"""A bigger scripted support team for the tool-order and routing examples.

  orchestrator    picks a route for a request with keyword rules
  policy-agent    tools: get_order, check_policy
  billing-agent   tools: get_payment, request_approval, issue_refund,
                         send_receipt, log_case
  shipping-agent  tools: get_shipment, reship_item

Like agents.py, it needs no model and no network, and it records each
run as the same OpenTelemetry spans through agents.Run. Here the tools
depend on each other's results, so the order of the calls changes what
the system does:
  check_policy  takes the amount that get_order returns
  issue_refund  needs the payment_id that get_payment returns
  send_receipt  reads the refund rows, so before a refund it says $0
issue_refund still does not check for an approval, and it does not stop
a second refund of the same order.

This is a bigger version of the team in agents.py. There, check_policy
reads the amount itself and billing has three tools. Here check_policy
takes the amount from get_order's result, billing has five tools, and a
shipping agent joins.

A person writes down what each case expects: the customer's intention
and, for a refund, whether policy should approve it. A refund that
policy rightly turns down stops at policy, so the expected route
depends on both. The route, what the reply must name, the rows the run
must write, the facts each receiver needs and the tool order all follow
from that label (EXPECT below).
"""
from contextlib import contextmanager

import agents as A
import attribution
import tracing as T
from agents import ORCH, POLICY, BILLING
from sequence import AFTER

SHIPPING = "shipping-agent"

SCHEMA = """
CREATE TABLE orders(
  order_id TEXT PRIMARY KEY, email TEXT, amount INTEGER, status TEXT);
CREATE TABLE payments(payment_id TEXT, order_id TEXT, amount INTEGER);
CREATE TABLE approvals(order_id TEXT, amount INTEGER, ts TEXT);
CREATE TABLE refunds(
  order_id TEXT, payment_id TEXT, amount INTEGER, ts TEXT);
CREATE TABLE outbox(to_addr TEXT, body TEXT, ts TEXT);
CREATE TABLE shipments(order_id TEXT, kind TEXT, ts TEXT);
CREATE TABLE case_log(order_id TEXT, note TEXT, ts TEXT);
"""

# order_id, email, amount, status, number of card charges.
# Every amount that can be refunded is over the $100 approval limit, so
# money_rule in attribution.py, which wants an approval before any
# refund, agrees with the policy on every case.
ORDERS = [
    ("B-2002", "lee@example.com", 240, "delivered", 1),
    ("C-3003", "ana@example.com", 160, "delivered", 1),
    ("D-4004", "raj@example.com", 135, "delivered", 1),
    ("E-5005", "mia@example.com", 180, "delivered", 1),
    ("F-6006", "sam@example.com", 210, "delivered", 1),
    ("H-8008", "tom@example.com", 75, "in transit", 1),
    ("J-9009", "eve@example.com", 50, "in transit", 1),
    ("K-1010", "ian@example.com", 90, "in transit", 1),
    ("Q-1515", "noa@example.com", 40, "in transit", 1),
    ("L-1111", "zoe@example.com", 120, "delivered", 2),
    ("M-1212", "max@example.com", 140, "delivered", 2),
    ("N-1313", "amy@example.com", 130, "delivered", 2),
    ("P-1414", "kai@example.com", 110, "delivered", 2),
]
AMOUNT = {oid: amount for oid, _, amount, _, _ in ORDERS}


def schema_sql():
    rows = []
    for oid, email, amount, status, charges in ORDERS:
        rows.append(f"INSERT INTO orders VALUES"
                    f"('{oid}', '{email}', {amount}, '{status}');")
        for k in range(1, charges + 1):
            rows.append(f"INSERT INTO payments VALUES"
                        f"('P{k}-{oid}', '{oid}', {amount});")
    return SCHEMA + "\n".join(rows)


# ---- tools ---------------------------------------------------------------

def check_policy(db, clock, order_id, reason, amount):
    return {"refundable": reason.lower() in A.REFUNDABLE_REASONS,
            "needs_approval": amount > A.APPROVAL_LIMIT}


def get_payment(db, clock, order_id):
    rows = db.execute("SELECT p.payment_id, p.amount, o.email"
                      " FROM payments p JOIN orders o USING(order_id)"
                      " WHERE order_id = ? ORDER BY p.payment_id",
                      (order_id,)).fetchall()
    return {"payment_id": rows[-1][0], "charged": rows[-1][1],
            "email": rows[-1][2], "charges": len(rows)}


def issue_refund(db, clock, order_id, amount, payment_id):
    known = db.execute("SELECT 1 FROM payments WHERE payment_id = ?"
                       " AND order_id = ?", (payment_id, order_id)
                       ).fetchone()
    if known is None:
        return {"error": f"unknown payment_id {payment_id!r}"}
    db.execute("INSERT INTO refunds VALUES (?, ?, ?, ?)",
               (order_id, payment_id, amount, clock.now()))
    return {"refunded": amount}


def send_receipt(db, clock, order_id, to):
    total = db.execute("SELECT COALESCE(SUM(amount), 0) FROM refunds"
                       " WHERE order_id = ?", (order_id,)).fetchone()[0]
    body = f"Receipt: ${total} refunded for {order_id}."
    db.execute("INSERT INTO outbox VALUES (?, ?, ?)",
               (to, body, clock.now()))
    return {"sent": body}


def log_case(db, clock, order_id, note):
    db.execute("INSERT INTO case_log VALUES (?, ?, ?)",
               (order_id, note, clock.now()))
    return {"logged": True}


def get_shipment(db, clock, order_id):
    status = db.execute("SELECT status FROM orders WHERE order_id = ?",
                        (order_id,)).fetchone()[0]
    return {"shipping_status": status}


def reship_item(db, clock, order_id):
    db.execute("INSERT INTO shipments VALUES (?, 'replacement', ?)",
               (order_id, clock.now()))
    return {"reshipped": True}


TOOLS = {
    POLICY: {"get_order": A.get_order, "check_policy": check_policy},
    BILLING: {"get_payment": get_payment,
              "request_approval": A.request_approval,
              "issue_refund": issue_refund,
              "send_receipt": send_receipt, "log_case": log_case},
    SHIPPING: {"get_shipment": get_shipment, "reship_item": reship_item},
}
# The agent that should make each call. order_gaps uses it to name an
# owner for a needed call that never happened.
OWNER = {tool: agent for agent, tools in TOOLS.items() for tool in tools}

# Each tool's arguments, built from the facts the agent holds so far:
# the handoff plus its own earlier results. A call made too early gets
# a default (amount 0, payment_id None), the way a model fills a value
# it has not looked up yet.
ARGS = {
    "get_order": lambda f: {"order_id": f["order_id"]},
    "check_policy": lambda f: {"order_id": f["order_id"],
                               "reason": f["reason"],
                               "amount": f.get("amount", 0)},
    "get_payment": lambda f: {"order_id": f["order_id"]},
    "request_approval": lambda f: {"order_id": f["order_id"],
                                   "amount": f["amount"]},
    "issue_refund": lambda f: {"order_id": f["order_id"],
                               "amount": f["amount"],
                               "payment_id": f.get("payment_id")},
    "send_receipt": lambda f: {"order_id": f["order_id"],
                               "to": f["email"]},
    "log_case": lambda f: {"order_id": f["order_id"],
                           "note": "refund case"},
    "get_shipment": lambda f: {"order_id": f["order_id"]},
    "reship_item": lambda f: {"order_id": f["order_id"]},
}


def make_doer(run, agent, facts):
    def do(name):
        facts.update(run.tool(agent, name, **ARGS[name](facts)))
    return do


# ---- the agents ----------------------------------------------------------
# A `plan` argument replaces the agent's own choice with a fixed list of
# tool calls. The scripted order runs use it.

POLICY_PLAN = ["get_order", "check_policy"]
REFUND_PLAN = ["get_payment", "request_approval", "issue_refund",
               "send_receipt", "log_case"]


def policy_agent(run, task, plan=None):
    f = dict(task)
    do = make_doer(run, POLICY, f)
    for name in plan or POLICY_PLAN:
        do(name)
    return {k: f[k] for k in ("order_id", "email", "amount",
                              "refundable", "needs_approval")}


def billing_agent(run, task, plan=None):
    """After a policy verdict, refund. Called alone, fix a double charge."""
    f = dict(task)
    do = make_doer(run, BILLING, f)
    steps = REFUND_PLAN
    if "refundable" not in task:            # no policy verdict
        do("get_payment")
        if f["charges"] < 2:
            return {"order_id": f["order_id"],
                    "billing": "no problem found"}
        f.update(amount=f["charged"],
                 needs_approval=f["charged"] > A.APPROVAL_LIMIT)
        steps = REFUND_PLAN[1:]
    if plan is None:
        plan = [t for t in steps
                if t != "request_approval" or f["needs_approval"]]
    for name in plan:
        do(name)
    return {"order_id": f["order_id"], "refunded": f.get("refunded", 0)}


def shipping_agent(run, task, plan=None):
    f = dict(task)
    do = make_doer(run, SHIPPING, f)
    do("get_shipment")
    damaged = f["reason"] in ("damaged", "broken")
    if damaged and f["shipping_status"] == "delivered":
        do("reship_item")
        return {"order_id": f["order_id"], "reshipped": True}
    return {"order_id": f["order_id"], "status": f["shipping_status"]}


AGENTS = {POLICY: policy_agent, BILLING: billing_agent,
          SHIPPING: shipping_agent}

# ---- the orchestrator ----------------------------------------------------
# The router picks a route from keywords, and the first rule that
# matches wins. Each rule looks sensible alone. The mistakes come when
# one message holds words from two intentions: "arrived" was written
# for late parcels and also catches "arrived damaged".

ROUTES = [
    ("charged", [BILLING]),
    ("arrived", [SHIPPING]),
    ("where", [SHIPPING]),
    ("refund", [POLICY, BILLING]),
]


def keyword_router(text):
    for word, route in ROUTES:
        if word in text.lower():
            return list(route)
    return [POLICY, BILLING]


def answer(reply):
    oid = reply["order_id"]
    if reply.get("refunded"):
        return (f"Your refund of ${reply['refunded']} for order {oid}"
                f" has been issued.")
    if "refunded" in reply:
        return f"We could not refund order {oid}. A person will follow up."
    if reply.get("reshipped"):
        return f"A replacement for order {oid} is on its way."
    if "status" in reply:
        return f"Order {oid} is {reply['status']}."
    if reply.get("refundable") is False:
        return f"Sorry, order {oid} is not refundable."
    return f"We found no billing problem on order {oid}."


def run_case(case, conversation_id, route=None, plans=None):
    """Run one request end to end. Returns (db, answer, spans).

    `route` replaces the router's choice of agents, and `plans` maps an
    agent to a fixed list of tool calls for that agent.
    """
    run = A.Run(conversation_id, tools=TOOLS, schema=schema_sql())
    plans = plans or {}
    agents = route or keyword_router(case["text"])

    def orchestrator(req):
        handoff = {"order_id": req["order_id"], "reason": req["reason"]}
        reply = None
        for agent in agents:
            run.chat("tool_call")              # decides who goes next
            reply = run.invoke(
                agent, ORCH, handoff,
                lambda t, a=agent: AGENTS[a](run, t, plans.get(a)))
            if reply.get("refundable") is False:
                break                          # policy said no
            handoff = dict(reply)
        return {"answer": answer(reply)}

    request = {"order_id": case["order_id"], "reason": case["reason"],
               "text": case["text"]}
    out = run.invoke(ORCH, "customer", request, orchestrator)
    run.db.commit()
    return run.db, out["answer"], T.take_spans()


# ---- what each case expects ---------------------------------------------
# The intention is what the customer wants. For a refund, the expected
# route also depends on the policy verdict, so a person writes one of the
# four labels below for each case. Everything the checks expect follows
# from it.

REFUND_ROWS = {"approvals": 1, "refunds": 1, "outbox": 1, "case_log": 1}
# The facts each receiver needs, keyed by (sender, receiver) as NEEDS
# in attribution.py is. The handoffs here are dicts, so read_facts
# reads them as they are, and needs_approval becomes the fact approval.
REFUND_NEEDS = {(ORCH, POLICY): ["order_id", "reason"],
                (POLICY, ORCH): attribution.REFUND_FACTS,
                (ORCH, BILLING): attribution.REFUND_FACTS}
EXPECT = {
    "refund": {
        "intent": "refund", "want": [POLICY, BILLING], "say": "amount",
        "rows": REFUND_ROWS, "after": AFTER, "needs": REFUND_NEEDS},
    "refund turned down": {
        "intent": "refund", "want": [POLICY], "say": "not refundable",
        "rows": {}, "after": {"check_policy": ["get_order"]},
        "needs": {(ORCH, POLICY): REFUND_NEEDS[(ORCH, POLICY)]}},
    "parcel status": {
        "intent": "parcel status", "want": [SHIPPING],
        "say": "in transit", "rows": {}, "after": {"get_shipment": []},
        "needs": {(ORCH, SHIPPING): ["order_id"]}},
    "charged twice": {
        "intent": "charged twice", "want": [BILLING], "say": "amount",
        "rows": REFUND_ROWS,
        "after": {"issue_refund": ["get_payment", "request_approval"],
                  "send_receipt": ["issue_refund"], "log_case": []},
        "needs": {(ORCH, BILLING): ["order_id"]}},
}


def case(order_id, expect, reason, text):
    spec = EXPECT[expect]
    say = (str(AMOUNT[order_id]) if spec["say"] == "amount"
           else spec["say"])
    return {"order_id": order_id, "expect": expect,
            "intent": spec["intent"], "reason": reason, "text": text,
            "want": spec["want"], "must": [order_id, say],
            "rows": spec["rows"], "after": spec["after"],
            "needs": spec["needs"]}


ORDER_CASE = case("B-2002", "refund", "broken",
                  "My blender B-2002 is broken. I need a refund.")
WRONG_AGENT_CASE = case("C-3003", "refund", "damaged",
                        "Order C-3003 arrived damaged. Refund please.")
REFUSED_CASE = case("F-6006", "refund turned down", "changed my mind",
                    "I changed my mind about the lamp from F-6006."
                    " Can I get a refund?")
# Twelve made-up requests, four per intention. Three of them were
# written to trip the keyword router, one per intention: C-3003
# ("arrived"), K-1010 ("charged") and N-1313 ("where").
CASES = [
    ORDER_CASE,
    WRONG_AGENT_CASE,
    case("D-4004", "refund", "wrong item",
         "Wrong item in D-4004. Please refund it."),
    case("E-5005", "refund", "defective",
         "The kettle from E-5005 is defective, I want a refund."),
    case("H-8008", "parcel status", "status",
         "Where is my parcel H-8008?"),
    case("J-9009", "parcel status", "status",
         "Order J-9009 has not arrived yet."),
    case("K-1010", "parcel status", "status",
         "I was charged for K-1010 last week. When will it ship?"),
    case("Q-1515", "parcel status", "status", "Where is Q-1515 now?"),
    case("L-1111", "charged twice", "charged twice",
         "I was charged twice for L-1111."),
    case("M-1212", "charged twice", "charged twice",
         "You charged me two times for order M-1212."),
    case("N-1313", "charged twice", "charged twice",
         "I paid twice for N-1313, where is my money?"),
    case("P-1414", "charged twice", "charged twice",
         "My card was charged twice for P-1414."),
]


@contextmanager
def needs_for(case):
    """Use this case's handoff facts while attribute_failure() runs.

    attribution.NEEDS lists the facts per handoff, which works while
    each agent does one job. Billing does two jobs here (a refund after
    policy, a double charge alone), and each job needs different facts.
    """
    with attribution.use_needs(case["needs"]):
        yield


def rows_written(db):
    """Rows the run added, per table (orders and payments are seed data)."""
    out = {}
    for t in ("approvals", "refunds", "outbox", "shipments", "case_log"):
        n = db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        if n:
            out[t] = n
    return out
