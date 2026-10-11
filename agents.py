"""A made-up refund system with three agents. It needs no model and no network.

  orchestrator   takes the customer's request, delegates it, answers
  policy-agent   tools: get_order, check_policy
  billing-agent  tools: request_approval, issue_refund, send_email

Each agent is a scripted stand-in for a real one. It is plain Python
that decides from the message it receives and does the same thing on
every run. Because no model runs, every score in this harness comes
from the scoring code. The tools share one in-memory SQLite database,
and billing's tools stamp the rows they write with a fake clock.

The business rule says approval comes before any money moves.
issue_refund does not check it, and many real tool backends do not
check it either.

The agents pass work the way several agent frameworks do by default.
A tool result stays with the agent that called the tool. The policy
agent answers the orchestrator in prose, and the orchestrator writes
billing's task in its own words, from that answer alone. Only the
policy agent's rules tool knows the approval limit. The orchestrator
and billing act on what they are told. So billing asks for approval
first only if policy's answer said a manager must approve the refund,
and the orchestrator passed that on.
"""
import json
import re
import sqlite3
from datetime import datetime, timedelta

from opentelemetry.trace import SpanKind

import tracing as T

ORCH, POLICY, BILLING = "orchestrator", "policy-agent", "billing-agent"
MODEL_NAME = "scripted-stand-in"   # names the scripted code (no model runs)
PROVIDER_NAME = "local.scripted"   # the spec allows custom values here
APPROVAL_LIMIT = 100               # refunds above this need approval

SCHEMA = """
CREATE TABLE orders(
  order_id TEXT PRIMARY KEY, email TEXT, amount INTEGER, status TEXT);
CREATE TABLE approvals(order_id TEXT, amount INTEGER, ts TEXT);
CREATE TABLE refunds(order_id TEXT, amount INTEGER, ts TEXT);
CREATE TABLE outbox(to_addr TEXT, body TEXT, ts TEXT);
INSERT INTO orders VALUES('A-1001', 'kim@example.com', 450, 'delivered');
"""

REFUNDABLE_REASONS = {"damaged", "broken", "defective", "wrong item"}


class Clock:
    """A fake clock. Each now() call adds one second, so every run
    writes the same timestamps."""

    def __init__(self):
        self.t = datetime(2026, 10, 7, 9, 0, 0)

    def now(self):
        self.t += timedelta(seconds=1)
        return self.t.strftime("%H:%M:%S")


def new_db(schema=SCHEMA):
    db = sqlite3.connect(":memory:")
    db.executescript(schema)
    return db


# ---- tools ---------------------------------------------------------------

def get_order(db, clock, order_id):
    row = db.execute(
        "SELECT order_id, email, amount, status FROM orders"
        " WHERE order_id = ?", (order_id,)).fetchone()
    if row is None:
        return {"error": "not found"}
    return dict(zip(["order_id", "email", "amount", "status"], row))


def check_policy(db, clock, order_id, reason):
    amount = db.execute("SELECT amount FROM orders WHERE order_id = ?",
                        (order_id,)).fetchone()[0]
    return {"refundable": reason.lower() in REFUNDABLE_REASONS,
            "needs_approval": amount > APPROVAL_LIMIT}


def request_approval(db, clock, order_id, amount, note=""):
    db.execute("INSERT INTO approvals VALUES (?, ?, ?)",
               (order_id, amount, clock.now()))
    return {"approved": True}


def issue_refund(db, clock, order_id, amount):
    db.execute("INSERT INTO refunds VALUES (?, ?, ?)",
               (order_id, amount, clock.now()))
    db.execute("UPDATE orders SET status = 'refunded' WHERE order_id = ?",
               (order_id,))
    return {"refunded": amount}


def send_email(db, clock, to, body):
    db.execute("INSERT INTO outbox VALUES (?, ?, ?)", (to, body, clock.now()))
    return {"sent": True}


TOOLS_BY_AGENT = {
    POLICY: {"get_order": get_order, "check_policy": check_policy},
    BILLING: {"request_approval": request_approval,
              "issue_refund": issue_refund, "send_email": send_email},
}


# ---- tracing helpers -----------------------------------------------------

def _as_messages(role, facts, finish=None):
    """Return a one-message GenAI message list as a JSON string.

    The message's only text part holds the text of a message in prose,
    or the facts as JSON when the message is a dict.
    """
    text = facts if isinstance(facts, str) else json.dumps(facts)
    msg = {"role": role, "parts": [{"type": "text", "content": text}]}
    if finish:
        msg["finish_reason"] = finish
    return json.dumps([msg])


def _message_event(span, sender, receiver, facts):
    """Record one message an agent sends or receives as a span event."""
    span.add_event(T.MESSAGE_EVENT, {
        "from": sender, "to": receiver, "content": json.dumps(facts)})


class Run:
    """One run, with its own database and clock. Its spans form one trace."""

    def __init__(self, conversation_id, tools=None, schema=SCHEMA):
        self.db, self.clock = new_db(schema), Clock()
        self.tools = tools or TOOLS_BY_AGENT
        self.conv = conversation_id
        self.n_calls = 0

    def chat(self, finish):
        """Record a model turn as a chat span. The stand-in has no tokens
        or messages to put on it."""
        with T.TRACER.start_as_current_span(
                f"chat {MODEL_NAME}", kind=SpanKind.CLIENT,
                attributes={T.OP: "chat", T.PROVIDER: PROVIDER_NAME,
                            T.MODEL: MODEL_NAME, T.FINISH: [finish]}):
            pass

    def tool(self, agent, name, **args):
        """Record the model turn that picks the tool, then run the tool."""
        self.chat("tool_call")
        self.n_calls += 1
        with T.TRACER.start_as_current_span(
                f"execute_tool {name}", kind=SpanKind.INTERNAL,
                attributes={T.OP: "execute_tool", T.TOOL_NAME: name,
                            T.TOOL_CALL_ID: f"call_{self.n_calls}",
                            T.TOOL_TYPE: "function",
                            T.AGENT_NAME: agent,
                            T.CONVERSATION_ID: self.conv,
                            # The spec makes tool args and results opt-in.
                            # We store both as JSON.
                            T.TOOL_ARGS: json.dumps(args)}) as span:
            result = self.tools[agent][name](self.db, self.clock, **args)
            span.set_attribute(T.TOOL_RESULT, json.dumps(result))
        return result

    def invoke(self, agent, sender, facts, body):
        """Run `body(facts)` as `agent` inside its own invoke_agent span.

        The span records the incoming facts and the agent's reply, both
        as events and as message attributes, so the trace shows exactly
        what each agent received and sent back.
        """
        with T.TRACER.start_as_current_span(
                f"invoke_agent {agent}", kind=SpanKind.INTERNAL,
                attributes={T.OP: "invoke_agent", T.AGENT_NAME: agent,
                            T.CONVERSATION_ID: self.conv,
                            T.MODEL: MODEL_NAME,
                            T.INPUT_MESSAGES: _as_messages("user", facts)}
                ) as span:
            _message_event(span, sender, agent, facts)
            reply = body(facts)
            self.chat("stop")
            _message_event(span, agent, sender, reply)
            span.set_attribute(T.OUTPUT_MESSAGES,
                               _as_messages("assistant", reply, "stop"))
        return reply


# ---- the three agents ----------------------------------------------------
# The scripted agents read their messages with these patterns. The
# scorers read messages with their own code, read_facts in scorers.py,
# so no check grades an agent by the agent's own reading.

ORDER_ID = r"\b([A-Z]-\d{4})\b"
DOLLARS = r"\$(\d+)"
EMAIL = r"([\w.]+@[\w.]+\w)"


def _find(pattern, text):
    return re.search(pattern, text).group(1)


# The sentence policy adds to its answer when the refund needs a
# manager's approval first.
APPROVAL_LINE = (f"It is over ${APPROVAL_LIMIT}, so a manager must approve"
                 " it before any money goes out.")


def policy_agent(run, task, reports_approval=True):
    """Look the order up, check the rules, and answer in prose.

    The tool results stay with the policy agent. Its answer says
    whether the order can be refunded, what it cost and who paid. With
    reports_approval=False it leaves out the manager's approval, which
    its check_policy result still says is needed.
    """
    order = run.tool(POLICY, "get_order", order_id=task["order_id"])
    verdict = run.tool(POLICY, "check_policy", order_id=task["order_id"],
                       reason=task["reason"])
    oid = order["order_id"]
    if not verdict["refundable"]:
        return f"Order {oid} cannot be refunded for this reason."
    answer = (f"Order {oid} can be refunded. It arrived {task['reason']},"
              f" which the policy covers. It cost ${order['amount']},"
              f" paid by {order['email']}.")
    if verdict["needs_approval"] and reports_approval:
        answer += " " + APPROVAL_LINE
    return answer


def write_brief(answer):
    """The orchestrator's task for billing, in its own words.

    It passes on what policy's answer says, and nothing more.
    """
    oid, email = _find(ORDER_ID, answer), _find(EMAIL, answer)
    amount, reason = _find(DOLLARS, answer), _find(r"arrived (\w+)", answer)
    brief = (f"Refund order {oid}: ${amount} to {email}. It arrived"
             f" {reason} and policy says it can be refunded.")
    if "approve" in answer.lower():
        limit = _find(r"over \$(\d+)", answer)
        brief += (f" It is over ${limit}, so get a manager's approval"
                  " first, then refund.")
    return brief


def make_billing_agent(heeds_approval=True):
    """Billing reads its brief with its own rule: if the brief says to
    get approval first, it asks for approval, then refunds. With
    heeds_approval=False (the control run) it refunds first anyway and
    files the approval after. A brief that says nothing about approval
    gets a refund only."""
    def billing_agent(run, brief):
        oid, email = _find(ORDER_ID, brief), _find(EMAIL, brief)
        amount = int(_find(DOLLARS, brief))
        first = "approval first" in brief.lower()

        def approve():
            run.tool(BILLING, "request_approval", order_id=oid,
                     amount=amount, note="approval needed per policy")

        def refund():
            run.tool(BILLING, "issue_refund", order_id=oid, amount=amount)

        if first and heeds_approval:
            approve()
            refund()
        elif first:
            refund()
            approve()
        else:
            refund()
        run.tool(BILLING, "send_email", to=email,
                 body=f"Refund of ${amount} issued for {oid}.")
        return {"order_id": oid, "refunded": amount}
    return billing_agent


def run_system(customer_request, conversation_id, reports_approval=True,
               heeds_approval=True, recorded=None):
    """Run the orchestrator end to end. Returns (db, answer, spans).

    reports_approval: whether the policy agent's answer says a manager
    must approve the refund first. In the failing run it does not.
    heeds_approval: whether billing follows a brief that says so. In the
    control run it does not. recorded: (results, answer) to replay the
    policy agent with. Its tools return the recorded results, keyed by
    tool name, and it sends `answer` in place of its own.
    """
    tools = TOOLS_BY_AGENT
    if recorded is not None:
        tools = {**TOOLS_BY_AGENT, POLICY: {
            name: (lambda db, clock, _r=r, **args: _r)
            for name, r in recorded[0].items()}}
    run = Run(conversation_id, tools=tools)
    billing = make_billing_agent(heeds_approval)

    def policy(task):
        said = policy_agent(run, task, reports_approval)
        return said if recorded is None else recorded[1]

    def orchestrator(req):
        run.chat("tool_call")              # decides to ask policy first
        answer = run.invoke(POLICY, ORCH,
                            {"order_id": req["order_id"],
                             "reason": req["reason"]}, policy)
        if "can be refunded" not in answer:
            return {"answer": "Sorry, this order is not refundable."}
        run.chat("tool_call")              # decides to hand off to billing
        done = run.invoke(BILLING, ORCH, write_brief(answer),
                          lambda t: billing(run, t))
        return {"answer": f"Your refund of ${done['refunded']} for order "
                          f"{done['order_id']} has been issued."}

    reply = run.invoke(ORCH, "customer", customer_request, orchestrator)
    run.db.commit()
    return run.db, reply["answer"], T.take_spans()


def replay_from_answer(customer_request, results, answer, conversation_id):
    """Run the case again on a fresh database with one fact changed.

    Policy's tools return the recorded `results` (tool name to result),
    its answer is replaced by `answer`, and the orchestrator and billing
    run again. Returns (db, spans).
    """
    db, _, spans = run_system(customer_request, conversation_id,
                              recorded=(results, answer))
    return db, spans


# ---- the runs used in the article ----------------------------------------

ORDER = "A-1001"
# Runs A, B and C all start from this request, so only the policy
# agent's answer and billing's order of calls differ between them.
REQUEST = {"order_id": ORDER, "reason": "broken",
           "text": "My order A-1001 arrived broken. I need a refund."}
