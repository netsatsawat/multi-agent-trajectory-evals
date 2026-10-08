"""A made-up three-agent refund system. No model, no network.

  orchestrator   takes the customer's request, delegates, answers
  policy-agent   tools: get_order, check_policy
  billing-agent  tools: request_approval, issue_refund, send_email

Every agent is a deterministic stand-in: plain Python that decides from
the message it was handed, so every score in this harness comes from
the scoring code, not from a model. All tools write to one in-memory
SQLite database with a shared fake clock.

Business rule: approval before the money moves. issue_refund does NOT
enforce it, the way many real tool backends do not.

Billing's own instructions (the stand-in's 'prompt'): if the task says
approval is needed, get it first; otherwise refund, then file an
approval record for the audit trail. So billing's behaviour depends on
what the orchestrator's handoff tells it.
"""
import json
import sqlite3
from datetime import datetime, timedelta

from opentelemetry.trace import SpanKind

import tracing as T

ORCH, POLICY, BILLING = "orchestrator", "policy-agent", "billing-agent"
MODEL_NAME = "scripted-stand-in"   # not a model: deterministic code
PROVIDER_NAME = "local.scripted"   # custom value, allowed by the spec
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
    """Fake clock, one second per tick, so timestamps are repeatable."""

    def __init__(self):
        self.t = datetime(2026, 10, 7, 9, 0, 0)

    def now(self):
        self.t += timedelta(seconds=1)
        return self.t.strftime("%H:%M:%S")


def new_db():
    db = sqlite3.connect(":memory:")
    db.executescript(SCHEMA)
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
    """Semconv-shaped message list, content kept as a JSON text part."""
    msg = {"role": role,
           "parts": [{"type": "text", "content": json.dumps(facts)}]}
    if finish:
        msg["finish_reason"] = finish
    return json.dumps([msg])


def _message_event(span, sender, receiver, facts):
    """Record what crossed between two agents, as a span event."""
    span.add_event(T.MESSAGE_EVENT, {
        "from": sender, "to": receiver, "content": json.dumps(facts)})


class Run:
    """One run of the whole system: one DB, one clock, one trace."""

    def __init__(self, conversation_id):
        self.db, self.clock = new_db(), Clock()
        self.conv = conversation_id
        self.n_calls = 0

    def chat(self, finish):
        """A model turn. The stand-in has no tokens or messages."""
        with T.TRACER.start_as_current_span(
                f"chat {MODEL_NAME}", kind=SpanKind.CLIENT,
                attributes={T.OP: "chat", T.PROVIDER: PROVIDER_NAME,
                            T.MODEL: MODEL_NAME, T.FINISH: [finish]}):
            pass

    def tool(self, agent, name, **args):
        """One model turn that picks the tool, then the tool call."""
        self.chat("tool_call")
        self.n_calls += 1
        with T.TRACER.start_as_current_span(
                f"execute_tool {name}", kind=SpanKind.INTERNAL,
                attributes={T.OP: "execute_tool", T.TOOL_NAME: name,
                            T.TOOL_CALL_ID: f"call_{self.n_calls}",
                            T.TOOL_TYPE: "function",
                            T.AGENT_NAME: agent,
                            T.CONVERSATION_ID: self.conv,
                            # Opt-in content attributes, kept as JSON.
                            T.TOOL_ARGS: json.dumps(args)}) as span:
            result = TOOLS_BY_AGENT[agent][name](self.db, self.clock, **args)
            span.set_attribute(T.TOOL_RESULT, json.dumps(result))
        return result

    def invoke(self, agent, sender, facts, body):
        """Run `body(facts)` as `agent`, inside its own invoke_agent span.

        The message in (facts) and the reply out are both recorded on
        this span, so the trace shows exactly what crossed.
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

def policy_agent(run, task):
    order = run.tool(POLICY, "get_order", order_id=task["order_id"])
    policy = run.tool(POLICY, "check_policy", order_id=task["order_id"],
                      reason=task["reason"])
    return {"order_id": order["order_id"], "email": order["email"],
            "amount": order["amount"], **policy}


def make_billing_agent(heeds_approval_flag=True):
    def billing_agent(run, task):
        oid, amount = task["order_id"], task["amount"]
        flag = task.get("needs_approval") and heeds_approval_flag
        if flag:
            run.tool(BILLING, "request_approval", order_id=oid,
                     amount=amount, note="approval needed per policy")
            run.tool(BILLING, "issue_refund", order_id=oid, amount=amount)
        else:
            run.tool(BILLING, "issue_refund", order_id=oid, amount=amount)
            run.tool(BILLING, "request_approval", order_id=oid,
                     amount=amount, note="filed after refund for audit")
        run.tool(BILLING, "send_email", to=task["email"],
                 body=f"Refund of ${amount} issued for {oid}.")
        return {"order_id": oid, "refunded": amount}
    return billing_agent


def run_system(customer_request, handoff_keys, conversation_id,
               heeds_approval_flag=True):
    """Run the orchestrator end to end. Returns (db, answer, spans).

    `handoff_keys` decides which facts the orchestrator copies from the
    policy agent's reply into its handoff to billing. Dropping one is
    the bug this harness is about.
    """
    run = Run(conversation_id)
    billing = make_billing_agent(heeds_approval_flag)

    def orchestrator(req):
        run.chat("tool_call")              # decide: ask policy first
        verdict = run.invoke(POLICY, ORCH,
                             {"order_id": req["order_id"],
                              "reason": req["reason"]},
                             lambda t: policy_agent(run, t))
        if not verdict["refundable"]:
            return {"answer": "Sorry, this order is not refundable."}
        run.chat("tool_call")              # decide: hand off to billing
        handoff = {k: verdict[k] for k in handoff_keys}
        done = run.invoke(BILLING, ORCH, handoff,
                          lambda t: billing(run, t))
        return {"answer": f"Your refund of ${done['refunded']} for order "
                          f"{done['order_id']} has been issued."}

    reply = run.invoke(ORCH, "customer", customer_request, orchestrator)
    run.db.commit()
    return run.db, reply["answer"], T.take_spans()


def replay_billing(task, conversation_id):
    """Run billing alone on a given handoff, on a fresh database."""
    run = Run(conversation_id)
    run.invoke(BILLING, ORCH, task,
               lambda t: make_billing_agent()(run, t))
    run.db.commit()
    return run.db, T.take_spans()


# ---- the runs used in the article ----------------------------------------

ORDER = "A-1001"
REQUEST_A = {"order_id": ORDER, "reason": "broken",
             "text": "My order A-1001 arrived broken. Refund please."}
REQUEST_B = {"order_id": ORDER, "reason": "damaged",
             "text": "Order A-1001 arrived damaged, photo attached."}

# The failing orchestrator summarises the policy reply and drops the
# approval flag. The correct one passes it on.
HANDOFF_DROPS_FLAG = ["order_id", "amount", "email", "refundable"]
HANDOFF_COMPLETE = HANDOFF_DROPS_FLAG + ["needs_approval"]
