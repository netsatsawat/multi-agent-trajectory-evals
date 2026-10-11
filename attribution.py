"""Find the first wrong step in a multi-agent trace and the agent that owns it.

The article prints the code between '# >>> excerpt N' and '# <<< excerpt'
word for word. check_excerpts.py pulls those blocks out and runs them
apart from the rest of the harness. attribute_failure takes the reader
as an argument, scorers.read_facts in this repo, so the excerpts import
nothing.
"""
from contextlib import contextmanager

# >>> excerpt 1
import json
OP, AGENT = "gen_ai.operation.name", "gen_ai.agent.name"
TOOL, ARGS = "gen_ai.tool.name", "gen_ai.tool.call.arguments"
RESULT = "gen_ai.tool.call.result"
def steps_from_spans(spans):
    """Flatten a multi-agent trace into time-ordered steps."""
    by_id = {s.context.span_id: s for s in spans}
    def owner(s):  # nearest invoke_agent span at or above s
        while s.attributes.get(OP) != "invoke_agent":
            s = by_id[s.parent.span_id]
        return s.attributes[AGENT]
    steps = []
    for s in spans:
        if s.attributes.get(OP) == "execute_tool":
            steps.append((s.start_time, {
                "kind": "tool", "agent": owner(s),
                "tool": s.attributes[TOOL],
                "args": json.loads(s.attributes[ARGS]),
                "result": json.loads(s.attributes[RESULT])}))
        for e in s.events:
            if e.name == "agent.message":
                a = e.attributes
                steps.append((e.timestamp, {
                    "kind": "message", "agent": a["from"],
                    "to": a["to"],
                    "facts": json.loads(a["content"])}))
    return [st for _, st in sorted(steps, key=lambda t: t[0])]
# <<< excerpt


# >>> excerpt 2
REFUND_FACTS = ["order_id", "amount", "email", "approval"]
NEEDS = {  # (sender, receiver): the facts the receiver needs
    ("orchestrator", "policy-agent"): ["order_id", "reason"],
    ("policy-agent", "orchestrator"): REFUND_FACTS,
    ("orchestrator", "billing-agent"): REFUND_FACTS,
}
def handoff_gaps(step, known, sent):
    """Facts the receiver needs that the sender knew but did
    not pass on, or passed on changed. sent: the facts read
    out of the message by code or, for text, by a judge."""
    gaps = []
    for key in NEEDS.get((step["agent"], step["to"]), []):
        got = sent.get(key, "<missing>")
        if key not in known:
            gaps.append(f"{key}: never reached {step['agent']}")
        elif got != known[key]:
            gaps.append(
                f"{key}: knew {known[key]!r}, sent {got!r}"
            )
    return gaps
# <<< excerpt


# >>> excerpt 3
def money_rule(step, approved):
    """Approval before money moves. Returns a reason or ''."""
    a = step["args"]
    if step["tool"] == "request_approval":
        approved[a["order_id"]] = a["amount"]
    if step["tool"] == "issue_refund":
        if approved.get(a["order_id"], 0) < a["amount"]:
            return "refund before approval"
    return ""
def attribute_failure(steps, read):
    """First wrong step (the cause) and first broken rule
    (the symptom), each as (step, agent, what, reasons).
    read() pulls the facts out of a message or tool result."""
    known, approved = {}, {}
    cause = symptom = None
    for n, st in enumerate(steps, 1):
        who = st["agent"]
        if st["kind"] == "message":
            sent = read(st["facts"])
            gaps = handoff_gaps(st, known.get(who, {}), sent)
            known.setdefault(st["to"], {}).update(sent)
            what = f"handoff to {st['to']}"
            wrong = (n, who, what, gaps) if gaps else None
        else:
            known.setdefault(who, {}).update(read(st["result"]))
            why = money_rule(st, approved)
            wrong = (n, who, st["tool"], [why]) if why else None
            symptom = symptom or wrong
        cause = cause or wrong
    return cause, symptom
# <<< excerpt


def first_wrong(*found):
    """The earliest of several (step, agent, what, reasons) findings.

    attribute_failure() returns its cause in this shape, and so do
    order_gaps() in sequence.py and wrong_route() in routing.py. The
    cause of a failed run is the finding with the lowest step number.
    On a tie, the finding passed in first wins.
    """
    found = [f for f in found if f]
    return min(found, key=lambda f: f[0]) if found else None


@contextmanager
def use_needs(table):
    """Use another NEEDS table while the code inside the block runs.

    handoff_gaps reads the module-level NEEDS, so swapping it keeps the
    article's excerpt 2 unchanged. team.needs_for uses this when one
    agent does two jobs, and run_demo.py uses it to show what a table
    keyed by receiver alone would name as the cause.
    """
    global NEEDS
    saved = NEEDS
    NEEDS = table
    try:
        yield
    finally:
        NEEDS = saved
