"""Walk a multi-agent trace and say which agent's step went wrong first.

Blocks between '# >>> excerpt N' and '# <<< excerpt' are the code the
article prints. check_excerpts.py extracts them and runs them alone.
"""

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
NEEDS = {"policy-agent": ["order_id", "reason"],
         "billing-agent": ["order_id", "amount", "email",
                           "needs_approval"]}
def handoff_gaps(step, known):
    """Facts the receiver needs that the sender knew but did
    not pass on, or passed on changed."""
    gaps = []
    for key in NEEDS.get(step["to"], []):
        sent = step["facts"].get(key, "<missing>")
        if key not in known:
            gaps.append(f"{key}: never reached {step['agent']}")
        elif sent != known[key]:
            gaps.append(f"{key}: knew {known[key]!r}, "
                        f"sent {sent!r}")
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
def blame(steps):
    """First wrong step (the cause) and first broken rule
    (the symptom), each as (step, agent, what, reasons)."""
    known, approved = {}, {}
    cause = symptom = None
    for n, st in enumerate(steps, 1):
        who = st["agent"]
        if st["kind"] == "message":
            gaps = handoff_gaps(st, known.get(who, {}))
            known.setdefault(st["to"], {}).update(st["facts"])
            what = f"handoff to {st['to']}"
            wrong = (n, who, what, gaps) if gaps else None
        else:
            known.setdefault(who, {}).update(st["result"])
            why = money_rule(st, approved)
            wrong = (n, who, st["tool"], [why]) if why else None
            symptom = symptom or wrong
        cause = cause or wrong
    return cause, symptom
# <<< excerpt
