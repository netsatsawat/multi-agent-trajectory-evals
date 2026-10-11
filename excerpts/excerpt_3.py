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
