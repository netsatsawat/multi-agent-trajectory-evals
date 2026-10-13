def money_rule(step, returns):
    """No refund before the item is back. A reason or ''."""
    a, r = step["args"], step["result"]
    if "return_status" in r:
        returns[a["order_id"]] = r["return_status"]
    if step["tool"] == "issue_refund":
        if returns.get(a["order_id"]) != "received":
            return "refund before the item came back"
    return ""
def attribute_failure(steps, read):
    """First wrong step (the cause) and first broken rule
    (the symptom), each as (step, agent, what, reasons).
    read() pulls the facts out of a message or tool result."""
    known, returns = {}, {}
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
            why = money_rule(st, returns)
            wrong = (n, who, st["tool"], [why]) if why else None
            symptom = symptom or wrong
        cause = cause or wrong
    return cause, symptom
