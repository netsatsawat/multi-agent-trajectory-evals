AFTER = {  # tool: the tools that must succeed before it runs
    "check_policy": ["get_order"],
    "issue_refund": ["check_policy", "get_payment",
                     "request_approval"],
    "send_receipt": ["issue_refund"],
    "log_case": []}  # needed, at any time
ONCE = {"issue_refund"}  # may succeed only one time per case
def order_gaps(steps, after=AFTER, once=ONCE, owner=None):
    """Calls made before a call they need, repeats of a
    once-only call and needed calls that never succeeded, as
    (step, agent, what, reasons). owner: tool -> its agent."""
    done, gaps = set(), []
    for n, st in enumerate(steps, 1):
        if st["kind"] != "tool":
            continue
        tool = st["tool"]
        why = [f"ran before {t} succeeded"
               for t in after.get(tool, []) if t not in done]
        if tool in once and tool in done:
            why.append("ran again")
        if why:
            gaps.append((n, st["agent"], tool, why))
        if "error" not in st["result"]:
            done.add(tool)
    needed = set(after).union(*after.values())
    for t in sorted(needed - done):
        gaps.append((len(steps) + 1, (owner or {}).get(t, "-"),
                     t, ["never succeeded"]))
    return gaps
