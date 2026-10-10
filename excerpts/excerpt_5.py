def wrong_route(steps, want, router="orchestrator"):
    """First handoff to an agent other than the one the case
    needs, as (step, agent, what, reasons), or None."""
    k = 0
    for n, st in enumerate(steps, 1):
        if (st["kind"] != "message" or st["agent"] != router
                or st["to"] == "customer"):
            continue
        need = want[k] if k < len(want) else "no more handoffs"
        if st["to"] != need:
            return (n, router, f"handoff to {st['to']}",
                    [f"want {need}"])
        k += 1
    if k < len(want):
        return (len(steps) + 1, router, "no handoff",
                [f"want {want[k]}"])
    return None
