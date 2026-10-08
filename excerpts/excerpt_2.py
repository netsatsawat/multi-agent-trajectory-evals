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
