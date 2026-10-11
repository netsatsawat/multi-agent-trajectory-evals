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
