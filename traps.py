"""Two matcher bugs that let a bad trajectory pass.

Both naive matchers below are short and look reasonable. Both return
True on inputs where the money moved without a valid check.
"""


def naive_contains_all(reference, actual):
    """'Every reference tool appears somewhere in the run.'

    all() over an empty list is True, so an empty reference (a test case
    whose reference failed to load, or was never written) passes any run.
    """
    names = [c["tool"] for c in actual]
    return all(r["tool"] in names for r in reference)


def greedy_in_order(reference, actual):
    """'Walk the run; advance through the reference on each match.'

    Anything that does not match the next expected step is skipped as
    noise, including a refund issued before the approval. It also
    returns True for an empty reference (0 == 0).
    """
    i = 0
    for call in actual:
        if i < len(reference) and call["tool"] == reference[i]["tool"]:
            i += 1
    return i == len(reference)
