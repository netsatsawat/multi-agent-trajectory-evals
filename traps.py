"""Two short matchers with bugs that let a bad trajectory pass.

They look reasonable, but both return True on a run where money moved
without a valid check.
"""


def naive_contains_all(reference, actual):
    """Pass if every reference tool appears somewhere in the run.

    all() of an empty list is True, so an empty reference passes any run.
    A test case ends up with an empty reference when loading it fails
    or nobody wrote one.
    """
    names = [c["tool"] for c in actual]
    return all(r["tool"] in names for r in reference)


def greedy_in_order(reference, actual):
    """Walk the run and move one step through the reference on each match.

    The matcher skips any call that does not match the next expected
    step and treats it as noise, even a refund issued before the
    approval. It also returns True for an empty reference, since 0 == 0.
    """
    i = 0
    for call in actual:
        if i < len(reference) and call["tool"] == reference[i]["tool"]:
            i += 1
    return i == len(reference)
