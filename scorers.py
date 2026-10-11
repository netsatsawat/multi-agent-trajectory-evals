"""Score an agent's trajectory, the list of tool calls it made.

The scorers are plain Python and need no model or network. The rule
for each argument and the check on what billing's tools wrote to
SQLite come unchanged from the single-agent harness. So does the
in-order matcher, which avoids the bugs shown in traps.py. In this
multi-agent harness, run_demo.py runs the in-order matcher once for
each agent. read_facts pulls the facts out of a message for the payload
check, from a dict or from text. The last function, final_answer_check,
grades the outcome instead of the trajectory. It checks the outcome
against the customer's intention.
"""
import json
import re

COMPARE, IGNORE, JUDGE = "compare", "ignore", "judge"
ARG_POLICY = {
    "check_policy": {"order_id": COMPARE, "reason": JUDGE},
    "request_approval": {
        "order_id": COMPARE, "amount": COMPARE, "note": IGNORE},
    "send_email": {"to": COMPARE, "body": IGNORE},
}
def mismatch(ref, act, judge):
    """Return why act differs from ref, or "" if it matches."""
    if ref["tool"] != act["tool"]:
        return f"tool {act['tool']} != {ref['tool']}"
    rules = ARG_POLICY.get(ref["tool"], {})
    for name in sorted(ref["args"].keys() | act["args"].keys()):
        rule = rules.get(name, COMPARE)  # args with no rule must match exactly
        want = ref["args"].get(name)
        got = act["args"].get(name)
        if rule == IGNORE:
            continue
        if rule == JUDGE:
            ok = judge(name, want, got)
        else:
            ok = want == got
        if not ok:
            return f"{name} {got!r} != {want!r}"
    return ""


# In a real setup the judge is an LLM grader with a rubric, and it can be
# a local model. This stand-in uses a fixed synonym table instead, so the
# run repeats exactly and the demo can print every verdict.
SAME_MEANING = {
    "damaged": {"damaged", "broken", "arrived damaged", "defective"},
}
JUDGE_LOG = []


def stub_judge(name, want, got):
    ok = got is not None and (
        got == want or got.lower() in SAME_MEANING.get(want, set()))
    JUDGE_LOG.append((name, got, want, ok))
    return ok


# ---- reading the facts in a message ---------------------------------------
# A message between agents can be a dict or plain text. read_facts turns
# either into a dict of facts, so the payload check can compare them.
# Code reads the order ID, the amount and the email. For the approval
# rule, a real setup asks a model judge one closed question: what does
# this message say about manager approval? The answers are "before
# refund", "already given", "side task" and "none". This stand-in
# answers from a fixed phrase table, so the run repeats exactly. Like
# SAME_MEANING, the table only knows the scripted messages in agents.py.
FACT_PATTERNS = {"order_id": r"\b([A-Z]-\d{4})\b", "amount": r"\$(\d+)",
                 "email": r"([\w.]+@[\w.]+\w)"}
APPROVAL_SAYS = [  # (answer, phrases). The first phrase found wins.
    ("before refund", ["approve it before", "approval first"]),
    ("already given", ["already approved"]),
    ("side task", ["also file"]),
]
NO_APPROVAL = "none"
APPROVAL_LOG = []   # (phrase found or None, answer), one per text read


def read_approval(text):
    """Answer the closed question about approval for one message."""
    low = text.lower()
    for answer, phrases in APPROVAL_SAYS:
        hit = next((p for p in phrases if p in low), None)
        if hit:
            APPROVAL_LOG.append((hit, answer))
            return answer
    APPROVAL_LOG.append((None, NO_APPROVAL))
    return NO_APPROVAL


def read_facts(content):
    """The facts in a message or a tool result, as a dict.

    A dict is read as it is, and needs_approval becomes the fact
    approval: "before refund" when true, because the business rule puts
    approval before any money moves, and "none" when false. Text is read
    with FACT_PATTERNS and the approval phrase table.
    """
    if isinstance(content, dict):
        facts = dict(content)
        if "needs_approval" in facts:
            facts["approval"] = ("before refund" if facts["needs_approval"]
                                 else NO_APPROVAL)
        return facts
    facts = {}
    for key, pattern in FACT_PATTERNS.items():
        m = re.search(pattern, content)
        if m:
            facts[key] = int(m.group(1)) if key == "amount" else m.group(1)
    facts["approval"] = read_approval(content)
    return facts


def strict_judge(name, want, got):
    """Compare free text exactly, as mismatch() does for args with no rule."""
    return want == got


def exact_match(reference, actual, judge):
    if not reference:
        return False, "empty reference: nothing to check"
    if len(reference) != len(actual):
        first = next(
            (n for n, (r, a) in enumerate(zip(reference, actual), 1)
             if mismatch(r, a, judge)), None)
        where = (f"; step {first}: "
                 f"{mismatch(reference[first - 1], actual[first - 1], judge)}"
                 if first else "")
        return False, f"{len(actual)} steps vs {len(reference)}{where}"
    for n, (r, a) in enumerate(zip(reference, actual), 1):
        why = mismatch(r, a, judge)
        if why:
            return False, f"step {n}: {why}"
    return True, f"all {len(reference)} steps identical"


def in_order(reference, actual, judge, prefix=False):
    """Check that the reference calls appear in order.

    An extra call in between passes only if its tool is not in the
    reference, so a refund made too early fails instead of being
    skipped as noise. With prefix=True, a run that stops before the
    end of the reference still passes. Returns (passed, step, why).
    """
    if not reference:
        return False, 0, "empty reference: nothing to check"
    ref_tools = {c["tool"] for c in reference}
    i = 0
    for step, call in enumerate(actual, 1):
        if i < len(reference):
            why = mismatch(reference[i], call, judge)
            if not why:
                i += 1
                continue
            if call["tool"] == reference[i]["tool"]:
                return False, step, why
        got = call["tool"]
        if got in ref_tools:
            want = (reference[i]["tool"]
                    if i < len(reference) else "no more calls")
            return False, step, f"got {got}, want {want}"
    if i < len(reference) and not prefix:
        missing = reference[i]["tool"]
        return False, len(actual) + 1, f"missing {missing}"
    return True, None, f"{i} of {len(reference)} in order"


def any_order(reference, actual, judge):
    """Every reference call must appear once, in any order."""
    if not reference:
        return False, "empty reference: nothing to check"
    used = set()
    for r in reference:
        hit = next((j for j, a in enumerate(actual)
                    if j not in used and not mismatch(r, a, judge)), None)
        if hit is None:
            near = next((mismatch(r, a, judge) for a in actual
                         if a["tool"] == r["tool"]), "tool never called")
            return False, f"no match for {r['tool']} ({near})"
        used.add(hit)
    return True, f"all {len(reference)} reference calls found"


def precision_recall(reference, actual, judge):
    """Match calls one to one, ignoring order.

    Returns precision, recall, the number matched and the extra tools.
    """
    if not reference or not actual:
        raise ValueError("precision/recall undefined on an empty list")
    used, matched = set(), 0
    for r in reference:
        hit = next((j for j, a in enumerate(actual)
                    if j not in used and not mismatch(r, a, judge)), None)
        if hit is not None:
            used.add(hit)
            matched += 1
    extras = [a["tool"] for j, a in enumerate(actual) if j not in used]
    return matched / len(actual), matched / len(reference), matched, extras


def billing_rows(db, order_id, amount):
    """Check the rows billing's tools wrote to SQLite, and their order.

    This is part of the tool check on billing's path, so it is specific
    to this refund system. It never reads what the agents said.
    """
    q = lambda sql: db.execute(sql, (order_id,)).fetchall()
    refunds = q("SELECT amount, ts FROM refunds WHERE order_id = ?")
    approvals = q("SELECT amount, ts FROM approvals WHERE order_id = ?")
    status = q("SELECT status FROM orders WHERE order_id = ?")[0][0]
    checks = []
    checks.append((f"one refund row of {amount}",
                   [r[0] for r in refunds] == [amount],
                   f"refunds={[r[0] for r in refunds]}"))
    checks.append((f"one approval row of {amount}",
                   [a[0] for a in approvals] == [amount],
                   f"approvals={[a[0] for a in approvals]}"))
    if refunds and approvals:
        a_ts, r_ts = approvals[0][1], refunds[0][1]
        checks.append(("approval before refund", a_ts < r_ts,
                       f"approval {a_ts}, refund {r_ts}"))
    else:
        checks.append(("approval before refund", False, "row missing"))
    checks.append(("order status is refunded", status == "refunded",
                   f"status={status}"))
    return checks


def final_answer_check(text, must_include):
    """Check the outcome against the customer's intention.

    The customer's intention is what the customer wants, here a refund
    of $450 for order A-1001. The outcome is what the system produces,
    and this check reads one part of it, the reply. The reply passes
    when it names every item in must_include, here the order ID and the
    refund amount. billing_rows checks the refund row itself, as part
    of billing's tool check.
    """
    missing = [m for m in must_include if m not in text]
    return not missing, (f"missing {missing}" if missing
                         else f"mentions {', '.join(must_include)}")
