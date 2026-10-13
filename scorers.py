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
    "create_return": {
        "order_id": COMPARE, "amount": COMPARE, "note": IGNORE},
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
# Code reads the order ID, the amount and the email. For the return
# rule, a real setup asks a model judge one closed question: what does
# this message say about getting the item back before the refund? The
# answers are "before refund", "already back", "side task" and "none".
# This stand-in answers from a fixed phrase table, so the run repeats
# exactly. Like SAME_MEANING, the table only knows the scripted
# messages in agents.py. The agents never call it: the orchestrator
# and billing read their messages with their own keyword rules.
FACT_PATTERNS = {"order_id": r"\b([A-Z]-\d{4})\b", "amount": r"\$(\d+)",
                 "email": r"([\w.]+@[\w.]+\w)"}
RETURN_SAYS = [  # (answer, phrases). The first phrase found wins.
    ("before refund", ["after the item comes back",
                       "once the item is back"]),
    ("already back", ["item is already back"]),
    ("side task", ["also send a return label"]),
]
NO_RETURN = "none"
RETURN_LOG = []   # (phrase found or None, answer), one per text read


def read_return(text):
    """Answer the closed question about the return for one message."""
    low = text.lower()
    for answer, phrases in RETURN_SAYS:
        hit = next((p for p in phrases if p in low), None)
        if hit:
            RETURN_LOG.append((hit, answer))
            return answer
    RETURN_LOG.append((None, NO_RETURN))
    return NO_RETURN


def read_facts(content):
    """The facts in a message or a tool result, as a dict.

    A dict is read as it is. requires_return becomes the fact return:
    "before refund" when true, because the rule pays the refund only
    after the item comes back, and "none" when false. For the blender
    team in team.py, needs_approval becomes the fact approval in the
    same way. Text is read with FACT_PATTERNS and the return phrase
    table.
    """
    if isinstance(content, dict):
        facts = dict(content)
        if "requires_return" in facts:
            facts["return"] = ("before refund" if facts["requires_return"]
                               else NO_RETURN)
        if "needs_approval" in facts:
            facts["approval"] = ("before refund" if facts["needs_approval"]
                                 else "none")
        return facts
    facts = {}
    for key, pattern in FACT_PATTERNS.items():
        m = re.search(pattern, content)
        if m:
            facts[key] = int(m.group(1)) if key == "amount" else m.group(1)
    facts["return"] = read_return(content)
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


def in_order(reference, actual, judge, due=None):
    """Check that the reference calls appear in order.

    An extra call in between passes only if its tool is not in the
    reference, so a refund made too early fails instead of being
    skipped as noise. due is how many reference calls are due in this
    run, all of them by default. With due=1, the run must stop after
    the first reference call. A later reference call fails as not due
    yet, as a refund does while every run ends before the item comes
    back. Returns (passed, step, why).
    """
    if not reference:
        return False, 0, "empty reference: nothing to check"
    ref_tools = {c["tool"] for c in reference}
    i = 0
    for step, call in enumerate(actual, 1):
        if due is not None and i >= due and call["tool"] in ref_tools:
            return False, step, f"got {call['tool']}, not due yet"
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
    if i < (len(reference) if due is None else due):
        missing = reference[i]["tool"]
        return False, len(actual) + 1, f"missing {missing}"
    later = len(reference) - i
    return True, None, (f"{i} of {len(reference)} in order"
                        + (f", {later} not due yet" if later else ""))


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
    """Check the rows billing's tools wrote to SQLite.

    This is part of the tool check on billing's path, so it is specific
    to this refund system. It never reads what the agents said. Every
    run ends before the item comes back, so a good run leaves one
    return row with its label sent, no refund row, and an email with
    the label. No return here is ever received, so a refund row alone
    breaks the rule and no timestamp is needed. A run that went on past
    the item's arrival would need the time the return was received and
    the time of the refund.
    """
    q = lambda sql: db.execute(sql, (order_id,)).fetchall()
    returns = q("SELECT amount, status FROM returns WHERE order_id = ?")
    refunds = [r[0] for r in
               q("SELECT amount FROM refunds WHERE order_id = ?")]
    mail = [m[0] for m in q("SELECT body FROM outbox WHERE to_addr ="
                            " (SELECT email FROM orders"
                            " WHERE order_id = ?)")]
    back = any(status == "received" for _, status in returns)
    return [
        (f"return row: {amount}, label sent",
         returns == [(amount, "label sent")], f"returns={returns}"),
        ("no refund before item is back", back or not refunds,
         f"refunds={refunds}, item {'back' if back else 'not back'}"),
        ("return label emailed",
         any("return label" in m.lower() for m in mail),
         f"outbox={mail}"),
    ]


def final_answer_check(text, must_include):
    """Check the outcome against the customer's intention.

    The customer's intention is what the customer wants, here a refund
    of $450 for order A-1001. The outcome is what the system produces,
    and this check reads one part of it, the reply. The reply passes
    when it names every item in must_include, here the order ID and the
    refund amount. billing_rows checks the return and refund rows
    themselves, as part of billing's tool check.
    """
    missing = [m for m in must_include if m not in text]
    return not missing, (f"missing {missing}" if missing
                         else f"mentions {', '.join(must_include)}")
