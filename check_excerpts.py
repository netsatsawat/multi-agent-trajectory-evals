"""Check that the code blocks the article prints still run on their own.

The article's code blocks, called excerpts here, are in attribution.py,
sequence.py and routing.py.
This script pulls each one out, checks its size, writes it to
excerpts/excerpt_N.py and runs it. An excerpt must fit in 30 lines of
65 characters or less, with no blank lines. Each one runs in a fresh
namespace that holds only the excerpts before it, so an excerpt fails
here if it needs a name that only the harness defines. The article
prints the blocks in the order 1, 4, 5, 2, 3, so excerpts 4 and 5 run
with only excerpt 1 before them.
"""
import re
from pathlib import Path

import agents as A
import attribution
import routing
import sequence
import team as TM

SOURCES = ["attribution.py", "sequence.py", "routing.py"]
BLOCK = re.compile(r"# >>> excerpt (\d+)\n(.*?)# <<< excerpt", re.S)


def extract():
    found = {}
    for src in SOURCES:
        for num, body in BLOCK.findall(Path(src).read_text()):
            found[int(num)] = body.rstrip("\n")
    return [found[k] for k in sorted(found)]


def lint(n, code):
    lines = code.split("\n")
    longest = max(len(x) for x in lines)
    blanks = sum(1 for x in lines if not x.strip())
    ok = len(lines) <= 30 and longest <= 65 and blanks == 0
    print(f"excerpt {n}: {len(lines)} lines, longest {longest} chars, "
          f"{blanks} blank lines -> {'OK' if ok else 'FAIL'}")
    return ok


def run_excerpt(n, code, ns):
    """Run one excerpt inside ns and print the names it defines."""
    names_before = set(ns)
    exec(compile(code, f"excerpt_{n}", "exec"), ns)
    new = sorted(k for k in set(ns) - names_before
                 if not k.startswith("__"))
    print(f"excerpt {n} defines: {', '.join(new)}")


def main():
    blocks = extract()
    assert len(blocks) == 5, f"expected 5 excerpts, found {len(blocks)}"
    assert all(lint(n, b) for n, b in enumerate(blocks, 1))
    Path("excerpts").mkdir(exist_ok=True)
    for n, code in enumerate(blocks, 1):
        Path(f"excerpts/excerpt_{n}.py").write_text(code + "\n")

    # Record real OpenTelemetry spans from three runs of the agents.
    _, _, bad = A.run_system(A.REQUEST_A, A.HANDOFF_DROPS_FLAG, "x1")
    _, _, good = A.run_system(A.REQUEST_B, A.HANDOFF_COMPLETE, "x2")
    _, _, ctrl = A.run_system(A.REQUEST_B, A.HANDOFF_COMPLETE, "x3",
                              heeds_approval_flag=False)

    ns = {}
    # Excerpt 1 runs alone. It rebuilds the steps in time order and
    # tags each step with the agent that owns it.
    run_excerpt(1, blocks[0], ns)
    only_1 = dict(ns)                  # what excerpts 4 and 5 may use
    steps = ns["steps_from_spans"](bad)
    who = [(s["agent"], s.get("tool") or "msg->" + s["to"])
           for s in steps]
    print(f"excerpt 1  steps_from_spans(failing run) -> {who}")
    assert who[5] == (A.ORCH, "msg->" + A.BILLING)
    assert who[6] == (A.BILLING, "issue_refund")
    assert len(steps) == 11

    # Excerpt 2 runs with only excerpt 1 before it and checks handoffs.
    run_excerpt(2, blocks[1], ns)
    handoff = steps[5]
    known = {**steps[4]["facts"]}      # what policy told the orchestrator
    gaps = ns["handoff_gaps"](handoff, known)
    print(f"excerpt 2  handoff_gaps(failing handoff) -> {gaps}")
    assert gaps == ["needs_approval: knew True, sent '<missing>'"]
    good_steps = ns["steps_from_spans"](good)
    gaps = ns["handoff_gaps"](good_steps[5], good_steps[4]["facts"])
    print(f"excerpt 2  handoff_gaps(correct handoff) -> {gaps}")
    assert gaps == []

    # Excerpt 3 runs with excerpts 1 and 2 before it and finds the
    # agent that caused the failure.
    run_excerpt(3, blocks[2], ns)
    cause, symptom = ns["attribute_failure"](steps)
    print(f"excerpt 3  attribute_failure(failing run) -> cause {cause}")
    print(f"{'symptom':>52} {symptom}")
    assert cause[:3] == (6, A.ORCH, "handoff to " + A.BILLING)
    assert symptom == (7, A.BILLING, "issue_refund",
                       ["refund before approval"])
    r = ns["attribute_failure"](good_steps)
    print(f"excerpt 3  attribute_failure(correct run) -> {r}")
    assert r == (None, None)
    r = ns["attribute_failure"](ns["steps_from_spans"](ctrl))
    print("excerpt 3  attribute_failure(control: billing ignores the"
          f" flag) -> {r}")
    assert r[0] == r[1] and r[0][1] == A.BILLING

    # The excerpts must agree with attribution.py, which the demo imports.
    for spans in (bad, good, ctrl):
        mine = ns["attribute_failure"](ns["steps_from_spans"](spans))
        theirs = attribution.attribute_failure(
            attribution.steps_from_spans(spans))
        assert mine == theirs
    print("excerpts match the harness: OK")

    # Excerpt 4 runs with only excerpt 1 before it and checks the order
    # of tool calls against the dependency graph (team.py, case B-2002).
    ns4 = dict(only_1)
    run_excerpt(4, blocks[3], ns4)
    order = {}
    for name, plans in (
            ("as designed", {}),
            ("policy order swapped",
             {A.POLICY: ["check_policy", "get_order"]}),
            ("refund sent twice",
             {A.BILLING: TM.REFUND_PLAN[:3] + ["issue_refund"]
              + TM.REFUND_PLAN[3:]}),
            ("payment lookup skipped",
             {A.BILLING: TM.REFUND_PLAN[1:]})):
        _, _, spans = TM.run_case(TM.ORDER_CASE, "x4", plans=plans)
        order[name] = spans
        gaps = ns4["order_gaps"](ns4["steps_from_spans"](spans))
        print(f"excerpt 4  order_gaps({name}) -> "
              f"{gaps[0] if gaps else gaps}")
    first = lambda k: ns4["order_gaps"](
        ns4["steps_from_spans"](order[k]))
    assert first("as designed") == []
    assert first("policy order swapped")[0] == (
        3, A.POLICY, "check_policy", ["ran before get_order succeeded"])
    assert first("refund sent twice")[0] == (
        10, A.BILLING, "issue_refund", ["ran again"])
    assert first("payment lookup skipped")[0] == (
        8, A.BILLING, "issue_refund", ["ran before get_payment succeeded"])
    # With owner, a needed call that never succeeded gets an agent.
    skipped = ns4["steps_from_spans"](order["payment lookup skipped"])
    last = ns4["order_gaps"](skipped, owner=TM.OWNER)[-1]
    print(f"excerpt 4  order_gaps(payment lookup skipped, owner) -> last"
          f" {last}")
    assert last == (13, A.BILLING, "issue_refund", ["never succeeded"])
    assert ns4["order_gaps"](skipped)[-1][1] == "-"

    # Excerpt 5 runs with only excerpt 1 before it and checks the route.
    ns5 = dict(only_1)
    run_excerpt(5, blocks[4], ns5)
    case = TM.WRONG_AGENT_CASE
    _, _, spans = TM.run_case(case, "x5")
    r = ns5["wrong_route"](ns5["steps_from_spans"](spans), case["want"])
    print(f"excerpt 5  wrong_route(refund sent to shipping) -> {r}")
    assert r == (2, A.ORCH, "handoff to " + TM.SHIPPING,
                 ["want " + A.POLICY])
    ok = ns5["wrong_route"](ns5["steps_from_spans"](order["as designed"]),
                            TM.ORDER_CASE["want"])
    print(f"excerpt 5  wrong_route(refund routed right) -> {ok}")
    assert ok is None

    # Excerpts 4 and 5 must agree with the modules run_ext.py imports.
    for spans in list(order.values()) + [spans]:
        steps = attribution.steps_from_spans(spans)
        assert ns4["order_gaps"](steps) == sequence.order_gaps(steps)
        assert (ns4["order_gaps"](steps, owner=TM.OWNER)
                == sequence.order_gaps(steps, owner=TM.OWNER))
        for want in (TM.ORDER_CASE["want"], case["want"]):
            assert (ns5["wrong_route"](steps, want)
                    == routing.wrong_route(steps, want))
    print("excerpts 4 and 5 match the harness: OK")


if __name__ == "__main__":
    main()
