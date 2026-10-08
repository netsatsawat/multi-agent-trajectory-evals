"""Check that the code blocks the article prints still run on their own.

The article's code blocks, called excerpts here, are in attribution.py.
This script pulls each one out, checks its size, writes it to
excerpts/excerpt_N.py and runs it. An excerpt must fit in 30 lines of
65 characters or less, with no blank lines. Each one runs in a fresh
namespace that holds only the excerpts before it, so an excerpt fails
here if it needs a name that only the harness defines.
"""
import re
from pathlib import Path

import agents as A
import attribution

SOURCES = ["attribution.py"]
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
    assert len(blocks) == 3, f"expected 3 excerpts, found {len(blocks)}"
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

    # Excerpt 3 runs with excerpts 1 and 2 before it and assigns blame.
    run_excerpt(3, blocks[2], ns)
    cause, symptom = ns["blame"](steps)
    print(f"excerpt 3  blame(failing run) -> cause {cause}")
    print(f"                                 symptom {symptom}")
    assert cause[:3] == (6, A.ORCH, "handoff to " + A.BILLING)
    assert symptom == (7, A.BILLING, "issue_refund",
                       ["refund before approval"])
    r = ns["blame"](good_steps)
    print(f"excerpt 3  blame(correct run) -> {r}")
    assert r == (None, None)
    r = ns["blame"](ns["steps_from_spans"](ctrl))
    print(f"excerpt 3  blame(control: billing ignores the flag) -> {r}")
    assert r[0] == r[1] and r[0][1] == A.BILLING

    # The excerpts must agree with attribution.py, which the demo imports.
    for spans in (bad, good, ctrl):
        mine = ns["blame"](ns["steps_from_spans"](spans))
        theirs = attribution.blame(attribution.steps_from_spans(spans))
        assert mine == theirs
    print("excerpts match the harness: OK")


if __name__ == "__main__":
    main()
