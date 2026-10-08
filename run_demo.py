"""Run the multi-agent worked example end to end and print every verdict.

Usage: .venv/bin/python run_demo.py

The demo needs no model, network or API key. It writes the spans of
runs A, B and C to ./traces/.
"""
import json
import random

import agents as A
import scorers as S
import tracing as T
from attribution import NEEDS, blame, handoff_gaps, steps_from_spans
from passk import pass_at_k, pass_hat_k
from traps import greedy_in_order, naive_contains_all

OID, AMOUNT = A.ORDER, 450

# One reference path per agent. For policy and billing the path is their
# tool calls. The orchestrator has no tools, so its path is the two
# agents it delegates to, written as 'delegate' calls.
REFS = {
    A.ORCH: [{"tool": "delegate", "args": {"to": A.POLICY}},
             {"tool": "delegate", "args": {"to": A.BILLING}}],
    A.POLICY: [
        {"tool": "get_order", "args": {"order_id": OID}},
        {"tool": "check_policy",
         "args": {"order_id": OID, "reason": "damaged"}}],
    A.BILLING: [
        {"tool": "request_approval",
         "args": {"order_id": OID, "amount": AMOUNT}},
        {"tool": "issue_refund",
         "args": {"order_id": OID, "amount": AMOUNT}}],
}


def verdict(ok):
    return "PASS" if ok else "FAIL"


def line(label, ok, detail=""):
    print(f"  {label:<30}{verdict(ok)}  {detail}".rstrip())


def fmt_args(d):
    return ", ".join(f"{k}={v!r}" for k, v in d.items())


# ---- printing the trace --------------------------------------------------

def print_tree(spans):
    """Print the spans and agent messages as an indented tree in time order."""
    kids = {}
    for s in spans:
        parent = s.parent.span_id if s.parent else None
        kids.setdefault(parent, []).append(s)

    def walk(span, depth):
        """Print a span's events and child spans in time order."""
        pad = "  " * depth
        items = [(e.timestamp, "event", e) for e in span.events
                 if e.name == T.MESSAGE_EVENT]
        items += [(c.start_time, "span", c)
                  for c in kids.get(span.context.span_id, [])]
        for _, kind, x in sorted(items, key=lambda t: t[0]):
            if kind == "event":
                ea = x.attributes
                print(f"  {pad}[event agent.message] "
                      f"{ea['from']} -> {ea['to']}: {ea['content']}")
                continue
            a = x.attributes
            extra = ""
            if a.get(T.OP) == "execute_tool":
                extra = (f"  args={a[T.TOOL_ARGS]}"
                         f"  result={a[T.TOOL_RESULT]}")
            print(f"  {pad}{x.name}{extra}")
            walk(x, depth + 1)

    for root in kids.get(None, []):
        print(f"  {root.name}")
        walk(root, 1)


def print_steps(steps):
    for n, st in enumerate(steps, 1):
        if st["kind"] == "tool":
            print(f"  {n:>2}. [{st['agent']}] {st['tool']}"
                  f"({fmt_args(st['args'])})")
        else:
            print(f"  {n:>2}. [{st['agent']}] message to {st['to']}: "
                  f"{json.dumps(st['facts'])}")


# ---- scorers that read the steps -----------------------------------------

def calls_by_agent(steps):
    """Split the steps into one call list per agent for the path check.

    The orchestrator's messages to policy and billing count as its
    'delegate' calls.
    """
    out = {A.ORCH: [], A.POLICY: [], A.BILLING: []}
    for st in steps:
        if st["kind"] == "tool":
            out[st["agent"]].append(
                {"tool": st["tool"], "args": st["args"]})
        elif st["agent"] == A.ORCH and st["to"] in REFS:
            out[A.ORCH].append(
                {"tool": "delegate", "args": {"to": st["to"]}})
    return out


def path_checks(steps):
    per_agent = calls_by_agent(steps)
    results = {}
    for agent, ref in REFS.items():
        ok, step, why = S.in_order(ref, per_agent[agent], S.stub_judge)
        tools = ", ".join(c["tool"] if c["tool"] != "delegate"
                          else f"delegate->{c['args']['to']}"
                          for c in per_agent[agent])
        detail = f"step {step}: {why}" if step else why
        line(f"{agent} path", ok, f"{detail}  [{tools}]")
        results[agent] = ok
    return results


def handoff_checks(steps):
    """Track facts the way blame() does and print a verdict for each
    handoff to an agent listed in NEEDS."""
    known, all_ok = {}, True
    for n, st in enumerate(steps, 1):
        if st["kind"] == "tool":
            known.setdefault(st["agent"], {}).update(st["result"])
            continue
        if st["to"] in NEEDS:
            gaps = handoff_gaps(st, known.get(st["agent"], {}))
            label = f"{st['agent']} -> {st['to']}"
            need = ", ".join(NEEDS[st["to"]])
            line(label, not gaps,
                 f"step {n}: " + ("; ".join(gaps) if gaps
                                  else f"carries {need}"))
            all_ok = all_ok and not gaps
        known.setdefault(st["to"], {}).update(st["facts"])
    return all_ok


def final_answer_from(steps):
    for st in steps:
        if st["kind"] == "message" and st["to"] == "customer":
            return st["facts"]["answer"]
    return ""


def print_blame(steps):
    cause, symptom = blame(steps)
    if cause is None:
        print("  none: every handoff carries what the next agent needs,"
              " and no rule is broken")
        return cause, symptom
    n, who, what, reasons = cause
    print(f"  first wrong step: step {n}, owner {who}, {what}")
    for r in reasons:
        print(f"    - {r}")
    if symptom:
        sn, swho, swhat, sreasons = symptom
        print(f"  first broken rule: step {sn}, by {swho}, {swhat}"
              f" ({'; '.join(sreasons)})")
        if symptom == cause:
            print(f"  verdict: blame {who}; its own action is the first"
                  " wrong step")
        else:
            print(f"  verdict: blame {who} (cause, step {n});"
                  f" {swho}'s {swhat} at step {sn} is a symptom")
    return cause, symptom


# ---- one run -------------------------------------------------------------

def score_run(title, request, handoff_keys, trace_file, **kw):
    print(f"\n=== {title} ===")
    print(f'customer: "{request["text"]}"')
    db, answer, spans = A.run_system(
        request, handoff_keys, f"conv_{trace_file}", **kw)
    T.save_json(spans, f"traces/{trace_file}.json")
    counts = T.span_counts(spans)
    print(f"trace: {counts.get('invoke_agent', 0)} invoke_agent, "
          f"{counts.get('chat', 0)} chat, "
          f"{counts.get('execute_tool', 0)} execute_tool spans "
          f"-> traces/{trace_file}.json")
    print("span tree (the stand-in's chat spans carry no content):")
    print_tree(spans)
    steps = steps_from_spans(spans)
    print("steps rebuilt from the spans, in time order:")
    print_steps(steps)
    answer_in_trace = final_answer_from(steps)
    assert answer_in_trace == answer
    print(f'final answer: "{answer}"')

    S.JUDGE_LOG.clear()
    print("-- per-agent path (in-order, per-argument policy) --")
    path_checks(steps)
    for name, got, want, jok in sorted(set(S.JUDGE_LOG)):
        print(f"  judge (stub) on {name}: {got!r} vs {want!r} -> "
              f"{'same meaning' if jok else 'different'}")
    print("-- handoffs (does each fact the next agent needs cross?) --")
    handoff_checks(steps)
    print("-- whole-system end state in SQLite --")
    for label, ok, detail in S.end_state(db, OID, AMOUNT):
        line(label, ok, detail)
    print("-- final answer --")
    ok, why = S.final_answer_check(answer, [OID, str(AMOUNT)])
    line("names the order and amount", ok, why)
    print("-- failure attribution (walk the tree in time order) --")
    cause, symptom = print_blame(steps)
    return steps, cause, symptom


def replay_check(steps):
    """Re-run billing alone with its handoff plus the fact it was missing.

    If billing then asks for approval first, its early refund in the
    failing run was a symptom of the orchestrator's dropped fact.
    """
    print("\n=== Replay: billing alone, given the fact it was missing ===")
    known, handoff = {}, None
    for st in steps:
        if st["kind"] == "message":
            if st["agent"] == A.ORCH and st["to"] == A.BILLING:
                handoff = dict(st["facts"])
            known.setdefault(st["to"], {}).update(st["facts"])
    restored = dict(handoff)
    restored["needs_approval"] = known[A.ORCH]["needs_approval"]
    print(f"  handoff as sent:     {json.dumps(handoff)}")
    print(f"  with fact restored:  {json.dumps(restored)}")
    db, spans = A.replay_billing(restored, "conv_replay")
    calls = calls_by_agent(steps_from_spans(spans))[A.BILLING]
    print("  billing now calls: "
          + ", ".join(c["tool"] for c in calls))
    ok, step, why = S.in_order(REFS[A.BILLING], calls, S.stub_judge)
    line("billing path", ok, why)
    for label, ok, detail in S.end_state(db, OID, AMOUNT):
        if label == "approval before refund":
            line(label, ok, detail)


# ---- matcher traps and pass^k (kept from the single-agent harness) ------

def call(tool, amount=AMOUNT):
    return {"tool": tool, "args": {"order_id": OID, "amount": amount}}


def traps():
    print("\n=== Two matcher traps ===")
    print("Trap 1: the reference list is empty (it failed to load).")
    print("  run: issue_refund(450)")
    run = [call("issue_refund")]
    line("naive contains-all", naive_contains_all([], run),
         "all([]) is True")
    line("greedy in-order", greedy_in_order([], run), "0 of 0 matched")
    ok, step, why = S.in_order([], run, S.stub_judge)
    line("fixed in-order", ok, why)

    print("Trap 2: reference approve, refund."
          " Run refund, approve, refund.")
    ref = [call("request_approval"), call("issue_refund")]
    run = [call("issue_refund"), call("request_approval"),
           call("issue_refund")]
    line("naive contains-all", naive_contains_all(ref, run),
         "both tools appear somewhere")
    line("greedy in-order", greedy_in_order(ref, run),
         "skipped the first refund as noise")
    ok, step, why = S.in_order(ref, run, S.stub_judge)
    line("fixed in-order", ok, f"step {step}: {why}")


def system_passes(db, steps):
    paths = all(S.in_order(REFS[a], c, S.stub_judge)[0]
                for a, c in calls_by_agent(steps).items())
    state = all(c[1] for c in S.end_state(db, OID, AMOUNT))
    return paths and state and blame(steps)[0] is None


def repeated_runs(n=8, seed=2026, p_bad=0.25):
    print(f"\n=== pass^k over {n} repeated runs ===")
    print(f"Stand-in for a flaky orchestrator: each run drops the approval"
          f" flag from the billing handoff with probability {p_bad}"
          f" (random.Random({seed})).")
    rng = random.Random(seed)
    results = []
    for _ in range(n):
        keys = (A.HANDOFF_DROPS_FLAG if rng.random() < p_bad
                else A.HANDOFF_COMPLETE)
        db, _, spans = A.run_system(A.REQUEST_B, keys, "conv_rep")
        results.append(system_passes(db, steps_from_spans(spans)))
    marks = " ".join("P" if r else "F" for r in results)
    print("run results (P = every path, handoff, end-state check passes):"
          f" {marks}")
    print(f"c = {sum(results)} passes out of n = {n}")
    for k in (1, 2, 4, 8):
        print(f"  k={k}:  pass^k = {pass_hat_k([results], k):.3f}"
              f"   pass@k = {pass_at_k([results], k):.3f}")


if __name__ == "__main__":
    steps_a, _, _ = score_run(
        "Run A: the orchestrator drops the approval flag",
        A.REQUEST_A, A.HANDOFF_DROPS_FLAG, "run_a")
    replay_check(steps_a)
    score_run("Run B: the correct system",
              A.REQUEST_B, A.HANDOFF_COMPLETE, "run_b")
    score_run("Run C (control): full handoff, billing ignores the flag",
              A.REQUEST_B, A.HANDOFF_COMPLETE, "run_c",
              heeds_approval_flag=False)
    traps()
    repeated_runs()
