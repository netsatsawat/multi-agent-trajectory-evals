"""Run the multi-agent worked example end to end and print every verdict.

Usage: .venv/bin/python run_demo.py

The demo needs no model, network or API key. It writes the spans of
runs A, B and C to ./traces/.
"""
import json
import random

import agents as A
import attribution
import scorers as S
import tracing as T
from attribution import (
    NEEDS, attribute_failure, handoff_gaps, steps_from_spans, use_needs)
from passk import pass_at_k, pass_hat_k
from routing import wrong_route
from traps import greedy_in_order, naive_contains_all

OID, AMOUNT = A.ORDER, 450

# One reference path per agent: the tool calls its rules need, in order.
# The orchestrator has no tools. Its handoffs are checked in grade 2,
# where ROUTE lists the agents it must hand the work to.
ROUTE = [A.POLICY, A.BILLING]
REFS = {
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
    """Split the tool steps into one call list per agent for the path
    check."""
    out = {}
    for st in steps:
        if st["kind"] == "tool":
            out.setdefault(st["agent"], []).append(
                {"tool": st["tool"], "args": st["args"]})
    return out


def path_checks(steps):
    per_agent = calls_by_agent(steps)
    results = {}
    for agent, ref in REFS.items():
        calls = per_agent.get(agent, [])
        ok, step, why = S.in_order(ref, calls, S.stub_judge)
        tools = ", ".join(c["tool"] for c in calls)
        detail = f"step {step}: {why}" if step else why
        line(f"{agent} path", ok, f"{detail}  [{tools}]")
        results[agent] = ok
    return results


def route_check(steps, want, router=A.ORCH):
    """Print the routing verdict: did the router hand the work to the
    agents in `want`, in that order? wrong_route reports a trace step,
    the same way attribute_failure and order_gaps do."""
    r = wrong_route(steps, want, router)
    sent = [f"step {n}: {st['to']}" for n, st in enumerate(steps, 1)
            if st["kind"] == "message" and st["agent"] == router
            and st["to"] != "customer"]
    line("route (who got the work)", r is None,
         ", ".join(sent) if r is None
         else f"step {r[0]}: {r[2]}, {r[3][0]}")
    return r


def handoff_checks(steps):
    """Track facts the way attribute_failure() does and print a verdict
    for each message whose (sender, receiver) pair is listed in NEEDS.
    Then print what the stand-in judge read about approval in each of
    those messages that is plain text."""
    known, all_ok, readings = {}, True, []
    for n, st in enumerate(steps, 1):
        if st["kind"] == "tool":
            known.setdefault(st["agent"], {}).update(
                S.read_facts(st["result"]))
            continue
        S.APPROVAL_LOG.clear()
        sent = S.read_facts(st["facts"])
        pair = (st["agent"], st["to"])
        if pair in NEEDS:
            gaps = handoff_gaps(st, known.get(st["agent"], {}), sent)
            label = f"{st['agent']} -> {st['to']}"
            # On a pass, print the approval the message passed on, so a
            # brief that passes on "none" shows it.
            need = ", ".join(f"{k}={sent[k]!r}" if k == "approval" else k
                             for k in NEEDS[pair])
            line(label, not gaps,
                 f"step {n}: " + ("; ".join(gaps) if gaps
                                  else f"as the sender knew: {need}"))
            all_ok = all_ok and not gaps
            readings += [(n, hit, ans) for hit, ans in S.APPROVAL_LOG]
        known.setdefault(st["to"], {}).update(sent)
    for n, hit, ans in readings:
        found = repr(hit) if hit else "no approval phrase"
        print(f"  judge (stub) on approval, step {n}: {found} -> {ans}")
    return all_ok


def final_answer_from(steps):
    for st in steps:
        if st["kind"] == "message" and st["to"] == "customer":
            return st["facts"]["answer"]
    return ""


def print_attribution(steps):
    """Print who caused the failure (the cause) and where it showed
    (the symptom), in plain words."""
    cause, symptom = attribute_failure(steps, S.read_facts)
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
    verdict = f"  cause: {who}, step {n} ({what})."
    if symptom == cause:
        verdict += " It is also the first broken rule."
    elif symptom:
        verdict += f" symptom: {swho}'s {swhat} at step {sn}"
    print(verdict)
    return cause, symptom


# ---- one run -------------------------------------------------------------

def score_run(title, request, trace_file, **kw):
    print(f"\n=== {title} ===")
    print(f'customer: "{request["text"]}"')
    db, answer, spans = A.run_system(request, f"conv_{trace_file}", **kw)
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
    print("-- grade 1: each agent's path (in order, a rule per argument) --")
    path_checks(steps)
    for name, got, want, jok in sorted(set(S.JUDGE_LOG)):
        print(f"  judge (stub) on {name}: {got!r} vs {want!r} -> "
              f"{'same meaning' if jok else 'different'}")
    print("-- grade 1, billing's tools: what they wrote to SQLite --")
    for label, ok, detail in S.billing_rows(db, OID, AMOUNT):
        line(label, ok, detail)
    print("-- grade 2: each handoff (the right agent,"
          " and each fact the next agent needs) --")
    route_check(steps, ROUTE)
    handoff_checks(steps)
    print("-- grade 3: the outcome,"
          " checked against the customer's intention --")
    ok, why = S.final_answer_check(answer_in_trace, [OID, str(AMOUNT)])
    line("reply names order and amount", ok, why)
    print("-- failure attribution (walk the tree in time order) --")
    cause, symptom = print_attribution(steps)
    return steps, cause, symptom


# The payload check keyed by receiver alone, as excerpt 2 was before
# it listed policy's answer to the orchestrator. by_receiver writes such
# a table with a key for every sender, so handoff_gaps can read it.
BY_RECEIVER = {A.POLICY: ["order_id", "reason"],
               A.BILLING: attribution.REFUND_FACTS}
SENDERS = ["customer", A.ORCH, A.POLICY, A.BILLING]


def by_receiver(table):
    return {(s, r): keys for r, keys in table.items() for s in SENDERS
            if s != r}


def keying_check(steps_a):
    """Score the failing run and the correct run again with NEEDS keyed
    by receiver alone, then keyed by sender and receiver.

    Without a list for messages to the orchestrator, no check reads
    policy's answer, and on the failing run the cause moves to billing,
    the symptom. With one list for every message to the orchestrator,
    even the correct run fails, at the customer's own request, because
    the orchestrator needs different facts from each sender.
    """
    print("\n=== Runs A and B with the payload check keyed three ways ===")
    _, _, spans_b = A.run_system(A.REQUEST, "conv_keying")
    runs = [("failing run", steps_a),
            ("correct run", steps_from_spans(spans_b))]
    tables = [
        ("keyed by receiver, nothing listed for the orchestrator",
         by_receiver(BY_RECEIVER)),
        ("keyed by receiver, one list for the orchestrator",
         by_receiver({**BY_RECEIVER, A.ORCH: attribution.REFUND_FACTS})),
        ("keyed by sender and receiver, as NEEDS is", NEEDS),
    ]
    for label, table in tables:
        print(f"  {label}")
        for name, steps in runs:
            with use_needs(table):
                cause = attribute_failure(steps, S.read_facts)[0]
            if cause is None:
                print(f"    {name}: cause none")
                continue
            n, who, what, _ = cause
            print(f"    {name}: cause {who}, step {n} ({what})")


def replay_check(steps):
    """Put the approval sentence back into policy's answer and run the
    case again on a fresh database. Policy's tools return what the
    trace recorded, and the orchestrator and billing run again.

    If billing then asks for approval first and every check passes, its
    refund with no approval in the failing run was a symptom of the fact
    policy's answer left out.
    """
    print("\n=== Replay: policy's answer with the approval sentence"
          " put back ===")
    request = steps[0]["facts"]
    results = {st["tool"]: st["result"] for st in steps
               if st["kind"] == "tool" and st["agent"] == A.POLICY}
    answer = next(st["facts"] for st in steps if st["kind"] == "message"
                  and st["agent"] == A.POLICY and st["to"] == A.ORCH)
    restored = f"{answer} {A.APPROVAL_LINE}"
    print(f"  answer as sent:      {json.dumps(answer)}")
    print(f"  with fact restored:  {json.dumps(restored)}")
    db, spans = A.replay_from_answer(request, results, restored,
                                     "conv_replay")
    replayed = steps_from_spans(spans)
    brief = next(st["facts"] for st in replayed if st["kind"] == "message"
                 and st["agent"] == A.ORCH and st["to"] == A.BILLING)
    print(f"  orchestrator now writes: {json.dumps(brief)}")
    calls = calls_by_agent(replayed)[A.BILLING]
    print("  billing now calls: "
          + ", ".join(c["tool"] for c in calls))
    ok, step, why = S.in_order(REFS[A.BILLING], calls, S.stub_judge)
    line("billing path", ok, why)
    for label, ok, detail in S.billing_rows(db, OID, AMOUNT):
        if label == "approval before refund":
            line(label, ok, detail)
    cause = attribute_failure(replayed, S.read_facts)[0]
    line("every check, all three grades", system_passes(db, replayed),
         "cause: none" if cause is None
         else f"cause: {cause[1]}, step {cause[0]}")


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
    """True when a run passes all three grades: each agent's path with
    billing's tool check, each handoff (the route and the facts), and
    the outcome checked against the customer's intention."""
    calls = calls_by_agent(steps)
    paths = all(S.in_order(ref, calls.get(a, []), S.stub_judge)[0]
                for a, ref in REFS.items())
    route_ok = wrong_route(steps, ROUTE) is None
    tools = all(c[1] for c in S.billing_rows(db, OID, AMOUNT))
    outcome_ok = S.final_answer_check(final_answer_from(steps),
                                      [OID, str(AMOUNT)])[0]
    return (paths and route_ok and tools and outcome_ok
            and attribute_failure(steps, S.read_facts)[0] is None)


def repeated_runs(n=8, seed=2026, p_bad=0.25):
    print(f"\n=== pass^k over {n} repeated runs ===")
    print(f"Stand-in for a flaky policy agent: each run's answer leaves"
          f" out the approval with probability {p_bad}"
          f" (random.Random({seed})).")
    rng = random.Random(seed)
    results = []
    for _ in range(n):
        reports = rng.random() >= p_bad      # one draw per run
        db, _, spans = A.run_system(A.REQUEST, "conv_rep",
                                    reports_approval=reports)
        results.append(system_passes(db, steps_from_spans(spans)))
    marks = " ".join("P" if r else "F" for r in results)
    print(f"run results (P = all three grades pass): {marks}")
    print(f"c = {sum(results)} passes out of n = {n}")
    for k in (1, 2, 4, 8):
        print(f"  k={k}:  pass^k = {pass_hat_k([results], k):.3f}"
              f"   pass@k = {pass_at_k([results], k):.3f}")


if __name__ == "__main__":
    steps_a, _, _ = score_run(
        "Run A: the policy agent's answer leaves out the approval",
        A.REQUEST, "run_a", reports_approval=False)
    keying_check(steps_a)
    replay_check(steps_a)
    score_run("Run B: the correct system", A.REQUEST, "run_b")
    score_run("Run C (control): the brief says approval first,"
              " billing refunds first anyway",
              A.REQUEST, "run_c", heeds_approval=False)
    traps()
    repeated_runs()
