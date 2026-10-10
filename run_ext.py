"""Tool order and routing, on the bigger scripted team in team.py.

Usage: .venv/bin/python run_ext.py

Like run_demo.py, it needs no model, network or API key, and it prints
the same output on every run. Part 1 measures the order of tool calls
on one refund (B-2002), then tests each order checker on every order of
billing's five tools. Part 2 measures routing: one request sent to the
wrong agent (C-3003), every route forced on one case per intention,
twelve requests through the keyword router, and a refund that policy
rightly turns down (F-6006).
"""
from itertools import permutations

import scorers as S
import team as TM
from agents import BILLING, ORCH, POLICY
from attribution import (
    attribute_failure, first_wrong, handoff_gaps, steps_from_spans)
from routing import route_of, route_table, wrong_route
from run_demo import calls_by_agent, line, print_steps, route_check
from sequence import AFTER, ONCE, order_gaps

CASE = TM.ORDER_CASE
OID, AMOUNT, EMAIL, CARD = "B-2002", 240, "lee@example.com", "P1-B-2002"


def call(tool, **args):
    return {"tool": tool, "args": {"order_id": OID, **args}}


EXACT = {  # every call in one fixed order, for exact match
    POLICY: [call("get_order"),
             call("check_policy", reason="broken", amount=AMOUNT)],
    BILLING: [call("get_payment"),
              call("request_approval", amount=AMOUNT),
              call("issue_refund", amount=AMOUNT, payment_id=CARD),
              call("send_receipt", to=EMAIL),
              call("log_case", note="refund case")],
}
# For the in-order matcher, the calls the rules need, in order. Billing's
# get_payment is left out: no rule fixes its place, and the refund's
# payment_id argument already shows whether it ran first. Every refund
# needs one case note, at any time. A list has to put it somewhere, so
# REQUIRED puts it last. NO_NOTE leaves it out, and the case note gets
# a check of its own. Policy's two calls have only one valid order, as
# in run_demo.py's REFS.
REQUIRED = {
    POLICY: EXACT[POLICY],
    BILLING: [c for c in EXACT[BILLING] if c["tool"] != "get_payment"],
}
NO_NOTE = {
    POLICY: REQUIRED[POLICY],
    BILLING: [c for c in REQUIRED[BILLING] if c["tool"] != "log_case"],
}


def tool_effects(db):
    """What the tools changed in SQLite for B-2002 (grade 1)."""
    q = lambda sql: db.execute(sql, (OID,)).fetchall()
    refunds = q("SELECT amount, payment_id, ts FROM refunds"
                " WHERE order_id = ?")
    approvals = q("SELECT amount, ts FROM approvals WHERE order_id = ?")
    mail = [m[0] for m in q("SELECT body FROM outbox WHERE to_addr ="
                            " (SELECT email FROM orders"
                            " WHERE order_id = ?)")]
    notes = [n[0] for n in q("SELECT note FROM case_log"
                             " WHERE order_id = ?")]
    first = bool(refunds) and [a[0] for a in approvals] == [AMOUNT] \
        and approvals[0][1] < refunds[0][2]
    when = (f"approval {approvals[0][1]}" if approvals else "no approval",
            f"refund {refunds[0][2]}" if refunds else "no refund")
    return [
        (f"one refund of {AMOUNT}, right card",
         [r[:2] for r in refunds] == [(AMOUNT, CARD)],
         f"refunds={[r[:2] for r in refunds]}"),
        (f"approval of {AMOUNT} first", first, ", ".join(when)),
        (f"receipt says ${AMOUNT}",
         mail == [f"Receipt: ${AMOUNT} refunded for {OID}."],
         f"outbox={mail}"),
        ("one case note", len(notes) == 1, f"case_log={notes}"),
    ]


def score_order_run(steps, db, answer):
    per = calls_by_agent(steps)
    exact = {a: S.exact_match(EXACT[a], per.get(a, []), S.stub_judge)
             for a in (POLICY, BILLING)}
    in_ord = {a: S.in_order(REQUIRED[a], per.get(a, []), S.stub_judge)
              for a in (POLICY, BILLING)}
    gaps = order_gaps(steps, owner=TM.OWNER)
    eff = tool_effects(db)
    out = S.final_answer_check(answer, CASE["must"])
    return exact, in_ord, gaps, eff, out


def note_apart(steps):
    """The in-order matcher on NO_NOTE, plus a second check that
    billing logged the case note exactly once, at any time."""
    per = calls_by_agent(steps)
    in_ord = all(S.in_order(NO_NOTE[a], per.get(a, []), S.stub_judge)[0]
                 for a in (POLICY, BILLING))
    notes = sum(c["tool"] == "log_case" for c in per.get(BILLING, []))
    return in_ord and notes == 1


def who(found):
    """A finding in plain words: the agent, the step and what it did."""
    if not found:
        return "none"
    return f"{found[1]}, step {found[0]} ({found[2]})"


def billing_handoff(steps):
    """The handoff check (excerpt 2) on the orchestrator's message to
    billing, with the facts the orchestrator knew by then."""
    known = {}
    for n, st in enumerate(steps, 1):
        if st["kind"] == "tool":
            known.setdefault(st["agent"], {}).update(st["result"])
            continue
        if st["agent"] == ORCH and st["to"] == BILLING:
            return n, st["facts"], handoff_gaps(st, known.get(ORCH, {}))
        known.setdefault(st["to"], {}).update(st["facts"])
    return None, {}, ["no handoff to billing"]


# ---- part 1: tool order ----------------------------------------------------

BILL = TM.REFUND_PLAN
MUTATIONS = [
    ("as designed", {}),
    ("case note logged first",
     {BILLING: ["log_case"] + BILL[:4]}),
    ("refund before approval",
     {BILLING: ["get_payment", "issue_refund", "request_approval",
                "send_receipt", "log_case"]}),
    ("receipt before refund",
     {BILLING: ["get_payment", "request_approval", "send_receipt",
                "issue_refund", "log_case"]}),
    ("refund sent twice",
     {BILLING: BILL[:3] + ["issue_refund"] + BILL[3:]}),
    ("payment lookup skipped",
     {BILLING: BILL[1:]}),
    ("policy checks before it looks up the order",
     {POLICY: ["check_policy", "get_order"]}),
]


def tool_order():
    print("=== Grade 1, tool order: one refund,"
          f" {len(MUTATIONS)} scripted orders ===")
    print(f'customer: "{CASE["text"]}"')
    print("dependency graph (tool: tools that must succeed before it):")
    for tool, needs in AFTER.items():
        print(f"  {tool}: {', '.join(needs) or 'nothing (any time)'}")
    print(f"may succeed only once: {', '.join(sorted(ONCE))}")
    print("no edge, so any order: get_payment and request_approval;"
          " log_case anywhere")
    summary, routes_ok = [], 0
    for title, plans in MUTATIONS:
        db, answer, spans = TM.run_case(CASE, "conv_order", plans=plans)
        steps = steps_from_spans(spans)
        exact, in_ord, gaps, eff, out = score_order_run(steps, db, answer)
        print(f"\n-- {title} --")
        for agent in (POLICY, BILLING):
            calls = [c["tool"] for c in calls_by_agent(steps)[agent]]
            print(f"  {agent} calls: {', '.join(calls)}")
        print(f'  reply: "{answer}"')
        for agent, (ok, why) in exact.items():
            line(f"exact match ({agent})", ok, why)
        for agent, (ok, step, why) in in_ord.items():
            line(f"in order ({agent})", ok,
                 f"step {step}: {why}" if step else why)
        line("dependency graph", not gaps,
             "; ".join(f"step {n}: {what} {', '.join(r)}"
                       for n, _, what, r in gaps) or "every edge holds")
        for label, ok, detail in eff:
            line(label, ok, detail)
        n, facts, hgaps = billing_handoff(steps)
        line("handoff facts to billing", not hgaps,
             f"step {n}: " + ("; ".join(hgaps) if hgaps else
                              f"needs_approval={facts['needs_approval']}"
                              " passed on as policy reported it"))
        line("reply names order and amount", out[0], out[1])
        cause = attribute_failure(steps)[0]
        root = first_wrong(cause, *gaps)
        print(f"  cause from the handoff and money rules: {who(cause)}")
        print(f"  cause with the order check added: {who(root)}")
        routes_ok += wrong_route(steps, CASE["want"]) is None
        summary.append((title,
                        all(r[0] for r in exact.values()),
                        all(r[0] for r in in_ord.values()),
                        not gaps, all(c[1] for c in eff), out[0],
                        who(root)))
    print("\nsummary (P = pass, F = fail):")
    heads = ["exact", "in order", "graph", "effects", "reply"]
    print(f"  {'run':<44}" + "".join(f"{h:<10}" for h in heads)
          + "cause (first wrong step)")
    for title, *oks, root in summary:
        print(f"  {title:<44}"
              + "".join(f"{'P' if ok else 'F':<10}" for ok in oks)
              + root)
    print(f"route check (grade 2) on these runs: {routes_ok} of"
          f" {len(MUTATIONS)} went policy-agent, then billing-agent,"
          " as the refund expects")


def no_errors(steps):
    """The same steps with every tool result treated as a success."""
    return [dict(st, result={}) if st["kind"] == "tool" else st
            for st in steps]


def checker_test():
    print("\n=== Testing the order checkers on every order of billing's"
          " five tools (5! = 120 orders) ===")
    print("policy runs as designed. A right order is one where every"
          " tool effect check and the reply pass.")
    runs = []
    for plan in permutations(TM.REFUND_PLAN):
        db, answer, spans = TM.run_case(
            CASE, "conv_sweep", plans={BILLING: list(plan)})
        steps = steps_from_spans(spans)
        exact, in_ord, gaps, eff, out = score_order_run(steps, db, answer)
        refunded = db.execute("SELECT COUNT(*) FROM refunds").fetchone()[0]
        runs.append({
            "plan": plan, "steps": steps, "refunded": refunded > 0,
            "right": all(c[1] for c in eff) and out[0],
            "exact match": all(r[0] for r in exact.values()),
            "in order": all(r[0] for r in in_ord.values()),
            "in order, case note checked apart": note_apart(steps),
            "dependency graph": not gaps, "reply check alone": out[0]})
    good = [r for r in runs if r["right"]]
    bad = [r for r in runs if not r["right"]]
    print(f"right orders: {len(good)} of {len(runs)}")
    print(f"  {'checker':<36}{'accepts, of the right':>24}"
          f"{'accepts, of the wrong':>24}")
    for name in ("exact match", "in order",
                 "in order, case note checked apart", "dependency graph",
                 "reply check alone"):
        print(f"  {name:<36}{sum(r[name] for r in good):>16} of"
              f" {len(good):<5}{sum(r[name] for r in bad):>16} of"
              f" {len(bad)}")
    print("what each checker got wrong:")
    kept = [r["plan"] for r in good if r["exact match"]]
    print(f"  exact match accepts {len(kept)} right order, the one in its"
          f" reference: {', '.join(kept[0])}")
    dropped = [r["plan"] for r in good if not r["in order"]]
    early = sum(p.index("log_case") < p.index("send_receipt")
                for p in dropped)
    print(f"  in order rejects {len(dropped)} right orders."
          + (" In each one the case note comes before the receipt"
             if early == len(dropped) else
             f" In {early} of them the case note comes before the"
             " receipt"))
    fooled = {r["plan"] for r in bad if r["reply check alone"]}
    paid = {r["plan"] for r in bad if r["refunded"]}
    print(f"  the reply check accepts {len(fooled)} wrong orders."
          + (" They are the wrong orders in which a refund went through"
             if fooled == paid else
             f" A refund went through in {len(fooled & paid)} of them"))
    agree = sum(r["dependency graph"] == r["right"] for r in runs)
    print(f"  the dependency graph agrees with the database checks on"
          f" {agree} of {len(runs)} orders, because both come from the"
          " same rules")
    print("checker bugs this test catches (the graph with one change):")
    for tool, need in (("issue_refund", "request_approval"),
                       ("send_receipt", "issue_refund"),
                       ("issue_refund", "get_payment")):
        graph = {t: [x for x in ns if (t, x) != (tool, need)]
                 for t, ns in AFTER.items()}
        n = sum(not order_gaps(r["steps"], after=graph) for r in bad)
        print(f"  edge {need} -> {tool} left out:"
              f" accepts {n} wrong orders")
        if need == "get_payment":
            n = sum(not order_gaps(no_errors(r["steps"]), after=graph)
                    for r in bad)
            print(f"  same edge left out, and a failed call counted as"
                  f" done: accepts {n} wrong orders")
    plans = dict(MUTATIONS)["refund sent twice"]
    _, _, spans = TM.run_case(CASE, "conv_twice", plans=plans)
    gaps = order_gaps(steps_from_spans(spans), once=set())
    print("  no once-only rule: the refund sent twice"
          f" {'fails' if gaps else 'passes'} the graph")


# ---- part 2: routing -----------------------------------------------------

def short(route):
    return " then ".join(a.split("-")[0] for a in route) or "none"


def row_diff(got, want):
    extra = {t: n - want.get(t, 0) for t, n in got.items()
             if n > want.get(t, 0)}
    missing = {t: n - got.get(t, 0) for t, n in want.items()
               if got.get(t, 0) < n}
    return extra, missing


def causes(case, steps):
    """The cause from the handoff and money rules alone, and the cause
    with the routing and order checks added."""
    with TM.needs_for(case):
        cause = attribute_failure(steps)[0]
    root = first_wrong(cause, wrong_route(steps, case["want"]),
                       *order_gaps(steps, after=case["after"],
                                   owner=TM.OWNER))
    return cause, root


def handoff_facts(steps, needs, router=ORCH):
    """The handoff check (excerpt 2) on each of the router's handoffs,
    the way run_demo.handoff_checks does it. Returns the gaps found and
    the receivers with no facts listed in `needs`. Run it inside
    TM.needs_for(case), so handoff_gaps reads this case's lists."""
    known, gaps, unlisted = {}, [], []
    for n, st in enumerate(steps, 1):
        if st["kind"] == "tool":
            known.setdefault(st["agent"], {}).update(st["result"])
            continue
        if st["agent"] == router and st["to"] != "customer":
            if st["to"] in needs:
                gaps += [f"step {n}: {g}" for g in
                         handoff_gaps(st, known.get(router, {}))]
            else:
                unlisted.append(st["to"])
        known.setdefault(st["to"], {}).update(st["facts"])
    return gaps, unlisted


def wrong_agent_run(case):
    print("\n=== Grade 2, routing: one request sent to the wrong agent ===")
    print(f'customer: "{case["text"]}"')
    print(f"intention: {case['intent']} -> route expected:"
          f" {short(case['want'])}; reply must name {case['must']}")
    db, answer, spans = TM.run_case(case, "conv_route")
    steps = steps_from_spans(spans)
    print("steps rebuilt from the spans, in time order:")
    print_steps(steps)
    print(f'final answer: "{answer}"')
    for agent, calls in calls_by_agent(steps).items():
        print(f"  {agent} did its own job:"
              f" {', '.join(c['tool'] for c in calls)}")
    route_check(steps, case["want"])
    with TM.needs_for(case):
        hgaps, unlisted = handoff_facts(steps, case["needs"])
    if hgaps:
        detail = "; ".join(hgaps)
    elif unlisted:
        detail = (f"no facts listed for {', '.join(unlisted)} on a"
                  f" {case['intent']}, so nothing to check")
    else:
        detail = "every listed fact arrived"
    line("handoff facts", not hgaps, detail)
    got = TM.rows_written(db)
    line("rows the tools wrote", got == case["rows"],
         f"wrote {got}, want {case['rows']}")
    ok, why = S.final_answer_check(answer, case["must"])
    line("reply names order and amount", ok, why)
    cause, root = causes(case, steps)
    print(f"  cause from the handoff and money rules: {who(cause)}")
    print(f"  cause with the routing and order checks added: {who(root)}")


def forced_routes():
    print("\n=== Grade 2: every route for one case of each intention"
          " (route forced, no router) ===")
    picks = [c for c in TM.CASES
             if c["order_id"] in ("C-3003", "H-8008", "L-1111")]
    routes = [c["want"] for c in picks]
    for case in picks:
        print(f"{case['intent']} ({case['order_id']}),"
              f" route expected: {short(case['want'])}")
        for route in routes:
            db, answer, spans = TM.run_case(case, "conv_forced",
                                            route=list(route))
            steps = steps_from_spans(spans)
            took = route_of(steps)
            rows = TM.rows_written(db)
            extra, missing = row_diff(rows, case["rows"])
            reply_ok = S.final_answer_check(answer, case["must"])[0]
            sent = short(route)
            if took != route:
                sent += f" (stopped after {short(took)})"
            print(f"  sent to {sent:<44}"
                  f"route {'PASS' if took == case['want'] else 'FAIL'}"
                  f"  rows {'PASS' if rows == case['rows'] else 'FAIL'}"
                  f"  reply {'PASS' if reply_ok else 'FAIL'}")
            print(f'    reply "{answer}"; extra rows {extra or "none"};'
                  f" missing rows {missing or 'none'}")


def routing_batch():
    print(f"\n=== Grade 2 across cases: {len(TM.CASES)} made-up requests"
          " through the keyword router ===")
    print(f"  {'order':<8}{'intention':<15}{'expected':<21}{'taken':<21}"
          f"route rows reply cause (first wrong step)")
    pairs, misses = [], []
    for case in TM.CASES:
        db, answer, spans = TM.run_case(case, "conv_batch")
        steps = steps_from_spans(spans)
        got = route_of(steps)
        rows = TM.rows_written(db)
        reply_ok = S.final_answer_check(answer, case["must"])[0]
        root = causes(case, steps)[1]
        print(f"  {case['order_id']:<8}{case['intent']:<15}"
              f"{short(case['want']):<21}{short(got):<21}"
              f"{'P' if got == case['want'] else 'F':<6}"
              f"{'P' if rows == case['rows'] else 'F':<5}"
              f"{'P' if reply_ok else 'F':<6}{who(root)}")
        pairs.append((case["intent"], tuple(got), got == case["want"]))
        if got != case["want"]:
            extra, missing = row_diff(rows, case["rows"])
            misses.append((case["order_id"], short(got), answer, extra,
                           missing))
    right = sum(p[2] for p in pairs)
    print(f"routed as expected: {right} of {len(pairs)}")
    intents = list(dict.fromkeys(p[0] for p in pairs))
    cols = list(dict.fromkeys(tuple(c["want"]) for c in TM.CASES))
    table = route_table((i, g) for i, g, _ in pairs)
    print("routes taken (rows: the customer's intention,"
          " columns: the route taken):")
    print(f"  {'':<16}" + "".join(f"{short(r):>21}" for r in cols)
          + f"{'other':>8}")
    for intent in intents:
        other = sum(n for (i, r), n in table.items()
                    if i == intent and r not in cols)
        print(f"  {intent:<16}" + "".join(f"{table[(intent, r)]:>21}"
                                          for r in cols)
              + f"{other:>8}")
    print("what each wrong route did:")
    for oid, took, answer, extra, missing in misses:
        print(f'  {oid} sent to {took}: reply "{answer}";'
              f" extra rows {extra or 'none'};"
              f" missing rows {missing or 'none'}")


def refusal_test(case):
    print("\n=== False-alarm test: a refund that policy rightly"
          " turns down ===")
    print(f'customer: "{case["text"]}"')
    print(f"expected: {case['expect']} -> route {short(case['want'])};"
          f" reply must name {case['must']}")
    db, answer, spans = TM.run_case(case, "conv_refused")
    steps = steps_from_spans(spans)
    print(f'  reply: "{answer}"')
    route_check(steps, case["want"])
    gaps = order_gaps(steps, after=case["after"], owner=TM.OWNER)
    line("dependency graph", not gaps,
         "; ".join(f"step {n}: {what} {', '.join(r)}"
                   for n, _, what, r in gaps) or "every edge holds")
    got = TM.rows_written(db)
    line("rows the tools wrote", got == case["rows"],
         f"wrote {got or 'nothing'}, want {case['rows'] or 'nothing'}")
    ok, why = S.final_answer_check(answer, case["must"])
    line("reply names order and refusal", ok, why)
    print(f"  cause: {who(causes(case, steps)[1])}")
    plain = TM.case(case["order_id"], case["intent"], case["reason"],
                    case["text"])
    print(f"the same run, labelled with the intention alone"
          f" ({plain['intent']}, route {short(plain['want'])}):")
    print(f"  cause: {who(causes(plain, steps)[1])}")


if __name__ == "__main__":
    tool_order()
    checker_test()
    wrong_agent_run(TM.WRONG_AGENT_CASE)
    forced_routes()
    routing_batch()
    refusal_test(TM.REFUSED_CASE)
