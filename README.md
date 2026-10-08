# multi-agent-trajectory-evals

[![reproduce](https://github.com/netsatsawat/multi-agent-trajectory-evals/actions/workflows/ci.yml/badge.svg)](https://github.com/netsatsawat/multi-agent-trajectory-evals/actions/workflows/ci.yml)

Score a multi-agent run step by step, check what each agent passes to the next, and find
which agent caused a failure. A test that reads only the final reply cannot tell you any of
that. In the run below, the reply is right, the money went out before anyone approved it,
and the agent that looks guilty is not the one to fix.

![A three-agent refund run where the orchestrator leaves the approval flag out of its handoff, billing refunds before approval, and a test on the final reply still passes](assets/hero-handoff.gif)

A customer asks for a refund on order A-1001, worth $450. The orchestrator agent sends the
case to a policy agent, which finds that a refund this size needs a manager's approval and
reports `needs_approval: true` back. The orchestrator then hands the case to a billing agent
but leaves that flag out. Billing, never told, refunds first and asks for approval one second
later. The customer still gets the right reply, so a test on the final reply passes. Billing
made the visible mistake, yet it followed its own rule with the facts it was given. The
orchestrator caused the failure when it dropped the approval flag, so if you grade only the
agent whose action looks wrong, you fix the wrong agent.

This repo is a worked example to read and copy from, with no package to install. The three
agents are scripted Python, so it runs offline with no model and no API key. The checks read
standard OpenTelemetry GenAI spans, so you can point them at your own agents (see
[Use it on your own agents](#use-it-on-your-own-agents)). It is the runnable companion to the
article *Multi-Agent Trajectory Evaluation, Explained Simply* (Towards AI, in draft). When an
image says "the article's harness", it means the code in this repo, and you do not need the
article to run or adapt it.

## Quickstart

```bash
uv venv --python 3.12 && uv pip install -r requirements.txt
.venv/bin/python run_demo.py
```

Without uv, `python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt` does the
same job. The only dependency is the OpenTelemetry SDK, pinned to 1.45.0.

The report has one block per run, and each block ends with its blame verdict under
`failure attribution`. Every run prints the same report, and CI checks that it still matches
the copy in `expected/`. On your machine,
`diff <(.venv/bin/python run_demo.py) expected/run_demo.txt` should print nothing.

A second script, `check_excerpts.py`, checks that the code blocks printed in the article
still run on their own. You need it only if you edit `attribution.py`.

## What it checks

The demo lays each run out as one numbered list of steps in time order. A step is one tool
call or one message between agents, and the failing run has 11 of them. Step 6 is the
orchestrator's handoff to billing, and step 7 is billing's refund. A handoff is the message an
agent sends when it passes work to another agent.

Every run gets four checks:

1. Each agent's path is matched in order against a short reference list for that agent
   (`REFS` in `run_demo.py`). Extra calls in between are fine, as long as their tool is not
   in the reference list. Each tool argument has its own rule: IDs and amounts must match
   exactly, free text goes to a judge, and free-form notes are ignored. The judge here is a
   fixed table of synonyms (`stub_judge` in `scorers.py`) standing in for a model, and it
   treats the customer's "broken" as the same as the reference's "damaged".
2. Each handoff is checked against the facts the receiving agent needs (`NEEDS` in
   `attribution.py`). For each fact, the check asks whether it arrived and whether its value
   matches what the sender knew at that moment.
3. The end state is read straight from the database: which rows the whole system left, and
   the order they were written in.
4. The final answer must name the right order and amount.

A failing run also gets a blame verdict. The first wrong step is the cause, and the agent
that owns it gets the blame. A rule broken later is a symptom.

![The failing run and the correct run scored by every check. Billing's path, the handoff to billing and the order of the database writes fail. The final answer passes. Blame goes to the orchestrator at step 6](assets/scored-run.png)

The failing run passes the final-answer check. Only three checks catch it: billing's path,
the handoff from the orchestrator, and the order of the database writes. Of those, the
handoff check points at the real cause:

![The handoff check going row by row through the four facts billing needs. Order ID, amount and email arrive unchanged, and needs_approval is missing](assets/handoff-check.gif)

## How it works

![How the harness fits together, from the scripted agents to the blame verdict, and which parts you replace, keep or write for your own system](assets/architecture.png)

`run_demo.py` starts each run. The three scripted agents in `agents.py` call their tools on
one shared SQLite database, and billing's tools write to it with a fake clock, so the
database timestamps repeat exactly. As the agents work, `agents.py` records the run as
OpenTelemetry spans, and `tracing.py` keeps them in memory. The names follow the GenAI
semantic conventions: an `invoke_agent` span per agent, a `chat` span per model turn, an
`execute_tool` span per tool call, and an `agent.message` event for each message between
agents. `steps_from_spans` in `attribution.py` turns that span tree into the numbered list of
steps and gives each step an owner. The path and handoff checks and `blame` run over that
list, and the end-state check reads the database.

The trace is a tree of spans, one bar per agent, model turn or tool call. The numbered
diamonds are messages between agents. At 5 the policy agent tells the orchestrator
`needs_approval: true`, and at 6 the orchestrator hands off to billing without it. The blame
walk reads the tree in time order:

![The failing run as a span tree. The orchestrator's handoff at step 6 is the first wrong step. Billing's refund at step 7 is the first broken rule, a symptom](assets/span-tree.png)

## The runs

`run_demo.py` prints these in order, and everything in this section comes from its output.

### Run A, the failing run

The orchestrator leaves `needs_approval` out of its handoff, so billing refunds before it
asks for approval. Blame goes to the orchestrator at step 6, and billing's refund at step 7
counts as the symptom.

### The replay

Billing runs alone on a fresh database, with the same handoff plus the one fact that was
missing. It now asks for approval first, and both of its checks pass. Fixing step 6 turns the
failure into a success, which is what makes step 6 the decisive step.

![The failing run as recorded, next to billing replayed alone with needs_approval put back. Both checks go from FAIL to PASS](assets/replay.gif)

### Run B, the correct run

The handoff carries the flag, and every check passes.

### Run C, a control

This run checks that blame follows the trace. The handoff is complete, but billing ignores
the flag and refunds first anyway, so blame lands on billing at step 7.

### Two matcher traps

The first trap is an empty reference list, which is what you get when a test's reference
fails to load. The second is a run that refunds, asks for approval, then refunds again. A
contains-all matcher and a greedy in-order matcher both pass these bad runs, and only the
fixed matcher (`in_order` in `scorers.py`) fails both.

![Both trap runs scored by three matchers. The two short matchers pass both bad runs, and the fixed matcher fails both](assets/matcher-traps.gif)

### Pass^k over 8 runs

The orchestrator drops the flag with a one-in-four chance on each run, seeded so it repeats.
Five of the eight runs pass, where a pass means every path, handoff and end-state check
passes. Pass^k, the chance that all k runs pass, falls from 0.625 at one run to 0.000 at
eight. Pass@k, the chance that at least one passes, reaches 1.000 at four runs, so by that
measure the system looks perfect.

![Eight runs, five passing. Pass^k falls to zero while pass@k climbs to one](assets/pass-k.png)

## Use it on your own agents

The scripted agents and `tracing.py` are the parts you replace (grey in the diagram under
How it works). You fill in a few tables and rules inside the scoring files (gold). The rest of
the scoring code stays as it is (blue).

1. Capture real spans. Instrument your framework with OpenTelemetry GenAI spans and turn on
   message content for test runs. In the Python instrumentation that means
   `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY` and
   `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental`. Content can hold personal
   data, which is why it is off by default. In your test, add the SDK's
   `InMemorySpanExporter` the way `tracing.py` does, run one case, and pass the exporter's
   `get_finished_spans()` to `steps_from_spans`. It takes the SDK's span objects, so spans
   saved as JSON or sent to a tracing backend have to be loaded back into that shape first.
2. Point `steps_from_spans` at your attribute keys by editing `OP`, `AGENT`, `TOOL`, `ARGS`
   and `RESULT` at the top of `attribution.py`, and pass it the spans of one trace ID at a
   time. Frameworks differ. Google's ADK, for example, writes tool arguments and results
   under `gcp.vertex.agent.tool_call_args` and `gcp.vertex.agent.tool_response`.
3. Record handoffs the way the code reads them. The conventions have no standard handoff
   event yet, so this harness reads each handoff from a span event named `agent.message`,
   with `from`, `to` and `content` attributes, where `content` holds the facts as JSON. Find
   how your framework records a handoff (often as a tool call) and turn it into that event.
4. Write down what your system must do: a short reference path per agent (`REFS` in
   `run_demo.py`), the facts each receiving agent needs (`NEEDS` in `attribution.py`), a rule
   for each tool argument (`ARG_POLICY` in `scorers.py`), your end-state rules (`end_state`
   in `scorers.py`), and the rules `blame` checks on each tool call (`money_rule` in
   `attribution.py`). As shipped, `money_rule` knows one rule, approval before a refund, so
   replace it with yours or `blame` will find only handoff gaps.

   ![Two calls from the failing run with a rule per argument, and three ways of scoring arguments on two cases. Only a rule for each argument gets both right](assets/argument-rules.png)

5. Test the matcher and the judge. If you change `in_order` or bring your own matcher, copy
   the two trap runs from `traps()` in `run_demo.py` and confirm your matcher fails both.
   Before a real model judge replaces `stub_judge` in a build gate, compare its verdicts with
   runs you labelled yourself.
6. Run each case several times, and report pass^k (`passk.py`) next to the pass rate.
7. When a case fails, find the first wrong step and replay it with the suspect fact
   restored, then fix one thing. Add two tests to CI: the recorded bad run must still get a
   FAIL from the checks, and the fixed system must pass every repeated run.

![The diagnosis loop: capture spans, rebuild the steps, find the first wrong step, blame its owner, name it, replay, then fix one thing and add two tests to CI](assets/diagnosis-loop.png)

Step 5 of that loop names the failure with a [MAST](https://arxiv.org/abs/2503.13657) failure
mode, such as information withholding or ignoring another agent's input, so you can count how
often each kind comes back.

## Files

| File | What it holds |
|---|---|
| `agents.py` | The three scripted agents, their tools, the shared SQLite database, the fake clock, and the code that records each run as spans |
| `tracing.py` | OpenTelemetry setup, with spans kept in memory and saved to `traces/` |
| `attribution.py` | `steps_from_spans`, `handoff_gaps`, `money_rule` and `blame`, printed in the article word for word |
| `scorers.py` | In-order path matching with a rule per argument, the end-state checks, the final-answer check and the stub judge |
| `traps.py` | The two short matchers that pass bad runs |
| `passk.py` | Pass^k and pass@k, with tau-bench's formulas |
| `run_demo.py` | Runs everything and prints every verdict |
| `check_excerpts.py` | Pulls the article's three code blocks out of `attribution.py`, checks their line limits, and runs each alone with only the blocks before it in scope |
| `excerpts/` | The three code blocks `check_excerpts.py` pulls out, one file each. CI fails if they drift |
| `expected/` | The output both scripts print, to diff against |

Every run writes its spans to `traces/`. Span IDs are random and span timestamps come from
the real clock (only the database uses the fake one), so the trace files change each time and
are not committed.

## Limits

The agents are scripted stand-ins that replay fixed steps, so every verdict repeats exactly.
Because no model runs, the harness shows what the checks catch but cannot say how often a
real model fails them. Run the checks on your own system to find that out.

The OpenTelemetry GenAI conventions are still marked Development. This harness names its
handoff event `agent.message`, which is not part of the spec.

`blame` checks two rules: every handoff carries the facts the receiver needs, and approval
comes before money moves. A failure with no rule written for it gets no blame from code.
Asking a model to assign blame does not solve this yet either. In the
[Who&When](https://arxiv.org/abs/2505.00212) study, which tested methods for naming the agent
behind a multi-agent failure, the best one named the right agent 53.5% of the time. Write a
rule for each failure you need blamed.

## Further reading

- [OpenTelemetry GenAI semantic conventions](https://github.com/open-telemetry/semantic-conventions-genai/tree/main/docs/gen-ai), and the [open issue on multi-agent traces](https://github.com/open-telemetry/semantic-conventions-genai/issues/243)
- [tau-bench](https://arxiv.org/abs/2406.12045), the source of pass^k
- [Who&When](https://arxiv.org/abs/2505.00212), on naming the agent and step behind a multi-agent failure
- [MAST](https://arxiv.org/abs/2503.13657), a study of why multi-agent systems fail
- [AgentRewardBench](https://arxiv.org/abs/2504.08942), on how often LLM judges of agent runs are wrong

---

Written by [Satsawat Natakarnkitkul](https://satsawat.ai). Companion article:
*Multi-Agent Trajectory Evaluation, Explained Simply* (Towards AI, in draft). Newsletter:
[AI in Practice](https://satsawat.ai/#subscribe). License: MIT.
