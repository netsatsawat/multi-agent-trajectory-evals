"""Check which agent the orchestrator handed each piece of work to.

When a model picks the next agent, many frameworks record the pick as a
tool call that names the target, so the route is already in the trace.
Here it is the `to` field of each agent.message event. The article
prints the block between the markers word for word. The helpers below
it score routing over a set of cases.
"""
from collections import Counter
from math import sqrt

# >>> excerpt 5
def wrong_route(steps, want, router="orchestrator"):
    """First handoff to an agent other than the one the case
    needs, as (step, agent, what, reasons), or None."""
    k = 0
    for n, st in enumerate(steps, 1):
        if (st["kind"] != "message" or st["agent"] != router
                or st["to"] == "customer"):
            continue
        need = want[k] if k < len(want) else "no more handoffs"
        if st["to"] != need:
            return (n, router, f"handoff to {st['to']}",
                    [f"want {need}"])
        k += 1
    if k < len(want):
        return (len(steps) + 1, router, "no handoff",
                [f"want {want[k]}"])
    return None
# <<< excerpt


def route_of(steps, router="orchestrator"):
    """The agents the router handed work to, in order."""
    return [st["to"] for st in steps if st["kind"] == "message"
            and st["agent"] == router and st["to"] != "customer"]


def route_table(rows):
    """Count (intention, route taken) pairs: the confusion table.

    `rows` holds one (intention, route) pair per case, with the route
    as a tuple of agent names.
    """
    return Counter(rows)


def wilson(k, n, z=1.96):
    """Wilson score interval for k successes out of n (95% by default).

    Use it for routing accuracy on cases sampled from real traffic. It
    stays inside 0 to 1 on small case sets, where the usual p +/- z*se
    interval can run past either end. An interval means nothing on
    cases written by hand, such as the 12 in team.py, so run_ext.py
    does not print one.
    """
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half
