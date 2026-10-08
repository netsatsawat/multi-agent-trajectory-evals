import json
OP, AGENT = "gen_ai.operation.name", "gen_ai.agent.name"
TOOL, ARGS = "gen_ai.tool.name", "gen_ai.tool.call.arguments"
RESULT = "gen_ai.tool.call.result"
def steps_from_spans(spans):
    """Flatten a multi-agent trace into time-ordered steps."""
    by_id = {s.context.span_id: s for s in spans}
    def owner(s):  # nearest invoke_agent span at or above s
        while s.attributes.get(OP) != "invoke_agent":
            s = by_id[s.parent.span_id]
        return s.attributes[AGENT]
    steps = []
    for s in spans:
        if s.attributes.get(OP) == "execute_tool":
            steps.append((s.start_time, {
                "kind": "tool", "agent": owner(s),
                "tool": s.attributes[TOOL],
                "args": json.loads(s.attributes[ARGS]),
                "result": json.loads(s.attributes[RESULT])}))
        for e in s.events:
            if e.name == "agent.message":
                a = e.attributes
                steps.append((e.timestamp, {
                    "kind": "message", "agent": a["from"],
                    "to": a["to"],
                    "facts": json.loads(a["content"])}))
    return [st for _, st in sorted(steps, key=lambda t: t[0])]
