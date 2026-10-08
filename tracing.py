"""Record agent runs as OpenTelemetry spans, all in memory.

Span and attribute names follow the OpenTelemetry GenAI semantic
conventions (v1.44.0, status: Development):
  invoke_agent {gen_ai.agent.name}   one per agent, kind INTERNAL;
                                     sub-agents nest under the orchestrator
  chat {gen_ai.request.model}        one per model turn
  execute_tool {gen_ai.tool.name}    one per tool call, kind INTERNAL
Messages that cross between agents are span events named
"agent.message" (from, to, content as JSON) and are mirrored into the
opt-in gen_ai.input.messages / gen_ai.output.messages attributes.
No exporter talks to the network. Spans land in an InMemorySpanExporter
and can be written to a local JSON file.
"""
import json
from pathlib import Path

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter)
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as G

EXPORTER = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(EXPORTER))
TRACER = _provider.get_tracer("refund-harness")

# Attribute keys used below, checked against the installed semconv package.
OP = G.GEN_AI_OPERATION_NAME
AGENT_NAME = G.GEN_AI_AGENT_NAME
CONVERSATION_ID = G.GEN_AI_CONVERSATION_ID
PROVIDER = G.GEN_AI_PROVIDER_NAME
MODEL = G.GEN_AI_REQUEST_MODEL
FINISH = G.GEN_AI_RESPONSE_FINISH_REASONS
TOOL_NAME = G.GEN_AI_TOOL_NAME
TOOL_CALL_ID = G.GEN_AI_TOOL_CALL_ID
TOOL_TYPE = G.GEN_AI_TOOL_TYPE
TOOL_ARGS = G.GEN_AI_TOOL_CALL_ARGUMENTS
TOOL_RESULT = G.GEN_AI_TOOL_CALL_RESULT
INPUT_MESSAGES = G.GEN_AI_INPUT_MESSAGES
OUTPUT_MESSAGES = G.GEN_AI_OUTPUT_MESSAGES
MESSAGE_EVENT = "agent.message"   # custom event name, not in the spec


def take_spans():
    """Return the spans finished since the last call, then clear."""
    spans = list(EXPORTER.get_finished_spans())
    EXPORTER.clear()
    return spans


def save_json(spans, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data = [json.loads(s.to_json()) for s in spans]
    Path(path).write_text(json.dumps(data, indent=2))


def span_counts(spans):
    counts = {}
    for s in spans:
        op = s.attributes.get(OP)
        counts[op] = counts.get(op, 0) + 1
    return counts
