"""Typed SSE stream events — the contract the Research UI consumes.

Every event serializes as ``event: <type>`` + a JSON data payload. The frontend
discriminates on the SSE event name (mirrors the POC's trace_schema, simplified).
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

from ai_graph_retrieval.core.models import TraceEdge, TraceNode, TraceStep


class StreamEvent(BaseModel):
    type: str

    def sse(self) -> dict[str, str]:
        return {"event": self.type, "data": self.model_dump_json(exclude={"type"})}


class MetaEvent(StreamEvent):
    type: Literal["meta"] = "meta"
    request_id: str
    intent: str
    strategy: str


class StepEvent(StreamEvent):
    type: Literal["step"] = "step"
    step: TraceStep


class TraceDeltaEvent(StreamEvent):
    type: Literal["trace_delta"] = "trace_delta"
    nodes: list[TraceNode] = []
    edges: list[TraceEdge] = []


class AnswerDeltaEvent(StreamEvent):
    type: Literal["answer_delta"] = "answer_delta"
    delta: str


class ClarificationRequestEvent(StreamEvent):
    """Emitted (turn 1) when the service needs the user to clarify before answering.
    The UI renders `questions` (options + a free-text "Other") and replies via
    POST /v1/graph/retrieve/resume with the same conversation_id."""

    type: Literal["clarification_request"] = "clarification_request"
    request_id: str
    conversation_id: str
    kind: Literal["clarification", "plan_review"] = "clarification"
    understanding: str | None = None  # populated for plan_review (D3)
    scope: list[str] = []
    questions: list[dict[str, Any]] = []


class FinalEvent(StreamEvent):
    type: Literal["final"] = "final"
    payload: dict[str, Any]


class ErrorEvent(StreamEvent):
    type: Literal["error"] = "error"
    code: str
    message: str
