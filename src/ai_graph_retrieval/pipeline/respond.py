"""Respond — the one place the response payload is shaped.

Mirrors `api.schemas.RetrieveResponse`. Both the normal path and the
conversational short-circuit build their payload here, so the two can never
drift apart.
"""
from __future__ import annotations

from ai_graph_retrieval.core.access import AccessContext
from ai_graph_retrieval.core.models import (
    Evidence,
    SourceDocument,
    SuggestedAction,
    UsageReport,
)
from ai_graph_retrieval.pipeline.state import RequestState


def build_payload(
    ctx: AccessContext,
    state: RequestState,
    *,
    answer: str,
    evidence: Evidence,
    usage_report: UsageReport,
    duration_ms: int,
    trace: dict | None = None,
    citations: list[SourceDocument] | None = None,
    suggested_actions: list[SuggestedAction] | None = None,
    answer_found: bool = True,
    answer_graph: dict | None = None,
) -> dict:
    """The final response payload (one shape for every path).

    ``citations`` are the documents the answer actually links inline (ranked
    best first). ``None`` — used by paths that never ran synthesis — falls back
    to the evidence's ranked source list (empty for conversational turns).

    ``answer_found`` — EVIDENCE PRESENCE, despite the field name: False when
    retrieval surfaced nothing at all, True when it surfaced anything. It does not
    read the answer, so it is NOT a verdict on whether the answer answered — it
    reports True on answers that open "I don't have information about…". The field
    keeps its name because clients branch on it; the trace score it feeds is emitted
    as ``evidence_found``, which is what it measures. Conversational replies keep
    the True default — the reply served the turn, and `strategy` already tells a
    client it was conversational."""
    cited = evidence.sources if citations is None else citations
    return {
        "request_id": ctx.request_id,
        "intent": state.intent,
        "strategy": state.strategy,
        # The executed mode — always the caller's own choice; there is no
        # automatic mode selection and no next-mode affordance.
        "mode": state.mode,
        # False ⇒ retrieval surfaced no evidence; the answer text explains and
        # the UI may render a distinct no-answer state. Evidence presence only,
        # never a phrase-match on the answer and never a judge LLM — so True
        # does NOT promise the answer answered.
        "answer_found": answer_found,
        "escalated": state.escalated,
        "partial": state.partial,
        "answer": answer,
        "citations": [s.model_dump() for s in cited],
        "evidence": evidence.model_dump(),
        "trace": trace,
        "usage": usage_report.model_dump(),
        "duration_ms": duration_ms,
        # Proposed only — this service never executes an action.
        "suggested_actions": [a.model_dump() for a in (suggested_actions or [])],
        # The entities the answer names and the relationships among them
        # (synthesis/answer_graph.py) — what a client draws as the graph behind
        # THIS answer. `evidence` stays the full pool; this is the subset of it
        # the answer actually rests on. Empty when nothing the pool holds is named.
        "answer_graph": answer_graph or {"nodes": [], "edges": []},
    }
