"""Request-scoped RunConfig — the resolved per-request overrides the routes hand
to the orchestrator.

Two independent axes, both optional (omit ⇒ today's behavior):
  * ``sources`` / ``embedding_space`` — per-request run selection resolved from
    ``run_registry`` (connectorType + runIdSuffix). ``sources=None`` ⇒ the
    orchestrator uses boot-time ``settings.graph_sources()``; ``embedding_space
    =None`` ⇒ the configured query embedder.
  * ``option_overrides`` — the caller's depth knobs (a subset of RequestOptions,
    only the keys actually provided). Clamped onto the mode's ExecutionProfile.

``resolved_meta`` carries what was actually applied (resolved sources, or the
fallback reason) so the orchestrator can log it and echo it in
``RetrieveResponse.options_applied``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ai_graph_retrieval.settings import GraphSource


@dataclass(frozen=True)
class EmbeddingSpace:
    """The embedding space a resolved run's vectors were built in. ``dimensions``
    is the load-bearing knob — the query MUST embed at the same dim as the index
    it is searched against. ``deployment`` overrides the embedding model name
    (None ⇒ keep the configured default)."""

    dimensions: int | None = None
    deployment: str | None = None


@dataclass(frozen=True)
class RequestRunConfig:
    # Per-request graph sources (None ⇒ settings.graph_sources()).
    sources: tuple[GraphSource, ...] | None = None
    # Per-request query embedding space (None ⇒ configured embedder).
    embedding_space: EmbeddingSpace | None = None
    # Only the depth knobs the caller actually provided, e.g. {"agent_rounds": 3}.
    option_overrides: dict = field(default_factory=dict)
    # Echo/telemetry: {"run_selection": "resolved|fallback|none", ...}.
    resolved_meta: dict = field(default_factory=dict)
