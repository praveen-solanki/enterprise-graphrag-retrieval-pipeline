"""The per-request record the pipeline stages fill in as a request progresses.

One instance per request, created at admission and threaded through the stages;
the audit/usage records (`pipeline.recording`) read it in the `finally` of the
request lifecycle, so it must hold whatever those records need even when the
request failed part-way.
"""
from __future__ import annotations

from dataclasses import dataclass

from ai_graph_retrieval.core.models import Evidence, UsageReport


@dataclass
class RequestState:
    """What the audit/usage records need, filled as the request progresses."""

    # Always the caller's explicit mode (there is no default mode); set at
    # admission from the validated request.
    mode: str
    intent: str | None = None
    strategy: str | None = None
    escalated: bool = False  # retained for API/audit contract; always False (no escalation)
    partial: bool = False  # answer synthesized from partial evidence after a retrieval timeout
    evidence: Evidence | None = None  # set only once a response was assembled
    usage: UsageReport | None = None
