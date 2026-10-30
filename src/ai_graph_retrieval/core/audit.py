"""Audit entry construction — the forensic record of every retrieval decision.

Carried from the POC's audit ledger (final_recommendations §Operate): each
request records *which documents were served to whom* so an ACL incident can be
investigated after the fact. Redaction rules:

  * principal tokens are NEVER recorded — only their count and a stable
    SHA-256 digest of the sorted set (enough to correlate, impossible to
    reverse);
  * the question is recorded as a digest, not text (questions can embed PII).

Entries flow to the structured log stream (always — that is the SIEM channel)
and optionally to the audit JSONL ledger file.
"""
from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import UTC, datetime

from ai_graph_retrieval.core.access import AccessContext
from ai_graph_retrieval.core.models import Evidence, UsageReport


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def principals_digest(principals: Sequence[str]) -> str:
    return _digest("\n".join(sorted(principals)))


def build_audit_entry(
    *,
    ctx: AccessContext,
    question: str,
    mode: str,
    intent: str | None,
    strategy: str | None,
    escalated: bool,
    evidence: Evidence | None,
    usage: UsageReport | None,
    duration_ms: int,
    outcome: str,
) -> dict:
    """One flat, JSON-safe audit record (used for both log line and ledger)."""
    sources = evidence.sources if evidence else []
    return {
        "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
        "request_id": ctx.request_id,
        "tenant_id": ctx.tenant_id,
        "principal_count": len(ctx.principals),
        "principals_sha256": principals_digest(ctx.principals),
        "question_sha256": _digest(question),
        "mode": mode,
        "intent": intent,
        "strategy": strategy,
        "escalated": escalated,
        "doc_ids_served": [s.doc_id for s in sources if s.doc_id],
        "entities": len(evidence.entities) if evidence else 0,
        "facts": len(evidence.facts) if evidence else 0,
        "llm_calls": usage.llm_calls if usage else 0,
        "prompt_tokens": usage.prompt_tokens if usage else 0,
        "completion_tokens": usage.completion_tokens if usage else 0,
        "cost_usd": usage.cost_usd if usage else 0.0,
        "duration_ms": duration_ms,
        "outcome": outcome,
    }
