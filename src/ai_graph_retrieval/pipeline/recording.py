"""Recording — the audit + usage records every request leaves behind.

Runs in the request's `finally`: success, pause, or failure, an audit record
(who asked, which documents were served, at what cost) and a usage record are
written. Both are non-blocking and never raise into the request path.
"""
from __future__ import annotations

import structlog

from ai_graph_retrieval.core.access import AccessContext
from ai_graph_retrieval.core.audit import build_audit_entry
from ai_graph_retrieval.core.costing import CostRates, estimate_cost_usd
from ai_graph_retrieval.core.models import Usage
from ai_graph_retrieval.observability.ledger import JsonlLedger
from ai_graph_retrieval.observability.usage_stats import UsageAggregator
from ai_graph_retrieval.pipeline.state import RequestState

logger = structlog.get_logger(__name__)


class RequestRecorder:
    def __init__(
        self,
        usage_ledger: JsonlLedger,
        audit_ledger: JsonlLedger,
        usage_stats: UsageAggregator | None,
        rates: CostRates,
    ) -> None:
        self._usage_ledger = usage_ledger
        self._audit_ledger = audit_ledger
        self._usage_stats = usage_stats
        self._rates = rates

    def record(
        self,
        ctx: AccessContext,
        question: str,
        state: RequestState,
        usage: Usage,
        outcome: str,
    ) -> None:
        report = state.usage
        if report is None:
            report = usage.snapshot()
            report.cost_usd = estimate_cost_usd(report, self._rates)

        entry = build_audit_entry(
            ctx=ctx, question=question, mode=state.mode,
            intent=state.intent, strategy=state.strategy,
            escalated=state.escalated, evidence=state.evidence,
            usage=report, duration_ms=usage.elapsed_ms(), outcome=outcome,
        )
        # The structured log stream is the durable audit channel (SIEM);
        # the file ledgers are optional local sinks.
        logger.info("audit_retrieval", **entry)
        self._audit_ledger.record(entry)
        usage_record = {
            "ts": entry["ts"],
            "request_id": ctx.request_id,
            "tenant_id": ctx.tenant_id,
            "intent": state.intent,
            "strategy": state.strategy,
            "escalated": state.escalated,
            "outcome": outcome,
            "duration_ms": entry["duration_ms"],
            **report.model_dump(),
        }
        self._usage_ledger.record(usage_record)
        if self._usage_stats is not None:
            self._usage_stats.add(usage_record)  # cheap per-replica cumulative (F3)
