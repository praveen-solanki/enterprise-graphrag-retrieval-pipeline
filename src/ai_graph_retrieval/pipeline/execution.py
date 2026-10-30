"""Execution — carry out the planner's decision under its wall-clock budget.

The mode's strategy runs to completion; there is no mid-flight escalation.
Retrieval runs under the mode deadline; synthesis runs AFTER it on its own
budget (`pipeline.synthesis`), so a retrieval timeout can still yield an
answer from partial evidence rather than a hard error.

ONE retrieval architecture serves every mode:

    the mode's strategy  →  the employment gate

Search and Super Research run the agent loop at their profile's budgets, under
ONE deadline, so a mode can spend its extra work but never exceed its wall clock.

The employment gate only exists for graphs that carry a people layer
(``settings.people_graph_enabled``) and runs outside the deadline: people who
have left must not reach the answer whichever path found them, and whether the
lane finished is not a property the rule may depend on (see
retrieval/person_status.py).
"""
from __future__ import annotations

import asyncio

import structlog

from ai_graph_retrieval.core.exceptions import RetrievalTimeout
from ai_graph_retrieval.core.models import EvidencePool
from ai_graph_retrieval.pipeline.planning import ExecutionDecision
from ai_graph_retrieval.retrieval.base import StrategyContext
from ai_graph_retrieval.retrieval.person_status import sweep_inactive_people
from ai_graph_retrieval.settings import Settings

logger = structlog.get_logger(__name__)


def has_evidence(pool: EvidencePool) -> bool:
    """True when the pool holds anything worth synthesizing from."""
    ev = pool.snapshot()
    return bool(
        ev.entities or ev.facts or ev.claims or ev.quotes or ev.sources
    )


class EvidenceCollector:
    def __init__(self, settings: Settings) -> None:
        self._s = settings

    async def collect(self, sc: StrategyContext, decision: ExecutionDecision) -> bool:
        """Fill the evidence pool with the mode's strategy under ONE deadline.
        Returns True when the budget expired but partial evidence survives for
        synthesis; raises RetrievalTimeout when nothing was gathered."""
        try:
            async with asyncio.timeout(decision.profile.budget_s):
                await decision.strategy.run(sc, decision.plan)
        except TimeoutError as exc:
            # No evidence gathered → nothing to synthesize; fail as before.
            # escalated is part of the emitted log schema (always False — no
            # escalation exists); kept so timeout dashboards see stable fields.
            if not (self._s.partial_on_timeout and has_evidence(sc.evidence)):
                logger.warning("retrieval_budget_exceeded",
                               strategy=decision.strategy.name, escalated=False,
                               **sc.ctx.log_fields())
                raise RetrievalTimeout() from exc
            logger.warning("retrieval_timeout_partial",
                           strategy=decision.strategy.name, escalated=False,
                           **sc.ctx.log_fields())
            await self._exclude_inactive_people(sc, decision)
            return True
        await self._exclude_inactive_people(sc, decision)
        # A stage that swallowed a mid-flight failure and continued on what was
        # gathered (agent/runtime) marks the context degraded — same user-facing
        # meaning as a timeout with surviving evidence, so the same `partial`.
        return sc.degraded

    async def _exclude_inactive_people(
        self, sc: StrategyContext, decision: ExecutionDecision
    ) -> None:
        """The employment gate's pool-side half, on every gated path out of retrieval.

        Here rather than inside a lane because it is a property of the evidence,
        not of the lane that found it: a lane cancelled by the deadline above
        still ships what it gathered, and a departed person would reach the
        answer if the rule were the last line of a lane.

        Outside the deadline deliberately — the budget it would inherit is the
        one that just expired, and a sweep skipped for want of time is the leak.
        Its own read is batched and capped (``person_identity_max_names``).

        Never the request's failure: the sweep is subtractive, so a read that
        fails leaves the same evidence the service would have had anyway.
        """
        if not (self._s.people_graph_enabled and decision.profile.employment_gate):
            return
        try:
            await sweep_inactive_people(sc)
        except Exception as exc:
            logger.warning("inactive_people_sweep_failed",
                           error=str(exc)[:200], error_type=type(exc).__name__,
                           **sc.ctx.log_fields())
