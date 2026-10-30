"""Planning — resolve the caller's mode into one executable decision.

Routing is model-driven only where a model is needed (`QueryRouter`, one forced-
function classification for conversational triage, the search query and the
enumeration verdict) and completely separated from execution: this stage turns
the classifier's `RetrievalPlan` into an `ExecutionDecision` — which strategy
runs, under the depth profile the caller's mode bought (`retrieval/profile.py`)
— and nothing here executes retrieval. `pipeline.execution` carries the
decision out.

The MODE is the user's explicit per-query choice and selects the strategy
DIRECTLY — Search or Super Research. There are no intent-keyed lanes: the
classifier's intent is telemetry (it reaches the response and the logs), never
a routing input. Time-bounded and people/topic questions are the agent's job
through its tools, not separate code paths.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

import structlog

from ai_graph_retrieval.connectors.llm.client import LlmClient
from ai_graph_retrieval.core.exceptions import RetrievalTimeout
from ai_graph_retrieval.core.models import Usage
from ai_graph_retrieval.retrieval.base import (
    ConversationalTurn,
    Mode,
    RetrievalPlan,
    RetrievalStrategy,
)
from ai_graph_retrieval.retrieval.profile import ExecutionProfile, profile_for
from ai_graph_retrieval.retrieval.router import QueryRouter
from ai_graph_retrieval.settings import Settings

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class ExecutionDecision:
    """Everything execution needs, resolved in one place: the classified plan,
    the strategy that will run, and the execution profile it runs under —
    the caller's mode resolved to its deadline and every depth cap
    (``retrieval/profile.py``)."""

    plan: RetrievalPlan
    strategy: RetrievalStrategy
    profile: ExecutionProfile


class ExecutionPlanner:
    def __init__(
        self,
        router: QueryRouter,
        settings: Settings,
        search_strategy: RetrievalStrategy,
        research_strategy: RetrievalStrategy,
    ) -> None:
        self._router = router
        self._s = settings
        self._search = search_strategy
        self._research = research_strategy

    async def resolve(
        self,
        question: str,
        history: list[dict],
        mode: Mode,
        usage: Usage,
        llm: LlmClient | None = None,
        settings: Settings | None = None,
    ) -> ConversationalTurn | ExecutionDecision:
        """Route (small, separately-bounded; one LLM classification) and resolve
        the entry: a ConversationalTurn short-circuits the pipeline with the
        classifier's reply; otherwise the strategy + profile to execute.

        ``llm`` overrides the classifier's client for this request (a
        per-request client); None ⇒ the router's own.
        ``settings`` is the per-request view (short-circuit / verify-the-decline
        overrides); None ⇒ the planner's boot-time settings, which the router
        already holds."""
        try:
            outcome = await asyncio.wait_for(
                self._router.plan(
                    question, history, mode, usage, llm=llm, settings=settings,
                ),
                timeout=self._s.router_timeout_s,
            )
        except TimeoutError as exc:
            # The entry classifier (fast tier) did not
            # answer within router_timeout_s — the wait_for cancels it, so the
            # provider's own error never surfaces (CancelledError is a BaseException,
            # slips past the router's except-Exception). Log the stage + budget here
            # so a planning stall is not indistinguishable from a retrieval timeout.
            logger.error(
                "query_planning_timeout",
                timeout_s=self._s.router_timeout_s,
                mode=mode.value if mode else None,
            )
            raise RetrievalTimeout("Query planning timed out") from exc
        if isinstance(outcome, ConversationalTurn):
            return outcome
        plan = outcome
        profile = profile_for(self._s, mode, plan.intent)
        if mode is Mode.SEARCH:
            return ExecutionDecision(plan=plan, strategy=self._search, profile=profile)
        return ExecutionDecision(plan=plan, strategy=self._research, profile=profile)
