"""Gate — the conditional pre-retrieval pause.

ONE gate, driven entirely by the entry classifier's own verdict: the same
fast-tier call that classifies the turn also judges whether a genuine
answer-changing ambiguity exists and, only then, writes 1-2 targeted
clarification questions for THIS query (`plan.clarification_questions`).
There is no separate planning/gate LLM, no predefined question templates, and
no mandatory review stage — clarification is conditional, not a pipeline step,
so the common case pays nothing and pauses never.

Which modes may pause is the SINGLE_ROUND_TRIP_MODES set (retrieval/base.py): it is
empty, so both Search and Super Research may pause. The pause raises
`MidResearchClarification` (a control-flow signal, not an error); the route layer
persists a session and answers 202 with the questions, and the `/retrieve/resume` turn
reruns with ``plan_review=False`` so answering never re-triggers the gate. The pause
happens BEFORE any retrieval spend, so nothing is discarded by pausing.

Fail-safe direction: everything about the path is proceed-biased. The
classifier is told an unnecessary question is the worse failure; a malformed
questions payload parses to none (router._clarifications); and the kill-switch
(``search_clarification_enabled``) turns the pause off without touching
classification.
"""
from __future__ import annotations

import structlog

from ai_graph_retrieval.clarification.models import (
    MidResearchClarification,
    build_question,
)
from ai_graph_retrieval.core.access import AccessContext
from ai_graph_retrieval.retrieval.base import SINGLE_ROUND_TRIP_MODES, Mode, RetrievalPlan
from ai_graph_retrieval.settings import Settings

logger = structlog.get_logger(__name__)


class ClarificationGates:
    def __init__(self, settings: Settings) -> None:
        self._s = settings

    def enforce(
        self,
        ctx: AccessContext,
        *,
        plan: RetrievalPlan,
        mode: Mode,
        plan_review: bool,
    ) -> None:
        """Raise the pause when the user must answer first; return to proceed.

        ``plan_review=False`` is the resume turn's guard: the enriched question
        already carries the user's answers, so it must never be re-questioned.
        Synchronous — the gate spends no I/O and no LLM; the classifier already
        paid for the verdict.
        """
        if not plan_review or not self._s.search_clarification_enabled:
            return
        if mode in SINGLE_ROUND_TRIP_MODES:
            return
        rows = plan.clarification_questions
        if not rows:
            return
        questions = [
            q.model_copy(update={"id": f"q{i}"})
            for i, q in enumerate(
                build_question(row["question"], row.get("options")) for row in rows
            )
        ]
        logger.info(
            "clarification_requested",
            source="entry_classifier", mode=mode.value, questions=len(questions),
            **ctx.log_fields(),
        )
        raise MidResearchClarification(questions)
