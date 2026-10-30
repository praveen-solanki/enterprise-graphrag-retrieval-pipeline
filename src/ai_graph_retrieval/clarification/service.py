"""ClarificationService — the single entry point the routes use for the
pause/resume flow.

Wraps the durable session store and owns the turn-1 (open) / turn-2 (resume)
coordination. No LLM lives here: clarification questions are written by the
entry classifier on the call it already pays for (retrieval/router.py), and
this service only moves sessions.
"""
from __future__ import annotations

import structlog

from ai_graph_retrieval.clarification.models import (
    AnswerItem,
    ClarificationQuestion,
    ClarificationSession,
)
from ai_graph_retrieval.clarification.store import ClarificationSessionStore
from ai_graph_retrieval.core.access import AccessContext
from ai_graph_retrieval.core.exceptions import InvalidRequest
from ai_graph_retrieval.settings import Settings

logger = structlog.get_logger(__name__)


class ClarificationService:
    """Session handling for the pause/resume flow. Spends NO LLM: the entry
    classifier writes any clarification questions on the call it already pays
    for, and this class only persists the session (turn 1), consumes it (the
    resume turn), and composes the enriched question."""

    def __init__(self, settings: Settings, store: ClarificationSessionStore) -> None:
        self._s = settings
        self._store = store

    async def start(self) -> None:
        self._store.start()

    async def aclose(self) -> None:
        await self._store.aclose()

    async def open_session(
        self,
        ctx: AccessContext,
        *,
        conversation_id: str,
        user_id: str | None,
        question: str,
        history: list[dict],
        questions: list[ClarificationQuestion],
        chat_conversation_id: str | None = None,
    ) -> ClarificationSession:
        """Persist the pending session."""
        session = ClarificationSession(
            conversation_id=conversation_id,
            tenant_id=ctx.tenant_id,
            user_id=user_id,
            original_question=question,
            history=history,
            questions=questions,
            chat_conversation_id=chat_conversation_id,
        )
        await self._store.open(session)
        logger.info(
            "clarification_opened",
            conversation_id=conversation_id,
            questions=len(questions),
            **ctx.log_fields(),
        )
        return session

    async def consume(self, ctx: AccessContext, conversation_id: str) -> ClarificationSession:
        """Atomically claim the session for this `/resume`. Fail-closed on miss."""
        session = await self._store.consume(ctx.tenant_id, conversation_id)
        if session is None:
            raise InvalidRequest("Unknown, expired, or already-answered conversation_id")
        return session

    @staticmethod
    def compose_enriched_question(session: ClarificationSession, answers: list[AnswerItem]) -> str:
        """Fold the original question + the user's answers into one enriched question."""
        by_id = {q.id: q for q in session.questions}
        lines = [f"Original question: {session.original_question}", "", "The user clarified:"]
        for a in answers:
            q = by_id.get(a.question_id)
            q_text = q.question if q else a.question_id
            labels: list[str] = []
            if q:
                label_by_id = {o.id: o.label for o in q.options}
                labels = [label_by_id.get(oid, oid) for oid in a.selected_option_ids]
            parts = [*labels]
            if a.free_text and a.free_text.strip():
                parts.append(a.free_text.strip())
            lines.append(f"- {q_text} -> {', '.join(p for p in parts if p) or '(no answer)'}")
        return "\n".join(lines)
