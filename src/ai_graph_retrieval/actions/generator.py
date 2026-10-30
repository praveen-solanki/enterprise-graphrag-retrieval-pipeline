"""Suggested next actions — one fast-tier LLM call, fully model-driven.

Runs AFTER synthesis, on the composed answer and the documents that answer actually
cited. It used to run concurrently with synthesis on the raw evidence
pool, which is why irrelevant suggestions appeared: the pool's top entities are
whatever ranked highest, and for a question whose subject was never found that is a
list of unrelated people and documents. Proposing follow-ups from that is proposing
follow-ups for a different question. The answer is the only thing that knows what
was actually established, so the answer is what this reads.

One small post-synthesis fast-tier call; token cost stays negligible next to
synthesis, and it is skipped entirely when retrieval surfaced no evidence.

Fail-open by contract: any error, timeout, or malformed output yields an empty list.
Suggestions must never delay or break a served answer.
"""
from __future__ import annotations

import asyncio
import json
import uuid

import structlog

from ai_graph_retrieval.actions.prompts import (
    PROPOSE_ACTIONS_TOOL,
    SUGGESTED_ACTIONS_SYSTEM_PROMPT,
)
from ai_graph_retrieval.connectors.llm.client import LlmClient
from ai_graph_retrieval.core.models import SourceDocument, SuggestedAction, Usage
from ai_graph_retrieval.settings import Settings

logger = structlog.get_logger(__name__)

# How much context the proposer sees. Deliberately small: the answer's opening
# carries what was established, and the cited titles name the subject matter. Named
# constants, not tuning knobs — the shape of the context is a design decision.
#
# A long answer is shown as HEAD + TAIL, not head alone: the closing lines are where
# an answer puts its qualifications ("no information was found about X", "this covers
# only the US entity"), and those are exactly what decides whether a follow-up is
# valid. Truncating them away reintroduces confident suggestions for things the
# answer walked back.
_MAX_ANSWER_HEAD_CHARS = 1_100
_MAX_ANSWER_TAIL_CHARS = 400
_ELLIPSIS = "\n[...]\n"
_MAX_TITLES = 6


class SuggestedActionGenerator:
    """Proposes follow-up actions. Never executes anything."""

    def __init__(self, llm: LlmClient, settings: Settings) -> None:
        self._llm = llm
        self._s = settings

    async def generate(
        self,
        question: str,
        answer: str,
        citations: list[SourceDocument],
        usage: Usage,
        llm: LlmClient | None = None,
    ) -> list[SuggestedAction]:
        """Propose follow-ups for THIS question given the answer it received.

        ``answer``/``citations`` come from synthesis, so a suggestion can only follow
        from something the answer actually said. An empty answer means synthesis
        produced nothing to follow up on, so there is nothing to propose and no call
        worth making.

        ``llm`` is the REQUEST's client when the caller has one: this generator is
        a container singleton built with the boot client, so without the override
        it would ignore any per-request client. None ⇒ the boot client.
        """
        if not self._s.suggested_actions_enabled or not question.strip():
            return []
        if not answer.strip():
            return []
        try:
            message = await (llm or self._llm).complete(
                [
                    {"role": "system", "content": SUGGESTED_ACTIONS_SYSTEM_PROMPT},
                    {"role": "user", "content": self._context(question, answer, citations)},
                ],
                tools=[PROPOSE_ACTIONS_TOOL],
                tool_choice={"type": "function", "function": {"name": "propose_actions"}},
                fast=True,  # latency-sensitive, simple output → fast tier
                usage=usage,
                purpose="suggested_actions",
            )
            args = json.loads(message.tool_calls[0].function.arguments)
        except (asyncio.CancelledError, KeyboardInterrupt):
            raise
        except Exception as exc:  # suggestions are additive — never fail the answer
            logger.warning("suggested_actions_failed", error=str(exc))
            return []
        return self._build(args.get("actions") or [])

    @staticmethod
    def _condense(answer: str) -> str:
        """The answer, bounded but keeping both ends: the opening states the outcome,
        the close carries the qualifications. Short answers pass through untouched."""
        if len(answer) <= _MAX_ANSWER_HEAD_CHARS + _MAX_ANSWER_TAIL_CHARS:
            return answer
        return (
            answer[:_MAX_ANSWER_HEAD_CHARS] + _ELLIPSIS + answer[-_MAX_ANSWER_TAIL_CHARS:]
        )

    def _context(
        self, question: str, answer: str, citations: list[SourceDocument]
    ) -> str:
        """What the user asked and what they were told. The answer states the outcome
        (including "the material does not cover this") and any closing caveat, which is
        what decides whether a follow-up makes sense. Cited titles name the subject
        matter; the raw evidence pool is deliberately NOT shown, because its long tail
        is what produced off-topic suggestions."""
        parts = [f"QUESTION:\n{question}", f"ANSWER GIVEN:\n{self._condense(answer)}"]
        titles = [c.title for c in citations[:_MAX_TITLES] if c.title]
        if titles:
            parts.append("DOCUMENTS THE ANSWER CITED:\n" + "\n".join(f"- {t}" for t in titles))
        return "\n\n".join(parts)

    def _build(self, raw: list) -> list[SuggestedAction]:
        """Validate + normalize the model's proposals into SuggestedAction objects.
        Anything malformed is skipped rather than raising."""
        out: list[SuggestedAction] = []
        if not isinstance(raw, list):
            return out
        for item in raw[: self._s.suggested_actions_max]:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or "").strip()
            if not label:
                continue
            params = item.get("parameters")
            parameters = (
                {str(k): str(v) for k, v in params.items() if v is not None}
                if isinstance(params, dict)
                else {}
            )
            out.append(
                SuggestedAction(
                    # Correlation handle a client can forward to whatever executes it; opaque
                    # by design so it never encodes routing meaning.
                    action_id=uuid.uuid4().hex,
                    label=label,
                    action_type=(str(item.get("action_type") or "").strip() or None),
                    parameters=parameters,
                )
            )
        return out
