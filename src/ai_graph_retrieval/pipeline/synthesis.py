"""Synthesis stage — compose the answer and propose the next actions.

Synthesis (streamed) runs over the prepared snapshot on its own budget, so a
retrieval timeout upstream still yields an answer from partial evidence.
Suggested next actions are generated AFTER composition, from the answer and the
documents it cited — proposing follow-ups from the evidence pool
instead produced suggestions for topics the answer never established. Actions
fail open — a suggestion problem can never degrade the answer.

`evidence_found` measures EVIDENCE PRESENCE, not answer quality: False when the
snapshot is empty, True when anything was retrieved. It never reads the answer,
so it cannot know whether the answer answered — measured 2026-08-24, it reported
True on 12 of 12 answers that opened "I don't have information about…".
Naming it for what it measures is the point: it is published as the response's
`answer_found` field and gates the suggestions call, and both consumers want
"was there anything to work from", not a verdict on the prose.

Whether the answer actually answered is deliberately UNMEASURED here (product
direction, 2026-08-08): no mode carries an LLM call whose only job is to validate
the finished answer — it added latency and cost without enough value — and a
phrase-match on the answer is forbidden (rule 8).
"""
from __future__ import annotations

import asyncio

import structlog

from ai_graph_retrieval.actions.generator import SuggestedActionGenerator
from ai_graph_retrieval.connectors.llm.client import LlmClient
from ai_graph_retrieval.core.exceptions import RetrievalTimeout
from ai_graph_retrieval.core.models import Evidence, SourceDocument, SuggestedAction, Usage
from ai_graph_retrieval.retrieval.base import AnswerShape, Mode, StrategyContext
from ai_graph_retrieval.settings import Settings
from ai_graph_retrieval.synthesis.composer import AnswerComposer, ComposedAnswer

logger = structlog.get_logger(__name__)


def _has_content(snapshot: Evidence) -> bool:
    """True when the snapshot holds anything an answer could rest on — the
    `evidence_found` signal, and the gate on spending the suggestions call."""
    return bool(
        snapshot.entities or snapshot.facts or snapshot.claims
        or snapshot.quotes or snapshot.sources
    )


class SynthesisStage:
    def __init__(
        self,
        settings: Settings,
        composer: AnswerComposer,
        action_generator: SuggestedActionGenerator | None,
    ) -> None:
        self._s = settings
        self._composer = composer
        # Proposes follow-up actions (None ⇒ feature off); never executes one.
        self._action_generator = action_generator

    async def run(
        self,
        sc: StrategyContext,
        snapshot: Evidence,
        *,
        research: bool,
        strategy: str,
        mode: Mode,
        answer_shape: AnswerShape | None = None,
        quote_ranked: bool = False,
    ) -> tuple[ComposedAnswer, list[SuggestedAction], bool]:
        """The composed answer + suggested actions + the evidence-found signal,
        under the synthesis budget. Research-grade token budget applies on the
        agent lane; ``mode`` is the caller's explicit mode (required — there is
        no default mode) and ``answer_shape`` the classifier's answer-organisation
        verdict, both carried to the composer. ``quote_ranked`` says the quality
        gate ordered the excerpts by relevance, so the digest keeps that order
        instead of re-sorting by document rank. ``strategy`` is the lane name,
        carried only for the timeout log."""
        try:
            async with asyncio.timeout(self._s.synthesis_budget_s):
                composed = await self._composer.compose(
                    sc, snapshot, research=research, mode=mode,
                    answer_shape=answer_shape, quote_ranked=quote_ranked,
                )
                # Suggestions read the finished answer (proposing follow-ups
                # without it produced off-topic ones).
                # Evidence presence, no LLM: nothing retrieved cannot have been
                # answered. The converse does NOT hold — evidence present says
                # nothing about whether the answer used it.
                evidence_found = _has_content(snapshot)
                actions = await self._suggest_actions(
                    sc.question, composed.text, composed.citations, sc.usage,
                    evidence_found=evidence_found,
                    # The request's client, so a per-request client override
                    # covers this call too — the generator
                    # itself is a singleton holding the boot client.
                    llm=sc.llm,
                )
        except TimeoutError as exc:
            logger.warning("synthesis_budget_exceeded",
                           strategy=strategy, **sc.ctx.log_fields())
            raise RetrievalTimeout("Synthesis timed out") from exc
        if not evidence_found:
            logger.info("no_evidence_retrieved", strategy=strategy, **sc.ctx.log_fields())
        return composed, actions, evidence_found

    async def _suggest_actions(
        self,
        question: str,
        answer: str,
        citations: list[SourceDocument],
        usage: Usage,
        *,
        evidence_found: bool,
        llm: LlmClient | None = None,
    ) -> list[SuggestedAction]:
        """Propose next actions (never execute). Additive: any failure yields []
        so a suggestion problem can never degrade the answer.

        Two guards make "additive" true rather than aspirational:

        * with no evidence the answer-found verdict is False by construction, so the
          suggestions would be discarded — don't spend the call at all;
        * its own deadline, far inside the stage budget. Sharing the stage timeout
          meant a hung fast-tier call could exhaust the whole synthesis budget and
          throw away a finished answer, which is the opposite of additive.
        """
        if self._action_generator is None or not evidence_found:
            return []
        try:
            async with asyncio.timeout(self._s.suggested_actions_timeout_s):
                return await self._action_generator.generate(
                    question, answer, citations, usage, llm=llm
                )
        except TimeoutError:
            logger.warning("suggested_actions_timed_out",
                           budget_s=self._s.suggested_actions_timeout_s)
            return []
        except (asyncio.CancelledError, KeyboardInterrupt):
            raise
        except Exception as exc:
            logger.warning("suggested_actions_unavailable", error=str(exc))
            return []
