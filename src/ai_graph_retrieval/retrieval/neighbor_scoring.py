"""Tiered neighbor scoring — budgeted LLM re-ranking for ambiguous hubs.

Carried from the POC's neighbor_scoring_service, which validated the gates
empirically. The cheap tier (cosine + log-support, computed in Cypher) is
always the baseline; the LLM tier fires only when ALL of:

  1. the fanout is genuinely large (> scoring_small_threshold), and
  2. the cheap ranking is ambiguous — the score gap at the keep-boundary is
     below scoring_separation_margin (a clean separation needs no LLM), and
  3. the per-request scoring budget has calls remaining.

Failure contract: the scorer NEVER degrades a request — any LLM error or
timeout returns the cheap order. Latency is bounded by scoring_timeout_s per
call; cost is bounded by the request's ``ExecutionProfile.scoring_budget``
(Super Research scoring_query_budget, Search search_scoring_budget —
a mode with 0 never builds a scorer at all;
see retrieval/deep_research.py::build_scorer).
"""
from __future__ import annotations

import asyncio
import json

import structlog

from ai_graph_retrieval.connectors.llm.client import LlmClient
from ai_graph_retrieval.core.models import RelationFact, Usage
from ai_graph_retrieval.settings import Settings

logger = structlog.get_logger(__name__)

_SCORE_TOOL = {
    "type": "function",
    "function": {
        "name": "score_relationships",
        "description": "Rate how relevant each candidate relationship is to the focus.",
        "parameters": {
            "type": "object",
            "properties": {
                "scores": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "index": {"type": "integer"},
                            "relevance": {"type": "number", "minimum": 0, "maximum": 1},
                        },
                        "required": ["index", "relevance"],
                    },
                }
            },
            "required": ["scores"],
        },
    },
}

_SYSTEM = (
    "You score how relevant each candidate relationship is to the given focus. "
    "For EVERY candidate index, return a relevance in [0, 1]: 1.0 directly helps "
    "answer the focus, 0.5 related but tangential, 0.0 unrelated noise. "
    "Call score_relationships exactly once with a score for every index."
)

# Blend: the LLM's judgment dominates; the cheap score breaks ties.
_W_LLM = 0.9
_W_CHEAP = 0.1

_CHARS_PER_TOKEN = 4  # rough packing estimate; only used to size batches


class ScoringBudget:
    """Per-request cap on LLM scoring calls (mutated only by the owning task)."""

    def __init__(self, calls: int) -> None:
        self._remaining = max(0, calls)

    @property
    def remaining(self) -> int:
        return self._remaining

    def try_consume(self) -> bool:
        if self._remaining <= 0:
            return False
        self._remaining -= 1
        return True


class NeighborScorer:
    def __init__(
        self, llm: LlmClient, settings: Settings, usage: Usage, budget: ScoringBudget
    ) -> None:
        self._llm = llm
        self._s = settings
        self._usage = usage
        self._budget = budget

    async def rescore(
        self, facts: list[RelationFact], focus: str, keep: int
    ) -> tuple[list[RelationFact], str]:
        """Return (top-`keep` facts, tier) — tier is "cheap" or "llm".

        The candidates arrive as the repository's fan-out concatenation — every
        row of the first graph, then the second, each block already score-ordered
        — so the list is NOT globally ranked. Every path below reads it
        positionally (``facts[:keep]``, and the boundary gap the escalation gate
        measures), which let the first graph fill `keep` and put both boundary
        candidates inside that one graph's block. The cheap score
        (``support_weight * log10(1+support)``, cypher.EXPAND_NEIGHBORS) is the
        same curve on every graph, so one sort makes the cheap order a real
        ranking and the boundary a real boundary.
        """
        facts = self._cheap_ranked(facts)
        if not self._should_escalate(facts, keep):
            return facts[:keep], "cheap"

        batches = self._pack(facts)
        if self._budget.remaining < len(batches):
            return facts[:keep], "cheap"  # not enough budget for the whole pack
        for _ in batches:
            self._budget.try_consume()

        try:
            async with asyncio.timeout(self._s.scoring_timeout_s):
                results = await asyncio.gather(
                    *(self._score_batch(batch, focus) for batch in batches)
                )
        except Exception as exc:  # never degrade the expansion — cheap order stands
            logger.warning("neighbor_scoring_failed_using_cheap", error=str(exc))
            return facts[:keep], "cheap"

        relevance: dict[int, float] = {}
        for offset, scores in results:
            for idx, rel in scores.items():
                relevance[offset + idx] = rel

        max_cheap = max((f.score for f in facts), default=1.0) or 1.0

        def _blended(pair: tuple[int, RelationFact]) -> tuple:
            idx, fact = pair
            cheap = fact.score / max_cheap
            # A candidate the model left out of its `scores` array is UNSCORED,
            # not irrelevant. Defaulting it to 0.0 zeroed the term carrying 90% of
            # the weight, so an omission silently deleted the row — and structural
            # edges, which sit at the cheap floor, were the first to go. Falling
            # back to its own cheap score leaves an unmentioned row where the
            # cheap ranking already put it.
            llm = relevance.get(idx)
            # The name tiebreak matches _cheap_ranked: without it equal blends fell
            # back to enumerate() order, so the same candidates ranked differently
            # when an earlier tool put them in the list in another order.
            return (not fact.structural,
                    -(_W_LLM * (cheap if llm is None else llm) + _W_CHEAP * cheap),
                    fact.target or "", fact.source or "")

        ranked = sorted(enumerate(facts), key=_blended)
        return [f for _, f in ranked[:keep]], "llm"

    # ── gates ──────────────────────────────────────────────────────────────

    @staticmethod
    def _cheap_ranked(facts: list[RelationFact]) -> list[RelationFact]:
        """The cheap ranking, applied across every graph rather than within each.

        Mirrors the per-graph ordering the expansion query already applies:
        structure first (a schema-asserted edge scores at the support curve's
        floor by construction, so ranking it on that curve puts the org backbone
        last and truncates it first), then ``score``; ``support`` breaks a score
        tie (the two move together, so this only orders rows the rounding
        collapsed), then the neighbour name keeps two identical requests
        identically ordered.
        """
        return sorted(
            facts,
            key=lambda f: (not f.structural, -f.score, -f.support, f.target or ""),
        )

    def _should_escalate(self, facts: list[RelationFact], keep: int) -> bool:
        if len(facts) <= max(self._s.scoring_small_threshold, keep):
            return False  # small fanout — cheap rank is fine
        boundary_gap = facts[keep - 1].score - facts[keep].score
        if boundary_gap >= self._s.scoring_separation_margin:
            return False  # cleanly separated — no ambiguity to resolve
        return self._budget.remaining > 0

    # ── batching + scoring ─────────────────────────────────────────────────

    def _pack(self, facts: list[RelationFact]) -> list[tuple[int, list[RelationFact]]]:
        """Split candidates into (offset, batch) chunks under the token budget."""
        char_budget = self._s.scoring_batch_token_budget * _CHARS_PER_TOKEN
        batches: list[tuple[int, list[RelationFact]]] = []
        start, used = 0, 0
        for i, fact in enumerate(facts):
            cost = len(self._line(i - start, fact))
            if used + cost > char_budget and i > start:
                batches.append((start, facts[start:i]))
                start, used = i, 0
            used += cost
        batches.append((start, facts[start:]))
        return batches

    @staticmethod
    def _line(idx: int, f: RelationFact) -> str:
        desc = (f.description or "")[:160]
        return f"{idx}. {f.source} {f.relationship} {f.target} — {desc}"

    async def _score_batch(
        self, batch: tuple[int, list[RelationFact]], focus: str
    ) -> tuple[int, dict[int, float]]:
        offset, facts = batch
        lines = "\n".join(self._line(i, f) for i, f in enumerate(facts))
        message = await self._llm.complete(
            [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": f"FOCUS:\n{focus}\n\nCANDIDATES:\n{lines}"},
            ],
            tools=[_SCORE_TOOL],
            tool_choice={"type": "function", "function": {"name": "score_relationships"}},
            fast=True,  # relationship re-ranking is a lightweight decision → fast tier
            usage=self._usage,
            purpose="neighbour_rerank",
        )
        args = json.loads(message.tool_calls[0].function.arguments)
        scores: dict[int, float] = {}
        for item in args.get("scores", []):
            idx = int(item["index"])
            if 0 <= idx < len(facts):
                scores[idx] = min(1.0, max(0.0, float(item["relevance"])))
        return offset, scores
