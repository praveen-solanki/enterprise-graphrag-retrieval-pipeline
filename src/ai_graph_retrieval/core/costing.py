"""Cost estimation — pure functions over a UsageReport.

Rates come from settings (USD per 1M tokens) so pricing changes are config,
never code. A deployment with no configured rate contributes 0 — the report
still carries exact token counts, so cost can always be recomputed offline.
"""
from __future__ import annotations

from dataclasses import dataclass

from ai_graph_retrieval.core.models import UsageReport
from ai_graph_retrieval.settings import Settings

_PER_1M = 1_000_000


@dataclass(frozen=True)
class CostRates:
    """Token rates for the two chat tiers (``llm_model`` and ``fast_model``) plus
    embeddings, resolved once from settings. Usage is bucketed per model name, so
    each bucket is priced at its own tier's rate."""

    fast_deployment: str
    chat_input_per_1m: float
    chat_output_per_1m: float
    chat_cached_input_per_1m: float
    fast_input_per_1m: float
    fast_output_per_1m: float
    fast_cached_input_per_1m: float
    embedding_per_1m: float

    @classmethod
    def from_settings(cls, s: Settings) -> CostRates:
        # Cached-input rate falls back to the full input rate when unset, and the
        # fast rates fall back to the chat rate, so a missing rate never
        # *under*-reports spend.
        chat_cached = (
            s.cost_chat_cached_input_per_1m
            if s.cost_chat_cached_input_per_1m is not None
            else s.cost_chat_input_per_1m
        )
        fast_in = (
            s.cost_fast_input_per_1m
            if s.cost_fast_input_per_1m is not None
            else s.cost_chat_input_per_1m
        )
        fast_out = (
            s.cost_fast_output_per_1m
            if s.cost_fast_output_per_1m is not None
            else s.cost_chat_output_per_1m
        )
        fast_cached = (
            s.cost_fast_cached_input_per_1m
            if s.cost_fast_cached_input_per_1m is not None
            else fast_in
        )
        return cls(
            fast_deployment=s.fast_model if s.fast_model != s.llm_model else "",
            chat_input_per_1m=s.cost_chat_input_per_1m,
            chat_output_per_1m=s.cost_chat_output_per_1m,
            chat_cached_input_per_1m=chat_cached,
            fast_input_per_1m=fast_in,
            fast_output_per_1m=fast_out,
            fast_cached_input_per_1m=fast_cached,
            embedding_per_1m=s.cost_embedding_per_1m,
        )

    def _rates_for(self, deployment: str) -> tuple[float, float, float]:
        """(input, output, cached_input) rates for a model bucket — the fast tier
        for ``fast_model`` (when it differs from ``llm_model``), the chat tier for
        everything else."""
        if deployment == self.fast_deployment:
            return self.fast_input_per_1m, self.fast_output_per_1m, self.fast_cached_input_per_1m
        return self.chat_input_per_1m, self.chat_output_per_1m, self.chat_cached_input_per_1m


def _prompt_cost(
    prompt_tokens: int, cached_tokens: int, rate_in: float, rate_cached: float
) -> float:
    """Price uncached prompt tokens at the full rate, cached ones at the discount."""
    cached = max(0, min(cached_tokens, prompt_tokens))
    uncached = prompt_tokens - cached
    return (uncached * rate_in + cached * rate_cached) / _PER_1M


def _tier_costs(
    rates: CostRates, deployment: str,
    prompt_tokens: int, completion_tokens: int, cached_prompt_tokens: int,
) -> tuple[float, float]:
    """Unrounded ``(input_cost, output_cost)`` for one call at ``deployment``'s tier —
    the single source of truth for token→USD, shared by ``call_cost_usd`` (per-span) and
    ``estimate_cost_usd`` (per-request) so the two can never diverge."""
    rate_in, rate_out, rate_cached = rates._rates_for(deployment)
    return (
        _prompt_cost(prompt_tokens, cached_prompt_tokens, rate_in, rate_cached),
        completion_tokens * rate_out / _PER_1M,
    )


def call_cost_usd(
    rates: CostRates,
    deployment: str,
    *,
    prompt_tokens: int,
    completion_tokens: int,
    cached_prompt_tokens: int = 0,
) -> tuple[float, float, float]:
    """``(input_cost, output_cost, total_cost)`` in USD for a SINGLE llm call, priced at
    ``deployment``'s tier rate (via ``_tier_costs``). Cached prompt tokens bill at the
    discounted rate; the rest at the full input rate."""
    input_cost, output_cost = _tier_costs(
        rates, deployment, prompt_tokens, completion_tokens, cached_prompt_tokens
    )
    return round(input_cost, 8), round(output_cost, 8), round(input_cost + output_cost, 8)


def embedding_cost_usd(rates: CostRates, tokens: int) -> float:
    """USD for a single embedding call (0 when the embedding rate is unconfigured)."""
    return round(tokens * rates.embedding_per_1m / _PER_1M, 8)


def estimate_cost_usd(report: UsageReport, rates: CostRates) -> float:
    """Blend per-deployment chat spend with embedding spend, rounded to 6 dp.

    Prompt tokens served from the provider's cache (OpenAI-compatible
    ``prompt_tokens_details.cached_tokens``) are billed at the discounted
    cached rate; the rest at the full input rate.
    """
    total = 0.0
    if report.by_deployment:
        for name, bucket in report.by_deployment.items():
            input_cost, output_cost = _tier_costs(
                rates, name, bucket.prompt_tokens, bucket.completion_tokens,
                bucket.cached_prompt_tokens,
            )
            total += input_cost + output_cost
    else:  # no per-deployment detail — price the aggregate at chat rates
        total += _prompt_cost(
            report.prompt_tokens,
            report.cached_prompt_tokens,
            rates.chat_input_per_1m,
            rates.chat_cached_input_per_1m,
        )
        total += report.completion_tokens * rates.chat_output_per_1m / _PER_1M
    total += report.embedding_tokens * rates.embedding_per_1m / _PER_1M
    return round(total, 6)
