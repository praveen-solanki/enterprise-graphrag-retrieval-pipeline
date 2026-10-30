"""Unit tests for the agent's stale-patience early-stop and per-round metering.

The SDK's only native bounds are the model stopping itself and ``max_turns``; the
``_AgentRunHooks`` on_llm_start hook enforces ``agent_stale_patience`` by watching
the evidence-pool size across rounds and raising ``_StalePatienceReached`` before the
model call once nothing new has been added for ``patience`` consecutive rounds. The
same hooks meter each round's tokens and give it its own LLM span.

    uv run pytest tests/test_agent_stale_patience.py -v
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from ai_graph_retrieval.agent.runtime import _AgentRunHooks, _StalePatienceReached
from ai_graph_retrieval.core.costing import CostRates
from ai_graph_retrieval.core.models import EntityHit, EvidencePool, Usage
from tests.support import make_settings


def _ctx():
    """A stand-in RunContextWrapper — the hook only reads ``.usage``."""
    return SimpleNamespace(usage=SimpleNamespace(input_tokens=123, output_tokens=45))


def _hooks(pool: EvidencePool, patience: int, usage: Usage | None = None) -> _AgentRunHooks:
    settings = make_settings()
    return _AgentRunHooks(
        pool, patience, usage=usage or Usage(),
        deployment=settings.llm_model,
        rates=CostRates.from_settings(settings),
        provider="openai",
    )


def _round_start(hook: _AgentRunHooks) -> None:
    """Fire the on_llm_start hook once (a round boundary)."""
    asyncio.run(hook.on_llm_start(_ctx(), agent=None, system_prompt="", input_items=[]))


# ── EvidencePool.size (the signal the hook watches) ──────────────────────────

def test_evidence_size_counts_and_dedupes():
    p = EvidencePool()
    assert p.size() == 0
    p.add_entities([EntityHit(name="A"), EntityHit(name="B")])
    assert p.size() == 2
    p.add_entities([EntityHit(name="A")])  # dup — no growth
    assert p.size() == 2


# ── the hook ─────────────────────────────────────────────────────────────────

def test_raises_after_patience_consecutive_stale_rounds():
    pool = EvidencePool()
    hook = _hooks(pool, patience=2)

    _round_start(hook)                       # round 1 start: size 0
    pool.add_entities([EntityHit(name="A")])  # round 1 produced something
    _round_start(hook)                        # round 2 start: size 1 (grew) → stale 0
    # rounds 2 and 3 add nothing:
    _round_start(hook)                        # round 3 start: size 1 == 1 → stale 1
    with pytest.raises(_StalePatienceReached) as ei:
        _round_start(hook)                    # round 4 start: still 1 → stale 2 == patience
    assert ei.value.stale_rounds == 2
    # usage is carried so the early-stop path can still meter cost exactly
    assert ei.value.usage.input_tokens == 123


def test_growth_resets_the_stale_counter():
    pool = EvidencePool()
    hook = _hooks(pool, patience=2)
    # A productive round every other round keeps it alive well past patience:
    # one stale round is always followed by growth, so stale never reaches 2.
    for i in range(6):
        _round_start(hook)
        if i % 2 == 0:
            pool.add_entities([EntityHit(name=f"E{i}")])


def test_patience_floor_of_one():
    # patience<=0 is clamped to 1: stop after the first no-growth round.
    pool = EvidencePool()
    hook = _hooks(pool, patience=0)
    _round_start(hook)          # round 1: size 0
    with pytest.raises(_StalePatienceReached):
        _round_start(hook)      # round 2: size still 0 → stale 1 >= 1


# ── per-round metering ────────────────────────────────────────────

def _response(prompt: int, completion: int, cached: int = 0):
    """A stand-in ModelResponse carrying one round's usage."""
    return SimpleNamespace(usage=SimpleNamespace(
        input_tokens=prompt, output_tokens=completion,
        input_tokens_details=SimpleNamespace(cached_tokens=cached),
    ))


def _round_end(hook: _AgentRunHooks, response) -> None:
    asyncio.run(hook.on_llm_end(_ctx(), agent=None, response=response))


def test_each_round_is_metered_as_its_own_call():
    # The defect: one add_llm for a whole multi-round run, so llm_calls under-reported
    # the agent's real call count and Langfuse saw no generations for the loop at all.
    usage = Usage()
    hook = _hooks(EvidencePool(), patience=3, usage=usage)
    _round_end(hook, _response(1_000, 100))
    _round_end(hook, _response(2_000, 200, cached=1_500))
    report = usage.snapshot()
    assert report.llm_calls == 2                     # two calls, not one run
    assert report.prompt_tokens == 3_000
    assert report.completion_tokens == 300
    assert report.cached_prompt_tokens == 1_500      # cache reads survive to the report
    assert hook.rounds_metered == 2


def test_rounds_are_bucketed_under_the_real_deployment():
    usage = Usage()
    hook = _hooks(EvidencePool(), patience=3, usage=usage)
    _round_end(hook, _response(500, 50))
    bucket = usage.snapshot().by_deployment[make_settings().llm_model]
    assert bucket.calls == 1
    assert bucket.prompt_tokens == 500


def test_a_round_without_usage_is_not_counted():
    # SDK shape drift must not invent a call or crash the run.
    usage = Usage()
    hook = _hooks(EvidencePool(), patience=3, usage=usage)
    _round_end(hook, SimpleNamespace(usage=None))
    assert usage.snapshot().llm_calls == 0
    assert hook.rounds_metered == 0


def test_hooks_close_is_safe_without_tracing():
    # With tracing unavailable the span is None throughout; close() must be a no-op
    # rather than an AttributeError on the error path.
    hook = _hooks(EvidencePool(), patience=3)
    hook.close()
    hook.close()
