"""A model failure inside the research loop must not become a 500.

Reported symptom: long-running Research requests eventually return 500 Internal
Server Error. The agent's model calls go through the OpenAI-Agents SDK straight to
LiteLLM, bypassing ``LlmClient``, so a provider error in a round never became
``LlmUnavailable`` — it left the error taxonomy and hit the catch-all handler, which
sanitises anything unknown into ``500 internal_error``. The longer a run goes, the
more likely it is: each round re-sends a bigger conversation until the gateway
rejects it outright.

Contract asserted here, matching what the turn cap and the stale-stop already do:
evidence gathered so far is kept and synthesis proceeds; only a run that gathered
NOTHING surfaces an error, and then a mapped 502 rather than a bare 500.

    uv run pytest tests/test_agent_runtime_failures.py -v
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from ai_graph_retrieval.agent import runtime
from ai_graph_retrieval.agent.runtime import ResearchAgent
from ai_graph_retrieval.clarification.models import MidResearchClarification
from ai_graph_retrieval.core.exceptions import LlmUnavailable
from ai_graph_retrieval.core.models import EntityHit, EvidencePool, Usage
from ai_graph_retrieval.retrieval.base import Intent, Mode
from tests.support import make_profile, make_settings


def _sc(pool: EvidencePool):
    """The slice of StrategyContext the agent run touches."""
    return SimpleNamespace(
        ctx=SimpleNamespace(log_fields=lambda: {}),
        question="how did retrieval evolve?",
        repo=None, embedder=None, evidence=pool, progress=None, usage=Usage(),
        seed_phrases=[], degraded=False,
        profile=make_profile(Mode.SUPER_RESEARCH, Intent.RESEARCH),
    )


def _raise(exc: BaseException):
    async def _boom(*_a, **_kw):
        raise exc
    return _boom


def test_a_provider_error_mid_loop_keeps_the_evidence_and_does_not_raise(monkeypatch):
    # The common long-run shape: the growing conversation is rejected outright. Two
    # rounds' findings are already in the pool, so the turn degrades to synthesising
    # them — the same contract as hitting the round cap.
    pool = EvidencePool()
    pool.add_entities([EntityHit(name="Enterprise Search"), EntityHit(name="AI Search")])
    monkeypatch.setattr(
        runtime.Runner, "run",
        _raise(RuntimeError("litellm.BadRequestError: prompt is too long: 214k > 200k tokens")),
    )
    sc = _sc(pool)
    asyncio.run(ResearchAgent(make_settings()).run(sc, None))   # no raise
    assert pool.size() == 2      # findings survive for synthesis
    # ...and the degrade is VISIBLE: the context is marked so the collector
    # folds it into the response's `partial` flag — a thinner answer must never
    # present itself as a complete one.
    assert sc.degraded is True


def test_a_provider_error_with_nothing_gathered_maps_to_a_502(monkeypatch):
    # Nothing to degrade to, so fail — but inside the taxonomy (502 llm_unavailable),
    # never as an unclassified 500.
    monkeypatch.setattr(runtime.Runner, "run", _raise(RuntimeError("gateway exploded")))
    with pytest.raises(LlmUnavailable):
        asyncio.run(ResearchAgent(make_settings()).run(_sc(EvidencePool()), None))


def test_a_clarification_pause_still_propagates(monkeypatch):
    # A mid-research question is a pause, not a failure: it must not be swallowed by
    # the new catch-all, or the HITL flow silently turns into an empty answer.
    monkeypatch.setattr(
        runtime.Runner, "run", _raise(MidResearchClarification([])),
    )
    with pytest.raises(MidResearchClarification):
        asyncio.run(ResearchAgent(make_settings()).run(_sc(EvidencePool()), None))


def test_client_disconnect_is_not_swallowed(monkeypatch):
    # Cancellation means the client left; converting it into a degraded answer would
    # keep burning budget on a request nobody is waiting for.
    monkeypatch.setattr(runtime.Runner, "run", _raise(asyncio.CancelledError()))
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(ResearchAgent(make_settings()).run(_sc(EvidencePool()), None))
