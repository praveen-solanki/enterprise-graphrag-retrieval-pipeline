"""Unit tests for the pipeline stages.

Each stage of the request lifecycle is independently constructable and
independently testable — these tests exercise every stage in isolation, with
fakes at the stage boundary, no network and no app wiring:

    planning   — mode→strategy selection, budget selection, router-timeout
                 handling, conversational pass-through
    gates      — when a pre-retrieval pause is (and is not) raised
    execution  — lane budget, partial-on-timeout semantics, seed fail-open
    quality    — censor → contradictions → trust ranking sequencing
    synthesis  — answer/actions concurrency, fail-open actions, budget
    respond    — the payload shape both response paths share

    uv run pytest tests/test_pipeline_stages.py -v
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ai_graph_retrieval.clarification.models import MidResearchClarification
from ai_graph_retrieval.core.censor import CensorRegistry
from ai_graph_retrieval.core.exceptions import RetrievalTimeout
from ai_graph_retrieval.core.models import (
    ChunkQuote,
    EvidencePool,
    SourceDocument,
    Usage,
)
from ai_graph_retrieval.pipeline.execution import EvidenceCollector, has_evidence
from ai_graph_retrieval.pipeline.gates import ClarificationGates
from ai_graph_retrieval.pipeline.planning import ExecutionDecision, ExecutionPlanner
from ai_graph_retrieval.pipeline.quality import QualityGate
from ai_graph_retrieval.pipeline.respond import build_payload
from ai_graph_retrieval.pipeline.state import RequestState
from ai_graph_retrieval.pipeline.synthesis import SynthesisStage
from ai_graph_retrieval.retrieval.base import (
    ConversationalTurn,
    Intent,
    Mode,
    RetrievalPlan,
    RetrievalStrategy,
)
from ai_graph_retrieval.retrieval.profile import ExecutionProfile
from tests.support import make_profile, make_settings

# ── shared fakes ──────────────────────────────────────────────────────────────


class _Ctx:
    request_id = "req-1"
    tenant_id = "t-1"
    principals = ("user:u1",)

    def log_fields(self) -> dict:
        return {"request_id": self.request_id}


class _Progress:
    @asynccontextmanager
    async def step(self, *_a, **_k):
        yield SimpleNamespace()

    async def graph_delta(self, **_k):
        pass


def _sc(
    evidence: EvidencePool | None = None, profile: ExecutionProfile | None = None
) -> SimpleNamespace:
    """A minimal StrategyContext-shaped object for stage-level tests."""
    return SimpleNamespace(
        ctx=_Ctx(), question="who owns the K8s dashboards?", history=[],
        evidence=evidence if evidence is not None else EvidencePool(),
        progress=_Progress(), usage=Usage(), embedder=None,
        settings=make_settings(), llm=object(), degraded=False,
        profile=profile if profile is not None else make_profile(Mode.SEARCH),
    )


class _Lane(RetrievalStrategy):
    """A stub lane: optionally slow, optionally dropping evidence first."""

    name = "stub_lane"

    def __init__(self, delay_s: float = 0.0, leaves_evidence: bool = False) -> None:
        self._delay_s = delay_s
        self._leaves_evidence = leaves_evidence
        self.ran = False

    async def run(self, sc, plan) -> None:
        self.ran = True
        if self._leaves_evidence:
            sc.evidence.add_sources([SourceDocument(doc_id="d1", title="Doc 1")])
        await asyncio.sleep(self._delay_s)


def _decision(strategy: RetrievalStrategy, budget_s: float = 5.0) -> ExecutionDecision:
    return ExecutionDecision(
        plan=RetrievalPlan(intent=Intent.ENTITY), strategy=strategy,
        profile=replace(make_profile(Mode.SEARCH), budget_s=budget_s),
    )


# ── planning ──────────────────────────────────────────────────────────────────


def test_planner_passes_a_conversational_turn_straight_through():
    # The entry classifier's short-circuit verdict is not an execution decision —
    # the planner hands it up unchanged for the orchestrator to answer with.
    turn = ConversationalTurn(category="greeting", reply="Hi! Ask me anything.")

    class _Router:
        async def plan(self, question, history, mode, usage, llm=None, settings=None):
            return turn

    planner = ExecutionPlanner(
        _Router(), make_settings(),
        search_strategy=_Lane(), research_strategy=_Lane(),
    )
    assert asyncio.run(planner.resolve("hi", [], Mode.SEARCH, Usage())) is turn


def _planner(plan: RetrievalPlan, **settings_overrides) -> ExecutionPlanner:
    class _Router:
        async def plan(self, question, history, mode, usage, llm=None, settings=None):
            return plan

    return ExecutionPlanner(
        _Router(), make_settings(**settings_overrides),
        search_strategy=_Lane(), research_strategy=_Lane(),
    )


def test_planner_budget_follows_the_mode():
    # The caller's mode alone sets the depth profile AND the strategy — the
    # classifier's verdict is telemetry (agent-led design).
    s = make_settings()
    search = asyncio.run(
        _planner(RetrievalPlan(intent=Intent.ENTITY)).resolve("q", [], Mode.SEARCH, Usage())
    )
    deep = asyncio.run(
        _planner(RetrievalPlan(intent=Intent.RESEARCH)).resolve(
            "q", [], Mode.SUPER_RESEARCH, Usage())
    )
    assert search.profile.budget_s == s.search_budget_s
    assert deep.profile.budget_s == s.super_research_budget_s
    assert search.strategy is not deep.strategy           # per-mode strategies


def test_planner_ignores_the_classifier_intent_for_strategy_choice():
    # No intent-keyed lanes remain: a temporal verdict on a Search request runs
    # the same agent lane as any other Search request — time windows are the
    # agent's job through its tools, not a separate code path.
    search_lane = _Lane()

    class _Router:
        async def plan(self, question, history, mode, usage, llm=None, settings=None):
            return RetrievalPlan(intent=Intent.TEMPORAL, months=3)

    planner = ExecutionPlanner(
        _Router(), make_settings(),
        search_strategy=search_lane, research_strategy=_Lane(),
    )
    decision = asyncio.run(planner.resolve("q", [], Mode.SEARCH, Usage()))
    assert decision.strategy is search_lane
    assert decision.plan.intent is Intent.TEMPORAL        # recorded, not routed


def test_planner_turns_a_router_stall_into_a_retrieval_timeout():
    class _StallingRouter:
        async def plan(self, question, history, mode, usage, llm=None, settings=None):
            await asyncio.sleep(0.2)

    planner = ExecutionPlanner(
        _StallingRouter(), make_settings(router_timeout_s=0.01),
        search_strategy=_Lane(), research_strategy=_Lane(),
    )
    with pytest.raises(RetrievalTimeout):
        asyncio.run(planner.resolve("q", [], Mode.SEARCH, Usage()))


# ── gates ─────────────────────────────────────────────────────────────────────
# One conditional, classifier-driven gate: no
# LLM is spent here — the entry classification already wrote any questions.
# The full contract is pinned in tests/test_ambiguous_query_routing.py; these
# two keep the stage's wiring honest at the pipeline level.


def test_the_gate_pauses_search_on_classifier_questions_only():
    gates = ClarificationGates(make_settings())
    ambiguous = RetrievalPlan(
        intent=Intent.ENTITY,
        clarification_questions=[{"question": "Which team?", "options": ["A", "B"]}],
    )
    with pytest.raises(MidResearchClarification):
        gates.enforce(_Ctx(), plan=ambiguous, mode=Mode.SEARCH, plan_review=True)
    # No questions => proceed; resume turn => proceed even with questions.
    gates.enforce(_Ctx(), plan=RetrievalPlan(intent=Intent.ENTITY),
                  mode=Mode.SEARCH, plan_review=True)
    gates.enforce(_Ctx(), plan=ambiguous, mode=Mode.SEARCH, plan_review=False)


# ── execution ─────────────────────────────────────────────────────────────────


def test_collector_runs_the_lane_and_reports_a_full_pass():
    lane = _Lane()
    collector = EvidenceCollector(make_settings())
    partial = asyncio.run(collector.collect(_sc(), _decision(lane)))
    assert lane.ran
    assert partial is False


def test_collector_times_out_to_partial_when_evidence_survives():
    lane = _Lane(delay_s=0.2, leaves_evidence=True)
    collector = EvidenceCollector(make_settings(partial_on_timeout=True))
    partial = asyncio.run(collector.collect(_sc(), _decision(lane, budget_s=0.02)))
    assert partial is True                     # partial evidence still reaches synthesis


def test_collector_times_out_hard_when_nothing_was_gathered():
    lane = _Lane(delay_s=0.2)
    collector = EvidenceCollector(make_settings(partial_on_timeout=True))
    with pytest.raises(RetrievalTimeout):
        asyncio.run(collector.collect(_sc(), _decision(lane, budget_s=0.02)))


def test_has_evidence_reflects_the_pool():
    pool = EvidencePool()
    assert not has_evidence(pool)
    pool.add_sources([SourceDocument(doc_id="d1", title="Doc 1")])
    assert has_evidence(pool)


# ── quality ───────────────────────────────────────────────────────────────────


def test_quality_gate_returns_a_ranked_synthesis_ready_snapshot():
    pool = EvidencePool()
    pool.add_sources([
        SourceDocument(doc_id="d1", title="Unquoted", relevance=0.2),
        SourceDocument(doc_id="d2", title="Quoted + relevant", relevance=0.9),
    ])
    pool.add_quotes([ChunkQuote(chunk_id="c1", doc_id="d2",
                                doc_title="Quoted + relevant", text="evidence")])
    snapshot = QualityGate(CensorRegistry(), make_settings()).curate(pool)
    assert [s.doc_id for s in snapshot.sources] == ["d2", "d1"]   # trust-ranked
    assert all(s.trust > 0 for s in snapshot.sources)
    assert snapshot.contradictions == []


def test_a_rung_can_decline_the_re_rank_and_keep_its_retrieval_order():
    """Evidence that arrives already ranked by its engine keeps that order:
    re-ordering it here would override a ranking with a score of our own. The
    rest of the pass still runs — notably the citation hover preview, taken from the
    retrieved quote — so declining the ORDER never costs the annotation."""
    pool = EvidencePool()
    pool.add_sources([
        SourceDocument(doc_id="d1", title="First result", relevance=0.2),
        SourceDocument(doc_id="d2", title="Second result", relevance=0.9),
    ])
    pool.add_quotes([ChunkQuote(chunk_id="c1", doc_id="d2", doc_title="Second result",
                                text="A verbatim chunk worth previewing.")])
    gate = QualityGate(CensorRegistry(), make_settings())
    snapshot = gate.curate(pool, trust_ranking=False)
    assert [s.doc_id for s in snapshot.sources] == ["d1", "d2"]   # retrieval order
    # ranked would have put d2 first on relevance alone
    assert [s.doc_id for s in gate.curate(pool).sources] == ["d2", "d1"]
    by_id = {s.doc_id: s for s in snapshot.sources}
    assert by_id["d2"].snippet == "A verbatim chunk worth previewing."


def test_a_quoted_document_becomes_a_citable_source():
    """A run answered from a
    graph chunk but shipped citations: [] — the quote carried its document's
    identity, yet nothing pooled it as a source, so citation resolution and the
    suggested-actions generator both starved. If a document's words are in the
    evidence, the document is citable."""
    pool = EvidencePool()
    pool.add_quotes([ChunkQuote(chunk_id="c1", doc_id="d9",
                                doc_title="Acme Goes Wild",
                                text="Starting from 31st August…", graph="wikipedia")])
    snapshot = QualityGate(CensorRegistry(), make_settings()).curate(pool)
    assert [s.doc_id for s in snapshot.sources] == ["d9"]
    stub = snapshot.sources[0]
    assert stub.title == "Acme Goes Wild" and stub.graph == "wikipedia"
    assert stub.trust > 0                       # ranked like any source
    assert stub.snippet == "Starting from 31st August…"   # hover preview


def test_quote_stubs_never_duplicate_an_existing_source():
    """Either-identifier dedup: a document already pooled by id OR by title
    (chunk-lane sources can be title-only) gains no second entry, so the
    pass is a no-op on every run whose quotes' documents are already sources."""
    pool = EvidencePool()
    pool.add_sources([
        SourceDocument(doc_id="d1", title="By Id", relevance=0.5),
        SourceDocument(title="By Title Only", relevance=0.4),
    ])
    pool.add_quotes([
        ChunkQuote(chunk_id="c1", doc_id="d1", doc_title="By Id", text="t1"),
        ChunkQuote(chunk_id="c2", doc_title="By Title Only", text="t2"),
        ChunkQuote(chunk_id="c3", doc_id="d1", doc_title="By Id", text="t3"),
    ])
    snapshot = QualityGate(CensorRegistry(), make_settings()).curate(pool)
    assert len(snapshot.sources) == 2


def test_an_untitled_quote_synthesizes_no_source():
    """No title = nothing to render as a citation chip."""
    pool = EvidencePool()
    pool.add_quotes([ChunkQuote(chunk_id="c1", doc_id="d1", text="orphan text")])
    snapshot = QualityGate(CensorRegistry(), make_settings()).curate(pool)
    assert snapshot.sources == []


def test_quote_stubs_append_after_a_kept_retrieval_order():
    """With trust_ranking=False, the retrieval order stays untouched and
    any quote-backed stub lands at the END — never re-ranking what the user
    already saw ranked."""
    pool = EvidencePool()
    pool.add_sources([
        SourceDocument(doc_id="d1", title="First result", relevance=0.2),
        SourceDocument(doc_id="d2", title="Second result", relevance=0.9),
    ])
    pool.add_quotes([ChunkQuote(chunk_id="c1", doc_id="d3",
                                doc_title="Graph-only doc", text="quoted")])
    snapshot = QualityGate(CensorRegistry(), make_settings()).curate(
        pool, trust_ranking=False
    )
    assert [s.doc_id for s in snapshot.sources] == ["d1", "d2", "d3"]


# ── synthesis ─────────────────────────────────────────────────────────────────


def _composed(text: str):
    class _Composer:
        async def compose(self, sc, snapshot, **_kw):
            return SimpleNamespace(text=text, citations=[])

    return _Composer()


def _answered_snapshot():
    """A snapshot with something in it — with no judge wired, non-empty evidence is
    what makes ``evidence_found`` True, which is the precondition for suggestions."""
    pool = EvidencePool()
    pool.add_sources([SourceDocument(doc_id="d1", title="Leave Policy", url="https://x/l")])
    return pool.snapshot()


def test_synthesis_stage_returns_answer_and_actions_together():
    class _Actions:
        async def generate(self, question, answer, citations, usage, llm=None):
            # The proposer receives the composed answer, not the evidence pool
            #, and the REQUEST's llm so provider overrides reach it.
            assert answer == "the answer"
            assert llm is not None, "the stage must hand over the request's client"
            return [SimpleNamespace(label="open dashboard")]

    stage = SynthesisStage(make_settings(), _composed("the answer"), _Actions())
    composed, actions, found = asyncio.run(
        stage.run(_sc(), _answered_snapshot(), research=False, strategy="stub", mode=Mode.SEARCH)
    )
    assert composed.text == "the answer"
    assert found is True
    assert len(actions) == 1


def test_no_evidence_means_the_action_generator_is_never_called():
    # Nothing retrieved ⇒ evidence_found is False by construction ⇒ suggestions would
    # be discarded. Spending a fast-tier call to produce something we will throw away
    # is pure waste, so the generator must not be invoked at all.
    calls = []

    class _Actions:
        async def generate(self, question, answer, citations, usage):
            calls.append(question)
            return [SimpleNamespace(label="never asked for")]

    stage = SynthesisStage(make_settings(), _composed("answer"), _Actions())
    _, actions, found = asyncio.run(
        stage.run(_sc(), EvidencePool().snapshot(), research=False,
                  strategy="stub", mode=Mode.SEARCH)
    )
    assert found is False
    assert actions == []
    assert calls == []


def test_a_hung_action_call_loses_the_suggestions_not_the_answer():
    # The suggestions have their own deadline. Sharing the stage budget meant a hung
    # fast-tier call could burn the whole synthesis budget and discard a finished
    # answer — the opposite of "additive".
    class _HangingActions:
        async def generate(self, question, answer, citations, usage):
            await asyncio.sleep(30)
            return [SimpleNamespace(label="too late")]

    stage = SynthesisStage(
        make_settings(suggested_actions_timeout_s=0.05),
        _composed("the answer"), _HangingActions(),
    )
    composed, actions, found = asyncio.run(
        stage.run(_sc(), _answered_snapshot(), research=False, strategy="stub", mode=Mode.SEARCH)
    )
    assert composed.text == "the answer"   # the answer still ships
    assert actions == []
    assert found is True


def test_actions_are_dropped_when_retrieval_found_nothing():
    # the reported defect was confident follow-ups for a subject the answer
    # never established ("who is managed by <person who does not exist>"). With an
    # empty snapshot there is nothing to follow up ON, so the proposer's output is
    # discarded whatever it returned.
    class _Actions:
        async def generate(self, question, answer, citations, usage):
            return [SimpleNamespace(label="View team structure")]

    stage = SynthesisStage(make_settings(), _composed("no information found"), _Actions())
    _, actions, found = asyncio.run(
        stage.run(_sc(), EvidencePool().snapshot(), research=False,
                  strategy="stub", mode=Mode.SEARCH)
    )
    assert found is False
    assert actions == []


def test_no_mode_spends_an_answer_validation_call():
    # the signal is evidence presence in every
    # mode — no LLM call whose only job is to validate the finished answer.
    for mode in Mode:
        stage = SynthesisStage(make_settings(), _composed("the answer"), None)
        _, _, found = asyncio.run(
            stage.run(_sc(), _answered_snapshot(), research=False,
                      strategy="stub", mode=mode)
        )
        assert found is True, mode


def test_synthesis_stage_actions_fail_open():
    class _BrokenActions:
        async def generate(self, question, answer, citations, usage):
            raise RuntimeError("fast tier down")

    stage = SynthesisStage(make_settings(), _composed("still answers"), _BrokenActions())
    composed, actions, _found = asyncio.run(
        stage.run(_sc(), EvidencePool().snapshot(), research=False,
                  strategy="stub", mode=Mode.SEARCH)
    )
    assert composed.text == "still answers"    # a suggestion problem never degrades the answer
    assert actions == []


def test_synthesis_stage_without_a_generator_yields_no_actions():
    stage = SynthesisStage(make_settings(), _composed("answer"), None)
    _, actions, _found = asyncio.run(
        stage.run(_sc(), EvidencePool().snapshot(), research=False,
                  strategy="stub", mode=Mode.SEARCH)
    )
    assert actions == []


def test_synthesis_stage_enforces_its_own_budget():
    class _SlowComposer:
        async def compose(self, sc, snapshot, **_kw):
            await asyncio.sleep(0.2)

    stage = SynthesisStage(make_settings(synthesis_budget_s=0.02), _SlowComposer(), None)
    with pytest.raises(RetrievalTimeout):
        asyncio.run(stage.run(_sc(), EvidencePool().snapshot(),
                              research=False, strategy="stub", mode=Mode.SEARCH))


# ── respond ───────────────────────────────────────────────────────────────────


def test_payload_citations_fall_back_to_the_ranked_pool_when_synthesis_cited_none():
    pool = EvidencePool()
    pool.add_sources([SourceDocument(doc_id="d1", title="Doc 1")])
    evidence = pool.snapshot()
    state = RequestState(mode="search", intent="entity", strategy="search")
    report = Usage().snapshot()

    fallback = build_payload(_Ctx(), state, answer="a", evidence=evidence,
                             usage_report=report, duration_ms=1, citations=None)
    assert [c["doc_id"] for c in fallback["citations"]] == ["d1"]

    cited = build_payload(_Ctx(), state, answer="a", evidence=evidence,
                          usage_report=report, duration_ms=1, citations=[])
    assert cited["citations"] == []            # explicit empty list is NOT the fallback


def test_payload_shape_matches_the_response_contract():
    # ``answer_found`` keeps its name deliberately: the internal signal
    # and the Langfuse score were renamed to `evidence_found`, which is what they
    # measure, but clients branch on this field, so renaming it would break them
    # for no gain. Its DESCRIPTION was corrected instead — see respond.py.
    state = RequestState(mode="search", intent="entity", strategy="search")
    payload = build_payload(_Ctx(), state, answer="a",
                            evidence=EvidencePool().snapshot(),
                            usage_report=Usage().snapshot(), duration_ms=1)
    assert set(payload) == {
        "request_id", "intent", "strategy", "mode", "answer_found",
        "escalated", "partial", "answer", "answer_graph", "citations", "evidence", "trace",
        "usage", "duration_ms", "suggested_actions",
    }
    assert payload["suggested_actions"] == []


def test_a_degraded_context_surfaces_as_partial():
    # agent/runtime swallows a mid-loop provider failure when evidence was
    # already gathered (the long-run context-window shape) and marks the
    # context degraded. The collector must fold that into its `partial` return
    # — the same user-facing meaning as a timeout with surviving evidence.
    lane = _Lane(leaves_evidence=True)
    collector = EvidenceCollector(make_settings())
    sc = _sc()
    sc.degraded = True   # what the agent's swallow path sets
    assert asyncio.run(collector.collect(sc, _decision(lane))) is True
    clean = _sc()
    assert asyncio.run(collector.collect(clean, _decision(_Lane()))) is False
