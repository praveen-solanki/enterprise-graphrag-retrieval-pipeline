"""One depth table per mode — Search / Super Research — and it is real.

Each mode resolves to exactly one ``ExecutionProfile`` (retrieval/profile.py),
selected by the CALLER'S EXPLICIT MODE alone: there is no automatic mode
selection, no default mode, and no intent-based profile override. These tests
pin the properties that make the two modes genuinely different pipelines:

  * SEARCH runs a SHORT agent loop over the graphs — never the full loop;
  * SUPER_RESEARCH is the full agent loop, and never runs the loop twice;
  * every depth cap comes from the profile, so the lanes hold no private
    constants that would ignore the mode;
  * a nonsensical config override is repaired, never fatal.

    uv run pytest tests/test_execution_profiles.py -v
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ai_graph_retrieval.core.models import (
    ChunkQuote,
    DocumentText,
    EntityHit,
    EvidencePool,
    RelationFact,
    SourceDocument,
    Usage,
)
from ai_graph_retrieval.graph.repository import GraphReadRepository
from ai_graph_retrieval.pipeline.execution import EvidenceCollector
from ai_graph_retrieval.pipeline.planning import ExecutionDecision
from ai_graph_retrieval.retrieval.base import Intent, Mode, RetrievalPlan, RetrievalStrategy
from ai_graph_retrieval.retrieval.deep_research import (
    DeepResearchStrategy,
    SearchAgentStrategy,
    build_scorer,
)
from ai_graph_retrieval.retrieval.profile import (
    profile_for,
    search_profile,
    super_research_profile,
)
from tests.support import make_profile, make_settings

# ── shared fakes ──────────────────────────────────────────────────────────────


class _Ctx:
    request_id = "req-1"
    tenant_id = "t-1"
    principals = ("user:u1",)

    def log_fields(self) -> dict:
        return {"request_id": self.request_id}


class _Progress:
    def __init__(self) -> None:
        # (kind, summary) — the label is derived from the kind by the reporter
        # (core/narration), so a stub that recorded labels would be asserting
        # against something the call site no longer chooses.
        self.steps: list[tuple] = []

    @asynccontextmanager
    async def step(self, kind, summary=None, **_k):
        self.steps.append((kind, summary))
        yield SimpleNamespace()

    async def graph_delta(self, **_k):
        pass


class _Repo:
    """Counts what each lane actually asked the graph for."""

    # The repository contract the graph-chunk relevance pass reads.
    # None ⇒ no vector arm, so the pass returns without scoring.
    sources = ()
    vector_store = None

    def __init__(self, n_entities: int = 9) -> None:
        self._n = n_entities
        self.expanded: list[str] = []
        self.claims_for: list[str] = []
        self.texts_for: list[str] = []

    async def resolve_entities(self, ctx, phrase, embedding):
        return [EntityHit(name=f"E{i}") for i in range(self._n)]

    async def expand_entity(self, ctx, name, **_k):
        self.expanded.append(name)
        return [RelationFact(source=name, relationship="OWNS", target="thing")]

    async def fetch_claims(self, ctx, name, since):
        self.claims_for.append(name)
        return []

    async def source_documents(self, ctx, names):
        return [SourceDocument(doc_id="d1", title="Doc 1")]

    async def supporting_chunks(self, ctx, names):
        return []

    async def fetch_document_text(self, ctx, doc_id, max_chars=0):
        self.texts_for.append(doc_id)
        return DocumentText(chunks=[
            ChunkQuote(chunk_id=f"{doc_id}-c1", doc_id=doc_id,
                       doc_title="Doc 1", text="body text")])


class _Embedder:
    async def get(self, *_a, **_k):
        return [0.1, 0.2, 0.3]


def _sc(profile, *, evidence=None, repo=None, progress=None):
    return SimpleNamespace(
        ctx=_Ctx(), question="who owns the dashboards?", history=[],
        settings=make_settings(), repo=repo if repo is not None else _Repo(),
        llm=None, embedder=_Embedder(),
        evidence=evidence if evidence is not None else EvidencePool(),
        progress=progress if progress is not None else _Progress(),
        usage=Usage(), profile=profile, chunk_retriever=None, seed_phrases=[],
        degraded=False, historical=False,
    )


# ── the table's shape ─────────────────────────────────────────────────────────


def _profiles():
    s = make_settings()
    return search_profile(s), super_research_profile(s)


def test_search_is_the_light_agent_loop_at_its_own_budgets():
    s = make_settings()
    search, super_ = _profiles()
    assert search.agent_rounds == s.search_agent_rounds
    assert 0 < search.agent_rounds < super_.agent_rounds   # 4/2 vs the full 5/2
    assert search.deep_tool_rows is False   # Super Research keeps the wide row budget
    assert search.scoring_budget > 0
    # Search runs the same graph-text machinery as Super Research at its own,
    # smaller, settings-backed cap — 0 in config makes the graph enrichment-only.
    assert search.graph_text_tools is True
    assert search.graph_chunk_cap == s.search_graph_chunk_cap
    assert 0 < search.graph_chunk_cap < super_.graph_chunk_cap
    assert search.mid_research_clarify is False
    assert search.digest_quote_chars == s.search_digest_quote_chars


def test_super_research_extends_search_never_reimplements_it():
    # Same flow, extensions — more rounds, a larger graph-chunk allowance, the
    # mid-research pause and a larger excerpt budget.
    s = make_settings()
    search, super_ = _profiles()
    assert super_.agent_rounds == s.agent_max_rounds        # extension 1: more rounds
    assert super_.agent_patience == s.agent_stale_patience
    assert super_.deep_tool_rows is True                    # …with the wide row budget
    assert super_.graph_text_tools is True                  # extension 2: graph chunks…
    assert super_.graph_chunk_cap == s.super_research_graph_chunk_cap  # …bounded (TOTAL)
    assert super_.mid_research_clarify is True              # extension 3: mid-run pause
    # Extension 4: its OWN, larger excerpt budget.
    assert super_.digest_quote_chars == s.super_research_digest_quote_chars
    assert super_.digest_quote_chars > search.digest_quote_chars


def test_the_two_profiles_are_genuinely_different():
    search, super_ = _profiles()
    assert search != super_


def test_a_deeper_mode_never_gets_less_wall_clock():
    search, super_ = _profiles()
    assert search.budget_s <= super_.budget_s


# ── mode → profile resolution ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("mode", "expected"),
    [(Mode.SEARCH, "search"), (Mode.SUPER_RESEARCH, "super_research")],
)
def test_each_mode_resolves_to_its_own_profile(mode, expected):
    assert profile_for(make_settings(), mode, Intent.ENTITY).name == expected


@pytest.mark.parametrize("mode", list(Mode))
@pytest.mark.parametrize("intent", list(Intent))
def test_the_classifier_verdict_never_changes_the_depth_the_mode_bought(mode, intent):
    """The invariant of manual mode selection: the intent picks a lane WITHIN a
    mode, the mode alone picks the depth. Every (mode, intent) pair resolves to
    the mode's own profile — there is no intent-based override left."""
    assert profile_for(make_settings(), mode, intent).name == mode.value


# ── configuration cannot break the table ──────────────────────────────────────


@pytest.mark.parametrize(("bad", "field", "at_least"), [
    ({"search_budget_s": 0}, "search_budget_s", 0.001),  # timeout(0) expires instantly
    ({"search_budget_s": -5}, "search_budget_s", 0.001),
    ({"source_docs_per_anchor": -1}, "source_docs_per_anchor", 1),  # negative slice
    ({"super_research_graph_chunk_cap": -5}, "super_research_graph_chunk_cap", 1),
    ({"quotes_per_anchor": 0}, "quotes_per_anchor", 1),
    ({"agent_max_rounds": 0}, "agent_max_rounds", 1),
])
def test_a_nonsensical_override_is_repaired_and_the_service_still_boots(bad, field, at_least):
    """A config typo must never take the pipeline down.

    These values reach asyncio.timeout() and list slices directly, where 0/negative is
    not "unlimited" but actively wrong. Each has one safe repair, so it is applied and
    logged — the service starts and serves.
    """
    s = make_settings(**bad)                  # no raise
    assert getattr(s, field) >= at_least


def test_an_inverted_budget_ladder_is_repaired_upward_never_downward():
    """A deeper mode given LESS time than a shallower one is repaired by raising
    the deeper mode. Repairing downward would silently reduce a budget someone
    configured deliberately; upward only ever preserves the property."""
    s = make_settings(super_research_budget_s=100)     # below search's 1200
    assert s.search_budget_s <= s.super_research_budget_s


def test_relationships_that_would_make_a_setting_a_no_op_are_repaired():
    # An excerpt budget that cannot hold even ONE anchor's excerpt result
    # would undo the fix that made anchors contribute text. (One anchor, not
    # many anchors' worth: Search sizes its own digest budget in its profile.)
    s = make_settings(digest_quote_char_budget=10_000)
    assert s.digest_quote_char_budget >= s.quotes_per_anchor * s.quote_chars


def test_a_valid_config_is_left_exactly_as_configured():
    """Repairs must only fire on genuinely broken values — a deliberate, coherent
    override has to survive untouched, or tuning becomes impossible."""
    s = make_settings(super_research_graph_chunk_cap=90,
                      digest_quote_char_budget=80_000, search_budget_s=900)
    assert s.super_research_graph_chunk_cap == 90
    assert s.digest_quote_char_budget == 80_000
    assert s.search_budget_s == 900


def test_the_shipped_defaults_pass_their_own_validation():
    # The guard is only useful if the values we ship satisfy it.
    s = make_settings()
    assert s.search_budget_s <= s.super_research_budget_s
    assert s.digest_quote_char_budget >= s.quotes_per_anchor * s.quote_chars


# ── grounding wider must not LOSE documents ───────────────────────────────────


def test_source_documents_budgets_per_anchor_so_adding_a_name_only_adds():
    """The eviction this fixes, asserted on the query itself.

    The old shape matched every anchor name at once, ranked by TOTAL mention count and
    cut at a fixed LIMIT — so adding an anchor could push out the documents of the
    entity that actually matched the question. Measured on dev: an anchor with 11
    documents kept 9 when it was the only name and 5 once four more names were added.
    That made "grounding wider" non-monotonic.

    Asserted structurally (the query text) rather than against a live graph, because
    this is a property of the Cypher shape: a per-anchor slice cannot be affected by
    what other anchors return.
    """
    from ai_graph_retrieval.graph import cypher

    q = cypher.SOURCE_DOCUMENTS
    assert "UNWIND $names AS anchor" in q          # each anchor evaluated on its own
    assert "[0..$per_anchor]" in q                 # …with its own budget
    assert "e.name IN $names" not in q             # not the old all-anchors-at-once match
    # The ACL predicate must survive untouched, or the runtime guard
    # (cypher.acl_guarded) silently stops rewriting it.
    assert "any(p IN d.allow_access_control WHERE p IN $principals)" in q
    assert "tenant_id = $tenant" not in q


def test_supporting_chunks_budgets_per_anchor_so_no_anchor_is_starved():
    """The starvation this fixes.

    One global LIMIT with no ORDER BY meant whichever anchors the planner traversed
    first consumed the whole budget. Measured on dev with 5 anchors and LIMIT 15: the
    first two took all 15 rows and THREE anchors returned zero excerpts. Those excerpts
    are the verbatim text the answer is grounded in and cited against, so an entity the
    question is squarely about could contribute nothing quotable.
    """
    from ai_graph_retrieval.graph import cypher

    q = cypher.SUPPORTING_CHUNKS
    assert "UNWIND $names AS anchor" in q
    assert "[0..$per_anchor]" in q
    assert "e.name IN $names" not in q          # not the old all-anchors-at-once match
    # via_names must be re-derived over ALL requested names after the per-anchor slice,
    # or a chunk selected for one anchor would under-report the others it mentions.
    assert "e2.name IN $names" in q
    assert "collect(DISTINCT e2.name) AS via_names" in q


@pytest.mark.parametrize(("method", "per_anchor_field"), [
    ("source_documents", "source_docs_per_anchor"),
    ("supporting_chunks", "quotes_per_anchor"),
])
def test_per_anchor_reads_size_their_ceiling_so_it_cannot_bind(method, per_anchor_field):
    """`limit` is derived from the per-anchor budget, so the overall ceiling can never
    re-create the eviction/starvation it used to cause."""
    captured = {}

    class _Repo(GraphReadRepository):
        def __init__(self):
            self._s = make_settings()

        async def _rows(self, query, ctx, **params):
            captured.update(params)
            return []

    asyncio.run(getattr(_Repo(), method)(_Ctx(), ["A", "B", "C"]))
    expected = getattr(make_settings(), per_anchor_field)
    assert captured["per_anchor"] == expected
    assert captured["limit"] == expected * 3


def test_the_excerpt_budgets_hold_what_the_per_anchor_reads_gather():
    """The per-anchor excerpt fix is only real if a digest budget can fit the
    result. The standard budget must hold at least one anchor's excerpts, and
    the Search modes' own budget (search_digest_quote_chars) is never narrower
    than the standard one — their evidence is strictly bulkier."""
    s = make_settings()
    assert s.digest_quote_char_budget >= s.quotes_per_anchor * s.quote_chars
    assert s.search_digest_quote_chars >= s.digest_quote_char_budget
    assert s.digest_max_quotes >= s.quotes_per_anchor


# ── entity descriptions reach synthesis whole ─────────────────────────────────


def test_entity_descriptions_are_truncated_on_a_word_boundary_at_the_configured_width():
    """Bounded, configurable, and never cut mid-word.

    The old cap was a hardcoded 280 applied with a plain slice, which clipped the
    derived text that explains what an entity IS — mid-word. The width is now a setting
    and the cut lands on whitespace.
    """
    from ai_graph_retrieval.synthesis.composer import AnswerComposer

    s = make_settings()
    long_desc = "Owns the platform. " * 60          # ~1140 chars, over every cap
    pool = EvidencePool()
    pool.add_entities([EntityHit(name="Platform Reliability", type="TEAM",
                                description=long_desc, score=1.0)])
    digest = AnswerComposer(s)._digest(pool.snapshot(), strict=False)

    line = next(ln for ln in digest.splitlines() if ln.startswith("- Platform Reliability"))
    shown = line.split(" — ", 1)[1]
    assert len(shown) <= s.digest_entity_desc_chars
    assert not shown.endswith(("Own", "Owns the platfor"))    # not a mid-word cut
    assert shown.split()[-1] in {"Owns", "the", "platform."}  # ends on a whole token
    # …and the width is genuinely wider than the constant it replaced.
    assert s.digest_entity_desc_chars > 280


def test_the_description_widths_are_configurable():
    from ai_graph_retrieval.synthesis.composer import AnswerComposer

    s = make_settings(digest_entity_desc_chars=30)
    pool = EvidencePool()
    pool.add_entities([EntityHit(name="X", type="TEAM",
                                description="a " * 200, score=1.0)])
    digest = AnswerComposer(s)._digest(pool.snapshot(), strict=False)
    shown = next(ln for ln in digest.splitlines()
                 if ln.startswith("- X")).split(" — ", 1)[1]
    assert len(shown) <= 30


# ── the agent lane reads its knobs from the profile ──────────────────────────


def test_the_lane_passes_the_profile_flags_to_the_agent(monkeypatch):
    """clarify + graph_text are per-mode capabilities and must arrive at the
    agent from the PROFILE — a hardcoded value here silently gives Search the
    mid-run pause or takes Super Research's graph chunks away. graph_text is
    the EFFECTIVE capability (2026-08-18): flag AND a non-zero cap — a zero
    cap (Search's setting at 0, or a per-request override of 0) must remove
    the graph-text tools rather than expose them inert."""
    built: list[dict] = []

    class _FakeAgent:
        def __init__(self, settings, *, clarify=True, graph_text=True, research=False):
            built.append({"clarify": clarify, "graph_text": graph_text,
                          "research": research})

        async def run(self, sc, scorer):
            return None

    import ai_graph_retrieval.agent.runtime as runtime_mod
    monkeypatch.setattr(runtime_mod, "ResearchAgent", _FakeAgent)
    plan = RetrievalPlan(intent=Intent.ENTITY)
    asyncio.run(SearchAgentStrategy().run(_sc(make_profile(Mode.SEARCH)), plan))
    asyncio.run(DeepResearchStrategy().run(
        _sc(make_profile(Mode.SUPER_RESEARCH, Intent.RESEARCH)), plan))
    # Search with the graph-chunk allowance zeroed (config or per-request
    # options.graph_chunk_cap=0): the capability is absent, exactly as before
    # the allowance existed.
    sc_zero = _sc(make_profile(Mode.SEARCH))
    sc_zero.profile = replace(sc_zero.profile, graph_chunk_cap=0)
    asyncio.run(SearchAgentStrategy().run(sc_zero, plan))
    # `research` is the depth-instruction flag: the full agent lane gets
    # the block telling it what its extra rounds are for, the light lane does not.
    assert built == [
        # Search (cap 8 by default)
        {"clarify": False, "graph_text": True, "research": False},
        # Super Research
        {"clarify": True, "graph_text": True, "research": True},
        # Search with cap 0
        {"clarify": False, "graph_text": False, "research": False},
    ]


def test_the_neighbor_re_ranker_is_built_only_for_a_mode_with_a_budget():
    assert build_scorer(_sc(make_profile(Mode.SEARCH, search_scoring_budget=0))) is None
    assert build_scorer(_sc(make_profile(Mode.SEARCH))) is not None
    # …and the master flag still wins.
    sc = _sc(make_profile(Mode.SEARCH))
    sc.settings = make_settings(scoring_enabled=False)
    assert build_scorer(sc) is None


# ── the collector runs the strategy and nothing else ─────────────────────────


def test_the_collector_has_no_retrieval_of_its_own():
    """The collector runs the strategy — nothing else — so a decision touches
    nothing but its lane."""

    class _Lane(RetrievalStrategy):
        name = "stub"

        def __init__(self):
            self.ran = False

        async def run(self, sc, plan) -> None:
            self.ran = True

    lane = _Lane()
    collector = EvidenceCollector(make_settings())
    decision = ExecutionDecision(
        plan=RetrievalPlan(intent=Intent.ENTITY), strategy=lane,
        profile=make_profile(Mode.SEARCH),
    )
    partial = asyncio.run(collector.collect(_sc(make_profile(Mode.SEARCH)), decision))
    assert lane.ran and partial is False


def test_the_profile_topup_is_on_for_both_agent_modes():
    """Both agent modes get the structured-profile pass from one setting."""
    s = make_settings()
    assert search_profile(s).profile_enrich_top == s.agent_profile_enrich_top > 0
    assert super_research_profile(s).profile_enrich_top == s.agent_profile_enrich_top


def test_the_mention_topup_is_on_for_both_agent_modes():
    """Both agent modes take the mention pass from one setting."""
    s = make_settings()
    assert search_profile(s).mention_enrich_top == s.agent_mention_enrich_top > 0
    assert super_research_profile(s).mention_enrich_top == s.agent_mention_enrich_top


def test_the_employment_gate_is_on_for_both_agent_modes():
    """The gate is on in both profiles; the collector applies it only when
    ``people_graph_enabled`` is on."""
    s = make_settings()
    assert search_profile(s).employment_gate is True
    assert super_research_profile(s).employment_gate is True
