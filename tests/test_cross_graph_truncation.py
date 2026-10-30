"""Cross-graph truncation: a positional cut must never decide which graph wins.

The repository fans out over every graph and CONCATENATES the per-graph result
lists (graph/repository.py::_fan_out) — every row of the first configured source,
then every row of the second. Each block is ordered on its own, so the
concatenation is several sorted runs glued together, never one ranking. Any
consumer that then takes a head-slice lets the FIRST graph spend the whole budget
and drops the second's stronger rows purely for arriving later.

Every path here has a genuine cross-graph comparable signal (claim recency, the
hop-normalised involvement score, publication date), so the fix is to rank before
the cut — not to interleave, which would discard that signal.

    uv run pytest tests/test_cross_graph_truncation.py -v
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from ai_graph_retrieval.agent.tools import _top_facts
from ai_graph_retrieval.core.models import (
    ClaimFact,
    EntityHit,
    EvidencePool,
    RelationFact,
)
from ai_graph_retrieval.retrieval.deep_research import enrich_claims
from ai_graph_retrieval.retrieval.neighbor_scoring import NeighborScorer, ScoringBudget
from ai_graph_retrieval.retrieval.person_status import PersonIdentities
from ai_graph_retrieval.retrieval.quality import rank_claims_by_recency
from tests.support import make_settings

# The two graphs in fan-out order: wikipedia is always built first
# (Settings.graph_sources), arXiv is appended.
WIKIPEDIA, ARXIV = "wikipedia", "arxiv"


# ── 1. enrich_claims ────────────────────────────────────────────────────────


class _Step:
    detail = ""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _TwoGraphRepo:
    """fetch_claims returning the fan-out shape: wikipedia's block, then
    arXiv's. Wikipedia alone can fill the per-entity budget."""

    def __init__(self, wikipedia: int, arxiv_from: str) -> None:
        self._wikipedia = wikipedia
        self._arxiv_from = arxiv_from

    async def person_identities(self, ctx, names):
        return PersonIdentities()

    async def fetch_claims(self, ctx, name, since_iso=None):
        rows = [
            ClaimFact(subject=name, predicate="noted", object=f"bk-{i}",
                      valid_from=f"2020-0{i + 1}-01", graph=WIKIPEDIA)
            for i in range(self._wikipedia)
        ]
        rows.append(
            ClaimFact(subject=name, predicate="noted", object="cf-newest",
                      valid_from=self._arxiv_from, graph=ARXIV)
        )
        return rows


def _sc(pool: EvidencePool, repo, *, per_entity: int = 5) -> SimpleNamespace:
    async def graph_delta(**kwargs):
        return None

    return SimpleNamespace(
        evidence=pool,
        repo=repo,
        ctx=SimpleNamespace(log_fields=lambda: {}),
        profile=SimpleNamespace(claim_enrich_top=10, profile_enrich_top=0),
        settings=SimpleNamespace(agent_claim_enrich_per_entity=per_entity,
                                 person_identity_max_names=400),
        progress=SimpleNamespace(step=lambda *a, **k: _Step(),
                                 graph_delta=graph_delta, span_parent=None),
    )


def test_a_full_wikipedia_block_cannot_evict_a_newer_arxiv_claim():
    """Wikipedia fills the per-entity budget on its own; the newest claim is
    arXiv's. Positional slicing dropped it — ranking keeps it."""
    pool = EvidencePool()
    pool.add_entities([EntityHit(name="Topic X")])
    repo = _TwoGraphRepo(wikipedia=5, arxiv_from="2026-08-02")

    asyncio.run(enrich_claims(_sc(pool, repo, per_entity=5)))

    kept = pool.snapshot().claims
    assert len(kept) == 5, "the per-entity budget is still enforced"
    assert "cf-newest" in {c.object for c in kept}
    # ...and it leads, because it is the newest claim of either graph.
    assert kept[0].object == "cf-newest"


def test_the_per_entity_budget_is_not_widened_to_make_room():
    """The fix reorders; it must not fetch or keep more than before."""
    pool = EvidencePool()
    pool.add_entities([EntityHit(name="Topic X")])
    repo = _TwoGraphRepo(wikipedia=20, arxiv_from="2026-08-02")

    asyncio.run(enrich_claims(_sc(pool, repo, per_entity=3)))

    assert len(pool.snapshot().claims) == 3


def test_rank_claims_by_recency_puts_undated_last():
    claims = [
        ClaimFact(subject="A", predicate="p", object="undated"),
        ClaimFact(subject="A", predicate="p", object="old", valid_from="2024-01-01"),
        ClaimFact(subject="A", predicate="p", object="new", valid_from="2026-01-01"),
    ]
    assert [c.object for c in rank_claims_by_recency(claims)] == [
        "new", "old", "undated",
    ]


# ── 2. NeighborScorer.rescore ───────────────────────────────────────────────


def _fact(target: str, score: float, graph: str, support: int = 1) -> RelationFact:
    return RelationFact(source="Hub", target=target, relationship="RELATED_TO",
                        score=score, support=support, graph=graph)


def _scorer(budget: int = 0) -> NeighborScorer:
    return NeighborScorer(llm=None, settings=make_settings(),
                          usage=SimpleNamespace(), budget=ScoringBudget(budget))


def test_the_cheap_slice_keeps_the_strongest_of_both_graphs():
    """Wikipedia supplies enough candidates to fill `keep` by itself, but
    arXiv holds the two strongest. With no scoring budget the cheap path
    runs — and it must still rank across graphs before cutting."""
    facts = [_fact(f"bk-{i}", score=0.10, graph=WIKIPEDIA) for i in range(6)]
    facts += [_fact("cf-top", score=0.90, graph=ARXIV),
              _fact("cf-second", score=0.80, graph=ARXIV)]

    kept, tier = asyncio.run(_scorer().rescore(facts, focus="anything", keep=3))

    assert tier == "cheap"
    assert [f.target for f in kept] == ["cf-top", "cf-second", "bk-0"]


def test_the_boundary_gap_is_measured_across_the_merged_ranking():
    """The escalation gate reads facts[keep-1] against facts[keep]. On the raw
    concatenation both sat inside the first graph's block, so the gap described
    that graph's internal curve and never the real keep boundary."""
    scorer = _scorer(budget=5)
    keep = 2
    # The fanout must clear scoring_small_threshold (30) for the gap gate to be
    # reached at all, so pad with low-scoring rows that sit far below `keep`.
    padding = [_fact(f"bk-pad-{i:02d}", score=0.01, graph=WIKIPEDIA)
               for i in range(30)]
    # Wikipedia's own head is cleanly separated at its internal keep boundary
    # (0.90 -> 0.20, a gap far above scoring_separation_margin). Globally, though,
    # the true boundary is 0.50 vs 0.49 — ambiguous, and worth escalating.
    facts = [
        _fact("bk-a", score=0.90, graph=WIKIPEDIA),
        _fact("bk-b", score=0.20, graph=WIKIPEDIA),
        *padding,
        _fact("cf-a", score=0.50, graph=ARXIV),
        _fact("cf-b", score=0.49, graph=ARXIV),
    ]
    ranked = scorer._cheap_ranked(facts)

    assert [f.target for f in ranked[:3]] == ["bk-a", "cf-a", "cf-b"]
    # Ranked: 0.90, 0.50, 0.49, 0.20, ... → boundary is 0.50 - 0.49 = 0.01.
    assert scorer._should_escalate(ranked, keep) is True
    # On the raw concatenation the gate compared bk-a(0.90) with bk-b(0.20)
    # — two rows of ONE graph, and neither of them the real boundary.
    assert scorer._should_escalate(facts, keep) is False


def test_a_clean_merged_separation_still_declines_to_escalate():
    """Ranking must not turn the gate into an always-escalate: a genuinely
    separated boundary stays cheap and spends no LLM budget."""
    scorer = _scorer(budget=5)
    facts = [
        _fact("cf-top", score=0.90, graph=ARXIV),
        _fact("bk-a", score=0.85, graph=WIKIPEDIA),
        *(_fact(f"bk-pad-{i:02d}", score=0.01, graph=WIKIPEDIA) for i in range(30)),
    ]

    ranked = scorer._cheap_ranked(facts)

    assert scorer._should_escalate(ranked, keep=2) is False
    assert scorer._budget.remaining == 5, "no budget spent on a clean boundary"


def test_the_margin_keeps_its_raw_curve_meaning_after_leveling():
    """Regression for the normalization/threshold interaction: expansion scores
    reach the gate on the raw support curve (leveling leaves a single or dominant
    graph untouched), so scoring_separation_margin keeps the separation it always
    measured. A full 0-1 rescale by the graph's own peak inflated this boundary
    gap past the margin and silently skipped the re-rank."""
    from ai_graph_retrieval.graph.repository import _level_scores_per_graph

    scorer = _scorer(budget=5)
    facts = [
        _fact("a", score=0.20, graph=WIKIPEDIA),
        _fact("b", score=0.16, graph=WIKIPEDIA),
        _fact("c", score=0.12, graph=WIKIPEDIA),
        *(_fact(f"pad-{i:02d}", score=0.01, graph=WIKIPEDIA) for i in range(30)),
    ]

    leveled = _level_scores_per_graph(facts)

    assert [f.score for f in leveled[:3]] == [0.2, 0.16, 0.12], "raw scale kept"
    # Boundary gap 0.16 - 0.12 = 0.04 < margin 0.05 → ambiguous → re-rank.
    assert scorer._should_escalate(scorer._cheap_ranked(leveled), keep=2) is True
    # Rescaled to the graph's own peak, the same gap reads 0.2 and the gate
    # would wrongly call the boundary cleanly separated.
    rescaled = [f.model_copy(update={"score": round(f.score / 0.20, 4)})
                for f in facts]
    assert scorer._should_escalate(scorer._cheap_ranked(rescaled), keep=2) is False


def test_cheap_ranking_is_deterministic_on_ties():
    """Two identical requests must order identically — equal scores fall back to
    support, then the neighbour name, never to arrival order."""
    a = [_fact("zeta", 0.5, WIKIPEDIA), _fact("alpha", 0.5, ARXIV)]
    b = [_fact("alpha", 0.5, ARXIV), _fact("zeta", 0.5, WIKIPEDIA)]
    scorer = _scorer()

    assert ([f.target for f in scorer._cheap_ranked(a)]
            == [f.target for f in scorer._cheap_ranked(b)] == ["alpha", "zeta"])


# ── 3. agent-facing contributor / author rows ───────────────────────────────


def test_top_facts_ranks_contributors_across_graphs_before_the_cut():
    """find_contributors / find_topic_people: the model's row window must not be
    a positional slice of the first graph's block."""
    facts = [_fact(f"doc-{i}", score=0.05, graph=WIKIPEDIA) for i in range(12)]
    facts += [_fact("cf-major", score=0.90, graph=ARXIV)]

    shown = _top_facts(facts, rows=12)

    assert len(shown) == 12, "the row budget is unchanged"
    assert shown[0].target == "cf-major"


def test_top_facts_ranks_authored_documents_by_date_across_graphs():
    """find_authored_documents orders by publication date, matching the query's
    own ORDER BY — so the newest work leads whichever graph holds it."""
    facts = [
        RelationFact(source="P", target="bk-old", relationship="authored",
                     date="2024-01-01", graph=WIKIPEDIA),
        RelationFact(source="P", target="bk-older", relationship="authored",
                     date="2023-01-01", graph=WIKIPEDIA),
        RelationFact(source="P", target="cf-newest", relationship="authored",
                     date="2026-08-01", graph=ARXIV),
    ]

    shown = _top_facts(facts, rows=2, by_date=True)

    assert [f.target for f in shown] == ["cf-newest", "bk-old"]


def test_top_facts_never_returns_more_than_the_row_budget():
    facts = [_fact(f"n-{i}", score=float(i), graph=WIKIPEDIA) for i in range(40)]
    assert len(_top_facts(facts, rows=12)) == 12


def test_the_pool_still_receives_every_fact_the_tool_fetched():
    """The row window bounds only what the MODEL sees. Truncating the pool too
    would lose evidence the digest ranks and cites."""
    pool = EvidencePool()
    facts = [_fact(f"doc-{i}", score=0.05, graph=WIKIPEDIA) for i in range(12)]
    facts += [_fact("cf-major", score=0.90, graph=ARXIV)]

    pool.add_facts(facts)
    shown = _top_facts(facts, rows=12)

    assert len(pool.snapshot().facts) == 13
    assert len(shown) == 12


# ── 4. paths verified correct for this issue stay unchanged ─────────────────


def test_single_graph_ordering_is_untouched_by_the_claim_rank():
    """One graph ⇒ the fetch is already globally ordered; ranking must be a
    no-op on it rather than a reshuffle."""
    ordered = [
        ClaimFact(subject="A", predicate="p", object="c1", valid_from="2026-01-01",
                  graph=WIKIPEDIA),
        ClaimFact(subject="A", predicate="p", object="c2", valid_from="2025-01-01",
                  graph=WIKIPEDIA),
        ClaimFact(subject="A", predicate="p", object="c3", valid_from="2024-01-01",
                  graph=WIKIPEDIA),
    ]
    assert rank_claims_by_recency(ordered) == ordered


def test_a_single_graph_expansion_keeps_its_query_order():
    """One graph's rows arrive score-ordered from Cypher; the cheap rank must
    preserve exactly that."""
    facts = [_fact("a", 0.9, WIKIPEDIA), _fact("b", 0.5, WIKIPEDIA),
             _fact("c", 0.1, WIKIPEDIA)]
    assert _scorer()._cheap_ranked(facts) == facts


def test_graph_chunk_budgeting_still_interleaves_rather_than_ranks():
    """budget_graph_quotes is the OTHER shape — chunk relevance is not always
    comparable across graphs, so that path interleaves by graph and is
    deliberately left alone here."""
    from ai_graph_retrieval.agent.tools import budget_graph_quotes
    from ai_graph_retrieval.core.models import ChunkQuote

    quotes = [ChunkQuote(chunk_id=f"bk{i}", text=f"wikipedia text {i}",
                         graph=WIKIPEDIA) for i in range(4)]
    quotes += [ChunkQuote(chunk_id="cf1", text="arxiv text", graph=ARXIV)]

    kept = budget_graph_quotes(EvidencePool(), 2, quotes)

    assert {q.graph for q in kept} == {WIKIPEDIA, ARXIV}


def test_get_claims_still_ranks_what_it_shows_the_model():
    """The tool's own ordering is unchanged — it shares the recency helper now,
    but the contract (newest first without a focus) is identical."""
    from ai_graph_retrieval.agent.tools import _rank_claims

    claims = [
        ClaimFact(subject="A", predicate="p", object="old", valid_from="2024-01-01",
                  graph=WIKIPEDIA),
        ClaimFact(subject="A", predicate="p", object="new", valid_from="2026-01-01",
                  graph=ARXIV),
    ]
    out = asyncio.run(_rank_claims(SimpleNamespace(embedder=None), claims, focus=None))

    assert [c.object for c in out] == ["new", "old"]


def test_the_contributor_json_shape_is_unchanged():
    """Ranking moved inside the comprehension; the emitted keys must not."""
    facts = [_fact("doc", score=0.5, graph=WIKIPEDIA, support=3)]
    payload = json.loads(json.dumps({
        "contributors": [
            {"person": f.source, "role": f.relationship, "document": f.target,
             "detail": f.description, "documents_matched": f.support,
             "graph": f.graph}
            for f in _top_facts(facts, 12)
        ],
    }))
    assert payload["contributors"] == [{
        "person": "Hub", "role": "RELATED_TO", "document": "doc",
        "detail": None, "documents_matched": 3, "graph": WIKIPEDIA,
    }]
