"""Cross-graph score comparability: a per-graph quantity must be scaled per graph.

Sibling to test_cross_graph_truncation, which covers the ORDERING half of the same
family of bug. This file covers the SCALE half: three signals are absolute values
produced independently by each graph — a Lucene score (its index's term statistics),
a mention count and an entity degree (its corpus's size and density), and an edge's
support count (how much text that graph holds per relationship). Each was normalised
against ONE max taken over the merged fan-out, which ranks the corpora against each
other instead of the candidates, and hands every row to whichever graph is larger,
denser or better-matching.

The fix is the same everywhere: the DAMPED per-graph correction
(retrieval/quality.py::graph_scalers, ``sqrt(own peak * ceiling)``): the dominant
graph keeps the global scale exactly, a weaker graph is lifted halfway toward it —
never onto it, because a weak graph's best is not automatically as strong as a
strong graph's best. Within one graph nothing moves — the tests below pin that too,
because a normalisation change that reshuffles a single-graph result is a
regression, not a fix.

    uv run pytest tests/test_cross_graph_score_comparability.py -v
"""
from __future__ import annotations

import asyncio
import math
from datetime import date

from ai_graph_retrieval.core.access import AccessContext
from ai_graph_retrieval.core.models import ChunkQuote, SourceDocument
from ai_graph_retrieval.graph.repository import GraphReadRepository
from ai_graph_retrieval.retrieval.quality import graph_peaks, rank_sources
from tests.support import make_settings

# Fan-out order: wikipedia is always built first (Settings.graph_sources).
WIKIPEDIA, ARXIV = "wikipedia", "arxiv"


def _ctx():
    return AccessContext(tenant_id="t", principals=("public",), request_id="r")


def _repo(rows, settings=None):
    """A repository whose fan-out returns `rows` — the concatenation shape."""
    repo = GraphReadRepository(None, settings or make_settings(), http=object())

    async def _stub_rows(_query, _ctx, **_params):
        return rows

    repo._rows = _stub_rows
    return repo


# ── graph_peaks ─────────────────────────────────────────────────────────────


def test_graph_peaks_is_per_graph():
    assert graph_peaks([(WIKIPEDIA, 3.0), (ARXIV, 9.0), (WIKIPEDIA, 5.0)]) == {
        WIKIPEDIA: 5.0, ARXIV: 9.0,
    }


def test_graph_peaks_buckets_untagged_rows_together():
    """A single-graph read never carries the tag; those rows share one bucket,
    which reproduces the previous global behaviour exactly."""
    assert graph_peaks([(None, 4.0), (None, 2.0)]) == {None: 4.0}


def test_graph_peaks_is_empty_for_no_rows():
    assert graph_peaks([]) == {}


# ── 1. entity lexical / BM25 ────────────────────────────────────────────────
#
# `db.index.fulltext.queryNodes` returns a Lucene score whose magnitude depends on
# the index that produced it. Each graph maintains its own index, so the raw numbers
# are on unrelated scales.

_LEXICAL_ROWS = [
    # Wikipedia's index happens to produce large raw scores for this term.
    {"entity_id": "bk-1", "name": "Search Wikipedia", "score": 14.2,
     "degree": 0, "graph": WIKIPEDIA},
    {"entity_id": "bk-2", "name": "Search Wikipedia Minor", "score": 4.0,
     "degree": 0, "graph": WIKIPEDIA},
    # arXiv's index scores its BEST match at 3.1 — the top entity of its
    # corpus, on a scale that has nothing to do with Wikipedia's.
    {"entity_id": "cf-1", "name": "Search arXiv", "score": 3.1,
     "degree": 0, "graph": ARXIV},
]


def _scores(hits):
    return {h.entity_id: h.score for h in hits}


def test_each_index_is_scaled_by_the_damped_divisor():
    """The dominant index keeps the old global scale; the weaker index's best is
    lifted halfway toward it — not onto it."""
    repo = _repo(_LEXICAL_ROWS)

    hits = asyncio.run(repo.resolve_entities(_ctx(), "search", embedding=None))
    scores = _scores(hits)

    w = make_settings().entity_w_lexical
    assert scores["bk-1"] == round(w, 4)
    assert scores["cf-1"] == round(w * (3.1 / math.sqrt(3.1 * 14.2)), 4)
    # Lifted above its raw global share, below the dominant index's best.
    assert w * (3.1 / 14.2) < scores["cf-1"] < scores["bk-1"]
    # ...and the runner-up is scaled against the same divisor as its own best.
    assert scores["bk-2"] == round(w * (4.0 / 14.2), 4)


def test_a_graphs_best_match_is_no_longer_buried_by_another_indexs_scale():
    """The regression: globally normalized, cf-1 scored 3.1/14.2 = 0.22 of the
    lexical weight and sorted BELOW Wikipedia's mediocre second-place row."""
    repo = _repo(_LEXICAL_ROWS)

    hits = asyncio.run(repo.resolve_entities(_ctx(), "search", embedding=None))

    ranked = [h.entity_id for h in hits]
    assert ranked.index("cf-1") < ranked.index("bk-2")


def test_single_graph_lexical_normalization_is_unchanged():
    """One index ⇒ one peak ⇒ exactly the previous global division."""
    rows = [
        {"entity_id": "a", "name": "A", "score": 10.0, "degree": 0, "graph": WIKIPEDIA},
        {"entity_id": "b", "name": "B", "score": 5.0, "degree": 0, "graph": WIKIPEDIA},
        {"entity_id": "c", "name": "C", "score": 2.5, "degree": 0, "graph": WIKIPEDIA},
    ]
    hits = asyncio.run(_repo(rows).resolve_entities(_ctx(), "search", embedding=None))

    w = make_settings().entity_w_lexical
    assert _scores(hits) == {
        "a": round(w, 4), "b": round(w * 0.5, 4), "c": round(w * 0.25, 4),
    }


def test_an_untagged_lexical_row_still_scores():
    """No `graph` tag (single-graph deployments) must not divide by zero or drop
    the row's score to nothing."""
    rows = [{"entity_id": "a", "name": "A", "score": 4.0, "degree": 0}]
    hits = asyncio.run(_repo(rows).resolve_entities(_ctx(), "search", embedding=None))

    assert _scores(hits) == {"a": round(make_settings().entity_w_lexical, 4)}


def test_an_all_zero_lexical_fanout_does_not_divide_by_zero():
    rows = [{"entity_id": "a", "name": "A", "score": 0.0, "degree": 0,
             "graph": WIKIPEDIA}]
    hits = asyncio.run(_repo(rows).resolve_entities(_ctx(), "search", embedding=None))

    assert _scores(hits) == {"a": 0.0}


# ── 2. entity degree ────────────────────────────────────────────────────────
#
# `degree` is an edge count inside one graph, so a denser corpus lifts every one of
# its entities against a sparser one.


def test_degree_is_scaled_against_its_own_graphs_density():
    """A sparse graph's hub is its corpus's most connected entity: the lift
    raises its boost toward the dense hub's — not onto it, because ten edges
    are still not a thousand. Scale-PRESERVING, like the edge leveling below:
    the dense graph's term stays the raw log-degree develop computed."""
    rows = [
        # Wikipedia is dense: its hub carries 1000 edges.
        {"entity_id": "bk-hub", "name": "BK Hub", "score": 0.0, "degree": 1000,
         "graph": WIKIPEDIA},
        # arXiv is sparse: its hub carries 10 — but it IS the hub.
        {"entity_id": "cf-hub", "name": "CF Hub", "score": 0.0, "degree": 10,
         "graph": ARXIV},
    ]
    hits = asyncio.run(_repo(rows).resolve_entities(_ctx(), "search", embedding=None))

    w = make_settings().entity_w_degree
    bk_log, cf_log = math.log10(1001), math.log10(11)
    scores = _scores(hits)
    # The dominant graph keeps its raw term — the pre-split value exactly.
    assert scores["bk-hub"] == round(w * bk_log, 4)
    # The sparse hub is lifted by sqrt(ceiling / own peak): halfway, in log space.
    assert scores["cf-hub"] == round(w * math.sqrt(cf_log * bk_log), 4)
    # Lifted above its raw term, below the dense hub.
    assert round(w * cf_log, 4) < scores["cf-hub"] < scores["bk-hub"]


def test_degree_still_orders_entities_within_a_graph():
    """Correcting the cross-graph scale must not flatten the signal itself."""
    rows = [
        {"entity_id": "hub", "name": "Hub", "score": 0.0, "degree": 100,
         "graph": WIKIPEDIA},
        {"entity_id": "thin", "name": "Thin", "score": 0.0, "degree": 1,
         "graph": WIKIPEDIA},
    ]
    hits = asyncio.run(_repo(rows).resolve_entities(_ctx(), "search", embedding=None))

    assert [h.entity_id for h in hits] == ["hub", "thin"]
    assert _scores(hits)["hub"] > _scores(hits)["thin"]


def test_a_single_graph_keeps_the_raw_degree_term():
    """The lift is a CROSS-GRAPH correction and nothing else: on one graph every
    peak IS the ceiling, the lift is 1.0, and the term is the raw
    `entity_w_degree * log10(1 + degree)` it has always been. A divisor here once
    re-weighted degree against the lexical signal on every request — by a factor
    that moved with whatever hub this result set happened to contain. How much
    connectedness is worth is `entity_w_degree`'s decision, not the corpus's."""
    w = make_settings().entity_w_degree
    rows = [
        {"entity_id": "hub", "name": "Hub", "score": 0.0, "degree": 100_000,
         "graph": WIKIPEDIA},
        {"entity_id": "mid", "name": "Mid", "score": 0.0, "degree": 100,
         "graph": WIKIPEDIA},
    ]
    hits = asyncio.run(_repo(rows).resolve_entities(_ctx(), "search", embedding=None))

    assert _scores(hits) == {
        "hub": round(w * math.log10(100_001), 4),
        "mid": round(w * math.log10(101), 4),
    }


def test_a_zero_degree_fanout_does_not_divide_by_zero():
    rows = [{"entity_id": "a", "name": "A", "score": 0.0, "degree": 0,
             "graph": WIKIPEDIA}]
    hits = asyncio.run(_repo(rows).resolve_entities(_ctx(), "search", embedding=None))

    assert _scores(hits) == {"a": 0.0}


# ── 3. relationship support ─────────────────────────────────────────────────
#
# EXPAND_NEIGHBORS returns support_weight * log10(1+support); `support` counts the
# chunks of ONE graph evidencing the edge, so the graph holding more text per
# relationship wins every row.


def _edge(target: str, score: float, graph: str, support: int = 1) -> dict:
    return {"source": "Hub", "target": target, "relationship": "RELATED_TO",
            "support": support, "score": score, "graph": graph}


def test_expansion_scores_are_leveled_per_graph():
    """The dominant graph keeps the raw curve byte-for-byte (downstream absolute
    thresholds — the neighbour scorer's separation margin — keep their meaning);
    the weaker graph is lifted by sqrt(ceiling/peak), order and spacing intact."""
    rows = [
        _edge("bk-strong", 0.30, WIKIPEDIA, support=100),
        _edge("bk-weak", 0.15, WIKIPEDIA, support=10),
        _edge("cf-strong", 0.06, ARXIV, support=3),
    ]
    facts = asyncio.run(_repo(rows).expand_entity(_ctx(), "Hub"))
    by_target = {f.target: f.score for f in facts}

    assert by_target == {
        "bk-strong": 0.3, "bk-weak": 0.15,
        "cf-strong": round(0.06 * math.sqrt(0.30 / 0.06), 4),
    }
    assert 0.06 < by_target["cf-strong"] < by_target["bk-strong"]


def test_a_sparse_graphs_best_edge_is_not_buried_by_a_dense_ones_middling_edge():
    """The regression: on raw support, cf-top (0.06) sorted below every
    Wikipedia row, so a text-poor graph never reached the digest's head."""
    rows = [
        _edge("bk-top", 0.30, WIKIPEDIA, support=100),
        _edge("bk-mid", 0.12, WIKIPEDIA, support=5),
        _edge("cf-top", 0.06, ARXIV, support=3),
    ]
    facts = asyncio.run(_repo(rows).expand_entity(_ctx(), "Hub"))
    ranked = sorted(facts, key=lambda f: -f.score)

    assert [f.target for f in ranked][:2] == ["bk-top", "cf-top"]


def test_expansion_order_within_a_graph_is_preserved():
    """Leveling is a positive per-graph scalar — a single graph keeps the query's
    raw scores and its own ORDER BY score DESC exactly."""
    rows = [
        _edge("a", 0.30, WIKIPEDIA), _edge("b", 0.20, WIKIPEDIA),
        _edge("c", 0.10, WIKIPEDIA),
    ]
    facts = asyncio.run(_repo(rows).expand_entity(_ctx(), "Hub"))

    assert [f.target for f in facts] == ["a", "b", "c"]
    assert [f.score for f in facts] == [0.3, 0.2, 0.1]


def test_support_itself_is_carried_through_unchanged():
    """Only the derived SCORE is rescaled — `support` is the evidence count the
    digest and the agent's tool payloads report, and must stay absolute."""
    rows = [_edge("a", 0.30, WIKIPEDIA, support=100),
            _edge("b", 0.06, ARXIV, support=3)]
    facts = asyncio.run(_repo(rows).expand_entity(_ctx(), "Hub"))

    assert {f.target: f.support for f in facts} == {"a": 100, "b": 3}


def test_an_all_zero_expansion_is_left_alone():
    rows = [_edge("a", 0.0, WIKIPEDIA), _edge("b", 0.0, WIKIPEDIA)]
    facts = asyncio.run(_repo(rows).expand_entity(_ctx(), "Hub"))

    assert [f.score for f in facts] == [0.0, 0.0]


# ── 4. document mentions ────────────────────────────────────────────────────
#
# `mentions` counts the chunks of ONE graph that name the anchor, and feeds
# `multisource` — 0.25 of the trust blend.


def _source(doc_id: str, mentions: int, graph: str) -> SourceDocument:
    return SourceDocument(doc_id=doc_id, title=doc_id, mentions=mentions,
                          graph=graph, date="2026-01-01")


def _trust(sources, quotes=None):
    ranked = rank_sources(sources, quotes or [], make_settings(),
                          today=date(2026, 1, 1))
    return {s.doc_id: s.trust for s in ranked}


def test_mentions_are_scaled_by_the_damped_divisor():
    """The dense graph's top document keeps the full multisource term; the sparse
    graph's top is lifted from its raw share toward it — not onto it."""
    s = make_settings()
    trust = _trust([_source("bk", 40, WIKIPEDIA), _source("cf", 4, ARXIV)])

    # Same trust on every other feature, so the gap is the multisource term alone:
    # cf's share rises from 4/40 to 4/sqrt(4*40), half the distance in log space.
    lifted = 4 / math.sqrt(4 * 40)
    assert trust["bk"] > trust["cf"]
    assert round(trust["bk"] - trust["cf"], 4) == round(
        s.trust_w_multisource * (1.0 - lifted), 4
    )


def test_a_sparse_graphs_top_document_is_not_outranked_on_corpus_size_alone():
    """The regression: cf-top scored 4/40 = 0.1 of the multisource term purely
    because the other corpus holds more chunks per document."""
    sources = [
        _source("bk-top", 40, WIKIPEDIA),
        _source("bk-mid", 6, WIKIPEDIA),
        _source("cf-top", 4, ARXIV),
    ]
    ranked = rank_sources(sources, [], make_settings(), today=date(2026, 1, 1))

    assert [s.doc_id for s in ranked][:2] == ["bk-top", "cf-top"]


def test_mentions_still_order_documents_within_a_graph():
    trust = _trust([_source("many", 40, WIKIPEDIA), _source("few", 4, WIKIPEDIA)])

    assert trust["many"] > trust["few"]


def test_single_graph_trust_is_unchanged_by_the_per_graph_split():
    """One graph ⇒ one peak ⇒ the previous global max, value for value."""
    sources = [_source("a", 10, WIKIPEDIA), _source("b", 5, WIKIPEDIA),
               _source("c", 1, WIKIPEDIA)]
    s = make_settings()
    total_w = (s.trust_w_relevance + s.trust_w_multisource + s.trust_w_recency
               + s.trust_w_support + s.trust_w_authority)
    expected = {
        doc: round((s.trust_w_multisource * (m / 10)
                    + s.trust_w_recency * 1.0
                    + s.trust_w_support * 0.4
                    + s.trust_w_authority * 0.5) / total_w, 4)
        for doc, m in (("a", 10), ("b", 5), ("c", 1))
    }
    assert _trust(sources) == expected


def test_untagged_sources_share_one_bucket():
    """Sources reached without a graph tag keep the old global normalisation."""
    a = SourceDocument(doc_id="a", title="a", mentions=10, date="2026-01-01")
    b = SourceDocument(doc_id="b", title="b", mentions=5, date="2026-01-01")

    trust = _trust([a, b])
    assert trust["a"] > trust["b"]


def test_zero_mentions_across_a_graph_do_not_divide_by_zero():
    """A graph reached only by the chunk lane carries mentions 0 on every row."""
    sources = [_source("bk", 10, WIKIPEDIA),
               SourceDocument(doc_id="cf", title="cf", mentions=0,
                              graph=ARXIV, date="2026-01-01")]

    trust = _trust(sources)
    assert trust["cf"] < trust["bk"]


def test_the_other_trust_features_are_untouched():
    """Only `multisource` changed: a quote-supported document still outranks an
    unsupported one at equal mentions, on the same graph."""
    sources = [_source("supported", 10, WIKIPEDIA),
               _source("bare", 10, WIKIPEDIA)]
    quotes = [ChunkQuote(chunk_id="c1", doc_id="supported", text="t",
                         graph=WIKIPEDIA)]

    trust = _trust(sources, quotes)
    assert trust["supported"] > trust["bare"]
