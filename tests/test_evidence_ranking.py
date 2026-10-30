"""Excerpt ranking, graph relevance, and windowed recency.

Three defects shared one cause — nothing scored a chunk against the question:

  * the digest ordered excerpts by their DOCUMENT's position in the trust-ranked
    source list. That position is an int, so every chunk of a document tied with
    every other and the character budget kept whichever the pool held first.
    Measured on one Super Research pool: 44 of 63 chunks dropped unscored, and
    within a single document relevance ranged 0.38-0.48 under one shared rank.
  * `relevance` (0.40, the blend's largest weight) is supplied only by the Search
    API, so graph-traversed documents scored 0.0 there and landed on one
    identical trust value — 32 of 48 sources in the measured run.
  * a question that NAMES its period had recency decay applied inside that
    window, pushing the window's own earliest items toward the truncation edge.

Every pass degrades to the previous behaviour when vectors are absent, so these
tests pin both directions: ranked when vectors exist, untouched when they do not.

    uv run pytest tests/test_evidence_ranking.py -v
"""
from __future__ import annotations

from datetime import date

from ai_graph_retrieval.core.models import ChunkQuote, SourceDocument
from ai_graph_retrieval.retrieval.quality import (
    document_relevance,
    rank_quotes,
    rank_sources,
)
from tests.support import make_settings


def _quote(cid: str, doc: str, rel: float = 0.0, text: str = "body") -> ChunkQuote:
    """``rel`` is the question-similarity retrieval/chunk_relevance stamps during
    retrieval; ranking here only reads it."""
    return ChunkQuote(chunk_id=cid, doc_id=doc, doc_title=doc.upper(),
                      text=text, relevance=rel)


# ── excerpt ranking ──────────────────────────────────────────────────────────
# The shared cosine itself is develop's (retrieval/similarity), tested in
# tests/test_chunk_relevance.py. These tests cover what this module adds: using
# the stamped score to ORDER excerpts and to give a graph document a relevance.


def test_relevance_beats_document_position():
    """The defect in miniature: the off-topic chunk belongs to the BEST-ranked
    document, so document-order ranking put it first."""
    quotes = [_quote("off", "top", 0.0), _quote("on", "low", 1.0)]
    sources = [SourceDocument(doc_id="top", title="TOP", trust=0.9),
               SourceDocument(doc_id="low", title="LOW", trust=0.1)]
    ranked = rank_quotes(quotes, sources, make_settings())
    assert [q.chunk_id for q in ranked] == ["on", "off"]


def test_chunks_of_one_document_are_no_longer_tied():
    """These three shared one document rank, so the budget kept whichever came
    first. They must now separate on their own scores."""
    quotes = [_quote("a", "d", 0.1), _quote("b", "d", 0.9), _quote("c", "d", 0.5)]
    ranked = rank_quotes(
        quotes, [SourceDocument(doc_id="d", title="D", trust=0.5)], make_settings())
    assert [q.chunk_id for q in ranked] == ["b", "c", "a"]


def test_ranking_returns_every_quote():
    """Ranking ORDERS; the digest's budget is what drops. A ranking that also
    filtered would lose evidence twice and silently."""
    quotes = [_quote(str(i), "d", 0.5) for i in range(6)]
    assert len(rank_quotes(quotes, [], make_settings())) == 6


def test_nothing_scored_leaves_the_callers_order_untouched():
    """The retrieval pass could not score (no vector store, no chunk index, mixed
    embedding spaces) — the digest's own document-order sort must still decide."""
    quotes = [_quote("z", "d"), _quote("a", "d")]      # all relevance 0.0
    assert [q.chunk_id for q in rank_quotes(quotes, [], make_settings())] == ["z", "a"]


def test_an_unscored_chunk_falls_back_to_its_documents_trust():
    """Partial coverage must not bury unscored chunks below genuinely irrelevant
    ones — trust is the tiebreak, as it was before."""
    quotes = [_quote("noscore", "trusted", 0.0), _quote("irrelevant", "untrusted", 0.01)]
    sources = [SourceDocument(doc_id="trusted", title="T", trust=0.9),
               SourceDocument(doc_id="untrusted", title="U", trust=0.0)]
    ranked = rank_quotes(quotes, sources, make_settings())
    assert [q.chunk_id for q in ranked] == ["noscore", "irrelevant"]


def test_both_weights_zero_disables_the_ranking():
    quotes = [_quote("z", "d", 0.1), _quote("a", "d", 0.9)]
    ranked = rank_quotes(
        quotes, [], make_settings(quote_w_semantic=0.0, quote_w_doc_trust=0.0))
    assert [q.chunk_id for q in ranked] == ["z", "a"]


def test_ranking_is_deterministic_across_identical_inputs():
    """Continuous scores make near-ties common; without a stable identifier
    tiebreak the citation order would vary between identical requests."""
    quotes = [_quote(c, "d", 0.5) for c in "badc"]     # every score identical
    runs = [[q.chunk_id for q in rank_quotes(quotes, [], make_settings())]
            for _ in range(3)]
    assert runs[0] == runs[1] == runs[2] == ["a", "b", "c", "d"]


# ── graph document relevance ─────────────────────────────────────────────────


def test_document_relevance_is_its_best_chunk_under_both_identifiers():
    """Max, not mean: one on-topic passage makes a document worth reading, and
    averaging would penalise long documents for unrelated sections. Keyed on
    doc_id AND title because either may be the only id a record carries."""
    rel = document_relevance([_quote("weak", "d", 0.2), _quote("strong", "d", 1.0)])
    assert rel["d"] == 1.0
    assert rel["D"] == 1.0


def test_document_relevance_is_empty_when_nothing_was_scored():
    assert document_relevance([_quote("a", "d")]) == {}


def test_a_chunk_score_supplies_the_relevance_a_graph_document_lacks():
    """The measured tie: graph documents carry relevance 0.0, so the blend's
    largest weight contributed nothing and they all scored alike."""
    s = make_settings()
    quotes = [_quote("on", "graphdoc", 1.0), _quote("off", "otherdoc", 0.0)]
    sources = [SourceDocument(doc_id="graphdoc", title="G"),
               SourceDocument(doc_id="otherdoc", title="O")]
    plain = rank_sources(sources, quotes, s, today=date(2026, 8, 27))
    assert plain[0].trust == plain[1].trust, "precondition: they tie today"

    scored = rank_sources(sources, quotes, s, today=date(2026, 8, 27),
                          relevance_by_key=document_relevance(quotes))
    assert scored[0].doc_id == "graphdoc"
    assert scored[0].trust > scored[1].trust


def test_an_api_supplied_relevance_is_never_overwritten():
    """The API's fused cosine/BM25 is computed over the whole index; a chunk score
    over the handful we happen to hold must not replace it."""
    s = make_settings()
    src = [SourceDocument(doc_id="api", title="A", relevance=0.9)]
    out = rank_sources(src, [_quote("c", "api")], s, today=date(2026, 8, 27),
                       relevance_by_key={"api": 0.0})
    baseline = rank_sources(src, [_quote("c", "api")], s, today=date(2026, 8, 27))
    assert out[0].trust == baseline[0].trust


# ── windowed recency ─────────────────────────────────────────────────────────


def test_a_stated_window_damps_the_recency_penalty():
    """A question that NAMED its period already stated its time scope; decaying
    inside that window counts it twice and pushes the window's earliest items
    toward the truncation edge."""
    s = make_settings()
    srcs = [SourceDocument(doc_id="old", title="O", date="2023-08-27"),
            SourceDocument(doc_id="new", title="N", date="2026-08-27")]
    unscoped = {x.doc_id: x.trust for x in
                rank_sources(srcs, [], s, today=date(2026, 8, 27))}
    scoped = {x.doc_id: x.trust for x in
              rank_sources(srcs, [], s, today=date(2026, 8, 27), time_scoped=True)}
    assert (unscoped["new"] - unscoped["old"]) > (scoped["new"] - scoped["old"])


def test_damping_of_one_restores_the_previous_behaviour_exactly():
    s = make_settings(trust_recency_windowed_damping=1.0)
    srcs = [SourceDocument(doc_id="old", title="O", date="2023-08-27"),
            SourceDocument(doc_id="new", title="N", date="2026-08-27")]
    a = [x.trust for x in rank_sources(srcs, [], s, today=date(2026, 8, 27))]
    b = [x.trust for x in rank_sources(srcs, [], s, today=date(2026, 8, 27),
                                       time_scoped=True)]
    assert a == b


def test_no_window_leaves_recency_at_full_weight():
    """The default path must be untouched: with no window asked, the newest of
    two editions still wins on recency."""
    s = make_settings()
    srcs = [SourceDocument(doc_id="old", title="O", date="2023-08-27"),
            SourceDocument(doc_id="new", title="N", date="2026-08-27")]
    ranked = rank_sources(srcs, [], s, today=date(2026, 8, 27))
    assert ranked[0].doc_id == "new"


# ── multi-graph fairness in the shared chunk budget ──────────────────────────
# `_fan_out` hands the budget every row of the first configured source, then
# every row of the second (Settings.graph_sources puts wikipedia first and
# appends arXiv). Filling a SHARED cap from that list in order meant the
# first graph took as much as it had: measured on the live graphs, arXiv
# contributed ZERO of Search's 8 slots and zero of Super Research's 40 on a
# question its own documents answered best.


def _graph_rows(*counts: tuple[str, int]) -> list[ChunkQuote]:
    """Quotes in `_fan_out` order — one source's rows in full, then the next."""
    return [
        ChunkQuote(chunk_id=f"{g}{i}", doc_id=f"{g}d{i}", doc_title=f"{g} {i}",
                   text=f"{g} text {i}", graph=g)
        for g, n in counts for i in range(n)
    ]


def test_a_shared_cap_is_split_between_the_graphs():
    from collections import Counter

    from ai_graph_retrieval.agent.tools import budget_graph_quotes
    from ai_graph_retrieval.core.models import EvidencePool

    rows = _graph_rows(("wikipedia", 50), ("arxiv", 50))
    for cap in (8, 40):  # Search's cap and Super Research's
        kept = budget_graph_quotes(EvidencePool(), cap, rows)
        assert Counter(q.graph for q in kept) == {"wikipedia": cap // 2,
                                                 "arxiv": cap // 2}


def test_a_graph_with_less_than_its_share_yields_the_remainder():
    """A fair START, not a quota: cap_per_document appends the overflow, so the
    budget still fills completely rather than stranding slots."""
    from collections import Counter

    from ai_graph_retrieval.agent.tools import budget_graph_quotes
    from ai_graph_retrieval.core.models import EvidencePool

    kept = budget_graph_quotes(
        EvidencePool(), 8, _graph_rows(("wikipedia", 2), ("arxiv", 50))
    )
    assert len(kept) == 8
    assert Counter(q.graph for q in kept) == {"wikipedia": 2, "arxiv": 6}


def test_a_single_graph_read_is_untouched():
    """No interleave to do, and the caller's order is the ranking."""
    from ai_graph_retrieval.agent.tools import budget_graph_quotes
    from ai_graph_retrieval.core.models import EvidencePool

    rows = _graph_rows(("wikipedia", 20))
    kept = budget_graph_quotes(EvidencePool(), 8, rows)
    assert [q.chunk_id for q in kept] == [f"wikipedia{i}" for i in range(8)]


def test_relevance_order_within_a_graph_survives_the_interleave():
    """`chunks_by_similarity` hands over a cross-graph cosine-sorted list. The
    interleave may move a graph's block, never reorder inside it."""
    from ai_graph_retrieval.agent.tools import budget_graph_quotes
    from ai_graph_retrieval.core.models import EvidencePool

    rows = _graph_rows(("wikipedia", 6), ("arxiv", 6))
    kept = budget_graph_quotes(EvidencePool(), 12, rows)
    for g in ("wikipedia", "arxiv"):
        assert ([q.chunk_id for q in kept if q.graph == g]
                == [q.chunk_id for q in rows if q.graph == g])


def test_the_budget_still_counts_what_the_pool_already_holds():
    """Fairness must not widen the cap: graph chunks already pooled still spend it."""
    from ai_graph_retrieval.agent.tools import budget_graph_quotes
    from ai_graph_retrieval.core.models import EvidencePool

    pool = EvidencePool()
    pool.add_quotes([
        ChunkQuote(chunk_id="already", text="pooled graph chunk", graph="wikipedia"),
    ])
    kept = budget_graph_quotes(
        pool, 4, _graph_rows(("wikipedia", 10), ("arxiv", 10))
    )
    assert len(kept) == 3  # 4 minus the one graph chunk already pooled


def test_duplicate_text_never_spends_a_slot():
    """Dedup runs before the count, and the interleave must not bypass it."""
    from ai_graph_retrieval.agent.tools import budget_graph_quotes
    from ai_graph_retrieval.core.models import EvidencePool

    pool = EvidencePool()
    pool.add_quotes([ChunkQuote(chunk_id="seed", text="Shared  TEXT", graph="wikipedia")])
    incoming = [
        ChunkQuote(chunk_id="dupe", text="shared text", graph="arxiv"),
        ChunkQuote(chunk_id="fresh", text="something else", graph="arxiv"),
    ]
    kept = budget_graph_quotes(pool, 8, incoming)
    assert [q.chunk_id for q in kept] == ["fresh"]


# ── the digest's excerpt budget ──────────────────────────────────────────────


def test_the_rungs_budget_is_never_shrunk_by_another_branch():
    """This began as a guard on the enumeration branch: `digest_agg_quote_char_budget`
    was sized against the 65k GLOBAL default, so taking it unconditionally SHRANK the
    window of any rung whose own budget was larger — backwards for the shape needing
    most breadth.

    That branch and its setting were deleted, so the specific bug cannot recur.
    What survives is the invariant it protected, and there are now two per-mode
    budgets to hold it against: a rung's own budget is the one that applies, and the
    deeper mode never reads less than the shallower one.
    """
    s = make_settings()
    # Each agent rung carries its own budget, and neither falls back to the global
    # default — that fallback is what let a global constant override a rung.
    assert s.search_digest_quote_chars > 0
    assert s.super_research_digest_quote_chars > 0
    # Super Research retrieves strictly more (up to 40 graph chunks against 8), so a
    # budget that did not rise with it would drop what the extra rounds paid to find.
    assert s.super_research_digest_quote_chars >= s.search_digest_quote_chars
    # ...and both clear the global default, which is a floor, not theirs.
    assert min(s.search_digest_quote_chars,
               s.super_research_digest_quote_chars) >= s.digest_quote_char_budget


# ── the seed carries the engine's per-chunk score ────────────────────────────


def test_api_chunks_of_one_document_keep_the_engines_order_through_ranking():
    """Without a per-chunk score, every chunk of an API document TIED in
    rank_quotes (relevance 0.0, one shared doc trust) and the tie broke on
    chunk_id — alphabetical UUIDs — discarding the engine's own order that the
    seed had just sorted by. When the digest budget binds, that decided which of
    a document's chunks survived by id instead of by the engine. The seed now
    stamps ``relevance=chunk.score``, so the chunks never tie and the engine's
    order rides through."""
    # ids chosen so the OLD tiebreak (alphabetical) would invert the score order
    quotes = [_quote("zz-best", "doc", 0.22), _quote("aa-weaker", "doc", 0.03)]
    sources = [SourceDocument(doc_id="doc", title="DOC", trust=0.5)]
    ranked = rank_quotes(quotes, sources, make_settings())
    assert [q.chunk_id for q in ranked] == ["zz-best", "aa-weaker"]

