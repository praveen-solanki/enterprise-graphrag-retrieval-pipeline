"""Trust ranking must lead with query relevance, not raw graph-mention count.

Regression guard for the "exact-match doc buried below a well-connected but
irrelevant doc" bug: a source carrying strong retrieval `relevance` (the fused
chunk cosine/BM25 the engine computed) should outrank a doc that is merely
mentioned by many graph entities. Relevance is the engine's signal — this file
does NOT reintroduce any hand-rolled term/title matching.

    uv run pytest tests/test_source_relevance_ranking.py -v
"""
from __future__ import annotations

from datetime import date

from ai_graph_retrieval.core.models import ChunkQuote, EvidencePool, SourceDocument
from ai_graph_retrieval.retrieval.quality import rank_sources
from tests.support import make_settings

_S = make_settings()
# Fixed reference date: rank_sources' recency term decays against `today`, so letting
# it default to the wall clock would make these assertions drift (and eventually flip)
# as real time passes. Every dated fixture below is relative to this date.
_TODAY = date(2026, 7, 27)


def _rank(srcs, quotes=None):
    return rank_sources(srcs, quotes or [], _S, today=_TODAY)


def test_relevant_low_mention_doc_outranks_irrelevant_high_mention_doc():
    # The K8-dashboards case: an on-topic chunk match (mentions 0) vs a well-connected
    # but irrelevant 2024 doc (mentions 7).
    srcs = [
        SourceDocument(doc_id="irrelevant", title="Semantic Chunking - ES Impl",
                       date="2024-08-12", mentions=7, relevance=0.0),
        SourceDocument(doc_id="match", title="K8s Dashboard",
                       date="2026-06-17", mentions=0, relevance=0.62),
    ]
    ranked = _rank(srcs)
    assert ranked[0].doc_id == "match"
    assert ranked[0].trust > ranked[1].trust


def test_mentions_still_break_ties_when_relevance_is_equal():
    # Relevance is the lead, not the only signal: with equal relevance the more
    # corroborated (higher-mention) doc still wins.
    srcs = [
        SourceDocument(doc_id="a", title="Doc A", mentions=1, relevance=0.5, date="2026-01-01"),
        SourceDocument(doc_id="b", title="Doc B", mentions=8, relevance=0.5, date="2026-01-01"),
    ]
    ranked = _rank(srcs)
    assert ranked[0].doc_id == "b"


def test_higher_relevance_wins_over_recency_and_authority():
    # A strongly-matching older doc beats a fresh but weakly-matching one.
    srcs = [
        SourceDocument(doc_id="fresh_weak", title="X", date="2026-07-01", relevance=0.1),
        SourceDocument(doc_id="old_strong", title="Y", date="2023-01-01", relevance=0.95),
    ]
    assert _rank(srcs)[0].doc_id == "old_strong"


def test_add_sources_merges_relevance_and_mentions():
    # Same doc from the chunk lane (relevance, no mentions) then graph (mentions, no
    # relevance) must end up carrying BOTH — not first-wins-drop.
    pool = EvidencePool()
    pool.add_sources([SourceDocument(doc_id="d", title="D", relevance=0.7)])
    pool.add_sources([SourceDocument(doc_id="d", title="D", mentions=5, url="u")])
    merged = pool.snapshot().sources
    assert len(merged) == 1
    assert merged[0].relevance == 0.7
    assert merged[0].mentions == 5
    assert merged[0].url == "u"


def test_ranking_is_stable_for_equal_trust():
    # Equal-scoring sources must keep their input order (stable sort), so repeated
    # requests present citations in the same sequence.
    srcs = [SourceDocument(doc_id=f"d{i}", title=f"Doc {i}", relevance=0.5,
                           mentions=1, date="2026-01-01") for i in range(6)]
    order = [s.doc_id for s in _rank(srcs)]
    assert order == [s.doc_id for s in _rank(srcs)]
    assert order == [f"d{i}" for i in range(6)]


# ── config validation + title-only support + order determinism ───────────────

def test_invalid_ranking_weights_are_rejected_at_startup():
    import pytest
    from pydantic import ValidationError
    for bad in ({"trust_w_relevance": float("nan")},
                {"trust_w_multisource": float("inf")},
                {"trust_w_recency": -0.1}):
        with pytest.raises(ValidationError):
            make_settings(**bad)


def test_title_only_source_earns_support_from_its_quote():
    # A chunk-lane source can be title-only; its verbatim quote must still count
    # as support, exactly as it would with a doc_id.
    from ai_graph_retrieval.core.models import ChunkQuote
    srcs = [SourceDocument(doc_id=None, title="Title Only Doc", relevance=0.5),
            SourceDocument(doc_id="other", title="Other", relevance=0.5)]
    quotes = [ChunkQuote(chunk_id="c1", doc_id=None, doc_title="Title Only Doc", text="x")]
    ranked = rank_sources(srcs, quotes, _S, today=_TODAY)
    by_title = {s.title: s.trust for s in ranked}
    assert by_title["Title Only Doc"] > by_title["Other"]   # support 1.0 vs 0.4


def test_equal_trust_ordering_is_input_order_independent():
    srcs = [SourceDocument(doc_id=f"d{i}", title=f"Doc {i}", relevance=0.5,
                           mentions=1, date="2026-01-01") for i in range(6)]
    forward = [s.doc_id for s in _rank(list(srcs))]
    backward = [s.doc_id for s in _rank(list(reversed(srcs)))]
    assert forward == backward             # identity tie-break, not arrival order


# ── citation hover preview ────────────────────────────────────────

def _doc(**kw):
    return SourceDocument(**{"doc_id": "d1", "title": "Doc", **kw})


def test_the_document_opening_is_preferred_over_the_retrieved_excerpt():
    # Ingestion stamps the document's own opening; that is what a hover card should
    # show, and it is the only option for a document with no excerpt at all.
    quotes = [ChunkQuote(chunk_id="c1", doc_id="d1", doc_title="Doc", text="A mid-document chunk.")]
    out = rank_sources([_doc(snippet="The document opens like this, plainly.")],
                       quotes, make_settings())
    assert out[0].snippet == "The document opens like this, plainly."


def test_a_useless_short_opening_falls_back_to_the_excerpt():
    # Real stamped openings run to a single character; an excerpt beats a preview the
    # reader cannot use.
    quotes = [ChunkQuote(chunk_id="c1", doc_id="d1", doc_title="Doc",
                         text="A genuinely informative chunk about leave policy.")]
    out = rank_sources([_doc(snippet="O")], quotes, make_settings())
    assert out[0].snippet.startswith("A genuinely informative chunk")


def test_a_document_with_neither_opening_nor_excerpt_has_no_preview():
    out = rank_sources([_doc()], [], make_settings())
    assert out[0].snippet is None


def test_a_preview_never_ends_on_a_stub_left_by_a_fixed_length_stamp():
    # Ingestion cuts at a character count, so a stored opening can end mid-word:
    # "…across the country. O".
    out = rank_sources([_doc(snippet="Invoy has a ten year history of proving things. O")],
                       [], make_settings())
    assert out[0].snippet == "Invoy has a ten year history of proving things.…"


def test_a_preview_does_not_repeat_the_title_it_sits_under():
    # The hover card shows the title already; an opening that restates it spends the
    # reader's ~100 characters saying nothing.
    out = rank_sources(
        [_doc(title="India Employee Handbook V-1.4.pdf",
              snippet="India Employee Handbook V-1.4.docx Table of Contents Introduction 4 Purpose")],
        [], make_settings())
    assert not out[0].snippet.lower().startswith("india employee handbook v-1.4.pdf")
    assert "Table of Contents" in out[0].snippet


def test_a_short_title_can_never_eat_the_opening():
    out = rank_sources([_doc(title="IT", snippet="ITSM tooling is reviewed each quarter by the team.")],
                       [], make_settings())
    assert out[0].snippet.startswith("ITSM tooling")


def test_the_preview_is_one_line_within_the_configured_width():
    quotes = [ChunkQuote(chunk_id="c1", doc_id="d1", doc_title="Doc",
                         text="Line one.\n\n| Col | Col |\n| --- | --- |\nLine two continues here " + "x" * 400)]
    s = make_settings()
    out = rank_sources([_doc()], quotes, s)
    assert "\n" not in out[0].snippet and "|" not in out[0].snippet
    assert len(out[0].snippet) <= s.citation_snippet_chars + 1   # +1 for the ellipsis


def test_the_preview_ends_on_a_completed_sentence_not_a_character_count():
    # A card that stops mid-thought reads as broken however tidy the count is, so the
    # width is a ceiling and the sentence is the target.
    text = ("This page walks you through submitting a ticket on behalf of a customer. "
            "It then covers escalation, which is a much longer second sentence that "
            "would run past the ceiling if we let it.")
    out = rank_sources([_doc(snippet=text)], [], make_settings())[0].snippet
    assert out.endswith("on behalf of a customer.")
    assert "…" not in out                      # nothing was truncated mid-thought


def test_an_abbreviation_does_not_end_the_preview_early():
    # "U.S.A." and "V-1.4" both contain periods that are not sentence ends.
    text = ("The 2025 Acme U.S.A. Employee Handbook V-1.4 sets out the terms that apply "
            "to every employee across the organisation and its subsidiaries worldwide.")
    out = rank_sources([_doc(snippet=text)], [], make_settings())[0].snippet
    assert "Employee Handbook V-1.4 sets out" in out


def test_a_trivially_short_first_sentence_does_not_become_the_preview():
    # "Overview." is a complete sentence and a useless card.
    text = ("Overview. Recognition and rewards are administered by the People team, who "
            "own the budget, the approval flow and the quarterly reporting cycle.")
    out = rank_sources([_doc(snippet=text)], [], make_settings())[0].snippet
    assert out != "Overview."
    assert "Recognition and rewards" in out


def test_a_preview_with_no_sentence_break_is_cut_on_a_word_boundary():
    text = "Segment Commercial Industry IT Services and Consulting Total ARR $57,720 " + "detail " * 40
    s = make_settings()
    out = rank_sources([_doc(snippet=text)], [], s)[0].snippet
    assert out.endswith("…") and len(out) <= s.citation_snippet_chars + 1
    assert not out.rstrip("…").endswith(" ")   # no dangling space before the ellipsis
