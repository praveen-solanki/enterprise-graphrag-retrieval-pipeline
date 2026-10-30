"""Dated evidence reaches the model newest first.

The reproduction (screenshot, "what leadership changes has the CEO announced?"):
the answer's sections ran April 2025 → … → July 2026, oldest at the top, even
though the synthesis prompt already asked for newest-first.

The prompt was not the bug. The digest RENDERS a date on FACTS and on SOURCES
but ordered both by RELEVANCE (structural/score/support) and by TRUST — so the
model was handed the dated items oldest-first and wrote them out in the order it
received them. CLAIMS was already date-ordered (`rank_claims_by_recency`), which
is why the failure looked inconsistent.

Selection is untouched: relevance still decides WHAT survives each cap, and the
date decides only how the survivors are PRESENTED.

STRUCTURED dates only. Every key here is a date the evidence carries as a field
— a claim's valid_from, a fact's date, a source document's date, and for an
excerpt its DOCUMENT's date. Reading an event date out of passage PROSE in code
(the `latest_date_in_text` month/date regex) was removed deliberately: the
digest does no text parsing, and a date stated only in prose is the synthesis
prompt's to read ("a date stated only in prose counts like a dated field").

    uv run pytest tests/test_dated_presentation_order.py -v
"""
from __future__ import annotations

from ai_graph_retrieval.core.models import (
    ChunkQuote,
    ClaimFact,
    Evidence,
    RelationFact,
    SourceDocument,
)
from ai_graph_retrieval.retrieval.quality import order_by_date_desc
from ai_graph_retrieval.synthesis.composer import AnswerComposer
from tests.support import make_settings


def _digest(ev: Evidence, **over) -> str:
    settings = make_settings()
    for k, v in over.items():
        setattr(settings, k, v)
    return AnswerComposer(settings)._build_digest(ev, strict=False)[0]


def _section(digest: str, header: str) -> list[str]:
    block = digest.split(f"{header}:\n", 1)[1].split("\n\n", 1)[0]
    return [line for line in block.splitlines() if line.startswith("- ")]


def _fact(target: str, date: str | None, **kw) -> RelationFact:
    return RelationFact(source="CEO", relationship="ANNOUNCED", target=target,
                        date=date, **kw)


# ── the reproduction ────────────────────────────────────────────────────────


def test_the_screenshots_ordering_is_inverted():
    """The exact shape of the reported answer: announcements spread across two
    years, arriving relevance-ranked with the OLDEST most relevant."""
    ev = Evidence(facts=[
        _fact("CX changes", "2026-05-01", structural=True, score=0.9),
        _fact("Platform reorg", "2026-06-01", score=0.5),
        _fact("Comp cycle", "2025-04-01", structural=True, score=0.95),
        _fact("July change", "2026-07-01", score=0.2),
    ])

    dates = [line.split("[dated: ")[1].rstrip("]") for line in _section(_digest(ev), "FACTS")]

    assert dates == sorted(dates, reverse=True)
    assert dates[0] == "2026-07-01" and dates[-1] == "2025-04-01"


def test_sources_carrying_dates_are_newest_first():
    """SOURCES renders `(date)` too, and was ordered by trust."""
    ev = Evidence(sources=[
        SourceDocument(doc_id="d1", title="April 2025 memo", date="2025-04-01"),
        SourceDocument(doc_id="d2", title="July 2026 memo", date="2026-07-01"),
        SourceDocument(doc_id="d3", title="May 2026 memo", date="2026-05-01"),
    ])

    assert _section(_digest(ev), "SOURCES") == [
        "- July 2026 memo (2026-07-01)",
        "- May 2026 memo (2026-05-01)",
        "- April 2025 memo (2025-04-01)",
    ]


# ── the ordering rule itself ────────────────────────────────────────────────


def test_mixed_event_and_documented_basis_order_on_the_same_axis():
    """Both bases carry their date in the same field, so they interleave by DATE
    rather than being grouped by basis."""
    ev = Evidence(facts=[
        _fact("documented change", "2026-03-01", date_basis="documented"),
        _fact("event change", "2026-08-01", date_basis="event"),
        _fact("older event", "2026-01-01", date_basis="event"),
    ])

    lines = _section(_digest(ev), "FACTS")

    assert "event change" in lines[0]
    assert "documented change" in lines[1]
    assert "older event" in lines[2]


def test_equal_dates_keep_their_relevance_order():
    """A date tie falls back to the ranking that selected them — stable, not
    arbitrary, and never re-sorted on an unrelated key."""
    ev = Evidence(facts=[
        _fact("lower relevance", "2026-05-01", score=0.1),
        _fact("higher relevance", "2026-05-01", score=0.9),
    ])

    lines = _section(_digest(ev), "FACTS")

    assert "higher relevance" in lines[0]
    assert "lower relevance" in lines[1]


def test_undated_rows_sort_last_and_keep_their_order():
    ev = Evidence(facts=[
        _fact("undated A", None, score=0.9),
        _fact("dated old", "2025-01-01", score=0.5),
        _fact("undated B", None, score=0.8),
        _fact("dated new", "2026-09-01", score=0.1),
    ])

    lines = _section(_digest(ev), "FACTS")

    assert "dated new" in lines[0]
    assert "dated old" in lines[1]
    # Undated rows are not sortable against dated ones; their own order stands.
    assert "undated A" in lines[2]
    assert "undated B" in lines[3]


def test_an_entirely_undated_section_is_unchanged():
    """Existing behaviour for evidence carrying no dates at all: pure relevance."""
    ev = Evidence(facts=[
        _fact("weak", None, score=0.1),
        _fact("strong", None, score=0.9),
        _fact("structural", None, structural=True, score=0.0),
    ])

    lines = _section(_digest(ev), "FACTS")

    assert "structural" in lines[0]     # structure still leads
    assert "strong" in lines[1]
    assert "weak" in lines[2]


def test_the_order_is_stable_across_repeated_identical_runs():
    ev = Evidence(
        facts=[_fact(f"change {i}", f"2026-0{i}-01", score=i / 10) for i in range(1, 8)],
        sources=[SourceDocument(doc_id=f"d{i}", title=f"memo {i}",
                                date=f"2026-0{i}-01") for i in range(1, 5)],
    )

    renders = {(tuple(_section(_digest(ev), "FACTS")),
                tuple(_section(_digest(ev), "SOURCES"))) for _ in range(10)}
    assert len(renders) == 1


# ── what must NOT change ────────────────────────────────────────────────────


def test_relevance_still_decides_what_survives_the_cap():
    """Ordering is presentation only. The OLDEST item is the most relevant here,
    so a date-based SELECTION would have dropped it — it must survive."""
    ev = Evidence(facts=[
        _fact("old but most relevant", "2020-01-01", structural=True, score=0.99),
        _fact("new and weak", "2026-01-01", score=0.01),
    ])

    lines = _section(_digest(ev, digest_max_facts=1), "FACTS")

    assert len(lines) == 1
    assert "old but most relevant" in lines[0]


def test_claims_keep_their_existing_recency_order():
    """CLAIMS was already correct and is not touched by this change."""
    ev = Evidence(claims=[
        ClaimFact(subject="X", predicate="p", object="old", valid_from="2024-01-01"),
        ClaimFact(subject="X", predicate="p", object="new", valid_from="2026-01-01"),
    ])

    lines = _section(_digest(ev), "CLAIMS")

    assert "new" in lines[0] and "old" in lines[1]


def test_the_fact_cap_is_still_enforced():
    ev = Evidence(facts=[_fact(f"c{i}", f"2026-01-{i:02d}") for i in range(1, 20)])

    assert len(_section(_digest(ev, digest_max_facts=5), "FACTS")) == 5


# ── the shared helper ───────────────────────────────────────────────────────


def test_the_helper_puts_undated_last_without_reordering_them():
    rows = [{"d": "2025-04-01", "n": "old"}, {"d": None, "n": "u1"},
            {"d": "2026-07-01", "n": "new"}, {"d": None, "n": "u2"}]

    assert [r["n"] for r in order_by_date_desc(rows, lambda r: r["d"])] == [
        "new", "old", "u1", "u2"]


def test_the_helper_treats_a_blank_date_as_undated():
    """Per-run graphs write "" where older ones omitted the key."""
    rows = [{"d": "", "n": "blank"}, {"d": "2026-01-01", "n": "dated"}]

    assert [r["n"] for r in order_by_date_desc(rows, lambda r: r["d"])] == [
        "dated", "blank"]


def test_the_helper_is_a_no_op_on_an_empty_list():
    assert order_by_date_desc([], lambda r: None) == []


# ── EXCERPTS: the section the reported answer's prose came from ─────────────


def _dated_excerpts(specs, **over):
    """(title, date, relevance) → the rendered EXCERPTS block."""
    sources = [SourceDocument(doc_id=f"d{i}", title=t, date=d, relevance=r)
               for i, (t, d, r) in enumerate(specs)]
    quotes = [ChunkQuote(chunk_id=f"c{i}", doc_id=f"d{i}", doc_title=t,
                         text=f"{t} body text. " * 8)
              for i, (t, _d, _r) in enumerate(specs)]
    return _section(_digest(Evidence(quotes=quotes, sources=sources), **over),
                    "EXCERPTS")


def test_excerpts_are_presented_newest_first():
    """The narrative prose in the reported answer came from EXCERPTS, which
    carry their document's date (_quote_date) but were ordered by SOURCE TRUST —
    so the model read the story oldest-first and wrote it out that way."""
    lines = _dated_excerpts([
        ("April 2025 memo", "2025-04-01", 0.9),    # most trusted, oldest
        ("Feb 2026 memo", "2026-02-01", 0.8),
        ("July 2026 memo", "2026-07-01", 0.2),     # least trusted, newest
    ])

    assert "July 2026 memo" in lines[0]
    assert "Feb 2026 memo" in lines[1]
    assert "April 2025 memo" in lines[2]


def test_undated_excerpts_follow_the_dated_ones():
    lines = _dated_excerpts([
        ("Undated handbook", None, 0.9),
        ("May 2026 memo", "2026-05-01", 0.1),
    ])

    assert "May 2026 memo" in lines[0]
    assert "Undated handbook" in lines[1]


def test_excerpt_selection_is_still_made_by_relevance_and_budget():
    """Ordering is presentation only: the trusted-but-old excerpt must still be
    the one KEPT when only one fits, exactly as before."""
    lines = _dated_excerpts(
        [("April 2025 memo", "2025-04-01", 0.9), ("July 2026 memo", "2026-07-01", 0.2)],
        digest_max_quotes=1,
    )

    assert len(lines) == 1
    assert "April 2025 memo" in lines[0]


def test_entirely_undated_excerpts_keep_their_existing_order():
    lines = _dated_excerpts([
        ("First doc", None, 0.9), ("Second doc", None, 0.5),
    ])

    assert "First doc" in lines[0]
    assert "Second doc" in lines[1]


def test_the_excerpt_order_is_stable_across_repeated_runs():
    specs = [(f"memo {i}", f"2026-0{i}-01", i / 10) for i in range(1, 6)]

    assert len({tuple(_dated_excerpts(specs)) for _ in range(10)}) == 1


# ── prose dates are the MODEL's to read, never code's ───────────────────────
#
# An excerpt's only structured date is its DOCUMENT's. A date stated inside the
# passage text ("in early February", "October 19, 2026") is deliberately NOT
# parsed in code — the `latest_date_in_text` month/date regex was removed — so
# passages of one document share their document date and keep their existing
# order among themselves, and the synthesis prompt's own rule ("a date stated
# only in prose counts exactly like a dated field") is what orders the prose
# case. These pin the removal and the boundary it draws.


def test_the_text_date_extractor_is_gone():
    """The removal itself, pinned: no month-name/date regex machinery remains in
    the quality module, and the composer no longer imports it."""
    from ai_graph_retrieval.retrieval import quality
    from ai_graph_retrieval.synthesis import composer

    for symbol in ("latest_date_in_text", "_MONTHS", "_MONTH_PREFIX",
                   "_MONTH_ALT", "_ISO_DATE", "_MONTH_YEAR"):
        assert not hasattr(quality, symbol), symbol
        assert not hasattr(composer, symbol), symbol


def test_same_document_passages_share_their_documents_date_and_keep_their_order():
    """The documented boundary of doc-date ordering: one rolling document, every
    passage keyed on the same document date, so the stable sort preserves their
    existing order — no prose date is read in code to separate them. Ordering
    the prose case is the synthesis prompt's job (asserted below)."""
    src = [SourceDocument(doc_id="d1", title="Org changes", date="2026-08-01")]
    quotes = [ChunkQuote(chunk_id=f"c{i}", doc_id="d1", doc_title="Org changes",
                         text=f"{label} announcement. " * 8)
              for i, label in enumerate(["April 2025", "February 2026", "July 2026"])]

    lines = _section(_digest(Evidence(quotes=quotes, sources=src)), "EXCERPTS")

    assert "April 2025" in lines[0]
    assert "February 2026" in lines[1]
    assert "July 2026" in lines[2]
    assert all("2026-08-01" in line for line in lines), \
        "each passage still renders its document's date for the model to weigh"


def test_excerpts_across_documents_order_by_their_documents_dates():
    """Cross-document ordering is untouched by the removal: the document date is
    a structured field, and newer documents' excerpts still come first."""
    src = [SourceDocument(doc_id="d1", title="Old doc", date="2024-01-01"),
           SourceDocument(doc_id="d2", title="New doc", date="2026-09-01")]
    quotes = [
        ChunkQuote(chunk_id="c1", doc_id="d1", doc_title="Old doc",
                   text="A passage naming no date at all. " * 8),
        ChunkQuote(chunk_id="c2", doc_id="d2", doc_title="New doc",
                   text="Another passage naming no date. " * 8),
    ]

    lines = _section(_digest(Evidence(quotes=quotes, sources=src)), "EXCERPTS")

    assert "New doc" in lines[0]
    assert "Old doc" in lines[1]


def test_a_prose_date_no_longer_moves_an_excerpt():
    """The inverse of the removed behavior, stated as the new contract: a recent
    document recapping an old event keys on ITS OWN date, not on the January
    2020 its text mentions — no date is parsed out of prose in code."""
    src = [SourceDocument(doc_id="d1", title="Recap", date="2026-09-01"),
           SourceDocument(doc_id="d2", title="Notice", date="2026-08-01")]
    quotes = [
        ChunkQuote(chunk_id="c1", doc_id="d1", doc_title="Recap",
                   text="Looking back at the January 2020 launch. " * 8),
        ChunkQuote(chunk_id="c2", doc_id="d2", doc_title="Notice",
                   text="The August 2026 rollout begins now. " * 8),
    ]

    lines = _section(_digest(Evidence(quotes=quotes, sources=src)), "EXCERPTS")

    assert "Recap" in lines[0]      # 2026-09-01, its document's own date
    assert "Notice" in lines[1]     # 2026-08-01


def test_nothing_is_dropped_by_the_removal():
    """Claims, facts, sources and excerpts ALL still reach the digest, dated
    sections newest first by their structured dates — removing the text-date
    extraction removed an ordering key, never content."""
    ev = Evidence(
        claims=[ClaimFact(subject="CPO", predicate="role", object="advisory",
                          valid_from="2026-07-31"),
                ClaimFact(subject="CRO", predicate="role", object="eliminated",
                          valid_from="2026-02-01")],
        facts=[_fact("June reorg", "2026-06-01"), _fact("April cycle", "2025-04-01")],
        sources=[SourceDocument(doc_id="d1", title="Digest", date="2026-08-01")],
        quotes=[ChunkQuote(chunk_id="c1", doc_id="d1", doc_title="Digest",
                           text="In June 2026 product strategy was reorganized. " * 6)],
    )

    digest = _digest(ev)

    claims = _section(digest, "CLAIMS")
    assert "CPO" in claims[0] and "CRO" in claims[1]      # newest → oldest
    facts = _section(digest, "FACTS")
    assert "June reorg" in facts[0] and "April cycle" in facts[1]
    assert "Digest" in _section(digest, "SOURCES")[0]
    assert "product strategy" in _section(digest, "EXCERPTS")[0]


def test_same_document_ordering_is_stable_across_repeated_runs():
    src = [SourceDocument(doc_id="d1", title="Org changes", date="2026-08-01")]
    quotes = [ChunkQuote(chunk_id=f"c{i}", doc_id="d1", doc_title="Org changes",
                         text=f"Announced March {i}, 2026 for the team. " * 6)
              for i in range(1, 8)]
    ev = Evidence(quotes=quotes, sources=src)

    assert len({tuple(_section(_digest(ev), "EXCERPTS")) for _ in range(10)}) == 1


def test_the_prompt_orders_by_each_items_own_date_not_by_evidence_order():
    """The defect: the rule used to end "the CLAIMS section is already given to
    you newest first; keep that order", which told the model to FOLLOW the order
    it received — and for excerpts from one document that order is oldest-first."""
    from ai_graph_retrieval.synthesis.prompts import SYNTHESIS_SYSTEM_PROMPT as P

    flat = " ".join(P.split())
    assert "ORDER BY EACH ITEM'S OWN DATE, NEVER BY THE ORDER THE EVIDENCE LISTED IT" in flat
    # The instruction that contradicted it must be gone.
    assert "keep that order" not in flat


def test_the_prompt_says_a_prose_date_counts_like_a_dated_field():
    """Where the only date signal is prose, the model is the only layer that can
    read it — so the rule has to say so explicitly."""
    from ai_graph_retrieval.synthesis.prompts import SYNTHESIS_SYSTEM_PROMPT as P

    flat = " ".join(P.split())
    assert "counts exactly like a dated field" in flat
    assert "an answer that follows the passage order it was given will come out backwards" in flat


def test_the_timeline_shape_still_asks_for_newest_first_sections():
    """The screenshot's answer was headed sections per period; that shape's own
    addendum must agree with the ordering rule."""
    from ai_graph_retrieval.retrieval.base import AnswerShape
    from ai_graph_retrieval.synthesis.prompts import SYNTHESIS_SHAPE_ADDENDA

    assert "newest first" in SYNTHESIS_SHAPE_ADDENDA[AnswerShape.TIMELINE]
