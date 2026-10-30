"""Near-duplicate documents share one excerpt-budget group.

Exact and containment dedup (``dedupe_quote_texts``) compare whole chunk texts,
so two copies of a page that differ ANYWHERE inside — a changed footer, an emoji
in the body — are neither equal nor contained, and both survived as separate
documents: separate round-robin slots, separate excerpt budget. Measured, two
arXiv pages of the same content under titles differing only by an emoji
took 23.8% of one query's budget.

The allocator now groups such documents together. It never drops evidence and
never splits a document's own chunks apart.

    uv run pytest tests/test_near_duplicate_documents.py -v
"""
from __future__ import annotations

from ai_graph_retrieval.core.models import ChunkQuote, Evidence, SourceDocument
from ai_graph_retrieval.synthesis.composer import (
    AnswerComposer,
    _near_duplicate_aliases,
)
from ai_graph_retrieval.synthesis.dedup import content_key
from tests.support import make_settings

# A shared body long enough to fingerprint (> digest_near_duplicate_prefix_chars).
SHARED = ("Release Readiness for 26.07. This page covers the hotpatch window, the "
          "readiness checklist, the sign-off owners and the rollback plan. ") * 5
DISTINCT = ("Leave entitlement policy. Employees accrue twenty days of paid leave "
            "per calendar year, accruing monthly from the joining date. ") * 5


def _quote(chunk_id: str, doc: str, text: str, graph: str = "arxiv") -> ChunkQuote:
    return ChunkQuote(chunk_id=chunk_id, doc_id=doc, doc_title=f"Title {doc}",
                      text=text, graph=graph)


def _rendered_docs(quotes, sources, **overrides) -> list[str]:
    s = make_settings(**overrides)
    _digest, _counts, rendered = AnswerComposer(s)._build_digest(
        Evidence(sources=sources, quotes=quotes), strict=False,
        quote_chars=overrides.pop("quote_chars", 1400),
    )
    return [r["doc"] for r in rendered]


def _sources(*docs: str) -> list[SourceDocument]:
    return [SourceDocument(doc_id=d, title=f"Title {d}") for d in docs]


# ── the fix ─────────────────────────────────────────────────────────────────


def test_a_near_duplicate_does_not_take_a_distinct_documents_budget():
    """The pair shares one slot, so the genuinely different document renders."""
    quotes = [
        _quote("q1", "cf-1", SHARED + "Footer: published by the platform team."),
        _quote("q2", "cf-2", SHARED + "Footer: republished 🚀 by enablement, extra note."),
        _quote("q3", "bk-1", DISTINCT, graph="wikipedia"),
    ]
    srcs = _sources("cf-1", "cf-2", "bk-1")

    assert _rendered_docs(quotes, srcs, digest_near_duplicate_prefix_chars=0) == [
        "Title cf-1", "Title cf-2",
    ]
    assert _rendered_docs(quotes, srcs) == ["Title cf-1", "Title bk-1"]


def test_the_alias_map_points_the_copy_at_the_first_document_seen():
    quotes = [_quote("q1", "cf-1", SHARED + "one"), _quote("q2", "cf-2", SHARED + "two")]

    assert _near_duplicate_aliases(quotes, 400) == {"cf-2": "cf-1"}


def test_grouping_is_deterministic_regardless_of_arrival_order():
    a, b = _quote("q1", "cf-1", SHARED + "one"), _quote("q2", "cf-2", SHARED + "two")

    assert _near_duplicate_aliases([a, b], 400) == {"cf-2": "cf-1"}
    assert _near_duplicate_aliases([b, a], 400) == {"cf-1": "cf-2"}


def test_a_document_is_fingerprinted_by_its_longest_chunk():
    """A short shared boilerplate chunk must not decide the fingerprint."""
    quotes = [
        _quote("q1", "cf-1", "Shared header. "),
        _quote("q2", "cf-1", SHARED + "the real body of document one"),
        _quote("q3", "cf-2", "Shared header. "),
        _quote("q4", "cf-2", SHARED + "the real body of document two"),
    ]

    assert _near_duplicate_aliases(quotes, 400) == {"cf-2": "cf-1"}


# ── genuinely different documents are preserved ─────────────────────────────


def test_distinct_documents_are_never_grouped():
    quotes = [_quote("q1", "d1", SHARED), _quote("q2", "d2", DISTINCT)]

    assert _near_duplicate_aliases(quotes, 400) == {}
    assert _rendered_docs(quotes, _sources("d1", "d2")) == ["Title d1", "Title d2"]


def test_documents_that_diverge_inside_the_prefix_stay_separate():
    """Two pages on one topic share wording, not an opening — the prefix is well
    past where real documents differ."""
    quotes = [
        _quote("q1", "d1", "Payroll runs on the 25th. " + SHARED),
        _quote("q2", "d2", "Payroll runs on the 28th. " + SHARED),
    ]

    assert _near_duplicate_aliases(quotes, 400) == {}


def test_a_shared_boilerplate_opening_alone_never_groups():
    """The prefix is only the CANDIDATE key: two templated documents that open
    with the same 400+ characters of form boilerplate but carry different
    substantive content must keep separate slots and budgets."""
    boiler = ("Change request form. Complete every section before submission. "
              "Approvals are required from the service owner and the security "
              "reviewer prior to any production deployment window. ") * 3
    quotes = [
        _quote("q1", "d1", boiler + ("We will migrate the billing database to the "
                                     "new cluster during the June window. ") * 6),
        _quote("q2", "d2", boiler + ("Rotate the signing keys for the partner API "
                                     "and update the JWKS endpoint by May. ") * 6),
    ]

    assert _near_duplicate_aliases(quotes, 400) == {}
    assert _rendered_docs(quotes, _sources("d1", "d2")) == ["Title d1", "Title d2"]


def test_a_true_copy_with_a_differing_footer_still_groups():
    """The verification tolerates a footer-sized differing tail — a republished
    copy whose body agrees end to end is one document for allocation."""
    body = SHARED * 2  # well past the prefix, so the check covers real body
    quotes = [
        _quote("q1", "cf-1", body + "Published by the platform team, June 2026."),
        _quote("q2", "cf-2", body + "Republished 🚀 by enablement with a new owner."),
    ]

    assert _near_duplicate_aliases(quotes, 400) == {"cf-2": "cf-1"}


def test_short_documents_are_never_collapsed_on_a_partial_key():
    """Below the prefix length there is not enough text to judge — a people
    record is often just a name and a role."""
    quotes = [_quote("q1", "p1", "Eva Stone, Engineer"),
              _quote("q2", "p2", "Eva Stone, Engineer")]

    assert _near_duplicate_aliases(quotes, 400) == {}


def test_no_evidence_is_dropped_only_regrouped():
    quotes = [_quote("q1", "cf-1", SHARED + "one"), _quote("q2", "cf-2", SHARED + "two")]
    _digest, counts, _rendered = AnswerComposer(make_settings())._build_digest(
        Evidence(sources=_sources("cf-1", "cf-2"), quotes=quotes),
        strict=False, quote_chars=100_000,
    )

    assert counts["quotes_pool"] == 2
    assert counts["quotes_rendered"] == 2


# ── unchanged behaviour ─────────────────────────────────────────────────────


def test_a_documents_own_chunks_are_never_split_apart():
    """The group stays the DOCUMENT: grouping maps documents together, it must
    never turn one document's chunks into separate groups."""
    quotes = [_quote(f"q{i}", "d1", f"chunk {i}. " + SHARED) for i in range(3)]

    assert _near_duplicate_aliases(quotes, 400) == {}


def test_zero_disables_the_grouping():
    quotes = [_quote("q1", "cf-1", SHARED + "one"), _quote("q2", "cf-2", SHARED + "two")]

    assert _near_duplicate_aliases(quotes, 0) == {}


def test_single_document_behaviour_is_unchanged():
    quotes = [_quote(f"q{i}", "d1", f"chunk {i}. " + SHARED) for i in range(3)]

    assert _rendered_docs(quotes, _sources("d1"), quote_chars=100_000) == [
        "Title d1", "Title d1", "Title d1",
    ]


def test_content_key_normalizes_whitespace_and_case_only():
    body = "Alpha Beta   Gamma\nDelta"

    assert content_key(body, prefix_chars=100) == "alpha beta gamma delta"
    assert content_key("", prefix_chars=100) == ""

