"""The title lookup takes the whole set, and budgets each title separately.

Two measured problems, one shape:

  * ONE TITLE PER CALL made covering a series cost a round per member. The agent
    prompt already tells the model to reach for the tool that takes a LIST —
    "repeating a single-subject tool once per member ... burns the round budget" —
    and this read gave it nothing to reach for. On "what all are the release items
    for this year" a calendar page named twelve releases, the loop looked up two of
    them by name, and the answer carried feature detail for six of twelve.
  * ONE GLOBAL LIMIT let a crowded title spend the rows the others needed. It
    already did so within a single title: "26.07" matches five "Hotpatch Release
    Readiness" pages on the dev arXiv graph, which pushes
    "26.07 - All you need to know" out of a five-row result entirely.

    uv run pytest tests/test_document_lookup.py -v
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from ai_graph_retrieval.graph import cypher
from ai_graph_retrieval.graph.repository import GraphReadRepository
from tests.support import make_settings

_CTX = SimpleNamespace(log_fields=lambda: {})


def _stub(rows: list[dict] | None = None, **overrides):
    """A repository stub that records what the read was asked for."""
    seen: dict = {}

    async def _rows(query, ctx, **params):
        seen["query"] = query
        seen.update(params)
        return rows or []

    return SimpleNamespace(_rows=_rows, _s=make_settings(**overrides)), seen


def _lookup(stub, titles, **kwargs):
    return asyncio.run(GraphReadRepository.document_lookup(stub, _CTX, titles, **kwargs))


# ── the read ────────────────────────────────────────────────────────────────


def test_every_title_goes_in_one_read_with_its_own_row_budget():
    stub, seen = _stub(documents_per_title=3)
    _lookup(stub, ["26.01", "26.02", "26.03"])

    assert seen["query"] is cypher.DOCUMENT_LOOKUP
    assert seen["titles"] == ["26.01", "26.02", "26.03"]
    assert seen["per_title"] == 3
    # Sized so it CANNOT bind and re-create the starvation the per-title slice
    # exists to prevent — the same treatment supporting_chunks gives its anchors.
    assert seen["limit"] == 3 * 3


def test_titles_are_deduped_and_blanks_dropped_in_order():
    stub, seen = _stub()
    _lookup(stub, [" 26.01 ", "", "26.02", None, "26.01", "   "])
    assert seen["titles"] == ["26.01", "26.02"]


def test_no_usable_title_means_no_read_at_all():
    stub, seen = _stub()
    assert _lookup(stub, ["", "  ", None]) == []
    assert _lookup(stub, []) == []
    assert seen == {}


def test_an_explicit_per_title_budget_wins_over_the_setting():
    stub, seen = _stub(documents_per_title=3)
    _lookup(stub, ["26.01"], per_title=10)
    assert (seen["per_title"], seen["limit"]) == (10, 10)


def test_rows_become_source_documents():
    stub, _ = _stub([
        {"doc_id": "d1", "title": "26.01 - All you need to know",
         "url": "http://x/1", "date": "2026-01-12", "source_type": "wiki"},
    ])
    docs = _lookup(stub, ["26.01"])
    assert [(d.doc_id, d.title) for d in docs] == [
        ("d1", "26.01 - All you need to know")]


# ── the query shape ─────────────────────────────────────────────────────────


def test_the_slice_is_per_title_so_one_title_cannot_starve_another():
    q = cypher.DOCUMENT_LOOKUP
    assert "UNWIND $titles AS wanted" in q
    assert "collect(DISTINCT d)[0..$per_title]" in q
    # the same title text matched by two graphs' copies is one row, not two
    assert "WITH DISTINCT d" in q


def test_the_lookup_stays_acl_gated_on_the_document():
    prepared = cypher.prepare(cypher.DOCUMENT_LOOKUP, "allow_access_control")
    assert (
        "($principals IS NULL OR any(p IN d.allow_access_control WHERE p IN $principals))"
        in prepared
    )


def test_ordering_within_a_title_uses_no_hand_rolled_relevance():
    """Date is the only signal available: there is no fulltext index on
    Document.title on either graph (checked 2026-08-27), so the engine has no
    match score to offer. Ranking by a property of the STRING — its length, how
    much of it the query covers — would be exactly the hand-rolled relevance
    rule 8 forbids, and it is what a future reader would be tempted to add here.
    """
    q = cypher.DOCUMENT_LOOKUP
    assert "ORDER BY d.published_at DESC" in q
    for banned in ("size(d.title)", "length(d.title)", "apoc.text",
                   "STARTS WITH toLower($", "levenshtein"):
        assert banned not in q
