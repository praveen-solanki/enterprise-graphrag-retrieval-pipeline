"""Recent changes keep the DOCUMENTED leg's date.

``cypher.RECENT_CHANGES`` is a UNION: the event leg dates a claim by
``cl.valid_from``, the documented leg by ``cl.documented_on``, and both come back
in one ``eff_date`` column the query has already ordered ``DESC``.

The mapping in ``recent_changes`` used to copy that column only when
``date_basis == "event"`` and drop it otherwise, so every documented-basis change
arrived undated. ``rank_claims_by_recency`` sorts undated rows last, so those
changes sank to the tail of a lane whose whole promise is newest-first — a 2026
documented change ranked below a 2024 event one.

    uv run pytest tests/test_recent_changes_documented_dates.py -v
"""
from __future__ import annotations

import asyncio

from ai_graph_retrieval.core.access import AccessContext
from ai_graph_retrieval.graph.repository import GraphReadRepository
from ai_graph_retrieval.retrieval.quality import rank_claims_by_recency
from tests.support import make_settings

# One row per UNION leg, newest first — the order the query's ORDER BY hands over.
_ROWS = [
    {"source": "Atlas", "relationship": "status_changed", "target": "GA",
     "date": "2026-05-01", "date_basis": "documented", "support": 2, "graph": "wikipedia"},
    {"source": "Atlas", "relationship": "released", "target": "v3",
     "date": "2026-02-01", "date_basis": "event", "support": 3, "graph": "wikipedia"},
]


def _repo(rows):
    repo = GraphReadRepository(None, make_settings(), http=object())

    async def _stub_rows(_query, _ctx, **params):
        return rows

    repo._rows = _stub_rows
    return repo


def _ctx():
    return AccessContext(tenant_id="t", principals=("public",), request_id="r")


def _changes(rows):
    return asyncio.run(_repo(rows).recent_changes(_ctx(), "2026-01-01"))


def test_the_documented_leg_keeps_its_date():
    """The defect: this row's date was discarded because it was not 'event'."""
    documented = _changes(_ROWS)[0]

    assert documented.valid_from == "2026-05-01"
    # The basis still labels it, so synthesis can keep saying "documented as of"
    # rather than presenting it as the date the thing happened.
    assert documented.date_basis == "documented"


def test_both_bases_stay_dated_and_rank_newest_first():
    ranked = rank_claims_by_recency(_changes(_ROWS))

    assert [c.valid_from for c in ranked] == ["2026-05-01", "2026-02-01"]
    assert [c.date_basis for c in ranked] == ["documented", "event"]


def test_a_documented_change_outranks_an_older_event_change():
    """Ordering is by DATE, not by basis — the bug made basis decide the order.

    Fed in the losing order, the newer documented row still has to come first;
    undated, it sorted last however recent it was.
    """
    ranked = rank_claims_by_recency(_changes([
        {"source": "Atlas", "relationship": "released", "target": "v1",
         "date": "2024-03-01", "date_basis": "event", "support": 1, "graph": "wikipedia"},
        {"source": "Atlas", "relationship": "status_changed", "target": "GA",
         "date": "2026-05-01", "date_basis": "documented", "support": 1, "graph": "wikipedia"},
    ]))

    assert [c.object for c in ranked] == ["GA", "v1"]


def test_a_genuinely_undated_row_stays_undated():
    """The dense-empty-date guard: per-run graphs write "" where older ones
    omitted the key (see cypher.FETCH_CLAIMS), and "" must not become a date."""
    changes = _changes([
        {"source": "Atlas", "relationship": "mentions", "target": "x",
         "date": "", "date_basis": None, "support": 1, "graph": "wikipedia"},
    ])

    assert changes[0].valid_from is None
