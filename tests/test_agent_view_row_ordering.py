"""The row-returning tools rank across graphs before bounding the agent's view.

``traverse_to_chunks`` and ``recent_change_excerpts`` hand the model raw rows.
Those rows arrive as the repository's fan-out concatenation — every row of the
first graph, then the second, each block already ordered by its own query — so
slicing them positionally let the first graph fill the window and hid the second
graph's stronger rows. What the agent cannot see it cannot follow up on.

The EvidencePool is unaffected: it keeps everything ``_keep_graph_quotes``
returns, and that budget's own graph fairness is unchanged.

    uv run pytest tests/test_agent_view_row_ordering.py -v
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from ai_graph_retrieval.agent.tools import _top_rows
from ai_graph_retrieval.core.models import EvidencePool
from tests.support import make_settings

WIKIPEDIA, ARXIV = "wikipedia", "arxiv"


def _row(chunk_id: str, graph: str, *, via_count: int = 1,
         changed_on: str = "2020-01-01") -> dict:
    return {"chunk_id": chunk_id, "doc_id": f"d-{chunk_id}",
            "doc_title": f"Title {chunk_id}", "text": f"body of {chunk_id}",
            "via_names": [], "via_count": via_count, "changed_on": changed_on,
            "graph": graph}


def _fan_out(wikipedia: int, *, via_count: int = 99,
             changed_on: str = "2026-08-01") -> list[dict]:
    """Wikipedia's block first, then one strictly stronger arXiv row."""
    return [_row(f"bk{i}", WIKIPEDIA) for i in range(wikipedia)] + [
        _row("cf-best", ARXIV, via_count=via_count, changed_on=changed_on)
    ]


# ── the fix ─────────────────────────────────────────────────────────────────


def test_a_stronger_second_graph_row_is_not_hidden_by_the_first_graphs_block():
    rows = _fan_out(12)

    for field in ("via_count", "changed_on"):
        shown = _top_rows(rows, 12, by=field)
        assert shown[0]["chunk_id"] == "cf-best", field
        assert any(r["graph"] == ARXIV for r in shown), field


def test_the_positional_slice_would_have_hidden_it():
    """Pins the defect the ranking fixes, so a regression is visible."""
    rows = _fan_out(12)

    assert not any(r["graph"] == ARXIV for r in rows[:12])


def test_ranking_is_by_the_field_the_query_already_ordered_on():
    rows = [_row("a", WIKIPEDIA, via_count=1), _row("b", ARXIV, via_count=5),
            _row("c", WIKIPEDIA, via_count=3)]

    assert [r["chunk_id"] for r in _top_rows(rows, 3, by="via_count")] == ["b", "c", "a"]


def test_missing_values_sort_last_without_raising():
    rows = [{"chunk_id": "a"}, _row("b", ARXIV, via_count=4)]

    assert [r["chunk_id"] for r in _top_rows(rows, 2, by="via_count")] == ["b", "a"]


def test_ordering_is_deterministic_across_identical_requests():
    rows = [_row("alpha", WIKIPEDIA, via_count=5), _row("zeta", ARXIV, via_count=5)]

    assert _top_rows(rows, 2, by="via_count") == _top_rows(
        list(reversed(rows)), 2, by="via_count"
    )


def test_the_row_budget_still_bounds_the_view():
    assert len(_top_rows(_fan_out(30), 12, by="via_count")) == 12


def test_single_graph_ordering_is_unchanged():
    """One graph arrives already ordered; ranking must be a no-op on it."""
    rows = [_row("a", WIKIPEDIA, via_count=9), _row("b", WIKIPEDIA, via_count=5),
            _row("c", WIKIPEDIA, via_count=1)]

    assert _top_rows(rows, 3, by="via_count") == rows


# ── the pool and the graph-chunk budget are untouched ───────────────────────


class _Step:
    detail: str | None = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _Repo:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    async def beam_traverse_chunks(self, ctx, seeds, hops=None):
        return list(self._rows)

    async def recent_change_chunks(self, ctx, since, topic=None):
        return list(self._rows)


def _rc(rows: list[dict], pool: EvidencePool) -> SimpleNamespace:
    async def graph_delta(**kwargs):
        return None

    return SimpleNamespace(
        repo=_Repo(rows), ctx=SimpleNamespace(log_fields=lambda: {}),
        settings=make_settings(), evidence=pool, usage=SimpleNamespace(),
        deep=False, graph_text=True, graph_chunk_cap=40, time_scope=("", ""),
        progress=SimpleNamespace(step=lambda *a, **k: _Step(),
                                 graph_delta=graph_delta, span_parent=None,
                                 _tool_seq=0),
    )


def _invoke(tool, rc, payload: dict) -> dict:
    return json.loads(asyncio.run(tool.on_invoke_tool(
        SimpleNamespace(context=rc, tool_name=tool.name, tool_call_id="t1",
                        run_config=None),
        json.dumps(payload),
    )))


def test_the_pool_keeps_every_row_the_view_bounds():
    from ai_graph_retrieval.agent.tools import traverse_to_chunks

    rows, pool = _fan_out(12), EvidencePool()
    out = _invoke(traverse_to_chunks, _rc(rows, pool), {"seed_entities": ["Seed"]})

    assert len(out["excerpts"]) == 12, "the agent's view is still bounded"
    assert len(pool.snapshot().quotes) == 13, "the pool keeps everything"
    assert any(q.graph == ARXIV for q in pool.snapshot().quotes)


def test_both_tools_surface_the_stronger_second_graph_row_to_the_agent():
    from ai_graph_retrieval.agent.tools import (
        recent_change_excerpts,
        traverse_to_chunks,
    )

    for tool, payload in ((traverse_to_chunks, {"seed_entities": ["Seed"]}),
                          (recent_change_excerpts, {})):
        out = _invoke(tool, _rc(_fan_out(12), EvidencePool()), payload)
        titles = [e["doc_title"] for e in out["excerpts"]]
        assert titles[0] == "Title cf-best", tool.name
