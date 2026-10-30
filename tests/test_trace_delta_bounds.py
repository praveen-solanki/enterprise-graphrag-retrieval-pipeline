"""trace_delta events are size-bounded — one merge, as many events as it takes.

The ws relay silently drops any frame past ~100-110 KB, and graph_delta used to
emit ONE TraceDeltaEvent per pool merge however large it was: a tool result
carrying hundreds of claims (each a node plus ASSERTS/SUPPORTED_BY edges with
the claim text embedded in every id) produced a single 100 KB+ event, dropped
right after its stage_done — 44 times in one 70-minute dev log window. These
pin the packing.

    uv run pytest tests/test_trace_delta_bounds.py -v
"""
from __future__ import annotations

import asyncio

from ai_graph_retrieval.core.events import TraceDeltaEvent
from ai_graph_retrieval.core.models import ClaimFact, GraphTraceBuilder
from ai_graph_retrieval.core.progress import ProgressReporter, _delta_chunks


def _claims(n: int, chars: int = 400) -> list[ClaimFact]:
    """Claims shaped like find_recent_changes output: long free-text objects,
    which the trace embeds into the claim node's id AND both its edge ids."""
    return [
        ClaimFact(subject=f"Subject {i}", predicate="announced",
                  object=f"announcement {i}: " + "x" * chars,
                  valid_from="2026-08-01")
        for i in range(n)
    ]


def _sizes(events: list[TraceDeltaEvent]) -> list[int]:
    return [len(e.model_dump_json().encode()) for e in events]


def test_a_huge_merge_is_emitted_as_several_bounded_events():
    trace = GraphTraceBuilder()
    emitted: list[TraceDeltaEvent] = []

    async def emit(event):
        emitted.append(event)

    reporter = ProgressReporter(emit, trace, trace_delta_max_bytes=49_152)
    asyncio.run(reporter.graph_delta(claims=_claims(300)))

    deltas = [e for e in emitted if isinstance(e, TraceDeltaEvent)]
    assert len(deltas) > 1, "300 long claims cannot be one event"
    for size in _sizes(deltas):
        assert size < 49_152, f"one delta serialized to {size} B — the relay drops it"


def test_the_slices_carry_everything_the_single_event_used_to():
    """Packing must lose nothing: the union of the slices is exactly the merge."""
    trace = GraphTraceBuilder()
    emitted: list[TraceDeltaEvent] = []

    async def emit(event):
        emitted.append(event)

    reporter = ProgressReporter(emit, trace, trace_delta_max_bytes=49_152)
    claims = _claims(300)
    asyncio.run(reporter.graph_delta(claims=claims))

    unsplit = GraphTraceBuilder()
    want_nodes, want_edges = unsplit.merge(claims=claims)
    got_nodes = [n for e in emitted for n in e.nodes]
    got_edges = [ed for e in emitted for ed in e.edges]
    assert got_nodes == want_nodes
    assert got_edges == want_edges


def test_an_ordinary_merge_stays_one_event():
    trace = GraphTraceBuilder()
    emitted: list[TraceDeltaEvent] = []

    async def emit(event):
        emitted.append(event)

    reporter = ProgressReporter(emit, trace, trace_delta_max_bytes=49_152)
    asyncio.run(reporter.graph_delta(claims=_claims(5, chars=60)))
    assert len(emitted) == 1


def test_a_single_oversized_item_still_ships_alone():
    """An item bigger than the budget cannot be split further — it gets an event
    to itself (and the sender's oversize log names it) rather than vanishing."""
    nodes, edges = GraphTraceBuilder().merge(claims=_claims(1, chars=80_000))
    chunks = _delta_chunks(nodes, edges, 49_152)
    assert [n for ns, _ in chunks for n in ns] == nodes, "no node may vanish"
    assert [e for _, es in chunks for e in es] == edges, "no edge may vanish"
    assert all(ns or es for ns, es in chunks), "no empty events"
