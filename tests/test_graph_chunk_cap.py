"""Super Research's graph-chunk allowance.

The agent's tools may add at most ``graph_chunk_cap`` graph chunks TOTAL across
both graphs (not per graph, not per tool call), and never a chunk whose text
the pool already holds. The cap bounds VOLUME, not standing: once kept, a graph
chunk is evidence of equal weight with the agent's graph findings — synthesis
weighs them together.

    uv run pytest tests/test_graph_chunk_cap.py -v
"""
from __future__ import annotations

from ai_graph_retrieval.agent.tools import ResearchContext, _keep_graph_quotes
from ai_graph_retrieval.core.models import ChunkQuote, EvidencePool
from tests.support import make_settings


def _rc(pool: EvidencePool, cap: int) -> ResearchContext:
    return ResearchContext(
        repo=None, ctx=None, embedder=None, settings=make_settings(),
        evidence=pool, progress=None, usage=None,
        graph_text=True, graph_chunk_cap=cap,
    )


def _q(cid: str, text: str, graph: str = "wikipedia") -> ChunkQuote:
    return ChunkQuote(chunk_id=cid, text=text, graph=graph)


def test_the_cap_is_total_across_both_graphs_not_per_graph():
    pool = EvidencePool()
    rc = _rc(pool, cap=3)
    first = _keep_graph_quotes(rc, [_q(f"b{i}", f"wikipedia chunk {i}") for i in range(2)])
    pool.add_quotes(first)
    second = _keep_graph_quotes(
        rc, [_q(f"c{i}", f"arxiv chunk {i}", graph="arxiv") for i in range(4)]
    )
    assert len(first) == 2
    assert len(second) == 1, "2 already used of 3 — ONE slot remains across graphs"


def test_a_chunk_the_api_already_returned_is_deduplicated():
    pool = EvidencePool()
    pool.add_quotes([_q("api1", "The  Shared\nChunk.", graph="other")])
    kept = _keep_graph_quotes(
        _rc(pool, cap=10),
        [_q("g1", "the shared chunk."), _q("g2", "a genuinely new chunk")],
    )
    # Same text under a different chunk id (whitespace/case differences) drops.
    assert [q.chunk_id for q in kept] == ["g2"]


def test_duplicates_within_one_batch_are_also_dropped():
    kept = _keep_graph_quotes(
        _rc(EvidencePool(), cap=10),
        [_q("g1", "same words"), _q("g2", "Same   WORDS"), _q("g3", "different")],
    )
    assert [q.chunk_id for q in kept] == ["g1", "g3"]


def test_a_zero_cap_keeps_nothing():
    kept = _keep_graph_quotes(_rc(EvidencePool(), cap=0), [_q("g1", "text")])
    assert kept == []
