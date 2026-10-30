"""Graph chunks are ranked by question similarity — from the right collection.

The digest gives graph chunks their own tier (they were 36-39 of the deduped pool
and rendered 0-2), and a tier needs an order. These tests pin the two properties
that make the score trustworthy:

  * every source is scored against ITS OWN chunk collection, with the question
    embedded at THAT collection's dimension — never one query vector reused across
    collections;
  * when the sources disagree on dimension the pass stamps NOTHING, because the
    digest merges both graphs into one ranked tier and cosines from different
    embedding spaces are not comparable.

    uv run pytest tests/test_chunk_relevance.py -v
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from ai_graph_retrieval.core.models import ChunkQuote, EvidencePool
from ai_graph_retrieval.retrieval.chunk_relevance import (
    score_graph_chunks,
)
from ai_graph_retrieval.retrieval.similarity import cosine


class _Store:
    """Records what was asked of which collection."""

    def __init__(self, dims: dict[str, int], vectors: dict[str, dict[str, list[float]]],
                 fail: bool = False) -> None:
        self._dims = dims
        self._vectors = vectors
        self._fail = fail
        self.asked: list[tuple[str, tuple[str, ...]]] = []

    async def vector_size(self, collection: str) -> int | None:
        return self._dims.get(collection)

    async def vectors_by_chunk_id(self, collection, chunk_ids, *, batch: int = 256):
        self.asked.append((collection, tuple(chunk_ids)))
        if self._fail:
            raise RuntimeError("qdrant down")
        return {c: v for c, v in self._vectors.get(collection, {}).items()
                if c in set(chunk_ids)}


class _Embedder:
    """Records the dimension each embed was requested at."""

    def __init__(self, vec: list[float] | None = None) -> None:
        self._vec = vec if vec is not None else [1.0, 0.0]
        self.dims: list[int | None] = []

    async def get(self, text: str, *, dimensions: int | None = None):
        self.dims.append(dimensions)
        return self._vec


def _sc(quotes, sources, store, embedder, question="who owns search?"):
    pool = EvidencePool()
    pool.add_quotes(quotes)
    return SimpleNamespace(
        question=question, evidence=pool, embedder=embedder,
        repo=SimpleNamespace(sources=sources, vector_store=store),
    )


def _src(name, collection):
    return SimpleNamespace(name=name, chunks_collection=collection)


def _q(chunk_id, graph, doc="d1"):
    return ChunkQuote(chunk_id=chunk_id, doc_id=doc, doc_title="Doc",
                      text=f"text {chunk_id}", graph=graph)


def test_the_shared_cosine_is_clamped_and_treats_a_missing_vector_as_zero():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine(None, [1.0]) == 0.0
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0      # zero-norm, not a crash


def test_each_source_is_scored_against_its_own_collection_at_its_own_dimension():
    """The explicit guard: a chunk is compared only with vectors from the graph it
    came from, and the question is embedded at that collection's width."""
    store = _Store(
        dims={"wiki-chunks": 2, "arxiv-chunks": 2},
        vectors={
            "wiki-chunks": {"b1": [1.0, 0.0], "b2": [0.0, 1.0]},
            "arxiv-chunks": {"c1": [1.0, 0.0]},
        },
    )
    emb = _Embedder([1.0, 0.0])
    quotes = [_q("b1", "wikipedia"), _q("b2", "wikipedia"), _q("c1", "arxiv"),
              _q("a1", "other")]
    sc = _sc(quotes, [_src("wikipedia", "wiki-chunks"), _src("arxiv", "arxiv-chunks")],
             store, emb)
    asyncio.run(score_graph_chunks(sc))

    by_id = {q.chunk_id: q for q in sc.evidence.snapshot().quotes}
    assert by_id["b1"].relevance == 1.0        # identical to the query vector
    assert by_id["b2"].relevance == 0.0        # orthogonal
    assert by_id["c1"].relevance == 1.0
    assert by_id["a1"].relevance == 0.0        # an API chunk is never scored

    # each collection was asked ONLY for its own graph's chunk ids
    asked = {c: set(ids) for c, ids in store.asked}
    assert asked == {"wiki-chunks": {"b1", "b2"}, "arxiv-chunks": {"c1"}}
    # and the question was embedded at each collection's dimension
    assert emb.dims == [2, 2]


def test_mixed_embedding_spaces_stamp_nothing_rather_than_rank_wrongly():
    """Two graphs at different widths: the cosines are not on one scale, and the
    digest merges both into ONE tier — so the pass declines to score at all."""
    store = _Store(
        dims={"wiki-chunks": 1536, "arxiv-chunks": 3072},
        vectors={"wiki-chunks": {"b1": [1.0, 0.0]}, "arxiv-chunks": {"c1": [1.0, 0.0]}},
    )
    emb = _Embedder([1.0, 0.0])
    quotes = [_q("b1", "wikipedia"), _q("c1", "arxiv")]
    sc = _sc(quotes, [_src("wikipedia", "wiki-chunks"), _src("arxiv", "arxiv-chunks")],
             store, emb)
    asyncio.run(score_graph_chunks(sc))

    assert all(q.relevance == 0.0 for q in sc.evidence.snapshot().quotes)
    assert store.asked == []          # nothing was even fetched
    assert emb.dims == []             # and nothing was embedded


def test_a_store_failure_leaves_the_order_alone_instead_of_failing_the_request():
    store = _Store(dims={"wiki-chunks": 2}, vectors={}, fail=True)
    quotes = [_q("b1", "wikipedia")]
    sc = _sc(quotes, [_src("wikipedia", "wiki-chunks")], store, _Embedder())
    asyncio.run(score_graph_chunks(sc))       # must not raise
    assert sc.evidence.snapshot().quotes[0].relevance == 0.0


def test_a_source_with_no_chunk_collection_is_skipped():
    store = _Store(dims={}, vectors={})
    quotes = [_q("b1", "wikipedia")]
    sc = _sc(quotes, [_src("wikipedia", "")], store, _Embedder())
    asyncio.run(score_graph_chunks(sc))
    assert store.asked == []
    assert sc.evidence.snapshot().quotes[0].relevance == 0.0


def test_an_unscored_chunk_keeps_zero_so_it_sorts_behind_scored_ones():
    """A chunk whose vector is missing from the collection is not dropped — it
    keeps 0.0 and falls to the back of its tier."""
    store = _Store(dims={"wiki-chunks": 2},
                   vectors={"wiki-chunks": {"b1": [1.0, 0.0]}})
    quotes = [_q("b1", "wikipedia"), _q("b2", "wikipedia")]
    sc = _sc(quotes, [_src("wikipedia", "wiki-chunks")], store, _Embedder([1.0, 0.0]))
    asyncio.run(score_graph_chunks(sc))
    by_id = {q.chunk_id: q.relevance for q in sc.evidence.snapshot().quotes}
    assert by_id == {"b1": 1.0, "b2": 0.0}
