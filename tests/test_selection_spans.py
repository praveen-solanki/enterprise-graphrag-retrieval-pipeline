"""What SELECTED the evidence is on the trace.

Two passes decide which evidence an answer can rest on, and neither produced an
observation: the graph chunk scoring that orders the digest's graph tier, and the
quality gate that orders the sources (which is the answer's citation numbering).
Both can also decline to do their job — an unavailable vector store, a mode that
keeps the engine's order — and a declined pass looks exactly like a pass that ran,
because the fallback is "leave the order alone".

These tests assert against a real in-memory OTEL exporter (the ``spans`` fixture),
so they exercise the actual instrumentation rather than a stand-in. Companion
suite: ``test_digest_spans.py`` covers what the digest and the citation resolver
report.

    uv run pytest tests/test_selection_spans.py -v
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from ai_graph_retrieval.core.models import (
    ChunkQuote,
    EvidencePool,
    SourceDocument,
)
from ai_graph_retrieval.graph.repository import _rows_by_graph
from ai_graph_retrieval.pipeline.quality import QualityGate
from ai_graph_retrieval.retrieval.chunk_relevance import score_graph_chunks
from tests.support import make_settings, span_attrs, span_output

pytest.importorskip("opentelemetry.sdk", reason="OTEL SDK required for span capture")


def _quote(chunk_id: str, graph: str, text: str = "text") -> ChunkQuote:
    return ChunkQuote(chunk_id=chunk_id, doc_id="d1", doc_title="Doc",
                      text=text, graph=graph)


# ── stage.quality_gate ──────────────────────────────────────────────────────


def _gate() -> QualityGate:
    from ai_graph_retrieval.core.censor import CensorRegistry

    return QualityGate(CensorRegistry(), make_settings())


def test_the_quality_gate_span_reports_what_it_ranked(spans):
    pool = EvidencePool()
    pool.add_sources([SourceDocument(doc_id="d1", title="Ranked Doc",
                                     url="http://x/1", relevance=0.8, mentions=3)])
    # a quote whose document the pool has no source for becomes a citable stub
    pool.add_quotes([ChunkQuote(chunk_id="c1", doc_id="d9", doc_title="Quote Only Doc",
                                text="verbatim", graph="wikipedia")])
    _gate().curate(pool)

    at = span_attrs(spans, "stage.quality_gate")
    assert at["trust_ranking"] is True
    assert at["sources_pool"] == 1
    assert at["sources_from_quotes"] == 1
    assert at["sources_out"] == 2
    assert at["sources_scored"] == 1          # only the API-seeded one has relevance
    assert at["sources_unusable"] == 1        # the stub has no url → never a citation
    assert at["trust_max"] >= at["trust_min"]
    head = span_output(spans, "stage.quality_gate")["head"]
    assert [r["title"] for r in head] == ["Ranked Doc", "Quote Only Doc"]
    assert head[0]["url"] is True and head[1]["url"] is False


def test_the_quality_gate_span_records_the_order_it_was_told_to_keep(spans):
    """A rung may decline the re-order; the span has to say so, or its head reads as a
    trust ranking that put a low-relevance document first."""
    pool = EvidencePool()
    pool.add_sources([
        SourceDocument(doc_id="d1", title="Engine First", relevance=0.1),
        SourceDocument(doc_id="d2", title="Engine Second", relevance=0.9),
    ])
    _gate().curate(pool, trust_ranking=False)

    at = span_attrs(spans, "stage.quality_gate")
    assert at["trust_ranking"] is False
    head = span_output(spans, "stage.quality_gate")["head"]
    assert [r["title"] for r in head] == ["Engine First", "Engine Second"]


# ── stage.chunk_relevance ───────────────────────────────────────────────────


class _Store:
    def __init__(self, dims, vectors) -> None:
        self._dims, self._vectors = dims, vectors

    async def vector_size(self, collection):
        return self._dims.get(collection)

    async def vectors_by_chunk_id(self, collection, chunk_ids, *, batch: int = 256):
        return {c: v for c, v in self._vectors.get(collection, {}).items()
                if c in set(chunk_ids)}


class _Embedder:
    async def get(self, _text, *, dimensions=None):
        return [1.0, 0.0]


def _sc(quotes, sources, store):
    pool = EvidencePool()
    pool.add_quotes(quotes)
    return SimpleNamespace(
        question="who owns search?", evidence=pool, embedder=_Embedder(),
        repo=SimpleNamespace(sources=sources, vector_store=store),
    )


def _source(name, collection):
    return SimpleNamespace(name=name, chunks_collection=collection)


def test_the_chunk_relevance_span_reports_the_score_distribution(spans):
    """A run where every chunk scores 0.30-0.35 says the question matched nothing
    in particular and the tier order is near-arbitrary. That is a different problem
    from a run whose top chunk scores 0.7, and a count cannot tell them apart.

    The 0.0 here is the case that matters: an orthogonal vector is a real score,
    and it must be counted as scored rather than mistaken for a chunk the pass
    never reached.
    """
    store = _Store({"wiki-chunks": 2},
                   {"wiki-chunks": {"g1": [1.0, 0.0], "g2": [0.0, 1.0]}})
    quotes = [_quote("g1", "wikipedia"), _quote("g2", "wikipedia"),
              _quote("g3", "wikipedia")]                 # no vector stored
    asyncio.run(score_graph_chunks(
        _sc(quotes, [_source("wikipedia", "wiki-chunks")], store)))

    at = span_attrs(spans, "stage.chunk_relevance")
    assert at["candidates"] == 3
    assert at["scored"] == 2
    assert at["unscored"] == 1                 # g3 has no stored vector
    assert at["dimension"] == 2
    assert (at["score_max"], at["score_min"]) == (1.0, 0.0)
    assert at["score_mean"] == 0.5
    assert json.loads(at["graph_counts"]) == {"wikipedia": 3}
    assert "skipped" not in at
    assert span_output(spans, "stage.chunk_relevance")["top"][0]["score"] == 1.0


@pytest.mark.parametrize(
    ("store", "sources", "reason"),
    [
        (None, [], "no_vector_store"),
        (_Store({"wiki-chunks": 2}, {}), [], "no_graph_chunks"),
        (_Store({}, {}), [("wikipedia", "")], "no_chunk_collection"),
        (_Store({"wiki-chunks": 2, "arxiv-chunks": 3}, {}),
         [("wikipedia", "wiki-chunks"), ("arxiv", "arxiv-chunks")],
         "mixed_embedding_spaces"),
    ],
)
def test_the_chunk_relevance_span_says_why_it_declined(spans, store, sources, reason):
    """Every way this pass can decline leaves the tier in document order, which is
    indistinguishable from having run. The reason is therefore an attribute."""
    quotes = [_quote("g1", "wikipedia"), _quote("c1", "arxiv")]
    asyncio.run(score_graph_chunks(
        _sc(quotes, [_source(n, c) for n, c in sources], store)))
    assert span_attrs(spans, "stage.chunk_relevance")["skipped"] == reason


# ── neo4j per-graph rows ────────────────────────────────────────────────────


def test_rows_are_counted_per_graph_and_untagged_rows_are_not_evidence():
    """Which graphs answered was already recorded; how many rows each returned is
    the number that carries the finding — a 48/2 split reads as a healthy
    multi-graph read under a list of names."""
    assert _rows_by_graph([
        {"graph": "wikipedia"}, {"graph": "wikipedia"}, {"graph": "arxiv"},
        {"no_graph": True},
    ]) == {"wikipedia": 2, "arxiv": 1}
    assert _rows_by_graph([]) == {}
