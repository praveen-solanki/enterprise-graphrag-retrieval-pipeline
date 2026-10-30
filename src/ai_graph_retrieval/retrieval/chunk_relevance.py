"""Question-similarity for GRAPH chunks, from the vectors ingestion already wrote.

WHY THIS EXISTS. The digest ranks its excerpts by relevance, and a tier needs an
order. A graph chunk carries no engine score (``relevance`` stays 0.0: it was
reached by traversal, not by a search), so this pass supplies one.

WHY EMBEDDINGS RATHER THAN A MODEL CALL. The vectors exist and are unused:
Qdrant holds one point per Neo4j chunk with the chunk_id in the payload's
``entity_id``. The question is embedded once here; the embedder memoises by
(dimension, text), so a repeat within a request is free. That makes the
marginal cost one embed plus one batched Qdrant read per graph, and the result
DETERMINISTIC — the same
question and vectors give the same order every time, which an LLM re-ranker
cannot promise.

THE COMPARABILITY RULE. A cosine is only meaningful against vectors from the same
embedding space. Each graph source is scored against ITS OWN chunk collection,
with the question embedded at THAT collection's dimension — never one query vector
reused across collections. And because the digest merges both graphs into one
ranked tier, scores from different spaces would be silently incomparable: if the
sources disagree on dimension, this pass stamps NOTHING and the tier keeps its
document order. Ranking is skipped rather than done wrongly.

Additive and fail-open throughout: any store or embedding failure leaves
``relevance`` at 0.0, and the digest falls back to the order it used before.
"""
from __future__ import annotations

import json

import structlog

from ai_graph_retrieval.observability.tracing import (
    record_io,
    set_span_attributes,
    stage_span,
)
from ai_graph_retrieval.retrieval.similarity import cosine

logger = structlog.get_logger(__name__)

# How many scored chunks the Output lists, best first. The head is what a reader
# checks against the answer ("the top-scored excerpt is about X — is that what I
# asked?"); the tail is what the character budget was never going to reach.
_TRACE_TOP_CHUNKS = 15


def graph_chunks_by_source(quotes) -> dict[str, list]:
    """Graph chunks grouped by the graph they came from, API chunks excluded.

    The split is by provenance, never by count: nothing here assumes a fixed
    share per graph, so whatever the agent happened to fetch from each
    graph is what gets scored.
    """
    out: dict[str, list] = {}
    for q in quotes:
        if q.graph and q.chunk_id:
            out.setdefault(q.graph, []).append(q)
    return out


async def score_graph_chunks(sc, quotes=None, *, stage: str = "chunk_relevance") -> None:
    """Stamp ``relevance`` on graph chunks, per source, in place.

    Traced as ``stage.chunk_relevance``. This pass decides the ORDER of the
    digest's graph tier, so "why did that excerpt lead the answer?" starts here —
    and every way it can decline to run (no vector store, no chunk collection,
    incomparable embedding spaces, an unreachable store) leaves the tier in
    document order, which is indistinguishable from having run unless the span
    says which happened. The skip reason is therefore an attribute, not a log
    line: a trace has to answer it on its own.

    ``quotes`` defaults to the pool. A caller about to DISCARD chunks passes its
    candidates instead, so the cut is made on the ranking rather than before it.
    """
    with stage_span(f"stage.{stage}") as span:
        await _score_graph_chunks(
            sc, span, sc.evidence.quotes if quotes is None else quotes)


async def _score_graph_chunks(sc, span, quotes) -> None:
    # The sources and the store come from the REPOSITORY, not from config:
    # per-request run selection can resolve different databases and collections
    # than the boot-time defaults, and a chunk must be scored against the
    # collection its own graph was read from.
    #
    # `vector_store` is the one genuinely optional dependency — it is None when the
    # vector arm is disabled — so it is the only thing checked here. The rest are
    # StrategyContext fields the pipeline always sets, and reading them directly
    # means a wiring mistake surfaces as an error instead of this pass quietly
    # scoring nothing.
    store = sc.repo.vector_store
    if store is None:
        set_span_attributes(span, skipped="no_vector_store")
        return
    sources = {s.name: s for s in sc.repo.sources}
    by_source = graph_chunks_by_source(quotes)
    candidates = sum(len(g) for g in by_source.values())
    set_span_attributes(span, candidates=candidates,
                        graphs=sorted(by_source), quotes=len(quotes))
    if not sources or not by_source:
        set_span_attributes(span, skipped="no_graph_chunks")
        return

    # Resolve each source's collection and its dimension FIRST, so the
    # comparability check happens before anything is stamped.
    plan: list[tuple[str, str, int, list]] = []
    for name, group in by_source.items():
        src = sources.get(name)
        collection = src.chunks_collection if src else ""
        if not collection:
            continue
        try:
            dim = await store.vector_size(collection)
        except Exception as exc:                      # pragma: no cover - defensive
            logger.warning("chunk_relevance_dim_failed",
                           graph=name, error=str(exc)[:200])
            dim = None
        if not dim:
            continue
        plan.append((name, collection, int(dim), group))
    if not plan:
        set_span_attributes(span, skipped="no_chunk_collection")
        return

    dims = {dim for _, _, dim, _ in plan}
    if len(dims) > 1:
        # Two graphs embedded at different widths: their cosines are not on the
        # same scale, and the digest merges both into ONE ranked tier. Ranking on
        # mixed scales would quietly prefer whichever space produces larger
        # numbers, so the tier keeps its document order instead.
        logger.warning("⚠️ chunk_relevance_skipped_mixed_spaces " + json.dumps(
            {"dimensions": sorted(dims),
             "graphs": {name: dim for name, _, dim, _ in plan}}, ensure_ascii=False))
        set_span_attributes(span, skipped="mixed_embedding_spaces",
                            dimensions=sorted(dims))
        return

    # The chunks this pass actually scored, kept explicitly rather than recovered
    # afterwards by looking for a non-zero ``relevance``: a chunk scored at exactly
    # 0.0 (an orthogonal vector — real, and the most interesting score there is)
    # is indistinguishable from an unscored one, so deriving the set from the value
    # both under-counts the work and hides the low end of the distribution.
    hits: list = []
    store_errors: list[str] = []
    for name, collection, dim, group in plan:
        try:
            query_vec = await sc.embedder.get(sc.question, dimensions=dim)
            if not query_vec:
                store_errors.append(f"{name}: no query vector")
                continue
            vectors = await store.vectors_by_chunk_id(
                collection, [q.chunk_id for q in group])
        except Exception as exc:
            logger.warning("chunk_relevance_failed", graph=name, error=str(exc)[:200])
            store_errors.append(f"{name}: {str(exc)[:120]}")
            continue
        for q in group:
            vec = vectors.get(q.chunk_id)
            if vec is None:
                continue                       # unscored: keeps 0.0, sorts last
            q.relevance = cosine(query_vec, vec)
            hits.append(q)

    logger.info("🎯 chunk_relevance " + json.dumps(
        {"graphs": {name: len(group) for name, _, _, group in plan},
         "dimension": next(iter(dims)), "scored": len(hits),
         "candidates": candidates},
        ensure_ascii=False))
    _record_scoring(span, plan, hits=hits, candidates=candidates,
                    dimension=next(iter(dims)), store_errors=store_errors)


def _record_scoring(span, plan, *, hits: list, candidates: int,
                    dimension: int, store_errors: list[str]) -> None:
    """The scoring outcome on the span: how much of the pool got a score, the
    DISTRIBUTION of those scores, and the head of the ranking.

    The distribution is the part worth having. A run where every chunk scores
    0.30-0.35 says the question matched nothing in particular and the tier order
    is close to arbitrary — which is a different problem from a run whose top
    chunk scores 0.7, and the two are indistinguishable from a count.
    """
    ranked = sorted(hits, key=lambda q: -q.relevance)
    attrs: dict[str, object] = {
        "dimension": dimension,
        "scored": len(hits),
        "unscored": candidates - len(hits),
        "graph_counts": json.dumps(
            {name: len(group) for name, _, _, group in plan}),
        "collections": sorted({c for _, c, _, _ in plan}),
    }
    if store_errors:
        attrs["store_errors"] = store_errors
    if ranked:
        scores = [q.relevance for q in ranked]
        attrs |= {
            "score_max": round(scores[0], 4),
            "score_min": round(scores[-1], 4),
            "score_mean": round(sum(scores) / len(scores), 4),
        }
    set_span_attributes(span, **attrs)
    record_io(span, output=json.dumps({
        "scored": len(hits), "candidates": candidates,
        "top": [{"score": round(q.relevance, 4), "graph": q.graph,
                 "doc": q.doc_title, "chars": len(q.text or "")}
                for q in ranked[:_TRACE_TOP_CHUNKS]],
    }, ensure_ascii=False, default=str))
