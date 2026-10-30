"""Quality gate — censor → contradictions → trust ranking, in that order.

Runs once, after retrieval and before synthesis, over an immutable snapshot of
the evidence pool. The ranking itself lives in `retrieval/quality.py`
(relevance-led); this stage only sequences the pass.

The ranking step is the one part a rung can decline (`ExecutionProfile. trust_ranking`).
It exists to impose an order where none was given — evidence fused from several stores
has no inherent one. A rung whose evidence arrives already ranked by the system that
retrieved it keeps that order instead: re-ranking there would reorder the sources, and
so the answer's citation numbers, against a score the user cannot see.

Traced as ``stage.quality_gate``. This stage decides the SOURCE ORDER, and that
order is the answer's citation numbering and the digest's excerpt priority — so
"why is this document first?" is a question about this stage, and it needs the
FEATURES behind each trust score (relevance, mentions, recency, support,
authority), not just the score. They go on the span's Output; the counts and the
distribution go on its attributes.
"""
from __future__ import annotations

import json

from ai_graph_retrieval.core.censor import CensorRegistry
from ai_graph_retrieval.core.models import Evidence, EvidencePool, SourceDocument
from ai_graph_retrieval.observability.tracing import (
    record_io,
    set_span_attributes,
    stage_span,
)
from ai_graph_retrieval.retrieval.quality import (
    detect_contradictions,
    document_relevance,
    rank_quotes,
    rank_sources,
    sources_from_quotes,
)
from ai_graph_retrieval.settings import Settings

# How many ranked sources the Output lists, best first. The head is what the
# citations lead with and what the digest's excerpt order follows.
_TRACE_TOP_SOURCES = 15


class QualityGate:
    def __init__(self, censor: CensorRegistry, settings: Settings) -> None:
        self._censor = censor
        self._s = settings

    def curate(
        self,
        pool: EvidencePool,
        *,
        trust_ranking: bool = True,
        time_scoped: bool = False,
    ) -> Evidence:
        """The synthesis-ready evidence: censored, contradiction-annotated, and —
        unless the mode declined it — sources trust-ranked and excerpts ordered
        by relevance to the question.

        ``trust_ranking=False`` keeps the pool's own order; the rest of the pass still
        runs, so declining the ORDER does not lose the annotation.

        ``time_scoped`` says the question named its own period, which damps the
        recency term — decaying inside a stated window penalises its earliest
        items twice.
        """
        with stage_span("stage.quality_gate", trust_ranking=trust_ranking) as span:
            raw = pool.snapshot()
            snapshot = self._censor.apply(raw)
            # Quote-backed documents the pool has no source for become citable
            # stubs — appended AFTER the pool's own sources, so a rung
            # that keeps retrieval order sees its ranking untouched; the
            # trust-ranked rungs re-sort everything anyway.
            stubs = sources_from_quotes(snapshot.sources, snapshot.quotes)
            contradictions = detect_contradictions(snapshot.claims)
            # Sources first: the excerpt ranking blends each chunk's score with its
            # DOCUMENT's trust, so trust has to be filled before quotes are ordered.
            ranked = rank_sources(
                snapshot.sources + stubs, snapshot.quotes, self._s,
                reorder=trust_ranking, time_scoped=time_scoped,
                relevance_by_key=document_relevance(snapshot.quotes),
            )
            quotes = snapshot.quotes
            if self._s.quote_ranking_enabled:
                quotes = rank_quotes(quotes, ranked, self._s)
            _record_gate(span, raw=raw, snapshot=snapshot, stubs=len(stubs),
                         contradictions=len(contradictions), ranked=ranked,
                         quotes=quotes,
                         scored=any(q.relevance for q in snapshot.quotes))
            return snapshot.model_copy(
                update={
                    "contradictions": contradictions,
                    "sources": ranked,
                    "quotes": quotes,
                }
            )


def _record_gate(span, *, raw: Evidence, snapshot: Evidence, stubs: int,
                 contradictions: int, ranked: list[SourceDocument],
                 quotes: list, scored: bool) -> None:
    """What the gate received, what the censor removed, and how the sources
    ordered — the three things a citation-order question needs.

    ``quote_ranking`` says whether the excerpts carried scores at all — an
    unscored run degrades to document order and looks identical in the output.

    ``sources_unusable`` is the one derived number here, and it earns its place:
    a source with no url can never become a citation (the resolver drops the
    chip), so a run whose ranked head is url-less ships fewer citations than its
    source count suggests — which reads as a synthesis problem and is not one.
    """
    trusts = [s.trust for s in ranked]
    set_span_attributes(
        span,
        entities=len(snapshot.entities),
        facts=len(snapshot.facts),
        claims=len(snapshot.claims),
        quotes=len(snapshot.quotes),
        sources_pool=len(snapshot.sources),
        sources_from_quotes=stubs,
        sources_out=len(ranked),
        contradictions=contradictions,
        # The censor runs before everything else; a drop here is invisible
        # downstream because the counts it changed are the ones every later
        # stage reports.
        censor_dropped=json.dumps({
            k: before - after for k, before, after in (
                ("entities", len(raw.entities), len(snapshot.entities)),
                ("facts", len(raw.facts), len(snapshot.facts)),
                ("claims", len(raw.claims), len(snapshot.claims)),
                ("quotes", len(raw.quotes), len(snapshot.quotes)),
                ("sources", len(raw.sources), len(snapshot.sources)),
            ) if before != after
        }),
        graphs=sorted({s.graph for s in ranked if s.graph}),
        sources_scored=sum(1 for s in ranked if s.relevance),
        # Whether the vector-backed passes ran, or degraded to document order.
        quote_ranking=scored,
        quotes_out=len(quotes),
        sources_dated=sum(1 for s in ranked if s.date),
        sources_unusable=sum(1 for s in ranked if not s.url),
        trust_max=round(max(trusts), 4) if trusts else None,
        trust_min=round(min(trusts), 4) if trusts else None,
        trust_mean=round(sum(trusts) / len(trusts), 4) if trusts else None,
    )
    record_io(span, output=json.dumps({
        "sources": len(ranked),
        "head": [{"title": s.title, "trust": s.trust,
                  "relevance": round(s.relevance, 4), "mentions": s.mentions,
                  "date": s.date, "graph": s.graph, "url": bool(s.url)}
                 for s in ranked[:_TRACE_TOP_SOURCES]],
    }, ensure_ascii=False, default=str))
