"""Post-retrieval censor — the late-binding second gate after ACL filtering.

Reserved seam carried from the POC (final_recommendations §Follow-on): some
future sources resolve permissions dynamically at read time (e.g. external
connector-backed objects) and cannot be settled by the ingestion-time ACL alone.
A censor is registered per ``source_type`` and runs over the evidence snapshot
just before synthesis.

Structural contract — censors may only REMOVE:
  a censor receives *copies* and returns the subset to keep; the registry
  intersects the returned keys with its input keys and re-emits the ORIGINAL
  objects. A buggy or malicious censor can therefore neither inject, widen,
  nor mutate evidence — only shrink it.
Default state is an empty registry: a pure pass-through.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import structlog

from ai_graph_retrieval.core.models import ChunkQuote, Evidence, SourceDocument

logger = structlog.get_logger(__name__)

# A censor sees the quotes/sources of ONE source_type and returns those to keep.
QuoteCensor = Callable[[Sequence[ChunkQuote]], Sequence[ChunkQuote]]
SourceCensor = Callable[[Sequence[SourceDocument]], Sequence[SourceDocument]]


class CensorRegistry:
    def __init__(self) -> None:
        self._source_censors: dict[str, SourceCensor] = {}

    def register(self, source_type: str, censor: SourceCensor) -> None:
        self._source_censors[source_type.lower()] = censor

    def apply(self, evidence: Evidence) -> Evidence:
        """Return evidence with censored sources (and their quotes) removed."""
        if not self._source_censors:
            return evidence

        kept_sources: list[SourceDocument] = []
        removed_doc_ids: set[str] = set()
        for source_type, group in self._group(evidence.sources).items():
            censor = self._source_censors.get(source_type)
            if censor is None:
                kept_sources.extend(group)
                continue
            # The censor sees copies; only keys it returns AND we sent survive,
            # and the ORIGINAL objects are re-emitted (mutations are discarded).
            sent = {self._key(s) for s in group}
            allowed = {self._key(s) for s in censor([s.model_copy() for s in group])} & sent
            for s in group:
                if self._key(s) in allowed:
                    kept_sources.append(s)
                elif s.doc_id:
                    removed_doc_ids.add(s.doc_id)

        if not removed_doc_ids and len(kept_sources) == len(evidence.sources):
            return evidence

        logger.info("censor_removed_sources", removed=len(removed_doc_ids))
        return evidence.model_copy(
            update={
                "sources": kept_sources,
                "quotes": [q for q in evidence.quotes if q.doc_id not in removed_doc_ids],
            }
        )

    @staticmethod
    def _key(s: SourceDocument) -> str:
        return s.doc_id or s.title

    @staticmethod
    def _group(sources: list[SourceDocument]) -> dict[str, list[SourceDocument]]:
        grouped: dict[str, list[SourceDocument]] = {}
        for s in sources:
            grouped.setdefault((s.source_type or "").lower(), []).append(s)
        return grouped
