"""Qdrant vector store — read-only ANN with an ACL ``MatchAny`` push-down.

One async client per process (mirrors ``Neo4jGraph``'s lifecycle). Its single
consumer is the entity ANN in ``GraphReadRepository.resolve_entities`` — the
semantic arm of hybrid entity resolution. When principals are passed they are
pushed into Qdrant as ``MatchAny`` over the point's ``acl`` array (a point
without ``acl`` never matches — fail-closed); ``principals=None`` skips the
ACL leg, and every chunk an entity later leads to is ACL-gated in Neo4j.

Tenant isolation is collection-scoped (each Qdrant collection is a single
tenant+run), so reads do not filter on ``tenant_id``.

``qdrant-client`` is imported lazily so the module loads without the package
(local dev without a vector store); it is required only when a search runs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import structlog

from ai_graph_retrieval.observability.tracing import (
    annotate_current_span,
    record_io,
    set_span_attributes,
    stage_span,
)
from ai_graph_retrieval.settings import Settings

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class VectorHit:
    id: str
    score: float
    payload: dict


class QdrantVectorStore:
    def __init__(self, settings: Settings) -> None:
        self._s = settings
        self._client = None  # lazy — built on first use

    def _ensure_client(self):
        if self._client is None:
            from qdrant_client import AsyncQdrantClient

            # httpx `verify`: a CA-bundle path when configured, else the bool. An
            # internal-CA dev endpoint needs the CA path (or verify off) or TLS
            # fails with "unable to get local issuer certificate".
            verify = self._s.qdrant_ca_cert or self._s.qdrant_verify_tls
            self._client = AsyncQdrantClient(
                url=self._s.qdrant_host,
                api_key=self._s.qdrant_api_key,
                port=self._s.qdrant_port,
                timeout=self._s.qdrant_timeout_s,
                prefer_grpc=self._s.qdrant_prefer_grpc,
                verify=verify,
            )
        return self._client

    def _build_filter(self, principals: list[str] | None):
        """Optional ACL ``MatchAny`` when principals are given.

        ``principals=None`` skips the ACL leg (router reads; the chunks they
        lead to are ACL-gated in Neo4j). A list adds ``MatchAny``.
        """
        from qdrant_client import models as qm

        if principals is None:
            return None
        return qm.Filter(
            must=[
                qm.FieldCondition(
                    key=self._s.qdrant_acl_field, match=qm.MatchAny(any=list(principals))
                )
            ]
        )

    async def search(
        self,
        collection: str,
        vector: list[float],
        *,
        top_k: int,
        principals: list[str] | None,
    ) -> list[VectorHit]:
        """ANN by cosine over the collection's default vector, with an optional
        ACL push-down. Never raises to the caller on a store error — returns []
        so retrieval degrades, not 500s."""
        client = self._ensure_client()
        # One span per ANN search, nested under the tool that asked for
        # it. The VECTOR is never an attribute — 1536 floats per read are noise;
        # its dimension is the useful part (a mismatch is the classic failure).
        # WHICH collection this tool searched, on the tool span itself.
        annotate_current_span(qdrant_collection=collection)
        with stage_span(
            "qdrant.search",
            store="qdrant",
            collection=collection,
            vector_dim=len(vector),
            top_k=top_k,
            acl_pushdown=bool(principals),
        ) as span:
            record_io(span, input=json.dumps(
                {"collection": collection, "top_k": top_k, "vector_dim": len(vector),
                 "acl_pushdown": bool(principals)}, ensure_ascii=False))
            return await self._search_in_span(span, client, collection, vector,
                                              top_k=top_k, principals=principals)

    async def _search_in_span(
        self, span, client, collection: str, vector: list[float], *,
        top_k: int, principals: list[str] | None,
    ) -> list[VectorHit]:
        try:
            resp = await client.query_points(
                collection_name=collection,
                query=vector,
                limit=top_k,
                query_filter=self._build_filter(principals),
                with_payload=True,
            )
        except Exception as exc:
            # Message-EMBEDDED so the failing collection + reason survive log
            # pipelines that surface only `message` (a missing/renamed collection or
            # a dim mismatch reads here, not just a bare "search failed").
            logger.warning("⚠️ qdrant_search_failed " + json.dumps(
                {"collection": collection, "top_k": top_k, "error": str(exc)[:200]},
                ensure_ascii=False))
            # Degrading to [] is the contract; the span says it happened so a
            # trace does not read as "the semantic arm found nothing relevant".
            set_span_attributes(span, store_error=str(exc)[:200], hits=0)
            return []
        hits = [
            VectorHit(id=str(p.id), score=float(p.score), payload=dict(p.payload or {}))
            for p in resp.points
        ]
        # What actually hit Qdrant: the collection, the ANN vector's dimension (not
        # the vector itself), the limit, the ACL push-down, and how many points came
        # back. The vector is elided — logging 1536 floats per read is pure noise.
        logger.info("🧭 qdrant_search " + json.dumps(
            {"store": "qdrant", "collection": collection, "vector_dim": len(vector),
             "top_k": top_k, "acl_principals": len(principals) if principals else 0,
             "hits": len(hits)}, ensure_ascii=False))
        set_span_attributes(span, hits=len(hits))
        # The scored ids are the useful output — the payloads are hydrated in
        # Neo4j afterwards, and the vectors are never worth printing.
        record_io(span, output=json.dumps(
            {"hits": len(hits),
             "top": [{"id": h.id, "score": round(h.score, 4)} for h in hits[:10]]},
            ensure_ascii=False), limit=2000)
        return hits

    async def vectors_by_chunk_id(
        self, collection: str, chunk_ids: list[str], *, batch: int = 256
    ) -> dict[str, list[float]]:
        """The stored vector for each chunk, keyed by chunk id. Missing ids are
        simply absent from the result.

        Ingestion writes one point per chunk with the Neo4j ``chunk_id`` in the
        payload's ``entity_id`` field. This RETRIEVES
        those vectors rather than running a filtered ANN search: we already know
        which chunks we hold and want an exact cosine against each, so an
        approximate nearest-neighbour query with a restrictive filter would be both
        the wrong tool and less deterministic.

        Never raises — an unavailable store returns {} and the caller keeps whatever
        order it already had, the same contract ``search`` has.
        """
        if not chunk_ids:
            return {}
        from qdrant_client import models as qm

        client = self._ensure_client()
        out: dict[str, list[float]] = {}
        with stage_span(
            "qdrant.vectors_by_chunk_id",
            store="qdrant", collection=collection, requested=len(chunk_ids),
        ) as span:
            for start in range(0, len(chunk_ids), batch):
                window = chunk_ids[start : start + batch]
                try:
                    points, _ = await client.scroll(
                        collection_name=collection,
                        scroll_filter=qm.Filter(must=[qm.FieldCondition(
                            key=self._s.qdrant_entity_id_field,
                            match=qm.MatchAny(any=window))]),
                        limit=len(window),
                        with_payload=[self._s.qdrant_entity_id_field],
                        with_vectors=True,
                    )
                except Exception as exc:
                    # Message-EMBEDDED, like qdrant_search_failed: a renamed or
                    # missing chunk collection has to be readable in a log pipeline
                    # that surfaces only `message`.
                    logger.warning("⚠️ qdrant_chunk_vectors_failed " + json.dumps(
                        {"collection": collection, "requested": len(window),
                         "error": str(exc)[:200]}, ensure_ascii=False))
                    break
                for p in points:
                    cid = (p.payload or {}).get(self._s.qdrant_entity_id_field)
                    vec = p.vector
                    # A named-vector collection hands back a mapping; this service
                    # only ever writes the default unnamed vector, so anything else
                    # is a collection we cannot score against — skip rather than
                    # guess which name is the text vector.
                    if cid and isinstance(vec, list):
                        out[str(cid)] = vec
            set_span_attributes(span, resolved=len(out))
        return out

    async def verify(self) -> bool:
        """One cheap round-trip for startup and the readiness probe.

        Both callers (`Container.start` and `/readinessProbe`) have called this
        since the probes were written, but the method never existed on this
        class — every check failed with AttributeError before any network I/O,
        so the log said `qdrant_connect_failed 'QdrantVectorStore' object has
        no attribute 'verify'` on a perfectly healthy endpoint and the probe
        could never report Qdrant as ok. Raises on an unreachable endpoint —
        the callers catch, log and degrade, exactly as they already do.
        """
        await self._ensure_client().get_collections()
        return True

    async def collection_exists(self, collection: str) -> bool:
        """Whether ``collection`` exists on this endpoint (startup source
        verification). Raises on transport errors — the caller decides how a
        'could not check' differs from a 'checked and missing'."""
        return bool(await self._ensure_client().collection_exists(collection))

    async def vector_size(self, collection: str) -> int | None:
        """Dimension of the collection's default (unnamed) vector, or None when it
        can't be determined (unreadable, or a named-vector collection this service
        doesn't use). Startup uses it to catch a query/index dimension mismatch,
        which at read time is invisible: Qdrant rejects the wrong-width vector and
        ``search`` swallows it to ``[]``, so the entity ANN just stops finding
        anything."""
        try:
            info = await self._ensure_client().get_collection(collection)
            params = info.config.params.vectors
        except Exception as exc:
            logger.warning("qdrant_collection_info_failed", collection=collection, error=str(exc))
            return None
        size = getattr(params, "size", None)
        return int(size) if size else None

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()
            logger.info("qdrant_client_closed")
