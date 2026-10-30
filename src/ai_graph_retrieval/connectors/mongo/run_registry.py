"""Read-only client for the ingestion ``run_registry`` collection.

ai-graph-ingestion writes one document per bulk run recording which Neo4j
database + Qdrant collections that run's data lives in (see
ai-graph-ingestion/repository/run_registry.py). A retrieval request carrying
``connectorType`` + ``runIdSuffix`` uses this client to resolve those names so
it reads the graph THAT run built.

Contract notes:
  * An explicit lookup is by the EXACT ``(tenant_id, connector_type,
    run_suffix)`` — the same triple ingestion enforces a unique index on — and
    does NOT filter on ``status`` (a FAILED run is still resolvable; the caller
    chose it explicitly). ``latest`` picks the newest COMPLETED run instead.
  * Records are immutable per suffix, so results (and negatives) are cached with
    a short TTL to spare Mongo a round-trip on every request.
  * ``pymongo`` is imported LAZILY (the `mongo` extra): the module stays
    importable — and the resolve logic unit-testable via an injected collection —
    without the driver present. The feature is off unless ``settings.mongo_uri``
    is set.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import structlog
from pydantic import BaseModel, ConfigDict

from ai_graph_retrieval.settings import Settings

logger = structlog.get_logger(__name__)


class RunRegistryUnavailable(Exception):
    """A transport/driver failure talking to run_registry (Mongo down, driver not
    installed, bad URI, …). Distinct from 'record not found' (which is a normal
    ``None``): the caller must fail EXPLICITLY, never mistake it for 'no record'
    and never fall back to a default database."""


class RunRecord(BaseModel):
    """The subset of a ``run_registry`` document retrieval needs. Frozen."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    connector_type: str  # as stored in Mongo, e.g. "wikipedia" | "arxiv"
    run_suffix: str
    status: str  # RUNNING | COMPLETED | FAILED — carried for logging, not gating
    neo4j_database: str
    qdrant_entities_collection: str
    # Chunk vectors for the same run; scores graph chunks against the question.
    qdrant_chunks_collection: str | None = None
    # From the record's ``run_config`` — the embedding space this run's vectors
    # were built in. The DIMENSION is the load-bearing knob (query vectors must
    # match the collection's width); the MODEL is used as the query model, or
    # mapped via settings.embedding_model_to_deployment. Absent on legacy docs ⇒ None
    # (caller keeps its configured embedder).
    embedding_model: str | None = None
    embedding_dimension: int | None = None

    @classmethod
    def from_doc(cls, doc: dict) -> RunRecord:
        run_config = doc.get("run_config") or {}
        return cls(
            run_id=doc.get("run_id") or doc.get("_id") or "",
            connector_type=doc["connector_type"],
            run_suffix=doc["run_suffix"],
            status=doc.get("status") or "UNKNOWN",
            neo4j_database=doc["neo4j_database"],
            qdrant_entities_collection=doc["qdrant_entities_collection"],
            qdrant_chunks_collection=doc.get("qdrant_chunks_collection"),
            embedding_model=run_config.get("embedding_model"),
            embedding_dimension=run_config.get("embedding_dimension"),
        )


class RunRegistryClient:
    """Resolves ``run_registry`` rows by ``(tenant, connector_type, suffix)``.

    Built by the container only when ``settings.mongo_uri`` is set. Two distinct
    outcomes, and the difference is load-bearing: ``None`` means CHECKED AND ABSENT
    (no such record), while a transport/driver failure raises
    :class:`RunRegistryUnavailable`. The route turns both into an explicit 422 —
    a run the caller NAMED must never silently resolve to a different database.
    """

    def __init__(self, settings: Settings, *, collection: Any | None = None) -> None:
        self._s = settings
        self._client = None  # lazy pymongo client
        # Test seam: an injected collection bypasses the driver entirely.
        self._collection = collection
        # key (tenant, connector_type, suffix) -> (expiry_monotonic, RunRecord|None)
        self._cache: dict[tuple[str, str, str], tuple[float, RunRecord | None]] = {}

    def _ensure_collection(self):
        if self._collection is not None:
            return self._collection
        from pymongo import MongoClient  # lazy — see module docstring

        self._client = MongoClient(self._s.mongo_uri, serverSelectionTimeoutMS=3000)
        self._collection = self._client[self._s.mongo_db][self._s.run_registry_collection]
        return self._collection

    async def resolve(
        self, tenant_id: str, connector_type: str, run_suffix: str
    ) -> RunRecord | None:
        """Exact lookup by the unique triple, no status filter. Returns the
        record, or None (not found OR any error) so the caller can fall back."""
        return await self._lookup(
            (tenant_id, connector_type, run_suffix),
            {"tenant_id": tenant_id, "connector_type": connector_type, "run_suffix": run_suffix},
            sort=None,
            label=run_suffix,
        )

    async def completed_tenants(self) -> list[dict]:
        """Completed runs grouped by tenant, newest first.

        The UI lists these so a caller picks a corpus instead of typing a tenant
        id. Each row is ``{tenant_id, connectors, latest_at}``. ``connectors``
        are the connector types that tenant has a COMPLETED run for.
        """
        try:
            collection = self._ensure_collection()
            docs = await asyncio.to_thread(
                lambda: list(collection.find(
                    {"status": "COMPLETED"},
                    {"_id": 0, "tenant_id": 1, "connector_type": 1, "created_at": 1},
                ))
            )
        except Exception as err:
            raise RunRegistryUnavailable(str(err)) from err

        grouped: dict[str, dict] = {}
        for doc in docs:
            tenant_id = doc.get("tenant_id") or ""
            if not tenant_id:
                continue
            row = grouped.setdefault(
                tenant_id, {"tenant_id": tenant_id, "connectors": set(), "latest_at": ""})
            if doc.get("connector_type"):
                row["connectors"].add(doc["connector_type"])
            created = doc.get("created_at") or ""
            if created > row["latest_at"]:
                row["latest_at"] = created
        rows = sorted(grouped.values(), key=lambda row: row["latest_at"], reverse=True)
        return [{**row, "connectors": sorted(row["connectors"])} for row in rows]

    async def latest(self, tenant_id: str, connector_type: str) -> RunRecord | None:
        """The newest COMPLETED run for a connector, or None when it has none.

        Used only when a connector has no explicit suffix and no boot-configured
        source. Cached for ``run_registry_cache_ttl_s`` like exact lookups, so a
        new ingestion run is picked up within one TTL."""
        return await self._lookup(
            (tenant_id, connector_type, "<latest>"),
            {"tenant_id": tenant_id, "connector_type": connector_type,
             "status": "COMPLETED"},
            sort=[("created_at", -1)],
            label="<latest>",
        )

    async def _lookup(
        self, cache_key: tuple, query: dict, *, sort: list | None, label: str
    ) -> RunRecord | None:
        now = time.monotonic()
        cached = self._cache.get(cache_key)
        if cached is not None and cached[0] > now:
            return cached[1]

        # What hits Mongo: db + collection + the exact find_one filter and sort.
        hit = {"store": "mongo", "db": self._s.mongo_db,
               "collection": self._s.run_registry_collection,
               "filter": query, "sort": sort, "requested": label}
        try:
            collection = self._ensure_collection()
            # pymongo is sync; run off the event loop so a slow Mongo can't block it.
            doc = await asyncio.to_thread(collection.find_one, query, sort=sort)
            record = RunRecord.from_doc(doc) if doc else None
        except Exception as err:
            # A transport/driver failure is NOT 'record not found'. Do NOT cache and
            # do NOT return None (that would masquerade as an absent record and could
            # trigger a wrong "no record" 4xx); raise so the caller fails explicitly.
            logger.warning("⚠️ run_registry_lookup_failed " + json.dumps(
                {**hit, "error": str(err)[:200]}, ensure_ascii=False, default=str))
            raise RunRegistryUnavailable(str(err)) from err

        if record is None:
            logger.warning("🍃 run_registry_not_found " + json.dumps(
                hit, ensure_ascii=False, default=str))
        else:
            logger.info("🍃 run_registry_resolved " + json.dumps(
                {**hit, "run_id": record.run_id, "status": record.status,
                 "neo4j_database": record.neo4j_database,
                 "entities_collection": record.qdrant_entities_collection,
                 "embedding_dimension": record.embedding_dimension},
                ensure_ascii=False, default=str))
        # Cache positives AND (genuine) negatives so a bad suffix doesn't hit Mongo
        # every call. Transport failures above are never cached.
        self._cache[cache_key] = (now + self._s.run_registry_cache_ttl_s, record)
        return record

    async def verify(self) -> bool:
        """Connectivity probe (readiness). True/False, never raises. Called only
        when ``settings.health_check_stores`` is on. Pings Mongo via the lazily
        built client, off the event loop (same as ``_lookup``); a ``ping`` is the
        standard lightweight liveness command."""
        try:
            collection = self._ensure_collection()
            await asyncio.to_thread(collection.database.client.admin.command, "ping")
            return True
        except Exception as exc:
            logger.warning("run_registry_verify_failed", error=str(exc))
            return False

    async def aclose(self) -> None:
        if self._client is not None:
            # pymongo close is sync + fast; keep it off the loop for symmetry.
            await asyncio.to_thread(self._client.close)
            self._client = None
