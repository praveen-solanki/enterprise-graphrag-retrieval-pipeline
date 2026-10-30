"""Composition root — the only place objects are constructed and wired.

Everything long-lived (driver, LLM client, repository, strategies,
orchestrator, ledgers) is built once at startup and torn down at shutdown;
everything request-scoped is created inside the orchestrator per request.
There are no module-level singletons anywhere in this codebase.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import structlog

from ai_graph_retrieval.actions.generator import SuggestedActionGenerator
from ai_graph_retrieval.clarification.service import ClarificationService
from ai_graph_retrieval.clarification.store import ClarificationSessionStore
from ai_graph_retrieval.connectors.llm.client import LlmClient
from ai_graph_retrieval.connectors.mongo.conversation_store import ConversationStore
from ai_graph_retrieval.connectors.mongo.run_registry import RunRegistryClient
from ai_graph_retrieval.connectors.neo4j.client import Neo4jGraph
from ai_graph_retrieval.connectors.neo4j.http_client import Neo4jHttpClient
from ai_graph_retrieval.connectors.qdrant.store import QdrantVectorStore
from ai_graph_retrieval.conversation.memory import ConversationMemory
from ai_graph_retrieval.conversation.ownership import ConversationOwnershipGuard
from ai_graph_retrieval.core.censor import CensorRegistry
from ai_graph_retrieval.graph.repository import GraphReadRepository
from ai_graph_retrieval.observability.ledger import JsonlLedger
from ai_graph_retrieval.observability.trace_log import host_only, redact_settings
from ai_graph_retrieval.observability.usage_stats import UsageAggregator
from ai_graph_retrieval.orchestrator import RetrievalOrchestrator
from ai_graph_retrieval.retrieval.deep_research import (
    DeepResearchStrategy,
    SearchAgentStrategy,
)
from ai_graph_retrieval.retrieval.router import QueryRouter
from ai_graph_retrieval.settings import Settings
from ai_graph_retrieval.synthesis.composer import AnswerComposer

logger = structlog.get_logger(__name__)


@dataclass
class Container:
    """Long-lived objects the app/routes/lifecycle actually reach for; pure
    build-time intermediates (repo, censor) stay local to ``build``."""

    settings: Settings
    graph: Neo4jGraph | None            # Bolt driver (None in Query-API mode)
    http_graph: Neo4jHttpClient | None  # Query API v2 transport (None in Bolt mode)
    llm: LlmClient
    usage_ledger: JsonlLedger
    audit_ledger: JsonlLedger
    orchestrator: RetrievalOrchestrator
    clarification: ClarificationService
    vector_store: QdrantVectorStore | None
    usage_stats: UsageAggregator
    ownership: ConversationOwnershipGuard
    conversation_memory: ConversationMemory
    # Durable conversation persistence. None ⇒ off (no mongo_uri or
    # graph_conversation_persist_enabled=false): turns are not stored and the
    # read-back endpoint always re-runs.
    conversation_store: ConversationStore | None
    # Per-request run selection (RunConfig). None ⇒ feature off (no mongo_uri):
    # requests always use boot-time settings.graph_sources().
    run_registry: RunRegistryClient | None

    @classmethod
    def build(cls, settings: Settings) -> Container:
        # Neo4j transport: Bolt by default; the Query API v2 client when
        # NEO4J_HTTP_URL is set (servers that don't expose Bolt;
        # ai-graph-ingestion writes over the same transport).
        http_graph = Neo4jHttpClient(settings) if settings.neo4j_http_url else None
        graph = None if http_graph else Neo4jGraph(settings)
        # Qdrant backs entity resolution's vector arm and chunk scoring, so it is
        # built whenever a host is configured.
        vector_store = QdrantVectorStore(settings) if settings.qdrant_host else None
        repo = GraphReadRepository(
            graph, settings, vector_store=vector_store, http=http_graph
        )
        # run_registry lookup for per-request run selection — built only when a
        # Mongo endpoint is configured (mirrors the vector_store gate above); off
        # ⇒ every request falls back to boot-time graph_sources().
        run_registry = RunRegistryClient(settings) if settings.mongo_uri else None
        llm = LlmClient(settings, asyncio.Semaphore(settings.llm_max_concurrency))
        censor = CensorRegistry()  # empty = pass-through; the extension seam
        clarification = ClarificationService(settings, ClarificationSessionStore(settings))
        usage_ledger = JsonlLedger(
            settings.usage_ledger_path, name="usage",
            queue_size=settings.ledger_queue_size,
        )
        usage_stats = UsageAggregator()
        ownership = ConversationOwnershipGuard(settings)
        conversation_memory = ConversationMemory(settings)
        # Durable turn persistence — gated on Mongo being configured AND the
        # feature switch (mirrors run_registry's gate above).
        conversation_store = (
            ConversationStore(settings)
            if settings.mongo_uri and settings.graph_conversation_persist_enabled
            else None
        )
        audit_ledger = JsonlLedger(
            settings.audit_ledger_path, name="audit",
            queue_size=settings.ledger_queue_size,
        )
        orchestrator = RetrievalOrchestrator(
            settings=settings,
            repo=repo,
            llm=llm,
            router=QueryRouter(llm, settings),
            composer=AnswerComposer(settings),
            censor=censor,
            usage_ledger=usage_ledger,
            audit_ledger=audit_ledger,
            search_strategy=SearchAgentStrategy(),
            research_strategy=DeepResearchStrategy(),
            usage_stats=usage_stats,
            action_generator=SuggestedActionGenerator(llm, settings),
            conversation_memory=conversation_memory,
            conversation_store=conversation_store,
        )
        return cls(settings=settings, graph=graph, http_graph=http_graph, llm=llm,
                   usage_ledger=usage_ledger,
                   audit_ledger=audit_ledger, orchestrator=orchestrator,
                   clarification=clarification, vector_store=vector_store,
                   usage_stats=usage_stats,
                   ownership=ownership, conversation_memory=conversation_memory,
                   conversation_store=conversation_store, run_registry=run_registry)

    async def start(self) -> None:
        """Async startup work: ledgers, sessions, clarification, then a best-effort warmup."""
        s = self.settings
        t_start = time.monotonic()
        components: list[dict] = []
        await self._log_startup()
        # Config-coherence checks: these mistakes are otherwise silent.
        if not s.acl_enforcement_enabled:
            logger.warning(
                "acl_enforcement_disabled",
                hint=(
                    "ACL retrieval filtering is OFF (AGR_ACL_ENFORCEMENT_ENABLED=false): "
                    "graph/vector reads ignore the caller's principals and may return "
                    "documents they are not permitted to see. Dev/test use only "
                    "— set AGR_ACL_ENFORCEMENT_ENABLED=true to restore fail-closed ACL."
                ),
            )
        t = time.monotonic()
        await self.usage_ledger.start()
        components.append({"name": "usage_ledger", "initTime": round(time.monotonic() - t, 3)})

        t = time.monotonic()
        await self.audit_ledger.start()
        components.append({"name": "audit_ledger", "initTime": round(time.monotonic() - t, 3)})

        t = time.monotonic()
        self.ownership.start()
        components.append({"name": "ownership", "initTime": round(time.monotonic() - t, 3)})

        t = time.monotonic()
        # Behind enable_multi_turn_conversation: don't open a Redis connection for
        # a feature that is disabled. aclose() is a no-op on an unstarted store.
        if self.conversation_memory.enabled:
            self.conversation_memory.start()
        components.append(
            {"name": "conversation_memory", "initTime": round(time.monotonic() - t, 3)}
        )

        t = time.monotonic()
        await self.clarification.start()
        components.append({"name": "clarification", "initTime": round(time.monotonic() - t, 3)})

        # Store connectivity probes (log-only) and the embedder warmup do NOT gate
        # readiness — run them in the BACKGROUND so startup returns immediately
        # instead of blocking on remote round-trips. Kept referenced so the loop
        # can't GC the task mid-flight.
        self._post_start_task = asyncio.create_task(self._post_start_background())

        logger.info(
            "server_startup_complete",
            serverStartupTime=round(time.monotonic() - t_start, 3),
            componentsInitialized=components,
        )

    async def _post_start_background(self) -> None:
        """Non-gating startup work, off the critical path: store probes + embedder
        warmup, run concurrently. Never fatal.

        The per-source probe runs ALWAYS: it is pure observability (which Neo4j
        database and Qdrant collection each source actually serves, and whether the
        query dimension matches) and it is the only place a misconfigured store
        surfaces before a user's query silently returns nothing.
        ``health_check_stores`` gates READINESS — whether store reachability may
        fail the readiness probe — not whether we log what we are wired to."""
        tasks = [("verify_graph_sources", self._verify_graph_sources())]
        if self.settings.health_check_stores:
            tasks.append(("verify_stores", self._verify_stores()))
        if self.settings.warmup_enabled:
            tasks.append(("warmup", self._warmup()))
        results = await asyncio.gather(*(c for _, c in tasks), return_exceptions=True)
        # `return_exceptions=True` COLLECTS failures — it does not report them.
        failed = [
            (name, res)
            for (name, _), res in zip(tasks, results, strict=True)
            if isinstance(res, BaseException)
        ]
        for name, exc in failed:
            logger.error(
                "post_start_check_failed",
                check=name, error=str(exc), error_type=type(exc).__name__, exc_info=exc,
            )
        logger.info("post_start_checks_complete", failed=[n for n, _ in failed])

    async def _log_startup(self) -> None:
        """One-time startup observability: the effective config (secrets masked,
        endpoints host-only), model selection, and the graph sources actually
        wired — so the logs show exactly what config is loaded before any traffic."""
        s = self.settings
        logger.info("config_loaded", environment=s.environment,
                    config=redact_settings(s))
        logger.info(
            "model_selection",
            chat_model=s.llm_model, fast_model=s.fast_model,
            llm_endpoint=host_only(s.llm_base_url or ""),
            embedding_model=s.embedding_model, embedding_dims=s.embedding_dimensions,
            embedding_endpoint=host_only(s.embedding_base_url or s.llm_base_url or ""),
        )
        logger.info(
            "graph_sources",
            sources=[
                {"name": g.name, "database": g.database,
                 "entities_collection": g.entities_collection,
                 "entity_index": g.entity_fulltext_index}
                for g in s.graph_sources()
            ],
            default_connectors=s.default_connectors,
            acl_enforcement=s.acl_enforcement_enabled,
            people_graph=s.people_graph_enabled,
        )
        # Per-request run selection: when on, a connector without a configured
        # source resolves its latest COMPLETED run from run_registry.
        logger.info(
            "run_registry",
            enabled=self.run_registry is not None,
            mongo_endpoint=host_only(s.mongo_uri or ""),
            mongo_db=s.mongo_db if self.run_registry is not None else None,
            collection=s.run_registry_collection if self.run_registry is not None else None,
        )
        store_on = self.conversation_store is not None
        logger.info(
            "conversation_store",
            enabled=store_on,
            mongo_db=s.mongo_db if store_on else None,
            sessions=s.graph_conversations_collection if store_on else None,
            messages=s.graph_messages_collection if store_on else None,
        )

    async def _verify_stores(self) -> None:
        """Live Neo4j + Qdrant CONNECTIVITY probes. LOG-ONLY — they don't gate
        readiness, so ``start()`` runs them in the BACKGROUND rather than blocking
        on it, and both run CONCURRENTLY. Scheduled only when
        ``settings.health_check_stores`` is on; the per-source data probe
        (``_verify_graph_sources``) runs unconditionally alongside it."""
        s = self.settings
        transport = "http_query_v2" if self.http_graph is not None else "bolt"
        neo = self.http_graph or self.graph

        async def _neo4j() -> None:
            t0 = time.monotonic()
            try:
                await neo.verify()
                logger.info("neo4j_connected", transport=transport, database=s.neo4j_database,
                            endpoint=host_only(s.neo4j_http_url or s.neo4j_uri or ""),
                            latency_ms=int((time.monotonic() - t0) * 1000))
            except Exception as exc:
                logger.error("neo4j_connect_failed", transport=transport, error=str(exc),
                             error_type=type(exc).__name__,
                             latency_ms=int((time.monotonic() - t0) * 1000))

        async def _qdrant() -> None:
            if self.vector_store is None:
                return
            t0 = time.monotonic()
            try:
                ok = await self.vector_store.verify()
                logger.info("qdrant_connected", endpoint=host_only(s.qdrant_host or ""),
                            ok=bool(ok), latency_ms=int((time.monotonic() - t0) * 1000))
            except Exception as exc:
                logger.error("qdrant_connect_failed", endpoint=host_only(s.qdrant_host or ""),
                             error=str(exc), error_type=type(exc).__name__)

        await asyncio.gather(_neo4j(), _qdrant(), return_exceptions=True)

    async def _verify_graph_sources(self) -> None:
        """Per-source data-availability check: one line per configured graph source
        stating whether its Neo4j DATABASE answers and its Qdrant COLLECTIONS exist
        — so 'this deployment silently serves only one source' is impossible to
        miss in the logs. Log-only: a missing source degrades at read time
        (graph_read_skipped); nothing here changes behavior. Sources are probed
        CONCURRENTLY so multi-graph startup isn't serialized."""
        async def _probe(source) -> None:
            db_ok = False
            try:
                if self.http_graph is not None:
                    await self.http_graph.rows("RETURN 1 AS ok", {}, database=source.database)
                else:
                    async with self.graph.session(database=source.database) as session:
                        result = await session.run("RETURN 1 AS ok")
                        await result.single()
                db_ok = True
            except Exception as exc:
                logger.error("graph_source_database_unavailable", source=source.name,
                             database=source.database, error=str(exc),
                             error_type=type(exc).__name__)
            collections: dict[str, bool] = {}
            dims: dict[str, int] = {}
            if self.vector_store is not None:
                # Only the entity collection is probed — the entity ANN is the
                # load-bearing Qdrant read; chunk scoring degrades on its own.
                for collection in (source.entities_collection,):
                    try:
                        collections[collection] = bool(
                            await self.vector_store.collection_exists(collection)
                        )
                    except Exception as exc:
                        # Without this the probe reports the collection as
                        # "missing" when Qdrant was merely unreachable — same
                        # verdict for readiness, different diagnosis in the logs.
                        collections[collection] = False
                        logger.error(
                            "graph_source_collection_probe_failed",
                            source=source.name, collection=collection,
                            error=str(exc), error_type=type(exc).__name__,
                        )
                    if collections[collection]:
                        size = await self.vector_store.vector_size(collection)
                        if size is not None:
                            dims[collection] = size
                missing = [name for name, ok in collections.items() if not ok]
                if missing:
                    logger.error("graph_source_collections_missing", source=source.name,
                                 missing=missing)
            # The query dimension MUST equal the collection's, or the entity ANN
            # returns nothing at all (the store swallows Qdrant's rejection to []).
            # Cheap to check once at startup; invisible if left to read time.
            query_dim = self.settings.embedding_dimensions
            mismatched = {name: size for name, size in dims.items() if size != query_dim}
            if mismatched:
                logger.error(
                    "graph_source_embedding_dimension_mismatch",
                    source=source.name, query_dimension=query_dim,
                    collection_dimensions=mismatched,
                    hint="entity ANN will return 0 hits — set AGR_EMBEDDING_DIMENSIONS "
                         "to the collection's dimension (or point at a collection built "
                         "at the query dimension)",
                )
            logger.info(
                "graph_source_verified"
                if db_ok and all(collections.values()) and not mismatched
                else "graph_source_degraded",
                source=source.name, database=source.database, database_ok=db_ok,
                collections=collections, vector_dimensions=dims,
                query_dimension=query_dim,
            )

        await asyncio.gather(
            *(_probe(src) for src in self.settings.graph_sources()),
            return_exceptions=True,
        )

    async def _warmup(self) -> None:
        """Prime the embedding endpoint (and TLS/connection pools) so the
        first real query does not pay the cold start. Never fatal."""
        try:
            async with asyncio.timeout(self.settings.warmup_timeout_s):
                await self.llm.embed(["warmup"])
            logger.info("warmup_completed")
        except Exception as exc:
            logger.warning("warmup_failed", error=str(exc))

    async def aclose(self) -> None:
        await self.usage_ledger.aclose()
        await self.audit_ledger.aclose()
        await self.clarification.aclose()
        await self.ownership.aclose()
        await self.conversation_memory.aclose()
        if self.vector_store is not None:
            await self.vector_store.aclose()
        if self.graph is not None:
            await self.graph.aclose()
        if self.http_graph is not None:
            await self.http_graph.aclose()
        if self.run_registry is not None:
            await self.run_registry.aclose()
        if self.conversation_store is not None:
            await self.conversation_store.aclose()
