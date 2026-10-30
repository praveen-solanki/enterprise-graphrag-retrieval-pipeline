"""FastAPI application factory."""
from __future__ import annotations

import asyncio
import signal
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from ai_graph_retrieval.api.errors import register_exception_handlers
from ai_graph_retrieval.api.middleware import RequestContextMiddleware
from ai_graph_retrieval.api.routes import conversations, health, retrieve, ui
from ai_graph_retrieval.container import Container
from ai_graph_retrieval.observability.logging import configure_logging
from ai_graph_retrieval.observability.tracing import (
    log_observability_state,
    setup_tracing,
    shutdown_tracing,
)
from ai_graph_retrieval.settings import Settings, get_settings

logger = structlog.get_logger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    # No-op when the tracing extra is absent or no exporter is configured — never fatal.
    setup_tracing(settings)
    log_observability_state(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        health.register_service("container")
        # Endpoints come from configuration alone (env / .env) — no probing, no
        # fallback. Startup verifies every configured graph source and collection
        # and logs what it found (see Container._log_startup).
        container = Container.build(settings)
        app.state.container = container
        # SIGTERM → mark not-ready immediately so the load balancer stops routing.
        asyncio.get_event_loop().add_signal_handler(
            signal.SIGTERM, lambda: health.set_ready(False)
        )
        await container.start()
        health.mark_initialized("container")
        health.set_ready(True)
        logger.info(
            "service_started",
            environment=settings.environment,
            chat_model=settings.llm_model,
            neo4j_database=settings.neo4j_database,
        )
        try:
            yield
        finally:
            health.set_ready(False)
            await container.aclose()
            shutdown_tracing()  # flush buffered spans to the exporters
            logger.info("service_stopped")

    _docs = settings.environment != "prod"
    app = FastAPI(
        title="ai-graph-retrieval",
        description=(
            "ACL-enforced GraphRAG inference over the Neo4j knowledge graph."
        ),
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/v1/graph/docs" if _docs else None,
        openapi_url="/v1/graph/openapi.json" if _docs else None,
        redoc_url=None,
    )
    app.add_middleware(RequestContextMiddleware)
    # CORS for direct UI access: tighten `cors_allow_origin_regex` to an allowlist
    # before exposing the service publicly.
    if settings.cors_enabled:
        app.add_middleware(
            CORSMiddleware,
            allow_origin_regex=settings.cors_allow_origin_regex,
            allow_credentials=True,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
        )
    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(ui.router)
    app.include_router(retrieve.router)
    app.include_router(conversations.router)
    return app
