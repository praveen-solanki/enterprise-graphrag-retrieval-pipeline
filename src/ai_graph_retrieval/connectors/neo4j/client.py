"""Async Neo4j driver lifecycle — one pooled driver per process.

Sessions are always opened in READ access mode: this service has no write path,
and the database enforces that server-side as defense in depth.
"""
from __future__ import annotations

import structlog
from neo4j import READ_ACCESS, AsyncDriver, AsyncGraphDatabase

from ai_graph_retrieval.settings import Settings

logger = structlog.get_logger(__name__)


class Neo4jGraph:
    def __init__(self, settings: Settings) -> None:
        self._database = settings.neo4j_database
        self._driver: AsyncDriver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
            max_connection_pool_size=settings.neo4j_pool_size,
            connection_timeout=settings.neo4j_connection_timeout_s,
        )

    def session(self, database: str | None = None):
        return self._driver.session(
            database=database or self._database,
            default_access_mode=READ_ACCESS,
        )

    async def verify(self) -> None:
        """Connectivity probe used by the readiness endpoint. Called only when
        ``settings.health_check_stores`` is on."""
        await self._driver.verify_connectivity()

    async def aclose(self) -> None:
        await self._driver.close()
        logger.info("neo4j_driver_closed")
