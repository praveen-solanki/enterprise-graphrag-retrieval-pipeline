"""Neo4j Query API v2 transport — for servers that don't expose Bolt.

The shared dev graph sits behind an HTTPS gateway that forwards only the Query
API (``POST /db/<database>/query/v2``); the Bolt port is unreachable from
anywhere that matters. `ai-graph-ingestion` writes through this same transport.
Selected by setting ``NEO4J_HTTP_URL``; Bolt (`graph/client.py`) stays the
default when it is unset.

Response contract (v2): ``{"data": {"fields": [...], "values": [[...], ...]},
"errors": [{"code", "message"}]?}`` — rows come back as JSON scalars, so dates
are ISO strings (the Cypher in this service already returns strings, never
temporal objects, keeping the two transports interchangeable).
"""
from __future__ import annotations

import asyncio
import json

import httpx
import structlog

from ai_graph_retrieval.core.exceptions import GraphUnavailable
from ai_graph_retrieval.settings import Settings

logger = structlog.get_logger(__name__)

# Idempotent reads: absorb dropped keepalives and transient gateway 5xx.
_MAX_ATTEMPTS = 3
_RETRY_BACKOFF_S = 0.05
_5XX_BODY_CHARS = 300


# The cluster answering "I have no reader for you" — a ROUTING state, not a bad
# query: the statement is fine and the same text succeeds on default routing.
# Matched on the transient code AND the READ-server wording together: the code
# alone also covers unrelated transient remote failures (retrying those without
# the access mode would mask a real fault), and the wording alone would match a
# future non-transient variant we should surface instead.
_READ_ROUTING_CODE = "Neo.TransientError.Statement.RemoteExecutionTransientError"


def _is_read_routing_failure(exc: Neo4jHttpError) -> bool:
    return exc.code == _READ_ROUTING_CODE and "read server" in str(exc).casefold()


class Neo4jHttpError(RuntimeError):
    """A Cypher/server-level error surfaced by the Query API (non-transport)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


class Neo4jHttpClient:
    def __init__(self, settings: Settings) -> None:
        self._base = (settings.neo4j_http_url or "").rstrip("/")
        self._default_db = settings.neo4j_database
        self._client = httpx.AsyncClient(
            auth=(settings.neo4j_username, settings.neo4j_password),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            timeout=httpx.Timeout(
                settings.neo4j_tx_timeout_s,
                connect=settings.neo4j_connection_timeout_s,
            ),
            # Keep connections alive across the agent's inter-round gaps so each call
            # doesn't re-handshake TLS (see neo4j_http_keepalive_expiry_s). Default
            # httpx expiry (5s) is shorter than those gaps.
            limits=httpx.Limits(
                max_keepalive_connections=settings.neo4j_http_max_keepalive,
                keepalive_expiry=settings.neo4j_http_keepalive_expiry_s,
            ),
        )

    async def rows(
        self,
        query: str,
        parameters: dict,
        *,
        database: str | None = None,
        log_fields: dict | None = None,
    ) -> list[dict]:
        """One statement against ``database`` (default) → rows as dicts (same shape
        as the Bolt ``result.data()``). The database is in the URL path, so one
        client serves every database on the endpoint (multi-graph fan-out).

        Routes to a READER, and falls back to default routing when the cluster
        cannot give it one. Both halves come from real dev-cluster failures:
        undeclared statements route as WRITES, so a writer-less cluster failed
        every read ("no longer accepts writes", 2026-08-10) — hence declaring
        READ. But the cluster then flipped the other way (2026-08-12: a writer
        and `readers []`), and READ-routed reads failed while write-routed ones
        succeeded. Declared intent stays READ because it is correct and it
        survives writer loss; the fallback keeps a read-only service answering
        when only a writer is left. Running a read-only statement on the writer
        is safe — it is what this client did for its whole life before READ was
        declared.
        """
        try:
            return await self._rows_once(
                query, parameters, database=database, log_fields=log_fields,
                access_mode="READ",
            )
        except Neo4jHttpError as exc:
            if not _is_read_routing_failure(exc):
                raise
            logger.warning(
                "⚠️ neo4j_read_routing_unavailable_falling_back",
                error=str(exc), **(log_fields or {}),
            )
            return await self._rows_once(
                query, parameters, database=database, log_fields=log_fields,
                access_mode=None,
            )

    async def _rows_once(
        self,
        query: str,
        parameters: dict,
        *,
        database: str | None,
        log_fields: dict | None,
        access_mode: str | None,
    ) -> list[dict]:
        """One attempt at one routing. ``access_mode=None`` omits the field, which
        is the server's default (write) routing."""
        fields = log_fields or {}
        url = f"{self._base}/db/{database or self._default_db}/query/v2"
        # accessMode READ: this service only ever READS the graph, and the Query
        # API routes an undeclared statement as a WRITE — so when the cluster has
        # no writer (leader election, writer-less routing state), every read in
        # the service failed with "no longer accepts writes" (observed on the
        # shared dev graph, 2026-08-10). Declaring the mode routes to readers,
        # which is both correct and what keeps reads up through writer loss.
        # "READ" is the documented Query API v2 enum value; verified against the
        # real dev server (2026-08-10): "READ"/"Read"/"read" all accepted (the
        # server matches case-insensitively), an invalid value is a 400 — the
        # canonical casing is used so a stricter server version changes nothing.
        payload: dict = {"statement": query, "parameters": parameters}
        if access_mode is not None:
            payload["accessMode"] = access_mode

        resp: httpx.Response | None = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                resp = await self._client.post(url, json=payload)
            except httpx.RemoteProtocolError as exc:
                # Kept-alive connection the server closed while idle → "server
                # disconnected" on reuse. Retry on a fresh connection.
                logger.info("🔄 neo4j_http_reconnect", attempt=attempt, **fields)
                if attempt >= _MAX_ATTEMPTS:
                    logger.warning("❌ neo4j_http_unavailable", error=str(exc), **fields)
                    raise GraphUnavailable() from exc
                await asyncio.sleep(_RETRY_BACKOFF_S * attempt)
                continue
            except httpx.HTTPError as exc:
                logger.warning("❌ neo4j_http_unavailable", error=str(exc), **fields)
                raise GraphUnavailable() from exc

            if resp.status_code >= 500:
                snippet = (resp.text or "")[:_5XX_BODY_CHARS]
                logger.warning(
                    "⚠️ neo4j_http_5xx",
                    status=resp.status_code,
                    body=snippet,
                    attempt=attempt,
                    **fields,
                )
                if attempt >= _MAX_ATTEMPTS:
                    raise GraphUnavailable()
                await asyncio.sleep(_RETRY_BACKOFF_S * attempt)
                continue
            break

        assert resp is not None  # loop always assigns or raises
        try:
            body = resp.json()
        except ValueError as exc:
            logger.warning("⚠️ neo4j_http_bad_body", status=resp.status_code, **fields)
            raise GraphUnavailable() from exc
        errors = body.get("errors") or []
        if errors:
            code = str(errors[0].get("code") or "")
            message = str(errors[0].get("message") or "")
            # Embed the failing statement so a message-only pipeline shows exactly
            # what hit Neo4j (Cypher collapsed + capped; ACL/per-source params dropped).
            logged_params = {k: v for k, v in (parameters or {}).items()
                             if k not in ("principals", "entity_index")}
            logger.error(
                "❌ neo4j_query_failed " + json.dumps(
                    {"store": "neo4j", "database": database or self._default_db,
                     "code": code, "error": message[:300],
                     "cypher": " ".join(query.split())[:1500],
                     "params": logged_params}, ensure_ascii=False, default=str),
                **fields)
            raise Neo4jHttpError(code, message)
        data = body.get("data") or {}
        cols = data.get("fields") or []
        return [dict(zip(cols, row, strict=False)) for row in data.get("values") or []]

    async def verify(self) -> None:
        """Connectivity + auth probe (readiness). Raises on failure. Called only
        when ``settings.health_check_stores`` is on."""
        rows = await self.rows("RETURN 1 AS ok", {})
        if not rows or rows[0].get("ok") != 1:
            raise GraphUnavailable()

    async def aclose(self) -> None:
        await self._client.aclose()
        logger.info("🔌 neo4j_http_client_closed")
