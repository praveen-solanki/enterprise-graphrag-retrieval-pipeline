"""Durable clarification-session store.

Redis-backed (the company `REDIS_CLOUD_*` cluster) in deployed mode, with an
in-process TTL fallback locally so the flow works without Redis. Key:
``agr:clarify:{tenant_id}:{conversation_id}``, TTL = ``clarification_session_ttl_s``.

``consume`` is **atomic** (Redis ``GETDEL``) — a double-submitted `/resume` can only
run once, so an answer is never processed twice.
"""
from __future__ import annotations

import time

import structlog

from ai_graph_retrieval.clarification.models import ClarificationSession
from ai_graph_retrieval.settings import Settings

logger = structlog.get_logger(__name__)


def _key(tenant_id: str, conversation_id: str) -> str:
    return f"agr:clarify:{tenant_id}:{conversation_id}"


class ClarificationSessionStore:
    def __init__(self, settings: Settings) -> None:
        self._s = settings
        self._redis = None
        self._ttl = settings.clarification_session_ttl_s
        self._local: dict[str, tuple[float, str]] = {}  # key -> (expires_at, json)

    def start(self) -> None:
        from ai_graph_retrieval.connectors.redis.client import redis_client

        self._redis = redis_client(self._s)

    async def open(self, session: ClarificationSession) -> None:
        key = _key(session.tenant_id, session.conversation_id)
        blob = session.model_dump_json()
        if self._redis is not None:
            await self._redis.set(key, blob, ex=int(self._ttl))
        else:
            now = time.monotonic()
            if len(self._local) > 10_000:  # local fallback must not grow unbounded
                self._local = {k: v for k, v in self._local.items() if v[0] > now}
            self._local[key] = (now + self._ttl, blob)

    async def consume(self, tenant_id: str, conversation_id: str) -> ClarificationSession | None:
        """Atomically fetch-and-delete the session (None if missing/expired/already used)."""
        key = _key(tenant_id, conversation_id)
        if self._redis is not None:
            blob = await self._redis.getdel(key)
        else:
            entry = self._local.pop(key, None)
            blob = entry[1] if entry and entry[0] > time.monotonic() else None
        if not blob:
            return None
        return ClarificationSession.model_validate_json(blob)

    async def aclose(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()
