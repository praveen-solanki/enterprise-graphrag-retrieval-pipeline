"""Conversation→owner binding — cross-session history-leak guard (E3).

The main `/retrieve` path accepts `history` from the caller. If a `conversation_id`
is supplied, we bind it to its first owner and reject any later request that
presents the same conversation under a *different* owner — so one user's prior
turns can never be injected into another user's synthesis prompt (POC
`ConversationRegistry`, Master §E1).

Backing store mirrors the clarification store: Redis (`REDIS_CLOUD_*`) when
configured so the binding holds across replicas, with an in-process TTL fallback
for local dev. First-writer-wins via Redis `SET NX`; a matching owner re-binds
(refreshes TTL) idempotently.

The upstream gateway remains the primary identity boundary; this is defence in
depth, off by default (`enforce_conversation_ownership`), and never fails a
request on a store error (fail-open on infra failure, closed on a real mismatch).
"""
from __future__ import annotations

import hashlib
import time

import structlog

from ai_graph_retrieval.settings import Settings

logger = structlog.get_logger(__name__)


def _key(tenant_id: str, conversation_id: str) -> str:
    return f"agr:owner:{tenant_id}:{conversation_id}"


def _owner_hash(owner: str) -> str:
    """Store a digest, never the raw owner id (audit/PII parity)."""
    return hashlib.sha256(owner.encode("utf-8")).hexdigest()[:32]


class ConversationOwnershipGuard:
    def __init__(self, settings: Settings) -> None:
        self._s = settings
        self._redis = None
        self._ttl = settings.conversation_ownership_ttl_s
        self._local: dict[str, tuple[float, str]] = {}  # key -> (expires_at, owner_hash)

    @property
    def enabled(self) -> bool:
        return self._s.enforce_conversation_ownership

    def start(self) -> None:
        from ai_graph_retrieval.connectors.redis.client import redis_client

        self._redis = redis_client(self._s)

    async def aclose(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()

    async def authorize(
        self, tenant_id: str, conversation_id: str, owner: str
    ) -> bool:
        """Bind (first use) or verify the conversation's owner. Returns False ONLY
        on a real cross-owner mismatch; True otherwise (including infra errors —
        fail-open on infrastructure, never on identity)."""
        if not conversation_id or not owner:
            return True
        key = _key(tenant_id, conversation_id)
        digest = _owner_hash(owner)
        try:
            existing = await self._bind_or_get(key, digest)
        except Exception as exc:  # store outage must not block the answer
            logger.warning("ownership_store_error_failing_open", error=str(exc))
            return True
        if existing is not None and existing != digest:
            logger.warning(
                "conversation_owner_mismatch",
                tenant_id=tenant_id,
                conversation_id=conversation_id,
            )
            return False
        return True

    async def _bind_or_get(self, key: str, digest: str) -> str | None:
        """Set the owner if unset (NX); return the current owner digest."""
        if self._redis is not None:
            # SET NX + refresh TTL for the owner; then read the authoritative value.
            await self._redis.set(key, digest, ex=int(self._ttl), nx=True)
            current = await self._redis.get(key)
            if current == digest:  # ours (just set, or re-bind) → refresh TTL
                await self._redis.expire(key, int(self._ttl))
            return current
        now = time.monotonic()
        if len(self._local) > 10_000:  # local fallback must not grow unbounded
            self._local = {k: v for k, v in self._local.items() if v[0] > now}
        entry = self._local.get(key)
        if entry and entry[0] > now:
            if entry[1] == digest:
                self._local[key] = (now + self._ttl, digest)  # refresh
            return entry[1]
        self._local[key] = (now + self._ttl, digest)  # first use / expired → bind
        return digest
