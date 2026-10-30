"""Shared Redis client construction.

One place for the connection/auth policy used by the clarification session
store, conversation memory and the conversation-ownership guard. Returns
``None`` when no host is configured — every caller treats that as "use the
in-process fallback".
"""
from __future__ import annotations

from ai_graph_retrieval.settings import Settings


def redis_client(settings: Settings):
    """An async Redis client per settings, or None (unconfigured)."""
    if not settings.redis_host:
        return None
    import redis.asyncio as redis  # optional `redis` extra; imported only when used

    return redis.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        username=settings.redis_username or None,
        password=settings.redis_password or None,
        decode_responses=True,
    )
