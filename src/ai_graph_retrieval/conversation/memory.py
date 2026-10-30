"""Server-side conversation memory.

The service no longer depends on the CLIENT to replay history each turn: when a
request carries a ``conversation_id`` and no ``history``, the recent turns are
loaded from this store, so an elliptical follow-up ("and who works under
them?") resolves even from a thin client. Client-supplied history always wins —
the store only fills the gap, keeping every existing caller's behavior intact.

Backing store mirrors the clarification/ownership stores: Redis
(``REDIS_CLOUD_*``) in deployed mode so the conversation holds across replicas,
with an in-process TTL fallback for local dev. Key:
``agr:history:{tenant_id}:{conversation_id}``, a list of JSON rows (one per
message), trimmed to the newest ``history_max_turns`` exchanges and expiring
``conversation_memory_ttl_s`` after the last turn.

Fail-open on infrastructure by contract: a store outage degrades to
"no stored history" (the turn still answers, without cross-turn context) and an
append failure is logged, never raised — memory must never break an answer.
Access control: reads/writes are tenant-scoped by key, and the routes layer's
ownership guard (E3) already binds a conversation_id to its first owner, so one
user's stored turns cannot be replayed into another's prompt.
"""
from __future__ import annotations

import json
import time

import structlog

from ai_graph_retrieval.core.access import AccessContext
from ai_graph_retrieval.settings import Settings

logger = structlog.get_logger(__name__)


def _key(tenant_id: str, conversation_id: str) -> str:
    return f"agr:history:{tenant_id}:{conversation_id}"


class ConversationMemory:
    def __init__(self, settings: Settings) -> None:
        self._s = settings
        self._redis = None
        self._ttl = settings.conversation_memory_ttl_s
        # history_max_turns counts MESSAGES everywhere it is used (the
        # orchestrator truncates `history[-history_max_turns:]`), so the store
        # keeps exactly that many rows — floor 2, one full user+assistant
        # exchange. Storing more would only be trimmed again downstream.
        self._max_rows = max(settings.history_max_turns, 2)
        self._local: dict[str, tuple[float, list[str]]] = {}  # key -> (expires_at, rows)

    @property
    def enabled(self) -> bool:
        return self._s.enable_multi_turn_conversation

    def start(self) -> None:
        from ai_graph_retrieval.connectors.redis.client import redis_client

        self._redis = redis_client(self._s)

    async def aclose(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()

    async def history(self, ctx: AccessContext, conversation_id: str) -> list[dict]:
        """The stored recent turns for this conversation (oldest first), shaped
        exactly like client-supplied history rows. [] when disabled, missing,
        expired, or on any store error (fail-open)."""
        if not self.enabled or not conversation_id:
            return []
        key = _key(ctx.tenant_id, conversation_id)
        try:
            if self._redis is not None:
                rows = await self._redis.lrange(key, 0, -1)
            else:
                entry = self._local.get(key)
                rows = entry[1] if entry and entry[0] > time.monotonic() else []
        except Exception as exc:  # a memory outage must never block the answer
            logger.warning("conversation_memory_read_failed", error=str(exc),
                           **ctx.log_fields())
            return []
        turns: list[dict] = []
        for row in rows:
            try:
                m = json.loads(row)
            except ValueError:
                continue  # skip a corrupt row rather than fail the conversation
            # Valid JSON that is not an object (a bare list/string/number) is
            # just as corrupt for our schema — skip it the same way.
            if not isinstance(m, dict):
                continue
            if m.get("role") in ("user", "assistant") and m.get("content"):
                turns.append({"role": m["role"], "content": m["content"]})
        return turns

    async def append(
        self, ctx: AccessContext, conversation_id: str, question: str, answer: str
    ) -> None:
        """Record one completed exchange. Best-effort: never raises.

        Concurrency: the exchange's two rows are written atomically (one Redis
        pipeline), so an exchange can never interleave internally. Two
        concurrent turns on the same conversation may append in completion
        order rather than submission order — accepted chat semantics; a lock
        here would trade a real availability cost for ordering no reader
        depends on."""
        if not self.enabled or not conversation_id or not (question or "").strip():
            return
        cap = self._s.conversation_memory_turn_chars
        rows = [
            json.dumps({"role": "user", "content": question[:cap]}),
            json.dumps({"role": "assistant", "content": (answer or "")[:cap]}),
        ]
        key = _key(ctx.tenant_id, conversation_id)
        try:
            if self._redis is not None:
                # One round trip: append both rows, keep only the newest window,
                # refresh the conversation's TTL.
                pipe = self._redis.pipeline()
                pipe.rpush(key, *rows)
                pipe.ltrim(key, -self._max_rows, -1)
                pipe.expire(key, int(self._ttl))
                await pipe.execute()
                return
            now = time.monotonic()
            if len(self._local) >= 10_000:  # local fallback must not grow unbounded
                self._local = {k: v for k, v in self._local.items() if v[0] > now}
                if len(self._local) >= 10_000:
                    # Nothing expired — evict the entries closest to expiry
                    # (oldest conversations) so the bound holds regardless.
                    survivors = sorted(self._local.items(), key=lambda kv: kv[1][0])
                    self._local = dict(survivors[-9_999:])
            entry = self._local.get(key)
            existing = entry[1] if entry and entry[0] > now else []
            self._local[key] = (now + self._ttl, (existing + rows)[-self._max_rows:])
        except Exception as exc:
            logger.warning("conversation_memory_write_failed", error=str(exc),
                           **ctx.log_fields())
