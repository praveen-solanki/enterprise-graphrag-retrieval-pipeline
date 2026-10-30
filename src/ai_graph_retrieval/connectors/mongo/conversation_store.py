"""Durable conversation persistence in MongoDB (stream-recovery support).

Every COMPLETED retrieval turn (sse / one-shot) records itself
here so a client whose stream hung, errored, or timed out can recover the
result over a plain REST call instead of losing it. Two collections in
``settings.mongo_db``:

  Document fields are **camelCase** (``conversationId``, ``messageId``, ``createdAt``,
  …); ``_id`` is left to Mongo (an ObjectId). The one exception is the nested
  ``payload`` on a completed ai_response — it is the RetrieveResponse contract dict
  and keeps its own (snake_case) keys so a client renders it identically to the SSE `final` event.

  * sessions (``graph_conversations``) — one document per ``(tenant, conversation)``,
    upserted every turn: title, timestamps, messageCount, lastAiMessageId.
  * messages (``graph_messages``) — TWO documents per turn, a ``user`` doc and an
    ``ai_response`` doc, linked by ``pairId`` (= the turn's ai_message_id) and
    ordered by a monotonic ``seq``. Both are created UP FRONT (``open_turn``); the
    ai_response doc then transitions ``pending → generating → completed`` (or
    ``failed``/``clarification``) and, on completion, carries the FULL
    ``build_payload()`` dict under ``payload`` — what the fallback reads back and
    returns verbatim as a ``RetrieveResponse``, so the shape can't drift. The
    ``status`` + ``updated_at`` are what the fallback reads to choose serve (done) /
    wait (in flight) / re-run (dead or absent).

Contract notes (mirrors :mod:`run_registry`):
  * Reads (``get_by_id``/``list_for_user``) return ``None``/empty for CHECKED-AND-
    ABSENT; a transport/driver failure raises :class:`ConversationStoreUnavailable`
    so the poll route can 404 (absent) vs 503 (Mongo down).
  * Every WRITE (``open_turn``/``mark_generating``/``complete_turn``/``set_status``)
    is BEST-EFFORT and NEVER raises — losing a history write must never fail a turn
    (same contract as ConversationMemory.append).
  * ``pymongo`` is imported LAZILY: the module stays importable and unit-testable
    via injected collections without the driver present. Off unless
    ``settings.mongo_uri`` is set and ``graph_conversation_persist_enabled``.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

import structlog

from ai_graph_retrieval.settings import Settings

logger = structlog.get_logger(__name__)

_TITLE_CHARS = 200  # session title = first question, truncated for list UIs


def _now_ms() -> int:
    """Current time as a Unix timestamp in MILLISECONDS — the stored format for every
    ``createdAt``/``updatedAt`` (a plain int, JS ``new Date(ms)`` ready; matches the
    service's existing ``timestamp_ms`` convention). Not a BSON date."""
    return int(time.time() * 1000)


class ConversationStoreUnavailable(Exception):
    """A transport/driver failure talking to the conversation store (Mongo down,
    driver missing, bad URI, …). Distinct from a normal ``None`` (record absent):
    a read that hits this must NOT be mistaken for 'no such conversation'."""


class ConversationStore:
    """Reads/writes the two conversation collections. Built by the container only
    when ``settings.mongo_uri`` is set and persistence is enabled."""

    def __init__(
        self,
        settings: Settings,
        *,
        sessions: Any | None = None,
        messages: Any | None = None,
    ) -> None:
        self._s = settings
        self._client = None  # lazy pymongo client
        # Test seams: injected collections bypass the driver entirely.
        self._sessions = sessions
        self._messages = messages
        # Transport-failure cooldown. Every lifecycle hook is awaited INLINE in
        # the turn, and a dead Mongo makes each operation wait out server
        # selection (`graph_store_timeout_ms`) before failing — measured shape:
        # ~3x that per turn (open → generating → complete), on EVERY turn, for
        # writes whose contract is already best-effort. After a failure the
        # store skips itself until the cooldown expires: writes no-op (their
        # contract), reads raise ConversationStoreUnavailable at once (the same
        # 503 they would have reached slowly). monotonic, so a clock step can
        # neither extend nor cut it.
        self._down_until = 0.0
        self._indexes_ensured = False

    def _cooling_down(self) -> bool:
        return time.monotonic() < self._down_until

    def _note_failure(self) -> None:
        cooldown = self._s.graph_store_retry_cooldown_s
        if cooldown > 0:
            self._down_until = time.monotonic() + cooldown
            logger.warning("conversation_store_cooldown", seconds=cooldown)

    async def _collections(self) -> tuple[Any, Any]:
        """The two collections, with the indexes ensured once per process.

        Every read this store issues was a collection scan: nothing anywhere
        created an index, and the query shapes are fixed (messageId point reads,
        (tenant, conversation) → seq for the poll, (tenant, user) → updatedAt for
        the list). Ensured lazily on first use — create_index is idempotent in
        Mongo, a no-op when the index already exists.

        The uniques are correctness, not just speed: ``open_turn`` upserts the
        session with check-then-act, so two concurrent first turns of one
        conversation could both insert; the unique key makes the second a
        conflict instead of a duplicate. Best-effort like every write here — a
        failed build (or a fake without create_index) logs once and the store
        runs unindexed, exactly as before.
        """
        sessions, messages = self._ensure_collections()
        if not self._indexes_ensured:
            self._indexes_ensured = True  # once per process, even on failure
            try:
                await asyncio.to_thread(
                    messages.create_index, "messageId", unique=True)
                await asyncio.to_thread(
                    messages.create_index,
                    [("tenantId", 1), ("conversationId", 1), ("seq", 1)])
                await asyncio.to_thread(
                    sessions.create_index,
                    [("tenantId", 1), ("conversationId", 1)], unique=True)
                await asyncio.to_thread(
                    sessions.create_index,
                    [("tenantId", 1), ("userId", 1), ("updatedAt", -1)])
            except Exception as exc:
                logger.warning("conversation_indexes_failed", error=str(exc)[:200])
        return sessions, messages

    def _ensure_collections(self) -> tuple[Any, Any]:
        if self._sessions is not None and self._messages is not None:
            return self._sessions, self._messages
        from pymongo import MongoClient  # lazy — see module docstring

        self._client = MongoClient(
            self._s.mongo_uri, serverSelectionTimeoutMS=self._s.graph_store_timeout_ms
        )
        db = self._client[self._s.mongo_db]
        self._sessions = db[self._s.graph_conversations_collection]
        self._messages = db[self._s.graph_messages_collection]
        return self._sessions, self._messages

    # ── write: turn lifecycle ─────────────────────────────────────────────────
    async def open_turn(
        self,
        *,
        tenant_id: str,
        conversation_id: str,
        user_id: str,
        user_message_id: str,
        ai_message_id: str,
        question: str,
        mode: str,
    ) -> None:
        """Create the turn's records up front: upsert the session, write the ``user``
        doc, and the ``ai_response`` doc at ``status="pending"``.

        Best-effort (never raises). Idempotent: re-opening the same ``ai_message_id``
        (the fallback re-running a stale/failed turn) resets it to ``pending`` WITHOUT
        re-incrementing ``message_count`` or reshuffling ``seq`` — so a turn is never
        double-counted or duplicated."""
        if self._cooling_down():
            return
        try:
            sessions, messages = await self._collections()
            now = _now_ms()

            existing = await asyncio.to_thread(messages.find_one, {"messageId": ai_message_id})
            if existing is not None:
                # Re-open: reset the ai doc to pending, touch the session, no recount.
                await asyncio.to_thread(
                    messages.update_one,
                    {"messageId": ai_message_id},
                    {"$set": {"status": "pending", "mode": mode, "updatedAt": now}},
                )
                await asyncio.to_thread(
                    sessions.update_one,
                    {"tenantId": tenant_id, "conversationId": conversation_id},
                    {"$set": {"updatedAt": now, "lastAiMessageId": ai_message_id}},
                    upsert=True,
                )
                return

            from pymongo import ReturnDocument  # lazy

            # `_id` is intentionally omitted everywhere — Mongo assigns the ObjectId.
            session = await asyncio.to_thread(
                sessions.find_one_and_update,
                {"tenantId": tenant_id, "conversationId": conversation_id},
                {
                    "$setOnInsert": {
                        "tenantId": tenant_id,
                        "conversationId": conversation_id,
                        "userId": user_id,
                        "title": (question or "").strip()[:_TITLE_CHARS],
                        "createdAt": now,
                    },
                    "$set": {"updatedAt": now, "lastAiMessageId": ai_message_id},
                    "$inc": {"messageCount": 2},
                },
                upsert=True,
                return_document=ReturnDocument.AFTER,
            )
            count = (session or {}).get("messageCount", 2)
            user_doc = {
                "messageId": user_message_id,
                "tenantId": tenant_id,
                "conversationId": conversation_id,
                "userId": user_id,
                "role": "user",
                "seq": count - 2,
                "pairId": ai_message_id,
                "createdAt": now,
                "updatedAt": now,
                "question": question,
            }
            ai_doc = {
                "messageId": ai_message_id,
                "tenantId": tenant_id,
                "conversationId": conversation_id,
                "userId": user_id,
                "role": "ai_response",
                "seq": count - 1,
                "pairId": ai_message_id,
                "createdAt": now,
                "updatedAt": now,
                "status": "pending",
                "mode": mode,
            }
            await asyncio.to_thread(
                messages.replace_one, {"messageId": user_message_id}, user_doc, upsert=True
            )
            await asyncio.to_thread(
                messages.replace_one, {"messageId": ai_message_id}, ai_doc, upsert=True
            )
        except Exception as exc:  # never fail a turn over a history write
            self._note_failure()
            logger.warning(
                "conversation_open_failed", error=str(exc)[:200],
                conversation_id=conversation_id, ai_message_id=ai_message_id,
            )

    async def mark_generating(self, ai_message_id: str) -> None:
        """Flip the ai_response doc to ``generating`` (synthesis has started)."""
        await self._update_ai(ai_message_id, {"status": "generating"})

    async def complete_turn(self, ai_message_id: str, payload: dict) -> None:
        """Finish the turn: store the full ``build_payload()`` dict and flip to
        ``completed`` — this is what the recovery poll reads back. The nested
        ``payload`` keeps the RetrieveResponse contract shape verbatim (snake_case);
        only the document's own top-level fields are camelCase."""
        await self._update_ai(ai_message_id, {
            "status": "completed",
            "payload": payload,
            "answer": payload.get("answer") or "",
            "partial": bool(payload.get("partial")),
            "requestId": payload.get("request_id"),
        })

    async def set_status(
        self,
        ai_message_id: str,
        status: str,
        *,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """Terminal set for a turn that produced no answer — ``failed`` (carrying the
        taxonomy ``errorCode``/``errorMessage`` so the recovery poll can surface the
        real error) or ``clarification``."""
        fields: dict = {"status": status}
        if error_code is not None:
            fields["errorCode"] = error_code
        if error_message is not None:
            fields["errorMessage"] = error_message
        await self._update_ai(ai_message_id, fields)

    async def _update_ai(self, ai_message_id: str, fields: dict) -> None:
        """Best-effort ``$set`` on the ai_response doc (keyed on ``messageId``),
        always bumping ``updatedAt``. Never raises."""
        if self._cooling_down():
            return
        try:
            _, messages = await self._collections()
            await asyncio.to_thread(
                messages.update_one,
                {"messageId": ai_message_id},
                {"$set": {**fields, "updatedAt": _now_ms()}},
            )
        except Exception as exc:
            self._note_failure()
            logger.warning(
                "conversation_status_update_failed",
                error=str(exc)[:200], ai_message_id=ai_message_id,
                status=fields.get("status"),
            )

    # ── reads ────────────────────────────────────────────────────────────────
    async def get_by_id(
        self, tenant_id: str, conversation_id: str, user_id: str | None
    ) -> dict | None:
        """The recovery poll read: a conversation's session + ordered messages,
        each `ai_response` carrying its `status` and — once completed — the full
        `payload` (the response schema exposes what the caller needs). ``None`` if
        unknown OR owned by another user (never leak another principal's thread).
        Raises :class:`ConversationStoreUnavailable` on transport failure."""
        if self._cooling_down():
            raise ConversationStoreUnavailable("store cooling down after failure")
        try:
            sessions, messages = await self._collections()
            session = await asyncio.to_thread(
                sessions.find_one,
                {"tenantId": tenant_id, "conversationId": conversation_id},
            )
            if session is None:
                return None
            # FAIL CLOSED on identity, not just on mismatch. The previous check
            # (`owner and user_id and owner != user_id`) passed whenever EITHER
            # side was empty — and the user header is optional at the transport
            # (`_access_context` requires only the tenant), so a caller with
            # internal auth and a tenant id but NO user identity could read any
            # conversation in the tenant by id. Exact match, empty included:
            # a caller with no identity reads only what was persisted with no
            # identity, which is nothing in any deployment that sends the header.
            if (session.get("userId") or "") != (user_id or ""):
                return None  # unknown and forbidden are the same 404, no leak
            rows = await self._find_list(
                messages,
                {"tenantId": tenant_id, "conversationId": conversation_id},
                sort=[("seq", 1)],
            )
        except ConversationStoreUnavailable:
            raise
        except Exception as err:
            self._note_failure()
            logger.warning(
                "conversation_get_failed",
                error=str(err)[:200],
                conversation_id=conversation_id,
            )
            raise ConversationStoreUnavailable(str(err)) from err
        return {"session": session, "messages": rows}

    async def list_for_user(
        self, tenant_id: str, user_id: str, *, limit: int, skip: int = 0
    ) -> dict:
        """The caller's conversations, newest-updated first, paginated by skip.
        Fetches ``limit + 1`` to compute ``nextCursor`` (skip-based). Raises on
        transport failure."""
        limit = max(1, min(limit, self._s.graph_conversations_list_max))
        if self._cooling_down():
            raise ConversationStoreUnavailable("store cooling down after failure")
        try:
            sessions, _ = await self._collections()
            rows = await self._find_list(
                sessions,
                {"tenantId": tenant_id, "userId": user_id},
                sort=[("updatedAt", -1)],
                skip=skip,
                limit=limit + 1,
            )
        except Exception as err:
            self._note_failure()
            logger.warning("conversation_list_failed", error=str(err)[:200])
            raise ConversationStoreUnavailable(str(err)) from err
        has_more = len(rows) > limit
        return {
            "conversations": rows[:limit],
            "nextCursor": (skip + limit) if has_more else None,
        }

    async def _find_list(
        self,
        collection: Any,
        query: dict,
        *,
        sort: list | None = None,
        skip: int = 0,
        limit: int = 0,
    ) -> list[dict]:
        """Run a find(+sort/skip/limit) chain to completion off the event loop."""

        def _run() -> list[dict]:
            cursor = collection.find(query)
            if sort:
                cursor = cursor.sort(sort)
            if skip:
                cursor = cursor.skip(skip)
            if limit:
                cursor = cursor.limit(limit)
            return list(cursor)

        return await asyncio.to_thread(_run)

    # ── lifecycle ──────────────────────────────────────────────────────────────
    async def verify(self) -> bool:
        """Connectivity probe (readiness). True/False, never raises."""
        try:
            sessions, _ = self._ensure_collections()
            await asyncio.to_thread(
                sessions.database.client.admin.command, "ping"
            )
            return True
        except Exception as exc:
            logger.warning("conversation_store_verify_failed", error=str(exc))
            return False

    async def aclose(self) -> None:
        if self._client is not None:
            await asyncio.to_thread(self._client.close)
            self._client = None
