"""Unit tests for ConversationStore against in-memory fake collections.

The fake implements just the pymongo surface the store touches (find_one,
find_one_and_update, update_one, replace_one, find+sort/skip/limit) with simple
equality matching — enough to exercise the write/read logic without a driver."""
from __future__ import annotations

import asyncio
import copy
from types import SimpleNamespace

import pytest

from ai_graph_retrieval.connectors.mongo.conversation_store import (
    ConversationStore,
    ConversationStoreUnavailable,
)
from tests.support import make_settings


class _ReturnDocument:
    BEFORE = False
    AFTER = True


def _matches(doc: dict, query: dict) -> bool:
    return all(doc.get(k) == v for k, v in query.items())


class _FakeCursor:
    def __init__(self, docs: list[dict]) -> None:
        self._docs = docs

    def sort(self, spec):
        for field, direction in reversed(spec):
            self._docs.sort(key=lambda d: d.get(field) or 0, reverse=direction < 0)
        return self

    def skip(self, n):
        self._docs = self._docs[n:]
        return self

    def limit(self, n):
        self._docs = self._docs[:n]
        return self

    def __iter__(self):
        return iter(self._docs)


class _FakeColl:
    """In-memory collection keyed by `_id`, simple equality queries."""

    def __init__(self) -> None:
        self._docs: list[dict] = []

    def find_one(self, query: dict, sort=None):
        for d in self._docs:
            if _matches(d, query):
                return copy.deepcopy(d)
        return None

    def find(self, query: dict) -> _FakeCursor:
        return _FakeCursor([copy.deepcopy(d) for d in self._docs if _matches(d, query)])

    def _apply(self, doc: dict, update: dict, *, on_insert: bool) -> None:
        if on_insert:
            for k, v in update.get("$setOnInsert", {}).items():
                doc[k] = v
        for k, v in update.get("$set", {}).items():
            doc[k] = v
        for k, v in update.get("$inc", {}).items():
            doc[k] = doc.get(k, 0) + v

    def find_one_and_update(self, query, update, upsert=False, return_document=False):
        for d in self._docs:
            if _matches(d, query):
                self._apply(d, update, on_insert=False)
                return copy.deepcopy(d) if return_document else None
        if not upsert:
            return None
        doc = dict(query)
        self._apply(doc, update, on_insert=True)
        self._docs.append(doc)
        return copy.deepcopy(doc) if return_document else None

    def update_one(self, query, update, upsert=False):
        for d in self._docs:
            if _matches(d, query):
                self._apply(d, update, on_insert=False)
                return
        if upsert:
            doc = dict(query)
            self._apply(doc, update, on_insert=True)
            self._docs.append(doc)

    def replace_one(self, query, doc, upsert=False):
        for i, d in enumerate(self._docs):
            if _matches(d, query):
                self._docs[i] = copy.deepcopy(doc)
                return
        if upsert:
            self._docs.append(copy.deepcopy(doc))


class _BoomColl:
    """Every op raises — simulates a transport/driver failure."""

    def find_one(self, *a, **k):
        raise RuntimeError("mongo down")

    def find(self, *a, **k):
        raise RuntimeError("mongo down")

    def find_one_and_update(self, *a, **k):
        raise RuntimeError("mongo down")

    def update_one(self, *a, **k):
        raise RuntimeError("mongo down")

    def replace_one(self, *a, **k):
        raise RuntimeError("mongo down")


@pytest.fixture(autouse=True)
def _return_document(monkeypatch):
    """Give the store a `pymongo.ReturnDocument` without the real driver."""
    fake_pymongo = SimpleNamespace(ReturnDocument=_ReturnDocument)
    monkeypatch.setitem(__import__("sys").modules, "pymongo", fake_pymongo)


def _store(sessions=None, messages=None):
    return ConversationStore(
        make_settings(mongo_uri="mongodb://x", mongo_db="db"),
        sessions=sessions if sessions is not None else _FakeColl(),
        messages=messages if messages is not None else _FakeColl(),
    )


def _open(store, **kw):
    defaults = {
        "tenant_id": "t1",
        "conversation_id": "c1",
        "user_id": "u1",
        "user_message_id": "um1",
        "ai_message_id": "am1",
        "question": "hello?",
        "mode": "search",
    }
    defaults.update(kw)
    asyncio.run(store.open_turn(**defaults))


def _ai(messages, ai_message_id="am1"):
    return next(d for d in messages._docs if d["messageId"] == ai_message_id)


def test_open_creates_user_and_pending_ai_docs_linked_and_ordered():
    sessions, messages = _FakeColl(), _FakeColl()
    store = _store(sessions, messages)
    _open(store)

    docs = sorted(messages._docs, key=lambda d: d["seq"])
    assert [d["role"] for d in docs] == ["user", "ai_response"]
    assert [d["seq"] for d in docs] == [0, 1]
    assert docs[0]["pairId"] == docs[1]["pairId"] == "am1"
    assert docs[1]["status"] == "pending"
    assert "payload" not in docs[1]  # not filled until completion
    assert "_id" not in docs[1]  # Mongo assigns the ObjectId; we don't set it

    session = sessions._docs[0]
    assert session["messageCount"] == 2
    assert session["conversationId"] == "c1"
    assert session["title"] == "hello?"
    assert session["lastAiMessageId"] == "am1"


def test_second_turn_increments_count_and_seq():
    sessions, messages = _FakeColl(), _FakeColl()
    store = _store(sessions, messages)
    _open(store)
    _open(store, user_message_id="um2", ai_message_id="am2", question="again?")

    assert sessions._docs[0]["messageCount"] == 4
    seqs = sorted(d["seq"] for d in messages._docs)
    assert seqs == [0, 1, 2, 3]


def test_status_transitions_pending_generating_completed():
    sessions, messages = _FakeColl(), _FakeColl()
    store = _store(sessions, messages)
    _open(store)
    assert _ai(messages)["status"] == "pending"

    asyncio.run(store.mark_generating("am1"))
    assert _ai(messages)["status"] == "generating"

    payload = {"answer": "the answer", "request_id": "r9", "mode": "search", "partial": True}
    asyncio.run(store.complete_turn("am1", payload))
    doc = _ai(messages)
    assert doc["status"] == "completed"
    assert doc["payload"] == payload
    assert doc["answer"] == "the answer"
    assert doc["partial"] is True


def test_set_status_terminal_states():
    sessions, messages = _FakeColl(), _FakeColl()
    store = _store(sessions, messages)
    _open(store)
    asyncio.run(store.set_status("am1", "failed", error_code="timeout",
                                 error_message="Retrieval exceeded its time budget"))
    doc = _ai(messages)
    assert doc["status"] == "failed"
    assert doc["errorCode"] == "timeout"
    assert doc["errorMessage"] == "Retrieval exceeded its time budget"


def test_get_by_id_includes_full_payload_on_completed():
    sessions, messages = _FakeColl(), _FakeColl()
    store = _store(sessions, messages)
    _open(store)
    payload = {"answer": "a", "citations": [], "mode": "search"}
    asyncio.run(store.complete_turn("am1", payload))

    out = asyncio.run(store.get_by_id("t1", "c1", "u1"))
    ai = next(m for m in out["messages"] if m["role"] == "ai_response")
    assert ai["status"] == "completed"
    assert ai["payload"] == payload  # the poll read exposes the full payload


def test_get_by_id_raises_on_transport_failure():
    store = _store(sessions=_BoomColl(), messages=_BoomColl())
    with pytest.raises(ConversationStoreUnavailable):
        asyncio.run(store.get_by_id("t1", "c1", "u1"))


def test_writes_never_raise_when_collection_fails():
    store = _store(sessions=_BoomColl(), messages=_BoomColl())
    _open(store)  # must not raise — best-effort contract
    asyncio.run(store.mark_generating("am1"))
    asyncio.run(store.complete_turn("am1", {"answer": "x"}))
    asyncio.run(store.set_status("am1", "failed"))


def test_reopen_resets_to_pending_without_recount():
    sessions, messages = _FakeColl(), _FakeColl()
    store = _store(sessions, messages)
    _open(store)
    asyncio.run(store.set_status("am1", "failed"))
    # Fallback re-runs the same turn: re-open with the SAME ids.
    _open(store)

    ai_docs = [d for d in messages._docs if d["role"] == "ai_response"]
    assert len(ai_docs) == 1  # not duplicated
    assert ai_docs[0]["status"] == "pending"  # reset
    assert sessions._docs[0]["messageCount"] == 2  # not double-counted


def test_get_by_id_assembles_ordered_messages():
    store = _store()
    _open(store)
    _open(store, user_message_id="um2", ai_message_id="am2", question="again?")

    out = asyncio.run(store.get_by_id("t1", "c1", "u1"))
    assert out is not None
    assert [m["seq"] for m in out["messages"]] == [0, 1, 2, 3]
    assert out["session"]["conversationId"] == "c1"


def test_get_by_id_unknown_and_ownership_mismatch_return_none():
    store = _store()
    _open(store)
    assert asyncio.run(store.get_by_id("t1", "missing", "u1")) is None
    assert asyncio.run(store.get_by_id("t1", "c1", "someone_else")) is None


def test_list_for_user_orders_paginates_and_cursors():
    sessions, messages = _FakeColl(), _FakeColl()
    store = _store(sessions, messages)
    # three conversations, distinct updated_at via seq of persists
    for i in range(3):
        _open(
            store,
            conversation_id=f"c{i}",
            user_message_id=f"um{i}",
            ai_message_id=f"am{i}",
        )
    # bump updatedAt ordering deterministically
    for i, d in enumerate(sessions._docs):
        d["updatedAt"] = i

    out = asyncio.run(store.list_for_user("t1", "u1", limit=2, skip=0))
    assert len(out["conversations"]) == 2
    assert out["conversations"][0]["updatedAt"] == 2  # newest first
    assert out["nextCursor"] == 2

    out2 = asyncio.run(store.list_for_user("t1", "u1", limit=2, skip=2))
    assert len(out2["conversations"]) == 1
    assert out2["nextCursor"] is None


def test_a_caller_without_a_user_identity_reads_nothing():
    """The user header is optional at the transport (only the tenant is required),
    so the ownership check must FAIL CLOSED: `owner and user_id and owner !=
    user_id` passed whenever either side was empty, which let a caller with
    internal auth and a tenant — but no user identity — read any conversation in
    the tenant by id."""
    store = _store()
    _open(store)
    assert asyncio.run(store.get_by_id("t1", "c1", None)) is None
    assert asyncio.run(store.get_by_id("t1", "c1", "")) is None
    # the rightful owner still reads it
    assert asyncio.run(store.get_by_id("t1", "c1", "u1")) is not None


# ── the transport-failure cooldown ───────────────────────────────────────────


class _DeadColl:
    """A collection whose every operation fails on transport, counting attempts —
    the shape of Mongo being down (each op waits out server selection, raises)."""

    def __init__(self) -> None:
        self.calls = 0

    def _boom(self, *a, **k):
        self.calls += 1
        raise RuntimeError("server selection timed out")

    find_one = find = find_one_and_update = update_one = replace_one = _boom


def test_a_failed_write_starts_a_cooldown_that_skips_the_store():
    """The lifecycle hooks are awaited INLINE in the turn, so with Mongo down a
    turn otherwise stalls ~3x server-selection timeout for best-effort writes.
    After one failure the store skips itself: no further transport attempt."""
    dead = _DeadColl()
    store = _store(sessions=_FakeColl(), messages=dead)
    _open(store)                                   # fails, starts the cooldown
    attempts = dead.calls
    assert attempts >= 1
    _open(store, ai_message_id="am2")              # skipped — no new attempt
    asyncio.run(store.mark_generating("am1"))      # skipped too
    assert dead.calls == attempts


def test_reads_fail_fast_during_the_cooldown_with_the_same_503():
    dead = _DeadColl()
    store = _store(sessions=dead, messages=_FakeColl())
    with pytest.raises(ConversationStoreUnavailable):
        asyncio.run(store.get_by_id("t1", "c1", "u1"))    # transport failure
    attempts = dead.calls
    with pytest.raises(ConversationStoreUnavailable):
        asyncio.run(store.get_by_id("t1", "c1", "u1"))    # cooldown: instant
    with pytest.raises(ConversationStoreUnavailable):
        asyncio.run(store.list_for_user("t1", "u1", limit=5))
    assert dead.calls == attempts


def test_zero_cooldown_disables_the_gate():
    """0 restores the pre-cooldown behaviour: every call retries immediately."""
    dead = _DeadColl()
    store = ConversationStore(
        make_settings(mongo_uri="mongodb://x", mongo_db="db",
                      graph_store_retry_cooldown_s=0.0),
        sessions=_FakeColl(), messages=dead,
    )
    _open(store)
    attempts = dead.calls
    _open(store, ai_message_id="am2")
    assert dead.calls > attempts


def test_the_cooldown_expires_and_the_store_recovers(monkeypatch):
    dead = _DeadColl()
    store = _store(sessions=_FakeColl(), messages=dead)
    _open(store)                                   # starts the cooldown
    attempts = dead.calls
    monkeypatch.setattr(store, "_down_until", 0.0)  # the clock ran out
    _open(store, ai_message_id="am2")               # tries the store again
    assert dead.calls > attempts


# ── index creation ───────────────────────────────────────────────────────────


class _IndexedColl(_FakeColl):
    def __init__(self) -> None:
        super().__init__()
        self.indexes: list = []

    def create_index(self, keys, **kwargs):
        self.indexes.append((keys, kwargs))


def test_the_query_indexes_are_ensured_once_on_first_use():
    """Every read was a collection scan (nothing created an index anywhere), and
    the session upsert is check-then-act — the unique key is what turns two
    concurrent first turns into one session instead of a duplicate."""
    sessions, messages = _IndexedColl(), _IndexedColl()
    store = _store(sessions=sessions, messages=messages)
    _open(store)
    assert ("messageId", {"unique": True}) in messages.indexes
    assert ([("tenantId", 1), ("conversationId", 1), ("seq", 1)], {}) in messages.indexes
    assert ([("tenantId", 1), ("conversationId", 1)], {"unique": True}) in sessions.indexes
    assert ([("tenantId", 1), ("userId", 1), ("updatedAt", -1)], {}) in sessions.indexes
    before = len(sessions.indexes) + len(messages.indexes)
    _open(store, ai_message_id="am2")          # second use: no re-ensure
    assert len(sessions.indexes) + len(messages.indexes) == before


def test_a_collection_without_create_index_still_serves():
    """The injected-fake path and a restricted deployment both lack create_index;
    the store must run unindexed rather than fail."""
    store = _store()                            # plain fakes: no create_index
    _open(store)
    assert asyncio.run(store.get_by_id("t1", "c1", "u1")) is not None
