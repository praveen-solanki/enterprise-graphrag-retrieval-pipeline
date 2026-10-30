"""GET /v1/graph/conversations (list) and /conversations/{id}/messages (the
WS-recovery poll).

Scoped to the gateway tenant+user headers; 404 for unknown/foreign threads; 503
when persistence is not configured or the store is down. Docs + responses are
camelCase. A completed ai_response carries the full payload; a failed one carries
the error. Endpoint functions are called directly with a fake store."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from ai_graph_retrieval.api.routes.conversations import (
    get_conversation_messages,
    list_conversations,
)
from ai_graph_retrieval.api.schemas import ConversationListResponse, ConversationMessagesResponse
from ai_graph_retrieval.connectors.mongo.conversation_store import ConversationStoreUnavailable
from tests.support import make_settings


class _ReadStore:
    def __init__(self, *, list_out=None, detail=None, raise_=False) -> None:
        self._list = list_out
        self._detail = detail
        self._raise = raise_

    async def list_for_user(self, tenant, user, *, limit, skip):
        if self._raise:
            raise ConversationStoreUnavailable("down")
        return self._list

    async def get_by_id(self, tenant, conv, user):
        if self._raise:
            raise ConversationStoreUnavailable("down")
        return self._detail


class _FakeRequest:
    headers = {"x-tenant-id": "t1", "x-user-id": "u1"}  # noqa: RUF012
    state = SimpleNamespace(request_id="r1")


def _container(store):
    return SimpleNamespace(settings=make_settings(), conversation_store=store)


def test_list_returns_summaries_and_cursor():
    store = _ReadStore(list_out={
        "conversations": [
            {"_id": "s2", "conversationId": "c2", "title": "second",
             "messageCount": 2, "updatedAt": 2},
            {"_id": "s1", "conversationId": "c1", "title": "first",
             "messageCount": 4, "updatedAt": 1},
        ],
        "nextCursor": 2,
    })
    out = asyncio.run(list_conversations(_FakeRequest(), limit=2, skip=0,
                                         container=_container(store)))
    assert isinstance(out, ConversationListResponse)
    assert [c.conversationId for c in out.conversations] == ["c2", "c1"]
    assert out.nextCursor == 2


def test_list_503_when_store_not_configured():
    with pytest.raises(HTTPException) as ei:
        asyncio.run(list_conversations(_FakeRequest(), limit=20, skip=0,
                                       container=_container(None)))
    assert ei.value.status_code == 503


def test_messages_ordered_with_full_payload_on_completed():
    store = _ReadStore(detail={
        "session": {"conversationId": "c1", "title": "first", "messageCount": 2},
        "messages": [
            {"messageId": "um1", "role": "user", "seq": 0, "question": "q"},
            {"messageId": "am1", "role": "ai_response", "seq": 1, "status": "completed",
             "answer": "a", "payload": {"answer": "a", "citations": []}},
        ],
    })
    out = asyncio.run(get_conversation_messages("c1", _FakeRequest(),
                                                container=_container(store)))
    assert isinstance(out, ConversationMessagesResponse)
    assert out.conversationId == "c1"
    assert [m.seq for m in out.messages] == [0, 1]
    ai = out.messages[1]
    assert ai.status == "completed"
    # The full result payload IS exposed on the recovery poll (the FE renders it).
    assert ai.payload == {"answer": "a", "citations": []}


def test_messages_surface_error_on_failed():
    store = _ReadStore(detail={
        "session": {"conversationId": "c1", "messageCount": 2},
        "messages": [
            {"messageId": "um1", "role": "user", "seq": 0, "question": "q"},
            {"messageId": "am1", "role": "ai_response", "seq": 1, "status": "failed",
             "errorCode": "timeout", "errorMessage": "Retrieval exceeded its time budget"},
        ],
    })
    out = asyncio.run(get_conversation_messages("c1", _FakeRequest(),
                                                container=_container(store)))
    ai = out.messages[1]
    assert ai.status == "failed"
    assert ai.errorCode == "timeout"
    assert ai.payload is None


def test_messages_still_pending_is_not_terminal():
    store = _ReadStore(detail={
        "session": {"conversationId": "c1", "messageCount": 2},
        "messages": [
            {"messageId": "um1", "role": "user", "seq": 0, "question": "q"},
            {"messageId": "am1", "role": "ai_response", "seq": 1, "status": "generating"},
        ],
    })
    out = asyncio.run(get_conversation_messages("c1", _FakeRequest(),
                                                container=_container(store)))
    ai = out.messages[1]
    assert ai.status == "generating"
    assert ai.payload is None  # FE keeps polling


def test_messages_404_for_unknown_or_foreign():
    store = _ReadStore(detail=None)  # store folds unknown + ownership-mismatch to None
    with pytest.raises(HTTPException) as ei:
        asyncio.run(get_conversation_messages("nope", _FakeRequest(),
                                              container=_container(store)))
    assert ei.value.status_code == 404


def test_messages_503_when_store_down():
    store = _ReadStore(raise_=True)
    with pytest.raises(HTTPException) as ei:
        asyncio.run(get_conversation_messages("c1", _FakeRequest(),
                                              container=_container(store)))
    assert ei.value.status_code == 503
