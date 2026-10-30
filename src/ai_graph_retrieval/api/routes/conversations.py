"""Conversation read endpoints — the recovery poll and the history list.

Recovery is poll-only: every turn is persisted by the orchestrator's lifecycle
hooks (`open_turn → mark_generating → complete_turn`/`set_status`), so a client
whose stream hung, errored, or timed out just polls
`GET /v1/graph/conversations/{id}/messages` until the latest `ai_response` reaches
a terminal `status` — `completed` carries the full result payload (same shape as the
SSE `final` event), `failed` carries the error. There is no re-run and no
server-side recompute: the durable copy is the source of truth.

Identity is taken from the identity headers via the retrieve router's helpers so
the two modules can't drift on how a caller is authenticated/scoped."""
from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request

from ai_graph_retrieval.api.deps import get_container, identity_headers, require_internal_auth
from ai_graph_retrieval.api.routes.retrieve import _access_context, _caller_owner
from ai_graph_retrieval.api.schemas import (
    ConversationListResponse,
    ConversationMessagesResponse,
)
from ai_graph_retrieval.connectors.mongo.conversation_store import ConversationStoreUnavailable
from ai_graph_retrieval.container import Container

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/v1/graph", dependencies=[Depends(require_internal_auth)])


@router.get(
    "/conversations",
    response_model=ConversationListResponse,
    dependencies=[Depends(identity_headers)],
)
async def list_conversations(
    request: Request,
    limit: int = Query(default=20, ge=1),
    skip: int = Query(default=0, ge=0),
    container: Container = Depends(get_container),
) -> ConversationListResponse:
    """The caller's conversations, newest-updated first. Scoped to the gateway
    tenant + user headers. Skip-based pagination; the response carries a
    `next_cursor` to pass back as `skip`."""
    ctx = await _access_context(request, container)
    store = _require_store(container)
    owner = _caller_owner(request, container, ctx)
    try:
        out = await store.list_for_user(ctx.tenant_id, owner, limit=limit, skip=skip)
    except ConversationStoreUnavailable as exc:
        raise HTTPException(status_code=503, detail="conversation store unavailable") from exc
    return ConversationListResponse(**out)


@router.get(
    "/conversations/{conversation_id}/messages",
    response_model=ConversationMessagesResponse,
    dependencies=[Depends(identity_headers)],
)
async def get_conversation_messages(
    conversation_id: str,
    request: Request,
    container: Container = Depends(get_container),
) -> ConversationMessagesResponse:
    """The recovery poll: a conversation's messages in `seq` order. Each
    `ai_response` carries its `status`; a `completed` one carries the full result
    `payload` (render it exactly like the SSE `final` event) and a `failed` one its
    `error_code`/`error_message`. A `pending`/`generating` message means "keep
    polling". 404 for an unknown id OR one owned by another user (the store folds an
    ownership mismatch into None so a thread is never disclosed across principals)."""
    ctx = await _access_context(request, container)
    store = _require_store(container)
    owner = _caller_owner(request, container, ctx)
    try:
        out = await store.get_by_id(ctx.tenant_id, conversation_id, owner)
    except ConversationStoreUnavailable as exc:
        raise HTTPException(status_code=503, detail="conversation store unavailable") from exc
    if out is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return ConversationMessagesResponse(
        conversationId=conversation_id, messages=out["messages"]
    )


def _require_store(container: Container):
    store = container.conversation_store
    if store is None:
        raise HTTPException(
            status_code=503, detail="conversation persistence not configured"
        )
    return store
