"""Retrieval endpoints: one-shot JSON, the SSE stream, and the clarify/resume turn
(the resume turn has both a JSON and an SSE variant — the latter streams the live
agent trace during the deep-research wait).

Clarifying-questions flow: when the entry classifier finds the question ambiguous,
the service persists a session, emits the questions (a `202 awaiting_clarification`
on `/retrieve`, or a `clarification_request` SSE event on `/retrieve/stream`), and
the UI answers on `POST /retrieve/resume`.

Streaming shape: the orchestrator runs as a background task pushing typed events into
a queue; the SSE generator drains it. Client disconnect cancels the worker.
"""
from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable

import structlog
from fastapi import APIRouter, Depends, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from ai_graph_retrieval.api.deps import get_container, identity_headers, require_internal_auth
from ai_graph_retrieval.api.schemas import (
    ClarificationResponse,
    ResumeRequest,
    RetrieveRequest,
    RetrieveResponse,
)
from ai_graph_retrieval.clarification.models import (
    ClarificationQuestion,
    MidResearchClarification,
)
from ai_graph_retrieval.container import Container
from ai_graph_retrieval.core.access import AccessContext
from ai_graph_retrieval.core.events import (
    ClarificationRequestEvent,
    ErrorEvent,
    FinalEvent,
    StreamEvent,
)
from ai_graph_retrieval.core.exceptions import (
    ConversationConflict,
    GraphQueryError,
    InvalidRequest,
)
from ai_graph_retrieval.core.progress import null_emit
from ai_graph_retrieval.core.run_config import EmbeddingSpace, RequestRunConfig
from ai_graph_retrieval.settings import GraphSource

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/v1/graph", dependencies=[Depends(require_internal_auth)])


@router.get("/tenants")
async def list_tenants(container: Container = Depends(get_container)) -> dict:
    """Tenants that have a COMPLETED ingestion run, newest first.

    The page fills its tenant dropdown from this. An empty list means Mongo is
    not configured or no run has finished. A registry outage is a 422, same as
    a failed run lookup — the page must not invent a tenant.
    """
    registry = container.run_registry
    if registry is None:
        return {"tenants": []}
    try:
        return {"tenants": await registry.completed_tenants()}
    except Exception as err:
        raise InvalidRequest(f"run registry lookup failed: {err}") from err

_QUEUE_MAX = 512
_SENTINEL = None


def _acl_override(body: RetrieveRequest | ResumeRequest) -> bool | None:
    """Per-request ACL-enforcement override from ``options.disabled_acl``:
    None (omitted) ⇒ defer to settings; else ``not disabled_acl`` (disabled_acl
    True ⇒ enforcement off)."""
    opts = getattr(body, "options", None)
    if opts is None or opts.disabled_acl is None:
        return None
    return not opts.disabled_acl


async def _access_context(
    request: Request,
    container: Container,
    *,
    acl_override: bool | None = None,
) -> AccessContext:
    """Build the caller's ACL identity from request headers.

    Tenant comes from ``settings.tenant_header`` (default ``x-tenant-id``), falling
    back to ``settings.default_tenant_id``. Principals come from
    ``settings.principals_header`` when one is configured and present (a
    comma-separated list set by a trusted gateway), else
    ``settings.default_principals`` (``["public"]`` — what the Wikipedia/arXiv
    connectors stamp on every document). There is no body-supplied
    tenant/principals path. ``acl_override`` (from ``options.disabled_acl``)
    overrides ``acl_enforcement_enabled`` for this request and is stamped onto the
    returned context so the repository honors it.
    """
    s = container.settings
    tenant = request.headers.get(s.tenant_header, "").strip() or s.default_tenant_id
    if not tenant:
        raise InvalidRequest(f"missing required header {s.tenant_header}")
    raw = request.headers.get(s.principals_header, "") if s.principals_header else ""
    principals = tuple(p.strip() for p in raw.split(",") if p.strip()) or tuple(
        s.default_principals
    )
    if not principals:
        raise InvalidRequest("no ACL principals for this request")
    return AccessContext(
        tenant_id=tenant, principals=principals,
        request_id=request.state.request_id, acl_enforced=acl_override,
    )


def _suffix_for(connector: str, run_id_suffix: str | dict[str, str] | None) -> str | None:
    """The run_suffix for one connector, or None ⇒ 'no explicit run'. None when
    runIdSuffix was omitted, when this connector's map key is absent or empty
    (""), or when the string form is blank. The string form applies to the single
    selected connector. Schema validation guaranteed the shape."""
    if run_id_suffix is None:
        return None
    if isinstance(run_id_suffix, dict):
        value = run_id_suffix.get(connector)
        return str(value) if value and str(value).strip() else None
    value = str(run_id_suffix)
    return value if value.strip() else None


def _embedding_space(settings, records: list) -> EmbeddingSpace | None:
    """Derive the query embedding space from the resolved run record(s). The
    dimension is load-bearing (query must match the collection). Several runs
    MUST share a dimension — one query embedder can't serve collections built at
    different dims — so a mismatch logs loudly and falls back to the configured
    embedder (space=None)."""
    dims = {r.embedding_dimension for r in records if r.embedding_dimension}
    if len(dims) > 1:
        logger.warning(
            "run_mixed_embedding_dimension",
            dimensions=sorted(dims),
            hint="the selected runs were built at different embedding dims — "
                 "using the configured embedder; vector recall may degrade.",
        )
        return None
    if not dims:
        return None
    (dimension,) = tuple(dims)
    model = next((r.embedding_model for r in records if r.embedding_model), None)
    deployment = (settings.embedding_model_to_deployment.get(model, model)
                  if model else None)
    return EmbeddingSpace(dimensions=dimension, deployment=deployment)


def _graph_source(settings, connector: str, record) -> GraphSource:
    return GraphSource(
        name=connector,
        database=record.neo4j_database,
        entities_collection=record.qdrant_entities_collection,
        chunks_collection=record.qdrant_chunks_collection or "",
        # Fulltext index name isn't stored per-run — ingestion creates a constant
        # name in every run DB.
        entity_fulltext_index=settings.run_fulltext_index_by_connector.get(
            connector, settings.entity_fulltext_index
        ),
    )


def _source_meta(source: GraphSource, run_suffix: str | None, selected_by: str) -> dict:
    """One resolved-source echo entry (for options_applied.run.sources / logs)."""
    return {
        "name": source.name,
        "database": source.database,
        "entities_collection": source.entities_collection,
        "run_suffix": run_suffix,
        "selected_by": selected_by,
    }


async def _lookup_run(registry, ctx: AccessContext, connector: str, suffix: str | None):
    """One run_registry lookup — exact by suffix, else the latest COMPLETED run.
    A transport failure is an explicit 422, never a silent fallback."""
    try:
        if suffix:
            return await registry.resolve(ctx.tenant_id, connector, suffix)
        return await registry.latest(ctx.tenant_id, connector)
    except Exception as err:
        logger.error(
            "run_registry_unavailable",
            connector=connector, run_suffix=suffix, error=str(err), **ctx.log_fields(),
        )
        raise InvalidRequest(
            f"run registry lookup failed for tenant={ctx.tenant_id!r} "
            f"connector={connector!r}: {err} — cannot resolve databases"
        ) from err


async def resolve_run(
    container: Container, ctx: AccessContext, body: RetrieveRequest | ResumeRequest
) -> RequestRunConfig:
    """Resolve the request's graph sources + embedding space + depth overrides.

    ``connectorType`` names the connector(s) to read (a string or a list);
    omitted ⇒ ``settings.default_connectors``. Per connector:

      * suffix PRESENT ⇒ resolve the EXACT run from run_registry (FAILED still
        resolves). A missing record, or Mongo not configured/unreachable, is an
        explicit 422 — a run the caller named must not silently become a
        different database.
      * else a source CONFIGURED in settings (``AGR_GRAPH_SOURCES``) under that
        name ⇒ use it.
      * else the connector's LATEST COMPLETED run in run_registry.
      * none of the above ⇒ 422 for a connector the request named; a DEFAULT
        connector with no run is skipped (422 only if none resolves).
    """
    s = container.settings
    overrides = body.options.model_dump(exclude_none=True) if body.options else {}
    registry = container.run_registry

    explicit = body.connectors()
    selected = explicit or list(s.default_connectors)
    if not selected:
        raise InvalidRequest("no connectorType given and AGR_DEFAULT_CONNECTORS is empty")
    configured = {g.name: g for g in s.graph_sources()}
    sources: list[GraphSource] = []
    records = []
    source_meta: list[dict] = []
    for connector in selected:
        suffix = _suffix_for(connector, body.runIdSuffix)
        if not suffix and connector in configured:
            source = configured[connector]
            sources.append(source)
            source_meta.append(_source_meta(source, None, "settings"))
            continue
        if registry is None:
            if not explicit:
                continue
            raise InvalidRequest(
                f"connector={connector!r} has no configured graph source and the run "
                "registry is not configured (AGR_MONGO_URI unset) to resolve one"
            )
        record = await _lookup_run(registry, ctx, connector, suffix)
        if record is None and not explicit and not suffix:
            logger.info("default_connector_has_no_run", connector=connector,
                        **ctx.log_fields())
            continue
        if record is None:
            wanted = f"run_suffix={suffix!r}" if suffix else "a COMPLETED run"
            raise InvalidRequest(
                f"no run_registry record for tenant={ctx.tenant_id!r} "
                f"connector={connector!r} with {wanted}"
            )
        source = _graph_source(s, connector, record)
        records.append(record)
        logger.info(
            "run_registry_used_for_search",
            connector=connector,
            run_suffix=record.run_suffix,
            selected_by="suffix" if suffix else "latest",
            run_id=record.run_id,
            status=record.status,
            neo4j_database=record.neo4j_database,
            qdrant_collection=record.qdrant_entities_collection,
            embedding_dimension=record.embedding_dimension,
            **ctx.log_fields(),
        )
        sources.append(source)
        source_meta.append(
            _source_meta(source, record.run_suffix, "suffix" if suffix else "latest"))

    if not sources:
        raise InvalidRequest(
            f"none of the default connectors {selected} has a configured graph source "
            f"or a COMPLETED run for tenant={ctx.tenant_id!r}"
        )
    return RequestRunConfig(
        sources=tuple(sources),
        embedding_space=_embedding_space(s, records),
        option_overrides=overrides,
        resolved_meta={
            # "resolved" ONLY when run_registry was actually consulted for at least
            # one source — the echo exists to answer "did a registry lookup pick
            # this database?".
            "run_selection": "resolved" if records else "settings",
            "sources": source_meta,
            "mongo_db": s.mongo_db if records else None,
        },
    )


def _caller_owner(request: Request, container: Container, ctx: AccessContext) -> str:
    """The caller's owner identity for conversation binding: the user header,
    else a `user:` principal."""
    return (
        request.headers.get(container.settings.user_header)
        or next((p for p in ctx.principals if p.startswith("user:")), "")
    )


async def _guard_conversation(
    request: Request,
    container: Container,
    ctx: AccessContext,
    *,
    conversation_id: str | None,
) -> None:
    """Bind a conversation to its FIRST presenter and reject any other owner.
    Binding happens on every conversation_id-bearing turn — including the
    history-less first turn — so a fresh id cannot later be claimed by someone
    else arriving with history."""
    guard = container.ownership
    if not guard.enabled or not conversation_id:
        return
    owner = _caller_owner(request, container, ctx)
    if not await guard.authorize(ctx.tenant_id, conversation_id, owner):
        raise ConversationConflict()


async def _open_pause(
    request: Request,
    container: Container,
    ctx: AccessContext,
    *,
    question: str,
    history: list[dict],
    questions: list[ClarificationQuestion],
    kind: str = "clarification",
    understanding: str | None = None,
    scope: list[str] | None = None,
    chat_conversation_id: str | None = None,
) -> ClarificationResponse:
    """Persist a session for a pipeline-raised pause (pre-research clarification
    or a mid-research clarification) and shape the turn-1 response — one shape for
    both, so the UI answers either on /retrieve/resume identically.

    ``chat_conversation_id`` — the client's chat thread on the paused request,
    carried in the session so the resumed turn reaches the same server-side
    conversation memory (the session's own id below is a one-shot handle)."""
    conversation_id = str(uuid.uuid4())
    await container.clarification.open_session(
        ctx,
        conversation_id=conversation_id,
        user_id=_caller_owner(request, container, ctx) or None,
        question=question,
        history=history,
        questions=questions,
        chat_conversation_id=chat_conversation_id,
    )
    logger.info("pause_opened", kind=kind, **ctx.log_fields())
    return ClarificationResponse(
        request_id=ctx.request_id, conversation_id=conversation_id,
        kind=kind, understanding=understanding, scope=scope or [], questions=questions,
    )


async def _prepare_resume(
    request: Request,
    container: Container,
    body: ResumeRequest,
) -> tuple[AccessContext, str, list[dict], str | None]:
    """Shared prelude for both resume paths (JSON + SSE): resolve identity, consume
    the one-shot session, enforce the recorded-owner match, and compose the
    enriched question + history (+ the chat thread id the pause was opened under,
    so the resumed turn reaches the same conversation memory). Kept in one place so
    the security-sensitive owner check can never drift between the two endpoints.

    A pending clarification belongs to the user who opened it: its enriched question
    + stored history must never resume under a different identity. Fail closed on a
    recorded-owner mismatch."""
    ctx = await _access_context(request, container, acl_override=_acl_override(body))
    session = await container.clarification.consume(ctx, body.conversation_id)
    caller = _caller_owner(request, container, ctx)
    if session.user_id and session.user_id != caller:
        logger.warning("resume_owner_mismatch", **ctx.log_fields())
        raise ConversationConflict()
    question = container.clarification.compose_enriched_question(session, body.answers)
    history = _history(body.history) or session.history
    return ctx, question, history, session.chat_conversation_id


def _clarification_event(
    ctx: AccessContext, resp: ClarificationResponse
) -> ClarificationRequestEvent:
    """One pause → one stream event; every streaming path shapes it here so the
    frame the UI answers from can never drift between endpoints."""
    return ClarificationRequestEvent(
        request_id=ctx.request_id, conversation_id=resp.conversation_id,
        kind=resp.kind, understanding=resp.understanding, scope=resp.scope,
        questions=[q.model_dump() for q in resp.questions],
    )


async def _run_stream_turn(
    request: Request,
    container: Container,
    ctx: AccessContext,
    body: RetrieveRequest,
    sink: Callable[[StreamEvent], Awaitable[None]],
    run_config: RequestRunConfig,
    *,
    conversation_id: str,
    user_message_id: str | None = None,
    ai_message_id: str | None = None,
    owner: str | None = None,
) -> None:
    """One /retrieve/stream turn, routing every event it can produce —
    clarification_request / meta / step / trace_delta / answer_delta / final /
    error — through `sink`. Never raises — every failure becomes an ErrorEvent.
    A pre-retrieval pause (the entry classifier's own clarification questions) is
    raised by the orchestrator's gates stage and lands in the except below.

    The conversation id + message-id pair are generated by the route so the
    orchestrator and the durable store key on the same ids."""
    try:
        payload = await container.orchestrator.retrieve(
            ctx=ctx,
            question=body.question,
            history=_history(body.history),
            mode=body.mode,
            include_trace=body.include_trace,
            emit=sink,
            conversation_id=conversation_id,
            run_config=run_config,
            user_message_id=user_message_id,
            ai_message_id=ai_message_id,
            owner=owner,
        )
        await sink(FinalEvent(payload=payload))
    except MidResearchClarification as pause:
        resp = await _open_pause(
            request, container, ctx, question=body.question,
            history=_history(body.history), questions=pause.questions,
            chat_conversation_id=conversation_id,
        )
        await sink(_clarification_event(ctx, resp))
    except GraphQueryError as exc:
        await sink(ErrorEvent(code=exc.code, message=exc.safe_message))
    except Exception:
        logger.exception("stream_worker_failed", **ctx.log_fields())
        await sink(ErrorEvent(code="internal_error", message="Internal error"))


async def _run_resume_turn(
    request: Request,
    container: Container,
    ctx: AccessContext,
    question: str,
    history: list[dict],
    chat_conversation_id: str | None,
    body: ResumeRequest,
    sink: Callable[[StreamEvent], Awaitable[None]],
    run_config: RequestRunConfig,
    *,
    user_message_id: str | None = None,
    ai_message_id: str | None = None,
    owner: str | None = None,
) -> None:
    """Resume-turn twin of `_run_stream_turn` (plan_review already reviewed, so
    only a mid-research clarification can pause it). Same never-raises contract."""
    try:
        payload = await container.orchestrator.retrieve(
            ctx=ctx, question=question, history=history, mode=body.mode,
            include_trace=body.include_trace, emit=sink,
            plan_review=False,
            conversation_id=chat_conversation_id,
            run_config=run_config,
            user_message_id=user_message_id,
            ai_message_id=ai_message_id,
            owner=owner,
        )
        await sink(FinalEvent(payload=payload))
    except MidResearchClarification as clar:
        resp = await _open_pause(
            request, container, ctx, question=question,
            history=history, questions=clar.questions,
            chat_conversation_id=chat_conversation_id,
        )
        await sink(_clarification_event(ctx, resp))
    except GraphQueryError as exc:
        await sink(ErrorEvent(code=exc.code, message=exc.safe_message))
    except Exception:
        logger.exception("resume_stream_worker_failed", **ctx.log_fields())
        await sink(ErrorEvent(code="internal_error", message="Internal error"))


Turn = Callable[[Callable[[StreamEvent], Awaitable[None]]], Awaitable[None]]


def _sse_turn_response(turn: Turn) -> EventSourceResponse:
    """The SSE transport for one turn: the worker pushes typed events into a
    queue, the generator drains it; client disconnect cancels the worker."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=_QUEUE_MAX)

    async def emit(event: StreamEvent) -> None:
        await queue.put(event)

    async def work() -> None:
        try:
            await turn(emit)
        finally:
            await queue.put(_SENTINEL)

    worker = asyncio.create_task(work())

    async def events():
        try:
            while True:
                event = await queue.get()
                if event is _SENTINEL:
                    break
                yield event.sse()
        finally:
            worker.cancel()  # client went away → stop retrieval immediately

    return EventSourceResponse(events())


@router.post(
    "/retrieve", response_model=RetrieveResponse, dependencies=[Depends(identity_headers)]
)
async def retrieve(
    body: RetrieveRequest,
    request: Request,
    container: Container = Depends(get_container),
):
    ctx = await _access_context(request, container, acl_override=_acl_override(body))
    await _guard_conversation(
        request, container, ctx, conversation_id=body.conversation_id,
    )
    run_config = await resolve_run(container, ctx, body)
    user_mid, ai_mid = _message_ids()
    # A pre-retrieval pause (the entry classifier's own clarification
    # questions) is raised by the orchestrator's gates stage — after entry
    # classification, on the call that already paid for the verdict — and
    # lands below.
    try:
        payload = await container.orchestrator.retrieve(
            ctx=ctx,
            question=body.question,
            history=_history(body.history),
            mode=body.mode,
            include_trace=body.include_trace,
            emit=null_emit,
            conversation_id=body.conversation_id,
            run_config=run_config,
            user_message_id=user_mid,
            ai_message_id=ai_mid,
            owner=_caller_owner(request, container, ctx),
        )
    except MidResearchClarification as clar:
        resp = await _open_pause(
            request, container, ctx, question=body.question,
            history=_history(body.history), questions=clar.questions,
            chat_conversation_id=body.conversation_id,
        )
        return JSONResponse(status_code=202, content=jsonable_encoder(resp))
    return RetrieveResponse(**payload)


@router.post(
    "/retrieve/resume", response_model=RetrieveResponse,
    dependencies=[Depends(identity_headers)],
)
async def retrieve_resume(
    body: ResumeRequest,
    request: Request,
    container: Container = Depends(get_container),
) -> RetrieveResponse:
    ctx, question, history, chat_cid = await _prepare_resume(request, container, body)
    run_config = await resolve_run(container, ctx, body)
    user_mid, ai_mid = _message_ids()
    try:
        payload = await container.orchestrator.retrieve(
            ctx=ctx,
            question=question,
            history=history,
            mode=body.mode,
            include_trace=body.include_trace,
            emit=null_emit,
            plan_review=False,  # already reviewed — don't re-trigger the pause
            conversation_id=chat_cid,
            run_config=run_config,
            user_message_id=user_mid,
            ai_message_id=ai_mid,
            owner=_caller_owner(request, container, ctx),
        )
    except MidResearchClarification as clar:
        # Even the enriched question can still be ambiguous — ask again via a fresh
        # session rather than failing (rare; the answer keeps narrowing each turn).
        resp = await _open_pause(
            request, container, ctx, question=question,
            history=history, questions=clar.questions,
            chat_conversation_id=chat_cid,
        )
        return JSONResponse(status_code=202, content=jsonable_encoder(resp))
    return RetrieveResponse(**payload)


@router.post(
    "/retrieve/stream", response_model=None, dependencies=[Depends(identity_headers)]
)
async def retrieve_stream(
    body: RetrieveRequest,
    request: Request,
    container: Container = Depends(get_container),
) -> EventSourceResponse:
    ctx = await _access_context(request, container, acl_override=_acl_override(body))
    await _guard_conversation(
        request, container, ctx, conversation_id=body.conversation_id,
    )
    run_config = await resolve_run(container, ctx, body)
    user_mid, ai_mid = _message_ids()
    conv_id = body.conversation_id or str(uuid.uuid4())  # one id for orchestrator + store
    owner = _caller_owner(request, container, ctx)

    def turn(sink):
        return _run_stream_turn(
            request, container, ctx, body, sink, run_config,
            conversation_id=conv_id,
            user_message_id=user_mid, ai_message_id=ai_mid, owner=owner,
        )

    return _sse_turn_response(turn)


@router.post(
    "/retrieve/resume/stream", response_model=None,
    dependencies=[Depends(identity_headers)],
)
async def retrieve_resume_stream(
    body: ResumeRequest,
    request: Request,
    container: Container = Depends(get_container),
) -> EventSourceResponse:
    """Streaming twin of POST /retrieve/resume. The JSON endpoint runs the (often
    1-2 min) deep-research agent loop with a no-op emitter and returns only the
    final payload; this one streams the same live trace (meta · step · trace_delta ·
    answer_delta) so a UI can visualize the agent working instead of showing a static
    spinner. The one-shot session is consumed synchronously up front (so an already-
    consumed/mismatched session fails as a normal HTTP error, not mid-stream); only
    the orchestrator loop runs in the background worker."""
    ctx, question, history, chat_cid = await _prepare_resume(request, container, body)
    run_config = await resolve_run(container, ctx, body)
    user_mid, ai_mid = _message_ids()
    owner = _caller_owner(request, container, ctx)

    def turn(sink):
        return _run_resume_turn(
            request, container, ctx, question, history, chat_cid, body, sink, run_config,
            user_message_id=user_mid, ai_message_id=ai_mid, owner=owner,
        )

    return _sse_turn_response(turn)


@router.get("/usage")
async def usage(container: Container = Depends(get_container)) -> dict:
    """Cumulative per-replica usage/cost: totals, per-mode volume, avg
    cost/query, recent rows, and the SLO targets. The JSONL usage ledger is
    the durable cross-replica record; this is the at-a-glance operator view.
    Behind internal auth (this router's dependency)."""
    snap = container.usage_stats.snapshot()
    s = container.settings
    snap["slo"] = {
        "fast_p95_s": s.slo_fast_p95_s,
        "research_p95_s": s.slo_research_p95_s,
        "first_token_s": s.slo_first_token_s,
    }
    return snap


def _history(turns) -> list[dict]:
    return [turn.model_dump() for turn in turns]


def _message_ids() -> tuple[str, str]:
    """A fresh ``(user_message_id, ai_message_id)`` pair for one turn. Generated in
    the route so the SAME ids reach the orchestrator and the durable store — which
    is what lets a client read a turn back by its ``ai_message_id``."""
    return str(uuid.uuid4()), str(uuid.uuid4())
