"""Distributed tracing on plain OpenTelemetry — cost + latency observability.

Spans export over OTLP/HTTP to any collector (Jaeger, Tempo, an OTel Collector) or
straight to Langfuse's OTLP endpoint. The ``langfuse.*`` attribute names below are
Langfuse's documented OTel mapping; other backends simply show them as attributes.

Everything here is **import-guarded and degrades gracefully**:

  * ``opentelemetry-sdk`` + ``opentelemetry-exporter-otlp-proto-http`` are the
    ``tracing`` extra. Without them ``setup_tracing`` is a logged no-op and
    ``llm_span``/``record_llm`` do nothing — the service runs identically.
  * With them installed, nothing is exported unless ``AGR_TRACING_ENABLED=true``
    and an endpoint is configured (``AGR_OTEL_EXPORTER_OTLP_ENDPOINT``, or the
    three ``AGR_LANGFUSE_*`` settings).

LLM TRACE COVERAGE. Every model call in every mode is a priced generation nested
under the turn's ``retrieve_span``:

    llm.chat          LlmClient.complete   — entry classification incl. the
                                             conditional clarification questions,
                                             neighbour re-ranking, suggested actions
    llm.chat.stream   LlmClient.stream_text — streamed synthesis (continuation
                                             rounds aggregate into the one span)
    llm.embed         LlmClient.embed      — every embedding call (query, entity
                                             ranking, claim-focus ranking)
    llm.chat.agent    _AgentRunHooks       — one span per research-agent round

Every span carries latency (span duration), model, tokens (incl. cache reads),
our computed cost, and the call's Input/Output (chat: messages + response text
or tool calls; embed: the embedded texts + vector count/dims; agent rounds:
the round's system+input items + assistant text/tool calls).

Two properties are pinned in ``tests/test_llm_span_coverage.py``: every call site
produces an ended span carrying non-zero cost, and the per-span costs sum to the
request's ``usage.cost_usd``.
"""
from __future__ import annotations

import base64
import json
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import structlog

from ai_graph_retrieval.settings import Settings

logger = structlog.get_logger("agr.tracing")

# The OTEL API + SDK. Absent ⇒ every entry point below becomes a no-op.
try:
    from opentelemetry import trace as _trace_api
    from opentelemetry.sdk.resources import Resource as _Resource
    from opentelemetry.sdk.trace import TracerProvider as _TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor as _BatchSpanProcessor
    from opentelemetry.trace import Status as _Status
    from opentelemetry.trace import StatusCode as _StatusCode

    _AVAILABLE = True
except ImportError:  # pragma: no cover - OTEL absent (base deps)
    _trace_api = _Status = _StatusCode = None  # type: ignore[assignment,misc]
    _Resource = _TracerProvider = _BatchSpanProcessor = None  # type: ignore[assignment,misc]
    _AVAILABLE = False

# Set True only after ``setup_tracing`` actually initialises the pipeline. ``llm_span``
# gates on this (not just ``_AVAILABLE``) so that when the package is present but tracing
# is disabled or init failed, LLM calls create NO spans — there is no provider to export
# them and the work would be pure overhead.
_INITIALIZED = False
_PROVIDER: Any = None


def _get_tracer(name: str) -> Any:
    return _trace_api.get_tracer(name)


def _set_llm_attributes(
    span: Any, *, model: str | None = None, provider: str | None = None,
    request_type: str | None = None, tokens_prompt: int | None = None,
    tokens_completion: int | None = None, tokens_total: int | None = None,
    cost_input: float | None = None, cost_output: float | None = None,
    cost_total: float | None = None, finish_reason: str | None = None,
    request_messages: str | None = None, response_choices: str | None = None,
) -> None:
    """The OTel GenAI semantic-convention attributes for one model call, plus the
    generation's Input/Output under Langfuse's observation keys."""
    attrs: dict[str, Any] = {
        "gen_ai.request.model": model,
        "gen_ai.system": provider,
        "gen_ai.operation.name": request_type,
        "gen_ai.usage.input_tokens": tokens_prompt,
        "gen_ai.usage.output_tokens": tokens_completion,
        "gen_ai.usage.total_tokens": tokens_total,
        "gen_ai.usage.cost": cost_total,
        "gen_ai.response.finish_reasons": finish_reason,
        "langfuse.observation.type": "generation",
        "langfuse.observation.input": request_messages,
        "langfuse.observation.output": response_choices,
    }
    del cost_input, cost_output  # carried by _set_langfuse_usage's cost_details
    for key, value in attrs.items():
        if value is not None:
            span.set_attribute(key, value)


# The request's correlation id, stamped on EVERY span the request creates.
# A contextvar rather than a threaded argument because the id has to
# reach spans opened deep in the call tree — a tool's Neo4j read, an embedding
# inside a claim ranking — without every layer in between growing a parameter.
# asyncio COPIES the context when a task is created, so the value reaches the
# Agents SDK's per-tool tasks too (unlike the parent SPAN, which must be passed
# explicitly: a copied contextvar carries values, not OTEL's active span).
_correlation_id: ContextVar[str | None] = ContextVar("agr_correlation_id", default=None)


def set_correlation_id(value: str | None) -> None:
    """Bind the correlation id for the rest of this request. Called once, at the
    top of the turn; every span created afterwards carries it, so a trace can be
    reassembled — or joined to the logs — from any single observation."""
    _correlation_id.set(value or None)


def correlation_id() -> str | None:
    """The current request's correlation id, if one is bound."""
    return _correlation_id.get()


def _stamp(span: Any) -> Any:
    """Put the correlation id on a freshly created span. Every span factory in
    this module goes through here, so a new span type cannot forget it."""
    if span is None:
        return span
    cid = _correlation_id.get()
    if cid:
        try:
            span.set_attribute("correlation_id", cid)
        except Exception:  # pragma: no cover - defensive
            logger.warning("correlation_stamp_failed", exc_info=True)
    return span


def annotate_current_span(**attrs: Any) -> None:
    """Add attributes to whatever span is CURRENT, from code that does not hold
    it.

    This is how a tool span learns its data sources: the repository and the
    vector store know which graphs and which collection they read, but they are
    called from inside the tool's span and have no reference to it. Inside a tool
    body the tool span IS current, so annotating "here" lands the source on the
    tool — visible beside the tool name, input and output, with no plumbing.
    """
    if not (_AVAILABLE and _trace_api is not None):
        return
    try:
        span = _trace_api.get_current_span()
        if span is None or not getattr(span, "is_recording", lambda: False)():
            return
        set_span_attributes(span, **attrs)
    except Exception:  # pragma: no cover - observability must never break a request
        logger.warning("annotate_current_span_failed", exc_info=True)


def _langfuse_configured(settings: Settings) -> bool:
    return bool(
        settings.langfuse_public_key and settings.langfuse_secret_key and settings.langfuse_host
    )


def _exporters(settings: Settings) -> list[Any]:
    """One OTLP/HTTP span exporter per configured destination."""
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    out: list[Any] = []
    if settings.otel_exporter_otlp_endpoint:
        endpoint = settings.otel_exporter_otlp_endpoint.rstrip("/")
        if not endpoint.endswith("/v1/traces"):
            endpoint += "/v1/traces"
        out.append(OTLPSpanExporter(endpoint=endpoint))
    if _langfuse_configured(settings):
        token = base64.b64encode(
            f"{settings.langfuse_public_key}:{settings.langfuse_secret_key}".encode()
        ).decode()
        out.append(OTLPSpanExporter(
            endpoint=f"{settings.langfuse_host.rstrip('/')}/api/public/otel/v1/traces",
            headers={"Authorization": f"Basic {token}"},
        ))
    return out


def setup_tracing(settings: Settings) -> bool:
    """Initialise the OTEL pipeline once.

    Returns ``True`` if tracing was initialised, ``False`` otherwise (package absent,
    disabled, nothing to export to, or an init error). Never raises — a tracing
    failure must not stop the service from serving traffic.
    """
    if not settings.tracing_enabled:
        logger.info("tracing_disabled", hint="AGR_TRACING_ENABLED=false")
        return False
    if not _AVAILABLE:
        logger.info(
            "tracing_unavailable",
            hint="install the `tracing` extra (opentelemetry-sdk + OTLP exporter).",
        )
        return False

    try:
        exporters = _exporters(settings)
        if not exporters:
            logger.info("tracing_no_exporter",
                        hint="set AGR_OTEL_EXPORTER_OTLP_ENDPOINT or AGR_LANGFUSE_*")
            return False
        provider = _TracerProvider(resource=_Resource.create({
            "service.name": settings.otel_service_name,
            "service.version": settings.service_version,
            "deployment.environment": settings.environment,
        }))
        for exporter in exporters:
            provider.add_span_processor(_BatchSpanProcessor(exporter))
        _trace_api.set_tracer_provider(provider)
        _disable_openai_agents_sdk_tracing()
        global _INITIALIZED, _PROVIDER
        _PROVIDER = provider
        _INITIALIZED = True
        logger.info(
            "tracing_initialized",
            service=settings.otel_service_name,
            environment=settings.environment,
            exporters={
                "otlp": bool(settings.otel_exporter_otlp_endpoint),
                "langfuse": _langfuse_configured(settings),
            },
        )
        return True
    except Exception:
        # Best-effort: a broken exporter/config must never take the service down.
        logger.exception("tracing_init_failed")
        return False


def observability_state(settings: Settings) -> dict[str, bool]:
    """Resolved exporter wiring, computed purely from ``settings`` — the single source
    of truth for the startup self-check. ``will_export`` is the bottom line: does
    anything leave the process?"""
    langfuse = _langfuse_configured(settings)
    otlp = bool(settings.otel_exporter_otlp_endpoint)
    return {
        "package_installed": _AVAILABLE,
        "tracing_enabled": settings.tracing_enabled,
        "langfuse": langfuse,
        "otlp": otlp,
        "will_export": _AVAILABLE and settings.tracing_enabled and (langfuse or otlp),
    }


def log_observability_state(settings: Settings) -> None:
    """Emit ONE structured ``observability_state`` line at startup, in every case
    (package absent / disabled / wired). Never raises."""
    logger.info("observability_state", **observability_state(settings))


def shutdown_tracing() -> None:
    """Flush + shut down the tracing pipeline (no-op when uninitialised)."""
    global _INITIALIZED, _PROVIDER
    _INITIALIZED = False
    if _PROVIDER is not None:
        try:
            _PROVIDER.shutdown()
        except Exception:
            logger.warning("tracing_shutdown_failed", exc_info=True)
        _PROVIDER = None


@contextmanager
def llm_span(name: str) -> Iterator[Any]:
    """Open a child span around one LLM/embedding call so its latency + cost land on
    their own span. Yields the span, or ``None`` when tracing is unavailable (the
    call then runs with zero tracing overhead)."""
    if not (_AVAILABLE and _INITIALIZED):
        yield None
        return
    with _get_tracer("agr.llm").start_as_current_span(name) as span:
        yield _stamp(span)


def start_llm_span(name: str, *, parent: Any = None) -> Any:
    """Start an LLM child span WITHOUT a ``with`` block, for callers that cannot wrap
    the call: the research agent runs inside the OpenAI-Agents SDK, whose start/end
    hooks fire as separate coroutines, so there is no single scope to wrap.

    Pass ``parent`` to nest the generation under a specific span — the agent's
    per-round span, which the SDK callbacks cannot supply through
    context. Without it the parent is taken from the current OTEL context: the
    request is already inside ``retrieve_span``, so the span still nests under
    the turn. Returns ``None`` when tracing is unavailable. **The caller must
    call ``end_span``**, including on the error path, or the span leaks.
    """
    if not (_AVAILABLE and _INITIALIZED):
        return None
    return _stamp(_get_tracer("agr.llm").start_span(name, context=_parent_context(parent)))


def end_span(span: Any) -> None:
    """End a span started by ``start_llm_span`` (no-op on ``None``). Never raises —
    a tracing failure must not surface as a failed answer."""
    if span is None:
        return
    try:
        span.end()
    except Exception:  # pragma: no cover - defensive
        logger.warning("span_end_failed", exc_info=True)


def _parent_context(parent: Any) -> Any:
    """An OTEL context that nests a new span under *parent*, or ``None`` for
    "use whatever is current". ``None`` when the OTEL API is absent, which the
    tracer treats the same way — so a missing API costs the explicit parent, not
    the span."""
    if parent is None or _trace_api is None:
        return None
    return _trace_api.set_span_in_context(parent)


@contextmanager
def stage_span(name: str | None, *, parent: Any = None, **attrs: Any) -> Iterator[Any]:
    """Trace one unit of NON-LLM work — a pipeline stage, an agent round, a tool
    call, a store read — so the waterfall accounts for the whole request instead
    of only its model calls.

    ``parent`` is an explicit parent SPAN. Pass it whenever the work runs in a
    different asyncio task than the parent: OTEL propagates context through a
    contextvar, and a task started elsewhere (the Agents SDK runs each tool call
    as its own task) inherits the context of ITS creation site, not of the span
    we want to nest under. Without an explicit parent those spans silently
    re-parent to the request root and the hierarchy flattens — the exact bug this
    API exists to prevent. ``None`` ⇒ the current context, which is correct for
    work that runs inline.

    An exception inside the block marks the span ERROR and propagates unchanged
    (OTEL's own context manager does that recording): tracing observes failures,
    it never swallows them. Yields ``None`` when tracing is unavailable, so
    callers need no branching. A falsy ``name`` is also a no-op, which lets a
    caller decide per call whether this unit of work deserves its own span
    without wrapping the ``with`` in a conditional.
    """
    if not (name and _AVAILABLE and _INITIALIZED):
        yield None
        return
    ctx = _parent_context(parent)
    # OTEL's context manager already records a propagating exception and sets
    # ERROR status, so this block adds no explicit recording — doing both wrote
    # the exception event twice.
    with _get_tracer("agr.stage").start_as_current_span(name, context=ctx) as span:
        _stamp(span)
        if attrs:
            set_span_attributes(span, **attrs)
        yield span


def start_stage_span(name: str, *, parent: Any = None, **attrs: Any) -> Any:
    """Non-context-managed twin of ``stage_span`` for spans whose start and end
    are separate callbacks — the agent's per-round span opens in ``on_llm_start``
    and closes when the NEXT round starts, so no ``with`` block can hold it.
    **The caller must call ``end_span``** on every path or the span leaks."""
    if not (_AVAILABLE and _INITIALIZED):
        return None
    ctx = _parent_context(parent)
    span = _stamp(_get_tracer("agr.stage").start_span(name, context=ctx))
    if attrs:
        set_span_attributes(span, **attrs)
    return span


def record_error(span: Any, exc: BaseException, *, code: str | None = None) -> None:
    """Mark *span* failed: OTEL status ERROR + the exception recorded.

    Without this a failed request was indistinguishable from a successful one in
    Langfuse — the trace simply had no output — so "how many requests failed?"
    was unanswerable from traces alone. Best-effort: observability
    must never turn a handled failure into an unhandled one.
    """
    if not (_AVAILABLE and span is not None and _Status is not None):
        return
    try:
        span.set_status(_Status(_StatusCode.ERROR, code or type(exc).__name__))
        span.record_exception(exc)
        if code:
            span.set_attribute("error.code", code)
    except Exception:  # pragma: no cover - defensive
        logger.warning("span_error_record_failed", exc_info=True)


def record_llm(span: Any, **attrs: Any) -> None:
    """Attach standardized ``llm.*`` / ``gen_ai.*`` attributes (model, tokens, cost,
    finish reason) to *span*. No-op when tracing is unavailable or the span is not
    recording. See ``_set_llm_attributes`` for the accepted keys.

    Also emits the Langfuse-native usage/cost attributes: Langfuse otherwise
    prices each generation from its own model registry, which does not know
    self-hosted or NIM model names, so it would show $0.00. Sending its own keys
    makes the trace show OUR computed cost, matching the API response.
    """
    if not (_AVAILABLE and span is not None):
        return
    # Langfuse-only keys, not GenAI semantic-convention attributes.
    cached = attrs.pop("tokens_cached", None)
    params = attrs.pop("model_parameters", None)
    _set_llm_attributes(span, **attrs)
    _set_langfuse_usage(
        span, {**attrs, "tokens_cached": cached, "model_parameters": params}
    )


def _set_langfuse_usage(span: Any, attrs: dict[str, Any]) -> None:
    """Mirror tokens + cost into the attributes Langfuse ingests natively.

    ``usage_details``/``cost_details`` are JSON objects keyed by unit; the keys used
    here (``input``/``output``/``cache_read_input_tokens``/``total``) are Langfuse's
    documented ones. Best-effort: a serialisation problem must never fail a request.
    """
    try:
        usage: dict[str, int] = {}
        if attrs.get("tokens_prompt") is not None:
            usage["input"] = int(attrs["tokens_prompt"])
        if attrs.get("tokens_completion") is not None:
            usage["output"] = int(attrs["tokens_completion"])
        if attrs.get("tokens_cached"):
            usage["cache_read_input_tokens"] = int(attrs["tokens_cached"])
        costs: dict[str, float] = {}
        for key, target in (("cost_input", "input"), ("cost_output", "output"),
                            ("cost_total", "total")):
            if attrs.get(key) is not None:
                costs[target] = float(attrs[key])
        if model := attrs.get("model"):
            span.set_attribute("langfuse.observation.model.name", str(model))
        if usage:
            span.set_attribute("langfuse.observation.usage_details", json.dumps(usage))
        if costs:
            span.set_attribute("langfuse.observation.cost_details", json.dumps(costs))
        # The settings the call ran UNDER. Without these a generation
        # explains what the model returned but not why — and "what did this run
        # with?" is the first question a cost or quality regression raises. Empty
        # dicts are skipped so a parameterless call shows no panel at all rather
        # than an empty one.
        if params := attrs.get("model_parameters"):
            span.set_attribute(
                "langfuse.observation.model.parameters", json.dumps(params)
            )
    except Exception:  # pragma: no cover - observability must never break a request
        logger.warning("langfuse_usage_attributes_failed", exc_info=True)


@contextmanager
def retrieve_span(name: str) -> Iterator[Any]:
    """Open the root span for one retrieve turn (one-shot / SSE stream / resume).

    Opened once per request, before any ``llm_span`` calls, so every LLM/embedding
    child span created during the turn nests under this one span in Langfuse instead
    of each looking like its own trace root — the FastAPI request span can't serve
    that role because ``LangfuseFilteringSpanProcessor`` drops it (its path isn't an
    LLM-gateway path); this span has no ``http.target`` attribute, so it passes the
    same filter that lets ``llm.*`` spans through. Yields the span, or ``None`` when
    tracing is unavailable."""
    if not (_AVAILABLE and _INITIALIZED):
        yield None
        return
    # Automatic exception recording is OFF here, deliberately. The
    # orchestrator distinguishes outcomes that OTEL cannot: a mid-research
    # CLARIFICATION PAUSE also leaves this block by raising, and auto-recording
    # would stamp that normal outcome ERROR — making "how many requests failed?"
    # over-count every paused turn. The orchestrator calls ``record_error`` on
    # the paths that are genuinely failures, and a pause simply leaves the status
    # unset.
    with _get_tracer("agr.retrieve").start_as_current_span(
        name, record_exception=False, set_status_on_exception=False,
    ) as span:
        yield _stamp(span)


def set_span_attributes(span: Any, **attrs: Any) -> None:
    """Attach arbitrary key/value attributes to *span* (``None`` values skipped).
    No-op when tracing is unavailable or *span* is ``None``. Unlike ``record_llm``,
    not restricted to the standardized ``llm.*``/``gen_ai.*`` keys — for
    request-level context (tenant, mode, intent, outcome) that has no SDK-defined
    attribute name."""
    if not (_AVAILABLE and span is not None):
        return
    for key, value in attrs.items():
        if value is not None:
            span.set_attribute(key, value)


# Bound for the root span's Input/Output previews. The OTEL SDK truncates
# attributes anyway; this keeps the trace list light on purpose — the FULL
# answer lives on the synthesis child span.
_ROOT_IO_CHARS = 4000


def record_root_io(span: Any, *, input: str | None = None,
                   output: str | None = None) -> None:
    """Set the ROOT retrieve span's Langfuse Input/Output.

    Langfuse renders a trace's top-level Input/Output from its root
    observation; ours carried neither, so reading a trace meant opening an LLM child. The
    question goes on at open, the answer (or the pause verdict) at close —
    a trace is then scannable from the list. Best-effort: observability must
    never fail a request."""
    if not (_AVAILABLE and span is not None):
        return
    try:
        if input:
            span.set_attribute("langfuse.observation.input",
                               str(input)[:_ROOT_IO_CHARS])
        if output:
            span.set_attribute("langfuse.observation.output",
                               str(output)[:_ROOT_IO_CHARS])
    except Exception:  # pragma: no cover - observability must never break a request
        logger.warning("root_io_attributes_failed", exc_info=True)


def record_applied_options(span: Any, options: dict[str, Any]) -> None:
    """Attach the request's EFFECTIVE RunConfig (``options_applied``) to the
    root retrieve span as ``options.*`` attributes.

    The same dict already rides the response and the ``options_applied`` log
    line; the trace was the one surface missing it, so telling an
    options-tweaked eval run apart from a default run in Langfuse meant a
    request_id cross-lookup into the logs. Scalars land as native attribute
    values (filterable in the trace UI); nested values (the ``run`` sources
    meta, ``ignored``) are JSON-encoded — OTEL attributes hold primitives
    only. Best-effort: observability must never fail a request."""
    if not (_AVAILABLE and span is not None):
        return
    try:
        for key, value in options.items():
            if value is None:
                continue
            if isinstance(value, str | bool | int | float):
                span.set_attribute(f"options.{key}", value)
            else:
                span.set_attribute(
                    f"options.{key}",
                    json.dumps(value, ensure_ascii=False, default=str),
                )
    except Exception:  # pragma: no cover - observability must never break a request
        logger.warning("applied_options_attributes_failed", exc_info=True)


def record_trace_identity(
    span: Any, *, session_id: str | None = None, user_id: str | None = None,
    tags: list[str] | None = None,
) -> None:
    """Set Langfuse's OWN trace fields — session, user, tags.

    These are not decoration: Langfuse's UI groups and filters by them. A
    ``session`` turns a clarification pause and its resume into one readable
    conversation instead of two unrelated traces; ``tags`` make "every failed
    Super Research run" a filter click rather than an attribute search. We were
    sending the same facts as generic attributes, which the UI cannot use.
    """
    if not (_AVAILABLE and span is not None):
        return
    try:
        if session_id:
            span.set_attribute("langfuse.session.id", str(session_id))
        if user_id:
            span.set_attribute("langfuse.user.id", str(user_id))
        if tags:
            # Langfuse reads tags as a JSON array on this key.
            span.set_attribute("langfuse.trace.tags", json.dumps(sorted(set(tags))))
    except Exception:  # pragma: no cover - observability must never break a request
        logger.warning("trace_identity_attributes_failed", exc_info=True)


def record_scores(span: Any, scores: dict[str, Any]) -> None:
    """Attach per-request QUALITY signals the service already computes, as
    ``langfuse.score.*`` attributes.

    This is what turns answer quality into a dashboard instead of a log grep:
    "% of runs shipping zero citations", "% hitting the round cap", "% paused".
    Values are numbers or booleans — anything else is skipped rather than
    stringified, because a score that is not comparable is not a score."""
    if not (_AVAILABLE and span is not None):
        return
    try:
        for name, value in scores.items():
            if isinstance(value, bool):
                span.set_attribute(f"langfuse.score.{name}", int(value))
            elif isinstance(value, (int, float)):
                span.set_attribute(f"langfuse.score.{name}", value)
    except Exception:  # pragma: no cover - observability must never break a request
        logger.warning("score_attributes_failed", exc_info=True)


def record_io(span: Any, *, input: str | None = None, output: str | None = None,
              limit: int = _ROOT_IO_CHARS) -> None:
    """Set a span's Langfuse Input/Output — the same two fields the root span and
    every generation carry, for spans that are neither (tool calls, stages).

    A tool span with Input/Output is what makes "which tool, with what
    arguments, returning what" answerable from the span itself rather than by
    pairing a round's output with the next round's input. Bounded:
    a tool result can be tens of KB and a trace is for reading."""
    if not (_AVAILABLE and span is not None):
        return
    try:
        if input:
            span.set_attribute("langfuse.observation.input", str(input)[:limit])
        if output:
            span.set_attribute("langfuse.observation.output", str(output)[:limit])
    except Exception:  # pragma: no cover - observability must never break a request
        logger.warning("span_io_attributes_failed", exc_info=True)


def _disable_openai_agents_sdk_tracing() -> None:
    """Suppress the OpenAI-Agents-SDK built-in tracing (the research agent uses that
    SDK). It POSTs traces to api.openai.com by default — noise here, and it would fail
    without a real OpenAI key. Our own LLM spans cover those calls."""
    try:
        from agents import set_tracing_disabled

        set_tracing_disabled(True)
        logger.debug("openai_agents_sdk_tracing_disabled")
    except ImportError:
        pass
    except Exception:
        logger.warning("openai_agents_sdk_tracing_disable_failed", exc_info=True)
