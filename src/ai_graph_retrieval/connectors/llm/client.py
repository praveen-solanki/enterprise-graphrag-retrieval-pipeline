"""LLM adapter — the only module that talks to the model providers.

One OpenAI-compatible endpoint family serves everything — NVIDIA NIM
(``integrate.api.nvidia.com``), a self-hosted vLLM, or any server speaking the
OpenAI protocol:
  * chat/reasoning → via **LiteLLM** (``openai/<model>`` at ``llm_base_url``).
    Callers pass OpenAI-shaped ``messages`` / ``tools`` / ``tool_choice`` and get
    OpenAI-shaped responses back. Two tiers: ``llm_model`` (heavy) and
    ``fast_model`` (lightweight decisions).
  * embeddings → the OpenAI SDK at ``embedding_base_url``. With
    ``embedding_input_type`` set (NIM's asymmetric embed models) the request
    carries ``input_type`` and omits ``dimensions``; empty ⇒ plain OpenAI shape
    with an explicit ``dimensions``.

A semaphore protects provider quota from bursts; transient failures retry with
exponential jitter (LiteLLM raises OpenAI-compatible exception types, so the same
``_TRANSIENT`` set applies); every call records tokens into the request's Usage.
"""
from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Any

import litellm
import structlog
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from ai_graph_retrieval.core.costing import CostRates, call_cost_usd, embedding_cost_usd
from ai_graph_retrieval.core.exceptions import LlmUnavailable
from ai_graph_retrieval.core.models import Usage
from ai_graph_retrieval.observability.trace_log import prompt_metadata
from ai_graph_retrieval.observability.tracing import llm_span, record_llm
from ai_graph_retrieval.settings import Settings

logger = structlog.get_logger(__name__)

# LiteLLM maps HTTP 502/503 to its own ServiceUnavailableError/BadGatewayError,
# whose MROs bypass openai.InternalServerError — include them or the most common
# transient gateway failures get zero retries.
_TRANSIENT = (
    RateLimitError, APITimeoutError, APIConnectionError, InternalServerError,
    litellm.exceptions.ServiceUnavailableError, litellm.exceptions.BadGatewayError,
)

# LiteLLM: don't phone home, don't print its own tracebacks, and silently drop
# any provider-unsupported params rather than 400.
litellm.telemetry = False
litellm.suppress_debug_info = True
litellm.drop_params = True
# LiteLLM's own retry handles the provider's 429 rate-limit with backoff so a
# burst paces instead of failing. Needed on both paths that
# bypass or under-cover LlmClient's tenacity: the research agent (calls litellm
# directly via the OpenAI-Agents SDK) and streamed synthesis (a 429 arrives BEFORE
# any token, so re-issuing the request can't duplicate output). complete()'s
# tenacity still adds a second safety layer — harmless for transient errors.
litellm.num_retries = 6

_DEFAULT_MAX_TOKENS = 4096


def litellm_model_name(model: str) -> str:
    """The LiteLLM model string for an OpenAI-compatible endpoint. The
    ``openai/`` prefix selects LiteLLM's OpenAI-protocol adapter; the rest is
    sent verbatim as the request's ``model`` (e.g. ``nvidia/nemotron-...``)."""
    return f"openai/{model}"


# Bounds for the embedding span's Input. A batch can be hundreds of document
# titles, so trace both ends of the size problem: how MANY texts are shown, and
# how much of each. The OTEL SDK truncates over-long attributes anyway; these
# bounds keep the value readable rather than merely legal.
_TRACE_EMBED_MAX_TEXTS = 10
_TRACE_EMBED_TEXT_CHARS = 200


def _embed_trace_input(texts: list[str]) -> str:
    """The embedded text, bounded, as the span's Input.

    Answers "which text produced this retrieval?" — the reason an llm.embed
    observation gets opened. Long texts are cut and a batch is capped, with the
    omission stated rather than silent, so nobody reads a trimmed list as complete.
    """
    shown = [
        (t or "")[:_TRACE_EMBED_TEXT_CHARS] for t in texts[:_TRACE_EMBED_MAX_TEXTS]
    ]
    if len(texts) > _TRACE_EMBED_MAX_TEXTS:
        shown.append(f"… +{len(texts) - _TRACE_EMBED_MAX_TEXTS} more not shown")
    # Wrapped as ONE chat message: trace UIs render request_messages through a
    # chat renderer, and a bare string array shows as rows of "undefined"
    # role/content. One user message with the texts as a list renders like
    # every chat span.
    content = "\n".join(f"- {t}" for t in shown)
    return json.dumps([{"role": "user", "content": content}], ensure_ascii=False)


def trace_model_params(
    params: dict[str, Any],
    tools: list[dict] | None = None,
    tool_choice: str | dict | None = None,
) -> dict[str, Any]:
    """The settings this call ran under, for the generation's model-parameters
    panel. Read from the params dict actually sent, so it cannot drift from
    the request the way a hand-maintained copy would.

    Only knobs that change the OUTPUT are included — endpoint, api key and
    timeout describe transport, not behavior, and a key must
    never reach a trace. A forced function is named because "which tool was it
    pinned to" is the first question a malformed structured output raises.
    """
    out: dict[str, Any] = {}
    for key in ("max_tokens", "temperature", "top_p"):
        if params.get(key) is not None:
            out[key] = params[key]
    if tools:
        out["tools_offered"] = len(tools)
    if isinstance(tool_choice, dict):
        if name := (tool_choice.get("function") or {}).get("name"):
            out["forced_function"] = name
    elif tool_choice:
        out["tool_choice"] = str(tool_choice)
    return out


def _chat_span_name(purpose: str | None) -> str:
    """The generation's name in the trace. A purpose makes the span self-describing
    ("llm.chat.entry_classification"); without one the call stays the generic
    "llm.chat", so an untagged caller degrades to today's behavior rather than
    breaking the trace."""
    return f"llm.chat.{purpose}" if purpose else "llm.chat"


# Prompts keep their stable blocks first and per-request content last: servers
# with automatic prefix caching (vLLM, NIM) reuse a repeated prefix, and one
# changed byte early invalidates everything after it.

# Appended after a length-truncated streaming round to continue the answer (see
# stream_text's completeness guard). Kept terse to avoid restating prior content.
_CONTINUE_NUDGE = (
    "Continue the answer from exactly where it stopped. Do not repeat any text you "
    "already wrote or restate earlier headings — continue seamlessly."
)


class LlmClient:
    def __init__(self, settings: Settings, semaphore) -> None:
        self._s = settings
        self._sem = semaphore
        # Rates for per-call span cost enrichment — same table as the per-request
        # UsageReport costing, so span cost never diverges from the request total.
        self._rates = CostRates.from_settings(settings)
        # Embeddings client. Optional: without an endpoint chat still works and
        # every embedding call fails as LlmUnavailable.
        base_url = settings.embedding_base_url or settings.llm_base_url
        self._embeddings_client: AsyncOpenAI | None = None
        if base_url:
            self._embeddings_client = AsyncOpenAI(
                base_url=base_url,
                # Local vLLM needs no key; the SDK still requires a value.
                api_key=settings.embedding_api_key or settings.llm_api_key or "none",
                timeout=settings.llm_timeout_s,
                max_retries=0,  # we own retries (tenacity) for observability
            )

    def with_settings(self, settings: Settings) -> LlmClient:
        """A per-request view with overridden settings that SHARES the
        concurrency semaphore and the embeddings client — chat is stateless per
        call (LiteLLM builds params from settings), so only ``_s`` differs.
        Cheap: no client is rebuilt."""
        clone = LlmClient.__new__(LlmClient)
        clone._s = settings
        clone._sem = self._sem
        clone._rates = CostRates.from_settings(settings)
        clone._embeddings_client = self._embeddings_client
        return clone

    def _retrying(self) -> AsyncRetrying:
        return AsyncRetrying(
            retry=retry_if_exception_type(_TRANSIENT),
            wait=wait_random_exponential(multiplier=1, max=20),
            stop=stop_after_attempt(self._s.llm_max_retries),
            reraise=True,
        )

    def _deployment(self, fast: bool) -> str:
        """The model for this call — what metering, spans and cost bucketing
        group by: the fast model for latency-sensitive lightweight decisions,
        the heavy model for reasoning + synthesis."""
        return self._s.fast_model if fast else self._s.llm_model

    def _chat_params(self, max_tokens: int | None, fast: bool = False) -> dict[str, Any]:
        """LiteLLM params for one chat call. ``max_tokens`` always has a value
        (explicit → configured floor → hard default)."""
        cap = max_tokens or self._s.llm_max_output_tokens or _DEFAULT_MAX_TOKENS
        params: dict[str, Any] = {
            "model": litellm_model_name(self._deployment(fast)),
            "api_base": self._s.llm_base_url,
            # Local vLLM needs no key; LiteLLM still requires a value.
            "api_key": self._s.llm_api_key or "none",
            "max_tokens": cap,
            "timeout": self._s.llm_timeout_s,
        }
        if self._s.llm_temperature is not None and self._s.llm_send_temperature:
            params["temperature"] = self._s.llm_temperature
        return params

    def _record(self, usage: Usage | None, usage_obj: Any, deployment: str) -> None:
        if not (usage and usage_obj):
            return
        usage.add_llm(
            getattr(usage_obj, "prompt_tokens", 0) or 0,
            getattr(usage_obj, "completion_tokens", 0) or 0,
            deployment=deployment,
            cached_prompt_tokens=self._cached_prompt_tokens(usage_obj),
        )

    def _enrich_span(
        self, span: Any, deployment: str, tier: str, prompt_tokens: int,
        completion_tokens: int, cached: int, finish_reason: str | None,
        *, request_messages: list[dict] | None = None, response_text: str | None = None,
        model_parameters: dict[str, Any] | None = None,
    ) -> None:
        """Attach model/token/cost/finish attrs to the LLM span. The span already
        carries latency (its own duration); this adds the cost + usage. No-op — and
        no cost math — when tracing is disabled (``span is None``).

        ``request_messages``/``response_text`` (chat calls only — never passed for
        embeddings) become the trace's Input/Output for this observation; the OTEL SDK
        truncates any value past ``span_attribute_value_length_limit`` (see
        ``TracingConfig``), so no manual length cap is needed here."""
        if span is None:
            return
        cost_in, cost_out, cost_total = call_cost_usd(
            self._rates, deployment, prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens, cached_prompt_tokens=cached,
        )
        record_llm(
            span, model=deployment, provider="openai", request_type=tier,
            tokens_prompt=prompt_tokens, tokens_completion=completion_tokens,
            # Cache reads explain a call whose cost is far below its token count.
            tokens_cached=cached,
            cost_input=cost_in, cost_output=cost_out, cost_total=cost_total,
            finish_reason=finish_reason,
            request_messages=(
                json.dumps(request_messages, ensure_ascii=False) if request_messages else None
            ),
            response_choices=response_text or None,
            # The settings the call ran under — rendered as the generation's
            # model-parameters panel.
            model_parameters=model_parameters or None,
        )

    @staticmethod
    def _response_text(msg: Any) -> str:
        """Text for the trace Output field: the assistant's content, or a compact
        summary of tool calls when a tool call replaced free text (``content`` is
        ``None`` on a pure tool-call turn)."""
        if msg.content:
            return msg.content
        tool_calls = getattr(msg, "tool_calls", None) or []
        if tool_calls:
            return json.dumps(
                [
                    {"name": tc.function.name, "arguments": tc.function.arguments}
                    for tc in tool_calls
                ],
                ensure_ascii=False,
            )
        return ""

    @staticmethod
    def _cached_prompt_tokens(usage_obj: Any) -> int:
        """Cache-READ tokens (prompt tokens served from cache). LiteLLM
        surfaces them either on usage.prompt_tokens_details.cached_tokens or directly
        as usage.cache_read_input_tokens depending on version — read both."""
        details = getattr(usage_obj, "prompt_tokens_details", None)
        via_details = int(getattr(details, "cached_tokens", 0) or 0) if details else 0
        via_direct = int(getattr(usage_obj, "cache_read_input_tokens", 0) or 0)
        return max(via_details, via_direct)

    # ── chat (OpenAI-compatible, via LiteLLM) ────────────────────────────────

    async def complete(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        max_tokens: int | None = None,
        fast: bool = False,
        usage: Usage | None = None,
        purpose: str | None = None,
    ):
        """One chat completion; returns the assistant message (OpenAI-shaped).

        ``purpose`` NAMES the generation in the trace. Every single-pass caller
        used to land on an identically-named ``llm.chat`` span, so a trace with
        entry classification, the out-of-scope probe, neighbour re-ranking and
        the action proposer showed four indistinguishable
        siblings — the decision flow was only recoverable by opening each one
        and reading its prompt. Callers pass their own name.

        ``fast=True`` routes to the fast/cheap model for lightweight decisions.
        """
        deployment = self._deployment(fast)
        tier = "fast" if fast else "heavy"
        logger.debug("llm_request", deployment=deployment, tier=tier,
                     tools_offered=len(tools or []), **prompt_metadata(messages))
        t0 = time.monotonic()
        with llm_span(_chat_span_name(purpose)) as span:
            try:
                async for attempt in self._retrying():
                    with attempt:
                        async with self._sem:
                            response = await litellm.acompletion(
                                messages=messages,
                                **self._chat_params(max_tokens, fast),
                                **({"tools": tools} if tools else {}),
                                **({"tool_choice": tool_choice} if tool_choice else {}),
                            )
            except _TRANSIENT as exc:
                logger.error("llm_exhausted_retries", deployment=deployment,
                             latency_ms=int((time.monotonic() - t0) * 1000), error=str(exc))
                raise LlmUnavailable() from exc
            except APIStatusError as exc:
                logger.error("llm_status_error", deployment=deployment, status=exc.status_code,
                             latency_ms=int((time.monotonic() - t0) * 1000), error=str(exc))
                raise LlmUnavailable() from exc

            self._record(usage, response.usage, deployment)
            u = response.usage
            cached = self._cached_prompt_tokens(u) if u else 0
            prompt_tokens = (getattr(u, "prompt_tokens", 0) or 0) if u else 0
            completion_tokens = (getattr(u, "completion_tokens", 0) or 0) if u else 0
            if not response.choices:
                # An empty choices list is an unusable response — surface it as an LLM
                # failure rather than an IndexError on the next line (the previous line
                # already admits choices can be empty).
                raise LlmUnavailable()
            choice = response.choices[0]
            finish_reason = choice.finish_reason
            msg = choice.message
            self._enrich_span(span, deployment, tier, prompt_tokens, completion_tokens,
                              cached, finish_reason, request_messages=messages,
                              response_text=self._response_text(msg),
                              model_parameters=trace_model_params(
                                  self._chat_params(max_tokens, fast),
                                  tools, tool_choice))
            logger.info(
                "llm_call", deployment=deployment, tier=tier,
                latency_ms=int((time.monotonic() - t0) * 1000),
                prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                cached_prompt_tokens=cached, cache_hit=cached > 0,
                finish_reason=finish_reason,
                tool_call=bool(getattr(msg, "tool_calls", None)),
            )
            return msg

    async def stream_text(
        self,
        messages: list[dict],
        *,
        max_tokens: int | None = None,
        fast: bool = False,
        usage: Usage | None = None,
        purpose: str | None = None,
    ) -> AsyncIterator[str]:
        """Token-stream a plain (tool-free) completion.

        ``purpose`` names the generation, exactly as on ``complete()``. Without one
        the span stays ``llm.chat.stream`` — a name describing how the call was
        transported rather than what it was for, which left the single most
        expensive observation in a request unlabelled.

        Completeness guard (POC parity): if a round stops because it hit the
        max_tokens cap (finish_reason == "length"), transparently continue for up
        to ``synthesis_max_continuations`` more rounds — each round gets a fresh
        max_tokens budget — so a long answer is never cut off mid-sentence. No
        retries: a failure (before or after first token) maps to LlmUnavailable
        and the orchestrator degrades; replaying a half-streamed answer would
        duplicate output.
        """
        deployment = self._deployment(fast)
        tier = "fast" if fast else "heavy"
        convo = list(messages)
        answer_so_far = ""
        rounds_left = self._s.synthesis_max_continuations
        # Accumulate this streamed call's usage across continuation rounds (each round
        # is a separately-billed call) for span cost enrichment.
        span_prompt = span_completion = span_cached = 0
        finish_reason: str | None = None
        logger.debug("llm_stream_request", deployment=deployment, **prompt_metadata(messages))
        t0 = time.monotonic()
        with llm_span(_chat_span_name(purpose) if purpose else "llm.chat.stream") as span:
            # Retries for a round that fails BEFORE emitting anything. Providers
            # return intermittent 500s, and a research turn makes many calls, so
            # the chance that one of them is unlucky grows with the length of the
            # turn. Re-issuing is
            # safe only while no token has reached the client; once text is out,
            # replaying would duplicate it, so that case still degrades (below).
            attempts_left = max(0, self._s.llm_max_retries - 1)
            try:
                while True:
                    finish_reason = None
                    produced = ""
                    round_usage = None
                    try:
                        async with self._sem:
                            stream = await litellm.acompletion(
                                messages=convo,
                                stream=True,
                                stream_options={"include_usage": True},
                                **self._chat_params(max_tokens, fast),
                            )
                            async for chunk in stream:
                                if getattr(chunk, "usage", None):
                                    round_usage = chunk.usage
                                if not chunk.choices:
                                    continue
                                choice = chunk.choices[0]
                                if choice.delta.content:
                                    produced += choice.delta.content
                                    yield choice.delta.content
                                if choice.finish_reason:
                                    finish_reason = choice.finish_reason
                    except (*_TRANSIENT, APIStatusError) as exc:
                        # Re-issue ONLY while the client has seen nothing: no token has
                        # been yielded this round and no earlier round produced text.
                        # Otherwise fall through to the handlers below, which degrade.
                        if attempts_left and not answer_so_far and not produced:
                            attempts_left -= 1
                            delay = min(2.0 * (self._s.llm_max_retries - attempts_left), 8.0)
                            logger.warning(
                                "llm_stream_retry_before_first_token",
                                deployment=deployment, attempts_left=attempts_left,
                                delay_s=delay, error=str(exc),
                            )
                            await asyncio.sleep(delay)
                            continue
                        raise
                    if round_usage is not None:
                        self._record(usage, round_usage, deployment)
                        span_prompt += getattr(round_usage, "prompt_tokens", 0) or 0
                        span_completion += getattr(round_usage, "completion_tokens", 0) or 0
                        span_cached += self._cached_prompt_tokens(round_usage)
                    answer_so_far += produced
                    # Continue only when the cap truncated the round AND budget remains
                    # AND the round produced text (guards against a no-progress loop).
                    if finish_reason != "length" or rounds_left <= 0 or not produced:
                        break
                    rounds_left -= 1
                    logger.info("stream_continuation", chars=len(answer_so_far),
                                remaining=rounds_left)
                    convo = [
                        *messages,
                        {"role": "assistant", "content": answer_so_far},
                        {"role": "user", "content": _CONTINUE_NUDGE},
                    ]
                self._enrich_span(span, deployment, tier, span_prompt, span_completion,
                                  span_cached, finish_reason, request_messages=messages,
                                  response_text=answer_so_far,
                                  model_parameters=trace_model_params(
                                      self._chat_params(max_tokens, fast)))
                logger.info(
                    "llm_stream_done", deployment=deployment,
                    latency_ms=int((time.monotonic() - t0) * 1000),
                    answer_chars=len(answer_so_far),
                    continuation_rounds=self._s.synthesis_max_continuations - rounds_left,
                )
            except _TRANSIENT as exc:
                logger.error("llm_stream_failed", deployment=deployment,
                             latency_ms=int((time.monotonic() - t0) * 1000), error=str(exc))
                raise LlmUnavailable() from exc
            except APIStatusError as exc:
                logger.error("llm_stream_status_error", status=exc.status_code, error=str(exc))
                raise LlmUnavailable() from exc

    # ── embeddings (OpenAI-compatible) ───────────────────────────────────────

    def _embed_params(self, dim: int) -> dict[str, Any]:
        """Request fields beyond model/input. NIM's asymmetric embed models take
        ``input_type`` ("query" here; ingestion embeds with "passage") and reject
        the OpenAI ``dimensions`` field; plain OpenAI-protocol servers take an
        explicit ``dimensions``."""
        if self._s.embedding_input_type:
            return {"extra_body": {"input_type": self._s.embedding_input_type,
                                   "truncate": "END"}}
        return {"dimensions": dim}

    async def embed(
        self,
        texts: list[str],
        *,
        dimensions: int | None = None,
        model: str | None = None,
        usage: Usage | None = None,
    ) -> list[list[float]]:
        # The query vector must match the dimension of the index being searched —
        # a mismatch silently collapses ANN recall.
        #
        # ``model`` overrides the configured model (per-request run selection —
        # a resolved run's embedding space); None ⇒ the configured default.
        if self._embeddings_client is None:
            logger.error("embedding_unavailable_no_endpoint")
            raise LlmUnavailable()
        dim = dimensions if dimensions is not None else self._s.embedding_dimensions
        deployment = model or self._s.embedding_model
        t0 = time.monotonic()
        with llm_span("llm.embed") as span:
            try:
                async for attempt in self._retrying():
                    with attempt:
                        async with self._sem:
                            response = await self._embeddings_client.embeddings.create(
                                model=deployment,
                                input=texts,
                                encoding_format="float",
                                **self._embed_params(dim),
                            )
            except _TRANSIENT as exc:
                logger.error("embedding_exhausted_retries",
                             deployment=deployment,
                             dim=dim, inputs=len(texts),
                             latency_ms=int((time.monotonic() - t0) * 1000), error=str(exc))
                raise LlmUnavailable() from exc
            except APIStatusError as exc:
                logger.error("embedding_status_error", deployment=deployment,
                             status=exc.status_code, error=str(exc))
                raise LlmUnavailable() from exc
            tokens = (response.usage.prompt_tokens if response.usage else 0) or 0
            if usage:
                usage.add_embedding(tokens)
            if span is not None:
                record_llm(
                    span, model=deployment, request_type="embedding",
                    tokens_prompt=tokens, tokens_total=tokens,
                    cost_total=embedding_cost_usd(self._rates, tokens),
                    # Input/Output were blank on every llm.embed observation, which
                    # made a trace unusable for the question it is most often opened
                    # to answer: which text produced this retrieval. Chat spans have
                    # always carried their prompt, so this is consistency, not a new
                    # exposure. The vector itself is never sent — thousands of
                    # floats are unreadable and would bloat every trace — only its shape.
                    request_messages=_embed_trace_input(texts),
                    response_choices=f"{len(response.data)} vector(s) x {dim}d",
                )
            logger.info(
                "embedding_call", deployment=deployment, dim=dim,
                inputs=len(texts), tokens=tokens,
                latency_ms=int((time.monotonic() - t0) * 1000),
            )
            return [item.embedding for item in response.data]
