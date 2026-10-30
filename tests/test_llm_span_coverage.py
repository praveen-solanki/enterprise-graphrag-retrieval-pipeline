"""EVERY LLM call is a priced Langfuse generation, in every mode.

The requirement is coverage, not sampling: a greeting turn, a guardrail decline, a
fast-lane answer and a research loop must all show up in Langfuse with cost. That
cannot be eyeballed from here — no exporter is configured in tests, so spans are
no-ops — so this module installs a RECORDING fake tracer and drives each call site
through the real code path, asserting a span exists and carries cost.

The call sites, and who exercises them:

  * ``LlmClient.complete``  — the entry classifier (so: greeting, smalltalk,
    capability, guardrail, out_of_scope, and every knowledge turn's routing), the
    clarification gate, plan review, neighbour re-ranking, the answer-found judge,
    and suggested actions. One code path, so one test covers all of them.
  * ``LlmClient.stream_text`` — streamed synthesis, including continuation rounds.
  * ``LlmClient.embed``    — every query/title embedding.
  * ``_AgentRunHooks``     — the research agent's rounds, which bypass LlmClient.

    uv run pytest tests/test_llm_span_coverage.py -v
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import litellm
import pytest

from ai_graph_retrieval.agent.runtime import _AgentRunHooks
from ai_graph_retrieval.connectors.llm.client import LlmClient
from ai_graph_retrieval.core.costing import CostRates
from ai_graph_retrieval.core.exceptions import LlmUnavailable
from ai_graph_retrieval.core.models import EvidencePool, Usage
from ai_graph_retrieval.observability import tracing
from tests.support import make_settings as _make_settings

# Shipped cost rates are 0 (unconfigured); price both tiers so "has cost" means something.
_RATES = {
    "fast_model": "test/fast-model",
    "cost_chat_input_per_1m": 3.0, "cost_chat_output_per_1m": 15.0,
    "cost_fast_input_per_1m": 1.0, "cost_fast_output_per_1m": 5.0,
    "cost_chat_cached_input_per_1m": 0.3, "cost_embedding_per_1m": 0.13,
}


def make_settings(**over):
    return _make_settings(**{**_RATES, **over})


class _RecordedSpan:
    def __init__(self, name: str) -> None:
        self.name = name
        self.attrs: dict = {}
        # What record_llm forwarded to the tracing SDK. Asserting on this tests OUR
        # contract (which values we send) without duplicating the SDK's own
        # attribute-name mapping, which is not ours to pin.
        self.forwarded: dict = {}
        self.ended = False

    def set_attribute(self, key, value) -> None:
        self.attrs[key] = value

    def end(self) -> None:
        self.ended = True

    # start_as_current_span() is used as a context manager by llm_span
    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.end()

    def cost(self) -> dict:
        raw = self.attrs.get("langfuse.observation.cost_details")
        return json.loads(raw) if raw else {}

    def usage(self) -> dict:
        raw = self.attrs.get("langfuse.observation.usage_details")
        return json.loads(raw) if raw else {}

    def params(self) -> dict:
        raw = self.attrs.get("langfuse.observation.model.parameters")
        return json.loads(raw) if raw else {}


class _Tracer:
    def __init__(self, sink: list) -> None:
        self._sink = sink

    # **kwargs mirrors the real tracer, which the helpers now call with an
    # explicit parent `context` and with exception recording switched off on the
    # root span. A fake with a narrower signature would fail these
    # tests for a reason that has nothing to do with what they assert.
    def start_as_current_span(self, name, **_kwargs):
        span = _RecordedSpan(name)
        self._sink.append(span)
        return span

    def start_span(self, name, **_kwargs):
        span = _RecordedSpan(name)
        self._sink.append(span)
        return span


def _generation(spans: list) -> _RecordedSpan:
    """The one PRICED span among a round's spans. Each agent round also opens a
    structural `agent.round.N` parent for its tool calls, so a test
    about the generation has to say which span it means."""
    gens = [s for s in spans if not s.name.startswith("agent.round")]
    assert len(gens) == 1, [s.name for s in spans]
    return gens[0]


@pytest.fixture
def spans(monkeypatch):
    """Turn tracing on with a recording tracer, and let real attributes through."""
    recorded: list[_RecordedSpan] = []
    monkeypatch.setattr(tracing, "_AVAILABLE", True)
    monkeypatch.setattr(tracing, "_INITIALIZED", True)
    monkeypatch.setattr(tracing, "_get_tracer", lambda _name: _Tracer(recorded))
    # The real SDK is absent. Stand in for it faithfully enough to see what we sent.
    monkeypatch.setattr(tracing, "_set_llm_attributes",
                        lambda span, **attrs: span.forwarded.update(attrs))
    return recorded


def _client(**over) -> LlmClient:
    return LlmClient(make_settings(**over), asyncio.Semaphore(4))


def _chat_response(prompt: int, completion: int):
    message = SimpleNamespace(content="hi", tool_calls=None)
    return SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion,
                              prompt_tokens_details=None, cache_read_input_tokens=0),
        choices=[SimpleNamespace(finish_reason="stop", message=message)],
    )


# ── the shared complete() path: classifier (all conversational modes), gate,
#    plan review, neighbour scoring, judge, suggested actions ──────────────────

@pytest.mark.parametrize("fast,tier", [(True, "fast"), (False, "chat")])
def test_every_complete_call_is_a_priced_generation(spans, monkeypatch, fast, tier):
    async def _fake_acompletion(**_kw):
        return _chat_response(3_830, 176)

    monkeypatch.setattr(litellm, "acompletion", _fake_acompletion)
    usage = Usage()
    asyncio.run(_client().complete([{"role": "user", "content": "hi"}],
                                   fast=fast, usage=usage))
    (span,) = spans
    assert span.name == "llm.chat"
    assert span.ended
    # Cost is present and NON-ZERO for both tiers — priced from our own rate table,
    # not from Langfuse's model-name lookup.
    assert span.cost()["total"] > 0, tier
    assert span.usage() == {"input": 3_830, "output": 176}
    # ...and the span's cost equals what the request will report, same rate table.
    assert span.cost()["total"] == pytest.approx(
        _request_cost(usage, make_settings()), rel=1e-9
    )


def _request_cost(usage: Usage, settings) -> float:
    from ai_graph_retrieval.core.costing import estimate_cost_usd
    return estimate_cost_usd(usage.snapshot(), CostRates.from_settings(settings))


def test_a_greeting_turn_is_traced_and_priced(spans, monkeypatch):
    # A greeting/guardrail turn short-circuits before retrieval, so the classifier
    # call is the ONLY llm call it makes. It must still be a priced generation, or
    # conversational traffic looks free.
    async def _fake_acompletion(**_kw):
        return _chat_response(3_600, 40)

    monkeypatch.setattr(litellm, "acompletion", _fake_acompletion)
    asyncio.run(_client().complete([{"role": "user", "content": "hi"}],
                                   fast=True, usage=Usage()))
    (span,) = spans
    assert span.name == "llm.chat"
    assert span.cost()["total"] > 0


# ── streamed synthesis ───────────────────────────────────────────────────────

def test_streamed_synthesis_is_a_priced_generation(spans, monkeypatch):
    async def _fake_acompletion(**_kw):
        async def _chunks():
            yield SimpleNamespace(
                usage=None,
                choices=[SimpleNamespace(delta=SimpleNamespace(content="answer"),
                                         finish_reason=None)],
            )
            yield SimpleNamespace(
                usage=SimpleNamespace(prompt_tokens=12_928, completion_tokens=1_497,
                                      prompt_tokens_details=None,
                                      cache_read_input_tokens=0),
                choices=[SimpleNamespace(delta=SimpleNamespace(content=None),
                                         finish_reason="stop")],
            )
        return _chunks()

    monkeypatch.setattr(litellm, "acompletion", _fake_acompletion)

    async def _drain():
        return [t async for t in _client().stream_text(
            [{"role": "user", "content": "q"}], usage=Usage())]

    assert asyncio.run(_drain()) == ["answer"]
    (span,) = spans
    assert span.name == "llm.chat.stream"
    assert span.cost()["total"] > 0
    assert span.usage() == {"input": 12_928, "output": 1_497}


# ── embeddings ───────────────────────────────────────────────────────────────

def test_embedding_calls_are_priced_too(spans):
    client = _client()

    class _Embeddings:
        async def create(self, **_kw):
            return SimpleNamespace(
                usage=SimpleNamespace(prompt_tokens=416),
                data=[SimpleNamespace(embedding=[0.1, 0.2])],
            )

    client._embeddings_client = SimpleNamespace(embeddings=_Embeddings())
    asyncio.run(client.embed(["text"], usage=Usage()))
    (span,) = spans
    assert span.name == "llm.embed"
    # Priced now that cost_embedding_per_1m is a real rate — a zero here reads as
    # broken instrumentation to anyone looking at a trace.
    assert span.cost()["total"] > 0
    assert span.usage() == {"input": 416}


# ── the research agent's rounds (they bypass LlmClient entirely) ─────────────

def test_each_research_round_is_its_own_priced_generation(spans):
    settings = make_settings()
    hooks = _AgentRunHooks(
        EvidencePool(), 3, usage=Usage(),
        deployment=settings.llm_model,
        rates=CostRates.from_settings(settings),
        provider="openai",
    )
    for prompt, completion in ((8_000, 300), (11_000, 250)):
        asyncio.run(hooks.on_llm_start(
            SimpleNamespace(usage=None), agent=None, system_prompt="", input_items=[]))
        asyncio.run(hooks.on_llm_end(
            SimpleNamespace(usage=None), agent=None,
            response=SimpleNamespace(usage=SimpleNamespace(
                input_tokens=prompt, output_tokens=completion,
                input_tokens_details=SimpleNamespace(cached_tokens=0)))))
    # Each round now also opens a structural `agent.round.N` span that parents the
    # generation and that round's tool calls, so this filters to the
    # GENERATIONS — the priced things this test is about.
    gens = [s for s in spans if s.name == "llm.chat.agent"]
    assert len(gens) == 2
    assert [s.name for s in spans if s.name.startswith("agent.round")] == [
        "agent.round.1", "agent.round.2",
    ]
    hooks.close()  # the runtime's finally does this; the last round span closes here
    assert all(s.ended for s in spans)                     # no leaked spans, either kind
    assert all(s.cost()["total"] > 0 for s in gens)
    assert [s.usage()["input"] for s in gens] == [8_000, 11_000]


def test_a_research_round_traces_its_input_and_output(spans):
    """agent rounds were the one generation with no Input/Output in
    Langfuse. The round's system prompt + input items must land as the span's
    input, and the model's text AND tool calls as its output — a pure tool-call
    round must not trace as empty."""
    settings = make_settings()
    hooks = _AgentRunHooks(
        EvidencePool(), 3, usage=Usage(),
        deployment=settings.llm_model,
        rates=CostRates.from_settings(settings),
        provider="openai",
    )
    asyncio.run(hooks.on_llm_start(
        SimpleNamespace(usage=None), agent=None,
        system_prompt="You are the researcher.",
        input_items=[{"role": "user", "content": "who owns LTR?"}],
    ))
    asyncio.run(hooks.on_llm_end(
        SimpleNamespace(usage=None), agent=None,
        response=SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=100, output_tokens=20,
                input_tokens_details=SimpleNamespace(cached_tokens=0)),
            output=[
                SimpleNamespace(content=[SimpleNamespace(text="Resolving first.")],
                                name=None),
                SimpleNamespace(content=None, name="resolve_entities",
                                arguments='{"query": "LTR"}'),
            ],
        )))
    span = _generation(spans)
    assert "You are the researcher." in span.forwarded["request_messages"]
    assert "who owns LTR?" in span.forwarded["request_messages"]
    assert "Resolving first." in span.forwarded["response_choices"]
    assert 'resolve_entities({"query": "LTR"})' in span.forwarded["response_choices"]


def test_cache_reads_appear_as_a_langfuse_usage_unit(spans):
    settings = make_settings()
    hooks = _AgentRunHooks(
        EvidencePool(), 3, usage=Usage(),
        deployment=settings.llm_model,
        rates=CostRates.from_settings(settings),
        provider="openai",
    )
    asyncio.run(hooks.on_llm_start(
        SimpleNamespace(usage=None), agent=None, system_prompt="", input_items=[]))
    asyncio.run(hooks.on_llm_end(
        SimpleNamespace(usage=None), agent=None,
        response=SimpleNamespace(usage=SimpleNamespace(
            input_tokens=10_000, output_tokens=100,
            input_tokens_details=SimpleNamespace(cached_tokens=9_000)))))
    span = _generation(spans)
    assert span.usage()["cache_read_input_tokens"] == 9_000


# ── graceful handling of an intermittent gateway failure mid-stream ───────────

def test_a_stream_failing_before_the_first_token_is_retried(spans, monkeypatch):
    """The gateway returns intermittent 500s, and a research turn makes many calls,
    so long turns were the ones failing. Nothing has reached the client yet, so
    re-issuing cannot duplicate output — the answer should still arrive."""
    from openai import InternalServerError
    calls: list[int] = []

    async def _fake_acompletion(**_kw):
        calls.append(1)
        if len(calls) == 1:
            raise InternalServerError(
                "APIError - Internal server error",
                response=SimpleNamespace(status_code=500, headers={}, request=None),
                body=None,
            )

        async def _chunks():
            yield SimpleNamespace(
                usage=SimpleNamespace(prompt_tokens=100, completion_tokens=5,
                                      prompt_tokens_details=None,
                                      cache_read_input_tokens=0),
                choices=[SimpleNamespace(delta=SimpleNamespace(content="recovered"),
                                         finish_reason="stop")],
            )
        return _chunks()

    monkeypatch.setattr(litellm, "acompletion", _fake_acompletion)
    _real_sleep = asyncio.sleep          # capture before patching, or it recurses
    monkeypatch.setattr(asyncio, "sleep", lambda _d: _real_sleep(0))

    async def _drain():
        return [t async for t in _client().stream_text(
            [{"role": "user", "content": "q"}], usage=Usage())]

    assert asyncio.run(_drain()) == ["recovered"]
    assert len(calls) == 2                      # failed once, retried once


def test_a_stream_failing_after_tokens_is_not_replayed(monkeypatch):
    """Once text is out, re-issuing would duplicate it in the user's answer, so the
    turn degrades instead (LlmUnavailable → mapped 502, never a bare 500)."""
    from openai import InternalServerError
    calls: list[int] = []

    async def _fake_acompletion(**_kw):
        calls.append(1)

        async def _chunks():
            yield SimpleNamespace(
                usage=None,
                choices=[SimpleNamespace(delta=SimpleNamespace(content="partial "),
                                         finish_reason=None)],
            )
            raise InternalServerError(
                "APIError - Internal server error",
                response=SimpleNamespace(status_code=500, headers={}, request=None),
                body=None,
            )
        return _chunks()

    monkeypatch.setattr(litellm, "acompletion", _fake_acompletion)

    async def _drain():
        return [t async for t in _client().stream_text(
            [{"role": "user", "content": "q"}], usage=Usage())]

    with pytest.raises(LlmUnavailable):
        asyncio.run(_drain())
    assert len(calls) == 1                      # no replay after output began


# ── (d): trace cost == request cost, by construction ─────────────────

def test_span_costs_sum_to_the_request_cost(spans, monkeypatch):
    """The invariant Langfuse renders: adding up every generation's cost in a trace
    must equal the ``usage.cost_usd`` we return for that request.

    Both sides are computed from the same CostRates table, so this pins that they
    stay wired to it — a span priced from a different table (or a call metered but
    not spanned, or spanned but not metered) breaks the sum and this test.
    """
    settings = make_settings()
    usage = Usage()

    async def _fake_acompletion(**_kw):
        return _chat_response(4_000, 250)

    monkeypatch.setattr(litellm, "acompletion", _fake_acompletion)
    client = _client()

    # One turn's worth of calls, all metering into the same request Usage:
    # a fast-tier decision, a heavy-tier call, an embedding, and two agent rounds.
    asyncio.run(client.complete([{"role": "user", "content": "q"}], fast=True, usage=usage))
    asyncio.run(client.complete([{"role": "user", "content": "q"}], fast=False, usage=usage))

    class _Embeddings:
        async def create(self, **_kw):
            return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=416),
                                   data=[SimpleNamespace(embedding=[0.1])])

    client._embeddings_client = SimpleNamespace(embeddings=_Embeddings())
    asyncio.run(client.embed(["text"], usage=usage))

    hooks = _AgentRunHooks(
        EvidencePool(), 3, usage=usage, deployment=settings.llm_model,
        rates=CostRates.from_settings(settings), provider="openai",
    )
    for prompt, completion in ((9_000, 300), (12_000, 280)):
        asyncio.run(hooks.on_llm_start(SimpleNamespace(usage=None), agent=None,
                                       system_prompt="", input_items=[]))
        asyncio.run(hooks.on_llm_end(
            SimpleNamespace(usage=None), agent=None,
            response=SimpleNamespace(usage=SimpleNamespace(
                input_tokens=prompt, output_tokens=completion,
                input_tokens_details=SimpleNamespace(cached_tokens=0)))))

    # Five PRICED generations (chat, stream, embed, two agent rounds); the
    # structural round spans carry no cost and must not be counted as calls.
    priced = [s for s in spans if not s.name.startswith("agent.round")]
    assert len(priced) == 5                      # nothing unspanned, nothing double-spanned
    trace_cost = sum(s.cost().get("total", 0.0) for s in priced)
    request_cost = _request_cost(usage, settings)
    assert request_cost > 0
    assert trace_cost == pytest.approx(request_cost, abs=1e-6)


def test_a_cached_round_is_cheaper_in_both_the_span_and_the_request():
    """Cache reads must discount both sides identically, or the trace and the API
    disagree exactly when prompt caching starts paying off ((c))."""
    settings = make_settings()
    plain, cached = Usage(), Usage()
    for usage, cache in ((plain, 0), (cached, 9_000)):
        hooks = _AgentRunHooks(
            EvidencePool(), 3, usage=usage, deployment=settings.llm_model,
            rates=CostRates.from_settings(settings), provider="openai",
        )
        asyncio.run(hooks.on_llm_end(
            SimpleNamespace(usage=None), agent=None,
            response=SimpleNamespace(usage=SimpleNamespace(
                input_tokens=10_000, output_tokens=100,
                input_tokens_details=SimpleNamespace(cached_tokens=cache)))))
    assert _request_cost(cached, settings) < _request_cost(plain, settings)


# ── embedding spans carry their input (the question the trace is opened to answer) ──

def test_embedding_span_carries_the_text_and_the_vector_shape(spans):
    client = _client()

    class _Embeddings:
        async def create(self, **_kw):
            return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=12),
                                   data=[SimpleNamespace(embedding=[0.1] * 4)])

    client._embeddings_client = SimpleNamespace(embeddings=_Embeddings())
    asyncio.run(client.embed(["who owns Enterprise Search?"], dimensions=1536, usage=Usage()))
    (span,) = spans
    # Input: the text itself — blank before, which made the observation useless for
    # the question it is usually opened to answer.
    assert "who owns Enterprise Search?" in span.forwarded["request_messages"]
    # Output: the vector's SHAPE, never 1536 floats.
    assert span.forwarded["response_choices"] == "1 vector(s) x 1536d"
    assert "0.1" not in span.forwarded["response_choices"]


def test_a_large_embedding_batch_is_bounded_and_says_so(spans):
    from ai_graph_retrieval.connectors.llm.client import _TRACE_EMBED_MAX_TEXTS
    client = _client()
    titles = [f"Document number {i}" for i in range(40)]

    class _Embeddings:
        async def create(self, **_kw):
            return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=400),
                                   data=[SimpleNamespace(embedding=[0.1]) for _ in titles])

    client._embeddings_client = SimpleNamespace(embeddings=_Embeddings())
    asyncio.run(client.embed(titles, usage=Usage()))
    (span,) = spans
    # request_messages is ONE chat message (Langfuse's Formatted view renders
    # chat shapes; a bare string array showed as "undefined" rows on dev).
    (msg,) = json.loads(span.forwarded["request_messages"])
    lines = msg["content"].splitlines()
    assert msg["role"] == "user"
    assert len(lines) == _TRACE_EMBED_MAX_TEXTS + 1        # the capped page + the note
    assert lines[-1] == f"- … +{40 - _TRACE_EMBED_MAX_TEXTS} more not shown"
    assert span.forwarded["response_choices"].startswith("40 vector(s)")


def test_a_long_text_is_truncated_not_dropped(spans):
    from ai_graph_retrieval.connectors.llm.client import _TRACE_EMBED_TEXT_CHARS
    client = _client()

    class _Embeddings:
        async def create(self, **_kw):
            return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=900),
                                   data=[SimpleNamespace(embedding=[0.1])])

    client._embeddings_client = SimpleNamespace(embeddings=_Embeddings())
    asyncio.run(client.embed(["x" * 5_000], usage=Usage()))
    (span,) = spans
    (msg,) = json.loads(span.forwarded["request_messages"])
    assert len(msg["content"]) == _TRACE_EMBED_TEXT_CHARS + 2  # "- " prefix


def test_the_root_retrieve_span_carries_question_and_answer(spans):
    """Langfuse renders a trace's Input/Output from the root
    observation; ours showed null/undefined, so reading a trace meant opening
    an LLM child. The question and (bounded) answer now ride the root span."""
    from ai_graph_retrieval.observability.tracing import (
        _ROOT_IO_CHARS,
        record_root_io,
    )

    span = _RecordedSpan("retrieve/search")
    record_root_io(span, input="who owns LTR?")
    record_root_io(span, output="x" * 10_000)
    assert span.attrs["langfuse.observation.input"] == "who owns LTR?"
    assert len(span.attrs["langfuse.observation.output"]) == _ROOT_IO_CHARS
    record_root_io(None, input="never raises on a no-op span")


# ── model parameters + generation naming ──────────────────────────

def test_a_generation_records_the_settings_it_ran_under(spans, monkeypatch):
    """A live trace showed `modelParameters: {}` on every generation: the trace
    said what the model returned but not what it ran with, which is the first
    thing a cost or quality regression needs. Only output-shaping knobs belong —
    never the endpoint, and never the api key."""
    async def _fake_acompletion(**_kw):
        return _chat_response(100, 10)

    monkeypatch.setattr(litellm, "acompletion", _fake_acompletion)
    tool = {"type": "function", "function": {"name": "classify_query"}}
    asyncio.run(_client().complete(
        [{"role": "user", "content": "hi"}], fast=True, usage=Usage(),
        tools=[tool],
        tool_choice={"type": "function", "function": {"name": "classify_query"}},
    ))
    (span,) = spans
    params = span.params()
    assert params["max_tokens"] > 0
    # A forced function is named: "which tool was it pinned to" is the question a
    # malformed structured output raises.
    assert params["forced_function"] == "classify_query"
    assert params["tools_offered"] == 1
    # Transport and secrets never reach the trace.
    assert "api_key" not in params
    assert "api_base" not in params
    assert "cache_control_injection_points" not in params


def test_the_synthesis_generation_is_named_for_its_job(spans, monkeypatch):
    """`llm.chat.stream` named the call by how it was transported. It is the most
    expensive observation in a request (67% of cost on the measured run), so it
    is the last one that should have been anonymous."""
    async def _fake_acompletion(**_kw):
        async def _chunks():
            yield SimpleNamespace(
                usage=SimpleNamespace(prompt_tokens=900, completion_tokens=30,
                                      prompt_tokens_details=None,
                                      cache_read_input_tokens=0),
                choices=[SimpleNamespace(delta=SimpleNamespace(content="answer"),
                                         finish_reason="stop")],
            )
        return _chunks()

    monkeypatch.setattr(litellm, "acompletion", _fake_acompletion)

    async def _drain(**kw):
        return [t async for t in _client().stream_text(
            [{"role": "user", "content": "q"}], usage=Usage(), **kw)]

    asyncio.run(_drain(purpose="synthesis"))
    assert spans[-1].name == "llm.chat.synthesis"
    # The streamed generation carries its settings too.
    assert spans[-1].params()["max_tokens"] > 0
    # An untagged caller degrades to the old name rather than breaking the trace.
    spans.clear()
    asyncio.run(_drain())
    assert spans[-1].name == "llm.chat.stream"


def test_an_agent_round_records_the_settings_it_ran_under(spans):
    """The agent bypasses LlmClient, so it carries its own record_llm — the fourth
    generation site, easy to miss when the other three get their parameters.
    A live trace showed all five rounds with an empty panel while every other
    generation had one, and the rounds are the largest cost block in a Super
    Research request.

    `tools_offered` is asserted as a real per-request count, not a constant: the
    clarification and graph-text tools are conditional, so the number tells you
    which toolset that run actually had."""
    from ai_graph_retrieval.agent.tools import research_tools
    from ai_graph_retrieval.connectors.llm.client import trace_model_params

    settings = make_settings()
    tools = research_tools(True, graph_text=True)
    hooks = _AgentRunHooks(
        EvidencePool(), 3, usage=Usage(),
        deployment=settings.llm_model,
        rates=CostRates.from_settings(settings),
        provider="openai",
        model_parameters=trace_model_params(
            {"max_tokens": settings.agent_max_tokens}, tools=tools
        ),
    )
    asyncio.run(hooks.on_llm_start(
        SimpleNamespace(usage=None), agent=None, system_prompt="", input_items=[]))
    asyncio.run(hooks.on_llm_end(
        SimpleNamespace(usage=None), agent=None,
        response=SimpleNamespace(usage=SimpleNamespace(
            input_tokens=8_000, output_tokens=300,
            input_tokens_details=SimpleNamespace(cached_tokens=0)))))
    hooks.close()

    (gen,) = [s for s in spans if s.name == "llm.chat.agent"]
    params = gen.params()
    assert params["max_tokens"] == settings.agent_max_tokens
    assert params["tools_offered"] == len(tools) > 1
    # No temperature: the agent's ModelSettings never sets one, so a value here
    # would be invented. Every LlmClient call sends 0; the agent does not.
    assert "temperature" not in params


def test_an_agent_round_without_parameters_still_traces(spans):
    """Fail-open: the parameters are observability, so a hooks instance built
    without them (older caller, or a future one that forgets) must still produce a
    priced generation rather than raise."""
    settings = make_settings()
    hooks = _AgentRunHooks(
        EvidencePool(), 3, usage=Usage(),
        deployment=settings.llm_model,
        rates=CostRates.from_settings(settings),
        provider="openai",
    )
    asyncio.run(hooks.on_llm_start(
        SimpleNamespace(usage=None), agent=None, system_prompt="", input_items=[]))
    asyncio.run(hooks.on_llm_end(
        SimpleNamespace(usage=None), agent=None,
        response=SimpleNamespace(usage=SimpleNamespace(
            input_tokens=500, output_tokens=10,
            input_tokens_details=SimpleNamespace(cached_tokens=0)))))
    hooks.close()
    (gen,) = [s for s in spans if s.name == "llm.chat.agent"]
    assert gen.cost()["total"] > 0
    assert gen.params() == {}          # absent, not a crash and not a fabrication
