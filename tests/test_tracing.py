"""Tracing integration tests — no network, no exporter required.

Covers the two things that matter for this feature:
  1. It degrades gracefully when the OTEL packages are absent or nothing is
     configured — setup is a no-op, span helpers do nothing, nothing raises.
  2. When configured, setup wires one exporter per destination (OTLP collector,
     Langfuse's OTLP endpoint) and nothing else.
Plus the per-call cost math the span enrichment relies on.

    uv run pytest tests/test_tracing.py -v
"""
from __future__ import annotations

import base64
import json
from types import SimpleNamespace

import pytest

from ai_graph_retrieval.core.costing import (
    CostRates,
    call_cost_usd,
    embedding_cost_usd,
)
from ai_graph_retrieval.observability import tracing


@pytest.fixture(autouse=True)
def _reset_initialized():
    """Keep the module-level ``_INITIALIZED`` flag from leaking between tests (the setup
    tests flip it on via the real global)."""
    tracing._INITIALIZED = False
    tracing._PROVIDER = None
    yield
    tracing._INITIALIZED = False
    tracing._PROVIDER = None


def _cfg(**overrides):
    """A duck-typed Settings with just the fields setup_tracing reads."""
    base = {
        "tracing_enabled": True,
        "otel_service_name": "ai-graph-retrieval",
        "service_version": "0.1.0",
        "environment": "dev",
        "otel_exporter_otlp_endpoint": None,
        "langfuse_public_key": None,
        "langfuse_secret_key": None,
        "langfuse_host": None,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


_LANGFUSE = {
    # Deliberately not key-shaped: "sk" is the real prefix Langfuse, Stripe and
    # OpenAI secret keys start with, so a secret-keyword scanner flags it on sight.
    "langfuse_public_key": "pk", "langfuse_secret_key": "not-a-real-secret",
    "langfuse_host": "https://cloud.langfuse.com/",
}


def _rates() -> CostRates:
    s = SimpleNamespace(
        llm_model="org/chat-model",
        fast_model="org/fast-model",
        cost_chat_input_per_1m=3.0,
        cost_chat_output_per_1m=15.0,
        cost_chat_cached_input_per_1m=0.3,
        cost_fast_input_per_1m=1.0,
        cost_fast_output_per_1m=5.0,
        cost_fast_cached_input_per_1m=None,
        cost_embedding_per_1m=0.13,
    )
    return CostRates.from_settings(s)


# ── graceful degradation ─────────────────────────────────────────────────────

def test_setup_returns_false_when_disabled():
    assert tracing.setup_tracing(_cfg(tracing_enabled=False)) is False


def test_setup_returns_false_when_unavailable(monkeypatch):
    monkeypatch.setattr(tracing, "_AVAILABLE", False)
    # Even with exporters "configured", an absent package is a clean no-op.
    assert tracing.setup_tracing(_cfg(otel_exporter_otlp_endpoint="http://collector:4318")) is False


def test_setup_returns_false_when_nothing_is_configured(monkeypatch):
    monkeypatch.setattr(tracing, "_AVAILABLE", True)
    monkeypatch.setattr(tracing, "_exporters", lambda _s: [])
    assert tracing.setup_tracing(_cfg()) is False
    assert tracing._INITIALIZED is False


def test_span_helpers_are_noops_when_unavailable(monkeypatch):
    monkeypatch.setattr(tracing, "_AVAILABLE", False)
    with tracing.llm_span("llm.chat") as span:
        assert span is None
        tracing.record_llm(span, model="x", tokens_prompt=1)  # must not raise
    tracing.shutdown_tracing()  # must not raise


def test_llm_span_is_noop_when_available_but_not_initialized(monkeypatch):
    # Package present, but setup_tracing never initialised the pipeline (disabled / failed).
    # No provider exists to export to, so the span helper must create NOTHING.
    monkeypatch.setattr(tracing, "_AVAILABLE", True)
    monkeypatch.setattr(tracing, "_INITIALIZED", False)
    with tracing.llm_span("llm.chat") as span:
        assert span is None


# ── setup with exporters (simulated provider) ────────────────────────────────

def test_setup_registers_one_processor_per_exporter(monkeypatch):
    added: list = []
    registered: list = []

    class _Provider:
        def __init__(self, resource):
            self.resource = resource

        def add_span_processor(self, processor):
            added.append(processor)

    monkeypatch.setattr(tracing, "_AVAILABLE", True)
    monkeypatch.setattr(tracing, "_exporters", lambda _s: ["otlp", "langfuse"])
    monkeypatch.setattr(tracing, "_TracerProvider", _Provider)
    monkeypatch.setattr(tracing, "_BatchSpanProcessor", lambda exp: ("batch", exp))
    monkeypatch.setattr(tracing, "_Resource", SimpleNamespace(create=lambda attrs: attrs))
    monkeypatch.setattr(tracing, "_trace_api",
                        SimpleNamespace(set_tracer_provider=registered.append))
    monkeypatch.setattr(tracing, "_disable_openai_agents_sdk_tracing", lambda: None)

    assert tracing.setup_tracing(_cfg()) is True
    assert added == [("batch", "otlp"), ("batch", "langfuse")]
    assert registered[0].resource["service.name"] == "ai-graph-retrieval"
    assert registered[0].resource["deployment.environment"] == "dev"
    assert tracing._INITIALIZED is True


def test_setup_never_raises_on_a_broken_exporter(monkeypatch):
    def _boom(_s):
        raise RuntimeError("bad endpoint")

    monkeypatch.setattr(tracing, "_AVAILABLE", True)
    monkeypatch.setattr(tracing, "_exporters", _boom)
    assert tracing.setup_tracing(_cfg()) is False


def test_exporters_target_the_collector_and_langfuse_otlp_endpoints():
    pytest.importorskip("opentelemetry.exporter.otlp.proto.http.trace_exporter")
    exporters = tracing._exporters(_cfg(otel_exporter_otlp_endpoint="http://c:4318/",
                                        **_LANGFUSE))
    otlp, langfuse = exporters
    assert otlp._endpoint == "http://c:4318/v1/traces"
    assert langfuse._endpoint == "https://cloud.langfuse.com/api/public/otel/v1/traces"
    token = base64.b64encode(b"pk:not-a-real-secret").decode()
    assert langfuse._headers["Authorization"] == f"Basic {token}"


# ── startup self-check: resolved exporter state ──────────────────────────────

def test_observability_state_wired_when_langfuse_creds_present(monkeypatch):
    monkeypatch.setattr(tracing, "_AVAILABLE", True)
    st = tracing.observability_state(_cfg(**_LANGFUSE))
    assert st == {
        "package_installed": True, "tracing_enabled": True,
        "langfuse": True, "otlp": False, "will_export": True,
    }


def test_observability_state_langfuse_false_when_creds_partial(monkeypatch):
    monkeypatch.setattr(tracing, "_AVAILABLE", True)
    # host missing → langfuse exporter won't wire → nothing exports
    st = tracing.observability_state(
        _cfg(langfuse_public_key="pk", langfuse_secret_key="not-a-real-secret")
    )
    assert st["langfuse"] is False
    assert st["will_export"] is False


def test_observability_state_otlp_alone_exports(monkeypatch):
    monkeypatch.setattr(tracing, "_AVAILABLE", True)
    st = tracing.observability_state(_cfg(otel_exporter_otlp_endpoint="http://c:4318"))
    assert st["otlp"] is True and st["langfuse"] is False
    assert st["will_export"] is True


def test_observability_state_no_export_when_package_absent(monkeypatch):
    monkeypatch.setattr(tracing, "_AVAILABLE", False)
    # even with full creds, an absent package means nothing leaves the process
    st = tracing.observability_state(_cfg(**_LANGFUSE))
    assert st["package_installed"] is False
    assert st["will_export"] is False


def test_observability_state_no_export_when_disabled(monkeypatch):
    monkeypatch.setattr(tracing, "_AVAILABLE", True)
    st = tracing.observability_state(
        _cfg(tracing_enabled=False, otel_exporter_otlp_endpoint="http://collector:4318")
    )
    assert st["otlp"] is True  # endpoint IS configured...
    assert st["will_export"] is False  # ...but tracing is disabled, so nothing exports


def test_log_observability_state_never_raises(monkeypatch):
    monkeypatch.setattr(tracing, "_AVAILABLE", True)
    tracing.log_observability_state(_cfg())  # must not raise regardless of config


# ── per-call cost math (feeds the cost attributes on each span) ──────────────

def test_call_cost_chat_tier_with_cache_discount():
    rates = _rates()
    ci, co, ct = call_cost_usd(
        rates, "org/chat-model",
        prompt_tokens=1_000_000, completion_tokens=1_000_000, cached_prompt_tokens=500_000,
    )
    # 500k uncached @3 + 500k cached @0.3 = 1.5 + 0.15 = 1.65 input; 1M output @15 = 15.
    assert ci == 1.65
    assert co == 15.0
    assert ct == 16.65


def test_call_cost_fast_tier_uses_the_fast_rate():
    rates = _rates()
    ci, co, _ = call_cost_usd(
        rates, "org/fast-model", prompt_tokens=1_000_000, completion_tokens=1_000_000,
    )
    assert ci == 1.0
    assert co == 5.0


def test_embedding_cost():
    assert embedding_cost_usd(_rates(), 1_000_000) == 0.13


# ── Langfuse-native usage/cost attributes ─────────────────────────


class _Span:
    """Records set_attribute calls."""

    def __init__(self) -> None:
        self.attrs: dict = {}

    def set_attribute(self, key, value) -> None:
        self.attrs[key] = value


def _record(monkeypatch, **kw) -> _Span:
    span = _Span()
    forwarded: dict = {}
    monkeypatch.setattr(tracing, "_AVAILABLE", True)
    monkeypatch.setattr(tracing, "_set_llm_attributes",
                        lambda _s, **a: forwarded.update(a))
    tracing.record_llm(span, **kw)
    span.forwarded = forwarded          # type: ignore[attr-defined]
    return span


def test_langfuse_cost_details_carry_our_computed_cost(monkeypatch):
    # Langfuse prices from its own model registry, which does not know self-hosted
    # or NIM model names. These keys make it use OUR numbers, so trace cost matches
    # the API's usage.cost_usd.
    span = _record(monkeypatch, model="org/fast-model", tokens_prompt=1_000,
                   tokens_completion=200, tokens_cached=800,
                   cost_input=0.001, cost_output=0.002, cost_total=0.003)
    assert json.loads(span.attrs["langfuse.observation.cost_details"]) == {
        "input": 0.001, "output": 0.002, "total": 0.003,
    }
    assert json.loads(span.attrs["langfuse.observation.usage_details"]) == {
        "input": 1_000, "output": 200, "cache_read_input_tokens": 800,
    }
    assert span.attrs["langfuse.observation.model.name"] == "org/fast-model"


def test_cache_token_key_is_not_forwarded_to_the_tracing_sdk(monkeypatch):
    # set_llm_attributes has no parameter for cache reads — forwarding it would
    # raise TypeError inside a live request.
    span = _record(monkeypatch, model="m", tokens_prompt=10, tokens_completion=1,
                   tokens_cached=5, cost_total=0.1)
    assert "tokens_cached" not in span.forwarded      # type: ignore[attr-defined]
    assert span.forwarded["tokens_prompt"] == 10      # type: ignore[attr-defined]


def test_embedding_style_call_gets_cost_without_completion_tokens(monkeypatch):
    span = _record(monkeypatch, model="nvidia/nemotron-3.5-embed-1b",
                   tokens_prompt=333, tokens_total=333, cost_total=0.00004)
    assert json.loads(span.attrs["langfuse.observation.usage_details"]) == {"input": 333}
    assert json.loads(span.attrs["langfuse.observation.cost_details"]) == {"total": 0.00004}


def test_no_cost_keys_emitted_when_nothing_to_report(monkeypatch):
    span = _record(monkeypatch, model="m")
    assert "langfuse.observation.cost_details" not in span.attrs
    assert "langfuse.observation.usage_details" not in span.attrs


# ── options_applied on the root retrieve span ────────────────────────────────


def test_applied_options_land_as_flat_filterable_attributes(monkeypatch):
    """The effective RunConfig already rides the response and the log line; the
    trace is the third surface. Scalars must stay native (the Langfuse UI can
    only filter primitives) and nested values (the ``run`` sources meta) are
    JSON-encoded rather than dropped."""
    monkeypatch.setattr(tracing, "_AVAILABLE", True)
    span = _Span()
    tracing.record_applied_options(span, {
        "agent_rounds": 1,
        "graph_chunk_cap": 0,
                "chat_smalltalk_shortcircuit": False,
        "run": {"sources": [{"name": "wikipedia"}]},
        "skipped": None,
    })
    assert span.attrs["options.agent_rounds"] == 1
    assert span.attrs["options.graph_chunk_cap"] == 0
    assert span.attrs["options.chat_smalltalk_shortcircuit"] is False
    assert json.loads(span.attrs["options.run"]) == {"sources": [{"name": "wikipedia"}]}
    assert "options.skipped" not in span.attrs


def test_applied_options_are_noops_without_a_span_or_package(monkeypatch):
    """Observability must never fail a request: no span (tracing off for the
    turn) and no package (local dev) are both silent no-ops."""
    monkeypatch.setattr(tracing, "_AVAILABLE", True)
    tracing.record_applied_options(None, {"agent_rounds": 1})  # must not raise
    monkeypatch.setattr(tracing, "_AVAILABLE", False)
    span = _Span()
    tracing.record_applied_options(span, {"agent_rounds": 1})
    assert span.attrs == {}
