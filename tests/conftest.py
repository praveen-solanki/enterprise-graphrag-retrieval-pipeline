"""Shared fixtures.

``spans`` lives here because two suites need it — the trace HIERARCHY tests (who
is whose parent) and the answer-path tests (what each span records) — and a
recording tracer copied into both would be free to drift in the way that matters:
if one copy stops wiring ``_INITIALIZED``, its tests pass by recording nothing.
"""
from __future__ import annotations

import pytest

from ai_graph_retrieval.settings import Settings

# Tests must not depend on a developer's local .env.
Settings.model_config["env_file"] = None


@pytest.fixture
def spans(monkeypatch):
    """A real tracer provider exporting into memory, wired into the tracing module
    the same way the SDK wires it — so tests exercise the actual span calls rather
    than a stand-in.

    Every span helper is a no-op until both flags are patched; patching them is
    what makes these tests test anything.
    """
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from ai_graph_retrieval.observability import tracing

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(tracing, "_AVAILABLE", True)
    monkeypatch.setattr(tracing, "_INITIALIZED", True)
    monkeypatch.setattr(tracing, "_get_tracer", lambda name: provider.get_tracer(name))
    return exporter
