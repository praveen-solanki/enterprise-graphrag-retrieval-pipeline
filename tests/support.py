"""Shared test helpers.

``make_settings`` builds a Settings instance that does NOT read the developer's
``.env`` (``_env_file=None``) and supplies the handful of required secrets with
throwaway values. Tests must never depend on a local ``.env`` — CI has none, and
ambient values would silently change what a test exercises.
"""
from __future__ import annotations

import json
import os

from dotenv import dotenv_values

from ai_graph_retrieval.retrieval.base import Intent, Mode
from ai_graph_retrieval.retrieval.profile import ExecutionProfile, profile_for
from ai_graph_retrieval.settings import Settings

_REQUIRED = {
    "neo4j_uri": "bolt://localhost:7687",
    "neo4j_password": "x",
    "llm_base_url": "https://example.invalid/v1",
    "llm_api_key": "x",
}

# ``_env_file=None`` silences pydantic's OWN dotenv source, but importing
# litellm (via the client module) calls ``load_dotenv()`` at import time and
# dumps the developer's .env into ``os.environ`` — and real env vars outrank
# every constructor switch. Scrub exactly the keys the .env defines (plus the
# AGR_ prefix) around construction, so the docstring's promise holds on a dev
# box and not just in CI.
_DOTENV_KEYS: frozenset[str] = frozenset(dotenv_values(".env") or ())
# Unprefixed aliases Settings also reads (e.g. a real NVIDIA_API_KEY in the shell).
_ALIASED_ENV: frozenset[str] = frozenset({
    "NVIDIA_API_KEY", "NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "NEO4J_DATABASE",
    "NEO4J_HTTP_URL", "QDRANT_HOST", "QDRANT_URL", "QDRANT_PORT", "QDRANT_API_KEY", "MONGO_URI",
    "MONGO_DB", "REDIS_HOST", "REDIS_PORT", "REDIS_USERNAME", "REDIS_PASSWORD",
    "ACL_ENFORCEMENT_ENABLED", "HEALTH_CHECK_STORES", "OTEL_EXPORTER_OTLP_ENDPOINT",
    "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST", "LANGFUSE_BASE_URL",
})


def make_settings(**overrides) -> Settings:
    """Settings with required secrets stubbed and the real .env ignored."""
    saved = {
        k: os.environ.pop(k)
        for k in list(os.environ)
        if k.startswith("AGR_") or k in _DOTENV_KEYS or k in _ALIASED_ENV
    }
    try:
        return Settings(_env_file=None, **{**_REQUIRED, **overrides})
    finally:
        os.environ.update(saved)


def make_profile(
    mode: Mode = Mode.SEARCH,
    intent: Intent = Intent.ENTITY,
    **overrides,
) -> ExecutionProfile:
    """The execution profile a request of this mode/intent resolves to.

    Built through the real ``profile_for`` rather than hand-assembled, so a test
    context always runs the profile production would give it (and a new profile
    field can't be silently missing from the fixtures). Default SEARCH: the
    middle mode, and the one whose profile carries every capability the shared
    passes read (short agent follow-up, enrich, re-ranking).
    """
    return profile_for(make_settings(**overrides), mode, intent)


def span_attrs(exporter, name: str) -> dict:
    """One recorded span's attributes, by span name.

    Raises ``StopIteration`` when the span is absent, which is deliberate: a
    missing span is the failure these assertions exist to catch, and a helper that
    returned ``{}`` would turn it into a confusing attribute-not-found error.
    Pairs with the ``spans`` fixture in conftest.
    """
    span = next(s for s in exporter.get_finished_spans() if s.name == name)
    return dict(span.attributes)


def span_output(exporter, name: str) -> dict:
    """A span's Langfuse Output, parsed. Every ``record_io`` call site in the
    service writes JSON, so a test reads it as data rather than matching strings."""
    return json.loads(span_attrs(exporter, name)["langfuse.observation.output"])


def make_people_settings(**overrides) -> Settings:
    """``make_settings`` with the optional people/HR layer switched on."""
    return make_settings(**{"people_graph_enabled": True, **overrides})
