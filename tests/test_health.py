"""Probe endpoint tests — run without K8s, no network required.

    uv run pytest tests/test_health.py -v
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ai_graph_retrieval.api.routes import health


def _app(container=None) -> FastAPI:
    app = FastAPI()
    app.include_router(health.router)
    if container is not None:
        app.state.container = container
    return app


def _mock_container(
    neo4j_ok: bool = True,
    qdrant_enabled: bool = False,
    qdrant_ok: bool = True,
    mongo_enabled: bool = False,
    mongo_ok: bool = True,
    check_stores: bool = False,
):
    mock_graph = AsyncMock()
    if neo4j_ok:
        mock_graph.verify = AsyncMock(return_value=None)
    else:
        mock_graph.verify = AsyncMock(
            side_effect=Exception("ServiceUnavailable: Unable to retrieve routing information")
        )

    mock_vector_store = None
    if qdrant_enabled:
        mock_vector_store = AsyncMock()
        mock_vector_store.verify = AsyncMock(return_value=qdrant_ok)

    mock_run_registry = None
    if mongo_enabled:
        mock_run_registry = AsyncMock()
        mock_run_registry.verify = AsyncMock(return_value=mongo_ok)

    mock_settings = MagicMock()
    # Store probes gate readiness only when this is True.
    mock_settings.health_check_stores = check_stores

    container = MagicMock()
    container.http_graph = mock_graph
    container.graph = mock_graph
    container.vector_store = mock_vector_store
    container.run_registry = mock_run_registry
    container.settings = mock_settings
    return container


@pytest.fixture(autouse=True)
def _reset():
    health._services.clear()
    health._ready = False
    yield
    health._services.clear()
    health._ready = False


# ── /startUpProbe ──────────────────────────────────────────────────────────────

def test_startup_passes_when_no_services_registered():
    r = TestClient(_app()).get("/startUpProbe")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_startup_fails_while_service_pending():
    health.register_service("neo4j")
    r = TestClient(_app()).get("/startUpProbe")
    assert r.status_code == 503
    assert "neo4j" in r.json()["pending"]


def test_startup_passes_after_all_initialized():
    health.register_service("neo4j")
    health.mark_initialized("neo4j")
    r = TestClient(_app()).get("/startUpProbe")
    assert r.status_code == 200


# ── /livenessProbe ─────────────────────────────────────────────────────────────

def test_liveness_always_200():
    r = TestClient(_app()).get("/livenessProbe")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ── /readinessProbe ────────────────────────────────────────────────────────────

def test_readiness_503_when_not_ready():
    """Default state: set_ready never called → 503."""
    r = TestClient(_app(_mock_container())).get("/readinessProbe")
    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "fail"
    assert body["probeType"] == "readiness"


def test_readiness_200_when_ready():
    health._ready = True
    r = TestClient(_app(_mock_container())).get("/readinessProbe")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "pass"
    assert body["probeType"] == "readiness"
    assert body["reason"] == []


# With health_check_stores OFF (default), store reachability does NOT gate
# readiness: an unreachable Neo4j or Qdrant must NOT flip readiness to 503.

def test_readiness_200_even_when_neo4j_unreachable():
    health._ready = True
    r = TestClient(_app(_mock_container(neo4j_ok=False))).get("/readinessProbe")
    assert r.status_code == 200
    assert r.json()["status"] == "pass"


def test_readiness_200_even_when_qdrant_unreachable():
    health._ready = True
    r = TestClient(
        _app(_mock_container(qdrant_enabled=True, qdrant_ok=False))
    ).get("/readinessProbe")
    assert r.status_code == 200
    assert r.json()["status"] == "pass"


def test_readiness_200_even_when_mongo_unreachable():
    health._ready = True
    r = TestClient(
        _app(_mock_container(mongo_enabled=True, mongo_ok=False))
    ).get("/readinessProbe")
    assert r.status_code == 200
    assert r.json()["status"] == "pass"


# With health_check_stores ON, store reachability gates readiness again.

def test_readiness_200_when_stores_checked_and_ok():
    health._ready = True
    r = TestClient(
        _app(_mock_container(qdrant_enabled=True, qdrant_ok=True, check_stores=True))
    ).get("/readinessProbe")
    assert r.status_code == 200
    body = r.json()
    assert body["neo4j"] == "ok" and body["qdrant"] == "ok"


def test_readiness_503_when_stores_checked_and_neo4j_unreachable():
    health._ready = True
    r = TestClient(
        _app(_mock_container(neo4j_ok=False, check_stores=True))
    ).get("/readinessProbe")
    assert r.status_code == 503
    assert r.json()["neo4j"] == "unreachable"


def test_readiness_503_when_stores_checked_and_qdrant_unreachable():
    health._ready = True
    r = TestClient(
        _app(_mock_container(qdrant_enabled=True, qdrant_ok=False, check_stores=True))
    ).get("/readinessProbe")
    assert r.status_code == 503
    assert r.json()["qdrant"] == "unreachable"


def test_readiness_200_when_stores_checked_and_mongo_ok():
    health._ready = True
    r = TestClient(
        _app(_mock_container(mongo_enabled=True, mongo_ok=True, check_stores=True))
    ).get("/readinessProbe")
    assert r.status_code == 200
    assert r.json()["mongo"] == "ok"


def test_readiness_503_when_stores_checked_and_mongo_unreachable():
    health._ready = True
    r = TestClient(
        _app(_mock_container(mongo_enabled=True, mongo_ok=False, check_stores=True))
    ).get("/readinessProbe")
    assert r.status_code == 503
    assert r.json()["mongo"] == "unreachable"


def test_readiness_200_when_stores_checked_and_mongo_not_configured():
    """run_registry unset (mongo feature off) → no mongo probe, no mongo key."""
    health._ready = True
    r = TestClient(
        _app(_mock_container(mongo_enabled=False, check_stores=True))
    ).get("/readinessProbe")
    assert r.status_code == 200
    assert "mongo" not in r.json()


def test_the_vector_store_actually_has_the_verify_the_probe_calls():
    """Both Container.start and /readinessProbe call vector_store.verify(),
    but every test above mocks it — which is how the method was missing from
    QdrantVectorStore for months (every probe failed with AttributeError on a
    healthy endpoint). Pin the real class: verify exists and is one
    get_collections round-trip."""
    import asyncio
    from unittest.mock import AsyncMock

    from ai_graph_retrieval.connectors.qdrant.store import QdrantVectorStore
    from tests.support import make_settings

    store = QdrantVectorStore(make_settings())
    store._client = AsyncMock()
    assert asyncio.run(store.verify()) is True
    store._client.get_collections.assert_awaited_once()
