"""Unit tests for the Neo4j Query API v2 HTTP transport.

Covers keep-alive expiry, reconnect on dropped idle connections, and retries on
transient gateway 5xx (idempotent reads).

    uv run pytest tests/test_neo4j_http_client.py -v
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import pytest

from ai_graph_retrieval.connectors.neo4j import http_client as hc
from ai_graph_retrieval.core.exceptions import GraphUnavailable


def _settings(**over):
    base = {
        "neo4j_http_url": "https://neo4j.example/",
        "neo4j_database": "neo4j",
        "neo4j_username": "u",
        "neo4j_password": "p",
        "neo4j_tx_timeout_s": 120.0,
        "neo4j_connection_timeout_s": 5.0,
        "neo4j_http_keepalive_expiry_s": 60.0,
        "neo4j_http_max_keepalive": 20,
    }
    base.update(over)
    return SimpleNamespace(**base)


def _resp(status=200, body=None, text=None):
    if text is not None:
        return httpx.Response(status, text=text)
    return httpx.Response(status, json=body if body is not None else {
        "data": {"fields": ["ok"], "values": [[1]]}
    })


def test_client_configured_with_keepalive_expiry(monkeypatch):
    captured = {}

    class _FakeAC:
        def __init__(self, **kw):
            captured.update(kw)

    monkeypatch.setattr(hc.httpx, "AsyncClient", _FakeAC)
    hc.Neo4jHttpClient(_settings(neo4j_http_keepalive_expiry_s=42.0))
    limits = captured["limits"]
    assert limits.keepalive_expiry == 42.0
    assert limits.max_keepalive_connections == 20


def test_reconnect_retry_on_dropped_keepalive(monkeypatch):
    """First POST hits a server-dropped idle connection (RemoteProtocolError); the
    retry on a fresh connection succeeds and rows come back."""

    async def _no_sleep(*_a, **_k):
        return None

    monkeypatch.setattr(hc.asyncio, "sleep", _no_sleep)
    c = hc.Neo4jHttpClient(_settings())
    calls = {"n": 0}

    async def _post(url, json):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.RemoteProtocolError("Server disconnected")
        return _resp()

    c._client = SimpleNamespace(post=_post, aclose=None)
    rows = asyncio.run(c.rows("RETURN 1 AS ok", {}))
    assert calls["n"] == 2                 # retried exactly once
    assert rows == [{"ok": 1}]


def test_reconnect_retry_exhausted_raises_unavailable(monkeypatch):
    """All reconnect attempts exhausted → GraphUnavailable."""

    async def _no_sleep(*_a, **_k):
        return None

    monkeypatch.setattr(hc.asyncio, "sleep", _no_sleep)
    c = hc.Neo4jHttpClient(_settings())
    calls = {"n": 0}

    async def _post(url, json):
        calls["n"] += 1
        raise httpx.RemoteProtocolError("Server disconnected")

    c._client = SimpleNamespace(post=_post)
    with pytest.raises(GraphUnavailable):
        asyncio.run(c.rows("RETURN 1 AS ok", {}))
    assert calls["n"] == hc._MAX_ATTEMPTS


def test_non_retryable_transport_error_raises_immediately():
    """A plain connect error (not the stale-keepalive case) is not retried."""
    c = hc.Neo4jHttpClient(_settings())
    calls = {"n": 0}

    async def _post(url, json):
        calls["n"] += 1
        raise httpx.ConnectError("refused")

    c._client = SimpleNamespace(post=_post)
    with pytest.raises(GraphUnavailable):
        asyncio.run(c.rows("RETURN 1 AS ok", {}))
    assert calls["n"] == 1                 # no retry on a non-RemoteProtocolError


def test_5xx_retried_then_succeeds(monkeypatch):
    """Transient gateway 500 is retried; a later 200 returns rows."""

    async def _no_sleep(*_a, **_k):
        return None

    monkeypatch.setattr(hc.asyncio, "sleep", _no_sleep)
    c = hc.Neo4jHttpClient(_settings())
    calls = {"n": 0}

    async def _post(url, json):
        calls["n"] += 1
        if calls["n"] == 1:
            return _resp(500, text='{"errors":[{"message":"gateway blip"}]}')
        return _resp()

    c._client = SimpleNamespace(post=_post)
    rows = asyncio.run(c.rows("RETURN 1 AS ok", {}))
    assert calls["n"] == 2
    assert rows == [{"ok": 1}]


def test_5xx_exhausted_raises_unavailable(monkeypatch):
    """Persistent 5xx across all attempts → GraphUnavailable."""

    async def _no_sleep(*_a, **_k):
        return None

    monkeypatch.setattr(hc.asyncio, "sleep", _no_sleep)
    c = hc.Neo4jHttpClient(_settings())
    calls = {"n": 0}

    async def _post(url, json):
        calls["n"] += 1
        return _resp(502, text="Bad Gateway")

    c._client = SimpleNamespace(post=_post)
    with pytest.raises(GraphUnavailable):
        asyncio.run(c.rows("RETURN 1 AS ok", {}))
    assert calls["n"] == hc._MAX_ATTEMPTS


def test_every_statement_declares_read_access_mode():
    """This service only ever READS the graph, and the Query API routes an
    undeclared statement as a WRITE — so a writer-less cluster (leader election,
    routing loss) failed every read with "no longer accepts writes" (shared dev
    graph, 2026-08-10). Declaring the mode routes to readers and keeps reads up
    through writer loss. "READ" is the documented Query API v2 enum value
    (verified against the real dev server: all casings accepted today, but the
    canonical form survives a stricter server version)."""
    c = hc.Neo4jHttpClient(_settings())
    seen = {}

    async def _post(url, json):
        seen.update(json)
        return _resp()

    c._client = SimpleNamespace(post=_post, aclose=None)
    asyncio.run(c.rows("RETURN 1 AS ok", {}))
    assert seen["accessMode"] == "READ"


_ROUTING_ERR = {
    "errors": [{
        "code": "Neo.TransientError.Statement.RemoteExecutionTransientError",
        "message": ("Remote execution failed with code N/A and message 'Failed to "
                    "obtain connection towards READ server. Known routing table is: "
                    "routers [], writers [server-2:7688], readers []'"),
    }]
}


def test_a_readerless_cluster_falls_back_to_default_routing():
    """Second real dev failure (2026-08-12): the cluster had a writer and
    `readers []`, so READ-routed statements failed while the same statement
    succeeded on default routing. A read-only service must keep answering — it
    ran on the writer for its whole life before READ was declared."""
    c = hc.Neo4jHttpClient(_settings())
    sent = []

    async def _post(url, json):
        sent.append(json.get("accessMode", "<unset>"))
        return _resp() if len(sent) > 1 else _resp(body=_ROUTING_ERR)

    c._client = SimpleNamespace(post=_post, aclose=None)
    rows = asyncio.run(c.rows("RETURN 1 AS ok", {}))
    assert sent == ["READ", "<unset>"]   # preferred routing first, then fallback
    assert rows == [{"ok": 1}]


def test_a_real_cypher_error_is_not_retried_as_a_routing_failure():
    """The fallback must not mask genuine faults: a syntax error (or any other
    transient remote failure that is NOT about READ routing) surfaces as-is,
    with no second call."""
    c = hc.Neo4jHttpClient(_settings())
    calls = {"n": 0}
    syntax = {"errors": [{"code": "Neo.ClientError.Statement.SyntaxError",
                          "message": "Invalid input"}]}

    async def _post(url, json):
        calls["n"] += 1
        return _resp(body=syntax)

    c._client = SimpleNamespace(post=_post, aclose=None)
    with pytest.raises(hc.Neo4jHttpError):
        asyncio.run(c.rows("RETURN", {}))
    assert calls["n"] == 1


def test_the_healthy_path_still_sends_read_and_makes_one_call():
    c = hc.Neo4jHttpClient(_settings())
    sent = []

    async def _post(url, json):
        sent.append(json.get("accessMode", "<unset>"))
        return _resp()

    c._client = SimpleNamespace(post=_post, aclose=None)
    asyncio.run(c.rows("RETURN 1 AS ok", {}))
    assert sent == ["READ"]
