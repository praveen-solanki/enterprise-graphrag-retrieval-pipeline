"""Per-request RunConfig: schema validation, graph-source resolution (configured
source, exact run, latest COMPLETED run), the embedding-space derivation, and
the depth-knob override clamping."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from ai_graph_retrieval.api.routes.retrieve import resolve_run
from ai_graph_retrieval.api.schemas import RetrieveRequest
from ai_graph_retrieval.connectors.mongo.run_registry import RunRegistryClient
from ai_graph_retrieval.core.access import AccessContext
from ai_graph_retrieval.core.exceptions import InvalidRequest
from ai_graph_retrieval.retrieval.profile import apply_overrides, search_profile
from ai_graph_retrieval.settings import GraphSource
from tests.support import make_settings


def _ctx() -> AccessContext:
    return AccessContext(tenant_id="demo", principals=("public",), request_id="req-1")


def _record(connector_type: str, suffix: str, dimension: int = 2048,
            created_at: str = "2026-08-10T00:00:00+00:00",
            status: str = "COMPLETED") -> dict:
    return {
        "run_id": f"idx:demo:{connector_type}:{suffix}",
        "tenant_id": "demo",
        "connector_type": connector_type,
        "run_suffix": suffix,
        "status": status,
        "created_at": created_at,
        "neo4j_database": f"g-demo-{connector_type}-{suffix}",
        "qdrant_entities_collection": f"g-demo-{connector_type}-entities-{suffix}",
        "qdrant_chunks_collection": f"g-demo-{connector_type}-chunks-{suffix}",
        "run_config": {
            "embedding_model": "nvidia/nemotron-3.5-embed-1b",
            "embedding_dimension": dimension,
        },
    }


class _FakeColl:
    """Stand-in for the run_registry Mongo collection: exact-triple match, or the
    latest by created_at when the query omits run_suffix (mirrors find_one+sort)."""

    def __init__(self, docs: list[dict]) -> None:
        self._docs = list(docs)
        self.calls = 0

    def find_one(self, query: dict, sort=None) -> dict | None:
        self.calls += 1
        matches = [
            d for d in self._docs
            if d["tenant_id"] == query["tenant_id"]
            and d["connector_type"] == query["connector_type"]
            and ("run_suffix" not in query or d["run_suffix"] == query["run_suffix"])
            and ("status" not in query or d.get("status") == query["status"])
        ]
        if not matches:
            return None
        if sort:
            field, direction = sort[0]
            matches.sort(key=lambda d: d.get(field) or "", reverse=direction < 0)
        return matches[0]


class _ListColl(_FakeColl):
    def find(self, query, projection=None):
        return [d for d in self._docs if all(d.get(k) == v for k, v in query.items())]


class _RaisingColl:
    def find_one(self, *_a, **_k):
        raise ConnectionError("mongo down")


def test_completed_tenants_groups_connectors_and_skips_failed_runs():
    docs = [
        _record("wikipedia", "old", created_at="2026-01-01T00:00:00+00:00"),
        {**_record("arxiv", "new", created_at="2026-08-01T00:00:00+00:00"),
         "tenant_id": "other"},
        {**_record("wikipedia", "mid", created_at="2026-06-01T00:00:00+00:00"),
         "tenant_id": "other"},
        {**_record("wikipedia", "fail", status="FAILED"), "tenant_id": "gone"},
    ]
    client = RunRegistryClient(make_settings(), collection=_ListColl(docs))
    rows = asyncio.run(client.completed_tenants())
    assert [r["tenant_id"] for r in rows] == ["other", "demo"]
    assert rows[0]["connectors"] == ["arxiv", "wikipedia"]
    assert rows[1]["connectors"] == ["wikipedia"]


def _container(settings, docs):
    registry = (RunRegistryClient(settings, collection=_FakeColl(docs))
                if docs is not None else None)
    return SimpleNamespace(settings=settings, run_registry=registry)


def _resolve(settings, docs, **body):
    req = RetrieveRequest(question="q", mode="search", **body)
    return asyncio.run(resolve_run(_container(settings, docs), _ctx(), req))


_CONFIGURED = GraphSource(name="wikipedia", database="wiki-db",
                          entities_collection="wiki-entities",
                          chunks_collection="wiki-chunks")


# ── schema validation ────────────────────────────────────────────────────────

def test_omitting_run_selection_is_valid():
    r = RetrieveRequest(question="q", mode="search")
    assert r.connectorType is None and r.runIdSuffix is None and r.options is None
    assert r.connectors() == []


def test_a_single_connector_or_a_list_is_accepted():
    assert RetrieveRequest(question="q", mode="search",
                           connectorType="wikipedia").connectors() == ["wikipedia"]
    assert RetrieveRequest(question="q", mode="search",
                           connectorType=["wikipedia", "arxiv"]).connectors() == [
        "wikipedia", "arxiv"]


def test_suffix_without_connector_is_rejected():
    with pytest.raises(ValueError):
        RetrieveRequest(question="q", mode="search", runIdSuffix="x")


def test_a_plain_string_suffix_needs_exactly_one_connector():
    with pytest.raises(ValueError):
        RetrieveRequest(question="q", mode="search", connectorType=["wikipedia", "arxiv"],
                        runIdSuffix="x")


def test_suffix_map_keys_must_be_selected_connectors():
    RetrieveRequest(question="q", mode="search", connectorType=["wikipedia", "arxiv"],
                    runIdSuffix={"wikipedia": "ab12"})
    with pytest.raises(ValueError):
        RetrieveRequest(question="q", mode="search", connectorType="wikipedia",
                        runIdSuffix={"arxiv": "ab12"})


@pytest.mark.parametrize("bad", ["wiki pedia", "-x", "a" * 65])
def test_malformed_connector_names_are_rejected(bad):
    with pytest.raises(ValueError):
        RetrieveRequest(question="q", mode="search", connectorType=bad)


# ── resolution ───────────────────────────────────────────────────────────────

def test_default_connectors_resolve_their_latest_completed_runs():
    s = make_settings()
    rc = _resolve(s, [
        _record("wikipedia", "old1", created_at="2026-08-01T00:00:00+00:00"),
        _record("wikipedia", "new1", created_at="2026-08-09T00:00:00+00:00"),
        _record("wikipedia", "fail", created_at="2026-08-10T00:00:00+00:00",
                status="FAILED"),
        _record("arxiv", "ax01"),
    ])
    assert [(src.name, src.database) for src in rc.sources] == [
        ("wikipedia", "g-demo-wikipedia-new1"), ("arxiv", "g-demo-arxiv-ax01")]
    assert rc.sources[0].chunks_collection == "g-demo-wikipedia-chunks-new1"
    assert rc.sources[0].entity_fulltext_index == "entity_search"
    assert [m["selected_by"] for m in rc.resolved_meta["sources"]] == ["latest", "latest"]
    assert rc.resolved_meta["run_selection"] == "resolved"


def test_a_default_connector_without_a_run_is_skipped():
    s = make_settings()
    rc = _resolve(s, [_record("wikipedia", "w1")])
    assert [src.name for src in rc.sources] == ["wikipedia"]


def test_no_default_connector_resolving_is_a_422():
    with pytest.raises(InvalidRequest):
        _resolve(make_settings(), [])


def test_an_explicitly_named_connector_without_a_run_is_a_422():
    with pytest.raises(InvalidRequest):
        _resolve(make_settings(), [_record("wikipedia", "w1")],
                 connectorType=["wikipedia", "arxiv"])


def test_an_explicit_suffix_resolves_that_exact_run_even_if_not_completed():
    s = make_settings()
    rc = _resolve(s, [_record("arxiv", "luup", status="FAILED"), _record("arxiv", "new")],
                  connectorType="arxiv", runIdSuffix="luup")
    assert rc.sources[0].database == "g-demo-arxiv-luup"
    assert rc.resolved_meta["sources"][0]["selected_by"] == "suffix"


def test_missing_exact_record_is_a_422():
    with pytest.raises(InvalidRequest):
        _resolve(make_settings(), [_record("arxiv", "a1")],
                 connectorType="arxiv", runIdSuffix="nope")


def test_a_configured_source_needs_no_registry():
    s = make_settings(graph_sources_config=[_CONFIGURED], default_connectors=["wikipedia"])
    rc = _resolve(s, None)
    assert rc.sources == (_CONFIGURED,)
    assert rc.resolved_meta["run_selection"] == "settings"
    assert rc.embedding_space is None


def test_a_suffix_bypasses_the_configured_source():
    s = make_settings(graph_sources_config=[_CONFIGURED])
    rc = _resolve(s, [_record("wikipedia", "w9")], connectorType="wikipedia",
                  runIdSuffix="w9")
    assert rc.sources[0].database == "g-demo-wikipedia-w9"


def test_named_connector_without_registry_or_source_is_a_422():
    with pytest.raises(InvalidRequest):
        _resolve(make_settings(), None, connectorType="arxiv")


def test_a_raising_registry_becomes_an_explicit_error_not_a_500():
    s = make_settings()
    container = SimpleNamespace(
        settings=s, run_registry=RunRegistryClient(s, collection=_RaisingColl()))
    req = RetrieveRequest(question="q", mode="search", connectorType="arxiv")
    with pytest.raises(InvalidRequest):
        asyncio.run(resolve_run(container, _ctx(), req))


def test_the_run_record_sets_the_query_embedding_space():
    rc = _resolve(make_settings(), [_record("wikipedia", "w1", dimension=1024)],
                  connectorType="wikipedia")
    assert rc.embedding_space.dimensions == 1024
    assert rc.embedding_space.deployment == "nvidia/nemotron-3.5-embed-1b"


def test_mixed_dimensions_drop_the_embedding_space():
    rc = _resolve(make_settings(), [_record("wikipedia", "w1", dimension=2048),
                                    _record("arxiv", "a1", dimension=1024)])
    assert len(rc.sources) == 2
    assert rc.embedding_space is None


def test_option_overrides_pass_through_on_a_resolved_run():
    rc = _resolve(make_settings(), [_record("wikipedia", "w1")],
                  connectorType="wikipedia", options={"agent_rounds": 1})
    assert rc.option_overrides == {"agent_rounds": 1}


def test_registry_caches_positive_and_negative_lookups():
    s = make_settings()
    coll = _FakeColl([_record("arxiv", "luup", status="FAILED")])
    client = RunRegistryClient(s, collection=coll)

    async def run():
        first = await client.resolve("demo", "arxiv", "luup")
        assert first is not None and first.status == "FAILED"
        await client.resolve("demo", "arxiv", "luup")  # cache hit
        assert coll.calls == 1
        assert await client.resolve("demo", "arxiv", "gone") is None
        await client.resolve("demo", "arxiv", "gone")  # negative cached
        assert coll.calls == 2

    asyncio.run(run())


# ── apply_overrides: clamp, echo, structural guards ───────────────────────────

def test_overrides_are_clamped_to_ceilings():
    s = make_settings()
    profile, applied = apply_overrides(search_profile(s), {"agent_rounds": 999}, s)
    assert profile.agent_rounds == s.options_max_agent_rounds
    assert applied["agent_rounds"] == s.options_max_agent_rounds


def test_graph_chunk_cap_applies_in_search_including_an_explicit_zero():
    """An explicit 0 is a real value, not a clamp artifact: it switches graph
    chunk text off for the run (the agent is then built without the graph-text
    tools)."""
    s = make_settings()
    prof, applied = apply_overrides(search_profile(s), {"graph_chunk_cap": 5}, s)
    assert prof.graph_chunk_cap == 5 and applied["graph_chunk_cap"] == 5
    assert "ignored" not in applied

    prof0, applied0 = apply_overrides(search_profile(s), {"graph_chunk_cap": 0}, s)
    assert prof0.graph_chunk_cap == 0 and applied0["graph_chunk_cap"] == 0
    assert not (prof0.graph_text_tools and prof0.graph_chunk_cap > 0)


def test_no_overrides_returns_the_same_profile_and_effective_echo():
    s = make_settings()
    base = search_profile(s)
    profile, applied = apply_overrides(base, {}, s)
    assert profile is base  # untouched
    assert applied["agent_rounds"] == base.agent_rounds
    assert "ignored" not in applied


def test_acl_override_maps_disabled_acl_to_enforcement():
    from ai_graph_retrieval.api.routes.retrieve import _acl_override

    def body(**opts):
        return RetrieveRequest(question="q", mode="search", options=opts or None)

    assert _acl_override(body()) is None                      # omitted ⇒ defer to settings
    assert _acl_override(body(disabled_acl=True)) is False     # ACL off
    assert _acl_override(body(disabled_acl=False)) is True     # ACL on


def test_llm_with_settings_shares_heavy_resources():
    from ai_graph_retrieval.connectors.llm.client import LlmClient

    s = make_settings()
    base = LlmClient(s, asyncio.Semaphore(1))
    over = base.with_settings(s.model_copy(update={"llm_model": "other/model"}))
    assert over._s.llm_model == "other/model" and base._s.llm_model != "other/model"
    assert over._sem is base._sem                              # shared concurrency
    assert over._embeddings_client is base._embeddings_client  # no client rebuilt


def test_repository_principals_honors_per_request_acl_override():
    from ai_graph_retrieval.graph.repository import GraphReadRepository

    s = make_settings(acl_enforcement_enabled=False)
    repo = GraphReadRepository(None, s, http=object())  # http sentinel satisfies the ctor guard

    def ctx(acl_enforced):
        return AccessContext(
            tenant_id="t", principals=("public",), request_id="r", acl_enforced=acl_enforced
        )

    assert repo._principals(ctx(None)) is None       # defer to settings (off) ⇒ bypass
    assert repo._principals(ctx(True)) == ["public"]  # override ON ⇒ enforce
    assert repo._principals(ctx(False)) is None      # override OFF ⇒ bypass


def test_acl_is_enforced_by_default():
    from ai_graph_retrieval.graph.repository import GraphReadRepository

    repo = GraphReadRepository(None, make_settings(), http=object())
    ctx = AccessContext(tenant_id="t", principals=("public",), request_id="r")
    assert repo._principals(ctx) == ["public"]
