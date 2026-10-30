"""Structural retrieval: multi-hop must yield the CONNECTIONS, and the relationship
vocabulary must name what traversal can actually filter on.

Two gaps, one theme — the graph holds structure the retrieval layer could not hand
back. ``BEAM_TRAVERSAL`` walks a multi-hop path, binds it only to ACL-check it, then
returns the chunks it lands on ("the chunk is the product — the graph is only the
router"), so a question about the connection itself had no read that answers it.
And ``list_relationship_types`` enumerated EXTRACTED_REL alone while
``EXPAND_NEIGHBORS`` filters on ``coalesce(r.relationship_type, type(r))`` over every
edge — so the structured vocabulary (REPORTS_TO, MANAGES, AUTHORED, …) was absent from
the tool the agent is told to call before choosing a type.

``ENTITY_PATHS`` returns one RelationFact per HOP, which is the existing evidence
type: it joins the same pool and the same digest FACTS section as every other
relationship read, so nothing new has to render it.

    uv run pytest tests/test_structural_retrieval.py -v
"""
from __future__ import annotations

import asyncio
import math

from ai_graph_retrieval.core.access import AccessContext
from ai_graph_retrieval.graph import cypher
from ai_graph_retrieval.graph.repository import GraphReadRepository
from tests.support import make_settings

WIKIPEDIA, ARXIV = "wikipedia", "arxiv"


def _ctx():
    return AccessContext(tenant_id="t", principals=("public",), request_id="r")


def _repo(rows, settings=None):
    """A repository whose fan-out returns `rows`, capturing the bound params."""
    repo = GraphReadRepository(None, settings or make_settings(), http=object())
    seen: dict = {}

    async def _stub_rows(query, _ctx, **params):
        seen["query"] = query
        seen.update(params)
        return rows

    repo._rows = _stub_rows
    return repo, seen


def _hop(source: str, target: str, depth: int, *, rel: str = "RELATED_TO",
         support: int = 1, score: float = 0.1, graph: str = WIKIPEDIA) -> dict:
    return {"source": source, "target": target, "relationship": rel,
            "direction": "outgoing", "support": support, "score": score,
            "depth": depth, "graph": graph}


# ── 1. multi-hop returns the path, not only the text it ends at ─────────────


def test_multi_hop_returns_one_relationship_fact_per_hop():
    """The chain A → B → C comes back as two edges, each naming its endpoints —
    the answer can state the connection instead of inferring it from prose."""
    rows = [_hop("A", "B", 1), _hop("B", "C", 2)]
    repo, _ = _repo(rows)

    facts = asyncio.run(repo.entity_paths(_ctx(), ["A"]))

    assert [(f.source, f.target, f.depth) for f in facts] == [
        ("A", "B", 1), ("B", "C", 2),
    ]


def test_hops_are_returned_in_path_order():
    """Depth order is the chain: a distant hop must never precede the adjacent
    one it hangs off, however strong its support."""
    rows = [_hop("A", "B", 1, score=0.1), _hop("B", "C", 2, score=0.9)]
    repo, _ = _repo(rows)

    facts = asyncio.run(repo.entity_paths(_ctx(), ["A"]))

    assert [f.depth for f in facts] == [1, 2]


def test_an_undirected_walk_reports_each_edge_once_not_once_per_orientation():
    """The walk is undirected, so the same edge is reached in BOTH orientations
    (A-B-C and A-C-B both use B-C). A group keyed on the endpoint pair kept each
    orientation as its own row — the same fact twice in the pool's
    (source, relationship, target) key and twice rendered in the digest. The
    grouping is therefore on the relationship itself, and the emitted row is
    anchored on the edge's own stored direction, so it is one deterministic row."""
    q = cypher.ENTITY_PATHS
    # Orientation-blind: the group key is the relationship, never the pair.
    assert "WITH r, min(depth) AS depth" in q
    assert "WITH a, r, b, min(depth)" not in q
    # Anchored on the stored edge, so both orientations emit the SAME row.
    assert "WITH startNode(r) AS a, r, endNode(r) AS b" in q
    assert "'outgoing' AS direction" in q


def test_a_target_narrows_the_traversal_to_paths_reaching_it():
    """"How does A relate to B" binds $target; without one the read is the
    neighbourhood and the predicate is a no-op."""
    repo, seen = _repo([])
    asyncio.run(repo.entity_paths(_ctx(), ["A"], target="B"))
    assert seen["target"] == "B"

    repo, seen = _repo([])
    asyncio.run(repo.entity_paths(_ctx(), ["A"]))
    assert seen["target"] is None


def test_the_hop_bound_is_substituted_and_clamped():
    """A var-length bound cannot be a Cypher parameter, so it is substituted —
    and clamped, which is what keeps the substitution injection-safe."""
    repo, seen = _repo([])
    asyncio.run(repo.entity_paths(_ctx(), ["A"], hops=3))
    assert "*1..3]" in seen["query"] and "__HOPS__" not in seen["query"]

    repo, seen = _repo([])
    asyncio.run(repo.entity_paths(_ctx(), ["A"], hops=99))
    assert "*1..4]" in seen["query"]

    repo, seen = _repo([])
    asyncio.run(repo.entity_paths(_ctx(), ["A"], hops=0))
    assert "*1..1]" in seen["query"]


def test_the_default_hop_bound_is_the_configured_beam():
    """Shares ``beam_hops`` with the chunk traversal — one depth for the service."""
    settings = make_settings()
    repo, seen = _repo([], settings)
    asyncio.run(repo.entity_paths(_ctx(), ["A"]))

    assert f"*1..{settings.beam_hops}]" in seen["query"]


def test_no_seeds_is_a_no_op_read():
    repo, seen = _repo([_hop("A", "B", 1)])
    assert asyncio.run(repo.entity_paths(_ctx(), [])) == []
    assert not seen, "no query should run without a seed"


def test_path_hops_reach_the_evidence_pool_as_facts():
    """One RelationFact per hop IS the existing evidence type, so the digest's
    FACTS section renders a path with no new plumbing."""
    from ai_graph_retrieval.core.models import EvidencePool

    repo, _ = _repo([_hop("A", "B", 1), _hop("B", "C", 2)])
    facts = asyncio.run(repo.entity_paths(_ctx(), ["A"]))

    pool = EvidencePool()
    pool.add_facts(facts)
    assert len(pool.snapshot().facts) == 2


def test_path_scores_are_leveled_per_graph_like_every_other_edge_read():
    """`score` derives from `support`, a per-graph chunk count — the same damped
    cross-graph correction expand_entity applies: the dominant graph keeps the raw
    curve, the weaker one is lifted by sqrt(ceiling/peak)."""
    rows = [_hop("A", "B", 1, score=0.30, graph=WIKIPEDIA),
            _hop("A", "C", 1, score=0.15, graph=WIKIPEDIA),
            _hop("A", "D", 1, score=0.06, graph=ARXIV)]
    repo, _ = _repo(rows)

    facts = asyncio.run(repo.entity_paths(_ctx(), ["A"]))

    assert {f.target: f.score for f in facts} == {
        "B": 0.3, "C": 0.15, "D": round(0.06 * math.sqrt(0.30 / 0.06), 4),
    }


# ── 2. the traversal is ACL-gated at every hop ──────────────────────────────


def test_every_node_and_edge_on_the_path_is_acl_gated():
    """A path is visible only if all of it is — one hidden hop hides the path,
    the same fail-closed contract EXPAND_NEIGHBORS has."""
    prepared = cypher.prepare(
        cypher.ENTITY_PATHS.replace("__HOPS__", "2"), "allow_access_control")

    assert ("all(n IN nodes(path)" in prepared
            and "($principals IS NULL OR any(p IN n.allow_access_control "
                "WHERE p IN $principals))" in prepared)
    assert ("all(r IN relationships(path)" in prepared
            and "($principals IS NULL OR any(p IN r.allow_access_control "
                "WHERE p IN $principals))" in prepared)


def test_plumbing_nodes_never_appear_on_a_path():
    """Chunks, claims and community clusters are provenance, not user-facing
    structure — the same exclusion EXPAND_NEIGHBORS makes."""
    assert "NOT (n:Chunk OR n:Claim OR n:Community)" in cypher.ENTITY_PATHS


def test_the_type_filter_matches_what_expand_entity_filters_on():
    """Both reads must accept the SAME type names, or a type discovered for one
    silently returns nothing on the other."""
    predicate = "coalesce(r.relationship_type, type(r)) IN $types"
    assert predicate in cypher.ENTITY_PATHS
    assert predicate in cypher.EXPAND_NEIGHBORS


# ── 3. relationship-type discovery matches what traversal can use ───────────


def test_the_vocabulary_covers_structured_edges_not_only_extracted_ones():
    """The regression: listing EXTRACTED_REL alone hid REPORTS_TO, MANAGES,
    AUTHORED and friends from the tool that exists to name the choices."""
    assert "coalesce(r.relationship_type, type(r))" in cypher.RELATIONSHIP_TYPES
    assert "[r:EXTRACTED_REL]" not in cypher.RELATIONSHIP_TYPES


def test_the_vocabulary_and_the_traversal_read_the_same_expression():
    """One expression, two queries: whatever the vocabulary NAMES,
    expand_entity's `$types` filter can match."""
    expr = "coalesce(r.relationship_type, type(r))"
    assert expr in cypher.RELATIONSHIP_TYPES and expr in cypher.EXPAND_NEIGHBORS


def test_the_vocabulary_excludes_the_same_plumbing_the_traversal_does():
    """A type the traversal will never return is not a choice worth offering."""
    assert "NOT (nbr:Chunk OR nbr:Claim OR nbr:Community)" in cypher.RELATIONSHIP_TYPES


def test_the_vocabulary_counts_each_edge_once():
    """A directed match: `-[r]-` would traverse every edge from both endpoints
    and double every count."""
    assert "-[r]->(nbr)" in cypher.RELATIONSHIP_TYPES


def test_the_vocabulary_order_is_deterministic():
    """Equal counts must not reorder between identical requests."""
    assert "ORDER BY edges DESC, type" in cypher.RELATIONSHIP_TYPES


def test_the_vocabulary_read_is_still_cached_and_limited():
    """Widening what it returns must not change how often it is read."""
    settings = make_settings()
    repo, seen = _repo([{"type": "REPORTS_TO", "edges": 741}], settings)

    first = asyncio.run(repo.relationship_types(_ctx(), limit=10))
    seen.clear()
    second = asyncio.run(repo.relationship_types(_ctx(), limit=10))

    assert first == second
    assert not seen, "the second call must be served from the TTL cache"


# ── 4. no regression to one-hop or single-graph behaviour ───────────────────


def test_one_hop_expansion_still_carries_no_depth():
    """`depth` is set by the path read alone; expand_entity's contract is
    unchanged, and None is what it has always meant."""
    rows = [{"source": "A", "target": "B", "relationship": "R",
             "support": 3, "score": 0.5, "graph": WIKIPEDIA}]
    repo, _ = _repo(rows)

    facts = asyncio.run(repo.expand_entity(_ctx(), "A"))

    assert [f.depth for f in facts] == [None]
    assert (facts[0].source, facts[0].target) == ("A", "B")


def test_one_hop_expansion_still_binds_its_own_parameters():
    """The shared edge-description helper must not change what the read asks for."""
    settings = make_settings()
    repo, seen = _repo([], settings)
    asyncio.run(repo.expand_entity(_ctx(), "A", relationship_types=["REPORTS_TO"]))

    assert seen["name"] == "A"
    assert seen["types"] == ["REPORTS_TO"]
    assert seen["limit"] == settings.neighbor_limit


def _entry_rows():
    import json as _json

    entries = _json.dumps([
        {"text": "A mentors B", "acl": ["public"], "documented_on": "2026-01-01"},
        {"text": "hidden", "acl": ["secret"], "documented_on": "2030-01-01"},
    ])
    return [{"source": "A", "target": "B", "relationship": "R", "support": 1,
             "score": 0.5, "graph": WIKIPEDIA, "descriptions_json": entries}]


def test_one_hop_edge_descriptions_are_unchanged_by_the_shared_helper():
    """Enforcement OFF: every well-formed entry serves, and the date comes from
    them — the documented null-principals bypass, unchanged by the extraction."""
    repo, _ = _repo(_entry_rows(), make_settings(acl_enforcement_enabled=False))

    facts = asyncio.run(repo.expand_entity(_ctx(), "A"))

    assert facts[0].description == "A mentors B hidden"
    assert facts[0].date == "2030-01-01"
    assert facts[0].date_basis == "documented"


def test_the_shared_helper_still_filters_entries_under_acl():
    """Enforcement ON: the description AND the date may only come from an entry
    the caller can read — a hidden entry must not leak its text or its date."""
    settings = make_settings(acl_enforcement_enabled=True)
    repo, _ = _repo(_entry_rows(), settings)

    facts = asyncio.run(repo.expand_entity(_ctx(), "A"))

    assert facts[0].description == "A mentors B"
    assert facts[0].date == "2026-01-01"


def test_path_hops_filter_edge_entries_under_acl_too():
    """The path read shares the helper, so it inherits the same guarantee."""
    settings = make_settings(acl_enforcement_enabled=True)
    rows = [{**_entry_rows()[0], "depth": 1}]
    repo, _ = _repo(rows, settings)

    facts = asyncio.run(repo.entity_paths(_ctx(), ["A"]))

    assert facts[0].description == "A mentors B"
    assert facts[0].date == "2026-01-01"


def test_a_single_graph_path_keeps_its_query_order():
    """One graph ⇒ the rescale divides by that graph's own peak, so the
    depth-then-score order the query produced survives untouched."""
    rows = [_hop("A", "B", 1, score=0.30), _hop("B", "C", 2, score=0.20),
            _hop("C", "D", 3, score=0.10)]
    repo, _ = _repo(rows)

    facts = asyncio.run(repo.entity_paths(_ctx(), ["A"]))

    assert [f.target for f in facts] == ["B", "C", "D"]
    assert [f.depth for f in facts] == [1, 2, 3]


def test_the_chunk_traversal_is_untouched():
    """BEAM_TRAVERSAL still returns chunks — the deliberately designed
    graph-as-router path is not what this change replaced."""
    assert "RETURN c.chunk_id AS chunk_id" in cypher.BEAM_TRAVERSAL
    assert "via_names" in cypher.BEAM_TRAVERSAL


# ── 5. the tool surface ─────────────────────────────────────────────────────


def test_the_path_tool_is_available_in_every_agent_mode():
    """It returns STRUCTURE, not chunk text, so the graph_text gate — which
    removes the tools whose return is graph text — must not remove it."""
    from ai_graph_retrieval.agent.tools import research_tools

    for graph_text in (True, False):
        names = {t.name for t in research_tools(graph_text=graph_text)}
        assert "find_entity_paths" in names

    # ...while the chunk traversal stays gated exactly as before.
    assert "traverse_to_chunks" not in {
        t.name for t in research_tools(graph_text=False)}
    assert "traverse_to_chunks" in {
        t.name for t in research_tools(graph_text=True)}
