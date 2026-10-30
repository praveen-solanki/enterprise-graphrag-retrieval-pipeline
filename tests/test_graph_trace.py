"""Unit tests for GraphTraceBuilder — the typed, incremental UI graph.

Covers the merge contract every emission relies on: each merge_* returns ONLY the
newly-seen nodes/edges (dedup), the typed node kinds carry their stable id scheme,
and the cross-type edges (HAS_CHUNK, MENTIONS, ASSERTS/ABOUT, SUPPORTED_BY) are
emitted only for the real associations the evidence carries.

    uv run pytest tests/test_graph_trace.py -v
"""
from __future__ import annotations

from ai_graph_retrieval.core.models import (
    ChunkQuote,
    ClaimFact,
    EntityHit,
    GraphTraceBuilder,
    RelationFact,
    SourceDocument,
)


def _ids(nodes) -> set[str]:
    return {n.id for n in nodes}


def _by_label(edges) -> dict[str, list]:
    out: dict[str, list] = {}
    for e in edges:
        out.setdefault(e.label, []).append(e)
    return out


# ── entities + facts (pre-existing behaviour is preserved) ───────────────────

def test_merge_entities_dedupes_and_tags_kind():
    b = GraphTraceBuilder()
    added = b.merge_entities([EntityHit(name="Tushar", type="Person", score=0.9)])
    assert _ids(added) == {"Tushar"}
    assert added[0].kind == "entity"
    assert added[0].type == "Person"
    # a second merge of the same name reports NOTHING new (dedup)
    assert b.merge_entities([EntityHit(name="Tushar")]) == []


def test_merge_facts_creates_entity_endpoints_and_edge():
    b = GraphTraceBuilder()
    new_nodes, new_edges = b.merge_facts(
        [RelationFact(source="Tushar", target="Platform", relationship="MEMBER_OF")]
    )
    assert _ids(new_nodes) == {"Tushar", "Platform"}
    assert all(n.kind == "entity" for n in new_nodes)
    assert len(new_edges) == 1
    assert new_edges[0].label == "MEMBER_OF"
    # re-merging is a no-op
    assert b.merge_facts(
        [RelationFact(source="Tushar", target="Platform", relationship="MEMBER_OF")]
    ) == ([], [])


# ── documents ────────────────────────────────────────────────────────────────

def test_merge_sources_builds_document_nodes():
    b = GraphTraceBuilder()
    added = b.merge_sources([
        SourceDocument(doc_id="d1", title="Travel Policy", url="http://x/d1",
                       date="2026-01-01", graph="wikipedia", trust=0.7),
        SourceDocument(title="Untitled-by-title"),  # no doc_id → id falls back to title
    ])
    ids = _ids(added)
    assert "doc:d1" in ids
    assert "doc:Untitled-by-title" in ids
    doc = next(n for n in added if n.id == "doc:d1")
    assert doc.kind == "document"
    assert doc.label == "Travel Policy"
    assert doc.url == "http://x/d1"
    assert doc.graph == "wikipedia"
    # dedup
    assert b.merge_sources([SourceDocument(doc_id="d1", title="Travel Policy")]) == []


# ── chunks: HAS_CHUNK (doc→chunk) + MENTIONS (chunk→entity) ──────────────────

def test_merge_quotes_emits_chunk_node_has_chunk_and_mentions():
    b = GraphTraceBuilder()
    nodes, edges = b.merge_quotes([
        ChunkQuote(chunk_id="c1", doc_id="d1", doc_title="Travel Policy",
                   text="…", graph="wikipedia", via=["Tushar", "Platform"]),
    ])
    node_ids = _ids(nodes)
    # chunk node + the two mentioned entity nodes are created
    assert "chunk:c1" in node_ids
    assert {"Tushar", "Platform"} <= node_ids
    chunk = next(n for n in nodes if n.id == "chunk:c1")
    assert chunk.kind == "chunk"
    assert chunk.graph == "wikipedia"

    by_label = _by_label(edges)
    # doc→chunk HAS_CHUNK (doc_id known)
    assert by_label["HAS_CHUNK"][0].source == "doc:d1"
    assert by_label["HAS_CHUNK"][0].target == "chunk:c1"
    # chunk→entity MENTIONS for each via name
    mentions = {(e.source, e.target) for e in by_label["MENTIONS"]}
    assert mentions == {("chunk:c1", "Tushar"), ("chunk:c1", "Platform")}


def test_merge_quotes_without_doc_id_skips_has_chunk():
    b = GraphTraceBuilder()
    _nodes, edges = b.merge_quotes([ChunkQuote(chunk_id="c9", text="…")])
    assert _by_label(edges).get("HAS_CHUNK") is None


# ── claims: ASSERTS/ABOUT (entity→claim) + SUPPORTED_BY (claim→chunk) ─────────

def test_merge_claims_links_subject_object_and_supporting_chunks():
    b = GraphTraceBuilder()
    # object "Platform" pre-exists as an entity so the ABOUT edge is drawn;
    # a literal object would NOT be fabricated into a node.
    b.merge_entities([EntityHit(name="Platform", type="Team")])
    nodes, edges = b.merge_claims([
        ClaimFact(subject="Tushar", predicate="works_on", object="Platform",
                  graph="wikipedia", supporting_chunk_ids=["c1"]),
    ])
    node_ids = _ids(nodes)
    claim_id = "claim:Tushar|works_on|Platform"
    assert claim_id in node_ids
    assert "Tushar" in node_ids  # subject entity created
    claim = next(n for n in nodes if n.id == claim_id)
    assert claim.kind == "claim"

    by_label = _by_label(edges)
    assert (by_label["ASSERTS"][0].source, by_label["ASSERTS"][0].target) == ("Tushar", claim_id)
    assert (by_label["ABOUT"][0].source, by_label["ABOUT"][0].target) == ("Platform", claim_id)
    sb = by_label["SUPPORTED_BY"][0]
    assert (sb.source, sb.target) == (claim_id, "chunk:c1")


def test_merge_claims_literal_object_gets_no_about_edge():
    b = GraphTraceBuilder()
    _nodes, edges = b.merge_claims([
        ClaimFact(subject="Tushar", predicate="joined_on", object="2021-05-01"),
    ])
    # subject→claim ASSERTS exists, but the literal object gets no ABOUT edge
    labels = _by_label(edges)
    assert "ASSERTS" in labels
    assert "ABOUT" not in labels


# ── snapshot carries the full typed graph ────────────────────────────────────

def test_snapshot_accumulates_every_kind():
    b = GraphTraceBuilder()
    b.merge_entities([EntityHit(name="Tushar", type="Person")])
    b.merge_sources([SourceDocument(doc_id="d1", title="Doc")])
    b.merge_quotes([ChunkQuote(chunk_id="c1", doc_id="d1", text="…", via=["Tushar"])])
    b.merge_claims([ClaimFact(subject="Tushar", predicate="p", object="o",
                              supporting_chunk_ids=["c1"])])
    snap = b.snapshot()
    kinds = {n.kind for n in snap.nodes}
    assert {"entity", "document", "chunk", "claim"} <= kinds
    labels = {e.label for e in snap.edges}
    assert {"HAS_CHUNK", "MENTIONS", "ASSERTS", "SUPPORTED_BY"} <= labels


# ── regression: emitted edges never dangle (endpoints are always ensured) ────

def test_edges_never_dangle_when_endpoints_not_separately_merged():
    """merge_quotes/merge_claims must ensure their edge endpoints exist, so HAS_CHUNK,
    SUPPORTED_BY, and MENTIONS never reference a node absent from the graph — even when
    the document/chunk/entity was NOT merged via merge_sources/merge_quotes/merge_entities."""
    b = GraphTraceBuilder()
    # a quote whose document + mentioned entity were never merged, and a claim whose
    # supporting chunk was never merged as a quote:
    b.merge_quotes([ChunkQuote(chunk_id="c1", doc_id="d1", text="…", via=["Alice"])])
    b.merge_claims([ClaimFact(subject="Alice", predicate="wrote", object="Doc",
                              supporting_chunk_ids=["c2"])])
    snap = b.snapshot()
    node_ids = {n.id for n in snap.nodes}
    dangling = [(e.label, e.source, e.target) for e in snap.edges
                if e.source not in node_ids or e.target not in node_ids]
    assert dangling == [], f"dangling edges: {dangling}"
    assert {"doc:d1", "chunk:c1", "chunk:c2", "Alice"} <= node_ids


def test_host_only_fails_closed_on_an_unparseable_url():
    """`host_only` is a REDACTION helper: a value it cannot parse must be
    replaced wholesale, never returned verbatim — returning the input on a
    parse failure would log exactly the embedded credentials it exists to
    strip."""
    from unittest.mock import patch

    from ai_graph_retrieval.observability import trace_log

    # Fixture assembled at runtime so no source literal is credential-shaped
    # (a URL literal with embedded userinfo trips secret scanners on every
    # scan; the values here are placeholders, not credentials).
    userinfo = ":".join(("fake-user", "fake-pass"))
    assert trace_log.host_only(f"bolt://{userinfo}@neo4j:7687/db") == "bolt://neo4j:7687"
    assert trace_log.host_only("plain-hostname") == "plain-hostname"  # no scheme, no userinfo
    with patch.object(trace_log, "urlsplit", side_effect=ValueError("boom")):
        out = trace_log.host_only(f"bolt://{userinfo}@broken")
    assert "fake-pass" not in out
    assert out == "<unparseable-url>"
