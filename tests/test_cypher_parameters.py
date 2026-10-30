"""Every parameter a query references must be one the repository can bind.

This is a merge guard. Retrieval stopped binding ``$tenant`` when the graphs
became per-run databases (each already single-tenant), but a query merged in
from another branch kept its ``person.tenant_id = $tenant`` predicate — and
Neo4j answers an unbound parameter with ParameterMissing, so that read failed on
EVERY call while the rest of the request succeeded. It surfaced only as a
warning line (``profile_enrichment_failed``); the answer just quietly lost the
job titles and departments it was supposed to gain.

The unit suite can't catch this with fake graphs, and an EXPLAIN sweep can't
either if it supplies a value for every parameter it finds. So assert the
contract directly against the query text.
"""
from __future__ import annotations

from ai_graph_retrieval.graph import cypher

# Bound by GraphReadRepository._run_one on every single execution.
# `include_inactive_people` is the employment gate every person-bearing read
# carries (cypher's <<current_person:…>> / <<current_subject>>), bound per
# request from the question's temporal reading.
ALWAYS_BOUND = {"principals", "entity_index", "include_inactive_people"}
# Passed per call site by the repository method that owns each query.
CALL_SITE = {
    "ids", "names", "seeds", "name", "title", "doc_id", "lucene", "topic",
    "limit", "chars", "per_anchor", "leg_limit", "types", "since", "now",
    "support_weight", "recency_days", "recency_weight", "degree_weight",
    # PEOPLE_PROFILES_BY_IDENTITY (cross-graph person merge): bound by
    # GraphReadRepository._identity_groups.
    "emails",
    # ENTITY_PATHS: bound by GraphReadRepository.entity_paths. `target` narrows
    # the traversal to paths reaching one named entity ("how does A relate to
    # B"); None ⇒ the predicate is true and the read is the neighbourhood.
    "target",
    # DOCUMENTS_BY_AUTHOR: bound by
    # GraphReadRepository.documents_by_author. `who` is the person the edge is
    # walked from; `until` closes the published_at window whose lower bound is
    # the already-listed `since`.
    "who", "until",
    # DOCUMENT_LOOKUP: bound by GraphReadRepository.document_lookup.
    # `titles` takes the whole set in one read; `per_title` is its per-title row
    # budget, so one crowded title cannot spend the rows the others needed —
    # the same pairing as `names`/`per_anchor` above.
    "titles", "per_title",
    # MENTIONED_ENTITIES: bound by GraphReadRepository.mentioned_entities
    # — the pool's graph chunk ids, whose mentioned entities the post-loop pass
    # hydrates.
    "chunk_ids",
    # PEOPLE_BY_GROUP: the group (department/team/manager) whose members are
    # enumerated, bound by the repository read that owns the query.
    "group",
    # CITED_MENTIONS / CITED_EDGES: the answer's citations, bound by
    # GraphReadRepository.cited_subgraph for the answer's graph.
    "doc_ids",
}
ALLOWED = ALWAYS_BOUND | CALL_SITE


def _queries() -> dict[str, str]:
    return {
        name: value
        for name in dir(cypher)
        if name.isupper()
        and isinstance(value := getattr(cypher, name), str)
        and "RETURN" in value.upper()
    }


def test_no_query_references_the_retired_tenant_parameter():
    """$tenant is not bound anywhere any more — a query using it always fails."""
    offenders = [name for name, query in _queries().items() if "$tenant" in query]
    assert not offenders, (
        f"{offenders} reference $tenant, which the repository no longer binds; "
        "Neo4j raises ParameterMissing and the read fails on every call"
    )


def test_every_referenced_parameter_is_one_the_repository_binds():
    import re

    unknown: dict[str, set[str]] = {}
    for name, query in _queries().items():
        # The PREPARED text, not the source constant: the ACL null-guard and the
        # employment gate are both expanded at run time, so a parameter one of
        # them introduces is invisible in the constant and would reach Neo4j
        # unbound — exactly the failure this guard exists to catch.
        prepared = cypher.prepare(query, "allow_access_control")
        params = set(re.findall(r"\$(\w+)", prepared)) - ALLOWED
        if params:
            unknown[name] = params
    assert not unknown, (
        f"queries reference parameters nothing binds: {unknown}. Either bind them "
        "at the call site or drop the predicate — an unbound parameter is not a "
        "degraded read, it is a failed one."
    )


# ── edge-level ACL (new per-run schema) ──────────────────────────────────────
# Ingestion stamps allow_access_control on every EXTRACTED_REL edge as the union
# of its supporting chunks' ACLs (verified equal on the live per-run graphs), so
# traversal gates the EDGE directly instead of running a chunk EXISTS per edge.


def test_expand_neighbors_gates_the_edge_and_the_neighbor_not_only_the_node():
    prepared = cypher.prepare(cypher.EXPAND_NEIGHBORS, "allow_access_control")
    # Edge ACL present and null-guarded (one flag still toggles enforcement)…
    assert (
        "($principals IS NULL OR any(p IN r.allow_access_control WHERE p IN $principals))"
        in prepared
    )
    # …alongside the neighbour's own node ACL — edge alone is not enough.
    assert (
        "($principals IS NULL OR any(p IN nbr.allow_access_control WHERE p IN $principals))"
        in prepared
    )
    # The per-edge chunk EXISTS is gone from this query.
    assert "c.chunk_id IN r.chunk_ids" not in prepared


def test_structured_edges_survive_acl_despite_carrying_no_properties():
    """REPORTS_TO / AUTHORED / … carry no ACL property; a bare edge-ACL predicate
    would evaluate null and silently drop every structured edge under enforcement.
    The type guard keeps them gated by the neighbour's ACL instead."""
    assert "type(r) <> 'EXTRACTED_REL'" in cypher.EXPAND_NEIGHBORS


def test_beam_traversal_gates_every_hop_by_the_edge_acl():
    prepared = cypher.prepare(cypher.BEAM_TRAVERSAL, "allow_access_control")
    assert "all(r IN relationships(path)" in prepared
    assert (
        "($principals IS NULL OR any(p IN r.allow_access_control WHERE p IN $principals))"
        in prepared
    )
    # Only the beam's EXTRACTED_REL leg drops the chunk hop; the co-mention leg
    # still reaches neighbours THROUGH a visible chunk, which stays chunk-gated.
    assert prepared.count("allow_access_control") >= 3


# ── dense empty claim dates (new per-run schema, 2026-08-14) ──────────────────
# The per-run graphs store valid_from/valid_to on 100% of claims with "" for
# "no date"; the old graphs omitted the key. Measured live before the guards:
# the documented_on leg was dead, undated claims stamped date_basis='event',
# and valid_to="" made EVERY open-ended claim is_current=false — current facts
# shipped as historical.


def test_fetch_claims_treats_empty_string_dates_as_absent():
    q = cypher.FETCH_CLAIMS
    # Fallback and date_basis decide on the NORMALIZED event date…
    assert "coalesce(nullIf(cl.valid_from, ''), nullIf(cl.documented_on, ''))" in q
    assert "CASE WHEN nullIf(cl.valid_from, '') IS NOT NULL" in q
    # …and is_current treats "" exactly like an absent bound.
    assert "nullIf(cl.valid_from, '') IS NULL OR cl.valid_from <= $now" in q
    assert "nullIf(cl.valid_to, '') IS NULL OR cl.valid_to >= $now" in q
    # No bare IS NULL date test survives anywhere in the query.
    assert "cl.valid_from IS NULL OR" not in q.replace(
        "nullIf(cl.valid_from, '') IS NULL OR", "")


# ── person-anchored authorship ───────────────────────────────
# The RCA case: DOCUMENT_CONTRIBUTORS is keyed on d.title, so asking it about a
# person returns documents TITLED after them. DOCUMENTS_BY_AUTHOR walks the same
# edge from the person. Both must stay ACL-gated on the document, and the window
# must reject undated rows rather than let them satisfy a lower bound.


def test_documents_by_author_gates_on_the_document_acl():
    """Authorship is visible iff the document is — the same rule the
    document-anchored direction already enforces."""
    prepared = cypher.prepare(cypher.DOCUMENTS_BY_AUTHOR, "allow_access_control")
    assert (
        "($principals IS NULL OR any(p IN d.allow_access_control WHERE p IN $principals))"
        in prepared
    )


def test_documents_by_author_walks_from_the_person_not_the_title():
    """The whole point of the query: the anchor is the PERSON. A title predicate
    here would recreate the bug it exists to fix."""
    q = cypher.DOCUMENTS_BY_AUTHOR
    assert "CONTAINS toLower($who)" in q
    assert "person.person_full_name" in q
    assert "toLower(d.title) CONTAINS" not in q


def test_documents_by_author_window_excludes_undated_documents():
    """An empty-string date sorts BELOW every real one, so a
    bare `>= $since` would let an undated document into every window."""
    q = cypher.DOCUMENTS_BY_AUTHOR
    assert "($since = '' OR (d.published_at >= $since AND d.published_at <> ''))" in q
    assert "($until = '' OR (d.published_at <= $until AND d.published_at <> ''))" in q


def test_documents_by_author_returns_the_date_so_a_window_can_be_shown():
    """The answer claimed no publication date was available while published_at was
    100% populated; the date has to travel with the row."""
    assert "d.published_at AS date" in cypher.DOCUMENTS_BY_AUTHOR
    assert "ORDER BY d.published_at DESC" in cypher.DOCUMENTS_BY_AUTHOR


def test_recent_change_documented_legs_match_dense_and_sparse_no_date():
    """The documented_on leg must fire for BOTH storage styles — key absent
    (old graphs) and key == '' (per-run graphs) — while the range predicates
    stay bare properties so each leg keeps its single-property index seek."""
    for q in (cypher.RECENT_CHANGES, cypher.RECENT_CHANGE_CHUNKS):
        assert "(cl.valid_from IS NULL OR cl.valid_from = '')" in q
        assert "cl.documented_on >= $since" in q
        assert "cl.valid_from >= $since" in q          # event leg: bare, seekable
        assert "nullIf(cl.valid_from, '') >= $since" not in q
        assert "nullIf(cl.documented_on, '') >= $since" not in q
