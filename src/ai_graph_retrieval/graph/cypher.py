"""Every Cypher statement in the service, in one reviewable file.

Security invariants (docs/acl/neo4j-graph-acl-design.md §3.3):
  * every derived element (Entity / relationship / Claim) is
    reachable only through a **visible chunk** — a Chunk whose ``allow_access_control``
    intersects ``$principals``. Multi-chunk provenance is a **union**: an edge/claim is
    visible iff ANY of its supporting chunks is visible;
  * Documents/Chunks are filtered on their own ACL (``allow_access_control``) directly;
  * there is NO free-form Cypher entry point anywhere in this service.

Tenant isolation is **not** applied in Cypher: each Neo4j database / Qdrant
collection is already scoped to a single tenant+run, so every node shares the
same ``tenant_id``. Filtering on ``$tenant`` was redundant.

Guards are composed via explicit placeholders (``<<chunk_visible>>``) instead of
str.format, because Cypher's ``{prop: $param}`` braces collide with Python
formatting. The repository injects ``$principals`` on every run.

ACL predicates are *null-guarded* (see ``acl_guarded`` below): bind ``$principals``
to the caller's token list to enforce ACL, or to ``None`` to disable
(``acl_enforcement_enabled``).
"""
from functools import cache
from re import compile as _re_compile

# A chunk the caller may read (`c` in scope).
_CHUNK_VISIBLE = "any(p IN c.allow_access_control WHERE p IN $principals)"

# Every ACL predicate in this file has the uniform shape
# ``any(p IN <var>.allow_access_control WHERE p IN $principals)`` (chunk, entity,
# neighbour, and document variants alike). The leading lookbehind keeps the
# rewrite idempotent — a predicate already wrapped in ``$principals IS NULL OR``
# is skipped on a second pass.
_ACL_PREDICATE = _re_compile(
    r"(?<!IS NULL OR )any\(p IN (\w+)\.allow_access_control WHERE p IN \$principals\)"
)


@cache
def acl_guarded(query: str) -> str:
    """Null-guard every ACL predicate so a single flag toggles enforcement.

    ``any(p IN x.allow_access_control WHERE p IN $principals)`` →
    ``($principals IS NULL OR any(p IN x.allow_access_control WHERE p IN $principals))``.

    Bind ``$principals`` to the caller's token list to enforce ACL, or to ``None``
    (predicate short-circuits true) to disable it — controlled by the repository
    from ``acl_enforcement_enabled``. Memoized.
    """
    return _ACL_PREDICATE.sub(
        r"($principals IS NULL OR any(p IN \g<1>.allow_access_control WHERE p IN $principals))",
        query,
    )


# WHO STILL WORKS HERE — the retrieval-side employment gate, written as
# ``<<current_person:VAR>>`` beside the ACL guard on every person-bearing read
# and expanded here. ``$include_inactive_people`` is bound on every execution
# (repository._run_one) from the request's own reading of the question, so ONE
# parameter switches the whole service between "current people only" and the
# historical view — no per-query variants, exactly as $principals does for ACL.
#
# UNKNOWN IS CURRENT, deliberately. ``coalesce(x, 'active')`` keeps every node
# that carries no status: some connectors write person nodes as identity stubs
# with no attributes at all, and a non-Person neighbour (a Document, a concept) has no
# such property either — reading absence as departure would empty both. The
# semantics are the same ones retrieval/person_status.is_inactive applies to a
# row, stated once per side of the wire.
_CURRENT_PERSON = _re_compile(r"<<current_person:(\w+)>>")
_CURRENT_PERSON_PREDICATE = (
    "($include_inactive_people OR ("
    r"toLower(coalesce(\g<1>.person_employment_status, 'active')) = 'active'"
    r" AND coalesce(\g<1>.person_departed_on, '') = ''))"
)

# The same gate for a chunk/document whose SUBJECT is a person — a people-object
# document (object_type='people'), whose title IS the person's name and whose
# text is their profile. The statuses arrive as the collected lists `ss`/`sd`
# (several Person nodes can share one people object), and ANY departed one
# condemns the row.
_CURRENT_SUBJECT = (
    "($include_inactive_people OR ("
    "none(s IN ss WHERE s IS NOT NULL AND toLower(s) <> 'active')"
    " AND none(x IN sd WHERE coalesce(x, '') <> '')))"
)


@cache
def prepare(query: str, acl_property: str) -> str:
    """Run-time query preparation: ACL null-guard + ACL property naming.

    The queries are written against the target contract's ``allow_access_control``;
    older graphs (the POC-lineage local build) store the same array under ``acl``.
    ``graph_acl_property`` renames the property at run time — memoized per
    (query, property) so it costs one pass ever."""
    guarded = _CURRENT_PERSON.sub(_CURRENT_PERSON_PREDICATE, acl_guarded(query))
    if acl_property != "allow_access_control":
        guarded = guarded.replace("allow_access_control", acl_property)
    return guarded


# The chunk→entity provenance leg, shared by every chunk query that surfaces
# ``via_names`` (the entities a visible chunk MENTIONS) for the trace's MENTIONS edges.
# The mentioned entity is reached THROUGH the already-``<<chunk_visible>>``-gated
# chunk, so no separate entity ACL check is needed.
_CHUNK_MENTIONS = "OPTIONAL MATCH (c)-[:MENTIONS]->(me:Entity)"


def _q(text: str) -> str:
    return (
        text.replace("<<chunk_visible>>", _CHUNK_VISIBLE)
        .replace("<<mentions>>", _CHUNK_MENTIONS)
        .replace("<<current_subject>>", _CURRENT_SUBJECT)
    )

# Entity is visible ⇔ the caller matches its stamped ACL. Ingestion maintains
# `allow_access_control` on every Entity as the UNION of its contributing
# documents' ACLs (recomputed on retraction) and creates the
# `entity_allow_access_control` index for exactly this check — equivalent to the
# old chunk-transitive EXISTS subquery, but indexed. Fail-closed: an entity
# without the property never matches.
_ENTITY_VISIBLE = "any(p IN e.allow_access_control WHERE p IN $principals)"


def _e(text: str) -> str:
    return _q(text).replace("<<entity_visible>>", _ENTITY_VISIBLE)


# ── Entity resolution ───────────────────────────────────────────────────────

# Vector-arm hydration. Entity embeddings live in QDRANT (`entity_embeddings`
# collection; Neo4j carries only a `has_vector` flag — there is NO Neo4j vector
# index). The repository runs the ANN in Qdrant, then hydrates the returned
# entity_ids here, ACL re-checked (defense in depth on the Qdrant push-down).
# `descriptions_json` is parsed + per-entry ACL-filtered client-side.
ENTITIES_BY_ID = _e("""
MATCH (e:Entity)
WHERE e.entity_id IN $ids AND <<entity_visible>> AND <<current_person:e>>
RETURN e.entity_id AS entity_id, e.name AS name, e.type AS type,
       e.description AS description, e.descriptions_json AS descriptions_json,
       coalesce(e.degree, 0) AS degree,
       e.person_employment_status AS employment_status,
       e.allow_access_control AS _acl_log
""")

# Lexical arm — fuzzy full-text, typo tolerant. `entity_search` covers name +
# aliases + person_search_terms (ingestion also builds the narrower
# `entity_aliases`; this one is the superset built for retrieval).
RESOLVE_BY_FULLTEXT = _e("""
CALL db.index.fulltext.queryNodes($entity_index, $lucene)
YIELD node AS e, score
WHERE 'Entity' IN labels(e) AND <<entity_visible>> AND <<current_person:e>>
RETURN e.entity_id AS entity_id, e.name AS name, e.type AS type,
       e.description AS description, e.descriptions_json AS descriptions_json,
       score AS score,
       coalesce(e.degree, 0) AS degree,
       e.person_employment_status AS employment_status,
       e.allow_access_control AS _acl_log
LIMIT $limit
""")

# ── Traversal ───────────────────────────────────────────────────────────────

# Neighbors of one entity, ranked by evidence support (+ neighbor degree as the
# tiebreak). Entity nodes carry NO embedding property (vectors live in Qdrant),
# so there is no cosine term here. Descriptions and dates live per-source inside
# `descriptions_json` (each entry ACL-tagged) — parsed + filtered client-side.
# The EDGE is anchored on its own source chunk, and the NEIGHBOR must be visible.
EXPAND_NEIGHBORS = _q("""
MATCH (e:Entity {name: $name})-[r]-(nbr)
  // The seed itself must be current: a departed person is not an anchor for
  // current evidence, and every fact this read emits carries their name as
  // its source. Same gate the neighbour gets below.
  WHERE <<current_person:e>>
  // Traverse EVERY relationship type off the entity, across both graphs — the
  // LLM-EXTRACTED_REL edges AND the structured ones (REPORTS_TO, MANAGES,
  // CONTRIBUTED_TO, AUTHORED, EDITED, AUTHOR_OF, MEMBER_OF, …). Only chunk/claim
  // provenance and internal community-clustering neighbours are skipped: those
  // are plumbing (surfaced via chunks/claims), not user-facing entity facts.
  AND NOT (nbr:Chunk OR nbr:Claim OR nbr:Community)
  AND ($types IS NULL OR coalesce(r.relationship_type, type(r)) IN $types)
  // EDGE-level ACL: ingestion stamps allow_access_control on every EXTRACTED_REL
  // edge as the UNION of its supporting chunks' ACLs (verified equal on the live
  // per-run graphs), so the edge is matched directly — same contract as entities,
  // a property check instead of a chunk EXISTS subquery per edge. The structured
  // edges (REPORTS_TO, AUTHORED, …) carry no properties at all and stay gated by
  // the neighbour's own ACL below; a null edge ACL must not kill them.
  AND (type(r) <> 'EXTRACTED_REL'
       OR any(p IN r.allow_access_control WHERE p IN $principals))
  // NODE-level ACL: the edge alone is not enough — the neighbour itself must be
  // visible to the caller, or a permitted edge would name a forbidden entity.
  AND any(p IN nbr.allow_access_control WHERE p IN $principals)
  // A neighbour who has left the organization is not a current connection, and
  // must not spend one of this read's LIMIT rows. Non-person neighbours carry no
  // status and are unaffected (see _CURRENT_PERSON).
  AND <<current_person:nbr>>
WITH e, r, nbr,
     coalesce(r.support, 1) AS support,
     // A schema-asserted edge, told apart by the ABSENCE of `support` — the
     // property relation extraction writes and the structured edges never carry.
     // Not a type list: a new structured verb is structural on arrival.
     r.support IS NULL AS structural,
     CASE WHEN startNode(r) = e THEN 'outgoing' ELSE 'incoming' END AS direction
RETURN e.name AS source,
       coalesce(nbr.name, nbr.title, nbr.person_full_name) AS target,
       coalesce(nbr.type, head([l IN labels(nbr) WHERE l <> 'Entity'])) AS target_type,
       nbr.description AS target_description_plain,
       nbr.descriptions_json AS target_descriptions_json,
       nbr.person_employment_status AS target_status,
       coalesce(r.relationship_type, type(r)) AS relationship,
       r.raw_types AS specific_relationships,
       r.description AS rel_description, r.descriptions_json AS descriptions_json,
       direction, support, structural,
       // The passages the edge was extracted from (absent on structural edges):
       // provenance for the answer's graph, never a ranking input.
       r.chunk_ids AS chunk_ids,
       round($support_weight * log10(1.0 + toFloat(support)), 4) AS score,
       nbr.allow_access_control AS _acl_log
// Structure FIRST. Every structural edge scores at the curve's floor (support is
// absent, so coalesce gives 1 and log10(2) is the minimum the curve produces), so
// they all tied and the ORDER BY degenerated to the neighbour-degree tiebreak —
// which ranks a high-degree CONCEPT above a Person, and puts the membership and
// reporting edges of an org question at the bottom of the LIMIT. Measured on one
// team entity: 112 edges returned, ALL at score 0.0452, 4 of them membership.
ORDER BY structural DESC, score DESC, coalesce(nbr.degree, 0) DESC
LIMIT $limit
""")

# ── Claims ──────────────────────────────────────────────────────────────────

# ABOUT-linked claims plus orphan claims keyed by subject. NOTE: ingestion
# writes `cl.subject` as the subject's ENTITY_ID (not its name), so the orphan
# arm matches on e.entity_id and the display subject returned is e.name. Claims
# remain visible only through their own source chunk.
#
# Validity: rather than dropping expired/future claims (the POC filtered them
# out, losing historical recall), each row carries `is_current` computed against
# $now so synthesis can present expired claims as historical and future-dated
# ones as predictions — never as current truth.
#
# DENSE EMPTY DATES: the per-run graphs store claim dates DENSELY —
# `valid_from`/`valid_to` exist on 100% of claims, `""` when the fact carries no
# date — while the old graphs stored them SPARSELY (no key at all). Every date
# read therefore normalizes `""` to NULL via nullIf(); a bare IS NULL / bare
# coalesce() treats `""` as a real date, which (measured live) killed the
# documented_on fallback, stamped date_basis='event' on undated claims, and —
# worst — made `valid_to >= $now` false for EVERY open-ended claim, so current
# facts shipped as is_current=false and synthesis presented them as historical.
FETCH_CLAIMS = _q("""
MATCH (e:Entity {name: $name})
  WHERE <<current_person:e>>
MATCH (cl:Claim)
WHERE ((cl)-[:ABOUT]->(e) OR cl.subject = e.entity_id OR cl.subject = e.name)
  AND EXISTS { MATCH (cl)-[:SUPPORTED_BY]->(c:Chunk) WHERE <<chunk_visible>> }
  AND ($since IS NULL
       OR coalesce(nullIf(cl.valid_from, ''), nullIf(cl.documented_on, '')) >= $since)
// Surface the visible supporting chunks (same (:Claim)-[:SUPPORTED_BY]->(:Chunk)
// provenance the EXISTS above gates on) so the trace can draw claim→chunk edges.
OPTIONAL MATCH (cl)-[:SUPPORTED_BY]->(c:Chunk) WHERE <<chunk_visible>>
WITH e, cl, collect(DISTINCT c.chunk_id) AS supporting_chunk_ids
// DISTINCT: $name can bind several same-named Entity nodes, each pairing with the same
// Claim to emit an identical tuple; without it those duplicates consume LIMIT slots.
RETURN DISTINCT e.name AS subject, cl.predicate AS predicate, cl.object AS object,
       cl.statement_type AS statement_type, cl.temporal_type AS temporal_type,
       coalesce(nullIf(cl.valid_from, ''), nullIf(cl.documented_on, '')) AS valid_from,
       nullIf(cl.valid_to, '') AS valid_to,
       CASE WHEN nullIf(cl.valid_from, '') IS NOT NULL
            THEN 'event' ELSE 'documented' END AS date_basis,
       coalesce(cl.support, 1) AS support,
       ((nullIf(cl.valid_from, '') IS NULL OR cl.valid_from <= $now)
        AND (nullIf(cl.valid_to, '') IS NULL OR cl.valid_to >= $now)) AS is_current,
       supporting_chunk_ids
ORDER BY coalesce(valid_from, '') DESC
LIMIT $limit
""")

# ── Temporal ────────────────────────────────────────────────────────────────

# CLAIM-anchored (matches what ingestion actually writes): EXTRACTED_REL edges
# carry no date properties — dated facts are :Claim nodes with `valid_from` /
# `documented_on`, both indexed (`claim_valid_from` / `claim_documented_on`).
# Two UNION legs so each date predicate uses its own single-property index
# (coalesce() over two properties can use neither — POC fix D14); the legs are
# disjoint (the documented leg only takes rows with no event date) so UNION
# never double-counts. Ordered by recency — entities carry no embedding for a
# similarity term (vectors live in Qdrant).
RECENT_CHANGES = _q("""
CALL () {
  MATCH (cl:Claim)
  // "" can never pass a >= $since range test, so the event leg needs no
  // emptiness guard — and the bare property keeps the claim_valid_from
  // index seek (a function around the property would forfeit it).
  WHERE cl.valid_from >= $since
  RETURN cl, cl.valid_from AS eff_date, 'event' AS date_basis
  UNION
  MATCH (cl:Claim)
  // Dense empty dates (see FETCH_CLAIMS): the per-run graphs write "" where
  // the old graphs omitted the key, so "no event date" must match BOTH — a
  // bare IS NULL left this whole leg dead on exactly the graphs where
  // documented_on finally exists (100% vs 2 claims). The documented_on range
  // stays a bare property for its own index seek.
  WHERE (cl.valid_from IS NULL OR cl.valid_from = '')
    AND cl.documented_on >= $since
  RETURN cl, cl.documented_on AS eff_date, 'documented' AS date_basis
}
WITH cl, eff_date, date_basis
WHERE EXISTS { MATCH (cl)-[:SUPPORTED_BY]->(c:Chunk) WHERE <<chunk_visible>> }
MATCH (cl)-[:ABOUT]->(e:Entity)
  WHERE <<current_person:e>>
RETURN e.name AS source, coalesce(cl.object, '') AS target, NULL AS target_type,
       cl.predicate AS relationship, NULL AS specific_relationships,
       NULL AS description, eff_date AS date, date_basis,
       coalesce(cl.support, 1) AS support, 0.0 AS score
ORDER BY date DESC
LIMIT $limit
""")

# ── Grounding: sources + verbatim quotes ────────────────────────────────────

# Documents are filtered on their OWN ACL — these feed user-facing citations.
#
# PER-ANCHOR budget, not one global LIMIT. The previous shape matched every name at
# once, ranked by TOTAL mention count and cut at a fixed limit — so adding an anchor
# entity could EVICT the documents of the entity that actually matched the question.
# Measured on dev: an anchor with 11 documents kept 9 of them when it was the only
# name, and 5 when four more names were added, because those names' documents
# outranked its tail. That made "grounding wider" non-monotonic — a deeper mode could
# legitimately return FEWER of the right documents than a shallower one.
#
# Now each anchor contributes its own top ``$per_anchor`` documents and the union is
# de-duplicated, so adding a name can only ADD documents. The caller sizes ``$limit``
# from ``$per_anchor * len($names)`` so the overall ceiling can never bind and
# re-introduce the eviction.
SOURCE_DOCUMENTS = """
UNWIND $names AS anchor
MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)-[:MENTIONS]->(e:Entity {name: anchor})
  WHERE any(p IN d.allow_access_control WHERE p IN $principals)
  AND <<current_person:e>>
WITH anchor, d, count(DISTINCT c) AS mentions
ORDER BY mentions DESC
WITH anchor, collect({doc_id: d.doc_id, title: d.title, url: d.url,
                      date: d.published_at, source_type: d.source_type,
                      // Opening of the document, stamped by ingestion, for the
                      // citation hover preview. Absent until ingestion ships it —
                      // null here simply means the preview falls back to the
                      // retrieved excerpt (retrieval/quality.py::_snippet).
                      snippet: d.doc_preview_text,
                      acl: d.allow_access_control,
                      mentions: mentions})[0..$per_anchor] AS docs
UNWIND docs AS row
WITH row.doc_id AS doc_id, row.title AS title, row.url AS url, row.date AS date,
     row.source_type AS source_type, row.snippet AS snippet,
     max(row.mentions) AS mentions,
     head(collect(row.acl)) AS _acl_log
RETURN doc_id, title, url, date, source_type, snippet, mentions, _acl_log
ORDER BY mentions DESC
LIMIT $limit
"""

# Hydrate chunk ids returned by the Qdrant CHUNK ANN, ACL re-checked here as
# defense in depth on the push-down — the same contract ENTITIES_BY_ID has.
#
# This is the only chunk query with no traversal: every other one starts from a
# resolved entity, so a question entity resolution finds nothing for could
# reach no graph text at all. Similarity needs no anchor, so it answers that.
#
# The ids come from the chunk collection's `entity_id` payload field, whose value
# is the chunk's `chunk_id` despite the name (the Qdrant POINT id is not).
#
# RETURN mirrors SUPPORTING_CHUNKS exactly — including the Person join for
# `subject_status` — so `_chunk_quote` builds an identical ChunkQuote whichever
# query produced the row, and synthesis can still flag a former employee.
#
# `via_names` is the chunk's real MENTIONS (the shared `<<mentions>>` leg), not
# an empty list: a passage found by similarity is the one chunk path with no
# anchor entity, so without this the entities it is ABOUT — the person a
# "who is the security engineer" answer names — were the only ones in the pool
# with no chunk→entity edge, and the post-loop mention pass had nothing to
# hydrate them from.
CHUNKS_BY_ID = _q("""
MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)
WHERE c.chunk_id IN $ids AND <<chunk_visible>>
OPTIONAL MATCH (subject:Person {person_user_id: d.doc_id})
  WHERE d.object_type = 'people'
<<mentions>>
WITH c, d, collect(DISTINCT subject.person_employment_status) AS ss,
     collect(DISTINCT subject.person_departed_on) AS sd,
     collect(DISTINCT me.name) AS via_names
WHERE <<current_subject>>
RETURN c.chunk_id AS chunk_id, d.doc_id AS doc_id, d.title AS doc_title,
       left(c.text, $chars) AS text,
       CASE WHEN 'Inactive' IN ss THEN 'Inactive'
            WHEN 'Active' IN ss THEN 'Active' ELSE null END AS subject_status,
       via_names,
       c.allow_access_control AS _acl_log
""")

# The entities the pool's GRAPH CHUNKS mention, hydrated — the post-loop mention pass
# (retrieval/deep_research.enrich_mentions). Chunks reach the pool through similarity,
# document reads and traversal, and the entity a passage is about arrives with it only
# as a NAME in `via_names`; nothing added it as an entity, so the answer could name a
# person the graph knows well while `evidence.entities` carried none of them (the answer
# named several people while the pool held none of them). Each row carries WHICH of the
# given chunks mention the entity, so the repository can rank by the question-similarity
# of those chunks (the score the digest already ranks the chunks by) — the entity a
# high-relevance passage is about outranks one that corporate boilerplate mentions forty
# times. `$limit` is a ceiling on candidates, not the ranking. Same ACL contract as
# ENTITIES_BY_ID: the entity itself must be visible.
MENTIONED_ENTITIES = _e("""
MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
WHERE c.chunk_id IN $chunk_ids AND <<entity_visible>>
WITH e, collect(DISTINCT c.chunk_id) AS chunk_ids
RETURN e.entity_id AS entity_id, e.name AS name, e.type AS type,
       e.description AS description, e.descriptions_json AS descriptions_json,
       coalesce(e.degree, 0) AS degree, chunk_ids,
       e.allow_access_control AS _acl_log
ORDER BY size(chunk_ids) DESC, degree DESC
LIMIT $limit
""")

# The relationships AMONG a set of entities — the induced subgraph of the pool's
# own entities, for the same post-loop pass. The agent's expand tool walks ONE
# entity's neighbourhood per call, so two entities the evidence found separately
# (a person from a passage, their manager from a profile) sit in the pool with no
# edge between them however directly the graph links them. One read over the
# whole set closes that: every edge whose BOTH endpoints are already evidence.
# RETURN mirrors EXPAND_NEIGHBORS exactly so the repository maps rows through the
# same `_relation_fact` path (ACL-filtered descriptions included). `a.name <
# b.name` emits each undirected pair once; `direction` is still relative to
# `source`, so the fact reads truthfully either way round. Same edge + node ACL
# contract as EXPAND_NEIGHBORS.
FACTS_AMONG = _q("""
MATCH (e:Entity)-[r]-(nbr:Entity)
  WHERE e.name IN $names AND nbr.name IN $names AND e.name < nbr.name
  AND (type(r) <> 'EXTRACTED_REL'
       OR any(p IN r.allow_access_control WHERE p IN $principals))
  AND any(p IN e.allow_access_control WHERE p IN $principals)
  AND any(p IN nbr.allow_access_control WHERE p IN $principals)
WITH e, r, nbr,
     coalesce(r.support, 1) AS support,
     CASE WHEN startNode(r) = e THEN 'outgoing' ELSE 'incoming' END AS direction
RETURN e.name AS source,
       coalesce(nbr.name, nbr.title, nbr.person_full_name) AS target,
       coalesce(nbr.type, head([l IN labels(nbr) WHERE l <> 'Entity'])) AS target_type,
       nbr.description AS target_description_plain,
       nbr.descriptions_json AS target_descriptions_json,
       nbr.person_employment_status AS target_status,
       coalesce(r.relationship_type, type(r)) AS relationship,
       r.raw_types AS specific_relationships,
       r.description AS rel_description, r.descriptions_json AS descriptions_json,
       direction, support, r.chunk_ids AS chunk_ids,
       round($support_weight * log10(1.0 + toFloat(support)), 4) AS score,
       nbr.allow_access_control AS _acl_log
ORDER BY score DESC, coalesce(nbr.degree, 0) DESC
LIMIT $limit
""")

# ── The cited documents' own subgraph (synthesis/answer_graph.py) ────────────
#
# WHY. The answer is written from cited documents, and the graph knows those
# same documents: their chunks MENTION the entities the passages are about, and
# every EXTRACTED_REL edge records the chunks it was read from (`r.chunk_ids`).
# So "which entities and relationships did the answer actually use" is a join,
# not a guess: the cited documents' visible chunks → what they mention → the
# edges extracted from exactly those chunks. Two reads because the two shapes
# (an entity row, an edge row) render differently; both are bounded by
# $doc_ids (the answer's citations, a handful) and a row ceiling.
#
# A document the graph never ingested simply matches nothing, and the answer's
# graph is built from the pool alone.
CITED_MENTIONS = _e("""
MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)-[:MENTIONS]->(e:Entity)
WHERE d.doc_id IN $doc_ids AND <<chunk_visible>> AND <<entity_visible>>
  AND <<current_person:e>>
WITH e, collect(DISTINCT d.doc_id) AS doc_ids, collect(DISTINCT c.chunk_id) AS chunk_ids
RETURN e.entity_id AS entity_id, e.name AS name, e.type AS type,
       e.description AS description, e.descriptions_json AS descriptions_json,
       coalesce(e.degree, 0) AS degree,
       e.person_employment_status AS employment_status,
       doc_ids, chunk_ids,
       e.allow_access_control AS _acl_log
ORDER BY size(chunk_ids) DESC, name
LIMIT $limit
""")

# Edges EXTRACTED FROM the cited passages: anchored on a mentioned entity (an
# edge's endpoints are mentioned by the chunk it was read from), then filtered to
# the edges whose own `chunk_ids` include that chunk. Same edge + node ACL
# contract as EXPAND_NEIGHBORS; `direction` is a constant because the pattern is
# directed on the stored edge.
CITED_EDGES = _q("""
MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)-[:MENTIONS]->(a:Entity)
WHERE d.doc_id IN $doc_ids AND <<chunk_visible>>
  AND any(p IN a.allow_access_control WHERE p IN $principals)
  AND <<current_person:a>>
MATCH (a)-[r:EXTRACTED_REL]->(b:Entity)
WHERE c.chunk_id IN r.chunk_ids
  AND any(p IN r.allow_access_control WHERE p IN $principals)
  AND any(p IN b.allow_access_control WHERE p IN $principals)
  AND <<current_person:b>>
WITH a, r, b, collect(DISTINCT d.doc_id) AS doc_ids, collect(DISTINCT c.chunk_id) AS chunk_ids
RETURN a.name AS source,
       coalesce(b.name, b.title, b.person_full_name) AS target,
       coalesce(b.type, head([l IN labels(b) WHERE l <> 'Entity'])) AS target_type,
       b.description AS target_description_plain,
       b.descriptions_json AS target_descriptions_json,
       b.person_employment_status AS target_status,
       coalesce(r.relationship_type, type(r)) AS relationship,
       r.raw_types AS specific_relationships,
       r.description AS rel_description, r.descriptions_json AS descriptions_json,
       'outgoing' AS direction, coalesce(r.support, 1) AS support,
       chunk_ids, doc_ids,
       round($support_weight * log10(1.0 + toFloat(coalesce(r.support, 1))), 4) AS score,
       b.allow_access_control AS _acl_log
ORDER BY size(chunk_ids) DESC, score DESC, source, target
LIMIT $limit
""")

# The cited documents' PASSAGES themselves — id, document, a text preview — so
# the answer's graph can draw the passage a fact or a claim came from. Bounded
# by $doc_ids (the citations) and a row ceiling; `$chars` is a preview width.
CITED_CHUNKS = _q("""
MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)
WHERE d.doc_id IN $doc_ids AND <<chunk_visible>>
RETURN c.chunk_id AS chunk_id, d.doc_id AS doc_id, d.title AS doc_title, d.url AS url,
       left(c.text, $chars) AS text,
       c.allow_access_control AS _acl_log
ORDER BY doc_id, chunk_id
LIMIT $limit
""")

# Visible chunk excerpts for the entities in play — the single-source text that
# grounds synthesis (the safe alternative to blended descriptions; ACL Leak 1).
#
# PER-ANCHOR budget, for the same reason as SOURCE_DOCUMENTS — but the failure here was
# STARVATION rather than eviction, and worse. Matching every name at once under one
# LIMIT with no ORDER BY meant whichever anchors the planner happened to traverse first
# consumed the entire budget. Measured on dev with 5 anchors and LIMIT 15: the first two
# took all 15 rows and THREE anchors returned zero excerpts. These excerpts are the
# verbatim text that grounds the answer, so an entity the question is squarely about
# could contribute nothing quotable. Per anchor, the same case yields 5/5/5/17/9.
SUPPORTING_CHUNKS = _q("""
UNWIND $names AS anchor
MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)-[:MENTIONS]->(e:Entity {name: anchor})
WHERE <<chunk_visible>> AND <<current_person:e>>
// The per-anchor slice below keeps a FIXED number of chunks, so without an order
// it kept whichever the planner happened to emit and two identical requests could
// support the same answer with different passages. chunk_id is the stable
// per-chunk identity, so the slice is reproducible.
WITH anchor, c ORDER BY c.chunk_id
WITH anchor, collect(DISTINCT c)[0..$per_anchor] AS cs
UNWIND cs AS c
WITH DISTINCT c
MATCH (d:Document)-[:HAS_CHUNK]->(c)
// `via_names` = which of the requested entities this chunk MENTIONS — the real
// chunk→entity association, for the trace's MENTIONS edges. Re-derived AFTER the
// per-anchor slice, over all requested names, so a chunk selected for one anchor still
// reports every other anchor it mentions (slicing per anchor must not narrow it).
OPTIONAL MATCH (c)-[:MENTIONS]->(e2:Entity)
  WHERE e2.name IN $names
// People-profile docs (object_type='people') carry the person's employment
// status on the Person node, not in the chunk text — join it in by the stable
// id (doc_id == person_user_id) so synthesis can flag a quote about a former
// employee. No name-matching: the id link is exact and offboarding-proof.
OPTIONAL MATCH (subject:Person {person_user_id: d.doc_id})
  WHERE d.object_type = 'people'
WITH c, d, collect(DISTINCT e2.name) AS via_names,
     collect(DISTINCT subject.person_employment_status) AS ss,
     collect(DISTINCT subject.person_departed_on) AS sd
WHERE <<current_subject>>
RETURN c.chunk_id AS chunk_id, d.doc_id AS doc_id, d.title AS doc_title,
       left(c.text, $chars) AS text,
       CASE WHEN 'Inactive' IN ss THEN 'Inactive'
            WHEN 'Active' IN ss THEN 'Active' ELSE null END AS subject_status,
       via_names,
       c.allow_access_control AS _acl_log
// The overall LIMIT is a second cut, and needs its own order for the same reason.
ORDER BY chunk_id
LIMIT $limit
""")

# Find documents by title substring (case-insensitive), filtered on the
# document's OWN ACL — the discovery step before get_document_text (B6). The
# agent otherwise has no way to obtain a doc_id for a document the user names.
# SEVERAL TITLES IN ONE READ, each with its OWN row budget.
#
# Two problems, one shape, and the fix is the per-anchor idiom SUPPORTING_CHUNKS
# and SOURCE_DOCUMENTS already use.
#
# ONE TITLE AT A TIME made covering a series cost a call per member. The agent
# prompt already says to reach for the tool that takes a LIST — "repeating a
# single-subject tool once per member ... burns the round budget" — and this read
# gave it nothing to reach for. For example, a page listing twelve releases
# led the loop to look up only two of them by name.
#
# ONE GLOBAL LIMIT would then let one crowded title eat the whole budget. It
# already does within a single title: a short title can match several pages,
# and ordering by date pushes the wanted one past a small limit. Slicing PER TITLE means a
# crowded title can only spend its own rows, never another title's.
#
# Ordering within a title stays `published_at DESC` because it is the only signal
# available: there is no fulltext index on Document.title, so the engine has no match
# score to offer, and inventing one here would be exactly the hand-rolled relevance rule
# 8 forbids. Adding that index at ingestion is the real fix for match-quality ordering.
DOCUMENT_LOOKUP = """
UNWIND $titles AS wanted
MATCH (d:Document)
WHERE toLower(d.title) CONTAINS toLower(wanted)
  AND any(p IN d.allow_access_control WHERE p IN $principals)
WITH wanted, d
  ORDER BY d.published_at DESC
WITH wanted, collect(DISTINCT d)[0..$per_title] AS ds
UNWIND ds AS d
WITH DISTINCT d
RETURN d.doc_id AS doc_id, d.title AS title, d.url AS url, d.published_at AS date,
       d.source_type AS source_type,
       // The document's stored opening, for the citation preview — every
       // Document-returning read carries it (see SOURCE_DOCUMENTS), so which
       // tool found a document no longer decides whether its citation has one.
       d.doc_preview_text AS snippet,
       d.allow_access_control AS _acl_log
ORDER BY d.published_at DESC
LIMIT $limit
"""

# One document's body text, chunk by chunk — for tables/rosters/lists that
# claim extraction misses (the POC's fetch_document_text / research enrichment).
# Guarded per chunk: a caller reads exactly the chunks their ACL allows, never
# "the document" as a unit. Ordering by chunk_id is best-effort (ingestion does
# not persist an ordinal today). `$chars` bounds EACH chunk (`enrich_chars_per_doc`
# is per-chunk, as its own docstring says); `$limit` bounds how many chunks a read
# may return, and the repository reports whether it bound rather than applying a
# second, silent character budget across the document — which made a "read this
# document" call return one arbitrary chunk of a roster.
DOCUMENT_TEXT = _q("""
MATCH (d:Document {doc_id: $doc_id})-[:HAS_CHUNK]->(c:Chunk)
WHERE <<chunk_visible>>
<<mentions>>
WITH c, d, collect(DISTINCT me.name) AS via_names
RETURN c.chunk_id AS chunk_id, d.doc_id AS doc_id, d.title AS doc_title,
       left(c.text, $chars) AS text, via_names,
       c.allow_access_control AS _acl_log
ORDER BY c.chunk_id
LIMIT $limit
""")

# ── Graph-as-router traversal to chunks (Neo4j only — no chunk vectors) ──────

# Beam-bounded multi-hop traversal that REACHES CHUNKS (POC A6 — "graph as a
# router to hidden chunks"). Per seed, neighbours come from TWO union legs (POC
# parity): EXTRACTED_REL up to N hops (each edge on the path must have a visible
# supporting chunk), and chunk CO-MENTION (entities that merely co-occur in a
# visible chunk — recall where relation extraction is sparse). Hub explosion is
# damped by the hop cap (the ``__HOPS__`` token — a settings-controlled int,
# substituted in the repository because a var-length bound cannot be a Cypher
# parameter) and a PER-SEED, PER-LEG limit inside the subquery. The final chunks
# are <<chunk_visible>>-gated; `via_names` records which entities reached each
# chunk (multi-hop bonus). NB: the path variable is named ``path`` — ``p`` would
# collide with the ACL guard's list variable.
BEAM_TRAVERSAL = _q("""
UNWIND $seeds AS seed_name
MATCH (s:Entity {name: seed_name})
  // The seed and (below) every node walked must be current: a departed person
  // is neither an anchor nor a bridge for current evidence, and a slot of
  // $leg_limit they spend is one an active colleague's material lost.
  WHERE <<current_person:s>>
CALL (s) {
  MATCH path = (s)-[:EXTRACTED_REL*1..__HOPS__]-(nbr:Entity)
      // Every hop is an EXTRACTED_REL, and every EXTRACTED_REL carries its own
      // stamped ACL (the union of its supporting chunks' — see EXPAND_NEIGHBORS),
      // so the path is gated per EDGE with a property check instead of one chunk
      // EXISTS subquery per hop. Fail-closed: an unstamped edge never matches.
      WHERE all(r IN relationships(path)
            WHERE any(p IN r.allow_access_control WHERE p IN $principals))
      AND all(n IN nodes(path) WHERE <<current_person:n>>)
  RETURN DISTINCT nbr
  LIMIT $leg_limit
  UNION
  MATCH (s)<-[:MENTIONS]-(c:Chunk)-[:MENTIONS]->(nbr:Entity)
  WHERE <<chunk_visible>> AND <<current_person:nbr>>
  RETURN DISTINCT nbr
  LIMIT $leg_limit
  UNION
  RETURN s AS nbr
}
WITH collect(DISTINCT nbr) AS ents
UNWIND ents AS e
MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)-[:MENTIONS]->(e)
WHERE <<chunk_visible>>
WITH c, d, collect(DISTINCT e.name) AS via_names
RETURN c.chunk_id AS chunk_id, d.doc_id AS doc_id, d.title AS doc_title,
       left(c.text, $chars) AS text, d.published_at AS date, d.source_type AS source_type,
       via_names, size(via_names) AS via_count,
       c.allow_access_control AS _acl_log
ORDER BY via_count DESC
LIMIT $limit
""")

# The SAME traversal BEAM_TRAVERSAL walks, returning the PATH instead of the
# chunks it ends at. BEAM_TRAVERSAL binds `path` only to ACL-check it and then
# discards it ("the chunk is the product — the graph is only the router"), so a
# question about the connection itself — how A relates to B, what sits between
# them — had no read that answers it: the structure was traversed and thrown away.
#
# One row per HOP, not per path: a hop is exactly a RelationFact, so the result
# joins the same evidence pool and digest FACTS section every other relationship
# read feeds, and a path is recoverable from its hops. `depth` is the hop's
# position, so the caller can order a path and see how far out it reached.
#
# Same guards as EXPAND_NEIGHBORS, which this deliberately mirrors: plumbing
# neighbours skipped, the optional type filter on the same
# `coalesce(relationship_type, type(r))` expression, per-EDGE ACL on
# EXTRACTED_REL and per-NODE ACL on everything the path passes through.
# Fail-closed at every hop — a path is visible only if all of it is.
ENTITY_PATHS = _q("""
UNWIND $seeds AS seed_name
MATCH (s:Entity {name: seed_name})
CALL (s) {
  MATCH path = (s)-[*1..__HOPS__]-(nbr:Entity)
  WHERE ($target IS NULL OR nbr.name = $target)
    AND all(n IN nodes(path)
            WHERE NOT (n:Chunk OR n:Claim OR n:Community)
            AND any(p IN n.allow_access_control WHERE p IN $principals)
            AND <<current_person:n>>)
    AND all(r IN relationships(path)
            WHERE ($types IS NULL OR coalesce(r.relationship_type, type(r)) IN $types)
            AND (type(r) <> 'EXTRACTED_REL'
                 OR any(p IN r.allow_access_control WHERE p IN $principals)))
  RETURN path
  ORDER BY length(path), nbr.name
  LIMIT $leg_limit
}
WITH path, relationships(path) AS rs
UNWIND range(0, size(rs) - 1) AS i
WITH rs[i] AS r, i + 1 AS depth
// One row per DISTINCT hop: the same edge recurs on every path through it, and
// an UNDIRECTED walk also reaches it in BOTH orientations (A-B-C and A-C-B both
// use B-C), so a group keyed on the endpoint pair kept each orientation as its
// own row — the same fact twice in the pool's (source, relationship, target)
// key and twice in the digest. Grouping on the relationship itself is
// orientation-blind; the emitted row is then anchored on the edge's own stored
// direction below, so two identical requests emit one identical row per edge.
// The shallowest occurrence wins, so an edge adjacent to a seed is not reported
// as distant because a longer path also used it.
WITH r, min(depth) AS depth
WITH startNode(r) AS a, r, endNode(r) AS b, depth
// BOTH ends coalesced the same way: a mid-path node is not always an :Entity
// with a `name` (a Document, a Person stub), whichever end of its edges it sits on.
RETURN coalesce(a.name, a.title, a.person_full_name) AS source,
       coalesce(b.name, b.title, b.person_full_name) AS target,
       coalesce(b.type, head([l IN labels(b) WHERE l <> 'Entity'])) AS target_type,
       b.person_employment_status AS target_status,
       coalesce(r.relationship_type, type(r)) AS relationship,
       r.raw_types AS specific_relationships,
       r.description AS rel_description, r.descriptions_json AS descriptions_json,
       // `a` IS startNode(r) by construction now, so the direction is a constant.
       'outgoing' AS direction,
       coalesce(r.support, 1) AS support, r.support IS NULL AS structural, depth,
       r.chunk_ids AS chunk_ids,
       round($support_weight * log10(1.0 + toFloat(coalesce(r.support, 1))), 4) AS score,
       b.allow_access_control AS _acl_log
ORDER BY depth, score DESC, target
LIMIT $limit
""")

# Temporal changes anchored on CLAIMS (with an indexed date range) → their
# supporting CHUNKS (POC B7). RECENT_CHANGES above returns claim tuples; this
# returns the verbatim chunk text that evidences a dated change, which is what
# synthesis quotes from. Two UNION legs so each date predicate uses its own
# single-property index instead of an OR-scan.
RECENT_CHANGE_CHUNKS = _q("""
CALL () {
  MATCH (cl:Claim) WHERE cl.valid_from >= $since
  RETURN cl
UNION
  // Same dense-empty-date guard as RECENT_CHANGES: "" is "no event date".
  MATCH (cl:Claim) WHERE (cl.valid_from IS NULL OR cl.valid_from = '')
    AND cl.documented_on >= $since
  RETURN cl
}
WITH DISTINCT cl
WHERE ($topic IS NULL OR toLower(coalesce(cl.subject, '')) CONTAINS toLower($topic))
  AND EXISTS { MATCH (cl)-[:SUPPORTED_BY]->(c:Chunk) WHERE <<chunk_visible>> }
WITH cl, coalesce(nullIf(cl.valid_from, ''), cl.documented_on) AS changed_on
ORDER BY changed_on DESC
LIMIT 200
MATCH (cl)-[:SUPPORTED_BY]->(c:Chunk)
WHERE <<chunk_visible>>
MATCH (d:Document)-[:HAS_CHUNK]->(c)
WITH DISTINCT c, d, changed_on
// Limit BEFORE MENTIONS enrichment so it expands only for the kept chunks, not every
// chunk supporting the up-to-200 recent claims.
ORDER BY changed_on DESC
LIMIT $limit
<<mentions>>
WITH c, d, changed_on, collect(DISTINCT me.name) AS via_names
RETURN c.chunk_id AS chunk_id, d.doc_id AS doc_id, d.title AS doc_title,
       left(c.text, $chars) AS text, d.published_at AS date, d.source_type AS source_type,
       changed_on, via_names
ORDER BY changed_on DESC
""")

# ── Vocabulary ──────────────────────────────────────────────────────────────

# DELIBERATE EXCEPTION to the chunk-anchor rule (the only one): this returns
# aggregate schema metadata — the closed relationship-type verb list with rough
# counts — never node/edge content. It is tenant-scoped only, because an
# ACL-anchored version would need per-principal-set caching and a per-edge
# existence check (full scan). Revisit if type names ever become sensitive.
#
# The vocabulary must name what EXPAND_NEIGHBORS can FILTER on, which is
# `coalesce(r.relationship_type, type(r))` over every edge to a non-plumbing
# neighbour — not EXTRACTED_REL alone. Listing only the extracted verbs left the
# structured edges (REPORTS_TO, MANAGES, MEMBER_OF, AUTHORED, …) absent from the
# tool the agent is told to call before choosing `relationship_types`, so an org
# question could only reach them by guessing a name the list never showed.
RELATIONSHIP_TYPES = """
MATCH (a:Entity)-[r]->(nbr)
WHERE NOT (nbr:Chunk OR nbr:Claim OR nbr:Community)
RETURN coalesce(r.relationship_type, type(r)) AS type, count(*) AS edges
ORDER BY edges DESC, type
LIMIT $limit
"""

# ONE folding rule for a GROUP name, shared by every read that matches one, so
# the attribute arm and the group-node arm cannot disagree about what "the same
# group" means. Case- and separator-folded: the graphs spell one group
# "Data-Science" and "Data Science" (both observed live), and a caller types
# either — a punctuation difference is not a different team. Still an equality,
# not a substring: folding widens the spellings that match one name, never the
# set of names.
def _folded(expr: str) -> str:
    return ("toLower(trim(replace(replace(replace("
            f"{expr}, '-', ' '), '_', ' '), '.', ' ')))")


# ── People & authorship ──────────────────────────────────────────────────────
# The graph carries a people/org layer (Person + AUTHORED / EDITED /
# CONTRIBUTED_TO / AUTHOR_OF) the topic layer ignores — it answers "who is X" and
# "who wrote / owns X". One rel-type alternation handles BOTH authorship grammars
# (AUTHORED/EDITED, or AUTHOR_OF); the person display fields are coalesced across
# the Person shapes different connectors write. The ACL guard is wrapped at
# run time (acl_guarded), so these stay fail-closed + ACL-ready
# even though enforcement is currently flag-bypassed. The `any(p IN … $principals)`
# list variable must stay `p` for acl_guarded to recognise it; the Person node is
# therefore named `person`.

# Resolve PEOPLE by name/role via the same fulltext index entity resolution uses
# (a graph whose people are bare identity nodes simply gets no hit,
# which is fine). Returns structured profile fields, not a blended description.
PEOPLE_RESOLVE = """
CALL db.index.fulltext.queryNodes($entity_index, $lucene) YIELD node AS person, score
WHERE 'Person' IN labels(person)
  AND any(p IN person.allow_access_control WHERE p IN $principals)
  AND <<current_person:person>>
RETURN person.name AS name,
       coalesce(person.person_full_name, person.user_name, person.name) AS full_name,
       person.person_job_title AS job_title,
       coalesce(person.person_department, person.person_team) AS department,
       coalesce(person.person_email, person.user_email) AS email,
       person.person_employment_status AS employment_status,
       coalesce(person.degree, 0) AS degree, score AS score
ORDER BY score DESC
LIMIT $limit
"""

# The STRUCTURED profile fields of named people, in ONE batched read.
#
# Ingestion writes a person's attributes as `person_*` node properties; the
# blended `description` is only populated when some document happened to
# describe them, so relying on it leaves most people blank. `PEOPLE_RESOLVE`
# already returns these fields, but it is a FULLTEXT SEARCH — usable for "find
# me people matching X", not for "describe these N people I already have". This
# query is the by-name counterpart: the answer enumerates people the traversal
# reached (relationship targets carry a name, never a job title), and this is
# how those names get their role/department/location without one search per
# person.
#
# Schema-driven, not query-driven: the caller passes whatever names the evidence
# holds; nothing here reads the question. Same ACL guard shape as every other
# read (the list variable must stay `p` for acl_guarded) — and, like every other
# read here, NO tenant predicate: this query arrived with one from a branch that
# still bound $tenant, and the repository no longer does, so it failed with
# ParameterMissing on every call (profile_enrichment_failed) until the predicate
# was dropped. Per-run databases are single-tenant; the filter was redundant.
# The person-profile row, shared by the name-keyed read and the identity-keyed
# read below — one contract, so a field added for profiles exists in both.
#
# The person's OWN source document — the people object ingestion writes as a
# Document whose doc_id IS the person_user_id (title = the person's name, url =
# their profile page) — is what makes a profile fact CITABLE: the fields
# returned here otherwise reach synthesis as bare entity lines with no document
# behind them, so people/org answers shipped without citations. Checked on the
# document's own ACL like every other Document read; graphs whose people carry
# no such document simply return NULLs here.
#
# `person_user_id` / `person_email` are the person's CROSS-GRAPH identity keys: one
# graph (an HR directory) may be the system of record for person attributes while
# another holds the same human under a different display name ("(Deactivated)" suffixes,
# dotted usernames), so an id/email join, never a name join, is what lets the repository
# merge the attributes across graphs. `user_email` coalesced for the old-schema graphs,
# same as PEOPLE_RESOLVE.
_PERSON_PROFILE_ROW = """
OPTIONAL MATCH (d:Document {doc_id: person.person_user_id})
  WHERE any(p IN d.allow_access_control WHERE p IN $principals)
RETURN person.name AS name,
       person.person_user_id AS person_user_id,
       coalesce(person.person_email, person.user_email) AS person_email,
       d.doc_id AS doc_id, d.title AS doc_title, d.url AS doc_url,
       d.source_type AS doc_source_type,
       // The people object's stored opening (name, title, reporting line) — the
       // citation preview for a person cited through their profile. Without it a
       // roster's citations carried a snippet only for members some other read
       // happened to quote (measured: 9 of 16 cited people previewless).
       d.doc_preview_text AS doc_snippet,
       person.person_job_title AS job_title,
       coalesce(person.person_department, person.person_team) AS department,
       person.person_location AS location,
       person.person_employment_status AS employment_status,
       // Org position, straight off the node — no traversal. `person_manager`
       // mirrors the REPORTS_TO edges (same 741 on the dev graph) but costs
       // nothing extra here, and `direct_reports`/`org_size` are counts the
       // graph cannot produce cheaply any other way.
       person.person_manager AS manager,
       person.direct_reports AS direct_reports,
       person.org_size AS org_size,
       // Set ONLY for people who have left, so a value means departed and NULL
       // means no departure recorded — never read it as proof of "active".
       person.person_departed_on AS departed_on
"""

PEOPLE_PROFILES = """
MATCH (person:Person)
WHERE person.name IN $names
  AND any(p IN person.allow_access_control WHERE p IN $principals)
  AND <<current_person:person>>
""" + _PERSON_PROFILE_ROW
# No LIMIT, deliberately: the result is already bounded by $names (the caller caps
# that list at agent_profile_enrich_top), and one name can match SEVERAL person
# nodes — three distinct "Sam Carter"s on the dev graph. A row limit sized to the
# name count therefore truncated OTHER people's rows away, silently returning no
# profile for names that had matched. Rows here are a handful of scalars, so the
# name list is the honest bound.

# The same profile row keyed by IDENTITY instead of name — the attribute-donor
# read behind the cross-graph merge. When a name-keyed row comes back as a bare
# identity stub (no job title, no department, no org fields), the repository
# fetches the SAME HUMAN's attribute-bearing node — whose display name may
# differ — by person_user_id/email.
# Rows from this read are donors only: the repository never emits them under
# their own name (their name may not be in the evidence at all). Bounded by
# $ids/$emails exactly as PEOPLE_PROFILES is bounded by $names.
PEOPLE_PROFILES_BY_IDENTITY = """
MATCH (person:Person)
WHERE (person.person_user_id IN $ids
       OR coalesce(person.person_email, person.user_email) IN $emails)
  AND any(p IN person.allow_access_control WHERE p IN $principals)
""" + _PERSON_PROFILE_ROW

# ENUMERATION, not relevance retrieval — the one read that SELECTS people on the
# structural attributes every other people read merely projects.
#
# WHY IT EXISTS. `person_department`, `person_team` and `person_manager` are
# returned by _PERSON_PROFILE_ROW on every profile read and were selected on by
# nothing, so "who are all the members of X" had no read that could answer it:
# PEOPLE_RESOLVE is a fuzzy fulltext match over names (a team name matches no
# person's name — measured, four team-name searches returned zero rows while
# every profiled person in the same run carried a department), PEOPLE_PROFILES
# needs the names the caller is trying to discover, and the contributor reads
# key on document titles and topics. Every one of them is top-K over text
# resemblance, so a valid member was omitted whenever another candidate scored
# higher — which is what top-K does, applied to a question that asked for a set.
#
# NO RANKING, DELIBERATELY. A set is not a ranking: rows come back in name order
# and the caller reports how many matched, so an incomplete answer is a visible
# truncation rather than a silent one. The equality is exact (case-folded), never
# a CONTAINS: a substring match on a department makes "Data" select everyone in
# "Data Science" AND "Data Platform", which is a different set than the one asked
# for. Nothing here reads the question — the caller passes a group name and the
# schema decides what matches.
PEOPLE_BY_GROUP = f"""
MATCH (person:Person)
WHERE {_folded("$group")} IN [
        {_folded("coalesce(person.person_department, '')")},
        {_folded("coalesce(person.person_team, '')")},
        {_folded("coalesce(person.person_manager, '')")}
      ]
  AND any(p IN person.allow_access_control WHERE p IN $principals)
  AND <<current_person:person>>
""" + _PERSON_PROFILE_ROW + """
ORDER BY name
LIMIT $limit
"""

# ── Structured groups: the :Team / :Department nodes reached by [:MEMBER_OF] ──
#
# THE SECOND REPRESENTATION OF "who is in X". The read above selects on the HR
# taxonomy ingestion writes as person PROPERTIES; both graphs ALSO carry group
# NODES — :Team, :Department — with structured [:MEMBER_OF] edges into them, and
# the two are not the same set (a scrum team is not an HR department, and the
# graphs record both). Neither representation is reachable from the other, and
# nothing in the toolset could anchor on a group node at all: entity resolution
# matches ':Entity', which a :Team node is not, so a structured team was
# invisible to every read the service has — no resolution, no expansion, no
# enumeration.
#
# A GROUP IS WHATEVER A PERSON IS A MEMBER_OF. No label list and no name list:
# the edge itself is what says the node is a group, so a label ingestion adds
# later is covered on arrival, and only the plumbing labels are excluded — the
# same exclusion EXPAND_NEIGHBORS applies, for the same reason.
#
# VISIBILITY. Structured edges carry NO properties at all (no ACL, no chunk ids)
# and the group node carries none either, so an ACL predicate on either would
# evaluate null and drop every row under enforcement — the trap
# EXPAND_NEIGHBORS's `type(r) <> 'EXTRACTED_REL'` guard exists for. The
# ACL-bearing node in this shape is the PERSON, and the person is also the thing
# being disclosed, so a group is visible THROUGH the members the caller may read:
# a group whose every member is invisible does not exist for that caller.
# Fail-closed by construction, and the same employment gate as every other people
# read.
_MEMBER_OF_GROUP = """
MATCH (person:Person)-[:MEMBER_OF]->(g)
WHERE NOT (g:Chunk OR g:Claim OR g:Community)
  AND any(p IN person.allow_access_control WHERE p IN $principals)
  AND <<current_person:person>>
"""

# The group node's own display label, coalesced the way EXPAND_NEIGHBORS
# coalesces a neighbour's: these nodes carry no `type` property, so the label is
# what says what kind of group it is.
_GROUP_NAME = "coalesce(g.name, g.title, '')"

# DISCOVERY — "which groups is this phrase about", the step that has to exist
# before an exact-match enumeration can be asked for anything. Deliberately
# SUBSTRING, in both directions, and deliberately only here: a caller types "the
# data science team" for a node called "Data-Science" and vice versa, and a
# discovery read that only matched exactly could never be the thing that finds
# the exact spelling. `members` is the caller-visible member count and the
# ordering signal — the graph's own, never a score of ours.
GROUPS_RESOLVE = _MEMBER_OF_GROUP + f"""
  AND {_folded(_GROUP_NAME)} <> ''
  AND ({_folded(_GROUP_NAME)} CONTAINS {_folded("$group")}
       OR {_folded("$group")} CONTAINS {_folded(_GROUP_NAME)})
WITH g, count(DISTINCT person) AS members
RETURN {_GROUP_NAME} AS name,
       head([l IN labels(g) WHERE l <> 'Entity']) AS type,
       members AS members
ORDER BY members DESC, name
LIMIT $limit
"""

# ENUMERATION — every member of the named group. EXACT (folded) equality, never
# CONTAINS, for the reason the attribute arm states: a substring match on "Data"
# selects Data Science AND Data Platform, which is a different set than the one
# asked for. Unranked, ordered by name, and the caller reports whether the
# ceiling bound — a set is not a ranking, and a member must never be droppable
# because another row scored higher.
#
# `WITH DISTINCT person` because one human can hold a MEMBER_OF edge into two
# same-named group nodes (a :Team and a :Department both called "Data Science"),
# which would otherwise spend two rows of the ceiling on one person.
#
# The SAME profile row as every other people read, so an enumerated member
# renders the columns a roster needs and goes through the one identity merge.
GROUP_MEMBERS = _MEMBER_OF_GROUP + f"""
  AND {_folded(_GROUP_NAME)} = {_folded("$group")}
WITH DISTINCT person
""" + _PERSON_PROFILE_ROW + """
ORDER BY name
LIMIT $limit
"""

# NAME -> CROSS-GRAPH IDENTITY, for the pass that gives one human one name.
# Two scalars per row: the display name exactly as every other
# people-bearing read emits it (same coalesce, so the keys match the names already
# in the evidence pool), and the email that identifies the human behind it.
#
# EMAIL, not the user id: each connector mints its own id space, so the SAME
# human's ids never match across graphs. Email does.
#
# Bounded by $names — the person-ish names the pool already holds — so this is one
# batched read per graph whose size follows the request, never the corpus. A name
# that belongs to no person node simply returns no row.
PERSON_IDENTITIES = """
MATCH (person:Person)
WHERE coalesce(person.person_full_name, person.user_name, person.name) IN $names
  AND any(p IN person.allow_access_control WHERE p IN $principals)
RETURN coalesce(person.person_full_name, person.user_name, person.name) AS name,
       toLower(coalesce(person.person_email, person.user_email, '')) AS person_email,
       person.person_employment_status AS employment_status,
       person.person_departed_on AS departed_on,
       coalesce(person.degree, 0) AS degree
"""

# THE SAME TWO SCALARS, KEYED BY EMAIL — the second half of the pass, and the half
# that makes a single handle fixable at all.
#
# The read above can only see the labels the evidence already carries: asked for
# "jane.doe" it returns that node and nothing else, because the same human's
# other node is called "Jane Doe" and that string was never in $names. A handle
# therefore looks unambiguous — one row, one label, no rename — and the answer
# would keep it.
#
# So the emails resolved above are fed back in here to collect EVERY node for those
# humans, in every graph, whatever each one is called. Same two-step, same reason,
# as PEOPLE_PROFILES -> PEOPLE_PROFILES_BY_IDENTITY: a name-keyed read cannot find
# the rows that disagree with the name.
PERSON_IDENTITIES_BY_EMAIL = """
MATCH (person:Person)
WHERE toLower(coalesce(person.person_email, person.user_email, '')) IN $emails
  AND any(p IN person.allow_access_control WHERE p IN $principals)
RETURN coalesce(person.person_full_name, person.user_name, person.name) AS name,
       toLower(coalesce(person.person_email, person.user_email, '')) AS person_email,
       person.person_employment_status AS employment_status,
       person.person_departed_on AS departed_on,
       coalesce(person.degree, 0) AS degree
"""

# WHO authored/edited/contributed to documents whose title matches $title. ACL +
# tenant anchor on the Document (authorship is visible iff the document is).
#
# ORDERED BY HOW MUCH OF THE MATCHING MATERIAL EACH PERSON ACTUALLY WROTE, because
# the LIMIT binds hard on a broad topic and something has to decide what survives
# it. It used to be `ORDER BY person` — an alphabetical slice — so a 60-row cap on
# a topic with hundreds of contributors returned the front of the alphabet and
# stopped: measured on "who is building Enterprise Search", the rows reached
# Abhimanyu, Aditya and Sam and never got as far as the four engineers with 19,
# 14, 5 and 2 matching documents each. The sibling query below already orders by
# date for exactly this reason ("the LIMIT then keeps the most recent work rather
# than an alphabetical slice"); this one never got the same treatment.
#
# `contributions` is a COUNT OF MATCHING EDGES — an engine signal of the same kind
# as `support` and `degree` used everywhere else, computed from the graph and never
# from the question's text.
#
# ONE ROW PER PERSON, so the LIMIT counts the thing the question asks for. This read
# answers WHO, and a person-per-document row shape made the limit count PAIRS: the
# most prolific contributor could spend it several times over and the answer would
# name one person where a team was asked for. Collapsing to the person makes the cap
# mean "up to $limit people", needs no per-person budget to stay fair, and loses
# nothing recoverable — `contributions` says how much they wrote and `document`
# names their most recent matching piece, while the full list of what any one of
# them wrote is a person-anchored read away.
DOCUMENT_CONTRIBUTORS = """
MATCH (person:Person)-[r:AUTHORED|EDITED|CONTRIBUTED_TO|AUTHOR_OF]->(d:Document)
  WHERE toLower(d.title) CONTAINS toLower($title)
  AND any(p IN d.allow_access_control WHERE p IN $principals)
  AND <<current_person:person>>
WITH person, r, d
  // Highest-degree node first, then newest document: `head(collect(...))` below
  // then takes the PRIMARY record's name and its most recent matching work.
  ORDER BY coalesce(person.degree, 0) DESC, d.published_at DESC
// ONE ROW PER HUMAN, not per node. Ingestion emits the same person as several
// Person nodes — a canonical one plus "(Deactivated)"/"(Unlicensed)" twins and
// run-together name artefacts — and they all carry the SAME person_user_id.
// Without grouping, one person can take several of the top rows of a people
// answer. Grouping on the identity field collapses them EXACTLY, with no name
// comparison of any kind — falling back to the name only for nodes with no id.
WITH coalesce(person.person_user_id,
              coalesce(person.person_full_name, person.user_name, person.name)) AS identity,
     person, r, d
WITH identity,
     count(DISTINCT d) AS contributions,
     head(collect(coalesce(person.person_full_name, person.user_name, person.name))) AS person,
     head(collect(person.person_job_title)) AS job_title,
     head(collect(coalesce(person.person_department, person.person_team))) AS department,
     head(collect(person.person_employment_status)) AS status,
     head(collect(type(r))) AS role,
     head(collect(d.title)) AS document
RETURN person, job_title, department, status, role, document, contributions
ORDER BY contributions DESC, person
LIMIT $limit
"""

# WHO WORKS ON A TOPIC — the people two hops from a concept, via what they wrote.
#
# The gap this closes: a concept node hardly links to people at all — its EXTRACTED_REL
# edges go overwhelmingly to other concepts, so one-hop expansion returns almost no
# people, while the people who actually work on it sit one hop further out, reachable as
# person —CONTRIBUTED_TO/AUTHORED→ (a page-level entity) —related→ (the concept).
# Nothing in the toolset walked that path, which is why "who is building it" could only
# ever be answered from whatever chunk text happened to name somebody.
#
# Two hops, both anchored: the topic end is the resolved concept, the person end is
# a real authorship edge. `mid` is whatever the corpus put between them — a page, a
# feature, a component — and is returned so the answer can say WHY the person is
# connected rather than asserting they work on the topic.
#
# Ranked by contribution count, and ONE ROW PER PERSON for the same reason as
# DOCUMENT_CONTRIBUTORS above: the question is who, so the LIMIT counts people. Both
# counts are edge counts — engine signals, not judgements about the question.
TOPIC_CONTRIBUTORS = """
MATCH (topic:Entity)
  WHERE toLower(topic.name) = toLower($topic)
  AND any(p IN topic.allow_access_control WHERE p IN $principals)
MATCH (topic)-[rel]-(mid:Entity)
  WHERE NOT (mid:Chunk OR mid:Claim OR mid:Community)
  AND (type(rel) <> 'EXTRACTED_REL'
       OR any(p IN rel.allow_access_control WHERE p IN $principals))
  AND any(p IN mid.allow_access_control WHERE p IN $principals)
MATCH (person:Person)-[r:AUTHORED|EDITED|CONTRIBUTED_TO|AUTHOR_OF]->(mid)
  WHERE any(p IN person.allow_access_control WHERE p IN $principals)
  AND <<current_person:person>>
WITH person, r, mid
  ORDER BY coalesce(person.degree, 0) DESC
// One row per HUMAN — same identity collapse as DOCUMENT_CONTRIBUTORS above.
WITH coalesce(person.person_user_id,
              coalesce(person.person_full_name, person.user_name, person.name)) AS identity,
     person, r, mid
WITH identity,
     count(DISTINCT mid) AS contributions,
     head(collect(coalesce(person.person_full_name, person.user_name, person.name))) AS person,
     head(collect(person.person_job_title)) AS job_title,
     head(collect(coalesce(person.person_department, person.person_team))) AS department,
     head(collect(person.person_employment_status)) AS status,
     head(collect(type(r))) AS role,
     head(collect(mid.name)) AS via
RETURN person, job_title, department, status, role, via, contributions
ORDER BY contributions DESC, person
LIMIT $limit
"""

# The SAME edge, walked from the PERSON: which documents did this person write.
# The mirror above is anchored on `d.title`, so asking it about a
# person matches whatever document happens to carry that person's NAME in its
# title — their profile page, the "Meet X" welcome post — and reports who wrote
# THOSE — "everything X has written" would come back as the concepts X is
# mentioned near, and the AUTHORED edges would stay unreachable.
#
# Newest first: an authorship question is nearly always "what has X been writing",
# and the LIMIT then keeps the most recent work rather than an alphabetical slice.
#
# $since/$until bound d.published_at — empty string means unbounded, the
# same sentinel RetrievalPlan.time_scope_from/to already use. The `<> ''` guard
# matters: an empty-string date sorts BELOW every
# real one, so without it an undated row would satisfy any `>=` lower bound.
# ACL is the Document's, exactly as above — authorship is visible iff the document is.
DOCUMENTS_BY_AUTHOR = """
MATCH (person:Person)-[r:AUTHORED|EDITED|CONTRIBUTED_TO|AUTHOR_OF]->(d:Document)
  WHERE toLower(coalesce(person.person_full_name, person.user_name, person.name))
        CONTAINS toLower($who)
  AND any(p IN d.allow_access_control WHERE p IN $principals)
  AND <<current_person:person>>
  AND ($since = '' OR (d.published_at >= $since AND d.published_at <> ''))
  AND ($until = '' OR (d.published_at <= $until AND d.published_at <> ''))
RETURN coalesce(person.person_full_name, person.user_name, person.name) AS person,
       person.person_job_title AS job_title,
       coalesce(person.person_department, person.person_team) AS department,
       person.person_employment_status AS status,
       type(r) AS role, d.title AS document, d.doc_id AS doc_id,
       d.url AS url, d.published_at AS date, d.source_type AS source_type,
       // Stored opening for the citation preview (same contract as every other
       // Document-returning read — see DOCUMENT_LOOKUP).
       d.doc_preview_text AS snippet
ORDER BY d.published_at DESC
LIMIT $limit
"""
