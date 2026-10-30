"""GraphReadRepository — the only gateway to Neo4j.

Every public method takes an `AccessContext` as its first argument and the base
runner injects ``$principals`` into every execution, so an unguarded query is
structurally impossible (HLD §6). Tenant isolation is DB/collection-scoped
(each Neo4j database / Qdrant collection is a single tenant+run), not query-level.
"""
from __future__ import annotations

import asyncio
import json
import math
import re
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

import structlog
from neo4j import Query
from neo4j.exceptions import Neo4jError, ServiceUnavailable, SessionExpired

from ai_graph_retrieval.connectors.neo4j.client import Neo4jGraph
from ai_graph_retrieval.connectors.neo4j.http_client import Neo4jHttpClient
from ai_graph_retrieval.connectors.qdrant.store import QdrantVectorStore
from ai_graph_retrieval.core.access import AccessContext
from ai_graph_retrieval.core.exceptions import GraphUnavailable
from ai_graph_retrieval.core.models import (
    ChunkQuote,
    CitedSubgraph,
    ClaimFact,
    DocumentText,
    EntityHit,
    RelationFact,
    SourceDocument,
)
from ai_graph_retrieval.graph import cypher
from ai_graph_retrieval.observability import acl_audit
from ai_graph_retrieval.observability.tracing import (
    annotate_current_span,
    record_io,
    set_span_attributes,
    stage_span,
)
from ai_graph_retrieval.retrieval.person_status import (
    INACTIVE,
    MANAGER_SEGMENT,
    PersonIdentities,
    departed_person_names,
    identity_key,
    is_inactive,
)
from ai_graph_retrieval.retrieval.quality import freshness, graph_peaks, graph_scalers
from ai_graph_retrieval.settings import GraphSource, Settings

logger = structlog.get_logger(__name__)

# Bound for Input/Output on store spans. A read can return hundreds of rows; a
# trace is for reading, so the payload is truncated rather than omitted.
_TRACE_IO_CHARS = 4000

# MENTIONED_ENTITIES candidate ceiling per requested slot (or per chunk, whichever
# is larger): one chunk mentions ~10 entities, and the ranking needs every
# candidate's chunk list before it can pick the top `limit`.
_MENTION_CANDIDATES_PER_SLOT = 10

# Reverse map {query text -> constant name} for every UPPER_CASE Cypher string in
# ``cypher`` — lets ``_rows`` name the query it ran in a diagnostic log without
# threading a label through every call site. Built once at import.
_QUERY_NAMES: dict[str, str] = {
    v: k for k, v in vars(cypher).items() if isinstance(v, str) and k.isupper()
}


def _rows_by_graph(rows: list[dict]) -> dict[str, int]:
    """Row count per source graph, for the read's span.

    WHICH graphs answered was already recorded; HOW MANY rows each returned was
    not, and that is the number that carries the finding. A fan-out where one
    graph returns 48 rows and the other 2 reads as a healthy multi-graph request
    under a list of names, and that imbalance is exactly what a corpus problem on
    one source looks like from here.

    A row the fan-out did not tag with a graph is not counted: the tag is added
    per source as rows are merged, so an untagged row is a shape problem in the
    query, not evidence about a graph.
    """
    out: dict[str, int] = {}
    for r in rows:
        if graph := r.get("graph"):
            out[str(graph)] = out.get(str(graph), 0) + 1
    return out


def _loggable(params: dict) -> dict:
    """Compact a query's bound params for one-line logging: long lists become
    ``[a, b, c, "…+N"]`` and long strings are truncated, so a $names batch or a
    stray embedding never bloats the line."""
    out: dict = {}
    for k, v in params.items():
        if isinstance(v, (list, tuple)):
            head = [str(x)[:60] for x in list(v)[:3]]
            out[k] = [*head, f"…+{len(v) - 3}"] if len(v) > 3 else head
        elif isinstance(v, str) and len(v) > 80:
            out[k] = v[:80] + "…"
        else:
            out[k] = v
    return out

# Row keys tried, in order, to name a fetched node in the ACL audit. `target` (the
# neighbour whose ACL is recorded) is preferred over the anchor `source`.
_NODE_ID_KEYS = ("entity_id", "chunk_id", "doc_id", "name", "target", "subject", "source")

# Parameters ``_run_one`` binds on every execution, so the logs and spans show
# what a read was ASKED rather than the machinery every read carries.
_BOUND_EVERYWHERE = frozenset({"principals", "entity_index", "include_inactive_people"})


def _node_id(row: dict) -> str | None:
    for k in _NODE_ID_KEYS:
        v = row.get(k)
        if v:
            return str(v)
    return None

# Cap for the derived-description text assembled from visible entries.
_DESCRIPTION_CHARS = 600


def _visible_entries(raw_json: str | None, principals: Sequence[str] | None) -> list[dict]:
    """The per-source entries of a ``*_json`` property this caller may read.

    Ingestion never blends derived text: ``descriptions_json`` (entities and
    EXTRACTED_REL edges) is a JSON list of entries, each tagged with its OWN
    source document's ACL (``allow_access_control`` on entity entries, ``acl``
    on relationship entries). Filtering per entry is what stops private text
    leaking through a shared entity (ACL Leak 1). Fail-closed: a malformed
    payload or an entry without a matching ACL token contributes nothing.

    ``principals is None`` ⇒ ACL enforcement disabled (AGR_ACL_ENFORCEMENT_ENABLED
    =false): the per-entry filter is bypassed and every well-formed entry is
    returned, matching the null-principals bypass in Cypher and Qdrant."""
    if not raw_json:
        return []
    try:
        entries = json.loads(raw_json)
    except (TypeError, ValueError):
        return []
    if not isinstance(entries, list):
        return []
    if principals is None:  # ACL enforcement off — no per-entry filtering
        return [e for e in entries if isinstance(e, dict)]
    pset = set(principals)
    return [
        e for e in entries
        if isinstance(e, dict)
        and any(p in pset for p in (e.get("allow_access_control") or e.get("acl") or []))
    ]


def _visible_description(raw_json: str | None, principals: Sequence[str] | None) -> str | None:
    entries = _visible_entries(raw_json, principals)
    texts = [t for e in entries if (t := (e.get("text") or "").strip())]
    joined = " ".join(texts)[:_DESCRIPTION_CHARS]
    return joined or None


def _acl_conditional_description(
    plain: str | None, raw_json: str | None, principals: Sequence[str] | None
) -> str | None:
    """The description the CALLER may read, from the new schema's two fields.

    The plain ``description`` is ingestion's blend over ALL source documents and
    carries no ACL tags; ``descriptions_json`` holds the same text split per
    source entry, each tagged with its own ACL. So which one is safe depends on
    whether ACL is enforced:

      * enforcement OFF (``principals is None``) — the plain blend is the richer,
        cheaper read; fall back to the unfiltered entries where ingestion wrote no
        blend (EXTRACTED_REL edges: descriptions_json 100%, plain ≈0%).
      * enforcement ON — ONLY the per-entry-filtered json. The plain blend must
        never win here, or text derived from a document the caller cannot see is
        shown whenever the blend happens to exist (it exists on ~100% of entities
        in the per-run graphs, so "prefer plain" under ACL is a leak, not a
        fallback). No plain fallback either — fail closed to nothing.
    """
    if principals is None:
        return (plain or "").strip() or _visible_description(raw_json, None)
    return _visible_description(raw_json, principals)


# A person's attributes are separate node properties; the digest renders ONE
# line per entity. Joined with "·" rather than prose so the answer model reads
# them as discrete facts it may use individually (a table's Role and Location
# columns) instead of quoting a sentence. Missing fields are omitted, never
# rendered as a placeholder — a blank in the source must not become text that
# looks like an answer.
_PROFILE_FIELDS = ("job_title", "department", "location")
# The label each profile field is rendered under. Unlabelled, the line was three
# bare values ("Staff Engineer · Platform · Bengaluru") and which one was the
# department rather than the location was left to the model to infer from the
# words themselves — fine for a title, guesswork for two proper nouns. Labelling
# them makes the roster columns readable straight off the digest, which is what
# a per-person answer has to fill.
_PROFILE_LABELS = {
    "job_title": "title",
    "department": "department",
    "location": "location",
}
_FORMER_EMPLOYEE = "FORMER EMPLOYEE — no longer with the organization"


def _org_parts(row: dict) -> list[str]:
    """Org position from the person node: who they report to, and how many people
    sit under them. Every field is CONDITIONAL in the graph — an individual
    contributor has no reports and the CEO has no manager — so each is included
    only when it carries a value, and a zero count says nothing worth printing."""
    parts: list[str] = []
    if manager := str(row.get("manager") or "").strip():
        parts.append(f"{MANAGER_SEGMENT}{manager}")
    reports, org_size = row.get("direct_reports"), row.get("org_size")
    if isinstance(reports, int) and reports > 0:
        parts.append(f"{reports} direct report{'s' if reports != 1 else ''}")
        # Only meaningful alongside the direct count, and only when it adds
        # something: for a manager with no sub-managers the two are equal.
        if isinstance(org_size, int) and org_size > reports:
            parts.append(f"{org_size} in org")
    return parts


def _profile_line(row: dict) -> str:
    """The renderable profile for one person row, or "" when nothing is set."""
    parts = [
        f"{_PROFILE_LABELS[key]}: {value}" for key in _PROFILE_FIELDS
        if (value := str(row.get(key) or "").strip())
    ]
    parts.extend(_org_parts(row))
    status = str(row.get("employment_status") or "").strip()
    # `person_departed_on` is written ONLY for people who have left, so a value
    # is the departure itself — the strongest signal available, and the only one
    # on graphs where employment_status is absent. Its NULL means "no departure
    # recorded", which is NOT a claim that the person is active, so nothing is
    # emitted for the empty case.
    if departed := str(row.get("departed_on") or "").strip():
        return " · ".join([f"{_FORMER_EMPLOYEE} (left {departed})", *parts])
    # Only a NON-active status is worth stating: "Active" is the unremarkable
    # default, while a former employee changes whether the person belongs in the
    # answer at all (the synthesis prompt's INACTIVE rule reads this).
    #
    # It LEADS the line rather than trailing it, and shouts. Trailing it was
    # measured worse: once the profile pass gave a departed colleague a
    # complete-looking "Data Scientist · Data Science · no longer with the
    # organization", the fast answer model skimmed the tail and listed them as a
    # roster row in 1 of 3 runs. The first token of the line is the one that
    # decides whether the person belongs in the answer at all.
    if is_inactive(status):
        parts.insert(0, _FORMER_EMPLOYEE)
    return " · ".join(parts)


# Every attribute _profile_line can render — what makes a row an attribute
# BEARER rather than an identity stub. Identity keys and doc fields are
# deliberately not in here: they say who the row is, not what it carries.
_PROFILE_ATTRS = (*_PROFILE_FIELDS, "employment_status", "manager",
                  "direct_reports", "org_size", "departed_on")


def _attr_count(row: dict) -> int:
    """How many renderable profile attributes the row actually carries — the
    completeness order the cross-graph merge picks its source row by."""
    return sum(1 for k in _PROFILE_ATTRS if row.get(k) not in (None, "", 0))


# The cross-graph identity of a person row. It lives in retrieval/person_status.py
# so the profile merge here and the departed-people resolution there group humans
# by exactly the same key — two answers off one grouping rule, never two rules.
_identity_key = identity_key


def _best_of_group(groups: dict[str, list[dict]], row: dict) -> dict:
    """The row this person's profile should render from: the most
    attribute-complete row of their identity group (ties keep first-seen order,
    which is fan-out order — deterministic), or the row itself when it carries
    no identity key. The caller keeps the ROW's name either way; only the
    attributes travel."""
    key = _identity_key(row)
    if key is None:
        return row
    return max(groups[key], key=_attr_count)


def _canonical_person_names(
    rows: Sequence[dict], authority: dict[str, int]
) -> dict[str, str]:
    """{display name -> the ONE name to use for that human}, across graphs.

    Rows are (name, person_email, degree, graph) from PERSON_IDENTITIES, fanned out
    over every graph. Humans are grouped by EMAIL — an exact key, never a name
    comparison — and each group's label is chosen by (source authority, DEGREE).
    Only actual renames are returned.

    DEGREE is the tiebreak, and it is not decoration: most duplicates are not
    cross-graph at all. Ingestion emits the same human as several nodes WITHIN one
    graph — a primary record plus "(Deactivated)"/"(Unlicensed)" twins. Without
    a tiebreak the label comes down to row order and can pick the twin ('Jane Doe
    (Unlicensed)' over 'Jane Doe'). Degree is the graph's own signal for which node is
    the real record, and it is the same tiebreak DOCUMENT_CONTRIBUTORS and
    TOPIC_CONTRIBUTORS already use for this exact shape — no string is inspected,
    so a suffix, a handle and a full name are treated identically.

    TWO GUARDS make the rewrite safe to apply blind:

    * a name that resolves to MORE THAN ONE email is left alone. Two humans do
      share a display name, and renaming there would merge two people into one —
      the mistake this pass exists to prevent, inverted.
    * ties on authority AND degree keep the first row seen, which is fan-out order
      (the source list), so the result is deterministic for identical input.

    Nothing here reads the question, the corpus, or the shape of a name.
    """
    best: dict[str, tuple[int, float, str]] = {}
    emails_by_name: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        name = str(r.get("name") or "").strip()
        email = str(r.get("person_email") or "").strip()
        if not name or not email:
            continue
        emails_by_name[name].add(email)
        rank = (authority.get(str(r.get("graph") or ""), 0),
                float(r.get("degree") or 0.0))
        current = best.get(email)
        if current is None or rank > current[:2]:
            best[email] = (*rank, name)
    renames: dict[str, str] = {}
    for name, emails in emails_by_name.items():
        if len(emails) != 1:
            continue                      # one name, several humans — leave it
        canonical = best[next(iter(emails))][2]
        if canonical != name:
            renames[name] = canonical
    return renames


def _apply_edge_description(row: dict, principals: Sequence[str] | None) -> None:
    """Fill an edge row's `description`/`date`/`date_basis` from its visible entries.

    Per-entry filtering does double duty on edges: the visible entries supply the
    description text AND the edge's date (each carries its source chunk's
    documented_on) — so under ACL the date can only come from an entry the caller
    may read. Same rule as ``_acl_conditional_description``, inlined because the
    entries are already parsed: ACL off ⇒ the blend may serve, ACL on ⇒ only the
    filtered entries. Edges reverse the entity coverage shape (plain ≈0%, json
    100%), so ACL-off leans on json in practice.
    """
    entries = _visible_entries(row.pop("descriptions_json", None), principals)
    texts = [t for e in entries if (t := (e.get("text") or "").strip())]
    dates = [d for e in entries if (d := e.get("documented_on"))]
    json_desc = " ".join(texts)[:_DESCRIPTION_CHARS] or None
    plain = row.pop("rel_description", None)
    row["description"] = (
        ((plain or "").strip() or json_desc) if principals is None else json_desc
    )
    row["date"] = max(dates) if dates else None
    row["date_basis"] = "documented" if dates else None


def _log_degree(degree) -> float:
    """Connectedness on a compressed scale — the degree boost's raw term."""
    return math.log10(1.0 + max(0, int(degree or 0)))


def _level_scores_per_graph(facts: list[RelationFact]) -> list[RelationFact]:
    """Damp the cross-graph scale gap in ``score``, keeping the read's own scale.

    Per-graph counts (edge support, contribution counts) hand the denser corpus
    the higher score on every row, so the ranking follows corpus size rather than
    the rows. Each graph is scaled by ``sqrt(ceiling / own peak)``: within a graph
    the spacing the scoring curve produced is untouched, the dominant graph keeps
    its raw values (so downstream absolute thresholds — the neighbour scorer's
    separation margin — keep their meaning), and a weaker graph is lifted halfway
    toward the ceiling, never onto it — a full rescale to the peak would make its
    best row read as strong as the dominant graph's best on density alone.

    A no-op on a single graph, and on an all-zero fanout.
    """
    peaks = graph_peaks((f.graph, f.score) for f in facts)
    ceiling = max(peaks.values(), default=0.0)
    if ceiling <= 0:
        return facts
    return [
        f.model_copy(update={"score": round(f.score * math.sqrt(ceiling / peak), 4)})
        if (peak := peaks.get(f.graph) or 0.0) > 0
        else f
        for f in facts
    ]


def _merged_vocabulary(rows: list[dict]) -> list[dict]:
    """One relationship vocabulary out of the fan-out's per-graph concatenation.

    ``$limit`` binds PER GRAPH and ``_fan_out`` concatenates, so slicing the
    merged list positionally returned the head of the FIRST graph and nothing
    else once both graphs filled their own limit — measured, 400 rows in, and
    the 60 the agent was shown were 100% graph one, while the membership verbs
    an org question needs (HAS_MEMBER, INCLUDES_MEMBER, …) sat in graph two.
    A type present in both graphs also spent two of those slots.

    Summing the counts per type makes the ordering a real cross-graph one and
    gives each type a single slot, so the same budget carries strictly more
    vocabulary. The tie-break on name keeps two identical requests identical.
    """
    edges: dict[str, int] = defaultdict(int)
    for r in rows:
        if name := str(r.get("type") or "").strip():
            edges[name] += int(r.get("edges") or 0)
    return [
        {"type": name, "edges": count}
        for name, count in sorted(edges.items(), key=lambda kv: (-kv[1], kv[0]))
    ]


def _relation_fact(row: dict) -> RelationFact:
    """One row→model mapping shared by every relationship-returning query."""
    return RelationFact(
        source=row["source"],
        target=row["target"],
        target_type=row.get("target_type"),
        target_status=row.get("target_status"),
        source_status=row.get("source_status"),
        relationship=row["relationship"],
        specific_relationships=row.get("specific_relationships"),
        description=row.get("description") or row.get("target_description"),
        direction=row.get("direction"),
        date=row.get("date"),
        date_basis=row.get("date_basis"),
        support=row.get("support") or 1,
        structural=bool(row.get("structural")),
        score=row.get("score") or 0.0,
        graph=row.get("graph"),
        depth=row.get("depth"),
        chunk_ids=[str(c) for c in (row.get("chunk_ids") or []) if c],
        origin=row.get("origin"),
    )


# Preview width for a cited passage on the answer's graph (a node label and its
# hover text) — a rendering width, not a retrieval knob.
_PASSAGE_PREVIEW_CHARS = 240


def _stamp_origin(facts: list[RelationFact], origin: str) -> list[RelationFact]:
    """Name the read that produced these facts (RelationFact.origin), in place."""
    for f in facts:
        f.origin = f.origin or origin
    return facts


def _chunk_quote(row: dict, *, text: str | None = None) -> ChunkQuote:
    """One row→ChunkQuote mapping shared by every chunk-returning query. `via` is
    the entity names the chunk MENTIONS (surfaced as ``via_names``), carried through
    so the trace can draw chunk→entity edges."""
    return ChunkQuote(
        chunk_id=row["chunk_id"],
        doc_id=row.get("doc_id"),
        doc_title=row.get("doc_title"),
        text=text if text is not None else (row.get("text") or "").strip(),
        subject_status=row.get("subject_status"),
        via=row.get("via_names") or [],
        graph=row.get("graph"),
    )


def _fuzzy_lucene(text: str) -> str:
    """User phrase → fuzzy Lucene query for the entity_aliases index.

    Words ≥5 chars get edit-distance-2 fuzziness ("Entrprice" still matches
    "Enterprise"); short words stay exact to avoid noise; terms are OR-ed.
    """
    parts = []
    for token in re.findall(r"[a-z0-9]+", (text or "").lower()):
        if len(token) < 2:
            continue
        parts.append(f"{token}~2" if len(token) >= 5 else token)
    return " ".join(parts)


class GraphReadRepository:
    def __init__(
        self,
        graph: Neo4jGraph | None,
        settings: Settings,
        *,
        vector_store: QdrantVectorStore | None = None,
        http: Neo4jHttpClient | None = None,
        sources: list[GraphSource] | None = None,
        include_inactive_people: bool = False,
    ) -> None:
        if graph is None and http is None:
            raise ValueError("GraphReadRepository needs a Bolt graph or an HTTP client")
        self._graph = graph
        self._http = http  # Query API v2 transport (servers that don't expose Bolt)
        self._s = settings
        # Entity ANN runs in Qdrant (`entity_embeddings`) — Neo4j has no vector
        # index. None ⇒ the vector arm of entity resolution is skipped.
        self._vectors = vector_store
        # Graph sources every read spans (one = single-graph; several = fan-out +
        # merge across Neo4j databases + Qdrant collections). Defaults to config.
        self._sources = sources or settings.graph_sources()
        # Whether people the graphs record as no longer employed may be returned
        # at all (bound as $include_inactive_people on every execution). False is
        # the default and the answer for a current-state question; a request whose
        # question is historical takes a view with it True (see
        # ``including_inactive_people`` and retrieval/person_status.py).
        self._include_inactive_people = include_inactive_people
        # (fetched_at, rows) for this repository's OWN sources — see
        # relationship_types. Not keyed by tenant: the vocabulary is a property of
        # the databases this instance reads, which are fixed at construction.
        self._vocab_cache: tuple[float, list[dict]] | None = None

    @property
    def sources(self) -> list[GraphSource]:
        """The graph sources this repository reads, in fan-out order.

        Read-only and deliberately exposed: per-request run selection means the
        resolved sources can differ from ``settings.graph_sources()``, so a pass
        that needs a source's Qdrant collection must ask the repository rather
        than re-reading config and getting the boot-time defaults.
        """
        return list(self._sources)

    @property
    def vector_store(self):
        """The Qdrant store, or None when the vector arm is disabled."""
        return self._vectors

    def with_sources(self, sources: list[GraphSource]) -> GraphReadRepository:
        """A request-scoped view over the SAME connections but a different graph
        source list — used by per-request run selection (RunConfig) so a request
        reads the Neo4j DB + Qdrant collections its resolved run built, without
        rebuilding the driver/http/vector clients. Gets its own (empty) vocab
        cache so a run's cached rows never leak across sources."""
        return GraphReadRepository(
            self._graph, self._s, vector_store=self._vectors, http=self._http,
            sources=list(sources),
            include_inactive_people=self._include_inactive_people,
        )

    def including_inactive_people(self, include: bool) -> GraphReadRepository:
        """A view over the SAME connections that does (or does not) return people
        the graphs record as no longer employed.

        Set once per request from the question's own temporal reading, so the
        employment gate is a property of the REQUEST rather than a flag each call
        site remembers to pass — the same shape ``with_sources`` uses for run
        selection. Returns ``self`` when nothing changes, which is the common
        case: only an explicitly historical question opens the gate."""
        if include == self._include_inactive_people:
            return self
        return GraphReadRepository(
            self._graph, self._s, vector_store=self._vectors, http=self._http,
            sources=list(self._sources), include_inactive_people=include,
        )

    # ── base runner ────────────────────────────────────────────────────────

    def _principals(self, ctx: AccessContext) -> list[str] | None:
        """The principal set to enforce, or ``None`` to bypass ACL filtering
        (``acl_enforcement_enabled`` off). ``None`` is the same "no ACL" sentinel
        the null-guarded Cypher predicates, the Qdrant store, and the per-source
        description filter all understand — one switch, no per-query variants.

        The per-request ``ctx.acl_enforced`` overrides the setting when present
        (from ``options.disabled_acl``); None defers to ``acl_enforcement_enabled``."""
        enforced = (
            self._s.acl_enforcement_enabled if ctx.acl_enforced is None else ctx.acl_enforced
        )
        return list(ctx.principals) if enforced else None

    async def _run_one(
        self, prepared: str, ctx: AccessContext, source: GraphSource, params: dict
    ) -> list[dict]:
        """Run a prepared query against ONE graph source's database; tag each row
        with the source name. ``entity_index`` is bound per-source so the
        entity-resolution fulltext query hits the right index name on each graph
        (unused by other queries — Neo4j ignores extra params)."""
        bound = {
            "principals": self._principals(ctx),
            "entity_index": source.entity_fulltext_index,
            # The employment gate every person-bearing read carries. Bound on
            # EVERY execution for the same reason $principals is: a read that
            # could forget it is a read that can leak a former employee into a
            # current-people answer.
            "include_inactive_people": self._include_inactive_people,
            **params,
        }
        if self._http is not None:
            rows = await self._http.rows(
                prepared, bound, database=source.database, log_fields=ctx.log_fields()
            )
        else:
            try:
                async with self._graph.session(database=source.database) as session:
                    result = await session.run(
                        Query(prepared, timeout=self._s.neo4j_tx_timeout_s), **bound
                    )
                    rows = await result.data()
            except (ServiceUnavailable, SessionExpired) as exc:
                logger.warning("neo4j_unavailable", graph=source.name,
                               error=str(exc), **ctx.log_fields())
                raise GraphUnavailable() from exc
            except Neo4jError as exc:
                # A failed query never reaches the graph_read log below, so embed
                # the Cypher + params here — the failing statement is self-contained.
                logger.error(
                    "❌ neo4j_query_failed " + json.dumps(
                        {"store": "neo4j", "graph": source.name, "database": source.database,
                         "code": exc.code, "error": str(exc)[:300],
                         "cypher": " ".join(prepared.split()),
                         "params": _loggable({k: v for k, v in bound.items()
                                              if k not in _BOUND_EVERYWHERE})},
                        ensure_ascii=False, default=str),
                    **ctx.log_fields())
                raise
        principals = bound.get("principals")
        for r in rows:
            if not isinstance(r, dict):
                continue
            r.setdefault("graph", source.name)
            # 🔒 ACL node audit: queries that carry the node's ACL surface it as
            # `_acl_log`; record node + acl + isACLMatch, then strip it so the row
            # shape downstream is unchanged. isACLMatch is None when ACL is off.
            if "_acl_log" in r:
                acl = r.pop("_acl_log") or []
                matched = None if principals is None else bool(set(principals) & set(acl))
                acl_audit.record(
                    node=_node_id(r), graph=source.name, acl=acl, is_match=matched
                )
        return rows

    async def _rows(self, query: str, ctx: AccessContext, **params: Any) -> list[dict]:
        name = _QUERY_NAMES.get(query, "?")
        # One span per named read, covering the fan-out over every
        # source — the unit an operator reasons about ("EXPAND_NEIGHBORS took
        # 10s"), not one span per database. Nests under the tool span that
        # triggered it, so a trace answers what a tool actually read. The raw
        # Cypher stays in the log line below: a span carries the query NAME and
        # shape, keeping Langfuse a behaviour tool rather than a query profiler.
        graphs = [src.name for src in self._sources]
        # WHICH graph this tool traversed, stamped on the TOOL span:
        # the repository knows the sources, the tool span is what a reader opens.
        # Inside a tool body that span is the current one, so annotating "here"
        # lands it beside the tool name, input and output.
        annotate_current_span(graphs_queried=graphs)
        query_params = {k: v for k, v in params.items()
                        if k not in _BOUND_EVERYWHERE}
        with stage_span(
            f"neo4j.{name}",
            store="neo4j",
            query_name=name,
            graphs=graphs,
            databases=[src.database for src in self._sources],
        ) as span:
            # Input/Output on the read itself: the bound parameters (compacted the
            # same way the log line compacts them) and what came back. Row COUNT
            # alone could not answer "did this graph contribute anything?".
            record_io(span, input=json.dumps(
                {"query": name, "params": _loggable(query_params)},
                ensure_ascii=False, default=str,
            ), limit=_TRACE_IO_CHARS)
            rows = await self._fan_out(query, ctx, params)
            by_graph = _rows_by_graph(rows)
            hit = sorted(by_graph)
            set_span_attributes(span, rows=len(rows), graphs_hit=hit,
                                rows_by_graph=json.dumps(by_graph))
            record_io(span, output=json.dumps(
                {"rows": len(rows), "rows_by_graph": by_graph,
                 "sample": _loggable(rows[0]) if rows else None},
                ensure_ascii=False, default=str,
            ), limit=_TRACE_IO_CHARS)
            # A graph that was queried but returned nothing is a real finding
            # (a missing collection, an ACL filter, a run mismatch) — surface it
            # on the tool, not only on this child.
            annotate_current_span(graphs_with_rows=hit)
            return rows

    async def _fan_out(
        self, query: str, ctx: AccessContext, params: dict[str, Any]
    ) -> list[dict]:
        # Null-guard the ACL predicates (memoized), then fan out across every
        # configured graph source (different Neo4j database + fulltext index
        # names) and merge the rows, each tagged with its source graph.
        # $principals is bound per-source (None ⇒ predicate always true, no filter).
        # One source ⇒ unchanged single-graph behavior; in multi-graph a per-source
        # failure is skipped so one bad graph never fails the whole request.
        prepared = cypher.prepare(query, self._s.graph_acl_property)
        if len(self._sources) == 1:
            rows = await self._run_one(prepared, ctx, self._sources[0], params)
        else:
            results = await asyncio.gather(
                *(self._run_one(prepared, ctx, s, params) for s in self._sources),
                return_exceptions=True,
            )
            rows = []
            for s, res in zip(self._sources, results, strict=True):
                if isinstance(res, BaseException):
                    logger.warning("graph_read_skipped", graph=s.name,
                                   error=str(res), **ctx.log_fields())
                    continue
                rows.extend(res)
        # Message-EMBEDDED (survives pipelines that surface only `message`): which
        # named Cypher ran, across which Neo4j database(s), and how many rows came
        # back. The named parameters (minus $principals/$entity_index, bound per
        # source) show what the query was actually asked.
        query_params = {k: v for k, v in params.items()
                        if k not in _BOUND_EVERYWHERE}
        logger.info(
            "🔎 graph_read " + json.dumps(
                {"store": "neo4j",
                 "query": _QUERY_NAMES.get(query, "?"),
                 "databases": [s.database for s in self._sources],
                 # The actual Cypher hitting Neo4j (ACL-guarded/renamed, whitespace
                 # collapsed to one line) + the params it was bound with.
                 "cypher": " ".join(prepared.split()),
                 "params": _loggable(query_params),
                 "rows": len(rows)},
                ensure_ascii=False, default=str),
            **ctx.log_fields(),
        )
        return rows

    # ── entity resolution ──────────────────────────────────────────────────

    async def resolve_entities(
        self,
        ctx: AccessContext,
        text: str,
        embedding: list[float] | None,
        limit: int | None = None,
    ) -> list[EntityHit]:
        """Hybrid, typo-tolerant, ACL-guarded entity resolution.

        Blends fuzzy full-text (typos, partial names) with semantic ANN over
        the Qdrant ``entity_embeddings`` collection (paraphrases; entities carry
        no vectors in Neo4j), plus a log-degree boost so well-connected entities
        beat thin variants. Degrades to lexical-only when the embedding or the
        vector store is unavailable. Merged on ``entity_id`` (ingestion's stable
        identity — names can repeat)."""
        limit = limit or self._s.resolve_limit
        merged: dict[str, dict] = {}

        lucene = _fuzzy_lucene(text)
        if lucene:
            rows = await self._rows(
                cypher.RESOLVE_BY_FULLTEXT, ctx, lucene=lucene, limit=self._s.fulltext_k
            )
            # Damped per INDEX: `score` is Lucene's, a function of the index that
            # produced it (term frequencies, document frequencies, average field
            # length), and each graph maintains its own. One global max scaled every
            # graph by whichever index produced the largest raw number, burying a
            # graph's best match behind another's mediocre one; a plain per-index
            # peak made every index's best a perfect 1.0 whatever it matched. The
            # sqrt-damped divisor lifts the weaker index halfway.
            scalers = graph_scalers((r.get("graph"), float(r["score"])) for r in rows)
            for r in rows:
                key = r.get("entity_id") or r["name"]
                hit = merged.setdefault(key, {**r, "score": 0.0, "via": set()})
                hit["score"] += self._s.entity_w_lexical * (
                    r["score"] / (scalers.get(r.get("graph")) or 1.0)
                )
                hit["via"].add("lexical")

        if embedding and self._vectors is not None:
            # ANN in Qdrant (ACL pushed down when enforced), then hydrate the ids
            # in Neo4j, re-guarded there. Fan out over every graph source's entity
            # collection and merge by id. Collections are already tenant-scoped.
            sim_by_id: dict[str, float] = {}
            id_field = self._s.qdrant_entity_id_field
            # None ⇒ ACL enforcement off ⇒ the store skips the MatchAny leg.
            principals = self._principals(ctx)
            for src in self._sources:
                points = await self._vectors.search(
                    src.entities_collection,
                    embedding,
                    top_k=limit * self._s.vector_overscan,
                    principals=principals,
                )
                for p in points:
                    # configured key first, then the two known generations' keys
                    eid = str(
                        p.payload.get(id_field)
                        or p.payload.get("_id")
                        or p.payload.get("entity_id")
                        or ""
                    )
                    if eid:
                        sim_by_id[eid] = max(sim_by_id.get(eid, 0.0), p.score)
            if sim_by_id:
                rows = await self._rows(cypher.ENTITIES_BY_ID, ctx, ids=list(sim_by_id))
                for r in rows:
                    key = r["entity_id"]
                    hit = merged.setdefault(key, {**r, "score": 0.0, "via": set()})
                    hit["score"] += self._s.entity_w_semantic * float(sim_by_id.get(key, 0.0))
                    hit["via"].add("semantic")

        # `degree` is an edge count inside ONE graph, so a denser corpus lifts every
        # one of its entities. Scale-PRESERVING, exactly like _level_scores_per_graph
        # and for the same reason: the dominant graph keeps its raw term — so a
        # single graph, and every request before the split, score byte-for-byte as
        # they always did — and a sparser graph's is lifted by sqrt(ceiling / own
        # peak), closing half the density gap without equating a sparse graph's hub
        # with a dense one's. A damped DIVISOR here did the cross-graph half right
        # but silently re-weighted degree against the lexical/semantic terms on
        # every request (single-graph included), by a factor that moved with
        # whatever peak this result set happened to contain; how much connectedness
        # is worth is `entity_w_degree`'s decision, not the corpus's.
        degree_peaks = graph_peaks(
            (h.get("graph"), _log_degree(h.get("degree"))) for h in merged.values()
        )
        degree_ceiling = max(degree_peaks.values(), default=0.0)
        degree_lift = {g: math.sqrt(degree_ceiling / p)
                       for g, p in degree_peaks.items() if p > 0}
        hits = [
            EntityHit(
                entity_id=h.get("entity_id"),
                name=h["name"], type=h.get("type"),
                description=_acl_conditional_description(
                    h.get("description"), h.get("descriptions_json"), self._principals(ctx)
                ),
                degree=h.get("degree") or 0,
                score=round(
                    h["score"]
                    + self._s.entity_w_degree
                    * (_log_degree(h.get("degree"))
                       * degree_lift.get(h.get("graph"), 1.0)),
                    4,
                ),
                via=sorted(h["via"]),
                graph=h.get("graph"),
                status=h.get("employment_status"),
            )
            for h in merged.values()
        ]
        hits.sort(key=lambda e: e.score, reverse=True)
        return hits[:limit]

    # ── traversal ──────────────────────────────────────────────────────────

    async def expand_entity(
        self,
        ctx: AccessContext,
        name: str,
        relationship_types: list[str] | None = None,
        limit: int | None = None,
    ) -> list[RelationFact]:
        """Support-ranked neighborhood of one entity. Edge/neighbor descriptions
        and the edge date are assembled from the per-source ``descriptions_json``
        entries the caller may read (rel entries carry ``documented_on``)."""
        rows = await self._rows(
            cypher.EXPAND_NEIGHBORS, ctx,
            name=name,
            types=relationship_types,
            support_weight=self._s.support_weight,
            limit=limit or self._s.neighbor_limit,
        )
        return _stamp_origin(self._neighbor_facts(rows, self._principals(ctx)), "expand")

    async def cited_subgraph(
        self, ctx: AccessContext, doc_ids: Sequence[str], *, limit: int | None = None
    ) -> CitedSubgraph:
        """What the graph records about the documents an answer cited — the
        entities their passages mention and the relationships extracted from
        those passages (cypher.CITED_MENTIONS / CITED_EDGES), for the answer's
        graph. Two batched reads, concurrent, both bounded by the citations.

        Additive and read-only: the pool is not touched, and a failure here
        costs the answer its graph's grounding, never the answer.
        """
        ids = list(dict.fromkeys(d for d in doc_ids if d))
        if not ids:
            return CitedSubgraph()
        limit = limit or self._s.answer_graph_cited_limit
        principals = self._principals(ctx)
        mention_rows, edge_rows, chunk_rows = await asyncio.gather(
            self._rows(cypher.CITED_MENTIONS, ctx, doc_ids=ids, limit=limit),
            self._rows(cypher.CITED_EDGES, ctx, doc_ids=ids,
                       support_weight=self._s.support_weight, limit=limit),
            self._rows(cypher.CITED_CHUNKS, ctx, doc_ids=ids,
                       chars=_PASSAGE_PREVIEW_CHARS, limit=limit),
        )
        entities: list[EntityHit] = []
        entity_docs: dict[str, list[str]] = {}
        entity_chunks: dict[str, list[str]] = {}
        chunk_docs: dict[str, str] = {}
        for r in chunk_rows:
            if r.get("chunk_id") and r.get("doc_id"):
                chunk_docs.setdefault(str(r["chunk_id"]), str(r["doc_id"]))
        for r in mention_rows:
            name = str(r.get("name") or "").strip()
            if not name:
                continue
            docs = [str(d) for d in (r.get("doc_ids") or []) if d]
            entity_docs[name] = list(dict.fromkeys([*entity_docs.get(name, []), *docs]))
            cids = [str(c) for c in (r.get("chunk_ids") or []) if c]
            entity_chunks[name] = list(dict.fromkeys([*entity_chunks.get(name, []), *cids]))
            for c in cids:
                # One document per chunk by construction; the first seen wins.
                chunk_docs.setdefault(c, docs[0] if docs else "")
            entities.append(EntityHit(
                entity_id=r.get("entity_id"), name=name, type=r.get("type"),
                description=_acl_conditional_description(
                    r.get("description"), r.get("descriptions_json"), principals),
                degree=int(r.get("degree") or 0), via=["cited"], graph=r.get("graph"),
                status=r.get("employment_status"),
            ))
        facts = _stamp_origin(self._neighbor_facts(edge_rows, principals), "cited")
        fact_docs: dict[str, list[str]] = {}
        for r, f in zip(edge_rows, facts, strict=True):
            key = f"{f.source}|{f.relationship}|{f.target}"
            docs = [str(d) for d in (r.get("doc_ids") or []) if d]
            fact_docs[key] = list(dict.fromkeys([*fact_docs.get(key, []), *docs]))
            for c in f.chunk_ids:
                chunk_docs.setdefault(c, docs[0] if docs else "")
        chunks = [
            ChunkQuote(chunk_id=str(r["chunk_id"]), doc_id=r.get("doc_id"),
                       doc_title=r.get("doc_title"), text=str(r.get("text") or ""),
                       graph=r.get("graph"))
            for r in chunk_rows if r.get("chunk_id")
        ]
        return CitedSubgraph(
            entities=entities, facts=facts, entity_docs=entity_docs,
            fact_docs=fact_docs, chunk_ids=list(chunk_docs), chunk_docs=chunk_docs,
            entity_chunks=entity_chunks, chunks=chunks,
        )

    async def facts_among(
        self, ctx: AccessContext, names: Sequence[str], *, limit: int
    ) -> list[RelationFact]:
        """The relationships AMONG ``names`` — the induced subgraph of the pool's
        own entities, one read (the post-loop mention pass).

        ``expand_entity`` walks one entity's neighbourhood per call, so two
        entities the evidence found separately sit in the pool unconnected however
        directly the graph links them. This returns every visible edge whose BOTH
        endpoints are already evidence, mapped through the same row→fact path as
        the expansion so descriptions and dates follow the same ACL rules.
        Bounded by ``names`` (the caller's pool) and ``limit``.
        """
        wanted = list(dict.fromkeys(n for n in names if n))
        if len(wanted) < 2 or limit <= 0:
            return []
        rows = await self._rows(
            cypher.FACTS_AMONG, ctx, names=wanted,
            support_weight=self._s.support_weight, limit=limit,
        )
        return _stamp_origin(self._neighbor_facts(rows, self._principals(ctx)), "induced")

    async def mentioned_entities(
        self, ctx: AccessContext, chunk_weights: Mapping[str, float], *, limit: int
    ) -> list[EntityHit]:
        """The entities the given chunks MENTION, hydrated (type, description,
        degree) — the post-loop mention pass's read.

        ``chunk_weights`` maps each chunk id to that chunk's question-similarity
        (``ChunkQuote.relevance``, the score the digest already ranks graph
        chunks by). A passage's relevance is SHARED by the entities it mentions:
        an entity's weight is the sum, over the passages that mention it, of that
        passage's relevance divided by how many of the pool's entities it
        mentions. So the person a focused passage is about (six mentions) ranks
        far above the forty names a marketing datasheet lists in one chunk, even
        when the datasheet scored slightly higher against the question — measured
        on the live graph, both a raw mention count and a best-passage rule
        returned investors and compliance acronyms and dropped the one person the
        answer named. Ties break on mention count, then degree. Unscored chunks
        (0.0 — vector store unavailable) contribute nothing, so the order falls
        back to count and degree. ``score`` carries the weight; ``via`` says how
        the entity reached the pool.
        """
        ids = list(dict.fromkeys(c for c in chunk_weights if c))
        if not ids or limit <= 0:
            return []
        # A ceiling on candidates, never the ranking: the ranking needs every
        # mentioned entity's chunk list, and one chunk mentions ~10 entities.
        rows = await self._rows(
            cypher.MENTIONED_ENTITIES, ctx, chunk_ids=ids,
            limit=max(limit, len(ids)) * _MENTION_CANDIDATES_PER_SLOT,
        )
        principals = self._principals(ctx)
        # How many candidate entities each chunk mentions — the denominator that
        # shares a passage's relevance out among them.
        fan_out: dict[str, int] = {}
        for r in rows:
            for c in r.get("chunk_ids") or []:
                if c in chunk_weights:
                    fan_out[c] = fan_out.get(c, 0) + 1
        hits: list[tuple[float, int, EntityHit]] = []
        for r in rows:
            if not r.get("name"):
                continue
            mentioning = [c for c in (r.get("chunk_ids") or []) if c in chunk_weights]
            weight = sum(chunk_weights[c] / fan_out[c] for c in mentioning)
            hits.append((weight, len(mentioning), EntityHit(
                entity_id=r.get("entity_id"),
                name=r["name"], type=r.get("type"),
                description=_acl_conditional_description(
                    r.get("description"), r.get("descriptions_json"), principals
                ),
                degree=r.get("degree") or 0,
                score=round(weight, 4),
                via=["mention"],
                graph=r.get("graph"),
            )))
        # The fan-out concatenates per graph; one comparable order across both.
        hits.sort(key=lambda h: (-h[0], -h[1], -h[2].degree, h[2].name))
        return [h for _, _, h in hits[:limit]]

    def _neighbor_facts(
        self, rows: list[dict], principals: Sequence[str] | None
    ) -> list[RelationFact]:
        """EXPAND_NEIGHBORS-shaped rows → facts, with the per-source
        ``descriptions_json`` entries filtered to what the caller may read."""
        facts = []
        for r in rows:
            _apply_edge_description(r, principals)
            r["target_description"] = _acl_conditional_description(
                r.pop("target_description_plain", None),
                r.pop("target_descriptions_json", None),
                principals,
            )
            facts.append(_relation_fact(r))
        # `score` is support_weight * log10(1+support), and `support` counts the
        # chunks of ONE graph that evidence the edge — so the graph holding more
        # text per relationship outranks the other on every row, whatever the edges
        # actually say. The damped leveling keeps the per-graph ORDER the query
        # produced and the raw curve's scale.
        return _level_scores_per_graph(facts)

    async def entity_paths(
        self,
        ctx: AccessContext,
        seeds: list[str],
        *,
        target: str | None = None,
        relationship_types: list[str] | None = None,
        hops: int | None = None,
        limit: int | None = None,
    ) -> list[RelationFact]:
        """The relationships ALONG multi-hop paths out of `seeds`, one fact per hop.

        The structural counterpart to ``beam_traverse_chunks``, which walks the same
        graph and returns the chunks it lands on. `target` narrows to paths that
        reach one named entity ("how does A relate to B"); without it this is the
        neighbourhood `hops` out. Each fact carries `depth`, so a path is
        recoverable from its hops and the caller can see how far out it reached.

        `hops` is substituted into the Cypher (a var-length bound cannot be a
        parameter) and clamped, exactly as ``beam_traverse_chunks`` does it.
        """
        if not seeds:
            return []
        hops = max(1, min(4, int(hops if hops is not None else self._s.beam_hops)))
        query = cypher.ENTITY_PATHS.replace("__HOPS__", str(hops))
        rows = await self._rows(
            query, ctx,
            seeds=list(seeds),
            target=target or None,
            types=relationship_types or None,
            support_weight=self._s.support_weight,
            leg_limit=self._s.beam_leg_limit,
            limit=limit or self._s.neighbor_limit,
        )
        principals = self._principals(ctx)
        facts = []
        for r in rows:
            _apply_edge_description(r, principals)
            facts.append(_relation_fact(r))
        # Same damped leveling as expand_entity: `score` is derived from `support`,
        # which counts the chunks of ONE graph evidencing the edge.
        return _stamp_origin(_level_scores_per_graph(facts), "paths")

    # ── people profiles ────────────────────────────────────────────────────

    async def fetch_people_profiles(
        self, ctx: AccessContext, names: Sequence[str]
    ) -> tuple[list[EntityHit], list[SourceDocument]]:
        """Structured profile fields for ``names``, as entities — ONE batched read —
        plus each profiled person's OWN source document, for citations.

        The answer's people usually arrive as relationship TARGETS (a name and
        nothing else) or as entities whose blended ``description`` ingestion
        never filled, which is why enumerated rosters render blank roles. This
        returns what the person nodes actually carry, so the digest can show it.

        The documents are the people objects (Document.doc_id == person_user_id;
        title = the person's name, url = their profile page). Without them a
        roster built from REPORTS_TO edges and these profile fields has NO
        document in the evidence pool that supports it, so synthesis — which can
        only cite documents it was shown — correctly cited nothing. A document
        is returned only for a person whose profile is actually kept: a doc for
        an ambiguous or empty row would put a citable source behind a fact this
        read declined to state.

        CROSS-GRAPH MERGE (one graph may be the system of record for person
        attributes): another graph's person node can be an identity stub — no
        job title, no department, no org fields — and the same human's display
        name often differs across graphs ("(Deactivated)" suffixes, dotted
        usernames), so the name-keyed read
        alone leaves those rendering as bare names. Rows are therefore grouped
        by IDENTITY (person_user_id, email as fallback — never by name), and a
        group whose rows are all attribute-empty gets ONE follow-up read that
        fetches the same humans' attribute-bearing nodes by identity. Within a
        group the most attribute-complete row is the profile everyone in the
        group renders — under each name the evidence actually carries; donor
        rows fetched by identity are never emitted under their own name.

        Not every name is a person and not every person is found: unmatched
        names simply return no row (no error, no placeholder). A person with no
        populated field at all — in any graph — is skipped rather than added as
        an empty entity.
        """
        if not names:
            return [], []
        # Unlimited by row count on purpose — the read is bounded by ``names``,
        # which the caller caps (see cypher.PEOPLE_PROFILES). The fan-out runs it
        # per source and merges; a name legitimately resolving in both graphs is
        # deduped by the evidence pool.
        rows = await self._rows(cypher.PEOPLE_PROFILES, ctx, names=list(names))
        return await self._profiles_from_rows(ctx, rows)

    async def _profiles_from_rows(
        self, ctx: AccessContext, rows: list[dict], *, keep_unprofiled: bool = False
    ) -> tuple[list[EntityHit], list[SourceDocument]]:
        """Profile rows → (entities, their own source documents).

        Shared by the name-keyed read above and the group ENUMERATION read below,
        so both spell a person the same way: one identity merge, one ambiguity
        rule, one profile line, one citable people-object document.

        ``keep_unprofiled`` is the enumeration's flag: there the ROW IS THE
        MEMBER, so a row whose profile renders empty (an identity stub) or whose name
        the ambiguity rule leaves bare still comes back as a name-only hit rather than
        silently shrinking the set. The name-keyed read keeps the old behaviour — a bare
        name there adds nothing the pool does not already hold.
        """
        groups = await self._identity_groups(ctx, rows)
        # A name matching several DIFFERENT people inside one graph is ambiguous:
        # the evidence pool keys entities by name, so attributing one of those
        # profiles would state someone else's job — and with org fields in the
        # line that becomes "23 direct reports" against the wrong person. We
        # cannot tell which one the evidence meant, so the name is left bare
        # (the answer keeps the name it already had). Judged per graph, because
        # the same human legitimately appears in both graphs — and judged on the
        # MERGED lines, after the identity groups resolve: two identity stubs
        # sharing a display name render identically-empty OWN lines (a pre-merge
        # check sees no conflict at all), but their donors render two DIFFERENT
        # people's profiles under that one name — the exact mistake this guard
        # exists to prevent (one stub name, two humans).
        seen: dict[tuple[str | None, str], set[str]] = defaultdict(set)
        for r in rows:
            if line := _profile_line(_best_of_group(groups, r)):
                seen[(r.get("graph"), str(r.get("name") or "").strip())].add(line)
        ambiguous = {name for (_graph, name), lines in seen.items() if len(lines) > 1}
        if ambiguous:
            logger.info(
                "people_profile_ambiguous_names", names=sorted(ambiguous),
                reason="one name matches several person nodes in a graph; profile omitted",
                **ctx.log_fields(),
            )
        profiles: list[EntityHit] = []
        docs: dict[str, SourceDocument] = {}
        for r in rows:
            name = str(r.get("name") or "").strip()
            # The row the profile renders FROM: the most attribute-complete row
            # of this human's identity group, or the row
            # itself when it carries no identity key. The NAME stays the row's
            # own — it is the string the evidence carries.
            source = _best_of_group(groups, r)
            line = _profile_line(source)
            if not name or ((not line or name in ambiguous) and not keep_unprofiled):
                continue
            if name in ambiguous:
                # The enumeration keeps the member — the row is a real element of
                # the set and dropping it undercounted the roster — but attributing
                # either human's profile, status or page would state someone else's
                # job, so the hit is the bare name.
                profiles.append(EntityHit(name=name, type="Person",
                                          entity_id=_identity_key(r),
                                          graph=r.get("graph")))
                continue
            profiles.append(EntityHit(
                name=name,
                type="Person",
                # The cross-graph identity (email, then user id), so the pool can
                # keep two DIFFERENT humans who share a display name apart —
                # name-keyed dedup merged them into one member.
                entity_id=_identity_key(source) or _identity_key(r),
                profile=line or None,
                # Also as a FIELD, not only inside the rendered line: the
                # cross-graph pass and the digest both need to read it back.
                status=INACTIVE if is_inactive(
                    source.get("employment_status"), source.get("departed_on")
                ) else source.get("employment_status"),
                # The person's own page, when their source document carries one —
                # the same union add_entities builds.
                url=source.get("doc_url") or r.get("doc_url"),
                graph=r.get("graph"),
            ))
            for row in (source, r):
                doc_id = str(row.get("doc_id") or "").strip()
                if doc_id and row.get("doc_title"):
                    docs.setdefault(doc_id, SourceDocument(
                        doc_id=doc_id,
                        title=str(row["doc_title"]),
                        url=row.get("doc_url"),
                        source_type=row.get("doc_source_type"),
                        snippet=row.get("doc_snippet"),
                        graph=row.get("graph"),
                    ))
                    break
        return profiles, list(docs.values())

    async def people_by_group(
        self, ctx: AccessContext, group: str, *, limit: int | None = None
    ) -> tuple[list[EntityHit], list[SourceDocument], bool]:
        """EVERY person the graph records in ``group`` — a department, a team, or
        a manager's name — as profiles, their documents, and whether the cap bound.

        The complete-set counterpart to every other people read, all of which are
        top-K over text resemblance: a fuzzy name match, a document-title
        substring, a topic name, or a list of names the caller already has. None
        of them can select on membership itself, which is what a membership
        question is actually asking about — so "who are all the members of X" was
        answered by whatever the ranking happened to surface, and a valid member
        was dropped whenever another candidate scored higher.

        MEMBERSHIP IS RECORDED TWO WAYS and this read covers both, because the
        caller cannot know which one holds the group they named: as person
        ATTRIBUTES (``person_department`` / ``person_team`` / ``person_manager``
        — the HR taxonomy) and as structured ``[:MEMBER_OF]`` edges into group
        NODES (``:Team``, ``:Department``). The two are genuinely different sets,
        and until the second arm existed a structured team returned nobody at all.

        Unranked by construction, so nothing here can prefer one member over
        another; the only bound is an explicit hub-explosion ceiling, per arm and
        per graph, and the third element of the result says whether it bound.
        Schema-driven: nothing reads the question, and the same call answers a
        department, a team, a scrum team and a reporting line because those are
        the shapes the schema carries.
        """
        group = (group or "").strip()
        if not group:
            return [], [], False
        limit = limit or self._s.people_enumeration_limit
        # BOTH representations of membership, in one call, because the caller
        # asking "who is in X" has no way to know which one the graph used for X
        # — and the two hold different sets (the HR department "Data Science" is
        # not the scrum team "Data-Science"). Concurrent: they are independent
        # reads and the ceiling binds per arm, so covering both costs one arm's
        # latency rather than two.
        arms = await asyncio.gather(
            self._rows(cypher.PEOPLE_BY_GROUP, ctx, group=group, limit=limit),
            self._rows(cypher.GROUP_MEMBERS, ctx, group=group, limit=limit),
        )
        # ONE ROW PER HUMAN, before the profile pass, keyed on the person's
        # CROSS-GRAPH IDENTITY — never the display name, which is exactly what
        # differs across graphs for one human ("jane.doe" / "Jane Doe") and
        # exactly what two DIFFERENT humans can share inside one graph. A
        # name-keyed dedup made both mistakes at once: it collapsed two
        # same-named humans into one member and counted one dual-spelled human
        # twice. The most attribute-complete row speaks for the human (the same
        # completeness order the profile merge picks its source row by, with
        # first-seen — fan-out order — breaking ties, so identical input
        # enumerates identically); a row with no identity key can only speak for
        # itself and is deduped on the one thing it can assert, its own
        # (graph, name).
        best: dict[object, dict] = {}
        for r in (row for arm in arms for row in arm):
            key = _identity_key(r) or (
                "row", r.get("graph"), str(r.get("name") or "").strip())
            prev = best.get(key)
            if prev is None or _attr_count(r) > _attr_count(prev):
                best[key] = r
        # One profile pass over the deduped union, so a member found by either
        # arm goes through the same identity merge and renders the same columns.
        # ``keep_unprofiled``: a member with no renderable attributes (an
        # identity stub) and a name the ambiguity rule leaves bare
        # STAY IN THE SET as name-only rows — an enumeration that silently drops
        # a member while reporting ``truncated: false`` is the exact failure this
        # read exists to remove.
        profiles, docs = await self._profiles_from_rows(
            ctx, list(best.values()), keep_unprofiled=True)
        # Per graph AND per arm: each binds its own $limit, so a merged count
        # says nothing about whether either read was cut.
        truncated = any(n >= limit for arm in arms
                        for n in _rows_by_graph(arm).values())
        return profiles, docs, truncated

    async def resolve_groups(
        self, ctx: AccessContext, text: str, *, limit: int | None = None
    ) -> list[EntityHit]:
        """The structured GROUP NODES (:Team, :Department, …) whose name matches
        ``text`` — the discovery step ``people_by_group`` needs a name for.

        Entity resolution matches ``:Entity``, and a group node is not one, so a
        structured team could not be resolved, expanded or enumerated by anything
        in the service. This is that missing step, and it is the group-node
        counterpart of ``resolve_entities`` -> ``expand_entity``: discover the
        exact name the graph uses, then enumerate it exactly.

        Returned as ``EntityHit`` (type = the node's own label, degree = the
        member count) so groups flow through the evidence pool and the digest
        like every other resolved thing. Nothing here reads the question: the
        match is the caller's phrase against the node's own name.
        """
        text = (text or "").strip()
        if not text:
            return []
        rows = await self._rows(
            cypher.GROUPS_RESOLVE, ctx, group=text,
            limit=limit or self._s.resolve_limit,
        )
        # One row per (name, kind) across graphs. The member count is the LARGEST
        # any single graph records, never the sum: the same human is a node in
        # both graphs, so summing would report a team as twice its size — and the
        # enumeration read is the authoritative answer to "how many" anyway.
        best: dict[tuple[str, str | None], tuple[int, dict]] = {}
        for r in rows:
            name = str(r.get("name") or "").strip()
            if not name:
                continue
            key, members = (name, r.get("type")), int(r.get("members") or 0)
            if members > best.get(key, (-1, None))[0]:
                best[key] = (members, r)
        return [
            EntityHit(
                name=str(r["name"]).strip(),
                type=r.get("type"),
                description=f"{members} recorded members" if members else None,
                degree=members,
                via=["groups"],
                graph=r.get("graph"),
            )
            for members, r in sorted(best.values(),
                                     key=lambda pair: (-pair[0], pair[1]["name"]))
        ]

    async def person_identities(
        self, ctx: AccessContext, names: Sequence[str]
    ) -> PersonIdentities:
        """Who the humans behind ``names`` are, and which of them have left.

        TWO batched reads, each fanned out over every graph:

        1. the given names -> the EMAIL of the human behind each. This is all a
           name-keyed read can do, and on its own it cannot fix anything: asked for
           "jane.doe" it returns that one node, because the same human's other
           node is called "Jane Doe" and that string was never asked for.
        2. those emails -> every node for those humans, whatever each graph calls
           them. This is the read that supplies the alternative labels.

        The same two-step, for the same reason, as PEOPLE_PROFILES ->
        PEOPLE_PROFILES_BY_IDENTITY. ``renames`` holds only the names that should
        change; empty means every name already agrees (or carries no email to join
        on), and the caller has nothing to do.

        ``departed`` is the SECOND answer off the same two reads, and it needs
        them both for the same reason the renames do: one graph can call a
        human Active while another holds their deactivated node, and only a
        read that collects every graph's node for one email can see that. THESE
        TWO READS ARE THE ONLY PERSON READS WITHOUT THE EMPLOYMENT GATE — gating
        them would hide exactly the rows the exclusion is looking for.
        """
        if not names:
            return PersonIdentities()
        authority = {s.name: s.person_name_authority for s in self._sources}
        rows = await self._rows(cypher.PERSON_IDENTITIES, ctx, names=list(names))
        emails = sorted({e for r in rows
                         if (e := str(r.get("person_email") or "").strip())})
        if emails:
            # The alternative labels. Additive: if this read fails the pass still
            # has the first read's rows, which fix any duplicate whose labels are
            # BOTH already in the evidence (a cross-graph pair the agent happened
            # to reach from both sides, or an ingestion twin).
            try:
                rows = rows + await self._rows(
                    cypher.PERSON_IDENTITIES_BY_EMAIL, ctx, emails=emails)
            except Exception as exc:
                logger.warning("person_identity_alias_read_failed",
                               error_type=type(exc).__name__, error=str(exc),
                               **ctx.log_fields())
        return PersonIdentities(
            renames=_canonical_person_names(rows, authority),
            departed=departed_person_names(rows),
            # The primary read returned, so the graphs answered the employment
            # question for exactly the names it was given. Set only here — every
            # path that did not get an answer builds the bare PersonIdentities().
            resolved=True,
            asked=frozenset(str(n).strip().casefold() for n in names),
        )

    async def _identity_groups(
        self, ctx: AccessContext, rows: list[dict]
    ) -> dict[str, list[dict]]:
        """Group profile rows by the person's cross-graph identity, fetching
        attribute donors for groups that have none.

        Key: ``person_user_id`` when present, else email — never the name. A
        group whose every row is attribute-empty (identity stubs from a graph
        that stores no person attributes) gets one batched follow-up read
        (PEOPLE_PROFILES_BY_IDENTITY) for the same humans' attribute-bearing
        nodes; donor rows join the group but are never emitted as entities.
        Additive: a failed donor read costs the merge, never the profiles."""
        groups: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            if key := _identity_key(r):
                groups[key].append(r)
        starved = [g for g in groups.values()
                   if not any(_attr_count(r) for r in g)]
        if not starved:
            return groups
        ids = sorted({str(r["person_user_id"]) for g in starved for r in g
                      if r.get("person_user_id")})
        emails = sorted({str(r["person_email"]) for g in starved for r in g
                         if r.get("person_email")})
        try:
            donors = await self._rows(
                cypher.PEOPLE_PROFILES_BY_IDENTITY, ctx, ids=ids, emails=emails,
            )
        except Exception as exc:  # merge is additive — never fatal
            logger.warning("people_profile_identity_merge_failed",
                           error_type=type(exc).__name__, error=str(exc),
                           **ctx.log_fields())
            return groups
        merged = 0
        for donor in donors:
            key = _identity_key(donor)
            if key in groups and _attr_count(donor):
                groups[key].append(donor)
                merged += 1
        if merged:
            logger.info("people_profile_identity_merge",
                        starved_groups=len(starved), donors=merged,
                        **ctx.log_fields())
        return groups

    # ── claims ─────────────────────────────────────────────────────────────

    async def fetch_claims(
        self, ctx: AccessContext, name: str, since_iso: str | None = None
    ) -> list[ClaimFact]:
        rows = await self._rows(
            cypher.FETCH_CLAIMS, ctx,
            name=name, since=since_iso, now=date.today().isoformat(),
            limit=self._s.claim_limit,
        )
        return [ClaimFact(**r) for r in rows]

    # ── temporal ───────────────────────────────────────────────────────────

    async def recent_changes(
        self, ctx: AccessContext, since_iso: str
    ) -> list[ClaimFact]:
        """Dated CLAIMS in the window, recency-ordered. Claim-anchored — dates
        live on :Claim (indexed), not on EXTRACTED_REL edges.

        Returned as claims, which is what the rows are: ``cl.predicate`` /
        ``cl.object`` about one entity, where the object is usually a literal
        ("21 September 2026", "High"). They used to be shaped as RelationFacts, so
        every literal became a graph NODE in the pool's facts — a client drawing
        the evidence graph showed a penetration-testing engagement fanning out to
        forty dates beside an answer about interns. A claim's object is a value,
        never a node.
        """
        rows = await self._rows(
            cypher.RECENT_CHANGES, ctx, since=since_iso, limit=self._s.recent_limit,
        )
        return [
            ClaimFact(
                subject=r["source"], predicate=r["relationship"],
                object=r.get("target") or "",
                # Both UNION legs carry a real date in `eff_date` (cypher.RECENT_CHANGES:
                # cl.valid_from for 'event', cl.documented_on for 'documented'), and the
                # query already ordered on it. Keeping only the event leg's date left every
                # documented-basis change undated, so rank_claims_by_recency sorted it into
                # the undated tail — the opposite of the newest-first this lane promises.
                # `date_basis` travels with the date and is what marks it as a
                # "documented as of" date rather than an event date (synthesis/prompts.py).
                valid_from=r.get("date") or None,
                date_basis=r.get("date_basis"),
                support=r.get("support") or 1,
                graph=r.get("graph"),
            )
            for r in rows
        ]

    # ── grounding ──────────────────────────────────────────────────────────

    async def source_documents(
        self, ctx: AccessContext, entity_names: list[str]
    ) -> list[SourceDocument]:
        if not entity_names:
            return []
        # `limit` is sized from the per-anchor budget so the overall ceiling can never
        # bind — that is what makes the result monotonic in `entity_names` (adding an
        # anchor can only ADD documents). It exists purely as a hard stop against a
        # pathological name list, never as the shaping cap it used to be.
        per_anchor = self._s.source_docs_per_anchor
        rows = await self._rows(
            cypher.SOURCE_DOCUMENTS, ctx, names=entity_names,
            per_anchor=per_anchor, limit=per_anchor * len(entity_names),
        )
        return [SourceDocument(**r) for r in rows]

    async def supporting_chunks(
        self, ctx: AccessContext, entity_names: list[str]
    ) -> list[ChunkQuote]:
        if not entity_names:
            return []
        # Per-anchor, like source_documents: one global LIMIT let the first anchors
        # traversed consume the whole budget and starved the rest of excerpts entirely.
        # `limit` is sized so it cannot bind and re-create that.
        per_anchor = self._s.quotes_per_anchor
        rows = await self._rows(
            cypher.SUPPORTING_CHUNKS, ctx,
            names=entity_names, chars=self._s.quote_chars,
            per_anchor=per_anchor, limit=per_anchor * len(entity_names),
        )
        return [_chunk_quote(r) for r in rows]

    async def chunks_by_similarity(
        self, ctx: AccessContext, embedding: list[float], *, limit: int
    ) -> list[ChunkQuote]:
        """Chunks most similar to the QUESTION — the one chunk path needing no entity.

        Every other route reaches chunk text by traversing from a resolved
        entity, so a question entity resolution finds nothing for reaches no
        graph text at all. Similarity has no such precondition.

        Fans out across every source that has a chunk index and merges on ONE
        comparable cosine scale, so a second graph competes on relevance rather
        than on its position in the merged row list. ACL is pushed into Qdrant
        and re-checked in Neo4j (defense in depth), exactly as the entity ANN does.
        """
        if not embedding or self._vectors is None or limit <= 0:
            return []
        sim: dict[str, float] = {}
        principals = self._principals(ctx)
        for src in self._sources:
            if not src.chunks_collection:
                continue
            points = await self._vectors.search(
                src.chunks_collection, embedding,
                top_k=limit * self._s.vector_overscan, principals=principals,
            )
            for p in points:
                # Named `entity_id` in the payload; the VALUE is the chunk_id.
                cid = str(p.payload.get(self._s.qdrant_entity_id_field) or "")
                if cid:
                    sim[cid] = max(sim.get(cid, 0.0), p.score)
        if not sim:
            return []
        rows = await self._rows(
            cypher.CHUNKS_BY_ID, ctx, ids=list(sim), chars=self._s.quote_chars,
        )
        quotes = [_chunk_quote(r) for r in rows]
        quotes.sort(key=lambda q: (-sim.get(q.chunk_id, 0.0), q.chunk_id))
        return quotes[:limit]

    async def beam_traverse_chunks(
        self, ctx: AccessContext, seeds: list[str], *, hops: int | None = None
    ) -> list[dict]:
        """Beam-bounded multi-hop traversal from seed entities to the visible
        chunks reachable within `hops` (POC A6). The chunk is the product — the
        graph is only the router. `hops` is substituted into the Cypher (a
        var-length bound cannot be a parameter); it is clamped to a small int so
        the substitution is injection-safe and hub-explosion stays bounded."""
        if not seeds:
            return []
        hops = max(1, min(4, int(hops if hops is not None else self._s.beam_hops)))
        query = cypher.BEAM_TRAVERSAL.replace("__HOPS__", str(hops))
        return await self._rows(
            query, ctx,
            seeds=list(seeds),
            leg_limit=self._s.beam_leg_limit,
            chars=self._s.quote_chars,
            limit=self._s.beam_chunk_limit,
        )

    async def recent_change_chunks(
        self, ctx: AccessContext, since: str, *, topic: str | None = None
    ) -> list[dict]:
        """Claim-date → supporting chunk quotes for a time window (POC B7). The
        verbatim chunk text that evidences a dated change, for synthesis to quote
        (the edge-anchored `recent_changes` returns relationship tuples instead)."""
        return await self._rows(
            cypher.RECENT_CHANGE_CHUNKS, ctx,
            since=since,
            topic=(topic or None),
            chars=self._s.quote_chars,
            limit=self._s.recent_change_chunk_limit,
        )

    async def document_lookup(
        self, ctx: AccessContext, titles: Sequence[str], *, per_title: int | None = None
    ) -> list[SourceDocument]:
        """Find documents by title substring (ACL-filtered) — the discovery step
        before fetch_document_text (B6).

        Takes MANY titles in one read, each with its own row budget:
        covering a series used to cost a call per member, and a single global limit
        let one crowded title spend rows the others needed. ``limit`` is sized so it
        cannot bind and re-create that — the same treatment ``supporting_chunks``
        gives its anchors, for the same reason.
        """
        wanted = list(dict.fromkeys(
            t.strip() for t in titles if t and t.strip()
        ))
        if not wanted:
            return []
        per_title = per_title or self._s.documents_per_title
        rows = await self._rows(
            cypher.DOCUMENT_LOOKUP, ctx, titles=wanted,
            per_title=per_title, limit=per_title * len(wanted),
        )
        return [SourceDocument(**r) for r in rows]

    async def fetch_document_text(
        self, ctx: AccessContext, doc_id: str, *, max_chars: int
    ) -> DocumentText:
        """One document's visible chunks, in best-effort order, each individually
        ACL-checked and each capped at ``max_chars`` — PER CHUNK, not per document.

        The cap used to be spent twice: bound into ``left(c.text, $chars)`` AND
        re-used as a running total across chunks, so the first chunk exhausted the
        whole document. Measured on the live graphs, a chunk of the documents that
        carry per-item detail runs to ~6,100 chars against a 6,000 budget — so a
        "read this document" call returned ONE arbitrary chunk (order is
        ``chunk_id``; ingestion persists no ordinal) while the tool told the model
        it had read the document from start to finish. That is exactly the roster,
        table and ownership-matrix case the read exists for. ``enrich_chars_per_doc``
        has always documented itself as per-chunk.

        The TOTAL is still bounded — by its own, much larger knob
        (``document_total_chars``), not by re-spending the per-chunk value: the
        graph-chunk cap downstream counts CHUNKS, never characters, so without a
        ceiling here one read could hand the agent chunk_limit x chars_per_chunk
        (300k) of context. A read the ceiling cuts is a partial read and says so.

        Completeness travels WITH the text: a caller cannot tell a partial read
        from a whole one by looking at the chunks, and neither could the agent.
        """
        if not doc_id:
            return DocumentText()
        limit = self._s.document_chunk_limit
        rows = await self._rows(
            cypher.DOCUMENT_TEXT, ctx,
            doc_id=doc_id, chars=max_chars, limit=limit,
        )
        quotes: list[ChunkQuote] = []
        remaining = self._s.document_total_chars
        clipped = False
        for r in rows:
            q = _chunk_quote(r)
            if not q.text:
                continue
            if remaining <= 0:
                clipped = True
                break
            if len(q.text) > remaining:
                q.text = q.text[:remaining]
                clipped = True
            remaining -= len(q.text)
            quotes.append(q)
        # Per GRAPH: the row limit binds per source and _fan_out concatenates, so a
        # merged count at the cap says nothing about whether either read was cut.
        truncated = clipped or any(
            n >= limit for n in _rows_by_graph(rows).values())
        return DocumentText(chunks=quotes, truncated=truncated)

    # ── vocabulary ─────────────────────────────────────────────────────────

    async def relationship_types(self, ctx: AccessContext, limit: int = 60) -> list[dict]:
        """The controlled relationship vocabulary of THIS repository's graph
        sources, TTL-cached (it is effectively static between ingestion runs).

        Cached per repository instance, not per tenant: the vocabulary is read
        out of the databases in ``self._sources``, which are fixed when the
        instance is built — so a tenant key stored N identical copies, and a
        run-scoped view (``with_sources``) already gets its own empty cache."""
        cached = self._vocab_cache
        if cached and time.monotonic() - cached[0] < self._s.rel_vocab_ttl_s:
            return cached[1][:limit]
        rows = await self._rows(
            cypher.RELATIONSHIP_TYPES, ctx, limit=self._s.rel_vocab_fetch)
        self._vocab_cache = (time.monotonic(), _merged_vocabulary(rows))
        return self._vocab_cache[1][:limit]

    # ── people & authorship ─────────────────────────────────────────────────

    async def find_people(
        self, ctx: AccessContext, text: str, *, limit: int | None = None
    ) -> list[EntityHit]:
        """Resolve people by name/role from the people layer. Returned as
        ``EntityHit`` (type=Person, description = job title · department) so they
        flow through the existing evidence pool + synthesis unchanged."""
        lucene = _fuzzy_lucene(text)
        if not lucene:
            return []
        rows = await self._rows(
            cypher.PEOPLE_RESOLVE, ctx, lucene=lucene, limit=limit or self._s.resolve_limit
        )
        hits: list[EntityHit] = []
        for r in rows:
            role = " · ".join(x for x in (r.get("job_title"), r.get("department")) if x)
            # Only reachable on a HISTORICAL request — the query's employment gate
            # drops these rows otherwise — so the flag is what keeps a departed
            # person from reading as a current one in the answer they belong in.
            gone = is_inactive(r.get("employment_status"))
            if gone:
                role = (f"{role} · " if role else "") + "INACTIVE — no longer with the organization"
            hits.append(
                EntityHit(
                    name=r.get("full_name") or r["name"],
                    type="Person",
                    description=role or None,
                    degree=r.get("degree") or 0,
                    score=round(float(r.get("score") or 0.0), 4),
                    via=["people"],
                    graph=r.get("graph"),
                    status=INACTIVE if gone else r.get("employment_status"),
                )
            )
        return hits

    async def document_contributors(
        self, ctx: AccessContext, title: str, *, limit: int | None = None
    ) -> list[RelationFact]:
        """Who authored/edited/contributed to documents matching ``title``.
        Returned as ``RelationFact`` (person —role→ document) so authorship joins
        the same evidence pool as every other relationship.

        ONE ROW PER PERSON, ordered by how much of the matching material each wrote,
        so a hard LIMIT on a broad topic counts people and keeps the ones most
        involved instead of the front of the alphabet — see the query's own note."""
        title = (title or "").strip()
        if not title:
            return []
        rows = await self._rows(
            cypher.DOCUMENT_CONTRIBUTORS, ctx, title=title,
            limit=limit or self._s.neighbor_limit,
        )
        facts: list[RelationFact] = []
        for r in rows:
            role = " · ".join(x for x in (r.get("job_title"), r.get("department")) if x)
            facts.append(
                RelationFact(
                    origin="authorship",
                    source=r["person"],
                    relationship=r.get("role") or "AUTHORED",
                    target=r.get("document") or title,
                    target_type="Document",
                    # A schema-asserted edge, like every other typed read: the
                    # traversal emits the same edge with structural=True, so
                    # omitting it here made the merged row depend on which read
                    # landed first.
                    structural=True,
                    description=role or None,
                    source_status=r.get("status"),
                    # Authorship is a typed edge, so support is 1 by nature — nothing
                    # corroborates "X wrote Y" beyond the edge itself. Involvement goes
                    # in SCORE, which is where every other read puts its ranking signal
                    # (documents_by_author puts recency there for the same reason).
                    #
                    # Putting the contribution count in `support` instead — which this
                    # briefly did — inverts the digest's ordering rather than fixing it:
                    # the counts of the two people-reads are DIFFERENT UNITS (documents
                    # here, topic-adjacent entities there), so the read whose unit is
                    # more plentiful wins on every row regardless of evidential
                    # strength. Measured: 0 of the top 20 facts were the direct ones.
                    support=1,
                    score=self._involvement(r.get("contributions")),
                    graph=r.get("graph"),
                )
            )
        return _level_scores_per_graph(facts)

    def _involvement(self, contributions, hops: int = 1) -> float:
        """A contribution count as a ranking SCORE: the same curve the neighbour
        expansion applies to edge support (``support_weight * log10(1+n)``), divided
        by the PATH LENGTH the evidence was reached over.

        One curve for both people-reads, so a count means the same thing whichever
        produced it, and log compression keeps a person with many loosely-related
        contributions from burying one with a handful of directly-titled ones.

        Dividing by hops is what stops the log curve alone deciding it. The two reads
        count different UNITS — documents whose title names the topic (direct) against
        topic-adjacent things contributed to (two hops) — and the looser unit is far
        more plentiful, so on raw count the indirect read wins every row: measured on
        one topic, 0 of the top 20 facts were the direct ones. Path length is the
        honest correction, because it is exactly what differs between them and it is
        read off the traversal rather than chosen: a person whose own document is named
        for the topic IS closer evidence than one who worked on something adjacent to
        it. No new constant is introduced — ``hops`` is the query's own shape.
        """
        n = max(0, int(contributions or 0))
        return round(
            self._s.support_weight * math.log10(1.0 + n) / max(1, hops), 4
        )

    async def topic_contributors(
        self, ctx: AccessContext, topic: str, *, limit: int | None = None
    ) -> list[RelationFact]:
        """The people who work on ``topic`` — two hops, via what they wrote.

        A concept node barely links to people — its edges go mostly to other
        concepts — so one-hop expansion returns almost none of the people who
        work on it. They sit one
        hop further out, reachable through the pages and components they authored.

        Facts are person —role→ the thing they contributed that connects them to the
        topic, so the answer can state the link rather than assert the association.
        ``support`` carries the contribution count (how many topic-adjacent things
        this person wrote), which is what the digest ranks on — the same treatment
        authorship facts get, for the same reason: an unranked fact is the first one
        dropped when the digest cap binds.
        """
        topic = (topic or "").strip()
        if not topic:
            return []
        rows = await self._rows(
            cypher.TOPIC_CONTRIBUTORS, ctx, topic=topic,
            limit=limit or self._s.neighbor_limit,
        )
        facts: list[RelationFact] = []
        for r in rows:
            role = " · ".join(x for x in (r.get("job_title"), r.get("department")) if x)
            facts.append(
                RelationFact(
                    origin="authorship",
                    source=r["person"],
                    relationship=r.get("role") or "CONTRIBUTED_TO",
                    target=r.get("via") or topic,
                    target_type="Entity",
                    structural=True,
                    description=role or None,
                    # The PERSON's status, and the person is the SOURCE of this
                    # edge — the target is the page or component they contributed
                    # to. Stamped on `target_status` it flagged the wrong end, so
                    # the digest printed "[INACTIVE]" beside a topic.
                    source_status=r.get("status"),
                    # Same convention as the direct read above: a typed edge carries
                    # support 1, and involvement is a SCORE. Both reads now scale their
                    # own count through the same curve, so neither structurally
                    # outranks the other on unit size alone.
                    support=1,
                    # Two hops: person -> a topic-adjacent thing -> the topic.
                    score=self._involvement(r.get("contributions"), hops=2),
                    graph=r.get("graph"),
                )
            )
        return _level_scores_per_graph(facts)

    async def documents_by_author(
        self,
        ctx: AccessContext,
        who: str,
        *,
        since: str = "",
        until: str = "",
        limit: int | None = None,
    ) -> tuple[list[RelationFact], list[SourceDocument]]:
        """Which documents ``who`` authored/edited/contributed to — the mirror of
        ``document_contributors``, walked from the PERSON instead of the document
       .

        Returns ``(facts, sources)``, the same two-part shape
        ``fetch_people_profiles`` uses. The facts are person —role→ document, so
        authorship reaches the evidence pool exactly as it does from the other
        direction; each carries the document's ``published_at`` as its date, which
        is what makes an asked window enforceable rather than asserted.

        The SOURCES half is what makes the answer citable. Citations resolve only
        against ``evidence.sources`` (synthesis/citations.py), so a document known
        to this request ONLY as a fact cannot be cited at all: the model's marker
        finds no matching source and the chip is dropped. Measured on the
        generalization sweep — a substantive answer about two people's
        contributions carried ZERO citations, while a run where the agent happened
        to look each title up separately carried 14. The tool already has the
        doc_id, url, title and date, so it registers them itself instead of
        depending on the agent making a second call per document.

        ``since``/``until`` are ``YYYY-MM-DD`` bounds; empty string means unbounded,
        the sentinel ``RetrievalPlan.time_scope_from``/``to`` already use.
        """
        who = (who or "").strip()
        if not who:
            return [], []
        rows = await self._rows(
            cypher.DOCUMENTS_BY_AUTHOR, ctx, who=who,
            since=(since or "").strip(), until=(until or "").strip(),
            limit=limit or self._s.neighbor_limit,
        )
        facts: list[RelationFact] = []
        sources: list[SourceDocument] = []
        for r in rows:
            role = " · ".join(x for x in (r.get("job_title"), r.get("department")) if x)
            title = r.get("document") or "(untitled)"
            # Registered so the answer can CITE what it lists. mentions and
            # relevance stay at their defaults: this document was reached by
            # traversal, not by a chunk match, and inventing either signal would
            # push it up the trust ranking on evidence that does not exist.
            # add_sources merges by doc_id and keeps the max of each signal, so a
            # document the chunk lane also found does not lose its relevance.
            sources.append(
                SourceDocument(
                    doc_id=r.get("doc_id"),
                    title=title,
                    url=r.get("url"),
                    date=(r.get("date") or None),
                    source_type=r.get("source_type"),
                    snippet=r.get("snippet"),
                    graph=r.get("graph"),
                )
            )
            facts.append(
                RelationFact(
                    origin="authorship",
                    source=r["person"],
                    relationship=r.get("role") or "AUTHORED",
                    target=title,
                    target_type="Document",
                    structural=True,
                    description=role or None,
                    source_status=r.get("status"),
                    # Publication IS the authoring event being dated, so 'event'
                    # rather than 'documented' (which marks a claim recorded later
                    # than it became true).
                    date=(r.get("date") or None),
                    date_basis="event" if r.get("date") else None,
                    support=1,
                    # Authorship is a typed edge, so `support` is 1 by nature —
                    # nothing corroborates "X wrote Y" beyond the edge itself.
                    # The synthesis digest sorts on (support, score) and truncates
                    # at digest_max_facts, so leaving score at its 0.0 default put
                    # these BELOW every extracted fact of the same support and made
                    # them first to be dropped from a large pool — a question
                    # spanning many people could lose exactly the rows it asked
                    # for. Scored with the house recency function rather than an
                    # invented support value: newest work ranks first, which is
                    # what "what has X written" wants and what the query's own
                    # ORDER BY already does.
                    score=round(
                        self._s.trust_w_recency
                        * freshness(
                            r.get("date"), date.today(),
                            self._s.trust_recency_halflife_days,
                        ),
                        4,
                    ),
                    graph=r.get("graph"),
                )
            )
        return facts, sources

    async def documents_by_authors(
        self,
        ctx: AccessContext,
        who: list[str],
        *,
        since: str = "",
        until: str = "",
        limit: int | None = None,
    ) -> tuple[list[RelationFact], list[SourceDocument]]:
        """``documents_by_author`` for a SET of people — one concurrent batch
       .

        A question like "who reports to X and what have each of them published"
        needs the authorship read once per member of the set. Doing that one name
        at a time made the answer's completeness depend on how many calls the agent
        chose to emit, which is what varied run to run: the same question named
        three reports on one run and thirteen on the next. The set is the unit the
        question asks about, so it is the unit the read takes.

        Fanned out per name rather than expressed as one Cypher IN-list ON PURPOSE:
        the row LIMIT then applies per PERSON, so one prolific author cannot
        consume the whole budget and leave the rest of the set looking as though it
        had published nothing. Same shape as the claim top-up's parallel batch.

        Names are de-duplicated (case-insensitively, on the trimmed form) before
        the reads — the set often arrives with a name in it twice, from two
        directions — and blanks are dropped. Returns the merged
        ``(facts, sources)``; a name with nothing recorded simply contributes
        nothing.
        """
        seen: set[str] = set()
        names: list[str] = []
        for raw in who or []:
            name = (raw or "").strip()
            key = name.lower()
            if name and key not in seen:
                seen.add(key)
                names.append(name)
        if not names:
            return [], []
        batches = await asyncio.gather(
            *(
                self.documents_by_author(ctx, name, since=since, until=until, limit=limit)
                for name in names
            )
        )
        facts: list[RelationFact] = []
        sources: list[SourceDocument] = []
        for f, s in batches:
            facts.extend(f)
            sources.extend(s)
        return facts, sources


def _clone(value: Any) -> Any:
    """Copy a read result so a cached entry can't be corrupted by a caller that
    pops rows or mutates models. The repository only ever returns lists of pydantic
    models or plain dicts — or a tuple of such lists (fetch_people_profiles)."""
    if isinstance(value, tuple):
        return tuple(_clone(v) for v in value)
    if isinstance(value, list):
        return [
            v.model_copy(deep=True) if hasattr(v, "model_copy")
            else (dict(v) if isinstance(v, dict) else v)
            for v in value
        ]
    return value


class CachingGraphRepository:
    """Per-request read-through cache over ``GraphReadRepository``.

    Built once per request (so entries never cross requests/tenants) and injected
    as ``StrategyContext.repo``. Transparently memoizes each async read by
    (method, args) — the research agent repeating an expansion or lookup within a
    run is served from cache instead of re-querying every graph. Results are
    deep-copied out so a caller that mutates rows/models can't corrupt the entry;
    a read with an unhashable arg (e.g. an embedding vector) is passed straight
    through, never cached. Pure optimization: no behavior change, no heuristics.
    """

    def __init__(self, repo: GraphReadRepository) -> None:
        self._repo = repo
        self._cache: dict = {}

    def __getattr__(self, name: str):
        attr = getattr(self._repo, name)
        if not asyncio.iscoroutinefunction(attr):
            return attr  # non-read attribute/method — expose the real one

        async def cached(ctx: AccessContext, *args: Any, **kwargs: Any):
            key = (name, args, tuple(sorted(kwargs.items())))
            try:
                hash(key)
            except TypeError:
                return await attr(ctx, *args, **kwargs)  # unhashable arg → don't cache
            if key not in self._cache:
                self._cache[key] = await attr(ctx, *args, **kwargs)
            return _clone(self._cache[key])

        return cached
