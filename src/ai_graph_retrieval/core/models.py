"""Domain models: retrieval evidence, the UI graph trace, and usage accounting.

Two mutable, request-scoped accumulators live here:

- `EvidencePool`   — everything retrieval found (deduped); synthesis reads it and
  the final response ships it.
- `GraphTraceBuilder` — the visited subgraph + ordered step list for the UI.

Concurrency contract: accumulators are mutated **only from the orchestrating
task**. Concurrent tool executions *return* their deltas and the runtime merges
them sequentially — no locks, no races (see HLD §7).
"""
from __future__ import annotations

import re
import time
import uuid
from collections.abc import Collection
from enum import Enum

from pydantic import BaseModel, Field, model_validator

# ── Provenance vocabulary ───────────────────────────────────────────────────

def normalize_source(value: str | None) -> str | None:
    """The `source` label for a provenance token (a graph name). Lowercased;
    None when there is nothing to say.

    Deliberately NOT a whitelist: a connector added tomorrow should appear under
    its own name rather than be dropped for being unrecognised here.
    """
    return (value or "").strip().lower() or None


# The status string every surface renders a departed person with. Defined here
# rather than imported from retrieval/person_status.py, which owns the SEMANTICS
# and sits a layer above these models.
_INACTIVE = "Inactive"

# Names carrying a separator are the only ones matched inside free text — see
# EvidencePool.drop_people for why a bare single token is not.
_SEPARATED = re.compile(r"[\s._-]")


def _name_pattern(names: Collection[str]) -> re.Pattern | None:
    """Word-boundary alternation over the multi-token names, or None."""
    parts = sorted((re.escape(n) for n in names if _SEPARATED.search(n)), key=len,
                   reverse=True)
    return re.compile(r"(?<!\w)(?:" + "|".join(parts) + r")(?!\w)",
                      re.IGNORECASE) if parts else None


# ── Evidence models ─────────────────────────────────────────────────────────


class EntityHit(BaseModel):
    entity_id: str | None = None  # ingestion's stable identity (names can repeat)
    name: str
    type: str | None = None
    description: str | None = None
    # The entity's own page (a person's profile/space, a site's home) when the
    # source supplies one. Rendered into the digest so a "who is X" answer can LINK the
    # person's area instead of only saying "try a people search". None for
    # graph-resolved entities (Neo4j entities carry no URL today) and tiles.
    url: str | None = None
    # Structured attributes the SOURCE stores as separate properties, rendered
    # as one line (e.g. a person's "Senior AI Engineer · Data Science ·
    # Rajasthan, IN"). Distinct from `description`, which is blended prose that
    # ingestion fills only when some document described the entity — for most
    # people it is empty, which is why an enumerated roster used to render bare
    # names. Filled by the post-loop profile pass (retrieval/deep_research.py).
    profile: str | None = None
    degree: int = 0
    score: float = 0.0
    via: list[str] = Field(default_factory=list)
    graph: str | None = None  # source graph (multi-graph provenance), None = single-graph
    # Employment status for a Person entity, straight off the node: Active |
    # Inactive (None = not a person, or no status recorded — which is NOT a
    # claim that the person is active; see retrieval/person_status.py). Carried
    # so the flag survives on an entity row that has no profile line to carry it.
    status: str | None = None


class RelationFact(BaseModel):
    source: str
    target: str
    relationship: str
    specific_relationships: list[str] | None = None
    description: str | None = None
    direction: str | None = None  # outgoing | incoming (relative to `source`)
    target_type: str | None = None
    target_status: str | None = None  # e.g. person employment status: Active | Inactive
    # The SOURCE end's employment status, for the reads whose person is the
    # source of the edge (authorship, contribution). Without it a former
    # employee's flag rode on the wrong endpoint and the digest printed it
    # beside the document or topic they contributed to.
    source_status: str | None = None
    date: str | None = None
    date_basis: str | None = None  # event | documented
    support: int = 1
    # A STRUCTURAL edge — one the schema asserts (REPORTS_TO, MANAGES, MEMBER_OF,
    # WORKS_IN, AUTHORED, …) rather than one relation extraction inferred from
    # prose. Structural edges carry no `support` property, so the support curve
    # scores every one of them at its floor (support_weight * log10(2)) and they
    # tie with each other: ranking then falls to the neighbour-degree tiebreak,
    # which prefers high-degree concepts to people, and truncation removes the
    # org backbone first. They are not weak evidence — they are a different KIND
    # of evidence, so they rank on their own key rather than on a curve built to
    # count corroborating chunks.
    structural: bool = False
    score: float = 0.0
    graph: str | None = None  # source graph (multi-graph provenance)
    # PROVENANCE, for the answer's graph (synthesis/answer_graph.py). ``chunk_ids``
    # are the graph chunks an extracted edge was read from (the edge's own
    # ``chunk_ids`` property; empty for a structural edge, which the schema
    # asserts rather than a passage). ``origin`` names the read that put the
    # fact in the pool — "expand" (the researcher followed this entity's
    # connections), "paths" (a multi-hop walk), "authorship", "induced" (the
    # post-loop fill-in of edges among entities already pooled), "cited" (read
    # back from a cited document's own passages). Neither ranks anything: they
    # let the graph shown beside the answer say WHY an edge is there, and keep
    # a relationship nobody traversed and no cited passage states off it.
    chunk_ids: list[str] = Field(default_factory=list)
    origin: str | None = None
    # Hops from the traversal seed (1 = adjacent). Set only by the multi-hop path
    # read; a one-hop expansion leaves it None, which is what it has always meant.
    depth: int | None = None


class ClaimFact(BaseModel):
    subject: str
    predicate: str
    object: str
    statement_type: str | None = None  # FACT | OPINION | PREDICTION
    temporal_type: str | None = None  # ATEMPORAL | STATIC | DYNAMIC
    valid_from: str | None = None
    valid_to: str | None = None
    date_basis: str | None = None
    support: int = 1
    is_current: bool = True  # validity window contains today (expired/future ⇒ False)
    # chunk_ids of the (:Claim)-[:SUPPORTED_BY]->(:Chunk) provenance, when surfaced —
    # drives the claim→chunk trace edges (empty when the lane's query didn't return it).
    supporting_chunk_ids: list[str] = Field(default_factory=list)
    graph: str | None = None  # source graph (multi-graph provenance)


class SourceDocument(BaseModel):
    doc_id: str | None = None
    title: str
    url: str | None = None  # doc permalink for citations, when ingestion stamps d.url; else None
    date: str | None = None
    source_type: str | None = None
    mentions: int = 0
    # Query relevance in [0,1]: fused chunk cosine/BM25 carried from the chunk lane
    # (0.0 for docs reached only via graph traversal). Feeds trust ranking so an
    # on-topic match isn't buried by a merely well-connected doc (retrieval/quality.py).
    relevance: float = 0.0
    trust: float = 0.0  # feature-blended trust score (retrieval/quality.py)
    graph: str | None = None  # source graph (multi-graph provenance)
    # Short preview for the citation hover card. Taken from a verbatim
    # excerpt this request ALREADY retrieved, so a hover costs no extra call and no
    # extra read — the whole point of shipping it on the citation payload. None when
    # the document reached the answer through graph traversal alone and no excerpt
    # was pulled for it; a client should fall back to showing the title.
    snippet: str | None = None
    # Citation marker number: set ONLY on entries of the response's `citations`
    # list, where ref=n means this document is the [n] placeholder in the answer
    # text (the explicit form of the positional citations[n-1] ↔ [n] contract).
    # Always None on evidence.sources entries.
    ref: int | None = None
    # WHICH SYSTEM the document lives in — "wikipedia", "arxiv", … — one vocabulary
    # every client renders from (icon, label, filter), whatever mode produced the
    # citation. Distinct from `graph`, which names the store this service READ (a Neo4j
    # graph) and is provenance for us, not a product name for a reader.
    #
    # Filled automatically below from `graph`, because the graph lanes' documents
    # arrive as raw Cypher rows (`SourceDocument(**row)`) and a field they had to
    # remember to set would silently be absent for exactly the modes nobody
    # re-tests.
    # None means the provenance is genuinely unknown — clients should fall back to
    # whatever they show today, never to a guess.
    source: str | None = None

    @model_validator(mode="after")
    def _default_source_from_graph(self) -> SourceDocument:
        """Fill `source` from `graph` when the caller did not set it.

        `model_copy` does not re-validate, so the trust-ranking and citation
        passes (which copy to add `trust`/`ref`) carry the value through
        untouched rather than recomputing it.
        """
        if self.source is None:
            self.source = normalize_source(self.graph)
        return self


class SuggestedAction(BaseModel):
    """One context-aware follow-up the user is likely to want next.

    This service only PROPOSES actions — it never executes them. `action_id` is the
    correlation handle a client forwards to whatever executes it; `action_type` and
    `parameters` are the model's own description of the intent, deliberately
    free-form so new capabilities need no schema change here.
    """

    action_id: str
    label: str                       # short user-facing text ("Apply for leave")
    action_type: str | None = None   # model's intent slug, routed by the client
    parameters: dict[str, str] = Field(default_factory=dict)


class Contradiction(BaseModel):
    """Two or more claims that assert conflicting values for the same
    subject+predicate — surfaced, never silently resolved."""

    subject: str
    predicate: str
    objects: list[str]
    detail: str | None = None


class ChunkQuote(BaseModel):
    chunk_id: str
    doc_id: str | None = None
    doc_title: str | None = None
    text: str
    subject_status: str | None = None  # for people-profile docs: Active | Inactive
    # entity names the chunk MENTIONS / that reached it during traversal — drives the
    # chunk→entity trace edges (empty when the lane's query didn't return it).
    via: list[str] = Field(default_factory=list)
    graph: str | None = None  # source graph (multi-graph provenance)
    # Question-similarity for a GRAPH chunk, filled by the lane from the chunk's
    # stored vector (Qdrant holds one vector per chunk, keyed by chunk_id).
    # 0.0 means "not scored".
    relevance: float = 0.0


class DocumentText(BaseModel):
    """One document's body text, and whether that is ALL of it.

    A partial read used to be indistinguishable from a complete one at every
    layer — the chunks look the same, so the tool reported "read from start to
    finish" either way and synthesis answered from a fragment. The flag is the
    read's own honesty about its row limit, carried to the caller that renders it.
    """

    chunks: list[ChunkQuote] = Field(default_factory=list)
    truncated: bool = False


class CitedSubgraph(BaseModel):
    """What the graph itself records about the documents an answer cited: the
    entities their passages MENTION and the relationships extracted FROM those
    passages — read back after synthesis for the answer's graph
    (synthesis/answer_graph.py).

    ``entity_docs`` / ``fact_docs`` map an entity name / a fact key
    (``source|relationship|target``) to the cited ``doc_id``s it came from, so
    a node or edge can point at the citation number that grounds it. ``chunk_ids``
    are the cited documents' visible graph chunks — the set a pool fact's own
    ``chunk_ids`` is checked against to say "this relationship is stated in [n]".
    """

    entities: list[EntityHit] = Field(default_factory=list)
    facts: list[RelationFact] = Field(default_factory=list)
    entity_docs: dict[str, list[str]] = Field(default_factory=dict)
    fact_docs: dict[str, list[str]] = Field(default_factory=dict)
    chunk_ids: list[str] = Field(default_factory=list)
    chunk_docs: dict[str, str] = Field(default_factory=dict)
    # entity name → the cited chunks that MENTION it, and the cited documents'
    # passages themselves (id, document, a text preview) — so the answer's graph
    # can draw the passage a fact came from, not only the fact.
    entity_chunks: dict[str, list[str]] = Field(default_factory=dict)
    chunks: list[ChunkQuote] = Field(default_factory=list)


class Evidence(BaseModel):
    """Immutable snapshot of the pool, embedded in the final response."""

    entities: list[EntityHit] = Field(default_factory=list)
    facts: list[RelationFact] = Field(default_factory=list)
    claims: list[ClaimFact] = Field(default_factory=list)
    sources: list[SourceDocument] = Field(default_factory=list)
    quotes: list[ChunkQuote] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)


class EvidencePool:
    """Deduplicating accumulator for everything retrieval surfaces."""

    def __init__(self) -> None:
        self._entities: dict[str, EntityHit] = {}
        self._facts: dict[tuple, RelationFact] = {}
        self._claims: dict[tuple, ClaimFact] = {}
        self._sources: dict[str, SourceDocument] = {}
        self._quotes: dict[str, ChunkQuote] = {}

    def add_entities(self, items: list[EntityHit]) -> list[EntityHit]:
        """Dedup by stable identity when the row carries one, else by name,
        ENRICHING a duplicate's missing fields instead of
        dropping the newcomer wholesale: one read's copy of an entity may carry
        its page url but no description, another's the description but no url —
        the pool keeps the union, whichever arrived first. Only empty fields
        are filled; a value already present is never overwritten. Enriched
        entries are not returned as "added" (they are not new trace nodes)."""
        added = []
        for e in items:
            key = self._entity_key(e)
            existing = self._entities.get(key)
            if existing is None:
                self._entities[key] = e
                added.append(e)
                continue
            for field_name in ("url", "description", "profile", "type",
                               "entity_id", "status"):
                if getattr(existing, field_name) is None and getattr(e, field_name):
                    setattr(existing, field_name, getattr(e, field_name))
        return added

    @staticmethod
    def _entity_key(e: EntityHit) -> object:
        """``entity_id`` when ingestion supplied one — names repeat, and two
        different humans sharing a display name were merged into one row. Falls
        back to the name, which is all a row without an identity can assert."""
        return ("id", e.entity_id) if e.entity_id else e.name

    def add_facts(self, items: list[RelationFact]) -> list[RelationFact]:
        # Merge, don't first-wins-drop (same reason as add_sources): the same edge
        # reaches the pool from several reads, and each carries a different part of
        # the picture — the traversal knows it is `structural`, a people read knows
        # the person's status, an authorship read knows the date. Keeping whichever
        # arrived first made the FACTS set depend on tool execution ORDER, so an
        # identical query could rank the org backbone differently between runs.
        # Every field below resolves order-independently (max / first-non-empty).
        added = []
        for f in items:
            key = (f.source, f.relationship, f.target)
            prev = self._facts.get(key)
            if prev is None:
                self._facts[key] = f
                added.append(f)
                continue
            self._facts[key] = prev.model_copy(update={
                # A schema-asserted edge stays structural however it arrived: the
                # flag is a property of the EDGE, and only some reads emit it.
                "structural": prev.structural or f.structural,
                "support": max(prev.support, f.support),
                "score": max(prev.score, f.score),
                "description": prev.description or f.description,
                "target_type": prev.target_type or f.target_type,
                "direction": prev.direction or f.direction,
                "source_status": prev.source_status or f.source_status,
                "target_status": prev.target_status or f.target_status,
                "date": prev.date or f.date,
                "date_basis": prev.date_basis or f.date_basis,
                "graph": prev.graph or f.graph,
                "specific_relationships": (prev.specific_relationships
                                           or f.specific_relationships),
                # Provenance accumulates: every chunk any read saw the edge in,
                # and the FIRST read that found it (the one the researcher's
                # path actually went through; a later fill-in adds nothing).
                "chunk_ids": list(dict.fromkeys([*prev.chunk_ids, *f.chunk_ids])),
                "origin": prev.origin or f.origin,
                # Nearest-to-seed wins; None means "not set by this read".
                "depth": (f.depth if prev.depth is None
                          else min(prev.depth, f.depth) if f.depth is not None
                          else prev.depth),
            })
        return added

    def add_claims(self, items: list[ClaimFact]) -> None:
        for c in items:
            self._claims.setdefault((c.subject, c.predicate, c.object), c)

    def add_sources(self, items: list[SourceDocument]) -> None:
        # Merge, don't first-wins-drop: the same doc can arrive from both the chunk
        # lane (carries `relevance`, mentions 0) and graph traversal (carries
        # `mentions`, relevance 0). A plain setdefault kept whichever came first and
        # silently lost the other's signal — so keep the max of each and fill any
        # gaps, so trust ranking sees both relevance AND corroboration for a doc.
        for s in items:
            key = s.doc_id or s.title
            prev = self._sources.get(key)
            if prev is None:
                self._sources[key] = s
                continue
            self._sources[key] = prev.model_copy(update={
                "mentions": max(prev.mentions, s.mentions),
                "relevance": max(prev.relevance, s.relevance),
                "url": prev.url or s.url,
                "date": prev.date or s.date,
                "source_type": prev.source_type or s.source_type,
                "graph": prev.graph or s.graph,
                # The citation preview and the connector label are carried by
                # SOME reads and not others (an authorship edge knows the
                # document, a source-documents read knows its opening), so the
                # first copy to arrive routinely lacks what the second brings.
                # Dropping them here is what made the same document's citation
                # show a snippet on one run and none on the next.
                "snippet": prev.snippet or s.snippet,
                "source": prev.source or s.source,
            })

    def add_quotes(self, items: list[ChunkQuote]) -> None:
        # LONGER TEXT WINS, don't first-wins-drop: the same chunk_id arrives at
        # two widths — `quote_chars` (2500) from the supporting-chunk/traversal
        # reads, `enrich_chars_per_doc` (6000) from a full document read — and
        # first-wins kept whichever landed first. The agent's own documented
        # sequence (get_source_documents, then get_document_text on the doc_id it
        # returned) produces exactly the losing order, so a document the model was
        # told it had read "from start to finish" reached synthesis as a prefix.
        # Same rule the digest's near-duplicate dedup already applies.
        for q in items:
            prev = self._quotes.get(q.chunk_id)
            if prev is None or len(q.text or "") > len(prev.text or ""):
                # Keep a relevance already stamped on the copy being replaced:
                # scoring runs over the pool, and the richer copy arrives unscored.
                if prev is not None and not q.relevance:
                    q.relevance = prev.relevance
                self._quotes[q.chunk_id] = q

    def drop_people(self, names: Collection[str]) -> dict[str, int]:
        """Remove every trace of ``names`` from the pool. Returns what went, by kind.

        The counterpart of ``rename_entities`` and one operation for the same
        reason: a person is not one row. The same human is an entity, both ends
        of relationships, the subject and the object of claims, the title of
        their own people-object document, a mention list on a chunk and a name
        inside chunk prose — so removing the entity alone leaves the answer able
        to name them from five other places. The caller decides WHO (see
        retrieval/person_status.py); this decides what "removed" means.

        Free text is matched on a WORD BOUNDARY and only for a name carrying a
        separator ("Jane Doe", "jane.doe"). A bare single-token label is too
        ambiguous to censor prose on — dropping every excerpt containing it would
        cost far more evidence than the leak it closes. A structured row whose
        endpoint or title merely CONTAINS such a name goes too ("Jane Doe
        (Unlicensed)", "Meet Jane Doe"): the digest would print the name.

        The URL counts as such a value, on the entity's own page and on a source
        document alike. The digest renders both verbatim (" — url: …"), and a
        person's page is addressed by the handle they are named with
        ("/people/jane.doe"), so a people object whose TITLE happens not to be
        the bare name still put the name in the answer context through its link.
        Same rule as the title, for the same reason: a row addressed by the
        departed person is a row about them.

        Surviving rows are SCRUBBED where prose can smuggle the name back in:
        a profile segment ("… · reports to Jane Doe") is removed, a blended
        description that names them is cleared. The row itself stays — it is
        evidence about someone else; only the prose naming the departed goes.
        """
        keys = {n.strip().casefold() for n in names if n and n.strip()}
        if not keys:
            return {}
        pattern = _name_pattern(keys)

        def _named(text: str | None) -> bool:
            return bool(pattern and text and pattern.search(text))

        def _hit(*values: str | None) -> bool:
            return any((v or "").strip().casefold() in keys or _named(v)
                       for v in values)

        before = (len(self._entities), len(self._facts), len(self._claims),
                  len(self._sources), len(self._quotes))
        self._entities = {k: e for k, e in self._entities.items()
                          if not _hit(e.name, e.url)}
        self._facts = {k: f for k, f in self._facts.items()
                       if not _hit(f.source, f.target)}
        self._claims = {k: c for k, c in self._claims.items()
                        if not _hit(c.subject, c.object)}
        self._sources = {k: s for k, s in self._sources.items()
                         if not _hit(s.title, s.url)}
        self._quotes = {
            k: q for k, q in self._quotes.items()
            if not (any(v.casefold() in keys for v in q.via)
                    or _named(q.doc_title) or _named(q.text))
        }
        scrubbed = 0
        for e in self._entities.values():
            if _named(e.profile):
                # Profiles are " · "-joined segments by construction
                # (graph/repository._profile_line), so the offending segment —
                # "reports to <a departed manager>" is the live case — can be
                # removed without losing the person's own role and department.
                e.profile = " · ".join(
                    seg for seg in e.profile.split(" · ") if not _named(seg)
                ) or None
                scrubbed += 1
            if _named(e.description):
                e.description = None
                scrubbed += 1
        for f in self._facts.values():
            if _named(f.description):
                f.description = None
                scrubbed += 1
        after = (len(self._entities), len(self._facts), len(self._claims),
                 len(self._sources), len(self._quotes))
        kinds = ("entities", "facts", "claims", "sources", "quotes")
        removed = {k: b - a for k, b, a in zip(kinds, before, after, strict=True)
                   if b != a}
        if scrubbed:
            removed["prose_scrubbed"] = scrubbed
        return removed

    def flag_people(self, names: Collection[str]) -> int:
        """Mark ``names`` as no longer with the organization, in place; returns how
        many evidence items were stamped.

        The historical counterpart of ``drop_people``: the evidence stays — a
        question about the past is entitled to it — but every row the digest
        renders now carries the status, so the person cannot be read as a current
        member. Only the STRUCTURED rows are stamped; an excerpt's prose is
        verbatim source text and is never rewritten.
        """
        keys = {n.strip().casefold() for n in names if n and n.strip()}
        if not keys:
            return 0
        stamped = 0
        for e in self._entities.values():
            if (e.name or "").casefold() in keys and e.status is None:
                e.status = _INACTIVE
                stamped += 1
        for f in self._facts.values():
            if (f.source or "").casefold() in keys and f.source_status is None:
                f.source_status = _INACTIVE
                stamped += 1
            if (f.target or "").casefold() in keys and f.target_status is None:
                f.target_status = _INACTIVE
                stamped += 1
        return stamped

    def rename_entities(self, mapping: dict[str, str]) -> int:
        """Rewrite entity names and EVERY reference to them, in place. Returns how
        many distinct names were rewritten.

        Why it has to be one operation over the whole pool: the pool
        keys entities by NAME, so the same human arriving under two labels is two
        entities, and every fact, claim and chunk-mention references one label or
        the other. Renaming entities alone would leave facts pointing at a name no
        entity carries — worse than the duplication it set out to fix. So the
        rewrite covers entity keys, fact endpoints, claim subjects/objects and the
        mention lists the trace draws from, together.

        Collapsing is the POINT and it goes through the normal add_* paths, so two
        entities that become one keep the union of their fields (add_entities) and
        two facts that become identical keep one row (add_facts) — the same merge
        rules that would have applied had both arrived under one name to begin with.
        """
        pairs = {old: new for old, new in mapping.items() if old and new and old != new}
        if not pairs:
            return 0
        entities, self._entities = self._entities, {}
        for e in entities.values():
            new = pairs.get(e.name)
            self.add_entities([e if new is None else e.model_copy(update={"name": new})])
        facts, self._facts = self._facts, {}
        for f in facts.values():
            src, tgt = pairs.get(f.source, f.source), pairs.get(f.target, f.target)
            self.add_facts([f if (src, tgt) == (f.source, f.target)
                            else f.model_copy(update={"source": src, "target": tgt})])
        claims, self._claims = self._claims, {}
        for c in claims.values():
            subj, obj = pairs.get(c.subject, c.subject), pairs.get(c.object, c.object)
            self.add_claims([c if (subj, obj) == (c.subject, c.object)
                             else c.model_copy(update={"subject": subj, "object": obj})])
        for q in self._quotes.values():
            if q.via and any(v in pairs for v in q.via):
                q.via = [pairs.get(v, v) for v in q.via]
        return len(pairs)

    @property
    def quotes(self) -> list[ChunkQuote]:
        """The pooled chunks THEMSELVES, not copies — a new list holding the same
        objects, so a post-loop pass can stamp a score on them in place (the
        graph-chunk relevance pass does exactly this). ``snapshot()`` builds a
        fresh Evidence and is the wrong door for that.
        """
        return list(self._quotes.values())

    @property
    def entity_names(self) -> list[str]:
        # The NAMES, not the dict keys: an entity may be keyed by its stable
        # identity, and every caller of this wants what the evidence calls it.
        return list(dict.fromkeys(e.name for e in self._entities.values()))

    def size(self) -> int:
        """Total deduped item count across every kind — the signal the agent's
        stale-patience stop watches: a round that adds nothing leaves this unchanged."""
        return (
            len(self._entities) + len(self._facts) + len(self._claims)
            + len(self._sources) + len(self._quotes)
        )

    def snapshot(self) -> Evidence:
        return Evidence(
            entities=list(self._entities.values()),
            facts=list(self._facts.values()),
            claims=list(self._claims.values()),
            sources=list(self._sources.values()),
            quotes=list(self._quotes.values()),
        )


# ── Graph trace (what the UI renders) ───────────────────────────────────────


class StepKind(str, Enum):
    PLAN = "plan"
    SEMANTIC_SEARCH = "semantic_search"
    TRAVERSE = "traverse"
    SOURCES = "sources"
    TOOL_CALL = "tool_call"
    SYNTHESIZE = "synthesize"


class StepStatus(str, Enum):
    STARTED = "started"
    COMPLETED = "completed"
    ERROR = "error"


class TraceNode(BaseModel):
    id: str
    label: str
    type: str | None = None
    # OPTIONAL enrichment so the UI renders every node kind generically. All default
    # to None/absent, so the pre-existing entity-only trace is unchanged. `kind`
    # discriminates the render: query | document | chunk | claim | entity (people are
    # entities distinguished by `type`).
    kind: str | None = None
    url: str | None = None  # document permalink (kind=document)
    date: str | None = None  # document/claim date
    score: float | None = None  # relevance/trust score, when known
    graph: str | None = None  # source graph (e.g. wikipedia | arxiv) for colouring
    status: str | None = None  # e.g. chunk subject_status, claim currency


class TraceEdge(BaseModel):
    id: str
    source: str
    target: str
    label: str
    support: int = 1


class TraceStep(BaseModel):
    id: str
    label: str
    kind: StepKind
    status: StepStatus
    # One short sentence about THIS step, in the reader's language ("Following
    # “Jane Doe”'s connections to find who reports to her"). Additive and
    # optional: a client that renders only `label` is unaffected. See
    # core/narration.py for the vocabulary and the rule that a summary describes
    # work attempted, never a result we do not have yet.
    summary: str | None = None
    # Free-form extra shown beside the step. NEVER an internal identifier — tool
    # names live in the structured log line (`tool=`), not on the wire.
    detail: str | None = None
    duration_ms: int | None = None


class GraphTrace(BaseModel):
    nodes: list[TraceNode] = Field(default_factory=list)
    edges: list[TraceEdge] = Field(default_factory=list)
    steps: list[TraceStep] = Field(default_factory=list)


def _doc_node_id(doc_id: str | None, title: str | None) -> str:
    """Stable id for a document node — `doc:<doc_id>` when ingestion stamped one,
    else `doc:<title>` (matches the id the UI/citations target)."""
    return f"doc:{doc_id or title or ''}"


def _clip(text: str, limit: int = 80) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


class GraphTraceBuilder:
    """Accumulates the visited subgraph; reports *new* nodes/edges per merge so
    the caller can stream incremental `trace_delta` events.

    Entities keep their NAME as their node id (unchanged); every other kind uses a
    prefixed id (`doc:` / `chunk:` / `claim:`) so nodes dedupe across lanes/sources
    and citations can target them. Each merge_* mirrors the same dedup contract:
    return ONLY the nodes/edges newly seen, so the caller streams a minimal delta."""

    def __init__(self) -> None:
        self._nodes: dict[str, TraceNode] = {}
        self._edges: dict[str, TraceEdge] = {}
        self.steps: list[TraceStep] = []

    # ── internal dedup helpers ──────────────────────────────────────────────

    def _add_node(self, node: TraceNode, out: list[TraceNode]) -> None:
        if node.id not in self._nodes:
            self._nodes[node.id] = node
            out.append(node)

    def _add_edge(
        self, edge_id: str, source: str, target: str, label: str, support: int,
        out: list[TraceEdge],
    ) -> None:
        if edge_id not in self._edges:
            edge = TraceEdge(
                id=edge_id, source=source, target=target, label=label, support=support,
            )
            self._edges[edge_id] = edge
            out.append(edge)

    # ── entities + entity↔entity facts (pre-existing behaviour) ─────────────

    def merge_entities(self, entities: list[EntityHit]) -> list[TraceNode]:
        added: list[TraceNode] = []
        for e in entities:
            self._add_node(
                TraceNode(id=e.name, label=e.name, type=e.type, kind="entity",
                          score=e.score or None, graph=e.graph),
                added,
            )
        return added

    def merge_facts(self, facts: list[RelationFact]) -> tuple[list[TraceNode], list[TraceEdge]]:
        new_nodes: list[TraceNode] = []
        new_edges: list[TraceEdge] = []
        for f in facts:
            for name, ntype in ((f.source, None), (f.target, f.target_type)):
                self._add_node(
                    TraceNode(id=name, label=name, type=ntype, kind="entity", graph=f.graph),
                    new_nodes,
                )
            self._add_edge(
                f"{f.source}|{f.relationship}|{f.target}",
                f.source, f.target, f.relationship, f.support, new_edges,
            )
        return new_nodes, new_edges

    # ── documents / chunks / claims (typed graph) ───────────────────────────

    def merge_sources(self, sources: list[SourceDocument]) -> list[TraceNode]:
        """`document` nodes — the citable sources retrieval grounded on."""
        added: list[TraceNode] = []
        for s in sources:
            self._add_node(
                TraceNode(
                    id=_doc_node_id(s.doc_id, s.title), label=s.title or s.doc_id or "(untitled)",
                    kind="document", url=s.url, date=s.date, graph=s.graph,
                    score=s.trust or None,
                ),
                added,
            )
        return added

    def merge_quotes(self, quotes: list[ChunkQuote]) -> tuple[list[TraceNode], list[TraceEdge]]:
        """`chunk` nodes + `document→chunk` (HAS_CHUNK) edges where the doc is known
        + `chunk→entity` (MENTIONS) edges from the entities that reached the chunk."""
        new_nodes: list[TraceNode] = []
        new_edges: list[TraceEdge] = []
        for q in quotes:
            chunk_id = f"chunk:{q.chunk_id}"
            self._add_node(
                TraceNode(
                    id=chunk_id, label=_clip(q.doc_title or q.chunk_id, 60),
                    kind="chunk", graph=q.graph, status=q.subject_status,
                ),
                new_nodes,
            )
            if q.doc_id:
                doc_id = _doc_node_id(q.doc_id, q.doc_title)
                # Ensure the document endpoint exists before the edge — a chunk always
                # belongs to a document; without this the HAS_CHUNK edge could dangle
                # when the quote's doc wasn't among the (separately-merged) sources.
                self._add_node(
                    TraceNode(id=doc_id, label=_clip(q.doc_title or q.doc_id, 48),
                              kind="document", graph=q.graph),
                    new_nodes,
                )
                self._add_edge(
                    f"{doc_id}|HAS_CHUNK|{chunk_id}", doc_id, chunk_id, "HAS_CHUNK", 1, new_edges,
                )
            for name in q.via:
                # the mentioned entity is a real graph node the traversal touched;
                # surface it so the MENTIONS edge is grounded (kind=entity, deduped).
                self._add_node(
                    TraceNode(id=name, label=name, kind="entity", graph=q.graph), new_nodes,
                )
                self._add_edge(
                    f"{chunk_id}|MENTIONS|{name}", chunk_id, name, "MENTIONS", 1, new_edges,
                )
        return new_nodes, new_edges

    def merge_claims(self, claims: list[ClaimFact]) -> tuple[list[TraceNode], list[TraceEdge]]:
        """`claim` nodes + `entity→claim` (ASSERTS/ABOUT) edges linking the claim's
        subject/object entities + `claim→chunk` (SUPPORTED_BY) provenance edges."""
        new_nodes: list[TraceNode] = []
        new_edges: list[TraceEdge] = []
        for c in claims:
            claim_id = f"claim:{c.subject}|{c.predicate}|{c.object}"
            self._add_node(
                TraceNode(
                    id=claim_id, label=_clip(f"{c.subject} {c.predicate} {c.object}", 80),
                    kind="claim", graph=c.graph, date=c.valid_from,
                    status="current" if c.is_current else "historical",
                ),
                new_nodes,
            )
            # link the subject/object entities to the claim when we know them. The
            # subject is the entity the claim is about (create it if unseen); the
            # object is only linked to an entity that already exists (it is often a
            # literal value, never fabricated into a node).
            if c.subject:
                self._add_node(
                    TraceNode(id=c.subject, label=c.subject, kind="entity", graph=c.graph),
                    new_nodes,
                )
                self._add_edge(
                    f"{c.subject}|ASSERTS|{claim_id}", c.subject, claim_id, "ASSERTS", 1, new_edges,
                )
            if c.object and c.object in self._nodes:
                self._add_edge(
                    f"{c.object}|ABOUT|{claim_id}", c.object, claim_id, "ABOUT", 1, new_edges,
                )
            for cid in c.supporting_chunk_ids:
                chunk_id = f"chunk:{cid}"
                # Ensure the supporting-chunk endpoint exists before the edge — the claim
                # is grounded in that chunk; without this the SUPPORTED_BY edge could
                # dangle when the chunk wasn't among the (separately-merged) quotes.
                self._add_node(TraceNode(id=chunk_id, label=cid, kind="chunk"), new_nodes)
                self._add_edge(
                    f"{claim_id}|SUPPORTED_BY|{chunk_id}",
                    claim_id, chunk_id, "SUPPORTED_BY", 1, new_edges,
                )
        return new_nodes, new_edges

    def merge(
        self,
        *,
        entities: list[EntityHit] | None = None,
        facts: list[RelationFact] | None = None,
        sources: list[SourceDocument] | None = None,
        quotes: list[ChunkQuote] | None = None,
        claims: list[ClaimFact] | None = None,
    ) -> tuple[list[TraceNode], list[TraceEdge]]:
        """Single entry point to merge any mix of graph elements; returns the newly
        seen ``(nodes, edges)``. Owns the multi-type orchestration so callers (the
        progress reporter) stay decoupled from the per-kind ``merge_*`` methods."""
        new_nodes: list[TraceNode] = []
        new_edges: list[TraceEdge] = []
        new_nodes.extend(self.merge_entities(entities or []))
        # Sources BEFORE quotes: a document node carries richer data (url/date/trust)
        # than the placeholder merge_quotes would ensure, and node ids dedupe first-wins.
        new_nodes.extend(self.merge_sources(sources or []))
        for nodes, edges in (
            self.merge_facts(facts or []),
            self.merge_quotes(quotes or []),
            self.merge_claims(claims or []),
        ):
            new_nodes.extend(nodes)
            new_edges.extend(edges)
        return new_nodes, new_edges

    def snapshot(self) -> GraphTrace:
        return GraphTrace(
            nodes=list(self._nodes.values()),
            edges=list(self._edges.values()),
            steps=list(self.steps),
        )


def new_step(
    label: str,
    kind: StepKind,
    detail: str | None = None,
    summary: str | None = None,
) -> TraceStep:
    return TraceStep(
        id=uuid.uuid4().hex[:12],
        label=label,
        kind=kind,
        status=StepStatus.STARTED,
        summary=summary,
        detail=detail,
    )


# ── Usage accounting ────────────────────────────────────────────────────────


class DeploymentUsage(BaseModel):
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_prompt_tokens: int = 0  # subset of prompt_tokens served from the provider cache


class UsageReport(BaseModel):
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_prompt_tokens: int = 0  # subset of prompt_tokens billed at the cached (discounted) rate
    embedding_calls: int = 0
    embedding_tokens: int = 0
    by_deployment: dict[str, DeploymentUsage] = Field(default_factory=dict)
    cost_usd: float = 0.0  # filled by core.costing at response assembly


class Usage:
    """Mutable per-request usage counter (merged into the final response).

    Token counts are bucketed per deployment so cost can be computed with
    per-deployment rates (chat vs router vs embeddings).
    """

    def __init__(self) -> None:
        self._report = UsageReport()
        self._started = time.monotonic()

    def add_llm(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        deployment: str | None = None,
        cached_prompt_tokens: int = 0,
    ) -> None:
        # cached_prompt_tokens is a subset of prompt_tokens (never additive).
        cached = max(0, min(cached_prompt_tokens, prompt_tokens))
        self._report.llm_calls += 1
        self._report.prompt_tokens += prompt_tokens
        self._report.completion_tokens += completion_tokens
        self._report.cached_prompt_tokens += cached
        if deployment:
            bucket = self._report.by_deployment.setdefault(deployment, DeploymentUsage())
            bucket.calls += 1
            bucket.prompt_tokens += prompt_tokens
            bucket.completion_tokens += completion_tokens
            bucket.cached_prompt_tokens += cached

    def add_embedding(self, tokens: int = 0) -> None:
        self._report.embedding_calls += 1
        self._report.embedding_tokens += tokens

    def elapsed_ms(self) -> int:
        return int((time.monotonic() - self._started) * 1000)

    def snapshot(self) -> UsageReport:
        return self._report.model_copy(deep=True)
