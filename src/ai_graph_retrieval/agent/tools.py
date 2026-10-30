"""The deep-research agent's toolset — OpenAI Agents SDK `@function_tool`s.

Each tool is a module-level function taking a ``RunContextWrapper[ResearchContext]``;
per-request identity + accumulators are injected via that context (never captured
in a closure), so the same tool objects are reused across requests while a tool
still cannot reach data outside the request's own graph sources and principals.

Invariants preserved from the hand-rolled loop:
  * ACL capture — every read goes through ``rc.repo`` with ``rc.ctx`` (the request's
    AccessContext); a tool structurally cannot query another principal set.
  * Race-free merge — tools write findings straight into ``rc.evidence``; the
    EvidencePool add_* methods contain no ``await``, so concurrent (parallel)
    tool calls on the single-threaded event loop can never interleave inside one.
  * Synthesis is elsewhere — tools only gather; the final answer is composed by
    the synthesis pass from the ACL-filtered evidence pool, never by the agent.

There is deliberately no free-form-Cypher tool.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, replace
from typing import Any

import structlog
from agents import FunctionTool, RunContextWrapper, function_tool

from ai_graph_retrieval.clarification.models import ClarificationQuestion, build_question
from ai_graph_retrieval.connectors.llm.embeddings import RequestEmbedder
from ai_graph_retrieval.core.access import AccessContext
from ai_graph_retrieval.core.models import (
    ChunkQuote,
    EvidencePool,
    StepKind,
    Usage,
)
from ai_graph_retrieval.core.narration import (
    preview,
    quoted,
    reasoning,
    subject_list,
    with_subject,
)
from ai_graph_retrieval.core.progress import ProgressReporter
from ai_graph_retrieval.graph.repository import GraphReadRepository
from ai_graph_retrieval.observability.tracing import record_io, stage_span
from ai_graph_retrieval.retrieval.base import since_iso
from ai_graph_retrieval.retrieval.neighbor_scoring import NeighborScorer
from ai_graph_retrieval.retrieval.quality import rank_claims_by_recency
from ai_graph_retrieval.retrieval.similarity import cosine
from ai_graph_retrieval.settings import Settings
from ai_graph_retrieval.synthesis.dedup import interleave_by_document

logger = structlog.get_logger(__name__)


def _rows(rc: ResearchContext) -> int:
    """How many rows a list-returning tool hands back to the model.

    Profile-scoped: this was a hardcoded 12 in every tool, so the research lane —
    which exists to look WIDER — saw exactly as many people, contributors or chunks
    per call as a bounded follow-up. The row budget is a depth cap like every other.
    """
    return rc.settings.agent_tool_rows_research if rc.deep else rc.settings.agent_tool_rows


def _rows_per_subject(rc: ResearchContext, subjects: int) -> int:
    """The row budget for a tool called with SEVERAL subjects at once.

    The cap above is a per-call budget written for one-subject tools. Applied
    unchanged to a set-wide call it becomes a per-SET budget, so asking about
    thirteen people returns fewer rows each than asking about one — the answer's
    completeness would then depend on whether the agent batched the call or not,
    which is precisely the run-to-run variance batching exists to remove. Scaling
    by the number of subjects keeps the budget per-subject, matching the per-person
    row limit the repository already applies. One subject is the unchanged case.
    """
    return _rows(rc) * max(1, subjects)


def _top_facts(facts: list, rows: int, *, by_date: bool = False) -> list:
    """The `rows` strongest facts, ranked across every graph before the cut.

    The pool keeps everything these tools fetch; this bounds only what the MODEL
    is shown. That row window used to be a positional head-slice of the fan-out's
    concatenation (every row of the first graph, then the second), so the first
    graph could fill it entirely and the agent never saw the second graph's
    stronger rows — and what the agent cannot see it cannot follow up on.

    Ranked by the signal each read already ordered its own per-graph block by:
    ``score`` — the involvement curve, hop-normalised and therefore comparable
    across graphs (repository._involvement) — or the publication ``date`` for
    authorship, where the query's own ORDER BY is newest-first. Undated rows sort
    last; the subject name keeps two identical requests identically ordered.
    """
    key = (
        (lambda f: (f.date or "", f.source or ""))
        if by_date
        else (lambda f: (f.score, f.source or ""))
    )
    return sorted(facts, key=key, reverse=True)[:rows]


def _top_rows(rows: list[dict], count: int, *, by: str) -> list[dict]:
    """The `count` strongest rows, ranked across every graph before the cut.

    The row sibling of ``_top_facts``, for the reads that hand the model raw rows:
    the repository fans out and concatenates per-graph blocks, so slicing them
    positionally let the first graph fill the window and hid the second graph's
    stronger rows from the agent. ``by`` is the field that read's own query
    already ordered each block by, so this restores that order across graphs
    rather than imposing a new one. Missing values sort last; ``chunk_id`` keeps
    two identical requests identically ordered.
    """
    return sorted(
        rows,
        key=lambda r: (r.get(by) is not None, r.get(by), str(r.get("chunk_id") or "")),
        reverse=True,
    )[:count]


@dataclass
class ResearchContext:
    """Per-request state the agent's tools operate over (injected via Runner)."""

    repo: GraphReadRepository
    ctx: AccessContext
    embedder: RequestEmbedder
    settings: Settings
    evidence: EvidencePool
    progress: ProgressReporter
    usage: Usage
    scorer: NeighborScorer | None = None
    # True when this run is the full agent loop (Super Research's own lane)
    # rather than a bounded
    # follow-up, so list-returning tools can hand back more rows per call (see _rows).
    deep: bool = False
    # Whether graph CHUNK TEXT may be returned (profile.graph_text_tools).
    # False: the graph is context enrichment only — claims/entities/
    # relationships/metadata — so get_source_documents skips the supporting
    # excerpts and the text-returning tools are not offered (research_tools below).
    graph_text: bool = True
    # Upper bound on graph chunks this run may ADD, total across both graphs
    # (profile.graph_chunk_cap). Enforced by _keep_graph_quotes.
    graph_chunk_cap: int = 0
    # Set by the request_clarification tool when the agent decides it needs the
    # user to disambiguate; the runtime raises it after the loop (see runtime.py).
    clarification: list[ClarificationQuestion] | None = None
    # The asked window (plan.time_scope_from/to), so find_authored_documents can
    # bound published_at from the CLASSIFIER's resolution of "this year" rather
    # than asking the model to restate a date it would have to work out.
    time_scope: tuple[str, str] = ("", "")


_DESC_CAP = 300  # chars of description shown to the model per item


def _listing(found, rows: list[dict], *, key: str) -> dict:
    """A list result that states its own truncation.

    Every list-returning tool cuts what it shows the model to ``_rows(rc)`` while
    the evidence pool keeps everything, and only ``get_claims`` said so. The agent
    is instructed to COVER THE WHOLE SET rather than a sample, and had no way to
    tell a complete set from a clipped one — so it could not know to ask for more,
    and an incomplete answer shipped as a confident one. Same
    ``returned``/``total_found`` pair ``get_claims`` already returns.
    """
    return {key: rows, "returned": len(rows), "total_found": len(found),
            "truncated": len(found) > len(rows)}


def _clip(text: str | None) -> str | None:
    if text and len(text) > _DESC_CAP:
        return text[: _DESC_CAP - 1] + "…"
    return text


def _rows_to_quotes(rows: list[dict]) -> list[ChunkQuote]:
    return [
        ChunkQuote(
            chunk_id=r["chunk_id"], doc_id=r.get("doc_id"),
            doc_title=r.get("doc_title"), text=(r.get("text") or "").strip(),
            via=r.get("via_names") or [], graph=r.get("graph"),
        )
        for r in rows
        if (r.get("text") or "").strip()
    ]


def _norm_text(text: str | None) -> str:
    """Whitespace-collapsed, case-folded chunk identity for cross-source dedup
    (the same text can arrive under different chunk ids from different
    graphs or reads)."""
    return " ".join((text or "").split()).casefold()


def budget_graph_quotes(
    evidence, cap: int, quotes: list[ChunkQuote]
) -> list[ChunkQuote]:
    """The graph-chunk budget: dedup-then-cap, against the CURRENT pool.

    At most ``cap`` graph chunks TOTAL across both graphs for the whole request
    — not per graph, not per call site — and never a chunk whose text is
    already in the pool: the dedup runs BEFORE the count, so a duplicate never
    wastes a budget slot. Shared by the agent's tools (``_keep_graph_quotes``)
    and the deterministic post-loop top-up
    (retrieval/deep_research.enrich_graph_chunks) so both draw from the SAME
    budget. The cap bounds VOLUME only.

    MULTI-GRAPH FAIRNESS. The cap is shared, and a multi-graph read arrives as
    ``_fan_out``'s concatenation — every row of the first configured source, then
    every row of the second (graph/repository.py; the order comes from
    ``Settings.graph_sources``). Filling a shared cap from that list in order
    meant the first graph took as much as it had and the second got only the
    remainder, even on a question the second graph's documents answered best.

    Interleaving by graph gives each source an equal share of the budget to
    compete for — a fair START, not a quota. ``interleave_by_document`` keyed on
    the GRAPH is a pure reordering, so a graph with fewer chunks runs out and the
    other's tail follows; nothing is discarded. Single-graph reads are untouched.

    Ordering WITHIN each graph is preserved (the helper is stable), so a caller
    that already sorted by relevance — ``chunks_by_similarity`` sorts by
    cross-graph cosine — keeps its ranking intact."""
    existing = evidence.snapshot().quotes
    seen = {_norm_text(q.text) for q in existing}
    # How much of each chunk the pool already holds, by chunk_id. Text-only dedup
    # could not see these at all: the SAME chunk arrives at two widths — 2500
    # chars from the supporting-chunk/traversal reads, 6000 from a full document
    # read — so the two copies normalise to different strings and the gate scored
    # the re-read as a brand-new chunk, spending a second slot on text already
    # pooled. The agent's own documented sequence (get_source_documents, then
    # get_document_text on the doc_id it returned) produces exactly that order.
    pooled: dict[str, int] = {
        q.chunk_id: len(q.text or "") for q in existing if q.chunk_id
    }
    used = len(existing)
    remaining = max(0, cap - used)
    graphs = {q.graph for q in quotes if q.graph}
    if len(graphs) > 1 and remaining:
        quotes = interleave_by_document(quotes, doc_of=lambda q: q.graph)
    kept: list[ChunkQuote] = []
    spent = 0
    for q in quotes:
        held = pooled.get(q.chunk_id) if q.chunk_id else None
        if held is not None:
            # Already pooled. A LONGER copy still passes — free, since the chunk
            # is already counted in `used` — because that is the only way the
            # richer text reaches the pool (EvidencePool.add_quotes keeps the
            # longer of the two). Anything not longer adds nothing and is dropped.
            if len(q.text or "") > held:
                pooled[q.chunk_id] = len(q.text or "")
                kept.append(q)
            continue
        if spent >= remaining:
            continue
        key = _norm_text(q.text)
        if key and key not in seen:
            seen.add(key)
            if q.chunk_id:
                pooled[q.chunk_id] = len(q.text or "")
            kept.append(q)
            spent += 1
    return kept


def _keep_graph_quotes(rc: ResearchContext, quotes: list[ChunkQuote]) -> list[ChunkQuote]:
    """The budget above, enforced at the tool-side entry of graph text into the
    pool (product direction, 2026-08-08). Dropped-by-budget is silent to the
    model but not to the pool: what was kept is what synthesis sees."""
    return budget_graph_quotes(rc.evidence, rc.graph_chunk_cap, quotes)


@function_tool
async def resolve_entities(wrapper: RunContextWrapper[ResearchContext], query: str) -> str:
    """Resolve a phrase to the most relevant entities (typo-tolerant, semantic).

    ALWAYS the first step for any concept the question mentions. Each result has a
    `degree` (connection count) — prefer expanding higher-degree matches.

    Args:
        query: the concept/phrase to resolve.
    """
    rc = wrapper.context
    async with rc.progress.step(
        StepKind.TOOL_CALL,
        with_subject(
            "Identifying what {subject} refers to in the knowledge graph",
            query, "Identifying what this question refers to in the knowledge graph",
        ),
        tool="resolve_entities"
    ) as step:
        embedding = await rc.embedder.get(query)
        hits = await rc.repo.resolve_entities(rc.ctx, query, embedding)
        step.detail = reasoning(
            f"Matched {quoted(query)} against known people, teams and topics",
            f"{len(hits)} candidates found" if hits else "nothing matched",
            f"strongest: {preview([h.name for h in hits])}" if hits else "",
        )
    new = rc.evidence.add_entities(hits)
    await rc.progress.graph_delta(entities=new)
    return json.dumps(_listing(hits, [
        {"name": h.name, "type": h.type, "description": _clip(h.description),
         "degree": h.degree, "score": h.score}
        for h in hits
    ], key="entities"))


@function_tool
async def expand_entity(
    wrapper: RunContextWrapper[ResearchContext],
    entity_name: str,
    focus: str,
    relationship_types: list[str] | None = None,
) -> str:
    """Get an entity's most relevant relationships, ranked against `focus`.

    Args:
        entity_name: exact entity name (from resolve_entities).
        focus: the sub-goal to rank relationships by.
        relationship_types: optional exact types from list_relationship_types
            (do not guess type names).
    """
    rc = wrapper.context
    types = relationship_types or None
    async with rc.progress.step(
        StepKind.TOOL_CALL,
        with_subject(
            "Following {subject}'s connections to see what relates to it",
            entity_name, "Following its connections to see what relates to it",
        ),
        tool="expand_entity"
    ) as step:
        if rc.scorer is not None:
            # Over-fetch the cheap support-ranked pool; the scorer's gates decide
            # whether an LLM re-rank against `focus` pays for itself on this hub.
            candidates = await rc.repo.expand_entity(
                rc.ctx, entity_name, relationship_types=types,
                limit=rc.settings.scoring_candidate_cap,
            )
            facts, _tier = await rc.scorer.rescore(
                candidates, focus or entity_name, keep=rc.settings.neighbor_limit
            )
            # The cap bound if the ranker had more than it was allowed to keep.
            truncated = len(candidates) > len(facts)
        else:
            facts = await rc.repo.expand_entity(
                rc.ctx, entity_name, relationship_types=types
            )
            # A full page back from a $limit-bound read is the read hitting it.
            truncated = len(facts) >= rc.settings.neighbor_limit
        step.detail = reasoning(
            f"Followed {quoted(entity_name)}'s connections in the knowledge graph",
            f"{len(facts)} relationships found" if facts else "no connections recorded",
            (f"kept those most relevant to {quoted(focus)}" if focus and facts else ""),
        )
    new = rc.evidence.add_facts(facts)
    await rc.progress.graph_delta(facts=new)
    return json.dumps({
        "entity": entity_name,
        "returned": len(facts),
        # An org chart cut at the cap is a PARTIAL neighbourhood: without this the
        # agent read a truncated fan-out as the entity's complete set of
        # relationships and reported a short roster as whole.
        "truncated": truncated,
        "relationships": [
            {"neighbor": f.target, "neighbor_type": f.target_type,
             "relationship": f.relationship, "specific": f.specific_relationships,
             "direction": f.direction, "date": f.date, "date_basis": f.date_basis,
             "support": f.support, "description": _clip(f.description)}
            for f in facts
        ],
    })


def _claim_text(c) -> str:
    return f"{c.subject} {c.predicate} {c.object or ''}".strip()


async def _rank_claims(rc: ResearchContext, claims: list, focus: str | None) -> list:
    """Order claims for the model: by embedding relevance to ``focus`` when the
    agent stated one, by date (newest first) otherwise.

    Both orderings fix a real defect of the raw fetch: the repository fans out
    over the graphs and CONCATENATES per-graph result lists, so the raw order
    is not even globally recency-sorted. Relevance is the embedder's job (rule
    8) — one batched embedding call over the claim texts, no LLM. Fail-open:
    any embedding failure falls back to the date ordering, because a worse
    order is still real claims while an error here would cost the whole call.
    """
    by_date = rank_claims_by_recency(claims)
    if not focus or len(claims) < 2:
        return by_date
    try:
        query_vec = await rc.embedder.get(focus)
        vec_by_text = await rc.embedder.embed_batch(
            list({_claim_text(c) for c in claims})
        )
        return sorted(
            by_date,
            key=lambda c: cosine(query_vec, vec_by_text.get(_claim_text(c))),
            reverse=True,
        )
    except Exception as exc:
        # Fail-open is right (a worse order still beats no claims), but silent
        # fail-open is not: a persistently broken embedder degrades every
        # `get_claims` result to date-order with nothing anywhere saying so,
        # and the symptom — "the agent cited the wrong claim" — looks like a
        # model problem rather than an embedding outage.
        logger.warning(
            "claim_relevance_sort_unavailable",
            error=str(exc),
            error_type=type(exc).__name__,
            claims=len(claims),
        )
        return by_date


@function_tool
async def get_claims(
    wrapper: RunContextWrapper[ResearchContext],
    entity_name: str,
    focus: str | None = None,
    months: int | None = None,
) -> str:
    """Dated, atomic claims about an entity (subject-predicate-object with validity
    dates and FACT/OPINION/PREDICTION typing). Use for grounding and precise 'when'.
    Returns the most relevant claims first when you give a `focus`, newest first
    otherwise.

    Args:
        entity_name: exact entity name.
        focus: optional sub-goal to rank the claims by — state one whenever you
            are looking for something specific about the entity.
        months: optional look-back window in months.
    """
    rc = wrapper.context
    async with rc.progress.step(
        StepKind.TOOL_CALL,
        with_subject(
            "Checking what has been recorded about {subject}, and when",
            entity_name, "Checking what has been recorded about this, and when",
        ),
        tool="get_claims"
    ) as step:
        claims = await rc.repo.fetch_claims(
            rc.ctx, entity_name, since_iso(int(months)) if months else None
        )
        ranked = await _rank_claims(rc, claims, focus)
        step.detail = reasoning(
            f"Collected {len(claims)} dated facts recorded about {quoted(entity_name)}"
            if claims else f"nothing dated is recorded about {quoted(entity_name)}",
            (f"ordered by relevance to {quoted(focus)}" if focus and len(claims) > 1
             else "newest first" if claims else ""),
        )
    # The pool keeps EVERYTHING fetched (synthesis applies its own digest caps);
    # the model sees the top of the ranked order under the row budget — the same
    # bounded-return contract as every other list tool, where this tool used to
    # hand back every row (up to 60 per graph) and dominate the round's tokens.
    rc.evidence.add_claims(claims)
    await rc.progress.graph_delta(claims=claims)
    return json.dumps({
        "returned": min(len(ranked), _rows(rc)),
        "total_found": len(ranked),
        "ordered_by": "relevance to focus" if focus and len(claims) > 1 else "newest first",
        "claims": [c.model_dump(exclude_none=True) for c in ranked[: _rows(rc)]],
    })


@function_tool
async def find_recent_changes(
    wrapper: RunContextWrapper[ResearchContext], topic: str, months: int | None = None
) -> str:
    """Dated facts added/changed in a time window, newest first.
    FIRST call for any time-bounded question.

    Args:
        topic: what to look for changes about (context for your own filtering —
            the window itself returns all dated changes you can see).
        months: look-back window in months (default 6).
    """
    rc = wrapper.context
    window = int(months) if months else rc.settings.default_temporal_months
    async with rc.progress.step(
        StepKind.TOOL_CALL,
        with_subject(
            "Looking for what has changed recently around {subject}",
            topic, "Looking for what has changed recently",
        ),
        tool="find_recent_changes"
    ) as step:
        claims = await rc.repo.recent_changes(rc.ctx, since_iso(window))
        step.detail = reasoning(
            f"Scanned the last {window} months of recorded changes",
            f"{len(claims)} dated updates found, newest first" if claims
            else "nothing changed in that window",
        )
    # Claims, not facts: a dated change is a statement ABOUT an entity whose
    # object is a value. Pooled as facts, every date became a node in the
    # evidence graph (see GraphReadRepository.recent_changes).
    rc.evidence.add_claims(claims)
    await rc.progress.graph_delta(claims=claims)
    return json.dumps({
        "since_months": window, "count": len(claims),
        "changes": [
            {"subject": c.subject, "predicate": c.predicate, "object": c.object,
             "date": c.valid_from, "date_basis": c.date_basis,
             "support": c.support, "graph": c.graph}
            for c in claims
        ],
    })


@function_tool
async def list_relationship_types(
    wrapper: RunContextWrapper[ResearchContext], limit: int | None = None
) -> str:
    """The closed relationship-type vocabulary in this workspace, with counts.
    Call before using relationship_types in expand_entity.

    Args:
        limit: max types to return (default 60).
    """
    rc = wrapper.context
    async with rc.progress.step(
        StepKind.TOOL_CALL,
        "Checking what kinds of connections exist here before following any",
        tool="list_relationship_types",
    ) as step:
        vocab = await rc.repo.relationship_types(rc.ctx, limit=int(limit) if limit else 60)
        step.detail = reasoning(
            f"Listed the {len(vocab)} kinds of connections this workspace records",
            "so only connections that really exist get followed",
        )
    return json.dumps(vocab)


@function_tool
async def get_source_documents(
    wrapper: RunContextWrapper[ResearchContext], entity_names: list[str]
) -> str:
    """The source documents behind a set of entities, for citations. Call once near
    the END with the main entities your findings rest on.

    Args:
        entity_names: the entities whose sources to gather.
    """
    rc = wrapper.context
    names = [str(n) for n in (entity_names or []) if n][: _rows(rc)]
    async with rc.progress.step(
        StepKind.SOURCES,
        "Pulling the documents behind these findings, and the passages that back them",
        tool="get_source_documents"
    ) as step:
        sources = await rc.repo.source_documents(rc.ctx, names)
        # Supporting excerpts are graph CHUNK TEXT — skipped when the profile
        # declares the graph enrichment-only: citations then rest on the
        # document metadata.
        quotes = (
            _keep_graph_quotes(rc, await rc.repo.supporting_chunks(rc.ctx, names))
            if rc.graph_text else []
        )
        step.detail = reasoning(
            f"Gathered {len(sources)} documents behind {preview(names)}"
            if sources else f"no documents recorded behind {preview(names)}",
            "each checked against your access permissions" if sources else "",
            "with the passages that back them" if quotes else "",
        )
    rc.evidence.add_sources(sources)
    rc.evidence.add_quotes(quotes)
    await rc.progress.graph_delta(sources=sources, quotes=quotes)
    return json.dumps([s.model_dump(exclude_none=True) for s in sources])


@function_tool
async def find_documents(
    wrapper: RunContextWrapper[ResearchContext], titles: list[str]
) -> str:
    """Find documents by (partial) TITLE when the user names specific documents
    ('the 26.04 release notes', 'the travel policy'). Returns doc_ids — read the
    content with get_document_text.

    PASS EVERY TITLE YOU NEED IN ONE CALL. When a page you have read names a series
    — releases, periods, versions, stages — each member has its own document, and
    looking up two of twelve spends a round per member while the rest stay a single
    line from whatever index or calendar listed them. Each title gets its own
    result budget, so one crowded title cannot crowd out the others.

    TITLE only. A PERSON's name here returns documents TITLED after them — their
    profile page, their welcome post — not the documents they wrote. For what a
    person or a group has written or published, use find_authored_documents, which
    walks authorship directly and takes the whole group in one call.

    Args:
        titles: titles, or distinctive parts of them — as many as the answer needs.
    """
    rc = wrapper.context
    wanted = [str(t).strip() for t in (titles or []) if t and str(t).strip()]
    which = subject_list(wanted)
    async with rc.progress.step(
        StepKind.SOURCES,
        f"Trying to locate the documents {which}" if which
        else "Trying to locate the documents referred to",
        tool="find_documents"
    ) as step:
        docs = await rc.repo.document_lookup(rc.ctx, wanted)
        step.detail = reasoning(
            f"Searched document titles for {preview(wanted)}",
            f"{len(docs)} candidates found" if docs else "no matching titles",
            f"closest: {preview([d.title for d in docs])}" if docs else "",
        )
    rc.evidence.add_sources(docs)
    await rc.progress.graph_delta(sources=docs)
    return json.dumps([
        d.model_dump(exclude_none=True, exclude={"mentions", "trust"}) for d in docs
    ])


@function_tool
async def get_document_text(
    wrapper: RunContextWrapper[ResearchContext], doc_ids: list[str]
) -> str:
    """Read DOCUMENTS' body text by doc_id (from get_source_documents/find_documents).
    Use when the answer likely lives in a table, roster, list, or exact wording that
    relationship/claim evidence cannot carry.

    Pass EVERY document you need read in ONE call. When the answer enumerates a series
    — releases, periods, versions, stages — each member's detail lives in its own
    document, and reading two of twelve produces an answer where two entries are
    developed and the rest are a single line from whatever index or calendar listed
    them. One call per document spends a round per member; one call spends one.

    Args:
        doc_ids: exact doc_ids, as many as the answer needs.
    """
    rc = wrapper.context
    ids = [str(d).strip() for d in (doc_ids or []) if d and str(d).strip()]
    # De-duplicated, order preserved: a set assembled from several reads routinely
    # carries the same doc_id twice, and a repeated read would spend graph-chunk
    # budget on text already pooled.
    seen: set[str] = set()
    ids = [i for i in ids if not (i in seen or seen.add(i))]
    if not ids:
        return json.dumps({"error": "no doc_ids given"})
    async with rc.progress.step(
        StepKind.SOURCES,
        "Reading the documents in full — the detail may sit in a table or list",
        tool="get_document_text"
    ) as step:
        # Concurrent, like every other set-wide read: the budget is enforced once on
        # the merged result, so the per-request graph-chunk cap still holds exactly.
        batches = await asyncio.gather(*(
            rc.repo.fetch_document_text(
                rc.ctx, i, max_chars=rc.settings.enrich_chars_per_doc)
            for i in ids
        ))
        fetched = [q for batch in batches for q in batch.chunks]
        # What the pool will hold once this call lands: the chunks it already had
        # plus the ones the graph-chunk budget admitted.
        readable = {q.chunk_id for q in rc.evidence.snapshot().quotes if q.chunk_id}
        quotes = _keep_graph_quotes(rc, fetched)
        readable |= {q.chunk_id for q in quotes if q.chunk_id}
        # Which of the requested documents came back whole. Reported, not assumed:
        # the step text used to claim "from start to finish" on every read. Judged
        # AFTER the budget, on two grounds — the row limit may have cut the read
        # short, and a chunk the graph-chunk cap refused is as absent from the
        # answer as one the query never returned.
        partial = {
            i for i, batch in zip(ids, batches, strict=True)
            if batch.truncated
            or any(q.chunk_id not in readable for q in batch.chunks)
        }
        titles = list(dict.fromkeys(q.doc_title for q in quotes if q.doc_title))
        step.detail = reasoning(
            f"Read {len(titles)} of {len(ids)} documents"
            + ("" if partial else " from start to finish")
            + f": {preview(titles)}" if quotes
            else "those documents added nothing new",
            "so tables, rosters and lists aren't missed" if quotes else "",
        )
    if not quotes:
        return json.dumps({"error": "documents not found, not accessible, or their "
                                    "text is already in evidence"})
    rc.evidence.add_quotes(quotes)
    await rc.progress.graph_delta(quotes=quotes)
    by_doc: dict[str, list] = {}
    for q in quotes:
        by_doc.setdefault(q.doc_id or q.doc_title or "?", []).append(q)
    return json.dumps({
        "documents": [
            {"doc_id": k, "title": v[0].doc_title,
             # Whether this IS the document or a part of it. A partial read used
             # to be indistinguishable from a complete one, so an answer drawn
             # from a fragment of a roster read as an answer drawn from the roster.
             "complete": k not in partial,
             "chunks_read": len(v),
             "text": "\n\n".join(q.text for q in v)}
            for k, v in by_doc.items()
        ],
    })


@function_tool
async def traverse_to_chunks(
    wrapper: RunContextWrapper[ResearchContext],
    seed_entities: list[str],
    hops: int | None = None,
) -> str:
    """Multi-hop from seed entities to the actual chunks that discuss them or
    their connected neighbours — reaches evidence 1-2 hops away that expanding a
    single entity would miss. Use when the answer likely sits in content linked
    THROUGH an entity rather than stated about it directly.

    Args:
        seed_entities: exact entity names to start from.
        hops: 1-2 (default 2).
    """
    rc = wrapper.context
    seeds = [str(n) for n in (seed_entities or []) if n][: _rows(rc)]
    if not seeds:
        return json.dumps({"error": "no seed_entities given"})
    async with rc.progress.step(
        StepKind.TRAVERSE,
        "Following the connections outward to reach material a direct search would miss",
        tool="traverse_to_chunks"
    ) as step:
        rows = await rc.repo.beam_traverse_chunks(rc.ctx, seeds, hops=int(hops) if hops else None)
        quotes = _keep_graph_quotes(rc, _rows_to_quotes(rows))
        step.detail = reasoning(
            f"Followed connections outward from {preview(seeds)}",
            f"reached {len(quotes)} related passages a direct search would miss"
            if quotes else "nothing new within reach",
        )
    rc.evidence.add_quotes(quotes)
    await rc.progress.graph_delta(quotes=quotes)
    return json.dumps({
        "seeds": seeds, "reached_chunks": len(quotes),
        **_listing(rows, [
            {"doc_title": r.get("doc_title"), "via": r.get("via_names"),
             "text": _clip((r.get("text") or "").strip())}
            for r in _top_rows(rows, _rows(rc), by="via_count")
        ], key="excerpts"),
    })


@function_tool
async def find_entity_paths(
    wrapper: RunContextWrapper[ResearchContext],
    seed_entities: list[str],
    target_entity: str | None = None,
    hops: int | None = None,
    relationship_types: list[str] | None = None,
) -> str:
    """Follow relationships several hops out and get back the CONNECTIONS themselves.

    Use for questions about structure — how two things are connected, what sits
    between them, what a thing reaches indirectly. expand_entity sees one hop;
    this walks further and returns each relationship along the way, so the chain
    can be stated rather than guessed.

    Args:
        seed_entities: exact entity names to start from.
        target_entity: only paths reaching this entity, for "how does A relate to B".
        hops: 1-4 (default 2).
        relationship_types: optional exact types from list_relationship_types.
    """
    rc = wrapper.context
    seeds = [str(n) for n in (seed_entities or []) if n][: _rows(rc)]
    if not seeds:
        return json.dumps({"error": "no seed_entities given"})
    async with rc.progress.step(
        StepKind.TRAVERSE,
        with_subject(
            "Tracing how {subject} connects to the rest of the graph",
            target_entity or (seeds[0] if len(seeds) == 1 else ""),
            "Tracing how these connect to the rest of the graph",
        ),
        tool="find_entity_paths",
    ) as step:
        facts = await rc.repo.entity_paths(
            rc.ctx, seeds, target=target_entity,
            relationship_types=relationship_types or None,
            hops=int(hops) if hops else None,
        )
        reached = len({f.target for f in facts})
        step.detail = reasoning(
            f"Traced connections outward from {preview(seeds)}"
            + (f" to {quoted(target_entity)}" if target_entity else ""),
            f"{len(facts)} relationships across {reached} entities"
            if facts else "no connecting path recorded",
        )
    new = rc.evidence.add_facts(facts)
    await rc.progress.graph_delta(facts=new)
    return json.dumps({
        "seeds": seeds, "target": target_entity,
        # `returned` used to count `facts` while the list below renders
        # `facts[:_rows(rc)]`, so the tool overstated its own result.
        "returned": min(len(facts), _rows(rc)), "total_found": len(facts),
        "truncated": len(facts) > _rows(rc),
        "relationships": [
            {"source": f.source, "relationship": f.relationship, "target": f.target,
             "target_type": f.target_type, "direction": f.direction,
             "depth": f.depth, "support": f.support,
             "description": _clip(f.description)}
            # NOT _top_facts: these rows are a PATH, already ordered
            # (depth, score) by the query. Re-ranking on score alone would show a
            # distant hop before the adjacent one it hangs off, and a chain read
            # out of order is not a chain.
            for f in facts[: _rows(rc)]
        ],
    })


@function_tool
async def find_relevant_chunks(
    wrapper: RunContextWrapper[ResearchContext], query: str
) -> str:
    """Find passages semantically similar to a question or topic — NO entity needed.

    Every other chunk tool starts from a named or resolved entity, so a question
    whose subject nothing resolves reaches no graph text at all. This one searches
    the chunk text itself, so use it when: entity resolution came back empty, the
    question is thematic rather than about a nameable thing, or you want to check
    whether anything else in the corpus bears on the question.

    Searches every configured graph on one comparable relevance scale, so the
    best passage wins wherever it lives.

    Args:
        query: the question or topic to find passages about — a full question
            works better here than a bare name, since the whole phrasing is
            matched against the text.
    """
    rc = wrapper.context
    query = (query or "").strip()
    if not query:
        return json.dumps({"error": "no query given"})
    async with rc.progress.step(
        StepKind.SEMANTIC_SEARCH,
        "Looking for passages that speak to the question, wherever they sit",
        tool="find_relevant_chunks",
    ) as step:
        # The request-scoped embedder memoises by text, so a repeat of the same
        # sub-goal costs nothing; None (embedding unavailable) degrades to no
        # results rather than failing the round.
        quotes = _keep_graph_quotes(rc, await rc.repo.chunks_by_similarity(
            rc.ctx, await rc.embedder.get(query), limit=_rows(rc),
        ))
        titles = list(dict.fromkeys(q.doc_title for q in quotes if q.doc_title))
        step.detail = reasoning(
            f"Found {len(quotes)} passages about “{preview([query])}”: {preview(titles)}"
            if quotes else "nothing in the material speaks to that directly",
        )
    if not quotes:
        return json.dumps({"chunks": []})
    rc.evidence.add_quotes(quotes)
    await rc.progress.graph_delta(quotes=quotes)
    return json.dumps({
        "chunks": [
            {"doc_title": q.doc_title, "text": _clip(q.text)} for q in quotes
        ],
    })


@function_tool
async def recent_change_excerpts(
    wrapper: RunContextWrapper[ResearchContext],
    topic: str | None = None,
    months: int | None = None,
) -> str:
    """Verbatim chunks behind dated changes in a time window, optionally filtered
    by topic — the exact wording of what changed (unlike find_recent_changes, which
    returns relationship summaries).

    Args:
        topic: optional topic filter.
        months: look-back window in months (default 6).
    """
    rc = wrapper.context
    window = int(months) if months else rc.settings.default_temporal_months
    async with rc.progress.step(
        StepKind.TRAVERSE,
        with_subject(
            "Reading what recently changed around {subject}",
            topic, "Reading what recently changed",
        ),
        tool="recent_change_excerpts",
    ) as step:
        rows = await rc.repo.recent_change_chunks(rc.ctx, since_iso(window), topic=topic or None)
        quotes = _keep_graph_quotes(rc, _rows_to_quotes(rows))
        step.detail = reasoning(
            f"Looked at the last {window} months of changes",
            f"pulled the exact wording behind {len(quotes)} of them"
            if quotes else "nothing new behind them",
        )
    rc.evidence.add_quotes(quotes)
    await rc.progress.graph_delta(quotes=quotes)
    return json.dumps({
        "since_months": window, "count": len(quotes),
        **_listing(rows, [
            {"doc_title": r.get("doc_title"), "changed_on": str(r.get("changed_on")),
             "text": _clip((r.get("text") or "").strip())}
            for r in _top_rows(rows, _rows(rc), by="changed_on")
        ], key="excerpts"),
    })


@function_tool
async def find_people(wrapper: RunContextWrapper[ResearchContext], query: str) -> str:
    """Look up PEOPLE by name or role, with their job title and department/team.

    Use for "who is X", "who leads / owns / manages Y" — prefer this over
    resolve_entities when the question is about a person or a role.

    One call per QUESTION, not one per person: this is how you FIND people you do
    not have yet. Job titles and departments for people already found are filled in
    for you after the loop, so calling this again for each member of a set you
    already have spends rounds on data you are getting anyway.

    Args:
        query: a person's name, role, or team (e.g. "head of platform").
    """
    rc = wrapper.context
    async with rc.progress.step(
        StepKind.TOOL_CALL,
        with_subject(
            "Finding people who match {subject}",
            query, "Finding the people this question is about",
        ),
        tool="find_people"
    ) as step:
        people = await rc.repo.find_people(rc.ctx, query)
        step.detail = reasoning(
            f"Searched people by name, role and team for {quoted(query)}",
            f"{len(people)} matches: {preview([p.name for p in people])}"
            if people else "no matching people",
        )
    new = rc.evidence.add_entities(people)
    await rc.progress.graph_delta(entities=new)
    return json.dumps(_listing(people, [
        {"name": h.name, "role": h.description, "graph": h.graph}
        for h in people
    ], key="people"))


@function_tool
async def find_groups(wrapper: RunContextWrapper[ResearchContext], query: str) -> str:
    """Find the TEAMS and DEPARTMENTS this workspace records as groups of people,
    with how many members each has.

    Call this FIRST for any question about a named group — "who is in X", "what
    does the X team own", "who leads X" — then pass a `name` from the result to
    list_group_members. resolve_entities cannot find these: a group is recorded
    as its own kind of node, not as a concept, so a team that exists here will
    look like it does not exist there.

    The match is a loose one on the group's name, so you may pass the words the
    question used ("the data science team"); what comes back is the exact
    spelling the organization uses, which is what the member list needs.

    Args:
        query: the group name or phrase from the question.
    """
    rc = wrapper.context
    name = (query or "").strip()
    if not name:
        return json.dumps({"error": "no query given"})
    async with rc.progress.step(
        StepKind.TOOL_CALL,
        with_subject(
            "Checking which teams or departments {subject} refers to",
            name, "Checking which teams or departments this refers to",
        ),
        tool="find_groups",
    ) as step:
        groups = await rc.repo.resolve_groups(rc.ctx, name)
        step.detail = reasoning(
            f"Looked for teams and departments matching {quoted(name)}",
            f"{len(groups)} found: {preview([g.name for g in groups])}"
            if groups else "no group of that name is recorded",
        )
    new = rc.evidence.add_entities(groups)
    await rc.progress.graph_delta(entities=new)
    return json.dumps(_listing(groups, [
        {"name": g.name, "kind": g.type, "members": g.degree, "graph": g.graph}
        for g in groups[: _rows(rc)]
    ], key="groups"))


@function_tool
async def list_group_members(
    wrapper: RunContextWrapper[ResearchContext], group: str
) -> str:
    """List EVERY person recorded in a department, team, or under a manager —
    the complete set, not the most relevant ones.

    Use this whenever the question asks for ALL of something about people: "who
    are all the members of X", "list everyone in X", "who is in X", "who reports
    to Y". It selects people by recorded membership — the department/team/manager
    on their own record, AND the team and department nodes they are members of —
    so nothing is ranked and nothing is dropped for scoring lower, unlike
    find_people, which matches names and roles by text and will return people
    whose NAME merely resembles the words you passed.

    Pass the group EXACTLY as the organization names it ("Data Science", not
    "data science team"): the match is on the recorded name, not a search. Call
    find_groups first when you do not already have that exact name. If this still
    returns nothing, no group of that name is recorded — fall back to
    resolve_entities on the group and expand_entity from there.

    `truncated: true` means the ceiling bound and the set is incomplete — say so
    rather than presenting what came back as everyone.

    Args:
        group: a department, team, or manager's name.
    """
    rc = wrapper.context
    name = (group or "").strip()
    if not name:
        return json.dumps({"error": "no group given"})
    async with rc.progress.step(
        StepKind.TOOL_CALL,
        with_subject(
            "Listing everyone recorded in {subject}",
            name, "Listing everyone recorded in this group",
        ),
        tool="list_group_members",
    ) as step:
        people, docs, truncated = await rc.repo.people_by_group(rc.ctx, name)
        step.detail = reasoning(
            f"Looked up the recorded members of {quoted(name)}",
            f"{len(people)} people: {preview([p.name for p in people])}"
            if people else "no one is recorded under that department, team or manager",
        )
    new = rc.evidence.add_entities(people)
    rc.evidence.add_sources(docs)
    await rc.progress.graph_delta(entities=new, sources=docs)
    return json.dumps({
        "group": name,
        "members": [
            {"name": p.name, "profile": p.profile, "graph": p.graph} for p in people
        ],
        # A set that hit the ceiling is a partial set, and says so.
        "returned": len(people), "truncated": truncated,
    })


@function_tool
async def find_contributors(
    wrapper: RunContextWrapper[ResearchContext], document_or_topic: str
) -> str:
    """Find WHO authored / edited / contributed to documents whose title matches
    the given text. Use for "who wrote / owns / authored X". Returns one row per
    person — their role, how many matching documents they wrote, and their most
    recent one — most involved first.

    Match is on the DOCUMENT TITLE, which makes this the read that finds people
    whose OWN work is named for the thing asked about, including specialists whose
    pages sit under a narrower name than the topic itself. On a topic question run
    it ALONGSIDE find_topic_people: that one walks the topic's own connections, so
    the two reach different people and either alone returns a partial team.

    For what a PERSON has written, use find_authored_documents — passing a person's
    name here returns documents titled after them (their profile or welcome post),
    not their work.

    Args:
        document_or_topic: a document title or topic phrase to match.
    """
    rc = wrapper.context
    async with rc.progress.step(
        StepKind.TOOL_CALL,
        with_subject(
            "Working out who authored or contributed to {subject}",
            document_or_topic, "Working out who authored or contributed to this",
        ),
        tool="find_contributors",
    ) as step:
        facts = await rc.repo.document_contributors(rc.ctx, document_or_topic)
        step.detail = reasoning(
            f"Traced who authored or edited documents about {quoted(document_or_topic)}",
            f"{len(facts)} contributors found: {preview([f.source for f in facts])}"
            if facts else "no recorded contributors",
        )
    new = rc.evidence.add_facts(facts)
    await rc.progress.graph_delta(facts=new)
    return json.dumps(_listing(facts, [
        {"person": f.source, "role": f.relationship, "document": f.target,
         "detail": f.description, "documents_matched": f.support,
         "graph": f.graph}
        for f in _top_facts(facts, _rows(rc))
    ], key="contributors"))


@function_tool
async def find_topic_people(
    wrapper: RunContextWrapper[ResearchContext], topic: str
) -> str:
    """Find WHO WORKS ON a topic, product, feature or area — the people building it.

    Use for "who is building / working on / behind X", "who owns X", "which
    engineers are on X". Give it the TOPIC, not a person's name.

    This walks from the topic to the people through the pages, components and
    features they wrote, which is how the people who build something are actually
    connected to it. Expanding the topic's own connections does NOT find them: a
    topic links overwhelmingly to other topics, and only rarely straight to a
    person. Returns each person with their role and WHAT connects them to the
    topic, most-involved first.

    Args:
        topic: the topic, product, feature or area (e.g. "Enterprise Search").
    """
    rc = wrapper.context
    async with rc.progress.step(
        StepKind.TOOL_CALL,
        with_subject(
            "Finding the people who work on {subject}",
            topic, "Finding the people who work on this",
        ),
        tool="find_topic_people",
    ) as step:
        facts = await rc.repo.topic_contributors(rc.ctx, topic)
        step.detail = reasoning(
            f"Traced who has been building {quoted(topic)}",
            f"{len({f.source for f in facts})} people: {preview([f.source for f in facts])}"
            if facts else "no contributors recorded for this topic",
        )
    new = rc.evidence.add_facts(facts)
    await rc.progress.graph_delta(facts=new)
    return json.dumps(_listing(facts, [
        {"person": f.source, "role": f.description, "contributed": f.target,
         "as": f.relationship, "items_contributed": f.support,
         # The PERSON's status, and the person is the edge's source — the
         # target is the page they contributed to.
         "status": f.source_status, "graph": f.graph}
        for f in _top_facts(facts, _rows(rc))
    ], key="people"))


@function_tool
async def find_authored_documents(
    wrapper: RunContextWrapper[ResearchContext], people: list[str]
) -> str:
    """Find the documents PEOPLE wrote — everything they authored, edited or
    contributed to, newest first.

    Use for "what has X written / published / authored", "everything X wrote",
    "X's posts / pages / docs" — and when the question asks what a GROUP has
    published (a team, a manager's reports), pass the whole group in ONE call
    rather than calling once per person. Every name gets its own allowance of
    documents, so a prolific author never crowds the others out.

    This is the opposite direction from find_contributors: that one takes a
    DOCUMENT and tells you who wrote it. Passing a person's name to it instead
    finds documents whose TITLE contains that name — their profile page, not
    their work — so use this tool whenever the subject is the person.

    Any time window the user asked for is applied automatically; do not restate
    dates.

    Args:
        people: one or more person names (e.g. ["Ada Lovelace"]).
    """
    rc = wrapper.context
    since, until = rc.time_scope
    who = subject_list(people)
    async with rc.progress.step(
        StepKind.TOOL_CALL,
        f"Collecting the documents written by {who}" if who
        else "Collecting the documents these people have written",
        tool="find_authored_documents",
    ) as step:
        facts, docs = await rc.repo.documents_by_authors(
            rc.ctx, people, since=since, until=until
        )
        window = f" published {since or '…'} to {until or '…'}" if (since or until) else ""
        step.detail = reasoning(
            f"Looked up documents authored by {who or 'these people'}{window}",
            f"{len(facts)} documents: {preview([f.target for f in facts])}"
            if facts else "nothing recorded for them",
        )
    new = rc.evidence.add_facts(facts)
    # The documents themselves, not only the authorship edges. Citations resolve
    # against evidence.sources alone, so without this the answer can LIST what
    # someone wrote and cite none of it (measured: 0 citations on a substantive
    # answer, against 14 on a run where the agent happened to look each title up).
    rc.evidence.add_sources(docs)
    await rc.progress.graph_delta(facts=new)
    # Date-ranked across graphs, then round-robined by AUTHOR: the window is
    # budgeted per subject (_rows_per_subject), so one prolific recent author
    # must not spend the rows the question allotted to the others.
    by_author = interleave_by_document(
        _top_facts(facts, len(facts), by_date=True), doc_of=lambda f: f.source
    )
    return json.dumps({
        "documents": [
            {"person": f.source, "role": f.relationship, "document": f.target,
             "published": f.date, "author_role": f.description, "graph": f.graph}
            for f in by_author[: _rows_per_subject(rc, len(people or []))]
        ],
    })


@function_tool
async def request_clarification(
    wrapper: RunContextWrapper[ResearchContext], question: str, options: list[str]
) -> str:
    """Pause and ask the USER a question when their input would genuinely steer
    the research — an ambiguous request, missing context only they can supply, a
    scope decision (broad vs deep, which branch), or a judgment call that is
    theirs to make. Don't use it for routine gaps the evidence can fill; a
    well-scoped request should run without a pause. After calling it, STOP
    immediately (do not keep researching or answer) — the user's reply will
    restart research.

    Args:
        question: one concise question that resolves the ambiguity or decision.
        options: 2-5 mutually-exclusive choices (the UI adds its own "Other").
    """
    rc = wrapper.context
    async with rc.progress.step(
        StepKind.PLAN,
        "Pausing to check what you meant before spending more time on this",
        tool="request_clarification"
    ) as step:
        rc.clarification = [build_question(question, options)]
        step.detail = reasoning(
            "The investigation reached a genuine fork",
            "pausing to ask which direction you meant",
            "your answer steers the rest of the research",
        )
    return json.dumps({"status": "clarification_requested", "stop": True})


_TRACE_TOOL_IO_CHARS = 4000


def _traced(tool: FunctionTool) -> FunctionTool:
    """Wrap one tool so its invocation is a span carrying the ARGUMENTS the agent
    passed and the RESULT it got back.

    Wrapping at ``on_invoke_tool`` rather than inside the tool bodies is what
    makes this exact and complete: the SDK hands this callable the raw arguments
    JSON and takes its return value, so every tool gets Input/Output with no
    per-tool code — and the span covers the WHOLE call, including the
    evidence-pool merge each tool does after its progress step closes. (The
    SDK's own ``on_tool_start`` hook cannot do this: it receives the tool object,
    not the call payload.)

    The parent is read from the progress reporter, where the agent's round hook
    publishes the current round span — the SDK runs each tool as its own asyncio
    task, so the round cannot be inherited from context.
    """
    async def _invoke(ctx: Any, args_json: str) -> Any:
        rc = getattr(ctx, "context", None)
        progress = getattr(rc, "progress", None)
        parent = getattr(progress, "span_parent", None)
        order = None
        if progress is not None:
            progress._tool_seq += 1
            order = progress._tool_seq
        with stage_span(
            f"tool.{tool.name}", parent=parent, tool_name=tool.name,
            tool_call_order=order,
        ) as span:
            record_io(span, input=args_json, limit=_TRACE_TOOL_IO_CHARS)
            result = await tool.on_invoke_tool(ctx, args_json)
            record_io(span, output=result if isinstance(result, str) else str(result),
                      limit=_TRACE_TOOL_IO_CHARS)
            return result

    return replace(tool, on_invoke_tool=_invoke)


def research_tools(
    clarify_enabled: bool = False, *, graph_text: bool = True, people: bool = False,
) -> list[FunctionTool]:
    """The agent's ACL-bound toolset (static objects; identity is injected per run).

    ``request_clarification`` is included only when mid-research clarification is
    on, so the capability is entirely absent (not just inert) when disabled.
    ``graph_text=False`` (a zero graph-chunk cap) removes the tools whose RETURN is
    graph chunk/document text entirely, for the same reason — a tool that is
    absent cannot be called, so the guarantee does not depend on the model's
    restraint. ``people`` adds the people-directory tools, which read properties
    only an HR-style connector writes (``people_graph_enabled``)."""
    tools = [
        resolve_entities,
        expand_entity,
        # Structure, not text: it returns relationships, so it stays in the base
        # set — the `graph_text` gate below removes the tools whose RETURN is
        # graph chunk/document text, which this is not.
        find_entity_paths,
        get_claims,
        find_recent_changes,
        list_relationship_types,
        get_source_documents,
        find_documents,
    ]
    if people:
        tools += [
            find_people,
            # Enumeration, beside the relevance reads: "all the members of X" is a
            # set question, and every other people tool answers it as top-K. The
            # pair: find_groups discovers the group node entity resolution cannot
            # see, list_group_members enumerates it.
            find_groups,
            list_group_members,
            find_contributors,
            find_topic_people,
            find_authored_documents,
        ]
    if graph_text:
        tools += [get_document_text, traverse_to_chunks, recent_change_excerpts,
                  find_relevant_chunks]
    if clarify_enabled:
        tools.append(request_clarification)
    # Every tool traced, by construction: no tool can be added to the set without
    # getting its span, which is the guarantee "no tool call is invisible" needs.
    return [_traced(t) for t in tools]
