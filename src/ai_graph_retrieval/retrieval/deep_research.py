"""The agent lanes — the loop as a RetrievalStrategy, at two budgets.

The agent is a *producer of evidence*: one pool, one
trace, same synthesis afterwards. Tools are bound to this request's identity
at run time (injected via the run context, never captured in a closure).

Search and Super Research share ONE implementation — Super Research extends
the Search flow (product direction, 2026-08-08) with more rounds, a bounded
graph-chunk allowance and the mid-research clarification tool; every
difference between them is a profile value, never a code branch.
"""
from __future__ import annotations

import asyncio

import structlog

from ai_graph_retrieval.core.models import Evidence, StepKind
from ai_graph_retrieval.core.narration import reasoning
from ai_graph_retrieval.observability.tracing import set_span_attributes, stage_span
from ai_graph_retrieval.retrieval.base import (
    RetrievalPlan,
    RetrievalStrategy,
    StrategyContext,
)
from ai_graph_retrieval.retrieval.chunk_relevance import score_graph_chunks
from ai_graph_retrieval.retrieval.neighbor_scoring import NeighborScorer, ScoringBudget
from ai_graph_retrieval.retrieval.person_identity import canonicalize_person_names
from ai_graph_retrieval.retrieval.quality import rank_claims_by_recency

logger = structlog.get_logger(__name__)


def build_scorer(sc: StrategyContext) -> NeighborScorer | None:
    """The tiered neighbor re-ranker for this request, or None when the mode is
    not granted any LLM scoring calls or the feature is off."""
    if not (sc.settings.scoring_enabled and sc.profile.scoring_budget > 0):
        return None
    return NeighborScorer(
        sc.llm, sc.settings, sc.usage, ScoringBudget(sc.profile.scoring_budget)
    )


def _ordered_names(
    ev: Evidence, *, skip: set[str], people_first: bool = False
) -> list[str]:
    """Every name the evidence surfaced, in the order the engines surfaced it.

    Traversal neighbors first (``fact.target``/``fact.source``: what the agent
    actually walked to is the strongest signal of what the answer will name),
    then the pool's entities (seed relevance order). Pure selection over pool
    order — no text matching, no ranking of our own, and nothing here reads the
    question (rule 8). ``skip`` is what each caller already considers described.

    DOCUMENTS ARE NOT CANDIDATES. A fact's target is as often a document as an
    entity (every authorship and contribution edge points at one), and a document
    title is not a name any of these passes can describe: it matches no Person
    node and no entity, so it returned nothing and spent a slot doing it.
    Measured, the claim budget went on "FAQ TSV file", "HRIS Certification
    document" and "[Virtual Roundtable] Professional Services". A title that is
    ALSO a pool entity stays — the exclusion is about what the row is, not about
    the string.

    ``people_first`` puts person-typed names at the head for the passes whose
    read only matches people, WITHOUT dropping anything: the budgets are spent
    off the head of this list in fact-INSERTION order, so one broad early read
    could take every slot before the question's own people were even in the pool
    (measured: 60 profile slots spent, 43 profiles filled, none of them in the
    department the question named, and not one of the five people the answer
    listed present as an entity). Ordering only — a mis-typed person loses
    priority, never presence.
    """
    entity_types = {e.name: (e.type or "").lower() for e in ev.entities}
    documents = {
        name for name in (
            [f.target for f in ev.facts if (f.target_type or "").lower() == "document"]
            + [s.title for s in ev.sources if s.title]
        ) if name and name not in entity_types
    }
    people: list[str] = []
    others: list[str] = []
    seen: set[str] = set()

    def consider(name: str | None, *, person: bool) -> None:
        # `and not person` guards the one real collision: a people-object
        # document's TITLE IS the person's name, so a person reached only as a
        # bare fact endpoint would otherwise be excluded by their own profile
        # page — the exact opposite of what this filter is for.
        if not name or name in skip or name in seen or (name in documents
                                                        and not person):
            return
        seen.add(name)
        target = people if (people_first and person) else others
        target.append(name)

    def is_person(name: str | None, declared: str | None, status: str | None) -> bool:
        # The row's own type, the pool's type for the same name, or the presence
        # of an employment status — which only a Person node carries.
        return (str(declared or "").lower() == "person"
                or entity_types.get(name or "") == "person"
                or bool(status))

    # Ranked, not pool-ordered: the budgets are spent off the HEAD of this list,
    # so insertion order decided which people got enriched and an identical query
    # could enrich a different set between runs. Same key the digest and the
    # neighbour ranker already sort facts by (structure first, then score,
    # support, name) — one ordering rule for the pool, not a new one here.
    for f in sorted(ev.facts,
                    key=lambda f: (not f.structural, -f.score, -f.support,
                                   f.target or "", f.source or "")):
        consider(f.target,
                 person=is_person(f.target, f.target_type, f.target_status))
        consider(f.source, person=is_person(f.source, None, f.source_status))
    # Entities by their own seed score, with the name breaking ties so two
    # equally-scored rows cannot swap between runs.
    for e in sorted(ev.entities, key=lambda e: (-e.score, -e.degree, e.name)):
        consider(e.name, person=is_person(e.name, e.type, None))
    return people + others


def _uncovered_names(sc: StrategyContext, cap: int) -> list[str]:
    """Names no CLAIM describes yet, capped — the claim pass's candidates.

    "Covered" means the name already appears as a claim SUBJECT: a claim ABOUT
    someone else that merely mentions the name does not describe it.
    """
    ev = sc.evidence.snapshot()
    return _ordered_names(
        ev, skip={c.subject for c in ev.claims}, people_first=True)[:cap]


def _unprofiled_names(sc: StrategyContext, cap: int) -> list[str]:
    """Names carrying no structured profile yet, capped — the profile pass's
    candidates.

    Deliberately NOT filtered by "has claims": the two passes describe different
    things. A person can have claims (expertise, reporting lines) and still no
    job title, and the node's own fields are the authoritative source for the
    columns a roster renders.
    """
    ev = sc.evidence.snapshot()
    return _ordered_names(
        ev, skip={e.name for e in ev.entities if e.profile}, people_first=True)[:cap]


def _unconnected_people(sc: StrategyContext, cap: int) -> list[str]:
    """People the pool holds but no relationship connects yet, capped.

    "Connected" means the name is already an endpoint of some fact: the
    structure that answers an org question is the edge between two people, so a
    person who only ever appears as an entity row has none of it.
    """
    ev = sc.evidence.snapshot()
    linked = {n for f in ev.facts for n in (f.source, f.target) if n}
    return [
        e.name for e in ev.entities
        if (e.type or "").lower() == "person" and e.name not in linked
    ][:cap]


async def enrich_profiles(sc: StrategyContext) -> None:
    """Post-loop profile top-up: the STRUCTURED attributes of what was found.

    The loop reaches most people as relationship TARGETS — a name and nothing
    else — and their blended ``description`` is only populated when some
    document happened to describe them. So an enumerated roster rendered bare
    names, or roles that varied run to run depending on which claims the agent
    chose to fetch.

    The node's own fields are the authoritative, complete source: ONE batched
    read (``fetch_people_profiles``) fills them for every candidate at once, so
    the answer's columns stop depending on the agent's tool choices. Schema-
    driven, never query-driven: nothing here reads the question, and the pass
    runs identically on every request in both agent modes.

    Additive only — a failure costs the profiles, never the answer.
    """
    cap = sc.profile.profile_enrich_top
    if cap <= 0:
        return
    picks = _unprofiled_names(sc, cap)
    if not picks:
        return
    # A stage span, not a progress step: these reads used to parent straight to
    # the ROOT span, leaving 9s of Neo4j work sitting in the waterfall with no
    # context. A progress step would also add a user-facing line to
    # the streamed UI, which this silent pass has never had.
    # getattr, not attribute access: this is an ADDITIVE pass whose contract is
    # "a failure costs the profiles, never the answer", and the span opens OUTSIDE
    # the try below — so a progress object without the attribute would kill the
    # request instead of skipping a top-up. Matches agent/tools.py, which reads it
    # the same defensive way.
    with stage_span("stage.profile_topup",
                    parent=getattr(sc.progress, "span_parent", None),
                    candidates=len(picks)):
        try:
            profiles, person_docs = await sc.repo.fetch_people_profiles(sc.ctx, picks)
        except Exception as exc:  # additive pass, never fatal
            logger.warning("profile_enrichment_failed", error_type=type(exc).__name__,
                           error=str(exc), **sc.ctx.log_fields())
            return
        if not profiles:
            return
        # Merge, don't replace: a name already in the pool keeps its description and
        # url and gains the profile (EvidencePool.add_entities enriches by name).
        added = sc.evidence.add_entities(profiles)
        # Each profiled person's OWN document (their people object) joins the pool's
        # sources: it is the document that supports the profile fields just added, and
        # the only way a roster assembled from structured edges and person properties
        # can carry citations — synthesis cites nothing it wasn't shown a document for.
        sc.evidence.add_sources(person_docs)
        logger.info("profile_enrichment", candidates=len(picks),
                    described=len(profiles), new_entities=len(added),
                    person_docs=len(person_docs),
                    **sc.ctx.log_fields())
        if added or person_docs:
            try:
                await sc.progress.graph_delta(entities=added, sources=person_docs)
            except Exception as exc:  # trace emission is additive too
                logger.warning("profile_enrichment_delta_failed",
                               error_type=type(exc).__name__, error=str(exc),
                               **sc.ctx.log_fields())


async def enrich_relationships(sc: StrategyContext) -> None:
    """Post-loop relationship top-up: connect the people the run found.

    Reporting lines and team membership are structured edges between Person
    nodes, so ``expand_entity`` is the only read that surfaces them — the
    people-directory reads return who someone IS and nothing about who they
    report to. The agent's instructions say exactly that, but whether it spends
    a round on the traversal is a per-run tool choice: measured, a run that
    skipped it came back with 27-36 people and ZERO relationships between them,
    and the same question answered with the structure on a later run.

    This is the same treatment the profile, claim and graph-chunk allowances
    already get: what the toolset can reach stops depending on which tools the
    loop happened to call. Schema-driven, never query-driven — nothing here
    reads the question, and the pass runs identically on every request in both
    agent modes. Additive only: a failed or slow read costs its own edges,
    never the rest of the batch or the answer.
    """
    cap = sc.profile.relationship_enrich_top
    if cap <= 0:
        return
    picks = _unconnected_people(sc, cap)
    if not picks:
        return
    async def _expand(name: str) -> list | None:
        # Per-read bound and per-read failure, inside the mode deadline: the
        # reads run in parallel, so the pass's wall-clock stays the same bound,
        # and a slow or failed read costs its own edges, never the batch's.
        try:
            async with asyncio.timeout(sc.settings.relationship_enrich_timeout_s):
                return await sc.repo.expand_entity(sc.ctx, name)
        except Exception as exc:  # additive pass, never fatal
            logger.warning("relationship_expand_failed", entity=name,
                           error_type=type(exc).__name__, error=str(exc),
                           **sc.ctx.log_fields())
            return None

    with stage_span("stage.relationship_topup",
                    parent=getattr(sc.progress, "span_parent", None),
                    candidates=len(picks)):
        results = await asyncio.gather(*(_expand(name) for name in picks))
        batches = [b for b in results if b is not None]
        # Only edges BETWEEN people already in the pool: this pass exists to
        # connect the set the run assembled, not to widen it — a person's full
        # neighbourhood is the agent's call and its budget, not this backstop's.
        known = {e.name for e in sc.evidence.snapshot().entities}
        facts = [
            f for batch in batches for f in batch
            if f.source in known and f.target in known
        ]
        added = sc.evidence.add_facts(facts)
        logger.info("relationship_enrichment", candidates=len(picks),
                    failed=len(results) - len(batches),
                    fetched=sum(len(b) for b in batches), kept=len(facts),
                    new_facts=len(added), **sc.ctx.log_fields())
        if added:
            try:
                await sc.progress.graph_delta(facts=added)
            except Exception as exc:  # trace emission is additive too
                logger.warning("relationship_enrichment_delta_failed",
                               error_type=type(exc).__name__, error=str(exc),
                               **sc.ctx.log_fields())


async def enrich_claims(sc: StrategyContext) -> None:
    """Post-loop claim top-up: describe what the loop found but did not detail.

    The agent-led redesign removed the old deterministic expansion
    (search_expand_top=30 / search_claim_top=10) and with it the per-entity
    facts — roles, titles, locations, status — for the people and things an
    answer ENUMERATES: a short loop finds the set (one expand call returns a
    whole team) but has no rounds left to describe each member, so the answer
    listed bare names. This pass claim-fetches the uncovered entities in ONE
    parallel batch after the loop — no agent rounds, no extra model calls —
    bounded by the profile's ``claim_enrich_top``.

    Additive only, so a failure here never costs the answer: the loop's
    evidence is already in the pool, and synthesis proceeds on it either way.
    """
    cap = sc.profile.claim_enrich_top
    if cap <= 0:
        return
    picks = _uncovered_names(sc, cap)
    if not picks:
        return
    async with sc.progress.step(
        StepKind.TRAVERSE,
        "Filling in details about the people and things found",
    ) as step:
        try:
            # since_iso=None DELIBERATELY, even when the question asked a time
            # window: these claims DESCRIBE the entities (roles, titles,
            # status), and a person's role claim routinely predates the asked
            # window — windowing here would strip the very designations the
            # pass exists to add. The window applies to the answer's ITEMS,
            # by the model, per the time-scope rules; describing claims carry
            # their validity dates and [expired]/[future] flags into the
            # digest for it to weigh.
            batches = await asyncio.gather(
                *(sc.repo.fetch_claims(sc.ctx, name, None) for name in picks)
            )
        except Exception as exc:  # additive pass, never fatal
            logger.warning("claim_enrichment_failed", error_type=type(exc).__name__,
                           error=str(exc), **sc.ctx.log_fields())
            step.detail = "Details lookup was unavailable — continuing with what was found"
            return
        # Keep only what DESCRIBES each pick — claims whose subject is the
        # fetched name — and a few per entity, newest first. The digest renders
        # the pool's first `digest_max_claims`; one entity's unbounded return
        # (≤60 per graph) could fill that window by itself and push every later
        # member's role claim out of the prompt.
        #
        # Ranked before the slice, never after: the batch is the fan-out's
        # concatenation (every row of the first graph, then the second), so a
        # positional cut let the first graph spend the whole per-entity budget
        # and dropped a newer claim from the second purely for arriving later.
        per_entity = max(0, sc.settings.agent_claim_enrich_per_entity)
        claims = []
        described = 0
        for name, batch in zip(picks, batches, strict=False):
            described_by = [c for c in batch if c.subject == name]
            kept = rank_claims_by_recency(described_by)[:per_entity]
            claims.extend(kept)
            described += bool(kept)
        sc.evidence.add_claims(claims)
        step.detail = reasoning(
            f"Looked up recorded details for {len(picks)} of the people and things found",
            f"details found for {described}" if described else "nothing further was recorded",
        )
    if claims:
        try:
            await sc.progress.graph_delta(claims=claims)
        except Exception as exc:  # trace emission is additive too — the claims
            # are already in the pool; a failed UI delta must not abort a
            # retrieval that has its evidence.
            logger.warning("claim_enrichment_delta_failed",
                           error_type=type(exc).__name__, error=str(exc),
                           **sc.ctx.log_fields())


async def enrich_graph_chunks(sc: StrategyContext) -> None:
    """Post-loop graph-chunk top-up (2026-08-18): deterministic, not
    agent-dependent.

    The agent MAY pull graph chunk text through its tools, but whether it does
    is a per-run tool choice — measured on Search's 2-round budget it usually
    spends its rounds on resolution and claims, so a configured graph-chunk
    allowance delivered nothing on most runs. This pass makes the allowance
    real: ONE batched supporting-chunks read for the pool's top anchors, drawn
    from the SAME budget the tools share (``budget_graph_quotes``: dedup against
    everything already pooled BEFORE counting, then cap), so agent-fetched and
    top-up chunks together never exceed ``graph_chunk_cap``.

    Gated on the EFFECTIVE capability (flag AND cap > 0) — with a cap of 0
    (config or per-request options.graph_chunk_cap) this pass does not run and
    no stage reads chunk text from the graph. Additive only: a failure costs
    the top-up, never the answer.
    """
    cap = sc.profile.graph_chunk_cap if sc.profile.graph_text_tools else 0
    if cap <= 0:
        return
    from ai_graph_retrieval.agent.tools import budget_graph_quotes

    ev = sc.evidence.snapshot()
    # The anchors whose text is most likely to matter: pool order (seed
    # relevance, then discovery order) — the same ordering the other top-ups
    # trust. Bounded by the cap itself: each anchor can contribute several
    # excerpts, so more anchors than kept-chunk slots is pure read waste.
    names = _ordered_names(ev, skip=set())[:cap]
    if not names:
        return
    # Stage span for the same reason as the profile top-up above.
    with stage_span("stage.graph_chunk_topup",
                    parent=getattr(sc.progress, "span_parent", None),
                    anchors=len(names), cap=cap):
        try:
            quotes = await sc.repo.supporting_chunks(sc.ctx, names)
        except Exception as exc:  # additive pass, never fatal
            logger.warning("graph_chunk_enrichment_failed",
                           error_type=type(exc).__name__, error=str(exc),
                           **sc.ctx.log_fields())
            return
        # RANK BEFORE THE CUT. `budget_graph_quotes` keeps by POSITION in the
        # per-anchor concatenation, and this read routinely returns several times
        # the cap (measured: 206 fetched, 40 kept — so 166 chunks were discarded
        # for where they sat in the list). The relevance pass that exists to order
        # this tier ran afterwards, over the survivors only, so it could never
        # rescue a roster chunk that arrived late. Scoring the CANDIDATES makes
        # the same cap a ranked selection at the same read cost; on any failure
        # every score stays 0.0 and the sort is stable, so the order is exactly
        # the one the budget used before.
        try:
            await score_graph_chunks(sc, quotes, stage="chunk_relevance_candidates")
        except Exception as exc:  # ordering only — never costs the top-up
            logger.warning("graph_chunk_candidate_scoring_failed",
                           error_type=type(exc).__name__, error=str(exc),
                           **sc.ctx.log_fields())
        quotes = sorted(quotes, key=lambda q: -(q.relevance or 0.0))
        kept = budget_graph_quotes(sc.evidence, cap, quotes)
        if not kept:
            return
        sc.evidence.add_quotes(kept)
        logger.info("graph_chunk_enrichment", anchors=len(names),
                    fetched=len(quotes), kept=len(kept), cap=cap,
                    **sc.ctx.log_fields())
        try:
            await sc.progress.graph_delta(quotes=kept)
        except Exception as exc:  # trace emission is additive too
            logger.warning("graph_chunk_enrichment_delta_failed",
                           error_type=type(exc).__name__, error=str(exc),
                           **sc.ctx.log_fields())


async def enrich_mentions(sc: StrategyContext) -> None:
    """Post-loop MENTION top-up: the entities the pool's graph chunks
    are about, and the relationships among the pool's entities.

    Two gaps, one pass, both measured on the live graphs (2026-08-28):

    * A passage reaches the pool through similarity, a document read or a
      traversal, and the entity it is ABOUT arrives with it only as a name in the
      chunk's mention list — nothing added it as an ENTITY. The answer named three
      security engineers the graph knows well; the pool held 31 entities and not
      one of them, so the digest's ENTITIES section could not describe the people
      the answer was about and the client's evidence graph showed none of them.
    * The expand tool walks ONE entity's neighbourhood per call, so entities the
      loop found separately sat in the pool with no edge between them however
      directly the graph links them (four facts in a pool of 31 entities).

    Read 1 hydrates the mentioned entities — ranked by how many of THIS
    request's retrieved passages mention each, then degree — through the pool's
    normal merge (an entity already present keeps its fields and gains nothing
    but the union). Read 2 returns every visible relationship whose both
    endpoints are already pool entities (the induced subgraph), so the digest's
    FACTS and the client's graph edges connect what the evidence connects.

    Schema-driven, never query-driven: the inputs are the pool's own chunk ids
    and entity names; nothing here reads the question. Additive only — a failure
    costs the top-up, never the answer.
    """
    cap = sc.profile.mention_enrich_top
    if cap <= 0:
        return
    ev = sc.evidence.snapshot()
    # Each pooled chunk carries the chunk's question-similarity (stamped by the relevance
    # pass, which runs before this one), the weight the read ranks by.
    chunk_weights = {
        q.chunk_id: (q.relevance or 0.0)
        for q in ev.quotes
    }
    if not chunk_weights:
        return
    with stage_span("stage.mention_topup",
                    parent=getattr(sc.progress, "span_parent", None),
                    chunks=len(chunk_weights), cap=cap) as span:
        try:
            hits = await sc.repo.mentioned_entities(sc.ctx, chunk_weights, limit=cap)
        except Exception as exc:  # additive pass, never fatal
            logger.warning("mention_enrichment_failed", error_type=type(exc).__name__,
                           error=str(exc), **sc.ctx.log_fields())
            return
        added = sc.evidence.add_entities(hits)
        # The induced subgraph over the PASSAGE-GROUNDED entities — the ranked
        # set just read, present or newly added — so a person found by passage
        # and their manager mentioned beside them get the REPORTS_TO edge between
        # them. Not the whole pool: resolution candidates no passage mentions
        # (typo variants of a company name, say) would only contribute the
        # relationships among themselves.
        facts = []
        fact_limit = sc.settings.agent_induced_facts_limit
        if fact_limit > 0 and hits:
            try:
                facts = sc.evidence.add_facts(await sc.repo.facts_among(
                    sc.ctx, [h.name for h in hits], limit=fact_limit,
                ))
            except Exception as exc:  # the entities are already pooled; keep them
                logger.warning("induced_facts_failed", error_type=type(exc).__name__,
                               error=str(exc), **sc.ctx.log_fields())
        set_span_attributes(
            span, mentioned=len(hits), new_entities=len(added), new_facts=len(facts),
        )
        logger.info("mention_enrichment", chunks=len(chunk_weights), mentioned=len(hits),
                    new_entities=len(added), new_facts=len(facts), cap=cap,
                    **sc.ctx.log_fields())
        if added or facts:
            try:
                await sc.progress.graph_delta(entities=added, facts=facts)
            except Exception as exc:  # trace emission is additive too
                logger.warning("mention_enrichment_delta_failed",
                               error_type=type(exc).__name__, error=str(exc),
                               **sc.ctx.log_fields())


class DeepResearchStrategy(RetrievalStrategy):
    name = "deep_research"

    async def run(self, sc: StrategyContext, plan: RetrievalPlan) -> None:
        # Imported here: agent.runtime imports retrieval.base, and this module
        # is imported by the container — a module-level import would be a cycle.
        from ai_graph_retrieval.agent.runtime import ResearchAgent

        agent = ResearchAgent(
            sc.settings,
            # Mid-research clarification is Super Research's capability
            # (profile flag, plus the settings kill-switch checked inside).
            clarify=sc.profile.mid_research_clarify,
            # EFFECTIVE graph-text capability: the flag alone is not enough — a
            # zero cap (Search's setting, or a per-request options.graph_chunk_cap
            # of 0) means no graph chunk may ever be kept, so the graph-text
            # tools are removed from the toolset entirely rather than exposed
            # and inert ("a tool that is absent cannot be called"). sc.profile
            # is the post-override profile, so this is per-request.
            graph_text=sc.profile.graph_text_tools and sc.profile.graph_chunk_cap > 0,
            # The FULL agent lane. `deep_tool_rows` is already the profile's "this is
            # the whole loop, not a bounded follow-up" signal — it is what grants the
            # wider per-call row budget (ResearchContext.deep) — so the depth
            # instructions travel with the budgets they exist to justify, and a mode
            # assembled with the full budgets can never get the light instructions.
            # Keyed on the profile, not on the strategy's own name, because Search
            # subclasses this strategy.
            research=sc.profile.deep_tool_rows,
        )
        await agent.run(sc, build_scorer(sc))
        # A mid-research clarification pause raises out of run() above and the
        # request ends this turn, so no top-up runs on a paused run.
        # Graph-chunk top-up FIRST: it dedups against everything already pooled
        # (the profile and claim passes add no chunks, so nothing is lost by
        # running ahead of them), and the
        # mention pass below reads the pool's chunk ids, so every chunk this
        # request will ever hold must be in before it runs.
        await enrich_graph_chunks(sc)
        # Score the graph chunks now the pool holds every one of them (no later
        # pass adds a chunk), so a single batched read per graph covers the whole
        # tier. Ordering only — it adds no evidence and removes none, and on any
        # failure the scores stay 0.0 and the digest keeps the document order it
        # used before. Before the mention pass, which ranks by these scores.
        await score_graph_chunks(sc)
        # Mentions: the entities those chunks are about, and the relationships
        # among them — BEFORE profiles and claims, so a person hydrated here is a
        # candidate for both describing passes.
        await enrich_mentions(sc)
        # The people passes read person-directory properties (profiles, reporting
        # lines, status) that only an HR-style connector writes — off unless the
        # deployment says its graphs carry them.
        people = sc.settings.people_graph_enabled
        # BEFORE the budgets are spent: the same human enters the pool under each
        # graph's own label ("Jane Doe" / "Jane A. Doe", "jane.doe" / "Jane Doe"),
        # and every label was a separate candidate. The profile cap and the claim
        # cap are spent off the head of that candidate list, so a duplicate
        # identity took a slot — and the claim pass an entire per-entity read —
        # that a person the answer had not described yet would otherwise have
        # had. One pass collapses them; it runs AGAIN below because the top-ups
        # themselves produce names. Mentions above are a name-producing read too,
        # so this collapse sees the people they hydrated.
        if people:
            await canonicalize_person_names(sc)
            # Profiles: one batched read that populates the columns a roster
            # renders; claims then add what the node fields do not carry.
            await enrich_profiles(sc)
            # After profiles: that pass is what types a bare relationship target
            # as a Person, and this one selects on that type.
            await enrich_relationships(sc)
        await enrich_claims(sc)
        # One human, one name — after every name-producing read, because it
        # rewrites the pool's entity KEYS and the facts that reference them, and a
        # later read emitting the old label would re-split the person it just merged.
        # The top-ups above are name-producing reads, so this run is what makes the
        # guarantee hold; the earlier one exists to protect their budgets. Idempotent
        # on names the first pass already canonicalised.
        if people:
            await canonicalize_person_names(sc)
        # Taking out the people who have left is NOT done here. It is a retrieval
        # BOUNDARY rule, applied once to whatever the pool holds by
        # pipeline/execution.EvidenceCollector — including when this lane never
        # reached its end because the budget expired. See retrieval/person_status.py.


class SearchAgentStrategy(DeepResearchStrategy):
    """Search Mode's lane: the SAME agent loop, at the search profile's smaller
    budgets (rounds/patience/tool rows), with a bounded graph-chunk allowance
    (``search_graph_chunk_cap``; 0 restores graph-as-enrichment-only) and no
    mid-research pause.

    Search has no fixed traversal step — the agent explores every configured
    graph through its tools and decides what is relevant.

    Subclassed rather than aliased purely so the trace and the response's
    `strategy` field report "search" — the behaviour is deliberately identical,
    and any change to the loop must apply to both modes.
    """

    name = "search"
