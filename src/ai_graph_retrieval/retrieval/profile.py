"""Execution profiles — one depth table per user-facing mode.

Every knob that decides HOW HARD the pipeline works for one request lives here,
resolved once per request from the caller's mode and handed to the stages on the
``StrategyContext``.

Two modes, two profiles, resolved by the CALLER'S EXPLICIT MODE alone — there
is no automatic mode selection, no intent-based lane routing, and no
intent-based profile override. The intent an entry classification produces is
telemetry; it never changes the strategy or the depth caps a user's mode choice
bought.

ONE retrieval architecture: the agent resolves the question's entities in the
knowledge graph and enriches from there — claims, relationships, metadata and a
bounded number of graph chunks. Super Research is that flow with more rounds, a
larger graph-chunk allowance, and the mid-research pause — never a separate
retrieval implementation.

Rule 8 (model-driven, never rule-driven) holds: a profile is a set of named,
query-agnostic constants selected by an EXPLICIT user mode. Nothing here reads
the question text and no value is derived from it — the model still owns every
scope, ranking and citation decision.

Rule 6 (config source of truth) holds: the values are ``Settings`` fields, so a
per-env override goes in that env's configuration. This module encodes only the
SHAPE of the table and how a mode resolves to a row.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import structlog

from ai_graph_retrieval.retrieval.base import Intent, Mode
from ai_graph_retrieval.settings import Settings

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class ExecutionProfile:
    """How hard the pipeline works for one request. Frozen: resolved once by the
    planner, then read (never mutated) by every stage downstream."""

    name: str
    # Wall-clock hang guard for retrieval (synthesis runs after it, on its own).
    budget_s: float
    # Agent rounds this mode may spend. 0 ⇒ the loop never runs; `agent_patience`
    # is the consecutive-no-new-evidence stop.
    agent_rounds: int
    agent_patience: int
    # LLM neighbor re-ranking calls allowed per request. 0 ⇒ cheap order only.
    scoring_budget: int
    # Post-agent-loop claim top-up: after the loop, the lane claim-fetches up to
    # this many entities the evidence surfaced but no claim describes yet, so the
    # items an answer enumerates carry their roles, titles and status without
    # spending agent rounds on them. 0 ⇒ the pass never runs.
    claim_enrich_top: int = 0
    # Post-loop relationship top-up: how many surfaced-but-unconnected people get
    # expanded so the edges BETWEEN the run's people are on record without
    # spending agent rounds on them. 0 ⇒ the pass never runs.
    relationship_enrich_top: int = 0
    # Post-loop profile top-up: how many surfaced entities get their structured
    # source fields fetched (one batched read). 0 ⇒ the pass never runs.
    profile_enrich_top: int = 0
    # Post-loop mention top-up: how many entities the pool's graph chunks mention
    # get hydrated into the pool (one batched read), plus the relationships among
    # the pool's entities. 0 ⇒ the pass never runs. Effective only where graph
    # chunks can exist — the pass reads the pool's chunk ids, so a zero
    # graph_chunk_cap run has nothing for it to read and it returns without a query.
    mention_enrich_top: int = 0
    # Excerpt char budget for the digest. 0 ⇒ use the Settings default. This is
    # the single biggest cost lever in the service, so it belongs beside the
    # other per-mode caps rather than as a mode check inside the composer.
    digest_quote_chars: int = 0
    # Whether the agent's list-returning tools use the WIDE row budget
    # (agent_tool_rows_research) instead of the bounded one (agent_tool_rows).
    # A profile flag, not a name check: keying on the profile's NAME string
    # silently breaks the moment a profile is renamed.
    deep_tool_rows: bool = False
    # Whether the agent may pull chunk/document TEXT from the graph (the
    # get_document_text / traverse_to_chunks / recent_change_excerpts tools and
    # get_source_documents' supporting excerpts), bounded by ``graph_chunk_cap``.
    # Graph chunks are never assumed to hold a document's complete source text.
    graph_text_tools: bool = False
    # Upper bound on graph chunks added by the agent's tools, TOTAL across every
    # graph source (not per source) for the whole request. 0 turns the capability
    # OFF for the run even with ``graph_text_tools=True`` — the effective flag the
    # agent is built from is ``graph_text_tools and graph_chunk_cap > 0``, so a
    # zero cap removes the graph-text tools rather than leaving them inert.
    graph_chunk_cap: int = 0
    # Whether the agent may pause mid-research to ask the user a clarifying
    # question (the request_clarification tool; also gated by the
    # ``mid_research_clarification_enabled`` kill-switch). Super Research only:
    # a mid-flight pause discards the run's evidence, which is a fair trade for
    # a long investigation and the wrong one for Search's bounded loop, whose
    # clarification chance already passed at the pre-retrieval gate.
    mid_research_clarify: bool = False
    # Whether the retrieval-boundary employment gate runs for this mode
    # (pipeline/execution.EvidenceCollector → person_status.sweep_inactive_people).
    # Only meaningful when the graph carries a people layer
    # (``settings.people_graph_enabled``); the collector checks both.
    employment_gate: bool = True
    # Whether the quality gate may RE-ORDER the sources by blended trust
    # (relevance + recency + graph mentions + authority). A profile flag rather
    # than a mode check inside the gate, for the same reason every other cap
    # lives here.
    trust_ranking: bool = True

    @property
    def agent_enabled(self) -> bool:
        """True when this mode is allowed to spend agent rounds at all."""
        return self.agent_rounds > 0


def search_profile(s: Settings) -> ExecutionProfile:
    """Search: a bounded AGENT LOOP over the knowledge graph.

    There is no fixed traversal step — the agent decides what to resolve,
    expand, claim-fetch and look up, through the same tools Super Research
    uses, at this mode's smaller budgets.

    ``graph_text_tools=True`` with ``graph_chunk_cap`` from
    ``search_graph_chunk_cap`` — the graph contributes bounded chunk text: every
    graph chunk is deduplicated against text already in the pool BEFORE it
    counts against the cap, and again (containment, longer copy wins) at digest
    time. A cap of 0 — the setting, or a per-request ``options.graph_chunk_cap``
    — switches the capability off for the run: the graph-text tools are then
    ABSENT from the agent's toolset.
    """
    return ExecutionProfile(
        name="search",
        budget_s=s.search_budget_s,
        agent_rounds=s.search_agent_rounds,
        agent_patience=s.search_agent_patience,
        scoring_budget=s.search_scoring_budget,
        relationship_enrich_top=s.agent_relationship_enrich_top,
        claim_enrich_top=s.agent_claim_enrich_top,
        profile_enrich_top=s.agent_profile_enrich_top,
        mention_enrich_top=s.agent_mention_enrich_top,
        digest_quote_chars=s.search_digest_quote_chars,
        graph_text_tools=True,
        graph_chunk_cap=s.search_graph_chunk_cap,
    )


def super_research_profile(s: Settings) -> ExecutionProfile:
    """Super Research: the SAME flow as Search — extended, never re-implemented.

    Three extensions over the Search profile, and nothing else differs:

    * MORE ROUNDS — ``agent_max_rounds`` with the wide tool-row budget
      (``deep_tool_rows``): the primary retrieval/reasoning difference.
    * MORE GRAPH CHUNKS — ``graph_chunk_cap`` from
      ``super_research_graph_chunk_cap``, TOTAL across every graph source, each
      deduplicated against text already in the pool.
    * MID-RESEARCH CLARIFICATION — ``mid_research_clarify=True``: the agent
      may pause once mid-run to ask the user a clarifying question when it
      finds the investigation genuinely ambiguous (distinct from the
      pre-retrieval pause both modes share).
    """
    return ExecutionProfile(
        name="super_research",
        budget_s=s.super_research_budget_s,
        agent_rounds=s.agent_max_rounds,
        agent_patience=s.agent_stale_patience,
        scoring_budget=s.scoring_query_budget,
        relationship_enrich_top=s.agent_relationship_enrich_top,
        claim_enrich_top=s.agent_claim_enrich_top,
        profile_enrich_top=s.agent_profile_enrich_top,
        mention_enrich_top=s.agent_mention_enrich_top,
        digest_quote_chars=s.super_research_digest_quote_chars,
        deep_tool_rows=True,
        graph_text_tools=True,
        graph_chunk_cap=s.super_research_graph_chunk_cap,
        mid_research_clarify=True,
    )


_BY_MODE = {
    Mode.SEARCH: search_profile,
    Mode.SUPER_RESEARCH: super_research_profile,
}


def profile_for(settings: Settings, mode: Mode, intent: Intent) -> ExecutionProfile:
    """The profile one request runs under: the caller's mode, nothing else.

    ``intent`` is accepted (and deliberately ignored) so the signature states
    the invariant: the classifier's verdict is telemetry and must never buy a
    request more depth than its mode did.
    """
    del intent
    return _BY_MODE[mode](settings)


# The caller-overridable knobs → floor. Ceilings come from Settings
# (options_max_*) so a per-env override can widen/narrow them.
_OVERRIDE_FLOORS = {
    "agent_rounds": 1,
    "graph_chunk_cap": 0,
}


def apply_overrides(
    profile: ExecutionProfile, overrides: dict, settings: Settings
) -> tuple[ExecutionProfile, dict]:
    """Clamp caller-supplied depth knobs onto the mode's profile.

    Repair-not-reject: each value is clamped to ``[floor, options_max_<knob>]``
    and a clamp is logged, never rejected. Both knobs are STRUCTURAL and can only
    be honored where the mode already does that work — ``agent_rounds`` is
    ignored on a no-agent profile and ``graph_chunk_cap`` where the profile reads
    no graph chunks. Returns the (possibly replaced) profile and an
    ``options_applied`` dict: the effective values + any ``ignored``.
    """
    ceilings = {
        "agent_rounds": settings.options_max_agent_rounds,
        "graph_chunk_cap": settings.options_max_graph_chunk_cap,
    }
    changes: dict[str, int] = {}
    ignored: dict[str, str] = {}
    for knob, ceiling in ceilings.items():
        requested = overrides.get(knob)
        if requested is None:
            continue
        # Structural guards: don't let an override create capability a mode disables.
        if knob == "agent_rounds" and profile.agent_rounds == 0:
            ignored[knob] = "mode runs no agent loop"
            continue
        if knob == "graph_chunk_cap" and not profile.graph_text_tools:
            ignored[knob] = "mode reads no graph chunks"
            continue
        clamped = max(_OVERRIDE_FLOORS[knob], min(int(requested), ceiling))
        if clamped != int(requested):
            logger.info(
                "option_clamped", knob=knob, requested=int(requested),
                applied=clamped, ceiling=ceiling,
            )
        changes[knob] = clamped

    new_profile = replace(profile, **changes) if changes else profile
    applied: dict = {knob: getattr(new_profile, knob) for knob in ceilings}
    if ignored:
        applied["ignored"] = ignored
    return new_profile, applied
