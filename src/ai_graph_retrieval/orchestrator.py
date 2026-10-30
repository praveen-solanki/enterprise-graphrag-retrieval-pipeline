"""RetrievalOrchestrator — the request lifecycle in one place.

admission (bounded) → plan (entry classification) → gates → execute → quality
→ synthesis → respond. Each phase is a pipeline stage (see `pipeline/`) with one job and
its own tests; this class only sequences them, owns the request envelope
(admission, logging span, audit/usage in `finally`), and shapes the response.
The emitter receives step/trace/answer events throughout (a no-op for the
non-streaming path).

The MODE (Search / Super Research) is the user's explicit per-query
choice — never selected or changed automatically. One model-driven entry
classification still decides everything WITHIN that choice (conversational
short-circuits, the self-contained rewrite, Search's temporal exception);
there is no post-retrieval escalation. Routing (`pipeline.planning`) is fully
separated from execution (`pipeline.execution`): the planner returns a
decision, the collector runs it.

Every request — success or failure — leaves an audit record (who asked, which
documents were served, at what cost) and a usage record. Both are non-blocking.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import replace
from typing import Any

import structlog

from ai_graph_retrieval.actions.generator import SuggestedActionGenerator
from ai_graph_retrieval.clarification.models import MidResearchClarification
from ai_graph_retrieval.connectors.llm.client import LlmClient
from ai_graph_retrieval.connectors.llm.embeddings import RequestEmbedder
from ai_graph_retrieval.connectors.mongo.conversation_store import ConversationStore
from ai_graph_retrieval.conversation.memory import ConversationMemory
from ai_graph_retrieval.core.access import AccessContext
from ai_graph_retrieval.core.censor import CensorRegistry
from ai_graph_retrieval.core.costing import CostRates, estimate_cost_usd
from ai_graph_retrieval.core.events import MetaEvent
from ai_graph_retrieval.core.exceptions import GraphQueryError, ServiceOverloaded
from ai_graph_retrieval.core.models import (
    CitedSubgraph,
    EvidencePool,
    GraphTraceBuilder,
    Usage,
)
from ai_graph_retrieval.core.progress import Emit, ProgressReporter
from ai_graph_retrieval.core.run_config import RequestRunConfig
from ai_graph_retrieval.graph.repository import CachingGraphRepository, GraphReadRepository
from ai_graph_retrieval.observability import acl_audit
from ai_graph_retrieval.observability.ledger import JsonlLedger
from ai_graph_retrieval.observability.tracing import (
    record_applied_options,
    record_error,
    record_root_io,
    record_scores,
    record_trace_identity,
    retrieve_span,
    set_correlation_id,
    set_span_attributes,
    stage_span,
)
from ai_graph_retrieval.observability.usage_stats import UsageAggregator
from ai_graph_retrieval.pipeline.execution import EvidenceCollector
from ai_graph_retrieval.pipeline.gates import ClarificationGates
from ai_graph_retrieval.pipeline.planning import ExecutionPlanner
from ai_graph_retrieval.pipeline.quality import QualityGate
from ai_graph_retrieval.pipeline.recording import RequestRecorder
from ai_graph_retrieval.pipeline.respond import build_payload
from ai_graph_retrieval.pipeline.state import RequestState
from ai_graph_retrieval.pipeline.synthesis import SynthesisStage
from ai_graph_retrieval.retrieval.base import (
    ConversationalTurn,
    Mode,
    RetrievalStrategy,
    StrategyContext,
)
from ai_graph_retrieval.retrieval.profile import apply_overrides
from ai_graph_retrieval.retrieval.router import QueryRouter
from ai_graph_retrieval.settings import Settings
from ai_graph_retrieval.synthesis.answer_graph import build_answer_graph
from ai_graph_retrieval.synthesis.composer import AnswerComposer

logger = structlog.get_logger(__name__)


class RetrievalOrchestrator:
    def __init__(
        self,
        settings: Settings,
        repo: GraphReadRepository,
        llm: LlmClient,
        router: QueryRouter,
        composer: AnswerComposer,
        censor: CensorRegistry,
        usage_ledger: JsonlLedger,
        audit_ledger: JsonlLedger,
        # Search Mode's lane: a bounded agent loop over the graph.
        search_strategy: RetrievalStrategy,
        # Super Research's lane — the full agent loop.
        research_strategy: RetrievalStrategy,
        usage_stats: UsageAggregator | None = None,
        action_generator: SuggestedActionGenerator | None = None,
        conversation_memory: ConversationMemory | None = None,
        # Durable conversation persistence in Mongo (None ⇒ off). Distinct from
        # conversation_memory (ephemeral Redis history for follow-ups): this writes
        # every completed turn so a client can read a result back by id.
        conversation_store: ConversationStore | None = None,
    ) -> None:
        self._s = settings
        self._repo = repo
        self._llm = llm
        self._rates = CostRates.from_settings(settings)
        self._admission = asyncio.Semaphore(settings.max_concurrent_requests)
        # Server-side conversation memory (None ⇒ client-replayed history only).
        self._memory = conversation_memory
        self._store = conversation_store
        # The pipeline, one stage per phase (see pipeline/__init__.py).
        self._planner = ExecutionPlanner(
            router, settings,
            search_strategy=search_strategy, research_strategy=research_strategy,
        )
        self._gates = ClarificationGates(settings)
        self._collector = EvidenceCollector(settings)
        self._quality = QualityGate(censor, settings)
        self._synthesis = SynthesisStage(settings, composer, action_generator)
        self._recorder = RequestRecorder(
            usage_ledger, audit_ledger, usage_stats, self._rates
        )

    async def retrieve(
        self,
        *,
        ctx: AccessContext,
        question: str,
        history: list[dict],
        mode: Mode,
        include_trace: bool,
        emit: Emit,
        plan_review: bool = True,
        conversation_id: str | None = None,
        run_config: RequestRunConfig | None = None,
        user_message_id: str | None = None,
        ai_message_id: str | None = None,
        owner: str | None = None,
    ) -> dict:
        """Run one retrieval end to end; returns the final response payload.

        ``plan_review`` — allow the pre-retrieval clarification pause. The
        /resume turn passes False so answering never re-triggers it.

        ``run_config`` — per-request RunConfig (resolved graph sources +
        embedding space + depth-knob overrides). None ⇒ boot-time
        graph_sources() + the mode's default profile (today's behavior).

        ``conversation_id`` — the chat thread, when the caller has one. With no
        client-supplied ``history``, recent turns are loaded from server-side
        memory so follow-ups resolve; every answered turn is recorded back.
        Client history always wins over the store (backward compatible).

        ``user_message_id`` / ``ai_message_id`` — the ids the streaming route
        already hands the client. Threaded through so the durable
        conversation store keys the persisted turn on the SAME ai_message_id the
        client holds, which is what a read-back looks up. Absent (the
        plain callers) ⇒ generated here, and returned in the payload so those
        callers can read the turn back too. ``owner`` is the calling user id (for
        the session's user scoping); None keeps the row tenant-scoped only.
        """
        user_message_id = user_message_id or str(uuid.uuid4())
        ai_message_id = ai_message_id or str(uuid.uuid4())
        # A turn always belongs to a conversation: mint one when the caller didn't
        # supply it (a placeholder like "string" is treated as real — the client
        # controls the thread). Returned in the payload so the caller can continue
        # the thread and poll it. The streaming routes pass their own id so the
        # stream and the persisted turn share it; this `or` no-ops for them.
        conversation_id = conversation_id or str(uuid.uuid4())
        try:
            await asyncio.wait_for(
                self._admission.acquire(), timeout=self._s.admission_timeout_s
            )
        except TimeoutError as exc:
            logger.warning("request_rejected_overloaded", **ctx.log_fields())
            raise ServiceOverloaded() from exc

        usage = Usage()
        state = RequestState(mode=mode.value)
        outcome = "ok"
        started = time.monotonic()
        acl_audit.begin()  # 🔒 collect per-node ACL checks → one summary at request end
        logger.info(
            "request_received", mode=mode.value, question_chars=len(question),
            history_turns=len(history), plan_review=plan_review,
            principals=len(ctx.principals or ()), **ctx.log_fields(),
        )
        # Span NAME carries the mode, so a trace list is readable without opening
        # each span to check an attribute. It was hardcoded "research_mode/stream"
        # for every request in every mode, which made the one field a trace UI
        # groups and filters by useless — every request looked like research.
        # Bind the correlation id BEFORE the root span opens, so every span the
        # turn creates — including the root — carries it. It is the
        # request id: unique per query, already the join key into the logs, and
        # already on every log line. A paused turn and its resume are two
        # requests, so they get two ids; the SESSION (conversation_id) is what
        # ties those together.
        set_correlation_id(ctx.request_id)
        with retrieve_span(f"retrieve/{mode.value}") as span:
            set_span_attributes(
                span, tenant_id=ctx.tenant_id, request_id=ctx.request_id,
                principal_count=len(ctx.principals or ()), mode=mode.value,
                question_chars=len(question), plan_review=plan_review,
            )
            # Trace grouping fields: the conversation is a
            # SESSION, so a clarification pause and its resume read as one
            # conversation instead of two unrelated traces; the caller is a
            # user; mode is a filterable tag from the start (intent/strategy/
            # outcome join it at close, once they are known).
            # ``user`` is the TENANT: that is the caller identity this context
            # actually carries. Per-person attribution would mean threading the
            # request's uid through AccessContext, which is a deliberate,
            # separate decision about putting a person id in traces.
            record_trace_identity(
                span, session_id=conversation_id, user_id=ctx.tenant_id,
                tags=[f"mode:{mode.value}"],
            )
            # The trace's top-level Input: the question, readable from the
            # trace list without opening a child span.
            record_root_io(span, input=question)
            try:
                # Durable persistence: create this turn's records UP FRONT (user +
                # ai_response at status=pending) so a read-back can see an
                # in-flight turn and wait instead of re-running it. Best-effort.
                if conversation_id and self._store is not None:
                    await self._store.open_turn(
                        tenant_id=ctx.tenant_id,
                        conversation_id=conversation_id,
                        user_id=owner or "",
                        user_message_id=user_message_id,
                        ai_message_id=ai_message_id,
                        question=question,
                        mode=mode.value,
                    )
                # Server-side conversation memory: fill in history ONLY when the
                # caller sent none — client-replayed history is authoritative.
                # Fail-open ([]); inside this try so the admission permit is
                # released (finally below) even if recall raises unexpectedly.
                if not history and conversation_id and self._memory and self._memory.enabled:
                    history = await self._memory.history(ctx, conversation_id)
                    if history:
                        logger.info("conversation_memory_recalled",
                                    history_turns=len(history), **ctx.log_fields())
                payload = await self._retrieve_admitted(
                    ctx=ctx, question=question, history=history, mode=mode,
                    include_trace=include_trace, emit=emit,
                    usage=usage, state=state, plan_review=plan_review,
                    run_config=run_config or RequestRunConfig(),
                    span=span, ai_message_id=ai_message_id,
                )
                # Stamp the turn's ids so every caller (streaming or not) gets them
                # back and can continue/poll the thread by id.
                payload["conversation_id"] = conversation_id
                payload["ai_message_id"] = ai_message_id
                payload["user_message_id"] = user_message_id
                # Record the completed exchange so the NEXT turn's follow-up
                # resolves. Best-effort (never raises); paused/failed turns are
                # deliberately not recorded — nothing was answered.
                if conversation_id and self._memory is not None:
                    await self._memory.append(
                        ctx, conversation_id, question, payload.get("answer") or ""
                    )
                # Durable persistence: the turn produced an answer — flip its
                # ai_response doc to completed and attach the full payload (what a
                # read-back returns). Best-effort.
                if conversation_id and self._store is not None:
                    await self._store.complete_turn(ai_message_id, payload)
                ev = payload.get("evidence") or {}
                up = payload.get("usage") or {}
                logger.info(
                    "request_completed", outcome=outcome, intent=state.intent,
                    strategy=state.strategy, duration_ms=int((time.monotonic() - started) * 1000),
                    answer_chars=len(payload.get("answer") or ""),
                    citations=len(payload.get("citations") or []),
                    entities=len(ev.get("entities") or []), facts=len(ev.get("facts") or []),
                    quotes=len(ev.get("quotes") or []), claims=len(ev.get("claims") or []),
                    llm_calls=up.get("llm_calls"), cost_usd=up.get("cost_usd"),
                    **ctx.log_fields(),
                )
                set_span_attributes(
                    span, llm_calls=up.get("llm_calls"), cost_usd=up.get("cost_usd"),
                    answer_chars=len(payload.get("answer") or ""),
                    citations=len(payload.get("citations") or []),
                )
                # Quality SIGNALS as trace scores — the service
                # already computes every one of these; as scores they become
                # trend lines ("% of runs shipping zero citations") instead of
                # numbers someone has to grep for.
                record_scores(span, {
                    # Named for what the flag MEASURES (evidence presence), not
                    # for the response field it is read from: as an answer-quality
                    # score `answer_found` read True on declining answers too.
                    "evidence_found": bool(payload.get("answer_found")),
                    "citations": len(payload.get("citations") or []),
                    "partial": bool(payload.get("partial")),
                    "suggested_actions": len(payload.get("suggested_actions") or []),
                    "llm_calls": up.get("llm_calls") or 0,
                    "cost_usd": up.get("cost_usd") or 0.0,
                })
                # ...and the trace's top-level Output: the answer (bounded).
                record_root_io(span, output=payload.get("answer") or "")
                return payload
            except MidResearchClarification as pause:
                # Not a failure: the pipeline paused to ask the user (the entry
                # classifier's clarification, or the agent's mid-research tool).
                # The route turns it into a clarify response; no answer was
                # synthesized this turn.
                outcome = "clarification"
                record_root_io(span, output="awaiting clarification — paused to "
                                            "ask the user; no answer this turn")
                logger.info("request_paused", kind=type(pause).__name__, intent=state.intent,
                            duration_ms=int((time.monotonic() - started) * 1000),
                            **ctx.log_fields())
                # The up-front records exist but no answer was produced — mark the
                # ai doc terminal so the fallback re-runs (re-raising the pause)
                # instead of waiting on it forever. Best-effort.
                if conversation_id and self._store is not None:
                    await self._store.set_status(ai_message_id, "clarification")
                raise
            except GraphQueryError as exc:
                outcome = exc.code
                # Mark the span failed too: a trace whose only symptom
                # was "no output" was indistinguishable from a success, so
                # "how many requests failed?" was unanswerable from traces alone.
                record_error(span, exc, code=exc.code)
                # Log the FULL failure, not just the code: the safe_message (which
                # stage — "Query planning timed out" vs "Synthesis timed out" vs a
                # store outage), the exception type, and the underlying cause chain.
                # A server-side failure (5xx) also carries the traceback (exc_info)
                # so the root exception — the one wrapped by `raise … from` — is in
                # the logs, not swallowed. Client faults (4xx) log without a stack.
                logger.error(
                    "request_failed",
                    outcome=outcome,
                    code=exc.code,
                    http_status=exc.http_status,
                    error=exc.safe_message,
                    error_type=type(exc).__name__,
                    cause=repr(exc.__cause__) if exc.__cause__ is not None else None,
                    duration_ms=int((time.monotonic() - started) * 1000),
                    exc_info=exc if exc.http_status >= 500 else None,
                    **ctx.log_fields(),
                )
                if conversation_id and self._store is not None:
                    await self._store.set_status(
                        ai_message_id, "failed", error_code=exc.code,
                        error_message=exc.safe_message,
                    )
                raise
            except asyncio.CancelledError as exc:  # client disconnected mid-request
                outcome = "cancelled"
                record_error(span, exc, code=outcome)
                logger.warning("request_cancelled",
                               duration_ms=int((time.monotonic() - started) * 1000),
                               **ctx.log_fields())
                raise
            except Exception as exc:
                outcome = "internal_error"
                record_error(span, exc, code=outcome)
                logger.exception("request_failed_unhandled", **ctx.log_fields())
                if conversation_id and self._store is not None:
                    await self._store.set_status(
                        ai_message_id, "failed", error_code=outcome,
                        error_message="Internal error",
                    )
                raise
            finally:
                set_span_attributes(span, outcome=outcome, intent=state.intent,
                                    strategy=state.strategy)
                # Same three facts as TAGS: attributes are searchable, tags are
                # what the trace list filters by.
                record_trace_identity(span, tags=[
                    f"mode:{mode.value}", f"outcome:{outcome}",
                    *([f"intent:{state.intent}"] if state.intent else []),
                    *([f"strategy:{state.strategy}"] if state.strategy else []),
                ])
                acl_audit.log_summary(ctx)  # 🔒 one ACL summary per request (all exit paths)
                self._admission.release()
                self._recorder.record(ctx, question, state, usage, outcome)

    async def _cited_subgraph(self, repo, ctx, citations) -> CitedSubgraph | None:
        """The graph's own record of the cited documents (mentions + extracted
        relationships), for the answer's graph. Never the request's failure: a
        read error or an empty citation list yields None and the graph is built
        from the pool alone. Traced as ``stage.answer_graph`` so a graph that
        came back bare can be told apart from a read that never ran."""
        doc_ids = [s.doc_id for s in citations if s.doc_id]
        with stage_span("stage.answer_graph", citations=len(doc_ids)) as span:
            if not doc_ids:
                set_span_attributes(span, skipped="no_citations")
                return None
            try:
                cited = await repo.cited_subgraph(ctx, doc_ids)
            except Exception as exc:  # additive: the graph loses its grounding, not the answer
                logger.warning("answer_graph_cited_read_failed",
                               error_type=type(exc).__name__, error=str(exc)[:200],
                               **ctx.log_fields())
                set_span_attributes(span, skipped="read_failed")
                return None
            set_span_attributes(span, cited_entities=len(cited.entities),
                                cited_facts=len(cited.facts), cited_chunks=len(cited.chunk_ids))
            return cited

    # ── the admitted request: the pipeline stages in order ───────────────────

    async def _retrieve_admitted(
        self,
        *,
        ctx: AccessContext,
        question: str,
        history: list[dict],
        mode: Mode,
        include_trace: bool,
        emit: Emit,
        usage: Usage,
        state: RequestState,
        plan_review: bool = True,
        run_config: RequestRunConfig | None = None,
        span: Any = None,
        ai_message_id: str | None = None,
    ) -> dict:
        run_config = run_config or RequestRunConfig()
        evidence = EvidencePool()
        trace = GraphTraceBuilder()
        progress = ProgressReporter(
            emit, trace, answer_delta_max_chars=self._s.answer_delta_max_chars,
            trace_delta_max_bytes=self._s.trace_delta_max_bytes,
        )
        history = history[-self._s.history_max_turns :]

        # Per-request settings overrides, resolved UP FRONT so they reach the
        # entry classifier (short-circuit) AND the rest of the request.
        req_settings, req_llm = self._s, self._llm
        val = run_config.option_overrides.get("chat_smalltalk_shortcircuit")
        if val is not None:
            req_settings = self._s.model_copy(
                update={"chat_smalltalk_shortcircuit": bool(val)})

        # 1. Plan: the entry call — conversational triage (greeting/guardrail/
        #    out-of-scope short-circuit with a model-written reply) and the
        #    self-contained rewrite of an elliptical follow-up. The MODE picks
        #    the strategy; the classifier's intent is telemetry.
        with stage_span("stage.entry_classification", mode=mode.value):
            outcome = await self._planner.resolve(
                question, history, mode, usage, llm=req_llm, settings=req_settings,
            )
        if isinstance(outcome, ConversationalTurn):
            return await self._conversational_response(ctx, outcome, state, usage, progress)
        decision = outcome
        plan = decision.plan
        # Original question is kept for audit + synthesis voice; retrieval +
        # synthesis run on the classifier's self-contained query — it resolves
        # references the same way for both search and the answer voice.
        search_query = plan.search_query or question
        if search_query != question:
            logger.info("query_contextualized", **ctx.log_fields())
        state.intent = plan.intent.value
        state.strategy = decision.strategy.name
        # Per-request RunConfig: clamp the caller's depth-knob overrides onto the
        # mode's profile (repair-not-reject) BEFORE anything reads it, so the log
        # and every stage see the effective values. `options_applied` is echoed on
        # the response; `run` records which graph sources were resolved / fell back.
        profile, options_applied = apply_overrides(
            decision.profile, run_config.option_overrides, self._s
        )
        # Sync the overridden profile back onto the decision so EVERY stage reads
        # one source of truth: the collector reads decision.profile, the
        # agent/composer read sc.profile — they must be
        # the SAME object or an override lands on one path but not the other.
        decision = replace(decision, profile=profile)
        options_applied = {**options_applied, "run": run_config.resolved_meta}
        # Echo the effective per-request ACL decision (from options.disabled_acl,
        # carried on ctx.acl_enforced; None ⇒ settings default) so it's visible
        # alongside the other applied options.
        acl_enforced_eff = (
            ctx.acl_enforced if ctx.acl_enforced is not None else self._s.acl_enforcement_enabled
        )
        options_applied["disabled_acl"] = not acl_enforced_eff
        logger.info(
            "retrieval_routed",
            intent=plan.intent.value,
            strategy=decision.strategy.name,
            # The depth profile that ran — the caller's mode's profile with any
            # per-request overrides applied; logged so the two stay auditable.
            profile=profile.name,
            agent_rounds=profile.agent_rounds,
            options_applied=options_applied,
            rationale=plan.rationale,
            **ctx.log_fields(),
        )
        await emit(MetaEvent(request_id=ctx.request_id,
                             intent=plan.intent.value, strategy=decision.strategy.name))

        # 1a. Gate: the conditional pre-retrieval clarification pause. Raises;
        # the route layer answers. No LLM spent here: the entry classification
        # above already judged the ambiguity and wrote any questions; the gate
        # only decides whether this MODE may pause on them (resume turns never
        # re-pause).
        self._gates.enforce(ctx, plan=plan, mode=mode, plan_review=plan_review)

        # Per-request run selection: when the request resolved its own graph
        # sources (connectorType+runIdSuffix → run_registry), read through a repo
        # scoped to that run's Neo4j DB + Qdrant collections; else the boot-time
        # graph_sources(). Same connections either way.
        base_repo = (
            self._repo.with_sources(list(run_config.sources))
            if run_config.sources else self._repo
        )
        # Per-request status gate (people graph only): people the graphs record
        # as no longer active are dropped IN CYPHER, so they never spend a row of
        # a ranked read's LIMIT. Opened only for a question the classifier read
        # as being about a period — its past people are the answer — and by the
        # kill switch. See retrieval/person_status.py.
        base_repo = base_repo.including_inactive_people(
            plan.historical or not req_settings.inactive_people_filter_enabled
            or not req_settings.people_graph_enabled
        )
        options_applied["chat_smalltalk_shortcircuit"] = req_settings.chat_smalltalk_shortcircuit
        # Third surface for the same dict: the root retrieve span, so a
        # trace is self-describing about the knobs the run executed with.
        record_applied_options(span, options_applied)
        # Message-EMBEDDED effective options (survives message-only log pipelines):
        # the depth knobs + toggles ACTUALLY applied after clamping/structural guards,
        # plus the resolved run selection — the single source of truth for "what
        # options took effect", mirroring options_applied echoed on the response.
        logger.info("options_applied "
                    + json.dumps(options_applied, ensure_ascii=False, default=str),
                    **ctx.log_fields())
        sc = StrategyContext(
            ctx=ctx, question=search_query, history=history, settings=req_settings,
            # Per-request read-through cache: dedupes repeated graph reads within
            # this request (esp. the research agent) without crossing requests.
            repo=CachingGraphRepository(base_repo), llm=req_llm,
            # Embed in the resolved run's space (dimension must match its
            # collection); None space ⇒ the configured default embedder.
            embedder=RequestEmbedder(req_llm, usage, space=run_config.embedding_space),
            evidence=evidence, progress=progress, usage=usage,
            # The profile the planner resolved, with per-request overrides applied:
            # every depth cap the strategy and the agent's tools apply is read from here.
            profile=profile,
            # The asked window the classifier resolved (empty = none): the
            # agent gathers within it and synthesis is held to it.
            time_scope=(plan.time_scope_from, plan.time_scope_to),
            # The classifier's reading of WHEN the question is about, so the
            # employment gate's pool-side half agrees with the Cypher half above.
            historical=plan.historical,
        )

        # 2. Execute the decision: the planned lane, under ONE deadline;
        #    partial evidence on timeout still reaches synthesis.
        state.partial = await self._collector.collect(sc, decision)

        # 3. Quality gate: censor → contradictions → trust ranking → excerpt
        #    ranking. The excerpt ranking reads the `relevance` already on each
        #    quote — retrieval/chunk_relevance scored the graph's chunks against
        #    the question during retrieval — so nothing is embedded or fetched
        #    again here.
        snapshot = self._quality.curate(
            evidence, trust_ranking=profile.trust_ranking,
            # The classifier resolved a window ⇒ the question stated its own
            # time scope and recency must not be counted against it twice.
            time_scoped=bool(plan.time_scope_from and plan.time_scope_to),
        )

        # 4. Synthesis (streamed) + suggested actions, on their own budget. The
        #    caller's mode sets the answer depth.
        # Durable persistence: retrieval is done, the answer is about to stream —
        # flip the stored turn to `generating` so a read-back poll reports it.
        # Best-effort. Short-circuit/decline paths return before here and go
        # straight pending → completed, which is fine.
        if ai_message_id and self._store is not None:
            await self._store.mark_generating(ai_message_id)
        composed, actions, evidence_found = await self._synthesis.run(
            # research selects the research-grade completion budget. Keyed on the
            # MODE (the user's depth choice), not the classifier's intent — the
            # intent is telemetry and must not buy a bigger answer budget.
            sc, snapshot, research=mode is Mode.SUPER_RESEARCH,
            strategy=decision.strategy.name, mode=mode,
            # The classifier's shape verdict, read off the question before
            # retrieval — carried as data the same way the time window is, so
            # synthesis applies it rather than re-deriving a shape per run.
            answer_shape=plan.answer_shape,
            # The excerpts arrive relevance-ordered only when the retrieval pass
            # scored them; otherwise the digest keeps its document-order sort.
            quote_ranked=(any(q.relevance for q in snapshot.quotes)
                          and self._s.quote_ranking_enabled),
        )

        # 5. Assemble the response payload.
        report = usage.snapshot()
        report.cost_usd = estimate_cost_usd(report, self._rates)
        state.usage = report
        state.evidence = snapshot
        # Message-EMBEDDED result summary ("what came back"): the deduped evidence
        # gathered, by kind, + whether synthesis judged the answer found and how long
        # it was. Reads at a glance whether retrieval found anything to answer from.
        logger.info(
            "📦 evidence " + json.dumps(
                {"entities": len(snapshot.entities), "facts": len(snapshot.facts),
                 "claims": len(snapshot.claims), "sources": len(snapshot.sources),
                 "quotes": len(snapshot.quotes), "contradictions": len(snapshot.contradictions),
                 "evidence_found": evidence_found,
                 "answer_chars": len(composed.text or "")},
                ensure_ascii=False),
            **ctx.log_fields(),
        )

        # The answer's graph: what the cited documents' own passages mention and
        # state, joined with what the answer names and what the researcher
        # traversed (synthesis/answer_graph.py). Read AFTER synthesis because it
        # is keyed on the citations the answer actually made; additive — a
        # failed read leaves the graph built from the pool alone.
        cited_subgraph = await self._cited_subgraph(base_repo, ctx, composed.citations)

        payload = build_payload(
            ctx, state, answer=composed.text, evidence=snapshot, usage_report=report,
            duration_ms=usage.elapsed_ms(),
            trace=trace.snapshot().model_dump() if include_trace else None,
            citations=composed.citations,
            suggested_actions=actions,
            # The response field keeps its name for client compatibility; the
            # signal behind it is evidence presence.
            answer_found=evidence_found,
            answer_graph=build_answer_graph(
                composed.text, snapshot, composed.citations, cited_subgraph),
        )
        # Echo the effective RunConfig (post-clamp knobs + resolved/fallback graph
        # sources) so a caller can detect silent clamps or a run-lookup fallback.
        payload["options_applied"] = options_applied
        # End-of-request: the databases this request actually read from, as ONE
        # JSON object embedded in the message string (some log pipelines surface
        # only `message` and drop bound kwargs). ``mongo_db`` names the
        # run_registry database ONLY when it was really consulted — a source
        # `selected_by` its boot config never touched Mongo, and naming it there
        # would suggest a lookup that never happened. Each source lists the Neo4j
        # database + Qdrant collection it read.
        eff_sources = list(run_config.sources) if run_config.sources else self._s.graph_sources()
        meta_by_name = {m["name"]: m for m in (run_config.resolved_meta.get("sources") or [])}
        from_registry = any(
            m.get("selected_by") in ("suffix", "latest") for m in meta_by_name.values()
        )
        db_summary = {
            "run_selection": run_config.resolved_meta.get("run_selection"),
            # The registry db that resolved the run(s); None when no lookup happened.
            "mongo_db": run_config.resolved_meta.get("mongo_db") if from_registry else None,
            "databases": [
                {
                    "connector": g.name,
                    "neo4j_db": g.database,
                    "qdrant_collection": g.entities_collection,
                    "run_suffix": meta_by_name.get(g.name, {}).get("run_suffix"),
                    "selected_by": meta_by_name.get(g.name, {}).get("selected_by"),
                }
                for g in eff_sources
            ],
        }
        logger.info("🗄️ request_databases " + json.dumps(db_summary, ensure_ascii=False))
        logger.info(
            "retrieval_completed",
            strategy=decision.strategy.name,
            escalated=state.escalated,
            duration_ms=payload["duration_ms"],
            llm_calls=report.llm_calls,
            cost_usd=report.cost_usd,
            entities=len(snapshot.entities),
            facts=len(snapshot.facts),
            suggested_actions=len(payload["suggested_actions"]),
            **ctx.log_fields(),
        )
        return payload

    # ── conversational short-circuit ─────────────────────────────────────────

    async def _conversational_response(
        self,
        ctx: AccessContext,
        turn: ConversationalTurn,
        state: RequestState,
        usage: Usage,
        progress: ProgressReporter,
    ) -> dict:
        """The entry classifier's model-written reply, with no retrieval and no
        heavy synthesis model. Still audited/metered via the normal `retrieve()`
        finally (state is populated here)."""
        answer = turn.reply
        empty = EvidencePool().snapshot()
        report = usage.snapshot()
        report.cost_usd = estimate_cost_usd(report, self._rates)
        state.intent = state.strategy = turn.category
        state.usage = report
        state.evidence = empty
        logger.info("conversational_shortcircuit", category=turn.category,
                    **ctx.log_fields())
        # Stream the reply as one delta so the SSE path renders it like any answer.
        await progress.answer_delta(answer)
        return build_payload(
            ctx, state, answer=answer, evidence=empty,
            usage_report=report, duration_ms=usage.elapsed_ms(),
        )
