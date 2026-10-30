"""Deep-research agent runtime — the OpenAI Agents SDK loop (via LiteLLM).

The SDK owns the reasoning/tool-selection loop (`Runner.run`); we own everything
that matters for correctness and safety:
  * the model is the configured heavy chat model on the OpenAI-compatible
    endpoint (NIM, vLLM, …) via the SDK's ``LitellmModel`` provider;
  * tools are ACL-bound via the injected ``ResearchContext`` and gather into the
    request's evidence pool;
  * the final answer is NOT the agent's output — it is composed downstream by the
    synthesis pass from the ACL-filtered evidence pool. The agent's text is
    discarded; only the evidence it gathered matters;
  * the SDK's own token usage is folded back into our per-request meter so cost
    accounting stays exact (the POC's D8 gap).

Tracing is disabled (the SDK's default exporter needs an OpenAI key and would
egress traces).
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

import structlog
from agents import Agent, ModelSettings, RunHooks, Runner, set_tracing_disabled
from agents.exceptions import MaxTurnsExceeded
from agents.extensions.models.litellm_model import LitellmModel

from ai_graph_retrieval.agent.prompts import agent_input, build_agent_system_prompt
from ai_graph_retrieval.agent.tools import ResearchContext, research_tools
from ai_graph_retrieval.clarification.models import MidResearchClarification
from ai_graph_retrieval.connectors.llm.client import (
    litellm_model_name,
    trace_model_params,
)
from ai_graph_retrieval.core.costing import CostRates, call_cost_usd
from ai_graph_retrieval.core.exceptions import LlmUnavailable
from ai_graph_retrieval.core.models import EvidencePool, Usage
from ai_graph_retrieval.observability.tracing import (
    end_span,
    record_llm,
    start_llm_span,
    start_stage_span,
)
from ai_graph_retrieval.retrieval.base import StrategyContext
from ai_graph_retrieval.retrieval.neighbor_scoring import NeighborScorer
from ai_graph_retrieval.settings import Settings

logger = structlog.get_logger(__name__)

set_tracing_disabled(True)  # no OpenAI trace egress; process-wide, idempotent


class _StalePatienceReached(Exception):
    """Control signal: the agent went ``patience`` consecutive rounds without adding
    any new evidence, so more rounds won't help — stop and synthesize on what we have.
    Carries the run usage (Runner.run didn't return normally) so cost stays exact."""

    def __init__(self, usage, stale_rounds: int) -> None:
        super().__init__(f"no new evidence for {stale_rounds} consecutive rounds")
        self.usage = usage
        self.stale_rounds = stale_rounds


def _response_output_text(response: Any) -> str | None:
    """The round's Output for the trace: assistant text plus a compact rendering
    of the tool calls it issued (name + arguments). Mirrors LlmClient's
    ``_response_text`` contract — a pure tool-call round must not trace as empty,
    because choosing tools IS what a research round produces. Defensive getattr
    walk: the SDK's output item types vary, and tracing never fails a run."""
    try:
        parts: list[str] = []
        for item in getattr(response, "output", None) or []:
            for c in getattr(item, "content", None) or []:
                text = getattr(c, "text", None)
                if text:
                    parts.append(text)
            name = getattr(item, "name", None)
            if name:  # a tool call item
                args = getattr(item, "arguments", None)
                parts.append(f"[tool call] {name}({args or ''})")
        return "\n".join(parts) or None
    except Exception:  # pragma: no cover - observability must never break a run
        return None


class _AgentRunHooks(RunHooks):
    """Two per-round jobs the SDK gives us no other place to do.

    1. STALE PATIENCE. The SDK's only native bounds are the model stopping on its own
       and ``max_turns``, so a model that keeps calling tools which surface nothing new
       would otherwise burn the full round budget (the dominant Deep Research latency +
       cost sink). We watch the dedup'd evidence-pool size at the start of each round;
       once it hasn't grown for ``patience`` rounds we raise BEFORE the model call,
       skipping the wasted round. The raise propagates cleanly out of Runner.run
       (verified: hook exceptions are not swallowed like tool exceptions).

    2. OBSERVABILITY. The agent's model calls bypass ``LlmClient``, so they
       produced no LLM span at all: the loop was invisible in Langfuse even though its
       tokens reached the request total, and one whole run counted as a single
       ``llm_calls``. Here each round gets its own span with model/tokens/cost AND the
       round's input/output, and its
       usage is metered as it happens — so the trace shows N generations for N rounds
       and the call count is truthful.
    """

    def __init__(
        self,
        evidence: EvidencePool,
        patience: int,
        *,
        usage: Usage,
        deployment: str,
        rates: CostRates,
        provider: str,
        progress: Any = None,
        model_parameters: dict[str, Any] | None = None,
    ) -> None:
        self._evidence = evidence
        self._patience = max(1, patience)
        self._last_size: int | None = None
        self._stale = 0
        self._usage = usage
        self._deployment = deployment
        self._rates = rates
        self._provider = provider
        # The settings every round of this run executes under. Built
        # once by ResearchAgent from the ModelSettings the SDK is handed, so the
        # trace reports what was SENT rather than a value re-derived here.
        self._model_parameters = model_parameters or {}
        self._span: Any = None
        self._round_input: str | None = None
        self.rounds_metered = 0
        # the ROUND span — a parent for the round's generation AND for
        # the tool calls that generation chose. It cannot be context-managed:
        # the model call and the tool executions arrive as separate SDK
        # callbacks, so the span opens on `on_llm_start` and closes when the
        # next round opens or the run ends. `progress` is where tool spans read
        # their parent from (ProgressReporter.span_parent).
        self._round_span: Any = None
        self._round_no = 0
        self._progress = progress

    async def on_llm_start(self, context, agent, system_prompt, input_items) -> None:
        size = self._evidence.size()
        # First round has no prior to compare against; after that, an unchanged size
        # means the previous round added nothing new.
        if self._last_size is not None and size == self._last_size:
            self._stale += 1
        else:
            self._stale = 0
        self._last_size = size
        if self._stale >= self._patience:
            raise _StalePatienceReached(context.usage, self._stale)
        # A previous round that ended abnormally (no on_llm_end) would leave its span
        # open; close it before starting the next so spans never nest by accident.
        self._close_span()
        # Open THIS round's parent span. It closes when the next round opens (or at
        # the run's end), so it spans the model call plus every tool that call
        # chose — which is what makes the waterfall read as
        # "round → LLM → tool → tool" instead of one flat list.
        self._open_round_span()
        self._span = start_llm_span("llm.chat.agent", parent=self._round_span)
        # The round's Input for the trace. Captured here because on_llm_end does not
        # receive it; the OTEL SDK truncates past span_attribute_value_length_limit,
        # so no manual cap. Serialization is best-effort — tracing never fails a run.
        try:
            self._round_input = json.dumps(
                {"system": system_prompt, "input": input_items},
                ensure_ascii=False, default=str,
            )
        except Exception:  # pragma: no cover - defensive; str() below always works
            self._round_input = str(input_items)

    async def on_llm_end(self, context, agent, response) -> None:
        u = getattr(response, "usage", None)
        prompt = int(getattr(u, "input_tokens", 0) or 0) if u else 0
        completion = int(getattr(u, "output_tokens", 0) or 0) if u else 0
        cached = (
            int(getattr(getattr(u, "input_tokens_details", None), "cached_tokens", 0) or 0)
            if u else 0
        )
        if u is not None:
            # Meter per ROUND: one add_llm per model call, so llm_calls counts calls
            # rather than counting a whole multi-round run as one.
            self._usage.add_llm(
                prompt, completion,
                deployment=self._deployment,
                cached_prompt_tokens=cached,
            )
            self.rounds_metered += 1
        if self._span is not None:
            cost_in, cost_out, cost_total = call_cost_usd(
                self._rates, self._deployment,
                prompt_tokens=prompt, completion_tokens=completion,
                cached_prompt_tokens=cached,
            )
            record_llm(
                self._span, model=self._deployment, provider=self._provider,
                request_type="agent", tokens_prompt=prompt, tokens_completion=completion,
                tokens_cached=cached,
                cost_input=cost_in, cost_output=cost_out, cost_total=cost_total,
                # Input/Output for the Langfuse observation, matching every
                # LlmClient span: the round's full model input, and its output —
                # assistant text and/or the tool calls it chose (a research round
                # usually "answers" with tool calls, so those ARE the output).
                request_messages=self._round_input,
                response_choices=_response_output_text(response),
                # The agent bypasses LlmClient, so its rounds carry their settings here.
                model_parameters=self._model_parameters,
            )
        self._close_span()

    def _open_round_span(self) -> None:
        """Close the previous round's span and open the next one, pointing the
        progress reporter's tool spans at it. Best-effort: a tracing problem
        must never interrupt the loop."""
        try:
            self._close_round_span()
            self._round_no += 1
            self._round_span = start_stage_span(
                f"agent.round.{self._round_no}", round_number=self._round_no,
            )
            if self._progress is not None:
                self._progress.span_parent = self._round_span
        except Exception:  # pragma: no cover - defensive
            logger.warning("round_span_open_failed", exc_info=True)

    def _close_round_span(self) -> None:
        """End the current round span (no-op when there is none) and stop
        parenting tool spans to it — the post-loop passes belong to their own
        stages, not to the last round."""
        if self._round_span is None:
            return
        end_span(self._round_span)
        self._round_span = None
        if self._progress is not None:
            self._progress.span_parent = None

    def _close_span(self) -> None:
        if self._span is not None:
            end_span(self._span)
            self._span = None

    def close(self) -> None:
        """Release any span still open (the run raised mid-round, hit the turn
        cap, or stopped on patience) — the generation AND the round span."""
        self._close_span()
        self._close_round_span()


def _already_found(sc: StrategyContext, *, names: int = 12, titles: int = 8) -> str:
    """A compact inventory of what the pool holds BEFORE the loop starts.

    Names and titles only, hard-capped: the point is to stop the agent
    re-searching, not to re-send the evidence. The full text is already going to
    synthesis, and duplicating it here would pay for the same tokens twice on
    every round — the agent's context is re-sent in full each time.
    """
    ev = sc.evidence.snapshot()
    if not (ev.entities or ev.sources):
        return ""
    lines = []
    if ev.entities:
        shown = [e.name for e in ev.entities[:names]]
        more = len(ev.entities) - len(shown)
        lines.append(f"- entities: {', '.join(shown)}"
                     + (f" (+{more} more)" if more > 0 else ""))
    if ev.sources:
        shown = [s.title for s in ev.sources[:titles]]
        more = len(ev.sources) - len(shown)
        lines.append(f"- documents: {'; '.join(shown)}"
                     + (f" (+{more} more)" if more > 0 else ""))
    if ev.quotes:
        lines.append(f"- {len(ev.quotes)} chunks already retrieved from those documents")
    return "\n".join(lines)


class ResearchAgent:
    """Builds the SDK agent once and drives one research run per request."""

    def __init__(
        self, settings: Settings, *, clarify: bool = True, graph_text: bool = True,
        research: bool = False,
    ) -> None:
        """``clarify`` — offer the mid-research clarification tool (subject to the
        feature flag). False for Search: its clarification chances already passed
        in the pre-retrieval gate, and pausing mid-flight discards the run's
        evidence. ``graph_text`` — offer the tools whose return is graph
        chunk/document text (profile.graph_text_tools and a non-zero chunk cap).
        ``research`` — the FULL agent lane (Super Research), which appends the
        depth block to the instructions: that lane is given several times the
        rounds of the light one, and its answer is expected to be correspondingly
        deeper, so its instructions have to say what the extra rounds are for."""
        self._s = settings
        self._graph_text = graph_text
        # Same rate table as LlmClient and the per-request UsageReport, so a round's
        # span cost can never disagree with the request total.
        self._rates = CostRates.from_settings(settings)
        # The heavy chat model, on the same OpenAI-compatible endpoint LlmClient uses.
        model = LitellmModel(
            model=litellm_model_name(settings.llm_model),
            base_url=settings.llm_base_url,
            # Local vLLM needs no key; the SDK still requires a value.
            api_key=settings.llm_api_key or "none",
        )
        self._deployment = settings.llm_model
        self._provider = "openai"
        # The request_clarification tool is offered only when mid-research
        # clarification is enabled (per-request identity still injected via context).
        clarify_enabled = settings.mid_research_clarification_enabled and clarify
        people = settings.people_graph_enabled
        tools = research_tools(clarify_enabled, graph_text=graph_text, people=people)
        temperature = (
            settings.llm_temperature
            if settings.llm_temperature is not None and settings.llm_send_temperature
            else None
        )
        # include_usage so the SDK reports tokens we can meter; max_tokens bounds
        # each round's completion.
        self._agent = Agent[ResearchContext](
            name="ResearchAgent",
            # Dynamic instructions: recomputed per run so the injected current
            # date/time is always fresh in a long-lived process (the SDK calls
            # this with the run context + agent; we need neither). The clarify
            # nudge is included only when the request_clarification tool is offered.
            # Static instructions: no per-request date, so the block is
            # byte-identical across requests and prefix caching can pay off.
            # The date rides on the input instead (see agent_input).
            instructions=build_agent_system_prompt(
                clarify_enabled=clarify_enabled, research=research, people=people,
            ),
            tools=tools,
            model=model,
            model_settings=ModelSettings(
                include_usage=True, max_tokens=settings.agent_max_tokens,
                # Same gate as LlmClient._chat_params: tool selection drives
                # WHICH evidence a run retrieves, so leaving sampling at the
                # provider default makes identical questions take different
                # tool paths.
                temperature=temperature,
            ),
        )
        # What every round of this agent runs under, for the generation's
        # model-parameters panel. Read off the SAME values just handed to ModelSettings,
        # and reusing the LlmClient allowlist so the set of traceable parameters is
        # defined once. `tools_offered` is per-request, not a constant: the
        # clarification tool and the graph-text tools are both conditional.
        #
        # Reports what was actually SENT: the temperature when the model accepts
        # one, nothing when it does not.
        self._model_parameters = trace_model_params(
            {"max_tokens": settings.agent_max_tokens}
            | ({} if temperature is None else {"temperature": temperature}),
            tools=tools,
        )

    async def run(self, sc: StrategyContext, scorer: NeighborScorer | None) -> None:
        """Drive the loop; findings land in ``sc.evidence`` (nothing is returned).
        The agent's final text is intentionally discarded — synthesis is separate.

        How many rounds this run may spend comes from the request's execution
        profile, not from a global: Super Research gets the full loop, Search a
        short follow-up.

        The pool is shared either way and is usually pre-filled by the chunk seed
        before the loop starts — but the model does not read the pool, so that on
        its own bought nothing: it could re-search for material already gathered.
        ``_already_found`` puts a summary of it in the agent's input, which is the
        only channel the model has.
        """
        rounds = sc.profile.agent_rounds
        rc = ResearchContext(
            repo=sc.repo, ctx=sc.ctx, embedder=sc.embedder, settings=self._s,
            evidence=sc.evidence, progress=sc.progress, usage=sc.usage, scorer=scorer,
            # The research lane's tools may return more rows per call than a bounded
            # follow-up's — it has the rounds and the wall clock to use them. A
            # profile FLAG, not a name check: keyed on the profile name, a rename
            # once turned agent_tool_rows_research into dead code silently.
            deep=sc.profile.deep_tool_rows,
            graph_text=self._graph_text,
            graph_chunk_cap=sc.profile.graph_chunk_cap,
            # Same window the prompt head states, so an authorship read ENFORCES
            # what the prompt only asks for.
            time_scope=getattr(sc, "time_scope", ("", "")),
        )
        hooks = _AgentRunHooks(
            sc.evidence, sc.profile.agent_patience,
            usage=sc.usage, deployment=self._deployment,
            rates=self._rates, provider=self._provider,
            # The reporter every tool's progress step flows through: the hook
            # points its span_parent at the current round so tool spans nest
            # under the model call that chose them.
            progress=sc.progress,
            model_parameters=self._model_parameters,
        )
        try:
            result = await Runner.run(
                self._agent,
                    agent_input(sc.question,
                                already_found=_already_found(sc),
                                time_scope=getattr(sc, "time_scope", ("", ""))),
                    context=rc,
                max_turns=rounds, hooks=hooks,
            )
            self._meter_usage(result, sc, hooks)
        except MaxTurnsExceeded as exc:
            # Routine agentic outcome for broad questions, NOT an error: every
            # round's findings already sit in sc.evidence — synthesis proceeds
            # on what was gathered (same contract as a timeout's partial answer).
            # It is the EXPECTED end of Search's short follow-up, whose cap is
            # small on purpose.
            logger.info("agent_turn_cap_reached", max_turns=rounds,
                        profile=sc.profile.name, **sc.ctx.log_fields())
            self._meter_usage(getattr(exc, "run_data", None), sc, hooks)
        except _StalePatienceReached as exc:
            # Same contract as the turn cap: findings are already in sc.evidence, we just
            # stopped burning rounds that were adding nothing. Synthesis proceeds.
            logger.info("agent_stale_stop", stale_rounds=exc.stale_rounds,
                        patience=sc.profile.agent_patience, **sc.ctx.log_fields())
            self._meter_usage_obj(exc.usage, sc, hooks)
        except (MidResearchClarification, asyncio.CancelledError, KeyboardInterrupt):
            # A pause is not a failure, and a cancellation is the client leaving.
            raise
        except Exception as exc:
            # ANY other failure inside a round. The agent's model calls go through the
            # SDK straight to LiteLLM, bypassing LlmClient's exception mapping, so a
            # provider error here never became LlmUnavailable — it left the error
            # taxonomy entirely and surfaced as a sanitized 500. The shape that hits
            # hardest on a LONG run is the growing conversation being rejected outright
            # (context-window/400-class), which is exactly the reported "long research
            # request eventually returns 500".
            #
            # Findings from completed rounds are already in sc.evidence, so degrade the
            # same way the turn cap and the stale-stop do: synthesize what was gathered.
            # With nothing gathered there is nothing to degrade TO, so surface a mapped
            # 502 (retryable, and honest about where it broke) rather than a bare 500.
            gathered = sc.evidence.size()
            logger.error("agent_round_failed", error_type=type(exc).__name__,
                         error=str(exc), evidence_gathered=gathered,
                         rounds=hooks.rounds_metered, **sc.ctx.log_fields())
            if not gathered:
                raise LlmUnavailable("The research agent could not complete") from exc
            # The degrade must be VISIBLE: mark the request partial so the
            # response says the answer came from incomplete evidence — the same
            # flag a retrieval timeout sets. Without it a thinner degraded
            # answer is indistinguishable from a complete one to the caller.
            sc.degraded = True
        finally:
            # A run that raised mid-round leaves that round's span open.
            hooks.close()
        # If the agent asked to clarify, raise AFTER the loop so it propagates
        # cleanly — an exception raised inside a tool is swallowed by the SDK.
        # The route catches it and returns the questions via the clarify/resume
        # transport; evidence gathered so far is simply discarded with the run.
        if rc.clarification:
            logger.info("mid_research_clarification", **sc.ctx.log_fields())
            raise MidResearchClarification(rc.clarification)

    def _meter_usage(self, result, sc: StrategyContext, hooks: _AgentRunHooks) -> None:
        """Fold the SDK run's token usage into our per-request meter (exact cost)."""
        try:
            u = result.context_wrapper.usage
        except AttributeError:  # SDK shape drift — never fail a served answer over metering
            logger.warning("agent_usage_unavailable")
            return
        self._meter_usage_obj(u, sc, hooks)

    def _meter_usage_obj(self, u, sc: StrategyContext, hooks: _AgentRunHooks) -> None:
        """Meter the run's aggregate usage — FALLBACK ONLY.

        Per-round metering happens in the hooks, which is what makes
        ``llm_calls`` count calls and gives each round its own Langfuse generation.
        Adding the aggregate on top would double-count every token, so this runs only
        when no round was metered — i.e. the SDK never fired ``on_llm_end`` (shape
        drift, or a failure before the first response). Cost stays exact either way.
        """
        if hooks.rounds_metered:
            logger.info(
                "agent_finished", rounds=hooks.rounds_metered,
                requests=getattr(u, "requests", 0) if u is not None else 0,
                metered="per_round", **sc.ctx.log_fields(),
            )
            return
        if u is None:
            logger.warning("agent_usage_unavailable")
            return
        logger.warning("agent_usage_metered_in_aggregate",
                       reason="no per-round usage seen from the SDK")
        cached = getattr(getattr(u, "input_tokens_details", None), "cached_tokens", 0) or 0
        sc.usage.add_llm(
            getattr(u, "input_tokens", 0) or 0,
            getattr(u, "output_tokens", 0) or 0,
            # Bucket under the model the loop actually ran on so per-model cost
            # stays truthful
            # (matches LlmClient._record).
            deployment=self._deployment,
            # Prompt-cache reads across the loop's rounds — so the report reflects
            # the discount (cache reads bill ~0.1x input).
            cached_prompt_tokens=cached,
        )
        logger.info(
            "agent_finished",
            requests=getattr(u, "requests", 0),
            input_tokens=getattr(u, "input_tokens", 0),
            cached_prompt_tokens=cached,
            output_tokens=getattr(u, "output_tokens", 0),
            **sc.ctx.log_fields(),
        )
