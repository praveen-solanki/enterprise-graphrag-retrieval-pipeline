"""User-selected retrieval modes: Search / Super Research.

The mode is the user's explicit per-query choice — there is no automatic mode
selection, no default mode, and no routing between modes. The mode never
changes what is TRUE, only how much work gathers the evidence. These tests pin
that contract:

  * the planner maps each mode to its own lane (Super Research always runs the
    deep research agent),
  * the composer keeps one grounded system prompt for every mode,
  * the response carries the executed mode and nothing steers the caller to a
    different one (`next_mode` is gone),
  * a request without a mode is rejected at the boundary, not defaulted.

    uv run pytest tests/test_retrieval_modes.py -v
"""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from ai_graph_retrieval.api.schemas import RetrieveRequest, RetrieveResponse
from ai_graph_retrieval.core.models import Evidence, EvidencePool, Usage
from ai_graph_retrieval.pipeline.gates import ClarificationGates
from ai_graph_retrieval.pipeline.respond import build_payload
from ai_graph_retrieval.pipeline.state import RequestState
from ai_graph_retrieval.retrieval.base import Intent, Mode, RetrievalPlan
from ai_graph_retrieval.retrieval.router import QueryRouter
from ai_graph_retrieval.synthesis.composer import AnswerComposer
from tests.support import make_settings

# ── the mode set itself ───────────────────────────────────────────────────────


def test_exactly_two_user_facing_modes_exist():
    """A mode added back must be a deliberate decision that revisits routing,
    profiles, synthesis tiers and the API contract — so this fails loudly rather
    than letting one arrive by enum edit alone."""
    assert {m.value for m in Mode} == {"search", "super_research"}


# ── router: the mode's execution guarantee ────────────────────────────────────


def _router(classified: RetrievalPlan) -> QueryRouter:
    args = {
        "intent": classified.intent.value,
        "anchor_phrases": classified.anchor_phrases,
    }
    tool_call = SimpleNamespace(function=SimpleNamespace(arguments=json.dumps(args)))

    class _Llm:
        async def complete(self, *a, **k):
            return SimpleNamespace(tool_calls=[tool_call])

    return QueryRouter(_Llm(), make_settings())


def test_the_router_rewrites_no_plan_for_any_mode():
    # Intent is telemetry since the agent-led redesign: the planner's
    # mode->strategy map decides what runs, so the router records the
    # classifier's verdict verbatim instead of forcing or downgrading it.
    classified = RetrievalPlan(intent=Intent.RESEARCH, anchor_phrases=["org chart"])
    for mode in Mode:
        plan = asyncio.run(_router(classified).plan("q", [], mode, Usage()))
        assert plan.intent is Intent.RESEARCH, mode
        assert plan.anchor_phrases == ["org chart"]   # classification is kept


def test_search_keeps_a_temporal_classification_untouched():
    classified = RetrievalPlan(intent=Intent.TEMPORAL)
    plan = asyncio.run(_router(classified).plan("q", [], Mode.SEARCH, Usage()))
    assert plan.intent is Intent.TEMPORAL


def test_super_research_gets_the_agent_lane_from_the_mode_not_the_intent():
    # The entry classifier still runs (the conversational short-circuit needs
    # its verdict), but the agent lane is the MODE's: the planner maps
    # SUPER_RESEARCH to the research strategy whatever intent was recorded.
    from ai_graph_retrieval.pipeline.planning import ExecutionPlanner

    classified = RetrievalPlan(intent=Intent.ENTITY, anchor_phrases=["a"])
    research = SimpleNamespace(name="deep_research")
    planner = ExecutionPlanner(
        _router(classified), make_settings(),
        search_strategy=SimpleNamespace(name="search"),
        research_strategy=research,
    )
    decision = asyncio.run(planner.resolve("q", [], Mode.SUPER_RESEARCH, Usage()))
    assert decision.strategy is research
    assert decision.plan.intent is Intent.ENTITY   # recorded, not rewritten


# ── composer: one grounded prompt, no per-mode depth directives ───────────────


class _Ctx:
    request_id = "r"
    tenant_id = "t"

    def log_fields(self):
        return {}


class _Progress:
    @asynccontextmanager
    async def step(self, *_a, **_k):
        yield SimpleNamespace()

    async def answer_delta(self, _t):
        pass


def _capturing_sc(captured: dict) -> SimpleNamespace:
    class _Llm:
        async def stream_text(self, messages, max_tokens=None, fast=False, usage=None, **_kwargs):
            captured["system"] = messages[0]["content"]
            captured["fast"] = fast
            yield "An answer long enough to not be degenerate."

    return SimpleNamespace(question="who works under Jane Doe?", history=[],
                           progress=_Progress(), llm=_Llm(), usage=Usage(), ctx=_Ctx())


def _compose(mode: Mode) -> dict:
    captured: dict = {}
    composer = AnswerComposer(make_settings())
    asyncio.run(composer.compose(_capturing_sc(captured), Evidence(), mode=mode))
    return captured


def _system_for(mode: Mode) -> str:
    return _compose(mode)["system"]


def test_the_per_mode_depth_directives_are_gone():
    # The old Fast/Balanced/Research ladder shaped answer depth with a per-mode
    # prompt paragraph. With the ladder removed, no mode carries one.
    for mode in Mode:
        assert "ANSWER DEPTH" not in _system_for(mode), mode


def test_the_answer_model_tier_follows_the_mode():
    # Search answers on the fast tier; Super Research keeps the heavy model for
    # the deepest report in the service.
    assert _compose(Mode.SEARCH)["fast"] is True
    assert _compose(Mode.SUPER_RESEARCH)["fast"] is False


def test_grounding_rules_survive_in_every_mode():
    """The mode shapes work, never truthfulness: the grounding contract stays in
    the prompt whatever the user selects.

    Asserted by SUBSTANCE rather than by a header string: a literal
    section-title check would pass or fail on wording rather than on protection,
    and would miss a future prompt that kept the heading and dropped the rule.
    """
    for mode in Mode:
        system = _system_for(mode).lower()
        # 1. closed world: answer only from what was retrieved
        assert "only" in system and "evidence" in system, mode
        # 2. no fabrication
        assert "never invent" in system or "never produce a statement" in system, mode
        # 3. retrieved content is data, not instructions to the model
        assert "never instructions" in system or "not executable" in system, mode


# ── response contract: the executed mode, and nothing steering past it ────────


def _payload(mode: Mode) -> dict:
    return build_payload(
        _Ctx(), RequestState(mode=mode.value, intent="entity", strategy="search"),
        answer="a", evidence=EvidencePool().snapshot(),
        usage_report=Usage().snapshot(), duration_ms=1,
    )


def test_payload_carries_the_executed_mode_and_no_refinement_affordance():
    for mode in Mode:
        p = _payload(mode)
        assert p["mode"] == mode.value
        assert "next_mode" not in p     # no routing between modes, not even a hint


def test_api_schemas_accept_and_roundtrip_every_mode():
    for mode in Mode:
        req = RetrieveRequest(question="who works under Jane?", mode=mode.value,
                              connectorType="wikipedia")
        assert req.mode is mode
        resp = RetrieveResponse(**_payload(mode))
        assert resp.mode == mode.value


def test_a_request_without_a_mode_is_rejected_not_defaulted():
    # The user picks the mode per query. A missing mode used to silently run
    # AUTO; now it is a validation error the caller can see and fix.
    with pytest.raises(ValidationError):
        RetrieveRequest(question="who works under Jane?")


@pytest.mark.parametrize("legacy", ["auto", "fast", "balanced", "research"])
def test_a_removed_mode_is_rejected_at_the_boundary(legacy):
    with pytest.raises(ValidationError):
        RetrieveRequest(question="q", mode=legacy)


def test_a_caller_still_sending_months_is_not_broken():
    # The field was removed with the lane redesign: time windows
    # are the model's call now. Deliberately IGNORED rather than rejected —
    # callers that still send it must keep working, they just stop steering.
    req = RetrieveRequest(question="q", mode="search", connectorType="wikipedia", months=6)
    assert not hasattr(req, "months")


# ── gates: only the classifier's own questions can pause, and only Search ────


def test_a_plan_without_questions_pauses_no_mode():
    # (See tests/test_ambiguous_query_routing.py for the full gate contract.)
    gates = ClarificationGates(make_settings())
    plan = RetrievalPlan(intent=Intent.TEMPORAL)
    for mode in Mode:
        gates.enforce(_Ctx(), plan=plan, mode=mode, plan_review=True)   # no raise


def test_former_employees_are_kept_out_of_enumerations_in_both_prompts():
    """A departed person must not appear both as a ROW in a members table AND in
    the "no longer with the organization" note below it. The writer's prompt
    carries the whole rule: a table row counts as listing them, note-or-nothing
    never both, and asking ABOUT former people flips it."""
    from ai_graph_retrieval.synthesis.prompts import SYNTHESIS_SYSTEM_PROMPT

    # the writer: a table row counts as listing them; note-or-nothing, never both
    assert "a table ROW is listing them" in SYNTHESIS_SYSTEM_PROMPT
    assert "Never both." in SYNTHESIS_SYSTEM_PROMPT
    # the flag is not a designation either — it never renders beside the name
    assert "The INACTIVE flag itself is not a role" in SYNTHESIS_SYSTEM_PROMPT
    # …and the escape hatch stays: asked-about former people ARE the answer
    assert "asks about former, past or departed people" in SYNTHESIS_SYSTEM_PROMPT
