"""Conditional clarification (agent-led redesign, 2026-08-08).

ONE mechanism: the entry classifier — on the call every request already pays for —
writes 1-2 targeted questions ONLY when a genuine answer-changing ambiguity exists
(`plan.clarification_questions`), and the gate decides whether this MODE may pause on
them (any mode not in SINGLE_ROUND_TRIP_MODES). No separate planning LLM, no static
templates, no mandatory review stage. Everything here is proceed-biased: a spurious
pause is worse than a missed one.

    uv run pytest tests/test_ambiguous_query_routing.py -v
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from ai_graph_retrieval.clarification.models import MidResearchClarification
from ai_graph_retrieval.core.models import Usage
from ai_graph_retrieval.pipeline.gates import ClarificationGates
from ai_graph_retrieval.retrieval.base import Intent, Mode, RetrievalPlan
from ai_graph_retrieval.retrieval.router import QueryRouter
from tests.support import make_settings

# ── classifier surfaces flags + questions ────────────────────────────────────


def _router_with_args(args: dict) -> QueryRouter:
    tool_call = SimpleNamespace(function=SimpleNamespace(arguments=json.dumps(args)))
    message = SimpleNamespace(tool_calls=[tool_call])

    class _FakeLlm:
        async def complete(self, *a, **k):
            return message

    return QueryRouter(_FakeLlm(), make_settings())


def _plan(router: QueryRouter, question: str = "Search", mode: Mode = Mode.SEARCH):
    return asyncio.run(router.plan(question, [], mode, Usage()))


def test_classifier_questions_reach_the_plan():
    plan = _plan(_router_with_args({
        "intent": "entity", "anchor_phrases": [],
        "clarification_questions": [
            {"question": "Which Enterprise Search did you mean?",
             "options": ["The product", "The internal tool"]},
        ],
    }))
    assert plan.clarification_questions == [
        {"question": "Which Enterprise Search did you mean?",
         "options": ["The product", "The internal tool"]},
    ]


def test_no_questions_is_the_default_and_malformed_payloads_parse_to_none():
    # Proceed-biased parse: anything not shaped like questions means no pause.
    for args in (
        {"intent": "entity", "anchor_phrases": []},                              # omitted
        {"intent": "entity", "anchor_phrases": [], "clarification_questions": None},
        {"intent": "entity", "anchor_phrases": [], "clarification_questions": "ask them"},
        {"intent": "entity", "anchor_phrases": [],
         "clarification_questions": ["a bare string"]},
        {"intent": "entity", "anchor_phrases": [],
         "clarification_questions": [{"options": ["A", "B"]}]},                  # no question
        {"intent": "entity", "anchor_phrases": [],
         "clarification_questions": [{"question": "   ", "options": ["A"]}]},    # blank
    ):
        assert _plan(_router_with_args(args)).clarification_questions == [], repr(args)


def test_no_fixed_question_limit_and_options_are_sane():
    # the model asks exactly as many questions
    # as understanding the request genuinely requires — no hard cap in code;
    # restraint is the prompt's job. Options stay bounded at 4 per question.
    rows = [{"question": f"Q{i}?", "options": [f"O{j}" for j in range(9)]}
            for i in range(5)]
    plan = _plan(_router_with_args(
        {"intent": "entity", "anchor_phrases": [], "clarification_questions": rows}
    ))
    assert len(plan.clarification_questions) == 5          # all genuine questions kept
    assert all(len(q["options"]) == 4 for q in plan.clarification_questions)


def test_string_options_are_split_not_iterated():
    """The pilot regression, re-pinned against the new parse: `options` sent as
    one comma-separated STRING must split into labels — the old gate iterated
    the string and showed five options labelled B, r, o, a, d."""
    plan = _plan(_router_with_args({
        "intent": "entity", "anchor_phrases": [],
        "clarification_questions": [
            {"question": "How broad should this be?", "options": "Broad, Narrow, Deep"},
        ],
    }))
    assert plan.clarification_questions[0]["options"] == ["Broad", "Narrow", "Deep"]


def test_every_mode_passes_the_plan_through_verbatim():
    # No mode rewrites the plan: the MODE picks the strategy in the planner and
    # the classifier's verdict — intent, anchors, questions — is
    # recorded exactly as classified. (The old copy-and-override normalization
    # once dropped a verdict field by rebuilding field-by-field; with no rewrite
    # at all, nothing can be dropped.)
    args = {"intent": "research", "anchor_phrases": ["releases"],
            "clarification_questions": [{"question": "Which?", "options": ["A", "B"]}]}
    for mode in (Mode.SUPER_RESEARCH, Mode.SEARCH):
        plan = _plan(_router_with_args(args), mode=mode)
        assert plan.intent is Intent.RESEARCH, mode
        assert plan.anchor_phrases == ["releases"], mode
        assert plan.clarification_questions, mode


# ── the gate: both modes may pause on the classifier's questions ───────────


class _Ctx:
    request_id = "req-1"

    def log_fields(self):
        return {"request_id": self.request_id}


_QUESTIONS = [
    {"question": "Which Jane did you mean?", "options": ["Jane Doe", "Jane Roe"]},
    {"question": "Which team's dashboards?", "options": ["Platform", "Data"]},
]


def _enforce(plan: RetrievalPlan, mode: Mode, *, plan_review=True, **settings_kw):
    ClarificationGates(make_settings(**settings_kw)).enforce(
        _Ctx(), plan=plan, mode=mode, plan_review=plan_review,
    )


def test_search_pauses_on_the_classifier_questions():
    plan = RetrievalPlan(intent=Intent.ENTITY, clarification_questions=_QUESTIONS)
    with pytest.raises(MidResearchClarification) as excinfo:
        _enforce(plan, Mode.SEARCH)
    questions = excinfo.value.questions
    assert [q.question for q in questions] == [
        "Which Jane did you mean?", "Which team's dashboards?",
    ]
    # Unique ids — build_question alone stamps every question "q0", and two
    # questions sharing an id make the resume answers ambiguous.
    assert len({q.id for q in questions}) == 2
    # Answerable by construction: at least two clickable options each.
    assert all(len(q.options) >= 2 for q in questions)


def test_search_with_no_questions_proceeds():
    _enforce(RetrievalPlan(intent=Intent.ENTITY), Mode.SEARCH)  # must not raise


def test_super_research_pauses_on_the_classifier_questions_too():
    # Same entry handling as Search.
    plan = RetrievalPlan(intent=Intent.ENTITY, clarification_questions=_QUESTIONS)
    with pytest.raises(MidResearchClarification):
        _enforce(plan, Mode.SUPER_RESEARCH)


def test_the_resume_turn_never_pauses_again():
    # plan_review=False is the /resume guard: the enriched question already
    # carries the user's answers; re-questioning it would loop the pause.
    plan = RetrievalPlan(intent=Intent.ENTITY, clarification_questions=_QUESTIONS)
    _enforce(plan, Mode.SEARCH, plan_review=False)  # must not raise


def test_the_kill_switch_turns_the_pause_off_without_touching_classification():
    plan = RetrievalPlan(intent=Intent.ENTITY, clarification_questions=_QUESTIONS)
    _enforce(plan, Mode.SEARCH, search_clarification_enabled=False)  # must not raise
    assert plan.clarification_questions, "the verdict itself stays (telemetry)"


# ── the context review's edit to the plan ─────────────────────────
# The review replaces exactly ONE field on the plan. It shipped as a 500 first
# time: RetrievalPlan is a pydantic model, and the orchestrator reached for
# `dataclasses.replace` — which is correct for the RetrievalDecision sitting
# beside it and raises TypeError on the plan. No test built a real Orchestrator,
# so nothing caught it until a live request did.


def test_replacing_the_questions_preserves_every_other_planned_field():
    plan = RetrievalPlan(
        intent=Intent.ENTITY, search_query="what did the team publish",
        time_scope_from="2026-01-01", time_scope_to="2026-08-25",
        anchor_phrases=["the team"], rationale="llm classifier",
        clarification_questions=_QUESTIONS,
    )
    reviewed = plan.model_copy(update={"clarification_questions": []})

    assert reviewed.clarification_questions == []
    # Everything the reviewer never looks at has to survive — the window most of
    # all, since the authorship read now enforces it.
    assert (reviewed.time_scope_from, reviewed.time_scope_to) == ("2026-01-01", "2026-08-25")
    assert reviewed.search_query == "what did the team publish"
    assert reviewed.anchor_phrases == ["the team"]
    # The original is untouched: the gate reads whichever object it was handed.
    assert plan.clarification_questions == _QUESTIONS


def test_the_plan_is_not_a_dataclass_which_is_why_model_copy_is_used():
    """Pins the reason for the mechanism. If RetrievalPlan ever becomes a
    dataclass this fails, and the comment at the call site needs revisiting."""
    import dataclasses

    plan = RetrievalPlan(intent=Intent.ENTITY, clarification_questions=_QUESTIONS)
    assert not dataclasses.is_dataclass(plan)
    with pytest.raises(TypeError):
        dataclasses.replace(plan, clarification_questions=[])
