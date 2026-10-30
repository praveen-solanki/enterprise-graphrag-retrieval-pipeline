"""People-enumeration questions get ONE deterministic structure.

The defect: ``answer_shape`` is chosen by the classifier, whose own instructions
make ``prose`` the safe default "whenever in doubt". A roster question phrased
without an explicit attribute list ("who is on the platform team?") reads as
exactly that doubtful case, so the same question came back ``prose`` in one run
and ``table`` in the next — the answer flipped between paragraphs, a numbered
list and a table across runs.

The signal is the MODEL'S: the classifier answers one dedicated boolean —
``people_enumeration``, judged from the intent of the question — and the router
acts only on that structured verdict: an unstructured shape (prose/none) on a
question the model marked as enumerating people settles to PER_MEMBER. There is
no query-text heuristic anywhere in the path (rule 8): no keyword list, no
phrase matching, no regex. The synthesis prompt carries the matching per-person
rule.

NOTE ON THE LITERAL QUESTION STRINGS BELOW: they are test FIXTURES, not rules.
Every test injects the classifier's verdict — the question text drives nothing,
and several tests prove exactly that by running the SAME strings under both
verdicts. Swap any question for any other and every assertion still holds.

    uv run pytest tests/test_people_roster_shape.py -v
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from ai_graph_retrieval.core.models import Usage
from ai_graph_retrieval.retrieval import router as router_module
from ai_graph_retrieval.retrieval.base import AnswerShape, Mode
from ai_graph_retrieval.retrieval.router import (
    _CLASSIFY_TOOL,
    QueryRouter,
    _answer_shape,
)
from tests.support import make_people_settings as make_settings

# Diverse formulations, deliberately spanning what any keyword list would catch,
# what it would miss, and what it would falsely catch. The router must treat
# every one of them identically — the flag decides, the text never does.
QUESTIONS = [
    "Who reports to Jane Doe?",                       # classic roster phrasing
    "who is on the design team",                        # bare roster, no attributes
    "give me the org chart under the CTO",              # no roster keyword at all
    "which engineers support the payments area?",       # role-holder enumeration
    "I need to know everybody involved in the launch",  # colloquial
    "quem faz parte da equipe de plataforma?",          # not even English
    "Who is Jane Doe?",                               # ONE person — not a set
    "list the services in production",                  # a set, but not of people
    "what is the release cadence?",                     # no people at all
]

UNSTRUCTURED = ["prose", "", None, "not_a_shape"]


# ── the flag is the only decision-maker ──────────────────────────────────────


@pytest.mark.parametrize("question", QUESTIONS)
@pytest.mark.parametrize("said", UNSTRUCTURED)
def test_a_positive_verdict_upgrades_an_unstructured_shape(question, said):
    """Whatever the phrasing and however the shape came back, the model's own
    people_enumeration=true settles prose/none to PER_MEMBER. ``question`` is
    unused by the code under test — parametrizing it proves that."""
    del question
    args = {"answer_shape": said, "people_enumeration": True}
    assert _answer_shape(args) is AnswerShape.PER_MEMBER


@pytest.mark.parametrize("question", QUESTIONS)
def test_a_negative_verdict_leaves_prose_alone_whatever_the_text_says(question):
    """The proof no phrase matching remains: the SAME strings — including the
    most roster-sounding ones — stay prose when the model said false."""
    del question
    args = {"answer_shape": "prose", "people_enumeration": False}
    assert _answer_shape(args) is AnswerShape.PROSE


def test_the_same_text_under_both_verdicts_lands_on_different_shapes():
    """Same question text, opposite flags, opposite outcomes — text-blindness
    stated as one assertion."""
    text_free = {"answer_shape": "prose"}
    assert _answer_shape({**text_free, "people_enumeration": True}) \
        is AnswerShape.PER_MEMBER
    assert _answer_shape({**text_free, "people_enumeration": False}) \
        is AnswerShape.PROSE


def test_every_positive_verdict_lands_on_the_same_shape():
    """Equivalence is the actual promise: not merely "structured", but the same
    structure for every formulation the model reads as the same ask."""
    shapes = {
        _answer_shape({"answer_shape": "prose", "people_enumeration": True})
        for _ in QUESTIONS
    }
    assert shapes == {AnswerShape.PER_MEMBER}


# ── precedence: only the unstructured verdicts are upgraded ──────────────────


def test_an_enumeration_that_is_also_a_progression_keeps_its_timeline():
    """TIMELINE outranks the upgrade, the same precedence the STRUCTURE list
    uses — a progression flattened into a roster loses the progression."""
    args = {"answer_shape": "timeline", "people_enumeration": True}
    assert _answer_shape(args) is AnswerShape.TIMELINE


def test_a_shape_the_classifier_already_structured_is_not_overridden():
    """TABLE is already per-person structure; the upgrade only rescues the
    unstructured verdicts, so a good verdict is left exactly as it came."""
    args = {"answer_shape": "table", "people_enumeration": True}
    assert _answer_shape(args) is AnswerShape.TABLE


# ── fail-open parsing ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("drifted", ["true", "True", 1, "yes", [True], {}, None])
def test_only_the_json_boolean_fires_the_upgrade(drifted):
    """A provider drifting the type must never flip the shape: anything but the
    boolean ``true`` reads as no signal, and the model's verdict stands."""
    args = {"answer_shape": "prose", "people_enumeration": drifted}
    assert _answer_shape(args) is AnswerShape.PROSE


def test_an_absent_flag_preserves_the_old_behaviour_exactly():
    """No signal means no override — the parse before this field existed."""
    assert _answer_shape({"answer_shape": "prose"}) is AnswerShape.PROSE
    assert _answer_shape({"answer_shape": "bogus"}) is None
    assert _answer_shape({}) is None


@pytest.mark.parametrize("said,expected", [
    ("prose", AnswerShape.PROSE),
    ("table", AnswerShape.TABLE),
    ("timeline", AnswerShape.TIMELINE),
    ("per_member", AnswerShape.PER_MEMBER),
])
def test_non_enumeration_questions_keep_the_existing_behaviour(said, expected):
    """Every other question is still the classifier's call, including its prose
    default."""
    args = {"answer_shape": said, "people_enumeration": False}
    assert _answer_shape(args) is expected


# ── end to end through the router, the house way: the verdict is injected ────


_KNOWLEDGE = {
    "category": "knowledge", "reply": "",
    "search_query": "q", "intent": "entity", "anchor_phrases": [],
}


def _plan_for(args: dict):
    tool_call = SimpleNamespace(function=SimpleNamespace(arguments=json.dumps(args)))

    class _Llm:
        async def complete(self, messages, **k):
            return SimpleNamespace(tool_calls=[tool_call])

    router = QueryRouter(_Llm(), make_settings())
    return asyncio.run(router.plan("any question at all", [], Mode.SEARCH, Usage()))


def test_the_plan_carries_the_upgraded_shape():
    plan = _plan_for({**_KNOWLEDGE, "answer_shape": "prose",
                      "people_enumeration": True})
    assert plan.answer_shape is AnswerShape.PER_MEMBER


def test_the_plan_keeps_the_models_shape_on_a_negative_verdict():
    plan = _plan_for({**_KNOWLEDGE, "answer_shape": "prose",
                      "people_enumeration": False})
    assert plan.answer_shape is AnswerShape.PROSE


# ── the contract with the classifier ─────────────────────────────────────────


def test_the_schema_forces_a_people_enumeration_verdict():
    """The field is a required boolean in the forced function call, so the model
    must answer the question on every classification — an optional field would
    quietly become an absent one."""
    schema = _CLASSIFY_TOOL["function"]["parameters"]
    assert schema["properties"]["people_enumeration"]["type"] == "boolean"
    assert "people_enumeration" in schema["required"]


def test_the_classifier_prompt_defines_the_field_by_intent():
    """The definition the model works from is about INTENT — a set of people —
    with the single-person and non-people exclusions stated, and it explicitly
    escapes the shape's prose-when-in-doubt default."""
    from ai_graph_retrieval.retrieval.router import _SYSTEM

    flat = " ".join(_SYSTEM.split())
    assert "people_enumeration" in flat
    assert "SET or GROUP of people" in flat
    assert "Judge the INTENT" in flat
    assert "ONE named or described individual" in flat
    assert "prose-when-in-doubt default does NOT apply to this field" in flat


def test_no_query_text_heuristic_remains_in_the_router():
    """The removal itself, pinned: the keyword lists and the text predicate are
    gone, so the module header's "no query-text heuristic" holds again."""
    for symbol in ("_ROSTER_ENUMERATORS", "_PEOPLE_NOUNS", "_SINGULAR_IDENTITY",
                   "is_people_roster_question"):
        assert not hasattr(router_module, symbol), symbol


# ── the prompts carry the matching rule, in EVERY mode ───────────────────────


def test_the_standard_prompt_orders_a_roster_per_person():
    from ai_graph_retrieval.synthesis.prompts import SYNTHESIS_SYSTEM_PROMPT as P

    flat = " ".join(P.split())
    assert "THE QUESTION ENUMERATES PEOPLE" in flat
    assert "one row per person" in flat
    # The failure mode it exists to forbid.
    assert "NEVER answer such a question as free-form prose" in flat


def test_the_per_member_addendum_still_renders_one_row_per_member():
    """The shape a positive verdict now resolves to must describe a table even
    for a bare "who?" ask, or the router would be pointing enumerations at an
    addendum that says nothing about them."""
    from ai_graph_retrieval.synthesis.prompts import SYNTHESIS_SHAPE_ADDENDA

    block = " ".join(SYNTHESIS_SHAPE_ADDENDA[AnswerShape.PER_MEMBER].split())
    assert "one row per member" in block
    assert "asked only WHO the members are" in block
    assert "Never degrade this shape to prose" in block


def test_the_prose_addendum_does_not_override_a_people_part():
    """The false-negative net: when the flag misses and prose reaches synthesis,
    the PROSE addendum itself still renders a people part per-person rather than
    weaving names through a paragraph."""
    from ai_graph_retrieval.synthesis.prompts import SYNTHESIS_SHAPE_ADDENDA

    block = " ".join(SYNTHESIS_SHAPE_ADDENDA[AnswerShape.PROSE].split())
    assert "For PEOPLE it is not optional" in block
    assert "one row per person" in block


# ── labelled profile segments feed the roster's columns ──────────────────────


def test_profile_segments_are_labelled_so_the_columns_are_unambiguous():
    """Unlabelled, the line was bare values ("Staff Engineer · Platform ·
    Bengaluru") and which one was the department rather than the location was
    left to the model to infer. A per-person table has to fill those columns."""
    from ai_graph_retrieval.graph.repository import _profile_line

    line = _profile_line({
        "job_title": "Staff Engineer", "department": "Platform",
        "location": "Bengaluru", "manager": "Jane Doe", "direct_reports": 3,
    })

    assert "title: Staff Engineer" in line
    assert "department: Platform" in line
    assert "location: Bengaluru" in line
    # The org segments already read as their own words and keep their wording —
    # person_identity.py matches the manager one by its MANAGER_SEGMENT prefix.
    assert "reports to Jane Doe" in line


def test_the_departed_marker_still_leads_the_labelled_line():
    """It LEADS deliberately (see _profile_line): the first token decides whether
    the person belongs in the answer at all. Labelling must not displace it."""
    from ai_graph_retrieval.graph.repository import _profile_line

    line = _profile_line({"job_title": "Analyst", "employment_status": "Inactive"})

    assert line.startswith("FORMER EMPLOYEE")
    assert "title: Analyst" in line


def test_an_empty_row_still_renders_nothing():
    """A blank in the source must not become text that looks like an answer."""
    from ai_graph_retrieval.graph.repository import _profile_line

    assert _profile_line({}) == ""


def test_the_manager_segment_is_still_parseable_by_person_identity():
    """The one cross-module contract on this string: person_identity splits on
    " · " and matches the manager segment by prefix."""
    from ai_graph_retrieval.graph.repository import _profile_line
    from ai_graph_retrieval.retrieval.person_status import MANAGER_SEGMENT

    line = _profile_line({"job_title": "Analyst", "manager": "Jane Doe"})
    segments = line.split(" · ")

    assert any(s.startswith(MANAGER_SEGMENT) for s in segments)
    assert [s[len(MANAGER_SEGMENT):] for s in segments
            if s.startswith(MANAGER_SEGMENT)] == ["Jane Doe"]
