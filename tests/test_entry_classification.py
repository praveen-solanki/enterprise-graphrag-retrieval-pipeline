"""Entry classification: one model call decides the whole entry.

The word-list heuristics (smalltalk sets, follow-up pronoun/opener gating) are
gone — the router's single fast-tier classification now produces the
conversational triage, the self-contained follow-up rewrite, and the lane, in
one flat forced-function call. These tests pin:

  * conversational categories short-circuit with the MODEL's reply (no canned
    text anywhere in code),
  * every parse is fail-safe TOWARD retrieval: unknown/missing category, a
    conversational verdict with no reply text, or a missing rewrite all degrade
    to "knowledge, question verbatim" — a wrong short-circuit hides answers,
  * the rewrite is used only when there is a conversation to resolve from and
    the flag allows it,
  * caller mode and the short-circuit flag override the classifier's verdicts.

NOTE ON THE LITERAL QUESTION STRINGS BELOW ("hi", "I want to play football",
"who owns Enterprise Search?"): they are test FIXTURES, not rules. Every test
injects the classifier's verdict through a fake LLM (``_router(args)``) — the
question text drives nothing; swap any question for any other and every
assertion still holds. Production code never inspects query text: the only
decision-maker is the model, and these tests exercise how the router acts on
the model's verdict.

    uv run pytest tests/test_entry_classification.py -v
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from ai_graph_retrieval.core.exceptions import LlmUnavailable
from ai_graph_retrieval.core.models import Usage
from ai_graph_retrieval.retrieval.base import (
    AnswerShape,
    ConversationalTurn,
    Intent,
    Mode,
    RetrievalPlan,
)
from ai_graph_retrieval.retrieval.router import QueryRouter
from tests.support import make_settings

_KNOWLEDGE = {
    "category": "knowledge", "reply": "",
    "search_query": "who owns Enterprise Search?",
    "intent": "entity", "anchor_phrases": ["Enterprise Search"],
}


def _router(args: dict, captured: dict | None = None, **settings_overrides) -> QueryRouter:
    """A QueryRouter whose LLM returns exactly one classify_query call carrying
    ``args`` — so we test the router's decision logic, not the model."""
    tool_call = SimpleNamespace(function=SimpleNamespace(arguments=json.dumps(args)))

    class _Llm:
        async def complete(self, messages, **k):
            if captured is not None:
                captured["messages"] = messages
            return SimpleNamespace(tool_calls=[tool_call])

    return QueryRouter(_Llm(), make_settings(**settings_overrides))


def _plan(router: QueryRouter, question: str = "who owns Enterprise Search?",
          history: list | None = None, mode: Mode = Mode.SEARCH):
    return asyncio.run(router.plan(question, history or [], mode, Usage()))


# ── conversational short-circuit ──────────────────────────────────────────────


@pytest.mark.parametrize("category", ["greeting", "smalltalk", "capability", "guardrail"])
def test_conversational_categories_shortcircuit_with_the_models_reply(category):
    # out_of_scope is carved out for Search (retrieval is the judge of an
    # unrecognized term).
    out = _plan(_router({**_KNOWLEDGE, "category": category,
                         "reply": "Hi! Ask me about your organization."}), "hi")
    assert isinstance(out, ConversationalTurn)
    assert out.category == category
    assert out.reply == "Hi! Ask me about your organization."  # model text, not canned


def test_knowledge_never_shortcircuits():
    out = _plan(_router(_KNOWLEDGE))
    assert isinstance(out, RetrievalPlan)
    assert out.intent is Intent.ENTITY


def test_flag_off_runs_the_pipeline_even_for_a_greeting():
    out = _plan(_router({**_KNOWLEDGE, "category": "greeting", "reply": "Hi!"},
                        chat_smalltalk_shortcircuit=False), "hi")
    assert isinstance(out, RetrievalPlan)      # verdict recorded but not honored


def test_per_request_settings_disable_shortcircuit_without_rebinding_the_router():
    """options.chat_smalltalk_shortcircuit lands as a per-call settings view —
    the boot-time router stays on, the request does not short-circuit."""
    router = _router({**_KNOWLEDGE, "category": "greeting", "reply": "Hi!"})
    off = make_settings(chat_smalltalk_shortcircuit=False)
    out = asyncio.run(router.plan("hi", [], Mode.SEARCH, Usage(), settings=off))
    assert isinstance(out, RetrievalPlan)


def test_conversational_turns_shortcircuit_in_every_mode():
    # The mode is a choice about how to work a KNOWLEDGE query. A persistent
    # 'Research' selector must not send 'thanks!' through the agent loop — the
    # short-circuit wins in every mode; a social category is honoured
    # everywhere, so it is what pins the shared rule.
    for mode in Mode:
        out = _plan(_router({**_KNOWLEDGE, "category": "smalltalk",
                             "reply": "Any time!"}), "thanks!", mode=mode)
        assert isinstance(out, ConversationalTurn), mode
        assert out.category == "smalltalk"


@pytest.mark.parametrize("category", ["guardrail", "out_of_scope"])
def test_super_research_declines_guardrail_but_searches_out_of_scope(category):
    # Same entry handling as Search (2026-08-08): an unrecognized term deserves
    # the investigation the caller asked for; a genuine guardrail still declines.
    out = _plan(_router({**_KNOWLEDGE, "category": category,
                         "reply": "That's outside what I cover."}),
                "I want to play football", mode=Mode.SUPER_RESEARCH)
    if category == "guardrail":
        assert isinstance(out, ConversationalTurn) and out.category == category
    else:
        assert isinstance(out, RetrievalPlan)


# ── fail-safes: everything malformed degrades toward retrieval ────────────────


def test_unknown_category_degrades_to_knowledge():
    out = _plan(_router({**_KNOWLEDGE, "category": "chitchat", "reply": "yo"}))
    assert isinstance(out, RetrievalPlan)


def test_missing_category_degrades_to_knowledge():
    args = {k: v for k, v in _KNOWLEDGE.items() if k != "category"}
    assert isinstance(_plan(_router(args)), RetrievalPlan)


def test_conversational_verdict_without_reply_text_degrades_to_knowledge():
    # Nothing to send the user — proceeding to retrieval beats an empty answer.
    out = _plan(_router({**_KNOWLEDGE, "category": "greeting", "reply": "   "}))
    assert isinstance(out, RetrievalPlan)


def test_missing_intent_fails_loud_not_into_a_wrong_lane():
    args = {k: v for k, v in _KNOWLEDGE.items() if k != "intent"}
    with pytest.raises(LlmUnavailable):
        _plan(_router(args))


# ── the enumeration flag (replaces the _AGG_RE query regex) ──────────────────


def test_the_classifier_no_longer_returns_a_complete_list_verdict():
    """Removed on purpose (2026-08-27). The verdict selected a second set of
    synthesis budgets and a second prompt block, and it decided the SAME question
    differently between runs — one query took the wide lane on one run and the
    narrow one on the next, rendering 9 excerpts against 3. Whether an answer must
    enumerate a set is now the answer model's reading of the question.

    A drifting provider that still sends the field must not resurrect the lane, so
    an unexpected key is ignored rather than carried on the plan.
    """
    from ai_graph_retrieval.retrieval.router import _CLASSIFY_TOOL

    schema = _CLASSIFY_TOOL["function"]["parameters"]
    assert "enumeration" not in schema["properties"]
    assert "enumeration" not in schema["required"]
    out = _plan(_router({**_KNOWLEDGE, "enumeration": True}))
    assert not hasattr(out, "enumeration")


# ── the follow-up rewrite ─────────────────────────────────────────────────────

_HISTORY = [{"role": "user", "content": "who leads the Platform team?"},
            {"role": "assistant", "content": "Alex Doe leads Platform."}]


def test_rewrite_is_used_when_there_is_history_to_resolve_from():
    # enable_follow_up_support ships default-OFF; the rewrite path is its gated
    # behavior, so this test turns it on explicitly.
    out = _plan(_router({**_KNOWLEDGE, "search_query": "who is Alex Doe's manager?"},
                        enable_follow_up_support=True),
                "and their manager?", history=_HISTORY)
    assert out.search_query == "who is Alex Doe's manager?"


def test_no_history_means_the_question_stands_verbatim():
    # With nothing to resolve from, a model paraphrase must never drift retrieval.
    out = _plan(_router({**_KNOWLEDGE, "search_query": "Enterprise Search ownership"}))
    assert out.search_query == "who owns Enterprise Search?"


def test_follow_up_support_off_keeps_the_question_verbatim():
    # The code default for the current release: history present, flag off →
    # the verbatim question retrieves, standalone.
    out = _plan(_router({**_KNOWLEDGE, "search_query": "who is Alex Doe's manager?"},
                        enable_follow_up_support=False),
                "and their manager?", history=_HISTORY)
    assert out.search_query == "and their manager?"


def test_follow_up_support_off_withholds_history_from_the_classifier_prompt():
    # Flag off ⇒ the turn is classified standalone: prior turns must not reach
    # the prompt at all (no wasted tokens, no history-swayed classification).
    captured: dict = {}
    router = _router(_KNOWLEDGE, captured=captured, enable_follow_up_support=False)
    _plan(router, "and their manager?", history=_HISTORY)
    user_msg = captured["messages"][-1]["content"]
    assert "Alex Doe" not in user_msg
    assert "CONVERSATION SO FAR" not in user_msg
    assert user_msg.endswith("MESSAGE: and their manager?")


def test_empty_rewrite_falls_back_to_the_question():
    out = _plan(_router({**_KNOWLEDGE, "search_query": ""}),
                "and their manager?", history=_HISTORY)
    assert out.search_query == "and their manager?"


# ── the classifier prompt sees only real conversational turns ─────────────────


def test_prompt_windows_real_turns_and_skips_tool_rows():
    captured: dict = {}
    # History reaches the classifier prompt only when follow-up support is on
    # (release default off) — this test covers the enabled windowing behavior.
    router = _router(_KNOWLEDGE, captured=captured, enable_follow_up_support=True)
    history = [
        {"role": "tool", "content": "raw tool output"},
        {"role": "system", "content": "system row"},
        {"role": "user", "content": ""},                      # empty: skipped
        {"role": "user", "content": "who leads Platform?"},
        {"role": "assistant", "content": "Alex Doe leads Platform."},
    ]
    _plan(router, "and their manager?", history=history)
    user_msg = captured["messages"][-1]["content"]
    assert "who leads Platform?" in user_msg
    assert "raw tool output" not in user_msg
    assert "system row" not in user_msg
    assert user_msg.endswith("MESSAGE: and their manager?")


def test_prompt_has_no_conversation_block_without_history():
    captured: dict = {}
    _plan(_router(_KNOWLEDGE, captured=captured), "who owns Enterprise Search?")
    user_msg = captured["messages"][-1]["content"]
    assert "CONVERSATION SO FAR" not in user_msg
    assert user_msg.endswith("MESSAGE: who owns Enterprise Search?")


def test_time_scope_parses_only_a_complete_iso_window():
    """The classifier resolves an asked period to concrete dates; code only
    checks the SHAPE (fail-open — a half or malformed window must read as
    'no window asked', never as a constraint the user didn't set)."""
    from ai_graph_retrieval.retrieval.router import _time_scope

    ok = _time_scope({"time_scope_from": "2026-02-09", "time_scope_to": "2026-08-09"})
    assert ok == {"time_scope_from": "2026-02-09", "time_scope_to": "2026-08-09"}
    assert _time_scope({}) == {}
    assert _time_scope({"time_scope_from": "2026-02-09", "time_scope_to": ""}) == {}
    assert _time_scope({"time_scope_from": "recently", "time_scope_to": "2026-08-09"}) == {}
    # inverted window is dropped too
    assert _time_scope({"time_scope_from": "2026-08-09", "time_scope_to": "2026-02-09"}) == {}


def test_agent_input_states_the_asked_window_as_data():
    from ai_graph_retrieval.agent.prompts import agent_input

    text = agent_input("what changed?", now="x", time_scope=("2026-02-09", "2026-08-09"))
    assert "ASKED TIME WINDOW: 2026-02-09 to 2026-08-09" in text
    assert "ASKED TIME WINDOW" not in agent_input("what changed?", now="x")


# ── answer_shape: the answer's outline, decided from the question ─────────────
# The classifier reads how the ANSWER should be organised off the QUESTION,
# before anything is retrieved, so the same question is answered in the same
# shape every time it is asked. Synthesis applies that verdict; it never
# re-derives a shape from whatever the evidence happened to look like this run.


@pytest.mark.parametrize("shape", ["timeline", "table", "per_member", "prose"])
def test_the_classifiers_shape_verdict_reaches_the_plan(shape):
    out = _plan(_router({**_KNOWLEDGE, "answer_shape": shape}))
    assert isinstance(out, RetrievalPlan)
    assert out.answer_shape is AnswerShape(shape)


@pytest.mark.parametrize("bad", [
    "grouped_by_month",   # a value from a drifting provider
    "",                   # emitted but empty
    None,                 # emitted as null
    123,                  # not even a string
])
def test_an_unusable_shape_degrades_to_no_opinion_rather_than_failing(bad):
    """Fail-OPEN, unlike `intent`. A wrong lane silently buys a research question
    a shallow answer, so the router fails loud on it; a wrong shape costs only
    consistency, and refusing to answer would be the worse trade. None ⇒ the
    synthesis prompt's own precedence rules decide, which is exactly how every
    answer was shaped before this field existed."""
    out = _plan(_router({**_KNOWLEDGE, "answer_shape": bad}))
    assert isinstance(out, RetrievalPlan)
    assert out.answer_shape is None


def test_a_missing_shape_is_not_an_error():
    """The field is `required` in the tool schema, but a provider that drops it
    must not take the request down with it."""
    out = _plan(_router(_KNOWLEDGE))          # no answer_shape key at all
    assert isinstance(out, RetrievalPlan)
    assert out.answer_shape is None


def test_the_tool_schema_offers_exactly_the_shapes_synthesis_can_render():
    """The schema is built from the enum, so the two cannot drift — a shape the
    model could return but synthesis has no block for would silently degrade to
    "no opinion" on every question that landed on it."""
    from ai_graph_retrieval.retrieval.router import _CLASSIFY_TOOL
    from ai_graph_retrieval.synthesis.prompts import SYNTHESIS_SHAPE_ADDENDA

    params = _CLASSIFY_TOOL["function"]["parameters"]
    offered = set(params["properties"]["answer_shape"]["enum"])
    assert offered == {s.value for s in AnswerShape}
    assert offered == {s.value for s in SYNTHESIS_SHAPE_ADDENDA}
    # Required, so the model states a shape on every knowledge turn rather than
    # omitting it on the questions it finds hardest to place.
    assert "answer_shape" in params["required"]


def test_the_shape_is_decided_before_retrieval_and_the_prompt_says_so():
    """The whole value of the field is that it depends on the QUESTION and not on
    what came back — that is what makes the same question get the same shape
    twice. If the prompt ever invites the model to wait and see, the guarantee is
    gone."""
    from ai_graph_retrieval.retrieval.router import _SYSTEM

    assert "STEP 4 — answer_shape" in _SYSTEM
    assert "from the\nquestion alone" in _SYSTEM
    assert "the same question asked twice is\nanswered in the same shape both times" in _SYSTEM
    # prose is the safe default: structure imposed on a question that did not ask
    # for it reads worse than plain paragraphs.
    assert "`prose` is the default and the safe answer" in _SYSTEM


def test_the_shape_verdict_is_logged_so_it_can_be_checked(capsys):
    """The field is not on the response payload, so without a log line the only
    way to see what shape was chosen is to infer it from the finished answer —
    which is circular when the thing you are verifying IS the answer's shape.

    capsys, not caplog: structlog writes to stdout rather than through the
    stdlib logging handlers pytest's caplog captures."""
    _plan(_router({**_KNOWLEDGE, "answer_shape": "timeline"}))
    out = capsys.readouterr().out
    assert "answer_shape_chosen" in out
    assert "shape=timeline" in out


def test_a_conversational_turn_logs_no_shape(capsys):
    """It never reaches synthesis, so there is no shape to record."""
    _plan(_router({**_KNOWLEDGE, "category": "greeting", "reply": "Hi!",
                   "answer_shape": "prose"}), "hi")
    assert "answer_shape_chosen" not in capsys.readouterr().out
