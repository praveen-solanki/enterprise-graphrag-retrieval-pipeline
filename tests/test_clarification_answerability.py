"""A pause the user cannot answer is worse than no pause at all.

Pilot reports: "the clarification appears but there are no questions / I can't
type anything, the run is stuck". The guarantee now lives in ONE place —
``build_question`` (used by the gate for the classifier's questions and by the
agent's request_clarification tool alike): every question that reaches the UI
carries at least two non-blank clickable options.
"""
from __future__ import annotations

from ai_graph_retrieval.clarification.models import build_question


def test_no_options_substitutes_an_answerable_pair():
    """`request_clarification` with no options once produced a pause with
    nothing to click — and `ResumeRequest.answers` needs at least one answered
    question, so the conversation could not be resumed at all."""
    q = build_question("Which Owen did you mean?", [])
    assert len(q.options) >= 2
    assert all(o.label.strip() for o in q.options)


def test_blank_option_labels_count_as_no_options():
    q = build_question("Which one?", ["", "   "])
    assert len(q.options) >= 2 and all(o.label.strip() for o in q.options)


def test_a_single_option_is_not_a_choice():
    q = build_question("Pick", ["Only one"])
    assert len(q.options) >= 2


def test_real_options_are_left_alone():
    q = build_question("Which team?", ["Platform", "Data", "Infra"])
    assert [o.label for o in q.options] == ["Platform", "Data", "Infra"]
