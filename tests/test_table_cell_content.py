"""Table cells hold one value each; overflow becomes continuation rows (CI-3).

The defect: both synthesis prompts push people answers hard toward tables (the
roster rule, the people-portion rule, "WHERE A TABLE FITS, PREFER IT") and toward
exhaustive enumeration — but no rule anywhere said what a CELL may contain. A
Markdown table cell cannot render a line break, so a model committed to a table
whose item carries several values for one column (a person with several authored
documents and their dates) had no legal separator: it either crammed the values
into one run-on cell — "Document A date Document B date", where no reader can
tell which date belongs with which document — or abandoned the table into mixed
prose. Which one happened was per-run luck, observed live (audit CI-3: the same
per-member documents question produced valid continuation rows one run and
crammed cells another).

Every layer below the prompt is exonerated by construction: the digest renders
one record per line, citation resolution is newline-safe (its patterns forbid
newlines and its whitespace repair is tab/space-scoped), and the API ships the
answer string verbatim. The inconsistency is introduced where the prompt
contract underdetermines the rendering, so the fix is one cell-content rule in
the prompt: a cell holds one value, and several values for
one column are CONTINUATION ROWS — stated as the single legal form, with an
explicit carve-out so "one row per item/member" can never contradict it.

    uv run pytest tests/test_table_cell_content.py -v
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from ai_graph_retrieval.core.models import EntityHit, Evidence
from ai_graph_retrieval.retrieval.base import AnswerShape, Mode
from ai_graph_retrieval.synthesis.composer import AnswerComposer
from ai_graph_retrieval.synthesis.prompts import (
    SYNTHESIS_CITATION_ADDENDUM,
    SYNTHESIS_SHAPE_ADDENDA,
    SYNTHESIS_SYSTEM_PROMPT,
)
from tests.support import make_settings

_STANDARD = " ".join(SYNTHESIS_SYSTEM_PROMPT.split())

# The rule's load-bearing phrases, asserted wherever the rule must hold.
_RULE = "A TABLE CELL HOLDS ONE ITEM'S VALUE FOR ONE COLUMN"
_NO_BREAK = "no raw newline, no <br>"
_CONTINUATION = "CONTINUATION ROWS"
_ASSOCIATION = "visibly belongs to its item"


# ── the rule is in both prompts ──────────────────────────────────────────────


def test_the_standard_prompt_states_the_cell_contract():
    assert _RULE in _STANDARD
    assert _NO_BREAK in _STANDARD
    # The failure mode it exists to forbid — several values run together so the
    # reader cannot tell which value belongs with which.
    assert "never several values run together as continuous text" in _STANDARD


def test_the_standard_prompt_names_continuation_rows_as_the_one_legal_form():
    """The defect was the ABSENCE of a legal form, so the rule must state one —
    not merely forbid cramming — or the model is back to inventing a separator
    per run."""
    assert _CONTINUATION in _STANDARD
    assert "name cell left EMPTY" in _STANDARD
    assert _ASSOCIATION in _STANDARD


def test_the_cell_rule_carves_out_the_one_row_per_item_shapes():
    """The TABLE and PER_MEMBER addenda ask for "one row per item/member" and
    "no second table". Without an explicit carve-out, continuation rows would
    contradict them — and two rules disagreeing inside one prompt is a run-to-run
    coin toss (the exact lesson of the people-portion fix)."""
    assert "never a second entry, a second table, or a change of shape" in _STANDARD
    assert "one row per item or member is satisfied by them" in _STANDARD


# ── the surrounding contract is untouched ────────────────────────────────────


def test_the_shape_addenda_are_unchanged():
    """The addenda still say which shape a question landed on; what a cell may
    hold is stated once, in each base prompt's STRUCTURE block — never
    duplicated per shape."""
    for block in SYNTHESIS_SHAPE_ADDENDA.values():
        assert _RULE not in block
    assert "one row per member" in SYNTHESIS_SHAPE_ADDENDA[AnswerShape.PER_MEMBER]
    assert "ONE table for the whole set" in SYNTHESIS_SHAPE_ADDENDA[AnswerShape.TABLE]


def test_citations_still_stay_out_of_cells():
    """The other half of keeping a cell one value: the citation addendum owns
    citation placement and still forbids chips anywhere inside a table."""
    flat = " ".join(SYNTHESIS_CITATION_ADDENDUM.split())
    assert "inside a TABLE — not in a cell, not appended to a row" in flat


# ── the rule reaches the model whatever the mode and shape verdict ───────────


class _Progress:
    @asynccontextmanager
    async def step(self, *_a, **_k):
        yield SimpleNamespace()

    async def answer_delta(self, _token):
        pass


class _CapturingLlm:
    def __init__(self, captured: dict):
        self._captured = captured

    async def stream_text(self, messages, max_tokens=None, fast=False,
                          usage=None, **_kwargs):
        self._captured["system"] = messages[0]["content"]
        self._captured["user"] = messages[-1]["content"]
        yield "A long enough answer for the degenerate-answer guard to pass."


def _person(name: str, title: str, dept: str) -> EntityHit:
    return EntityHit(
        name=name, type="Person",
        profile=f"title: {title} · department: {dept} · location: Bengaluru",
    )


# A multi-person pool of the kind the defect's answers were composed from.
# Names are fixture data, nothing keys on them.
_PEOPLE_EVIDENCE = Evidence(entities=[
    _person("Eva Stone", "Engineering Manager", "Platform"),
    _person("Victor Hale", "Lead Software Engineer", "Platform"),
    _person("Mila Irwin", "Senior Data Scientist", "Data Science"),
])


def _compose(question: str, mode: Mode, shape: AnswerShape | None) -> dict:
    captured: dict = {}
    sc = SimpleNamespace(
        question=question, history=[], progress=_Progress(),
        llm=_CapturingLlm(captured), usage=SimpleNamespace(),
        ctx=SimpleNamespace(log_fields=lambda: {}),
    )
    asyncio.run(AnswerComposer(make_settings()).compose(
        sc, _PEOPLE_EVIDENCE, mode=mode, answer_shape=shape,
    ))
    return captured


@pytest.mark.parametrize("mode", list(Mode))
@pytest.mark.parametrize("shape", [None, *AnswerShape])
def test_every_mode_and_shape_verdict_carries_the_cell_contract(mode, shape):
    """The defect was per-run inconsistency, so the guarantee is path coverage:
    whichever base prompt the mode selects and whichever shape the classifier
    ruled — including the PER_MEMBER verdict rosters now settle to — the model
    is told what a cell may hold and what the overflow form is."""
    captured = _compose(
        "list the documents each member of the platform team has authored",
        mode, shape,
    )
    system = " ".join(captured["system"].split())
    assert _RULE in system, (mode, shape)
    assert _CONTINUATION in system, (mode, shape)


def test_the_evidence_below_the_prompt_is_unchanged():
    """Information preservation: the digest still hands the model one labelled
    line per person — the fix constrains presentation only, never what evidence
    reaches the model."""
    captured = _compose(
        "who is on the platform team", Mode.SEARCH, AnswerShape.PER_MEMBER,
    )
    digest = captured["user"]
    for e in _PEOPLE_EVIDENCE.entities:
        (line,) = [row for row in digest.splitlines() if e.name in row]
        assert line.lstrip().startswith("- ")
        assert "title:" in line and "department:" in line


def test_the_existing_table_preference_and_guards_are_intact():
    """The cell rule constrains what a cell holds; it must not weaken the lean
    toward tables nor the guard against forcing them."""
    assert "WHERE A TABLE FITS, PREFER IT" in _STANDARD
    assert "A one-column table is not a table" in _STANDARD
