"""People embedded in a larger answer keep per-person structure.

The defect: a people enumeration that is only PART of a multi-part question
("how is our search improving — and who's building it?") escapes the router's
roster upgrade (no roster phrasing), lands on the classifier's ``prose``
default, and the base prompt's "never combine two shapes" rule then binds the
people section into a running paragraph — ``Name (title, details), Name (title,
details), …`` — where no reader can pick one person out. The PROSE addendum only
said a table "is welcome", so whether an answer structured its people flipped
run to run.

The structure is provably intact below the prompt (the digest renders one
labelled profile line per person), so the fix is where the loss happens: one
portion-scoped rule in BOTH synthesis prompts — several people with attributes
are per-person structure inside ANY shape — with an explicit carve-out in the
single-shape rule so the two can never contradict again.

    uv run pytest tests/test_people_portion_structure.py -v
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
    SYNTHESIS_SHAPE_ADDENDA,
    SYNTHESIS_SYSTEM_PROMPT,
)
from tests.support import make_settings

_STANDARD = " ".join(SYNTHESIS_SYSTEM_PROMPT.split())

# The rule's load-bearing phrases, asserted wherever the rule must hold.
_RULE = "A PEOPLE SET IS PER-PERSON STRUCTURE INSIDE ANY SHAPE"
_ONE_ROW = "one row per person"
_NO_THREADING = "NEVER thread several people"


# ── the rule is in the prompts, portion-scoped ───────────────────────────────


def test_the_standard_prompt_structures_people_inside_any_shape():
    assert _RULE in _STANDARD
    assert _ONE_ROW in _STANDARD
    # Portion-scoped, not another whole-answer shape: it must bind when the
    # people are one section of a larger prose answer.
    assert "ONE SECTION of a larger answer" in _STANDARD
    # The failure mode it exists to forbid — names and attributes concatenated
    # through a paragraph.
    assert _NO_THREADING in _STANDARD


def test_the_single_shape_rule_carves_out_the_people_rule():
    """"Never combine two shapes" is what previously bound a prose answer's
    people section into prose. The carve-out is what makes the two rules
    consistent instead of a run-to-run coin toss."""
    assert "Never combine two shapes" in _STANDARD
    assert "exception is the people rule" in _STANDARD


def test_the_prose_addendum_makes_people_structure_mandatory():
    """The addendum used to merely welcome a table while the base prompt forbade
    combining shapes — permissive against mandatory is how the format flipped.
    A PROSE verdict (the classifier's stated default) must now still structure
    its people portion."""
    block = SYNTHESIS_SHAPE_ADDENDA[AnswerShape.PROSE]
    assert "not optional" in block
    assert "one row per person" in block


def test_non_people_shape_addenda_are_unchanged():
    """The fix is scoped to people portions: the other shape verdicts keep their
    existing instructions verbatim."""
    assert "one \"## \" heading per period" in " ".join(
        SYNTHESIS_SHAPE_ADDENDA[AnswerShape.TIMELINE].split()
    )
    assert "ONE table for the whole set" in SYNTHESIS_SHAPE_ADDENDA[AnswerShape.TABLE]
    assert "one row per member" in SYNTHESIS_SHAPE_ADDENDA[AnswerShape.PER_MEMBER]


def test_existing_non_people_structure_rules_are_intact():
    """The surrounding STRUCTURE contract is untouched: shape is still chosen
    once, prose is still the fallback shape, and rosters still have their own
    whole-question rule."""
    assert "CHOOSE THE ANSWER'S SHAPE ONCE" in _STANDARD
    assert "OTHERWISE → short paragraphs" in _STANDARD
    assert "THE QUESTION ENUMERATES PEOPLE" in _STANDARD


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


# A multi-person pool of the kind the defect's answers were composed from —
# several people, each with a labelled structured profile. Names are fixture
# data, nothing keys on them.
_PEOPLE_EVIDENCE = Evidence(entities=[
    _person("Eva Stone", "Engineering Manager", "Platform"),
    _person("Victor Hale", "Lead Software Engineer", "Platform"),
    _person("Mila Irwin", "Senior Data Scientist", "Data Science"),
    _person("Rob Banks", "Principal Product Manager", "Product"),
    _person("Dana Mills", "Data Scientist", "Data Science"),
    _person("Carl Pate", "Senior Software Engineer", "Platform"),
])

# The defect's question class: a people enumeration embedded in a multi-part
# ask, in phrasings the roster heuristic does not (and need not) catch. The
# guarantee under test is phrasing-independent, which is what makes it general.
_EMBEDDED_PEOPLE_QUESTIONS = [
    "how is the company's own search getting smarter - and who's building it?",
    "what changed in enterprise search and who is behind it?",
    "tell me about the search roadmap and the people working on it",
]


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


@pytest.mark.parametrize("question", _EMBEDDED_PEOPLE_QUESTIONS)
def test_the_defect_conditions_reproduce_with_the_rule_present(question):
    """The reproduced failure path: an embedded-people question, a PROSE
    verdict, the research lane (the mode the report came from). The people
    reach the model one labelled line each — structure is intact below the
    prompt — and the prompt now carries the binding portion rule where before
    it carried the contradiction."""
    captured = _compose(question, Mode.SUPER_RESEARCH, AnswerShape.PROSE)

    # The digest hands the model per-person structured input: one ENTITIES
    # line per person, attributes labelled — so any prose mush is a
    # presentation choice, which the rule now forbids.
    digest = captured["user"]
    for e in _PEOPLE_EVIDENCE.entities:
        (line,) = [row for row in digest.splitlines() if e.name in row]
        assert line.lstrip().startswith("- ")
        assert "title:" in line and "department:" in line

    system = " ".join(captured["system"].split())
    assert _RULE in system
    assert "not optional" in system  # the PROSE addendum's people clause


@pytest.mark.parametrize("mode", list(Mode))
@pytest.mark.parametrize("shape", [None, *AnswerShape])
def test_every_mode_and_shape_verdict_still_binds_people_structure(mode, shape):
    """Equivalent questions must come back consistently structured HOWEVER the
    classifier ruled: the per-person rule has to reach the model in every
    (mode, shape) combination, including the PROSE-verdict path and the
    no-opinion path."""
    captured = _compose(_EMBEDDED_PEOPLE_QUESTIONS[0], mode, shape)
    system = " ".join(captured["system"].split())
    assert "one row per person" in system, (mode, shape)
    assert _RULE in system
    assert _NO_THREADING in system
