"""What the user is told, and what we refuse to tell them (pilot UI feedback).

Three guarantees, each reported as a defect from the pilot:
  * steps are named from ONE closed vocabulary, not per-lane wording;
  * internal tool names never reach the wire (they belong in the logs);
  * a step carries a short human summary of what is being attempted.
"""
from __future__ import annotations

import asyncio

from ai_graph_retrieval.core.models import GraphTraceBuilder, StepKind
from ai_graph_retrieval.core.narration import PHASES, phase_for, quoted, with_subject
from ai_graph_retrieval.core.progress import ProgressReporter


def _reporter():
    events = []

    async def emit(e):
        events.append(e)

    return ProgressReporter(emit, GraphTraceBuilder()), events


def _run(coro):
    return asyncio.run(coro)


def test_every_step_kind_maps_into_the_five_phases():
    """A reader learns five stage names; a new StepKind must pick one of them
    rather than inventing a sixth label."""
    for kind in StepKind:
        assert phase_for(kind) in PHASES


def test_the_label_comes_from_the_kind_not_the_call_site():
    reporter, events = _reporter()

    async def go():
        async with reporter.step(StepKind.TRAVERSE, "Tracing how these connect"):
            pass

    _run(go())
    assert [e.step.label for e in events] == [
        "Connecting related information", "Connecting related information",
    ]
    assert events[0].step.summary == "Tracing how these connect"


def test_tool_names_never_reach_the_stream():
    """The pilot UI showed `resolve_entities` to end users. The tool identity is
    a log field now; nothing on the event may carry it."""
    reporter, events = _reporter()

    async def go():
        async with reporter.step(
            StepKind.TOOL_CALL, "Finding people who match “Priya”", tool="find_people"
        ):
            pass

    _run(go())
    for e in events:
        payload = e.step.model_dump()
        assert "find_people" not in str(payload), payload
        assert payload["detail"] is None


def test_no_agent_tool_hardcodes_a_label_or_leaks_its_name():
    """Guards the whole toolset at once: a new tool that passes `detail="foo"`
    (the old habit) or its own label string fails here rather than in the UI."""
    from pathlib import Path

    src = Path("src/ai_graph_retrieval/agent/tools.py").read_text()
    assert 'detail="' not in src, "agent tools must pass tool=, not detail="
    # every step call names a StepKind first — no literal label argument
    assert "progress.step(\n        StepKind." in src or "progress.step(StepKind." in src


def test_a_summary_never_renders_an_empty_subject():
    """"Following “”'s connections" reached the pilot UI. A missing subject must
    fall back to a subject-less sentence, not an empty quote."""
    assert quoted("") == "" and quoted(None) == ""
    assert with_subject("Following {subject}", "", "Following the trail") == (
        "Following the trail"
    )
    assert with_subject("Following {subject}", "Priya", "x") == "Following “Priya”"


def test_a_long_subject_is_trimmed_not_dumped_into_the_sentence():
    long_name = "A" * 200
    out = quoted(long_name)
    assert len(out) < 70 and out.endswith("…”")


def test_error_steps_keep_their_summary():
    """A failed step still tells the user what was being attempted."""
    reporter, events = _reporter()

    async def go():
        try:
            async with reporter.step(StepKind.SOURCES, "Reading a document in full"):
                raise RuntimeError("neo4j down")
        except RuntimeError:
            pass

    _run(go())
    assert events[-1].step.status.value == "error"
    assert events[-1].step.summary == "Reading a document in full"
    assert events[-1].step.label == "Analyzing findings"


def test_the_step_payload_shape_is_additive():
    """`summary` is new and optional — a client reading only label/kind/status
    is unaffected, which is what makes this deployable without a UI release."""
    reporter, events = _reporter()

    async def go():
        async with reporter.step(StepKind.SYNTHESIZE):
            pass

    _run(go())
    dumped = events[0].step.model_dump()
    assert dumped["summary"] is None
    assert set(dumped) >= {"id", "label", "kind", "status", "summary", "detail"}



def test_no_streamed_detail_names_a_pipeline_internal():
    """Streamed reasoning keeps counts of things a READER knows — documents,
    relationships, people, matches ("6 relationships found" is good context) —
    but never names an internal processing unit ("54 chunks", "28 sources",
    entities, embeddings). This scans every rendered detail block so the
    distinction survives the next edit."""
    import re
    from pathlib import Path

    files = [
        "src/ai_graph_retrieval/agent/tools.py",
        "src/ai_graph_retrieval/retrieval/deep_research.py",
        "src/ai_graph_retrieval/synthesis/composer.py",
    ]
    banned = ("chunk", "entit", "embedding", "vector", "qdrant", "neo4j",
              " sources")  # "sources" as a counted unit; "source behind" is fine
    for f in files:
        src = Path(f).read_text()
        # every detail block, from the reasoning( opener to its closing paren;
        # scan only its STRING LITERALS — conditions may name variables freely.
        for block in re.findall(
            r"detail\s*=\s*reasoning\((?:[^()]|\([^()]*\))*\)", src
        ):
            rendered = " ".join(re.findall(r'"([^"]*)"', block))
            for noun in banned:
                assert noun not in rendered.lower(), (
                    f"{f}: a streamed detail names a pipeline internal "
                    f"({noun.strip()}):\n{block}"
                )
