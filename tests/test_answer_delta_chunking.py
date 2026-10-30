"""`answer_delta` re-slicing — smooth streaming regardless of provider chunking.

The gateway's stream chunks can arrive as multi-hundred-character blocks, which
render as jumpy block-appends in the UI. `ProgressReporter.answer_delta` now
re-slices every chunk to at most `answer_delta_max_chars` characters before
emitting, at the one point all answer text flows through — so SSE events and
SSE frames get the same granularity, and the joined text is unchanged.

    uv run pytest tests/test_answer_delta_chunking.py -v
"""
from __future__ import annotations

import asyncio

from ai_graph_retrieval.core.models import GraphTraceBuilder
from ai_graph_retrieval.core.progress import ProgressReporter
from tests.support import make_settings


def _deltas(text: str, max_chars: int) -> list[str]:
    emitted: list[str] = []

    async def emit(event) -> None:
        emitted.append(event.delta)

    reporter = ProgressReporter(
        emit, GraphTraceBuilder(), answer_delta_max_chars=max_chars
    )
    asyncio.run(reporter.answer_delta(text))
    return emitted


def test_a_big_provider_chunk_is_split_to_the_cap():
    text = "x" * 100
    deltas = _deltas(text, 24)
    assert all(len(d) <= 24 for d in deltas)
    assert "".join(deltas) == text  # nothing lost, nothing reordered


def test_a_small_chunk_passes_through_as_one_event():
    assert _deltas("short", 24) == ["short"]


def test_zero_disables_the_split():
    text = "y" * 500
    assert _deltas(text, 0) == [text]


def test_empty_deltas_are_still_dropped():
    assert _deltas("", 24) == []


def test_the_setting_exists_and_defaults_on():
    assert make_settings().answer_delta_max_chars == 24
