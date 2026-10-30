"""The orchestrator's durable-persistence hook: a COMPLETED turn is stored (so the
WS-fallback can read it back by ai_message_id), while a PAUSED or FAILED turn is
not (so the fallback correctly re-runs). Also covers the id-threading contract —
the stored turn keys on the exact ids the route handed the client.

Built on a bare orchestrator instance (only the attributes ``retrieve()`` touches
are set) with ``_retrieve_admitted`` monkeypatched, so the test exercises the
envelope — admission, the success/pause/fail branches, the persist hook — without
standing up the whole pipeline."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from ai_graph_retrieval.clarification.models import MidResearchClarification
from ai_graph_retrieval.core.access import AccessContext
from ai_graph_retrieval.core.exceptions import RetrievalTimeout
from ai_graph_retrieval.orchestrator import RetrievalOrchestrator
from ai_graph_retrieval.retrieval.base import Mode
from tests.support import make_settings


class _FakeStore:
    """Records the lifecycle calls in order so a test can assert the sequence."""

    def __init__(self) -> None:
        self.events: list[tuple] = []

    async def open_turn(self, **kwargs) -> None:
        self.events.append(("open", kwargs))

    async def mark_generating(self, ai_message_id) -> None:
        self.events.append(("generating", ai_message_id))

    async def complete_turn(self, ai_message_id, payload) -> None:
        self.events.append(("completed", ai_message_id, payload))

    async def set_status(self, ai_message_id, status, *, error_code=None,
                         error_message=None) -> None:
        self.events.append(("status", ai_message_id, status, error_code))

    @property
    def kinds(self) -> list[str]:
        return [e[0] for e in self.events]


def _orch(store, admitted):
    """A bare orchestrator with just the fields retrieve() reads, and a stubbed
    _retrieve_admitted (returns a payload, or raises)."""
    orch = object.__new__(RetrievalOrchestrator)
    orch._s = make_settings()
    orch._admission = asyncio.Semaphore(4)
    orch._memory = None
    orch._store = store
    orch._recorder = SimpleNamespace(record=lambda *a, **k: None)
    orch._retrieve_admitted = admitted
    return orch


def _ctx() -> AccessContext:
    return AccessContext(tenant_id="t1", principals=("public",), request_id="r1")


def _run(orch):
    return asyncio.run(
        orch.retrieve(
            ctx=_ctx(),
            question="who owns the roadmap?",
            history=[],
            mode=Mode.SEARCH,
            include_trace=False,
            emit=None,
            conversation_id="c1",
            user_message_id="um1",
            ai_message_id="am1",
            owner="u1",
        )
    )


def test_completed_turn_opens_then_completes_with_threaded_ids():
    store = _FakeStore()
    seen = {}

    async def admitted(**kwargs):
        seen.update(kwargs)  # capture what retrieve() threaded in
        return {"answer": "hi", "citations": [], "evidence": {}, "usage": {}}

    payload = _run(_orch(store, admitted))

    # Lifecycle order: open up front, complete on success.
    assert store.kinds == ["open", "completed"]
    open_kw = store.events[0][1]
    assert open_kw["ai_message_id"] == "am1"
    assert open_kw["user_message_id"] == "um1"
    assert open_kw["conversation_id"] == "c1"
    assert open_kw["user_id"] == "u1"
    assert open_kw["question"] == "who owns the roadmap?"
    assert open_kw["mode"] == "search"

    # complete_turn keyed on the same ai id; payload carries the stamped ids.
    _, ai_id, completed_payload = store.events[1]
    assert ai_id == "am1"
    assert completed_payload["ai_message_id"] == "am1"
    assert completed_payload["user_message_id"] == "um1"
    assert payload["ai_message_id"] == "am1"

    # ai_message_id is threaded into _retrieve_admitted (where mark_generating fires).
    assert seen["ai_message_id"] == "am1"


def test_paused_turn_is_marked_clarification():
    store = _FakeStore()

    async def admitted(**kwargs):
        raise MidResearchClarification(questions=[])

    with pytest.raises(MidResearchClarification):
        _run(_orch(store, admitted))
    assert store.kinds == ["open", "status"]
    assert store.events[1][1:] == ("am1", "clarification", None)


def test_failed_turn_is_marked_failed_with_code():
    store = _FakeStore()

    async def admitted(**kwargs):
        raise RetrievalTimeout("budget exceeded")

    with pytest.raises(RetrievalTimeout):
        _run(_orch(store, admitted))
    assert store.kinds == ["open", "status"]
    # RetrievalTimeout.code == "timeout"
    assert store.events[1][1:] == ("am1", "failed", "timeout")
