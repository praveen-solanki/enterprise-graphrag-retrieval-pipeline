"""Server-side conversation memory.

The store fills history ONLY when the client sent none, so a thin client gets
working follow-ups while every existing caller keeps its exact behavior. These
tests run the in-process fallback (no Redis in CI) and a raising fake for the
fail-open contract:

  * roundtrip: an appended exchange replays shaped exactly like client history,
  * the window: only the newest history_max_turns exchanges survive,
  * isolation: tenants and conversations never see each other's turns,
  * fail-open: store errors mean "no stored history", never a failed request,
  * disabled / blank inputs are no-ops.

    uv run pytest tests/test_conversation_memory.py -v
"""
from __future__ import annotations

import asyncio

from ai_graph_retrieval.conversation.memory import ConversationMemory
from tests.support import make_settings


class _Ctx:
    def __init__(self, tenant: str = "t-1") -> None:
        self.tenant_id = tenant
        self.request_id = "req-1"

    def log_fields(self) -> dict:
        return {"request_id": self.request_id}


def _memory(**overrides) -> ConversationMemory:
    # The feature ships behind enable_multi_turn_conversation (code default OFF
    # for the current release); these tests exercise the enabled behavior.
    overrides.setdefault("enable_multi_turn_conversation", True)
    mem = ConversationMemory(make_settings(**overrides))
    mem.start()  # no Redis configured in tests → in-process fallback
    return mem


def test_appended_exchange_replays_like_client_history():
    mem = _memory()
    asyncio.run(mem.append(_Ctx(), "c1", "Who is Ian Webb?",
                           "Ian Webb is a Senior Engineering Manager."))
    turns = asyncio.run(mem.history(_Ctx(), "c1"))
    assert turns == [
        {"role": "user", "content": "Who is Ian Webb?"},
        {"role": "assistant", "content": "Ian Webb is a Senior Engineering Manager."},
    ]


def test_only_the_newest_window_of_messages_survives():
    # history_max_turns counts MESSAGES (the orchestrator truncates
    # history[-history_max_turns:]), so the store keeps exactly that many rows.
    mem = _memory(history_max_turns=4)
    for i in range(5):
        asyncio.run(mem.append(_Ctx(), "c1", f"q{i}", f"a{i}"))
    turns = asyncio.run(mem.history(_Ctx(), "c1"))
    assert len(turns) == 4                                  # 4 messages = 2 exchanges
    assert [t["content"] for t in turns] == ["q3", "a3", "q4", "a4"]


def test_non_object_json_rows_are_skipped_not_fatal():
    # Valid JSON that is not an object (list/string/number) must be skipped like
    # any corrupt row — never raise on m.get(...).
    mem = _memory()
    asyncio.run(mem.append(_Ctx(), "c1", "q1", "a1"))
    key = next(iter(mem._local))
    exp, rows = mem._local[key]
    mem._local[key] = (exp, ['["a", "list"]', '"just a string"', "42", *rows])
    turns = asyncio.run(mem.history(_Ctx(), "c1"))
    assert [t["content"] for t in turns] == ["q1", "a1"]


def test_local_fallback_stays_bounded_even_when_nothing_expires():
    mem = _memory()
    for i in range(10_050):
        asyncio.run(mem.append(_Ctx(), f"c{i}", f"q{i}", f"a{i}"))
    assert len(mem._local) <= 10_000


def test_tenants_and_conversations_are_isolated():
    mem = _memory()
    asyncio.run(mem.append(_Ctx("t-1"), "c1", "q-t1", "a-t1"))
    assert asyncio.run(mem.history(_Ctx("t-2"), "c1")) == []      # other tenant
    assert asyncio.run(mem.history(_Ctx("t-1"), "c2")) == []      # other conversation
    assert len(asyncio.run(mem.history(_Ctx("t-1"), "c1"))) == 2  # own turns intact


def test_stored_content_is_capped_to_the_history_turn_contract():
    mem = _memory(conversation_memory_turn_chars=50)
    asyncio.run(mem.append(_Ctx(), "c1", "q", "x" * 5000))
    turns = asyncio.run(mem.history(_Ctx(), "c1"))
    assert len(turns[1]["content"]) == 50


def test_disabled_memory_is_a_complete_noop():
    mem = _memory(enable_multi_turn_conversation=False)
    asyncio.run(mem.append(_Ctx(), "c1", "q", "a"))
    assert asyncio.run(mem.history(_Ctx(), "c1")) == []
    assert mem.enabled is False


def test_blank_inputs_never_store():
    mem = _memory()
    asyncio.run(mem.append(_Ctx(), "", "q", "a"))       # no conversation id
    asyncio.run(mem.append(_Ctx(), "c1", "   ", "a"))   # blank question
    assert asyncio.run(mem.history(_Ctx(), "c1")) == []


def test_store_errors_fail_open_not_the_request():
    class _BrokenRedis:
        def pipeline(self):
            raise RuntimeError("redis down")

        async def lrange(self, *a, **k):
            raise RuntimeError("redis down")

    mem = _memory()
    mem._redis = _BrokenRedis()
    asyncio.run(mem.append(_Ctx(), "c1", "q", "a"))               # no raise
    assert asyncio.run(mem.history(_Ctx(), "c1")) == []           # no raise, empty


def test_corrupt_rows_are_skipped_not_fatal():
    mem = _memory()
    asyncio.run(mem.append(_Ctx(), "c1", "q", "a"))
    key = next(iter(mem._local))
    expires, rows = mem._local[key]
    mem._local[key] = (expires, ["not-json", *rows, '{"role": "tool", "content": "x"}'])
    turns = asyncio.run(mem.history(_Ctx(), "c1"))
    assert [t["content"] for t in turns] == ["q", "a"]  # corrupt + non-convo rows dropped
