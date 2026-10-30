"""Unit tests for the conversation ownership guard."""
from __future__ import annotations

import asyncio

from ai_graph_retrieval.conversation.ownership import ConversationOwnershipGuard
from ai_graph_retrieval.settings import Settings


def _guard() -> ConversationOwnershipGuard:
    # No redis_host → in-memory backend.
    s = Settings(
        neo4j_uri="bolt://x",
        neo4j_password="x",  # test fixture, not a real credential
        enforce_conversation_ownership=True,
    )
    return ConversationOwnershipGuard(s)


def test_ownership_first_owner_binds_and_matches():
    g = _guard()
    assert asyncio.run(g.authorize("t1", "conv1", "alice")) is True  # first use binds
    assert asyncio.run(g.authorize("t1", "conv1", "alice")) is True  # same owner ok


def test_ownership_rejects_different_owner():
    g = _guard()

    async def scenario() -> tuple[bool, bool]:
        first = await g.authorize("t1", "conv1", "alice")
        second = await g.authorize("t1", "conv1", "bob")
        return first, second

    first, second = asyncio.run(scenario())
    assert first is True
    assert second is False  # cross-user leak blocked


def test_ownership_tenant_isolated():
    g = _guard()

    async def scenario() -> tuple[bool, bool]:
        a = await g.authorize("t1", "conv1", "alice")
        # Same conversation_id under a different tenant is a distinct binding.
        b = await g.authorize("t2", "conv1", "bob")
        return a, b

    a, b = asyncio.run(scenario())
    assert a is True
    assert b is True


def test_ownership_noops_without_ids():
    g = _guard()
    assert asyncio.run(g.authorize("t1", "", "alice")) is True
    assert asyncio.run(g.authorize("t1", "conv1", "")) is True
