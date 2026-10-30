"""Identical queries must retrieve the identical evidence set.

Five confirmed sources of run-to-run drift, one section each:

1. the agent loop sent no temperature, so tool selection varied;
2. the chat client sent none, so synthesis and routing varied;
3. ``add_facts`` was first-writer-wins, so tool ORDER decided the FACTS set;
4. enrichment/top-up selection walked the pool in INSERTION order;
5. the blended re-rank had no tiebreak, so equal scores fell back to input order.

The answer's wording may still vary; the evidence reaching synthesis may not.

    uv run pytest tests/test_retrieval_determinism.py -v
"""
from __future__ import annotations

import asyncio
import random

from ai_graph_retrieval.core.models import (
    ClaimFact,
    EntityHit,
    Evidence,
    EvidencePool,
    RelationFact,
)
from ai_graph_retrieval.retrieval.deep_research import _ordered_names
from tests.support import make_settings


def _client(**over):
    from ai_graph_retrieval.connectors.llm.client import LlmClient

    return LlmClient(make_settings(**over), asyncio.Semaphore(1))


# ── 3. fact merging does not depend on which read arrived first ──────────────


def _authored(**over) -> RelationFact:
    base = {"source": "Jane Doe", "relationship": "AUTHORED", "target": "Charter"}
    return RelationFact(**{**base, **over})


def test_a_structural_edge_stays_structural_whichever_read_landed_first():
    """The defect: the traversal emits structural=True and the people reads did
    not, so the merged row depended on tool execution order — and `structural`
    is the FIRST key every ranking sorts on."""
    walked = _authored(structural=True, score=0.9)
    people_read = _authored(support=3, source_status="Active")

    first, second = EvidencePool(), EvidencePool()
    first.add_facts([walked])
    first.add_facts([people_read])
    second.add_facts([people_read])
    second.add_facts([walked])

    assert first.snapshot().facts[0].structural is True
    assert second.snapshot().facts[0].structural is True


def test_the_merged_fact_is_identical_in_both_arrival_orders():
    a = _authored(structural=True, score=0.9, target_type="Document")
    b = _authored(support=3, source_status="Active", date="2026-01-01",
                  date_basis="event", graph="wikipedia")

    first, second = EvidencePool(), EvidencePool()
    first.add_facts([a])
    first.add_facts([b])
    second.add_facts([b])
    second.add_facts([a])

    assert (first.snapshot().facts[0].model_dump()
            == second.snapshot().facts[0].model_dump())


def test_merging_keeps_the_strongest_value_of_each_signal():
    """Not "last writer wins" either — each field resolves order-independently."""
    pool = EvidencePool()
    pool.add_facts([_authored(support=1, score=0.2)])
    pool.add_facts([_authored(support=5, score=0.1, description="VP, Marketing")])

    (fact,) = pool.snapshot().facts
    assert (fact.support, fact.score) == (5, 0.2)
    assert fact.description == "VP, Marketing"


def test_a_genuinely_new_edge_is_still_reported_as_added():
    """add_facts returns what was NEW — the agent tools count on that to report
    what a round actually found, so merging must not inflate it."""
    pool = EvidencePool()
    assert len(pool.add_facts([_authored()])) == 1
    assert pool.add_facts([_authored(support=9)]) == []     # merged, not new


def test_the_typed_people_reads_emit_structural():
    """The three reads that produced schema-asserted edges without the flag."""
    import inspect

    from ai_graph_retrieval.graph import repository

    src = inspect.getsource(repository)
    # AUTHORED (x2: document_contributors, documents_by_author) + CONTRIBUTED_TO.
    assert src.count("structural=True") >= 4


# ── 4. enrichment picks by RANK, not by pool insertion order ─────────────────


def _evidence(seed: int) -> Evidence:
    """The same evidence, shuffled — what two identical runs differ by."""
    facts = [
        RelationFact(source=f"P{i}", relationship="R", target=f"T{i}",
                     score=i / 10, support=i, structural=i % 2 == 0)
        for i in range(8)
    ]
    entities = [
        EntityHit(name=f"E{i}", type="person", score=i / 10, degree=i)
        for i in range(5)
    ]
    rng = random.Random(seed)
    rng.shuffle(facts)
    rng.shuffle(entities)
    return Evidence(facts=facts, entities=entities)


def test_candidate_order_is_identical_however_the_pool_was_filled():
    """The defect directly: the budgets are spent off the HEAD of this list, so
    insertion order decided WHICH people got profile/claim enrichment."""
    orders = {
        tuple(_ordered_names(_evidence(seed), skip=set(), people_first=True))
        for seed in range(25)
    }
    assert len(orders) == 1


def test_structure_still_leads_the_candidate_order():
    """Deterministic must not mean re-ranked: the existing key (structure first,
    then score, support, name) is the one the digest already uses."""
    names = _ordered_names(_evidence(0), skip=set(), people_first=False)
    # T6 is structural with the highest score among structural rows.
    assert names[0] == "T6"


def test_skip_and_people_first_still_apply():
    ev = _evidence(0)
    names = _ordered_names(ev, skip={"T6", "P6"}, people_first=True)
    assert "T6" not in names and "P6" not in names
    # people_first hoists person-typed entities ahead of the rest, unchanged.
    assert names.index("E4") < names.index("T4")


# ── 5. the blended re-rank breaks ties deterministically ─────────────────────


def test_equal_blends_are_broken_by_name_not_by_input_order():
    from ai_graph_retrieval.retrieval.neighbor_scoring import NeighborScorer

    # Identical score/support/structural — only the name separates them.
    facts = [RelationFact(source="S", relationship="R", target=t, score=0.5)
             for t in ("delta", "alpha", "charlie", "bravo")]

    orders = set()
    for seed in range(15):
        shuffled = facts[:]
        random.Random(seed).shuffle(shuffled)
        orders.add(tuple(f.target for f in NeighborScorer._cheap_ranked(shuffled)))
    assert orders == {("alpha", "bravo", "charlie", "delta")}


def test_the_blended_key_carries_a_name_tiebreak():
    """The LLM tier's own sort, which fell back to enumerate() order."""
    import inspect

    from ai_graph_retrieval.retrieval import neighbor_scoring

    src = inspect.getsource(neighbor_scoring.NeighborScorer.rescore)
    assert 'fact.target or ""' in src


# ── 1 + 2. sampling is pinned on every chat surface ──────────────────────────


def test_the_chat_call_carries_the_temperature():
    """The chat client serves synthesis, routing and judging alike."""
    params = _client(llm_temperature=0.0)._chat_params(max_tokens=100)
    assert params["temperature"] == 0.0


def test_an_explicit_opt_out_still_wins():
    client = _client(llm_send_temperature=False)
    assert "temperature" not in client._chat_params(max_tokens=100)


def test_a_none_temperature_is_still_omitted():
    """None means "use the provider default" — an explicit configuration."""
    client = _client(llm_temperature=None)
    assert "temperature" not in client._chat_params(max_tokens=100)


def test_the_agent_loop_pins_the_same_temperature():
    """The agent chooses the TOOLS, so its sampling decides which evidence a run
    retrieves at all — it must follow the same switch as the chat client."""
    import inspect

    from ai_graph_retrieval.agent import runtime

    src = inspect.getsource(runtime.ResearchAgent.__init__)
    assert "settings.llm_send_temperature" in src
    assert "temperature=temperature" in inspect.getsource(runtime.ResearchAgent)


# ── the end-to-end property: same evidence in, same evidence out ─────────────


def test_repeated_identical_runs_reach_synthesis_with_the_same_evidence():
    """Search and Super Research share the pool and the digest, so the property
    is asserted on what both hand to synthesis: fill one pool per run with the
    same rows in a different order, and compare the snapshots."""
    facts = [
        RelationFact(source="Jane Doe", relationship="AUTHORED", target="Charter",
                     structural=True, score=0.8),
        RelationFact(source="Jane Doe", relationship="AUTHORED", target="Charter",
                     support=4, source_status="Active"),
        RelationFact(source="Ray Morgan", relationship="MEMBER_OF", target="Platform",
                     structural=True, score=0.5),
    ]
    claims = [ClaimFact(subject="Jane Doe", predicate="leads", object="Platform")]
    entities = [EntityHit(name="Jane Doe", type="person", score=0.9)]

    snapshots = set()
    for seed in range(20):
        rows = facts[:]
        random.Random(seed).shuffle(rows)
        pool = EvidencePool()
        pool.add_facts(rows)
        pool.add_claims(claims)
        pool.add_entities(entities)
        ev = pool.snapshot()
        snapshots.add(tuple(sorted(
            (f.source, f.relationship, f.target, f.structural, f.support, f.score)
            for f in ev.facts
        )))

    assert len(snapshots) == 1
