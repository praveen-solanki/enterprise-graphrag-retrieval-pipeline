"""Post-loop relationship top-up (retrieval/deep_research.enrich_relationships).

Reporting lines and team membership are structured edges between Person nodes,
so ``expand_entity`` is the only read that surfaces them. The agent's
instructions say so, but whether it spends a round on the traversal is a per-run
tool choice — measured, a run that skipped it came back with 27-36 people and
ZERO relationships between them. This pass gives that allowance the same
deterministic backstop the profile, claim and graph-chunk allowances already
have: what the toolset can reach stops depending on which tools the loop called.

    uv run pytest tests/test_relationship_enrichment.py -v
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from ai_graph_retrieval.core.models import EntityHit, EvidencePool, RelationFact
from ai_graph_retrieval.retrieval.deep_research import (
    _unconnected_people,
    enrich_relationships,
)
from ai_graph_retrieval.retrieval.profile import (
    search_profile,
    super_research_profile,
)
from tests.support import make_settings


class _Step:
    def __init__(self) -> None:
        self.detail: str | None = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class _Repo:
    """Records which names were expanded; returns the edges each one carries."""

    def __init__(self, edges: dict[str, list[RelationFact]] | None = None,
                 fail: bool = False) -> None:
        self.expanded: list[str] = []
        self._edges = edges or {}
        self._fail = fail

    async def expand_entity(self, ctx, name, relationship_types=None, limit=None):
        if self._fail:
            raise RuntimeError("graph down")
        self.expanded.append(name)
        return list(self._edges.get(name, []))


def _sc(pool: EvidencePool, repo: _Repo, cap: int) -> SimpleNamespace:
    async def graph_delta(**kwargs):
        return None

    return SimpleNamespace(
        evidence=pool,
        repo=repo,
        ctx=SimpleNamespace(log_fields=lambda: {}),
        profile=SimpleNamespace(relationship_enrich_top=cap),
        settings=make_settings(),
        progress=SimpleNamespace(step=lambda *a, **k: _Step(),
                                 graph_delta=graph_delta, span_parent=None),
    )


def _person(name: str) -> EntityHit:
    return EntityHit(name=name, type="Person")


def _roster() -> EvidencePool:
    """Three people found by the loop, with NO relationship between them —
    the measured shape this pass exists to repair."""
    pool = EvidencePool()
    pool.add_entities([_person("Manager A"), _person("Report B"), _person("Report C")])
    return pool


# ── the fix ─────────────────────────────────────────────────────────────────


def test_people_the_loop_never_connected_get_expanded():
    pool = _roster()
    repo = _Repo({"Manager A": [
        RelationFact(source="Report B", target="Manager A", relationship="REPORTS_TO"),
    ]})

    asyncio.run(enrich_relationships(_sc(pool, repo, cap=10)))

    assert set(repo.expanded) == {"Manager A", "Report B", "Report C"}
    facts = pool.snapshot().facts
    assert [(f.source, f.relationship, f.target) for f in facts] == [
        ("Report B", "REPORTS_TO", "Manager A"),
    ]


def test_the_answer_no_longer_depends_on_the_agent_spending_a_round():
    """The same roster, with and without the loop having called expand_entity:
    the structure on record must be the same either way."""
    edges = {"Manager A": [
        RelationFact(source="Report B", target="Manager A", relationship="REPORTS_TO"),
    ]}

    skipped = _roster()
    asyncio.run(enrich_relationships(_sc(skipped, _Repo(edges), cap=10)))

    called = _roster()
    called.add_facts(edges["Manager A"])
    asyncio.run(enrich_relationships(_sc(called, _Repo(edges), cap=10)))

    assert {(f.source, f.relationship, f.target) for f in skipped.snapshot().facts} == \
           {(f.source, f.relationship, f.target) for f in called.snapshot().facts}


def test_an_already_connected_person_is_not_re_expanded():
    pool = _roster()
    pool.add_facts([
        RelationFact(source="Report B", target="Manager A", relationship="REPORTS_TO"),
    ])
    repo = _Repo()

    asyncio.run(enrich_relationships(_sc(pool, repo, cap=10)))

    assert repo.expanded == ["Report C"]


def test_only_people_are_candidates():
    """The pass connects the run's PEOPLE; a topic's neighbourhood is the
    agent's call and its budget, not this backstop's."""
    pool = EvidencePool()
    pool.add_entities([_person("Manager A"), EntityHit(name="Payroll", type="CONCEPT")])
    repo = _Repo()

    asyncio.run(enrich_relationships(_sc(pool, repo, cap=10)))

    assert repo.expanded == ["Manager A"]


def test_edges_to_people_outside_the_pool_are_not_kept():
    """The pass connects the assembled set; it never widens it."""
    pool = _roster()
    repo = _Repo({"Manager A": [
        RelationFact(source="Manager A", target="Outsider Z", relationship="MANAGES"),
        RelationFact(source="Report B", target="Manager A", relationship="REPORTS_TO"),
    ]})

    asyncio.run(enrich_relationships(_sc(pool, repo, cap=10)))

    targets = {f.target for f in pool.snapshot().facts}
    assert "Outsider Z" not in targets
    assert {e.name for e in pool.snapshot().entities} == {
        "Manager A", "Report B", "Report C",
    }


def test_the_selector_is_query_agnostic_and_pool_ordered():
    pool = _roster()
    sc = _sc(pool, _Repo(), cap=2)

    assert _unconnected_people(sc, 2) == ["Manager A", "Report B"]


# ── the pass never costs the answer ─────────────────────────────────────────


def test_a_failed_expansion_never_costs_the_answer():
    pool = _roster()
    pool.add_facts([
        RelationFact(source="Report B", target="Manager A", relationship="REPORTS_TO"),
    ])

    asyncio.run(enrich_relationships(_sc(pool, _Repo(fail=True), cap=10)))

    assert len(pool.snapshot().facts) == 1


def test_a_failed_read_does_not_discard_the_batchs_successes():
    """Fault tolerance is per READ: one raising expansion and one hanging one
    must not cost the edges a third read already returned."""

    class _MixedRepo(_Repo):
        async def expand_entity(self, ctx, name, relationship_types=None,
                                limit=None):
            if name == "Report B":
                raise RuntimeError("graph down")
            if name == "Report C":
                await asyncio.sleep(3600)
            return await super().expand_entity(ctx, name)

    pool = _roster()
    repo = _MixedRepo({"Manager A": [
        RelationFact(source="Report B", target="Manager A", relationship="REPORTS_TO"),
    ]})
    sc = _sc(pool, repo, cap=10)
    sc.settings = make_settings(relationship_enrich_timeout_s=0.05)

    asyncio.run(enrich_relationships(sc))  # must not raise

    facts = pool.snapshot().facts
    assert [(f.source, f.relationship, f.target) for f in facts] == [
        ("Report B", "REPORTS_TO", "Manager A"),
    ], "the successful read's edges survive its siblings' failures"


def test_a_slow_batch_is_cut_at_its_own_bound_never_the_mode_deadline():
    """The pass carries its own asyncio.timeout, so a hung graph costs the
    edges and a bounded slice of wall-clock — never the shared retrieval
    deadline the remaining top-ups and synthesis run under."""

    class _HangingRepo(_Repo):
        async def expand_entity(self, ctx, name, relationship_types=None,
                                limit=None):
            await asyncio.sleep(3600)
            return []

    pool = _roster()
    sc = _sc(pool, _HangingRepo(), cap=10)
    sc.settings = make_settings(relationship_enrich_timeout_s=0.05)

    async def _bounded():
        # Far below the hang, comfortably above the pass's own bound: reaching
        # here proves the timeout fired inside the pass and was swallowed.
        async with asyncio.timeout(5):
            await enrich_relationships(sc)

    asyncio.run(_bounded())  # must not raise

    assert pool.snapshot().facts == []


def test_a_failed_trace_delta_never_costs_the_answer():
    async def boom(**kwargs):
        raise RuntimeError("ws down")

    pool = _roster()
    sc = _sc(pool, _Repo({"Manager A": [
        RelationFact(source="Report B", target="Manager A", relationship="REPORTS_TO"),
    ]}), cap=10)
    sc.progress.graph_delta = boom

    asyncio.run(enrich_relationships(sc))  # must not raise

    assert pool.snapshot().facts


def test_nothing_unconnected_means_no_read():
    pool = EvidencePool()
    pool.add_entities([_person("Manager A"), _person("Report B")])
    pool.add_facts([
        RelationFact(source="Report B", target="Manager A", relationship="REPORTS_TO"),
    ])
    repo = _Repo()

    asyncio.run(enrich_relationships(_sc(pool, repo, cap=10)))

    assert repo.expanded == []


# ── budgets, modes and ordering are unchanged ───────────────────────────────


def test_the_cap_bounds_the_batch_and_zero_disables_the_pass():
    pool = _roster()
    repo = _Repo()
    asyncio.run(enrich_relationships(_sc(pool, repo, cap=1)))
    assert repo.expanded == ["Manager A"]

    off = _Repo()
    asyncio.run(enrich_relationships(_sc(_roster(), off, cap=0)))
    assert off.expanded == []


def test_the_lane_runs_relationships_after_profiles_and_before_claims():
    """Ordering matters: enrich_profiles is what types a bare relationship
    target as a Person, and this pass selects on that type."""
    import inspect

    from ai_graph_retrieval.retrieval.deep_research import DeepResearchStrategy

    src = inspect.getsource(DeepResearchStrategy.run)
    assert (src.index("enrich_profiles(sc)")
            < src.index("enrich_relationships(sc)")
            < src.index("enrich_claims(sc)"))


def test_no_mode_budget_or_depth_knob_changed():
    """The pass adds a backstop; it must not buy any mode more retrieval."""
    s = make_settings()
    for profile in (search_profile(s), super_research_profile(s)):
        assert profile.agent_rounds == {
            "search": s.search_agent_rounds,
            "super_research": s.agent_max_rounds,
        }[profile.name]
        assert profile.graph_chunk_cap == {
            "search": s.search_graph_chunk_cap,
            "super_research": s.super_research_graph_chunk_cap,
        }[profile.name]
        assert profile.claim_enrich_top == s.agent_claim_enrich_top
