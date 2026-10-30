"""Post-loop MENTION top-up (retrieval/deep_research.enrich_mentions).

A passage reaches the pool with the entities it MENTIONS as bare names on the
chunk; nothing added them as entities, so an answer could name people the pool
never held and the client's evidence graph showed none of them. After the loop
the lane hydrates those entities in one read and adds the relationships AMONG
the pool's entities in a second — additive-only, bounded by the profile.

    uv run pytest tests/test_mention_enrichment.py -v
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from ai_graph_retrieval.core.models import (
    ChunkQuote,
    EntityHit,
    EvidencePool,
    RelationFact,
)
from ai_graph_retrieval.retrieval.deep_research import enrich_mentions


class _Repo:
    def __init__(self, *, fail: bool = False, facts_fail: bool = False) -> None:
        self.mention_calls: list[tuple[list[str], int]] = []
        self.fact_calls: list[tuple[list[str], int]] = []
        self._fail = fail
        self._facts_fail = facts_fail

    async def mentioned_entities(self, ctx, chunk_weights, *, limit):
        if self._fail:
            raise RuntimeError("graph down")
        self.mention_calls.append((dict(chunk_weights), limit))
        return [
            EntityHit(name="Alan Turing", type="PERSON", description="Security engineer",
                      score=2.0, via=["mention"], graph="wikipedia"),
            EntityHit(name="Grace Hopper", type="PERSON", score=1.0, via=["mention"],
                      graph="wikipedia"),
            # Already in the pool under a description: the merge keeps it, adds nothing.
            EntityHit(name="Security", type="CONCEPT", score=1.0, via=["mention"]),
        ][:limit]

    async def facts_among(self, ctx, names, *, limit):
        if self._facts_fail:
            raise RuntimeError("graph down")
        self.fact_calls.append((list(names), limit))
        return [RelationFact(source="Alan Turing", target="Grace Hopper",
                             relationship="REPORTS_TO", direction="outgoing")]


def _sc(pool: EvidencePool, repo: _Repo, *, cap: int = 40,
        fact_limit: int = 200) -> SimpleNamespace:
    deltas: list[dict] = []

    async def graph_delta(**kwargs):
        deltas.append(kwargs)

    sc = SimpleNamespace(
        evidence=pool, repo=repo,
        ctx=SimpleNamespace(log_fields=lambda: {}),
        profile=SimpleNamespace(mention_enrich_top=cap),
        settings=SimpleNamespace(agent_induced_facts_limit=fact_limit),
        progress=SimpleNamespace(graph_delta=graph_delta, span_parent=None),
    )
    sc.deltas = deltas
    return sc


def _pool_with_chunks() -> EvidencePool:
    pool = EvidencePool()
    pool.add_entities([EntityHit(name="Security", type="CONCEPT", description="The team")])
    pool.add_quotes([
        ChunkQuote(chunk_id="g1", doc_id="d1", doc_title="Alan Turing",
                   text="Alan is a Security Operations Engineer.", graph="wikipedia",
                   relevance=0.71),
        ChunkQuote(chunk_id="g2", doc_id="d2", doc_title="Who's who",
                   text="Grace Hopper heads security.", graph="arxiv"),
    ])
    return pool


def test_mentioned_entities_join_the_pool_and_get_connected():
    pool = _pool_with_chunks()
    repo = _Repo()
    sc = _sc(pool, repo)
    asyncio.run(enrich_mentions(sc))

    # Graph chunk ids only, each with the relevance the ranking weighs by.
    assert repo.mention_calls == [({"g1": 0.71, "g2": 0.0}, 40)]
    names = {e.name for e in pool.snapshot().entities}
    assert {"Alan Turing", "Grace Hopper", "Security"} <= names
    # The induced-subgraph read covers the passage-grounded set the read
    # returned — never resolution candidates no passage mentions.
    assert repo.fact_calls[0][0] == ["Alan Turing", "Grace Hopper", "Security"]
    facts = pool.snapshot().facts
    assert [(f.source, f.relationship, f.target) for f in facts] == [
        ("Alan Turing", "REPORTS_TO", "Grace Hopper"),
    ]
    # The UI's trace delta carries the new nodes and edges, never the merged one.
    [delta] = sc.deltas
    assert {e.name for e in delta["entities"]} == {"Alan Turing", "Grace Hopper"}
    assert len(delta["facts"]) == 1


def test_an_entity_already_pooled_keeps_its_fields():
    pool = _pool_with_chunks()
    asyncio.run(enrich_mentions(_sc(pool, _Repo())))
    [security] = [e for e in pool.snapshot().entities if e.name == "Security"]
    assert security.description == "The team"


def test_the_cap_bounds_the_read():
    pool = _pool_with_chunks()
    repo = _Repo()
    asyncio.run(enrich_mentions(_sc(pool, repo, cap=1)))
    assert repo.mention_calls == [({"g1": 0.71, "g2": 0.0}, 1)]
    assert "Grace Hopper" not in {e.name for e in pool.snapshot().entities}


def test_zero_cap_disables_the_pass_entirely():
    """Cap 0 means no read at all."""
    pool = _pool_with_chunks()
    repo = _Repo(fail=True)  # would raise if reached
    sc = _sc(pool, repo, cap=0)
    sc.progress = None
    asyncio.run(enrich_mentions(sc))
    assert repo.mention_calls == []


def test_no_graph_chunks_means_no_read():
    """A Search run at graph_chunk_cap=0 holds no graph chunks to read from."""
    pool = EvidencePool()
    repo = _Repo(fail=True)
    asyncio.run(enrich_mentions(_sc(pool, repo)))
    assert repo.mention_calls == []


def test_a_failed_read_never_costs_the_answer():
    pool = _pool_with_chunks()
    before = pool.size()
    asyncio.run(enrich_mentions(_sc(pool, _Repo(fail=True))))
    assert pool.size() == before


def test_a_failed_facts_read_keeps_the_hydrated_entities():
    pool = _pool_with_chunks()
    asyncio.run(enrich_mentions(_sc(pool, _Repo(facts_fail=True))))
    assert "Alan Turing" in {e.name for e in pool.snapshot().entities}
    assert pool.snapshot().facts == []


def test_zero_fact_limit_skips_only_the_facts_read():
    pool = _pool_with_chunks()
    repo = _Repo()
    asyncio.run(enrich_mentions(_sc(pool, repo, fact_limit=0)))
    assert repo.fact_calls == []
    assert "Alan Turing" in {e.name for e in pool.snapshot().entities}


# ── the repository's ranking: relevance shared per mention ───────────────────


def _repo_stub(rows):
    from types import SimpleNamespace

    from tests.support import make_settings

    async def _rows(query, ctx, **params):
        return rows

    return SimpleNamespace(_rows=_rows, _s=make_settings(), _principals=lambda ctx: None)


def test_a_focused_passage_outranks_a_list_chunk_that_scored_slightly_higher():
    """Measured live: the top-scoring passage was a marketing datasheet naming
    forty entities; a raw count and a best-passage rule both filled the cap with
    investors and dropped the one person the answer named. Sharing each
    passage's relevance among the entities it mentions is what ranks the
    person, mentioned by two focused passages, above every datasheet name."""
    from ai_graph_retrieval.graph.repository import GraphReadRepository

    datasheet = [f"Investor {i}" for i in range(40)]
    rows = [{"name": n, "type": "ORGANIZATION", "degree": 500, "chunk_ids": ["sheet"]}
            for n in datasheet]
    rows.append({"name": "Alan Turing", "type": "PERSON", "degree": 12,
                 "chunk_ids": ["profile", "whos-who"]})
    rows.append({"name": "Grace Hopper", "type": "PERSON", "degree": 20,
                 "chunk_ids": ["whos-who"]})
    weights = {"sheet": 0.523, "profile": 0.469, "whos-who": 0.515}
    hits = asyncio.run(GraphReadRepository.mentioned_entities(
        _repo_stub(rows), None, weights, limit=3))
    assert [h.name for h in hits] == ["Alan Turing", "Grace Hopper", "Investor 0"]
    # 0.469/1 + 0.515/2 for Ian; each investor shares 0.523 forty ways.
    assert hits[0].score == round(0.469 + 0.515 / 2, 4)
    assert hits[2].score == round(0.523 / 40, 4)
    assert hits[0].via == ["mention"]


def test_unscored_chunks_fall_back_to_count_then_degree():
    from ai_graph_retrieval.graph.repository import GraphReadRepository

    rows = [
        {"name": "Twice", "degree": 1, "chunk_ids": ["a", "b"]},
        {"name": "Once Big", "degree": 99, "chunk_ids": ["a"]},
        {"name": "Once Small", "degree": 1, "chunk_ids": ["b"]},
    ]
    hits = asyncio.run(GraphReadRepository.mentioned_entities(
        _repo_stub(rows), None, {"a": 0.0, "b": 0.0}, limit=10))
    assert [h.name for h in hits] == ["Twice", "Once Big", "Once Small"]
