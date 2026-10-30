"""A complete enumeration must survive the digest intact.

The enumeration read deliberately keeps unprofiled members (repository
``_profiles_from_rows``, ``keep_unprofiled``) because the ROW IS THE MEMBER —
dropping one undercounts the roster. The digest then re-capped that set with a
content tier (profile → description → bare name), which put exactly those
members last, so ``digest_max_entities`` dropped them and a roster reported
``truncated: false`` came back short.

Also covered here: person dedup by stable identity, deterministic enrichment and
supporting-chunk selection, and explicit neighbour truncation.

    uv run pytest tests/test_enumeration_survives_digest.py -v
"""
from __future__ import annotations

import random

from ai_graph_retrieval.core.models import (
    EntityHit,
    Evidence,
    EvidencePool,
    RelationFact,
)
from ai_graph_retrieval.retrieval.deep_research import _ordered_names
from ai_graph_retrieval.synthesis.composer import AnswerComposer
from tests.support import make_settings


def _entities(section: str) -> list[str]:
    """The names the ENTITIES section actually rendered."""
    return [line.split(" (")[0][2:]
            for line in section.splitlines() if line.startswith("- ")]


def _section(digest: str, header: str) -> str:
    block = digest.split(f"{header}:\n", 1)[1]
    return block.split("\n\n", 1)[0]


# ── 1. the cap must not prefer profiled entities over roster members ─────────


def test_an_unprofiled_roster_member_is_not_dropped_for_a_profiled_concept():
    """The defect: a bare-name PERSON from the enumeration ranked BELOW every
    profiled non-person, so the cap dropped the roster and kept the concepts."""
    settings = make_settings()
    settings.digest_max_entities = 3
    ev = Evidence(entities=[
        EntityHit(name="Topic A", type="CONCEPT", description="a described topic"),
        EntityHit(name="Topic B", type="CONCEPT", description="another topic"),
        EntityHit(name="Topic C", type="CONCEPT", description="a third topic"),
        # A arXiv identity stub: a real member with nothing renderable.
        EntityHit(name="Ray Morgan", type="Person"),
    ])

    rendered = _entities(_section(
        AnswerComposer(settings)._digest(ev, strict=False), "ENTITIES"))

    assert "Ray Morgan" in rendered


def test_every_member_of_a_complete_enumeration_survives_the_cap():
    """The whole roster, none of it profiled, against a cap it fits inside."""
    settings = make_settings()
    settings.digest_max_entities = 10
    roster = [EntityHit(name=f"Member {i}", type="Person") for i in range(8)]
    noise = [EntityHit(name=f"Topic {i}", type="CONCEPT", profile="x")
             for i in range(20)]

    rendered = _entities(_section(
        AnswerComposer(settings)._digest(
            Evidence(entities=noise + roster), strict=False), "ENTITIES"))

    assert {m.name for m in roster} <= set(rendered)


def test_the_richest_people_still_lead_inside_the_people_tier():
    """Deterministic, not flattened: profile still outranks description inside
    the tier, so a cap that binds ON PEOPLE keeps the most useful rows."""
    settings = make_settings()
    settings.digest_max_entities = 2
    ev = Evidence(entities=[
        EntityHit(name="Bare", type="Person"),
        EntityHit(name="Described", type="Person", description="d"),
        EntityHit(name="Profiled", type="Person", profile="title: VP"),
    ])

    rendered = _entities(_section(
        AnswerComposer(settings)._digest(ev, strict=False), "ENTITIES"))

    assert rendered == ["Profiled", "Described"]


def test_non_people_keep_the_original_content_tiering():
    """Unrelated digest behaviour is unchanged: among non-people, a profile
    still beats a description, which still beats a bare name."""
    settings = make_settings()
    settings.digest_max_entities = 2
    ev = Evidence(entities=[
        EntityHit(name="Bare", type="CONCEPT"),
        EntityHit(name="Described", type="CONCEPT", description="d"),
        EntityHit(name="Profiled", type="CONCEPT", profile="p"),
    ])

    rendered = _entities(_section(
        AnswerComposer(settings)._digest(ev, strict=False), "ENTITIES"))

    assert rendered == ["Profiled", "Described"]


def test_the_cap_itself_is_still_enforced():
    """A safety limit, not removed — just no longer biased against people."""
    settings = make_settings()
    settings.digest_max_entities = 4
    ev = Evidence(entities=[EntityHit(name=f"P{i}", type="Person") for i in range(30)])

    rendered = _entities(_section(
        AnswerComposer(settings)._digest(ev, strict=False), "ENTITIES"))

    assert len(rendered) == 4


def test_the_digest_ordering_is_stable_for_a_given_pool():
    """The tiering is a stable sort, so pool order survives inside a tier — the
    documented intent (seed relevance first). Determinism therefore comes from
    the POOL being deterministic (tests/test_retrieval_determinism.py), and the
    digest must not add drift of its own on top of it."""
    people = [EntityHit(name=f"P{i}", type="Person",
                        profile=("t" if i % 3 == 0 else None)) for i in range(12)]
    settings = make_settings()
    settings.digest_max_entities = 6

    orders = {
        tuple(_entities(_section(
            AnswerComposer(settings)._digest(
                Evidence(entities=people), strict=False), "ENTITIES")))
        for _ in range(10)
    }
    assert len(orders) == 1


def test_a_deterministic_pool_yields_a_deterministic_digest():
    """End to end: the pool dedups/merges order-independently (CI-8), so filling
    it in any order and rendering gives one identical ENTITIES section."""
    rows = [
        EntityHit(name="Jane Doe", type="Person", entity_id="p@x.com",
                  profile="title: VP"),
        EntityHit(name="Ray Morgan", type="Person", entity_id="r@x.com"),
        EntityHit(name="Platform", type="CONCEPT", description="a topic"),
    ]
    settings = make_settings()
    settings.digest_max_entities = 10

    sections = set()
    for seed in range(15):
        shuffled = rows[:]
        random.Random(seed).shuffle(shuffled)
        pool = EvidencePool()
        pool.add_entities(shuffled)
        sections.add(tuple(sorted(_entities(_section(
            AnswerComposer(settings)._digest(pool.snapshot(), strict=False),
            "ENTITIES")))))
    assert len(sections) == 1


# ── 2. distinct people with one display name stay distinct ──────────────────


def test_two_humans_sharing_a_name_are_not_merged_when_identity_differs():
    pool = EvidencePool()
    pool.add_entities([
        EntityHit(name="Gary", type="Person", entity_id="a@x.com", profile="SDE"),
        EntityHit(name="Gary", type="Person", entity_id="b@x.com", profile="PM"),
    ])

    entities = pool.snapshot().entities
    assert len(entities) == 2
    assert {e.profile for e in entities} == {"SDE", "PM"}


def test_the_same_human_twice_is_still_one_member():
    """Dedup is not disabled — it is keyed on the identity instead of the name."""
    pool = EvidencePool()
    pool.add_entities([
        EntityHit(name="Jane Doe", type="Person", entity_id="p@x.com"),
        EntityHit(name="Jane Doe", type="Person", entity_id="p@x.com",
                  profile="title: VP"),
    ])

    (entity,) = pool.snapshot().entities
    assert entity.profile == "title: VP"      # still enriched in place


def test_rows_without_an_identity_still_dedup_by_name():
    """Most entities carry no id at all; their behaviour must not change."""
    pool = EvidencePool()
    pool.add_entities([
        EntityHit(name="Platform", type="CONCEPT"),
        EntityHit(name="Platform", type="CONCEPT", url="https://x/platform"),
    ])

    (entity,) = pool.snapshot().entities
    assert entity.url == "https://x/platform"


def test_entity_names_still_reports_names_not_keys():
    """The departed-people sweep reads this; a stable-identity key must not leak."""
    pool = EvidencePool()
    pool.add_entities([EntityHit(name="Jane Doe", type="Person",
                                 entity_id="p@x.com")])

    assert pool.entity_names == ["Jane Doe"]


def test_renaming_still_works_when_entities_are_identity_keyed():
    pool = EvidencePool()
    pool.add_entities([EntityHit(name="jane.doe", type="Person",
                                 entity_id="p@x.com")])
    pool.rename_entities({"jane.doe": "Jane Doe"})

    assert pool.entity_names == ["Jane Doe"]


# ── 3. enrichment selection is deterministic ────────────────────────────────


def test_profile_enrichment_candidates_do_not_depend_on_insertion_order():
    """`enrich_profiles` spends its cap off the head of this list."""
    facts = [RelationFact(source=f"P{i}", relationship="R", target=f"T{i}",
                          score=i / 10, support=i, structural=i % 2 == 0)
             for i in range(10)]
    entities = [EntityHit(name=f"E{i}", type="person", score=i / 10)
                for i in range(6)]

    orders = set()
    for seed in range(20):
        f, e = facts[:], entities[:]
        random.Random(seed).shuffle(f)
        random.Random(seed + 99).shuffle(e)
        orders.add(tuple(_ordered_names(
            Evidence(facts=f, entities=e), skip=set(), people_first=True)))
    assert len(orders) == 1


# ── 4. supporting chunks are selected deterministically ─────────────────────


def test_supporting_chunks_orders_before_its_per_anchor_slice():
    """`collect(DISTINCT c)[0..$per_anchor]` keeps a fixed COUNT, so without an
    order two identical requests supported the answer with different passages."""
    from ai_graph_retrieval.graph import cypher

    q = cypher.SUPPORTING_CHUNKS
    slice_at = q.index("[0..$per_anchor]")
    assert "ORDER BY c.chunk_id" in q[:slice_at]
    # The overall LIMIT is a second cut and needs its own order.
    assert "ORDER BY chunk_id" in q[slice_at:]


# ── 5. neighbour truncation is explicit ─────────────────────────────────────


def test_expand_entity_reports_when_the_neighbour_cap_bound():
    import inspect

    from ai_graph_retrieval.agent import tools

    src = inspect.getsource(tools)
    assert '"truncated": truncated,' in src
    # Both branches (scored and unscored) have to compute it.
    assert "truncated = len(candidates) > len(facts)" in src
    assert "truncated = len(facts) >= rc.settings.neighbor_limit" in src
