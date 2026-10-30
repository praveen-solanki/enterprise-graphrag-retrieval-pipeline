"""One human gets one name, across graphs.

The bug, measured on the dev graphs (2026-08-27): 609 people exist in both graphs
and 209 under DIFFERENT labels, so the same person reached the answer twice —
"max.fields" (an Atlassian account label) beside "Max Fields", and an answer that
opened by explaining that it held both "Tom Baker" and "Tom Henry Baker", both
VP Engineering, one email.

These tests pin the four properties that make the rewrite safe to apply blind:

  * humans are grouped on an EXACT key (email) and the label is chosen by a
    configured source precedence — no name is ever compared to another name;
  * a name shared by two humans is left alone, because renaming there would merge
    two people (the same bug, inverted);
  * the rewrite covers entity keys AND every reference to them — facts, claims and
    chunk mentions — since a fact pointing at a name no entity carries is worse
    than the duplication;
  * any failure leaves every name exactly as it was.

    uv run pytest tests/test_person_identity.py -v
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from ai_graph_retrieval.core.models import (
    ChunkQuote,
    ClaimFact,
    EntityHit,
    EvidencePool,
    RelationFact,
)
from ai_graph_retrieval.graph.repository import _canonical_person_names
from ai_graph_retrieval.retrieval.person_identity import (
    candidate_names,
    canonicalize_person_names,
)
from ai_graph_retrieval.retrieval.person_status import PersonIdentities
from tests.support import make_people_settings as make_settings
from tests.support import span_attrs

# The real case, as it exists on the dev graphs.
TOM = "tom.baker@example.com"
AMAN = "max.fields@example.com"
AUTHORITY = {"wikipedia": 2, "arxiv": 1}


def _row(name: str, email: str, graph: str, degree: int = 0) -> dict:
    return {"name": name, "person_email": email, "graph": graph, "degree": degree}


# ── choosing the canonical name ─────────────────────────────────────────────


def test_the_graph_with_the_higher_authority_names_the_person():
    rows = [
        _row("Tom Henry Baker", TOM, "arxiv"),
        _row("Tom Baker", TOM, "wikipedia"),
        _row("max.fields", AMAN, "arxiv"),
        _row("Max Fields", AMAN, "wikipedia"),
    ]
    assert _canonical_person_names(rows, AUTHORITY) == {
        "Tom Henry Baker": "Tom Baker",
        "max.fields": "Max Fields",
    }


def test_only_actual_renames_are_returned():
    """A name that already agrees across graphs is not a rename, so the pool is
    never rewritten for it."""
    rows = [_row("Toby Cole", "toby@x", "wikipedia"),
            _row("Toby Cole", "toby@x", "arxiv")]
    assert _canonical_person_names(rows, AUTHORITY) == {}


def test_reversing_the_precedence_reverses_the_choice():
    """The precedence is configuration, not a judgement about the strings — so it
    is reversible, and nothing in the code prefers one graph."""
    rows = [_row("Tom Henry Baker", TOM, "arxiv"),
            _row("Tom Baker", TOM, "wikipedia")]
    assert _canonical_person_names(rows, {"wikipedia": 1, "arxiv": 2}) == {
        "Tom Baker": "Tom Henry Baker"}


def test_equal_authority_falls_back_to_degree_then_to_row_order():
    """Equal ranks disable the source preference without disabling the join: the
    graph's own signal decides, and identical degrees keep fan-out order."""
    flat = {"wikipedia": 0, "arxiv": 0}
    # degree decides
    rows = [_row("Tom Baker", TOM, "wikipedia", degree=2),
            _row("Tom Henry Baker", TOM, "arxiv", degree=9)]
    assert _canonical_person_names(rows, flat) == {"Tom Baker": "Tom Henry Baker"}
    # nothing decides ⇒ first row seen, deterministically
    rows = [_row("Tom Baker", TOM, "wikipedia"),
            _row("Tom Henry Baker", TOM, "arxiv")]
    assert _canonical_person_names(rows, flat) == {"Tom Henry Baker": "Tom Baker"}


def test_within_one_graph_the_primary_record_wins_on_degree():
    """MOST duplicates are not cross-graph. Ingestion emits the same human as
    several nodes in ONE graph — a primary record plus "(Deactivated)"/
    "(Unlicensed)" twins sharing an email — and on the dev arXiv graph 215
    emails carry more than one node, degree separating 201.

    A live run without this tiebreak picked the twin, which is how the pass
    briefly made three names WORSE ('Paul Reed' -> 'Paul Reed
    (Unlicensed)'). Degree is the graph's signal for the real record; no string is
    inspected, so the suffix itself is never what decides.
    """
    rows = [
        _row("Paul Reed (Unlicensed)", "chris@x", "arxiv", degree=5),
        _row("Paul Reed", "chris@x", "arxiv", degree=7),
    ]
    assert _canonical_person_names(rows, AUTHORITY) == {
        "Paul Reed (Unlicensed)": "Paul Reed"}


def test_source_authority_still_outranks_degree():
    """A high-degree arXiv node does not outrank the graph that carries the
    people directory — the ranks are the outer key, degree only breaks ties."""
    rows = [_row("max.fields", AMAN, "arxiv", degree=99),
            _row("Max Fields", AMAN, "wikipedia", degree=1)]
    assert _canonical_person_names(rows, AUTHORITY) == {"max.fields": "Max Fields"}


def test_a_name_shared_by_two_humans_is_never_renamed():
    """The guard that stops this pass merging two people. The arXiv graph
    holds 1,111 person nodes under 878 distinct account ids, so shared display
    names are real."""
    rows = [
        _row("Sam Carter", "sam.one@x", "arxiv"),
        _row("Sam Carter", "sam.two@x", "arxiv"),
        _row("Sam Lee Carter", "sam.one@x", "wikipedia"),
    ]
    assert _canonical_person_names(rows, AUTHORITY) == {}


def test_a_person_with_no_email_is_left_alone():
    """186 of the arXiv graph's person nodes carry no email. Without the join
    key there is nothing to join on, and a name is never guessed."""
    rows = [_row("someone.odd", "", "arxiv"),
            _row("Someone Odd", "", "wikipedia")]
    assert _canonical_person_names(rows, AUTHORITY) == {}


# ── rewriting the pool ──────────────────────────────────────────────────────


def _pool() -> EvidencePool:
    pool = EvidencePool()
    pool.add_entities([
        EntityHit(name="Tom Henry Baker", type="Person", profile="VP, Engineering"),
        EntityHit(name="Tom Baker", type="Person", url="http://x/tom"),
    ])
    pool.add_facts([
        RelationFact(source="Tom Henry Baker", target="Autonomous Agent Builder",
                     relationship="authored"),
        RelationFact(source="Tom Baker", target="Shift left", relationship="authored"),
    ])
    pool.add_claims([ClaimFact(subject="Tom Henry Baker", predicate="holds role",
                               object="VP")])
    pool.add_quotes([ChunkQuote(chunk_id="c1", doc_id="d1", doc_title="Doc",
                                text="…", via=["Tom Henry Baker", "Search"])])
    return pool


def test_renaming_collapses_the_duplicate_and_keeps_both_sides_fields():
    pool = _pool()
    assert pool.rename_entities({"Tom Henry Baker": "Tom Baker"}) == 1
    ev = pool.snapshot()
    assert [e.name for e in ev.entities] == ["Tom Baker"]
    only = ev.entities[0]
    # the union, exactly as if both had arrived under one name to begin with
    assert only.profile == "VP, Engineering"
    assert only.url == "http://x/tom"


def test_every_reference_follows_the_rename():
    """A fact pointing at a name no entity carries is worse than the duplication
    it replaced, so the rewrite is one operation over the whole pool."""
    pool = _pool()
    pool.rename_entities({"Tom Henry Baker": "Tom Baker"})
    ev = pool.snapshot()
    assert {f.source for f in ev.facts} == {"Tom Baker"}
    assert len(ev.facts) == 2                      # two documents, one author
    assert [c.subject for c in ev.claims] == ["Tom Baker"]
    assert ev.quotes[0].via == ["Tom Baker", "Search"]


def test_facts_that_become_identical_keep_one_row():
    pool = EvidencePool()
    pool.add_facts([
        RelationFact(source="max.fields", target="Search", relationship="authored"),
        RelationFact(source="Max Fields", target="Search", relationship="authored"),
    ])
    pool.rename_entities({"max.fields": "Max Fields"})
    assert len(pool.snapshot().facts) == 1


def test_an_empty_or_self_mapping_changes_nothing():
    pool = _pool()
    assert pool.rename_entities({}) == 0
    assert pool.rename_entities({"Tom Baker": "Tom Baker"}) == 0
    assert len(pool.snapshot().entities) == 2


# ── the pass ────────────────────────────────────────────────────────────────


def test_candidate_names_include_relationship_endpoints():
    """A person routinely enters the pool ONLY as a fact endpoint, with no entity
    row of their own — and those endpoints are the labels the duplicate-person
    answer was built from."""
    names = candidate_names(_pool().snapshot())
    assert "Tom Henry Baker" in names
    assert "Autonomous Agent Builder" in names   # the read filters, not this
    assert names == list(dict.fromkeys(names))   # ordered, no repeats


class _Repo:
    def __init__(self, mapping=None, fail: bool = False) -> None:
        self.mapping = mapping or {}
        self.fail = fail
        self.asked: list[str] = []

    async def person_identities(self, ctx, names):
        if self.fail:
            raise RuntimeError("neo4j down")
        self.asked = list(names)
        return PersonIdentities(renames=self.mapping)


def _sc(pool: EvidencePool, repo: _Repo, **overrides):
    return SimpleNamespace(
        evidence=pool, repo=repo, ctx=SimpleNamespace(log_fields=lambda: {}),
        settings=make_settings(**overrides), historical=False,
    )


def test_the_pass_rewrites_the_pool_and_reports_the_collapse(spans):
    pool = _pool()
    sc = _sc(pool, _Repo({"Tom Henry Baker": "Tom Baker"}))
    asyncio.run(canonicalize_person_names(sc))

    assert [e.name for e in pool.snapshot().entities] == ["Tom Baker"]
    at = span_attrs(spans, "stage.person_identity")
    assert at["renamed"] == 1
    assert at["entities_before"] == 2
    assert at["entities_after"] == 1
    assert at["entities_collapsed"] == 1


def test_a_failed_read_leaves_every_name_untouched(spans):
    pool = _pool()
    sc = _sc(pool, _Repo(fail=True))
    asyncio.run(canonicalize_person_names(sc))          # must not raise

    assert {e.name for e in pool.snapshot().entities} == {"Tom Baker",
                                                         "Tom Henry Baker"}
    assert span_attrs(spans, "stage.person_identity")["skipped"] == "read_failed"


def test_the_cap_bounds_the_read(spans):
    pool = EvidencePool()
    pool.add_entities([EntityHit(name=f"P{i}") for i in range(10)])
    repo = _Repo()
    asyncio.run(canonicalize_person_names(
        _sc(pool, repo, person_identity_max_names=3)))
    assert len(repo.asked) == 3
    assert span_attrs(spans, "stage.person_identity")["cap"] == 3


@pytest.mark.parametrize(("cap", "reason"), [(0, "disabled")])
def test_zero_disables_the_pass(spans, cap, reason):
    pool = _pool()
    repo = _Repo({"Tom Henry Baker": "Tom Baker"})
    asyncio.run(canonicalize_person_names(
        _sc(pool, repo, person_identity_max_names=cap)))
    assert repo.asked == []
    assert len(pool.snapshot().entities) == 2
    assert span_attrs(spans, "stage.person_identity")["skipped"] == reason


# ── the two reads ───────────────────────────────────────────────────────────

