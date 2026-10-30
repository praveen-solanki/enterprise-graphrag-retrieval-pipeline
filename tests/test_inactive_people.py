"""People who have left the organization never reach a current-people answer.

THE BUG. Departed colleagues were retrieved like anyone else and shipped to the
answer model with a flag on them, leaving it to decide whether a former employee
belongs in a current roster. By then the damage was already done twice over: the
person had spent a row of every ranked read's LIMIT and a slot of the digest's
entity cap, and the flag only ever rode on SOME of the evidence — the same name
still arrived as a claim object, as a chunk's prose, and as the title of their
own people-object document, all unmarked.

So the exclusion is a retrieval boundary in two halves, and these tests pin both:

  * IN CYPHER — every person-bearing read carries ``<<current_person:…>>`` before
    its LIMIT, so a departed person is never a candidate at all. The identity
    reads are the deliberate exception: they are what tells the pass below that a
    human is departed somewhere.
  * OVER THE POOL — the cross-graph sweep, which is the only thing that can see a
    human the intranet calls Active while arXiv holds their deactivated
    node, and the only thing that can reach a name arriving as free text.

Plus the two rules that keep it honest: UNKNOWN IS NOT DEPARTED (absence has an
established meaning on these graphs), and a HISTORICAL question keeps its people
— marked, so a past contributor is never read as a current one.

    uv run pytest tests/test_inactive_people.py -v
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import ClassVar

import pytest

from ai_graph_retrieval.core.models import (
    ChunkQuote,
    ClaimFact,
    EntityHit,
    Evidence,
    EvidencePool,
    RelationFact,
    SourceDocument,
)
from ai_graph_retrieval.graph import cypher
from ai_graph_retrieval.retrieval.base import Intent, RetrievalPlan, RetrievalStrategy
from ai_graph_retrieval.retrieval.person_status import (
    PersonIdentities,
    departed_person_names,
    exclude_inactive_people,
    is_inactive,
)
from ai_graph_retrieval.synthesis.composer import AnswerComposer
from tests.support import make_people_settings as make_settings
from tests.support import span_attrs

WIKIPEDIA, ARXIV = "wikipedia", "arxiv"

# One departed human and one current colleague, as the dev graphs hold them: the
# same person under two labels, one per graph, joined by email.
GONE, GONE_ALIAS = "Jane Doe", "jane.doe"
GONE_EMAIL = "jane.doe@example.com"
HERE, HERE_EMAIL = "Ray Morgan", "ray.morgan@example.com"
GONE_KEY = f"email:{GONE_EMAIL}"


def _row(name, email, graph, *, status=None, departed=None, degree=0) -> dict:
    return {"name": name, "person_email": email, "graph": graph, "degree": degree,
            "employment_status": status, "departed_on": departed}


# ── the status semantics ────────────────────────────────────────────────────
# One reading of the source systems' own fields, shared by the Cypher predicate
# and every Python row filter. Nothing here is a new status model.


@pytest.mark.parametrize("status", ["Inactive", "inactive", "Deactivated", "Terminated"])
def test_any_non_active_status_means_the_person_has_left(status):
    """'Active' is the one value that means still here. Every other non-empty
    value is read as departed rather than enumerated, so a new vocabulary word
    from a connector is excluded on arrival instead of passing as current."""
    assert is_inactive(status)


def test_a_recorded_departure_date_means_the_person_has_left():
    """`person_departed_on` is written ONLY for people who have left, so a value
    is the departure itself — and on a graph carrying no employment_status it is
    the only signal there is."""
    assert is_inactive(None, "2025-11-30")
    assert is_inactive("Active", "2025-11-30")


@pytest.mark.parametrize("status", [None, "", "  ", "Active", "ACTIVE"])
def test_a_missing_status_is_not_a_departure(status):
    """THE CONSERVATIVE HALF, and it is load-bearing. arXiv's person nodes
    are identity stubs with no attributes at all, and a NULL `person_departed_on`
    says "no departure recorded", never "this person is active". Reading either
    absence as a departure would delete every arXiv person from every
    answer."""
    assert not is_inactive(status)
    assert not is_inactive(status, None)
    assert not is_inactive(status, "")


# ── the Cypher gate ─────────────────────────────────────────────────────────


# Every read that can put a Person in front of the answer or spend a candidate
# slot on one, and the variable(s) each gate binds — the person it returns AND
# the anchor it is asked about (a departed person is not an anchor for current
# evidence either). A read added here without the gate is a new way in.
GATED = {
    "PEOPLE_RESOLVE": ("person",),           # find_people
    "PEOPLE_PROFILES": ("person",),          # the post-loop profile top-up
    "DOCUMENT_CONTRIBUTORS": ("person",),    # who wrote this
    "TOPIC_CONTRIBUTORS": ("person",),       # who works on this
    "DOCUMENTS_BY_AUTHOR": ("person",),      # what has this person written
    "EXPAND_NEIGHBORS": ("e", "nbr"),        # neighbour retrieval: seed + result
    "ENTITY_PATHS": ("n",),                  # structural/path: every node walked
    "BEAM_TRAVERSAL": ("s", "n", "nbr"),     # chunk beam: seed, path, co-mention
    "ENTITIES_BY_ID": ("e",),                # entity resolution (semantic arm)
    "RESOLVE_BY_FULLTEXT": ("e",),           # entity resolution (lexical arm)
    "FETCH_CLAIMS": ("e",),                  # claims anchored on the person
    "RECENT_CHANGES": ("e",),                # the changed entity can be a person
    "SOURCE_DOCUMENTS": ("e",),              # the mention anchor
    "SUPPORTING_CHUNKS": ("e",),             # the mention anchor
}


@pytest.mark.parametrize(("query", "gated_vars"), sorted(GATED.items()))
def test_every_person_bearing_read_gates_on_employment(query, gated_vars):
    prepared = cypher.prepare(getattr(cypher, query), "allow_access_control")
    for var in gated_vars:
        assert (
            f"toLower(coalesce({var}.person_employment_status, 'active')) = 'active'"
            in prepared
        ), f"{query} can reach a person nothing checked the employment status of"
        assert f"coalesce({var}.person_departed_on, '') = ''" in prepared
    assert "$include_inactive_people OR" in prepared


@pytest.mark.parametrize("query", sorted(GATED))
def test_the_gate_binds_before_the_limit_so_a_departed_person_spends_no_slot(query):
    """The whole point of gating in Cypher rather than filtering the rows after.
    A read that returns its LIMIT rows and drops the departed ones afterwards has
    already spent those slots — the active colleagues who would have filled them
    were never fetched."""
    prepared = cypher.prepare(getattr(cypher, query), "allow_access_control")
    gate = prepared.index("$include_inactive_people")
    limit = prepared.rfind("LIMIT")
    assert limit == -1 or gate < limit


@pytest.mark.parametrize("query", ["SUPPORTING_CHUNKS", "CHUNKS_BY_ID"])
def test_a_people_profile_chunk_is_gated_on_its_subject(query):
    """A people-object document is titled with the person's name and its text IS
    their profile — the purest "this person is here" evidence in the corpus, and
    it reaches the digest as an excerpt rather than as a person row."""
    prepared = cypher.prepare(getattr(cypher, query), "allow_access_control")
    assert "none(s IN ss WHERE s IS NOT NULL AND toLower(s) <> 'active')" in prepared
    assert "none(x IN sd WHERE coalesce(x, '') <> '')" in prepared


@pytest.mark.parametrize("query", ["PERSON_IDENTITIES", "PERSON_IDENTITIES_BY_EMAIL",
                                   "PEOPLE_PROFILES_BY_IDENTITY"])
def test_the_identity_reads_are_deliberately_not_gated(query):
    """The one exception, and it has to be one: these reads are what tell the
    cross-graph sweep that a human is departed in the OTHER graph. Gating them
    would hide exactly the rows the exclusion is looking for."""
    text = getattr(cypher, query)
    assert "<<current_person" not in text
    assert "$include_inactive_people" not in text


def test_the_identity_reads_carry_the_status_they_are_asked_for():
    for query in (cypher.PERSON_IDENTITIES, cypher.PERSON_IDENTITIES_BY_EMAIL):
        assert "person.person_employment_status AS employment_status" in query
        assert "person.person_departed_on AS departed_on" in query


def test_a_non_person_neighbour_is_untouched_by_the_gate():
    """Documents, concepts and teams carry no employment status. `coalesce(…,
    'active')` is what keeps the neighbour expansion from returning nothing at
    all once the gate is on."""
    prepared = cypher.prepare(cypher.EXPAND_NEIGHBORS, "allow_access_control")
    assert "coalesce(nbr.person_employment_status, 'active')" in prepared


# ── the gate is a property of the REQUEST ───────────────────────────────────


def test_the_repository_binds_the_gate_on_every_execution():
    """Bound beside $principals for the same reason: a read that could forget it
    is a read that can leak a former employee into a current-people answer."""
    from ai_graph_retrieval.graph.repository import GraphReadRepository

    bound: dict = {}

    class _Http:
        async def rows(self, prepared, params, database, log_fields):
            bound.update(params)
            return []

    repo = GraphReadRepository(None, make_settings(graph_sources_config=[
        {"name": "wikipedia", "database": "neo4j", "entities_collection": "e"}]), http=_Http())
    ctx = SimpleNamespace(principals=("p",), acl_enforced=False,
                          log_fields=lambda: {})
    asyncio.run(GraphReadRepository._run_one(
        repo, "MATCH (n) RETURN n", ctx, repo.sources[0], {}))
    assert bound["include_inactive_people"] is False


def test_a_historical_request_takes_a_view_with_the_gate_open():
    """One switch for the whole request, the same shape ``with_sources`` uses for
    run selection — never a flag each call site remembers to pass."""
    from ai_graph_retrieval.graph.repository import GraphReadRepository

    repo = GraphReadRepository(None, make_settings(), http=SimpleNamespace())
    assert repo.including_inactive_people(False) is repo      # the common case
    historical = repo.including_inactive_people(True)
    assert historical is not repo
    assert historical._include_inactive_people is True
    assert historical.sources == repo.sources


def test_run_selection_carries_the_gate_through():
    from ai_graph_retrieval.graph.repository import GraphReadRepository

    repo = GraphReadRepository(None, make_settings(), http=SimpleNamespace())
    scoped = repo.including_inactive_people(True).with_sources(repo.sources[:1])
    assert scoped._include_inactive_people is True


# ── which questions are historical ──────────────────────────────────────────


def test_only_the_classifiers_own_temporal_reading_opens_the_gate():
    """Read off what the classifier already decided, never off the question's
    text: a resolved window means the question named a period, and TEMPORAL is
    the intent for "what changed or happened within a time window"."""
    assert not RetrievalPlan(intent=Intent.ENTITY).historical
    assert not RetrievalPlan(intent=Intent.THEMATIC).historical
    assert not RetrievalPlan(intent=Intent.RESEARCH).historical
    assert RetrievalPlan(intent=Intent.TEMPORAL).historical
    assert RetrievalPlan(intent=Intent.ENTITY, time_scope_from="2024-01-01",
                         time_scope_to="2024-12-31").historical
    # A half window is dropped by the router before it gets here; if one ever
    # arrived it is not a period the question asked for.
    assert not RetrievalPlan(intent=Intent.ENTITY, time_scope_from="2024-01-01").historical


# ── cross-graph resolution ──────────────────────────────────────────────────


def test_one_graph_marking_a_person_departed_condemns_every_label():
    """The cross-graph rule. Wikipedia still says Active; arXiv holds the
    deactivated node for the same email. The human is not current, and BOTH
    labels go — including the one whose own row looked fine."""
    rows = [
        _row(GONE, GONE_EMAIL, WIKIPEDIA, status="Active"),
        _row(GONE_ALIAS, GONE_EMAIL, ARXIV, status="Inactive"),
        _row(HERE, HERE_EMAIL, WIKIPEDIA, status="Active"),
        _row(HERE, HERE_EMAIL, ARXIV),
    ]
    assert departed_person_names(rows) == frozenset({GONE, GONE_ALIAS})


def test_a_deactivated_node_condemns_the_human_in_every_graph():
    """The business definition (confirmed 2026-08-29): Deactivated means the
    person has LEFT the organization — never a license or account state — so one
    graph's Active row must not outvote another's Deactivated node for the same
    email. The human is suppressed globally, under every label they carry."""
    rows = [
        _row(GONE, GONE_EMAIL, WIKIPEDIA, status="Active"),
        _row(GONE_ALIAS, GONE_EMAIL, ARXIV, status="Deactivated"),
    ]
    assert departed_person_names(rows) == frozenset({GONE, GONE_ALIAS})


def test_a_human_active_in_every_graph_stays_available():
    """The converse: agreement that the person is Active leaves them alone."""
    rows = [
        _row(HERE, HERE_EMAIL, WIKIPEDIA, status="Active"),
        _row(HERE, HERE_EMAIL, ARXIV, status="Active"),
    ]
    assert departed_person_names(rows) == frozenset()


def test_the_join_is_on_identity_and_never_on_the_name():
    """Two humans can share a display name; grouping on it would condemn an
    active colleague for a stranger's departure."""
    rows = [
        _row("Sam Carter", "sam.carter@x", WIKIPEDIA, status="Inactive"),
        _row("Sam Carter", "sam.c@x", WIKIPEDIA, status="Active"),
    ]
    # The name is departed because ONE of the two humans carrying it is; that is
    # the honest answer from an identity join — and the guard against condemning
    # the other is that the label is shared, which the profile pass already
    # refuses to render. What must NOT happen is the reverse: an active human
    # surviving only because they share a label with a departed one.
    assert departed_person_names(rows) == frozenset({"Sam Carter"})


def test_a_row_with_no_identity_key_speaks_only_for_itself():
    rows = [
        _row("No Email Person", "", WIKIPEDIA, status="Inactive"),
        _row("Also No Email", "", WIKIPEDIA, status="Active"),
    ]
    assert departed_person_names(rows) == frozenset({"No Email Person"})


def test_a_departure_date_alone_is_enough_to_condemn_the_group():
    rows = [
        _row(GONE, GONE_EMAIL, WIKIPEDIA, departed="2025-11-30"),
        _row(GONE_ALIAS, GONE_EMAIL, ARXIV),
    ]
    assert departed_person_names(rows) == frozenset({GONE, GONE_ALIAS})


def test_nobody_is_departed_when_no_graph_records_a_status():
    """The graph may carry no status field at all. Silence is not a departure."""
    rows = [_row(HERE, HERE_EMAIL, WIKIPEDIA), _row(HERE, HERE_EMAIL, ARXIV)]
    assert departed_person_names(rows) == frozenset()


def test_the_resolution_keeps_both_spellings_of_a_departed_name():
    """The renames are applied to the pool first, so the structured rows carry
    the canonical label — but chunk prose and document titles are never
    rewritten, and they still spell the name the way their graph did. Dropping
    only the canonical form left exactly those free-text copies behind."""
    resolution = PersonIdentities(renames={GONE_ALIAS: GONE},
                                  departed=frozenset({GONE_ALIAS}))
    assert resolution.departed_labels() == {GONE, GONE_ALIAS}


# ── the pool sweep: every way a name can arrive ─────────────────────────────


def _leaky_pool() -> EvidencePool:
    """One departed person reaching the pool through every evidence type there
    is, and one active colleague who must survive all of it."""
    pool = EvidencePool()
    pool.add_entities([
        EntityHit(name=GONE, type="Person", profile="Staff Engineer"),
        EntityHit(name=HERE, type="Person", profile="Staff Engineer"),
    ])
    pool.add_facts([
        RelationFact(source=HERE, target=GONE, relationship="REPORTS_TO"),
        RelationFact(source=GONE, target="Search Platform",
                     relationship="contributed_to"),
        RelationFact(source=HERE, target="Search Platform",
                     relationship="contributed_to"),
    ])
    pool.add_claims([
        ClaimFact(subject=GONE, predicate="leads", object="Search Platform"),
        ClaimFact(subject="Search Platform", predicate="is owned by", object=GONE),
        ClaimFact(subject=HERE, predicate="leads", object="Ranking"),
    ])
    pool.add_sources([
        # A people object: the document IS the person.
        SourceDocument(doc_id="p-1", title=GONE, source_type="people"),
        SourceDocument(doc_id="d-1", title="Search Platform charter"),
    ])
    pool.add_quotes([
        ChunkQuote(chunk_id="c-1", doc_id="p-1", doc_title=GONE,
                   text="Staff Engineer, Search Platform.", subject_status="Inactive"),
        ChunkQuote(chunk_id="c-2", doc_id="d-1", doc_title="Search Platform charter",
                   text=f"The charter was written by {GONE} in 2024.", via=[GONE]),
        ChunkQuote(chunk_id="c-3", doc_id="d-1", doc_title="Search Platform charter",
                   text=f"{HERE} owns ranking quality.", via=[HERE]),
    ])
    return pool


def test_the_sweep_reaches_every_evidence_type_the_name_can_arrive_in():
    pool = _leaky_pool()
    removed = pool.drop_people({GONE})
    ev = pool.snapshot()
    blob = Evidence.model_dump_json(ev)
    assert GONE not in blob, "the departed person is still reachable in the pool"
    assert removed == {"entities": 1, "facts": 2, "claims": 2,
                       "sources": 1, "quotes": 2}


def test_the_sweep_keeps_everything_that_is_not_about_the_departed_person():
    pool = _leaky_pool()
    pool.drop_people({GONE})
    ev = pool.snapshot()
    assert [e.name for e in ev.entities] == [HERE]
    assert [(f.source, f.target) for f in ev.facts] == [(HERE, "Search Platform")]
    assert [c.subject for c in ev.claims] == [HERE]
    assert [s.title for s in ev.sources] == ["Search Platform charter"]
    assert [q.chunk_id for q in ev.quotes] == ["c-3"]


def test_the_sweep_matches_free_text_on_a_word_boundary():
    pool = EvidencePool()
    pool.add_quotes([
        ChunkQuote(chunk_id="c-1", text="Ask jane.doei@example.com about it."),
        ChunkQuote(chunk_id="c-2", text="Ask Jane Doe about it."),
    ])
    pool.drop_people({GONE, GONE_ALIAS})
    assert [q.chunk_id for q in pool.snapshot().quotes] == ["c-1"]


def test_a_single_token_name_is_never_matched_inside_prose():
    """A bare label is too ambiguous to censor prose on: dropping every excerpt
    containing it would cost far more evidence than the leak it closes. The
    structured rows still go."""
    pool = EvidencePool()
    pool.add_entities([EntityHit(name="Sam", type="Person")])
    pool.add_quotes([ChunkQuote(chunk_id="c-1", text="The same approach works.")])
    assert pool.drop_people({"Sam"}) == {"entities": 1}
    assert [q.chunk_id for q in pool.snapshot().quotes] == ["c-1"]


def test_a_source_title_that_contains_the_name_is_dropped_too():
    """The digest's SOURCES section prints titles verbatim, so "Meet Jane Doe"
    puts the name in the answer context even though the title is not the name."""
    pool = EvidencePool()
    pool.add_sources([
        SourceDocument(doc_id="d-1", title=f"Meet {GONE} — welcome post"),
        SourceDocument(doc_id="d-2", title="Search charter"),
    ])
    pool.drop_people({GONE})
    assert [s.title for s in pool.snapshot().sources] == ["Search charter"]


def test_a_fact_endpoint_that_contains_the_name_is_dropped_too():
    """Ingestion emits suffix twins — "Jane Doe (Unlicensed)" — which equality
    against the departed label would miss."""
    pool = EvidencePool()
    pool.add_facts([
        RelationFact(source=f"{GONE} (Unlicensed)", target="Search Platform",
                     relationship="contributed_to"),
        RelationFact(source=HERE, target="Search Platform",
                     relationship="contributed_to"),
    ])
    pool.drop_people({GONE})
    assert [f.source for f in pool.snapshot().facts] == [HERE]


def test_a_surviving_profile_is_scrubbed_of_the_departed_manager():
    """`person_manager` is a property on the REPORT's node, so an active person's
    profile line can name a departed manager — the one prose path no row filter
    reaches. The segment goes; the person's own role stays."""
    pool = EvidencePool()
    pool.add_entities([EntityHit(
        name=HERE, type="Person",
        profile=f"Engineer · Data Science · reports to {GONE} · 3 direct reports",
        description=f"Works closely with {GONE} on ranking.",
    )])
    removed = pool.drop_people({GONE})
    [e] = pool.snapshot().entities
    assert e.profile == "Engineer · Data Science · 3 direct reports"
    assert e.description is None
    assert removed == {"prose_scrubbed": 2}


def test_a_surviving_facts_description_is_scrubbed():
    pool = EvidencePool()
    pool.add_facts([RelationFact(
        source=HERE, target="Search Platform", relationship="contributed_to",
        description=f"Took over the charter {GONE} wrote.",
    )])
    pool.drop_people({GONE})
    [f] = pool.snapshot().facts
    assert f.description is None


def test_the_alias_spelling_still_reaches_prose_after_the_rename():
    """The end-to-end shape of the departed_labels fix: the pool was renamed to
    the canonical label, but a chunk still spells the name the arXiv way."""
    pool = EvidencePool()
    pool.add_quotes([ChunkQuote(chunk_id="c-1", doc_title="Search charter",
                                text=f"Written by {GONE_ALIAS} in 2024.")])
    resolution = PersonIdentities(renames={GONE_ALIAS: GONE},
                                  departed=frozenset({GONE_ALIAS}))
    pool.drop_people(resolution.departed_labels())
    assert pool.snapshot().quotes == []


def test_an_empty_set_changes_nothing():
    pool = _leaky_pool()
    assert pool.drop_people(set()) == {}
    assert pool.drop_people({"", "  "}) == {}
    assert len(pool.snapshot().entities) == 2


# ── the pass ────────────────────────────────────────────────────────────────


def _sc(pool: EvidencePool, *, historical: bool = False, **overrides):
    return SimpleNamespace(evidence=pool, historical=historical,
                           settings=make_settings(**overrides),
                           ctx=SimpleNamespace(log_fields=lambda: {}))


def test_a_current_question_loses_the_departed_person_before_anything_ranks(spans):
    sc = _sc(_leaky_pool())
    exclude_inactive_people(sc, PersonIdentities(departed=frozenset({GONE})))
    assert GONE not in Evidence.model_dump_json(sc.evidence.snapshot())
    attrs = span_attrs(spans, "stage.inactive_people")
    assert attrs["departed"] == 1 and attrs["historical"] is False
    assert attrs["removed_entities"] == 1


def test_a_historical_question_keeps_the_evidence_and_marks_it(spans):
    """Not destroying legitimate historical evidence is half the requirement; the
    other half is that a past contributor must not read as a current member."""
    sc = _sc(_leaky_pool(), historical=True)
    exclude_inactive_people(sc, PersonIdentities(departed=frozenset({GONE})))
    ev = sc.evidence.snapshot()
    assert GONE in Evidence.model_dump_json(ev)
    assert next(e for e in ev.entities if e.name == GONE).status == "Inactive"
    incoming = next(f for f in ev.facts if f.target == GONE)
    outgoing = next(f for f in ev.facts if f.source == GONE)
    assert incoming.target_status == "Inactive"
    assert outgoing.source_status == "Inactive"
    # The colleague who is still here gains nothing.
    assert next(e for e in ev.entities if e.name == HERE).status is None
    assert span_attrs(spans, "stage.inactive_people")["historical"] is True


def test_the_kill_switch_restores_the_previous_behaviour(spans):
    sc = _sc(_leaky_pool(), inactive_people_filter_enabled=False)
    exclude_inactive_people(sc, PersonIdentities(departed=frozenset({GONE})))
    assert GONE in Evidence.model_dump_json(sc.evidence.snapshot())
    assert span_attrs(spans, "stage.inactive_people")["skipped"] == "disabled"


def test_nothing_happens_when_nobody_has_left(spans):
    sc = _sc(_leaky_pool())
    exclude_inactive_people(sc, PersonIdentities())
    assert len(sc.evidence.snapshot().entities) == 2
    assert span_attrs(spans, "stage.inactive_people")["departed"] == 0


def test_the_cross_graph_verdict_suppresses_the_person_from_the_pool():
    """The whole cross-graph path in one piece: the identity rows say Active in
    one graph and Deactivated in the other for the same email, and the sweep
    takes the human out of the evidence under every label — while the colleague
    who is Active everywhere survives untouched."""
    rows = [
        _row(GONE, GONE_EMAIL, WIKIPEDIA, status="Active"),
        _row(GONE_ALIAS, GONE_EMAIL, ARXIV, status="Deactivated"),
        _row(HERE, HERE_EMAIL, WIKIPEDIA, status="Active"),
        _row(HERE, HERE_EMAIL, ARXIV, status="Active"),
    ]
    sc = _sc(_leaky_pool())
    exclude_inactive_people(sc, PersonIdentities(
        resolved=True, departed=departed_person_names(rows),
        asked=frozenset(n.casefold() for n in (GONE, GONE_ALIAS, HERE))))
    blob = Evidence.model_dump_json(sc.evidence.snapshot())
    assert GONE not in blob and GONE_ALIAS not in blob
    assert HERE in blob


# ── the slot the departed person no longer spends ───────────────────────────


def test_the_active_colleague_takes_the_slot_the_digest_cap_would_have_cut():
    """The pool-side half of "an inactive person must not consume a slot": the
    sweep runs BEFORE the digest's caps, so an active person who would have been
    truncated away is rendered instead."""
    s = make_settings(digest_max_entities=2)
    pool = EvidencePool()
    pool.add_entities([
        EntityHit(name=GONE, type="Person", profile="Staff Engineer"),
        EntityHit(name="Second Person", type="Person", profile="Engineer"),
        EntityHit(name=HERE, type="Person", profile="Engineer"),
    ])
    before, _, _ = AnswerComposer(s)._build_digest(pool.snapshot(), strict=False)
    assert HERE not in before                       # cut by the cap
    pool.drop_people({GONE})
    after, counts, _ = AnswerComposer(s)._build_digest(pool.snapshot(), strict=False)
    assert HERE in after and GONE not in after
    assert counts["entities_rendered"] == 2


# ── what the digest says when a departed person legitimately appears ────────


def test_the_digest_flags_the_person_at_whichever_end_of_the_edge_they_are():
    """Authorship and contribution edges are walked FROM the person, so flagging
    only the target printed "[INACTIVE]" beside the document they wrote."""
    ev = Evidence(facts=[
        RelationFact(source=GONE, target="Search charter", relationship="authored",
                     source_status="Inactive"),
        RelationFact(source=HERE, target=GONE, relationship="REPORTS_TO",
                     target_status="Inactive"),
    ])
    digest, _, _ = AnswerComposer(make_settings())._build_digest(ev, strict=False)
    assert f"{GONE} [INACTIVE — no longer with the organization] authored" in digest
    assert f"{GONE} [INACTIVE — no longer with the organization]" in digest
    assert "Search charter [INACTIVE" not in digest


def test_a_bare_person_entity_still_carries_the_flag():
    """A person reached as an entity with no profile line had nothing to carry
    the status; the row rendered as an unmarked current colleague."""
    ev = Evidence(entities=[EntityHit(name=GONE, type="Person", status="Inactive")])
    digest, _, _ = AnswerComposer(make_settings())._build_digest(ev, strict=False)
    assert f"- {GONE} (Person) [INACTIVE — no longer with the organization]" in digest


def test_the_flag_is_not_repeated_when_the_profile_line_already_leads_with_it():
    ev = Evidence(entities=[EntityHit(
        name=GONE, type="Person", status="Inactive",
        profile="FORMER EMPLOYEE — no longer with the organization · Staff Engineer")])
    digest, _, _ = AnswerComposer(make_settings())._build_digest(ev, strict=False)
    assert digest.count("no longer with the organization") == 1


def test_a_profile_from_the_graph_that_still_reads_active_does_not_hide_the_flag():
    """The cross-graph case the sweep exists for: the intranet still calls the
    person Active, so their profile line carries role and department but no
    FORMER EMPLOYEE — and `flag_people` stamps the status on exactly that row.
    Suppressing the flag on the profile's mere existence rendered them as an
    unmarked current colleague with a complete-looking profile."""
    pool = EvidencePool()
    pool.add_entities([EntityHit(
        name=GONE, type="Person",
        profile=f"Staff Engineer · Search Platform · reports to {HERE}")])
    assert pool.flag_people({GONE}) == 1
    digest, _, _ = AnswerComposer(make_settings())._build_digest(
        pool.snapshot(), strict=False)
    line = next(l for l in digest.splitlines() if l.startswith(f"- {GONE}"))
    assert "[INACTIVE — no longer with the organization]" in line
    assert "Staff Engineer" in line, "the profile itself stays — only the flag is added"


# ── the repository's person reads carry the status they were asked for ──────


def test_find_people_stamps_the_status_as_a_field_not_only_as_prose():
    """Only reachable on a historical request — the query's gate drops the row
    otherwise — but the pass and the digest both read the FIELD back."""
    from ai_graph_retrieval.graph.repository import GraphReadRepository

    async def _rows(query, ctx, **params):
        return [
            {"name": GONE, "full_name": GONE, "job_title": "Staff Engineer",
             "employment_status": "Inactive", "graph": WIKIPEDIA, "score": 1.0},
            {"name": HERE, "full_name": HERE, "job_title": "Engineer",
             "employment_status": "Active", "graph": WIKIPEDIA, "score": 0.9},
        ]

    stub = SimpleNamespace(_rows=_rows, _s=make_settings())
    hits = asyncio.run(GraphReadRepository.find_people(
        stub, SimpleNamespace(log_fields=lambda: {}), "engineer"))
    gone, here = hits
    assert gone.status == "Inactive"
    assert "INACTIVE — no longer with the organization" in gone.description
    assert here.status == "Active"


def test_the_contribution_reads_stamp_the_status_on_the_person_end():
    from ai_graph_retrieval.graph.repository import GraphReadRepository

    async def _rows(query, ctx, **params):
        return [{"person": GONE, "job_title": "Staff Engineer", "status": "Inactive",
                 "role": "AUTHORED", "via": "Search Platform", "document": "Charter",
                 "contributions": 3, "graph": WIKIPEDIA}]

    stub = SimpleNamespace(_rows=_rows, _s=make_settings(),
                           _involvement=GraphReadRepository._involvement.__get__(
                               SimpleNamespace(_s=make_settings())))
    ctx = SimpleNamespace(log_fields=lambda: {})
    for read in (GraphReadRepository.topic_contributors,
                 GraphReadRepository.document_contributors):
        [fact] = asyncio.run(read(stub, ctx, "Search Platform"))
        assert fact.source == GONE
        assert fact.source_status == "Inactive"
        assert fact.target_status is None, "the flag belongs on the person's end"


def test_the_profile_pass_stamps_the_status_as_a_field():
    from ai_graph_retrieval.graph.repository import GraphReadRepository

    async def _rows(query, ctx, **params):
        return [{"name": GONE, "person_email": GONE_EMAIL, "job_title": "Staff Engineer",
                 "employment_status": "Inactive", "graph": WIKIPEDIA}]

    async def _identity_groups(ctx, rows):
        return {GONE_KEY: list(rows)}

    stub = SimpleNamespace(_rows=_rows, _s=make_settings(),
                           _identity_groups=_identity_groups,
                           _principals=lambda ctx: None)
    # The row→evidence half is shared with the group ENUMERATION read, so it
    # lives in its own method; bind the real one onto the stub.
    stub._profiles_from_rows = (
        GraphReadRepository._profiles_from_rows.__get__(stub))
    profiles, _docs = asyncio.run(GraphReadRepository.fetch_people_profiles(
        stub, SimpleNamespace(log_fields=lambda: {}), [GONE]))
    assert [p.status for p in profiles] == ["Inactive"]
    assert profiles[0].profile.startswith("FORMER EMPLOYEE")


# ── end to end: query → retrieval → pool → digest ───────────────────────────


class _Repo:
    """A graph whose identity read reports one human departed in arXiv while
    Wikipedia still calls them Active — the cross-graph case Cypher cannot see.

    NAME-KEYED, like the real read (cypher.PERSON_IDENTITIES matches the names it
    is given): a human it was never asked about comes back as nothing. That is
    what makes this a real test of ``candidate_names`` — a stub that reported the
    departure whatever it was handed would pass every shape, including the ones
    whose name no read ever saw.
    """

    sources = ()
    vector_store = None
    # The graph's own rows for the two humans, keyed by the labels a caller can
    # ask for. The second read (…_BY_EMAIL) is what turns ONE matched label into
    # every label the human carries, so asking for either spelling finds both.
    _NODES: ClassVar[dict[str, list[dict]]] = {
        GONE: [_row(GONE, GONE_EMAIL, WIKIPEDIA, status="Active"),
               _row(GONE_ALIAS, GONE_EMAIL, ARXIV, status="Inactive")],
        GONE_ALIAS: [_row(GONE, GONE_EMAIL, WIKIPEDIA, status="Active"),
                     _row(GONE_ALIAS, GONE_EMAIL, ARXIV, status="Inactive")],
        HERE: [_row(HERE, HERE_EMAIL, WIKIPEDIA, status="Active")],
    }

    def __init__(self) -> None:
        self.asked: list[str] = []

    async def person_identities(self, ctx, names):
        self.asked = list(names)
        rows = [r for n in names for r in self._NODES.get(n, ())]
        return PersonIdentities(
            renames={GONE_ALIAS: GONE} if any(
                n in (GONE, GONE_ALIAS) for n in names) else {},
            departed=departed_person_names(rows),
            # The real read reports BOTH of these on a successful return
            # (graph/repository.person_identities); a stub that omitted them
            # would look to the sweep like a read that never answered.
            resolved=True,
            asked=frozenset(str(n).strip().casefold() for n in names),
        )

    async def fetch_people_profiles(self, ctx, names):
        return [], []

    async def expand_entity(self, ctx, name, **kwargs):
        return []

    async def fetch_claims(self, ctx, name, since_iso=None):
        return []

    async def supporting_chunks(self, ctx, names):
        return []


def _lane_sc(pool: EvidencePool, repo: _Repo, *, historical: bool):
    async def graph_delta(**kwargs):
        return None

    return SimpleNamespace(
        evidence=pool, repo=repo, ctx=SimpleNamespace(log_fields=lambda: {}),
        settings=make_settings(scoring_enabled=False),
        historical=historical, degraded=False, question="q", seed_phrases=[],
        profile=SimpleNamespace(
            claim_enrich_top=0, profile_enrich_top=0, relationship_enrich_top=0,
            mention_enrich_top=0,
            graph_text_tools=False, graph_chunk_cap=0, mid_research_clarify=False,
            deep_tool_rows=False, scoring_budget=0,
        ),
        progress=SimpleNamespace(graph_delta=graph_delta, span_parent=None),
        usage=None, llm=None, embedder=None,
    )


def _seeded_pool() -> EvidencePool:
    """What the loop leaves behind: the departed human under BOTH labels — the
    arXiv handle the agent resolved and the Wikipedia name a fact names."""
    pool = _leaky_pool()
    pool.add_entities([EntityHit(name=GONE_ALIAS, type="Person")])
    pool.add_facts([RelationFact(source=GONE_ALIAS, target="Ranking",
                                 relationship="contributed_to")])
    return pool


class _Lane(RetrievalStrategy):
    """A lane that fills the pool and does nothing else — no identity pass, no
    sweep. Standing in for EVERY lane on purpose: the invariant under test is
    that the rule holds for evidence, not that a particular lane remembers to
    apply it."""

    name = "stub"

    def __init__(self, fill=lambda pool: None) -> None:
        self._fill = fill

    async def run(self, sc, plan) -> None:
        self._fill(sc.evidence)


class _CancelledLane(_Lane):
    """A lane the retrieval deadline cancels after it has pooled evidence — the
    ``partial_on_timeout`` path, whose partial pool still reaches synthesis."""

    async def run(self, sc, plan) -> None:
        self._fill(sc.evidence)
        raise TimeoutError


def _collect(pool: EvidencePool, *, historical: bool = False,
             strategy: RetrievalStrategy | None = None,
             repo: object | None = None, **overrides) -> Evidence:
    """Run the REAL retrieval boundary over ``pool``.

    Through ``EvidenceCollector.collect`` rather than any lane's own tail: that
    is where the rule now lives, and a test that called a lane directly could
    not tell a rule that holds for every lane from one a single lane remembers.
    """
    from ai_graph_retrieval.pipeline.execution import EvidenceCollector
    from ai_graph_retrieval.pipeline.planning import ExecutionDecision

    settings = make_settings(**overrides)
    sc = _lane_sc(pool, repo if repo is not None else _Repo(), historical=historical)
    sc.settings = settings
    decision = ExecutionDecision(
        plan=RetrievalPlan(intent=Intent.ENTITY, search_query="q", rationale="t"),
        strategy=strategy or _Lane(),
        profile=SimpleNamespace(budget_s=30.0, employment_gate=True),
    )
    asyncio.run(EvidenceCollector(settings).collect(sc, decision))
    return pool.snapshot()


def test_the_boundary_leaves_no_trace_of_a_departed_person_in_the_answer_context():
    """The acceptance criterion, traced the whole way: the pool a lane left,
    through the identity pass, into the digest the answer model is given."""
    ev = _collect(_seeded_pool())
    digest, _, _ = AnswerComposer(make_settings())._build_digest(ev, strict=False)
    for label in (GONE, GONE_ALIAS):
        assert label not in digest
        assert label not in Evidence.model_dump_json(ev)
    assert HERE in digest


def test_the_boundary_keeps_and_marks_them_on_a_historical_question():
    ev = _collect(_seeded_pool(), historical=True)
    digest, _, _ = AnswerComposer(make_settings())._build_digest(ev, strict=False)
    assert GONE in digest
    # Canonicalised to one label first, then marked — never left as two people.
    assert GONE_ALIAS not in digest
    assert "no longer with the organization" in digest


def test_the_people_object_document_title_is_offered_to_the_identity_read():
    """A Search-seeded run can hold a person ONLY as their people object: no
    entity row, no edge, just a document whose title is their name."""
    from ai_graph_retrieval.retrieval.person_identity import candidate_names

    ev = Evidence(sources=[SourceDocument(doc_id="p-1", title=GONE,
                                          source_type="people")])
    assert candidate_names(ev) == [GONE]


# ── the rule is the BOUNDARY'S, not any lane's ──────────────────────────────
# The general invariant, stated once and checked against the shapes that broke
# it: whatever put a departed person in the pool, and however retrieval ended,
# the name is gone before anything is ranked. None of these name a lane.


# Every representation whose value IS a label of the person — so a path is
# covered because it is an evidence shape, never because someone remembered to
# write a test for the lane that produces it.
#
# WHY THIS LIST AND NOT "any text containing the name". The sweep drops a person
# it was told is departed, and only the graph can tell it that — keyed on the
# name (cypher.PERSON_IDENTITIES matches ``person.name IN $names``). So a person
# is reachable exactly when SOME field in the pool carries a label of theirs
# whole: that label resolves, the second read returns every other spelling they
# carry, and ``drop_people`` then reaches the embedded copies too — a title that
# merely contains the name, a url addressed by their handle, a description or a
# chunk that names them in prose (each pinned separately below).
#
# A person whose ONLY appearance anywhere in the pool is inside free text is out
# of reach by construction, and deliberately not faked into this list: finding
# them means extracting names from prose, which is a different mechanism (NER)
# with its own false-positive cost, not a gap in this one. ``prose_only_leak``
# below states that boundary rather than leaving it to be rediscovered.
LEAKS = {
    "entity": lambda p: p.add_entities([EntityHit(name=GONE, type="Person")]),
    "fact_source": lambda p: p.add_facts([
        RelationFact(source=GONE, target="Ranking", relationship="contributed_to")]),
    "fact_target": lambda p: p.add_facts([
        RelationFact(source=HERE, target=GONE, relationship="REPORTS_TO")]),
    "claim_subject": lambda p: p.add_claims([
        ClaimFact(subject=GONE, predicate="leads", object="Ranking")]),
    "claim_object": lambda p: p.add_claims([
        ClaimFact(subject="Ranking", predicate="is owned by", object=GONE)]),
    "source_title": lambda p: p.add_sources([
        SourceDocument(doc_id="p-1", title=GONE, source_type="people")]),
    "chunk_mention": lambda p: p.add_quotes([
        ChunkQuote(chunk_id="c-8", doc_title="Charter", text="Owned by the team.",
                   via=[GONE])]),
    "chunk_doc_title": lambda p: p.add_quotes([
        ChunkQuote(chunk_id="c-7", doc_title=GONE, text="Staff Engineer.")]),
    # The OTHER graph's spelling. Whichever one the pool carries resolves to the
    # same human, and the sweep then drops BOTH labels.
    "entity_alias": lambda p: p.add_entities([
        EntityHit(name=GONE_ALIAS, type="Person")]),
    # A manager is a STRING PROPERTY on their report's node, so this is the one
    # person-name slot that is structured at the source and prose by the time it
    # is pooled — no entity row, no edge. Read back through the same segment its
    # writer used (graph/repository._org_parts).
    "profile_manager": lambda p: p.add_entities([
        EntityHit(name=HERE, type="Person",
                  profile=f"Engineer · Data Science · reports to {GONE}")]),
}

# The embedded copies: text that CONTAINS a label rather than being one. Each is
# reachable only once the person is resolvable from somewhere — which is what
# `anchor` supplies, exactly as a real run would (their own row, an edge, their
# people object). What is pinned here is that resolving them ONCE clears every
# embedded copy too, so a name cannot re-enter through another representation
# after the person itself has been filtered.
EMBEDDED = {
    "source_title_containing": lambda p: p.add_sources([
        SourceDocument(doc_id="d-9", title=f"Meet {GONE} — welcome post")]),
    # The digest renders " — url: …" verbatim for a source AND for an entity,
    # and a person's page is addressed by the handle they are named with.
    "source_url": lambda p: p.add_sources([
        SourceDocument(doc_id="p-2", title="Employee record 4471",
                       url=f"https://intranet.example/people/{GONE_ALIAS}")]),
    "entity_url": lambda p: p.add_entities([
        EntityHit(name="Employee record 4471", type="people",
                  url=f"https://intranet.example/people/{GONE_ALIAS}")]),
    "chunk_prose": lambda p: p.add_quotes([
        ChunkQuote(chunk_id="c-9", doc_title="Charter",
                   text=f"The charter was written by {GONE} in 2024.")]),
    "alias_prose": lambda p: p.add_quotes([
        ChunkQuote(chunk_id="c-6", doc_title="Charter",
                   text=f"Reviewed by {GONE_ALIAS} last quarter.")]),
    "fact_description": lambda p: p.add_facts([
        RelationFact(source=HERE, target="Ranking", relationship="owns",
                     description=f"Took over the charter {GONE} wrote.")]),
}


def _assert_gone(ev: Evidence, shape: str) -> None:
    blob = Evidence.model_dump_json(ev)
    digest, _, _ = AnswerComposer(make_settings())._build_digest(ev, strict=False)
    for label in (GONE, GONE_ALIAS):
        assert label not in blob, f"{shape} carried the name into the evidence"
        assert label not in digest, f"{shape} carried the name into the digest"


@pytest.mark.parametrize("shape", sorted(LEAKS))
def test_no_evidence_shape_can_carry_a_departed_person_past_the_boundary(shape):
    """THE GENERAL INVARIANT. One departed human, entering through one evidence
    representation at a time, with no lane doing anything about it — and the
    name is not in the pool the digest is built from."""
    _assert_gone(_collect(EvidencePool(), strategy=_Lane(LEAKS[shape])), shape)


@pytest.mark.parametrize("shape", sorted(EMBEDDED))
def test_an_embedded_copy_of_the_name_goes_with_the_person_it_names(shape):
    """The second half of the invariant: once the person is filtered, their name
    must not walk back in through a different representation of the same human.

    The anchor is what a real run always has — the person is reachable from
    their own row — and the assertion is about everything ELSE in the pool."""
    def fill(pool):
        pool.add_entities([EntityHit(name=GONE, type="Person")])   # the anchor
        EMBEDDED[shape](pool)

    _assert_gone(_collect(EvidencePool(), strategy=_Lane(fill)), shape)


@pytest.mark.parametrize("shape", sorted(EMBEDDED))
def test_an_embedded_copy_is_reached_through_any_one_label_of_the_human(shape):
    """Cross-graph: the anchor is the OTHER graph's spelling, and it still clears
    the copies written the first graph's way. One human, one decision, whichever
    label the run happened to hold."""
    def fill(pool):
        pool.add_entities([EntityHit(name=GONE_ALIAS, type="Person")])
        EMBEDDED[shape](pool)

    _assert_gone(_collect(EvidencePool(), strategy=_Lane(fill)), shape)


def test_prose_only_leak_is_the_known_boundary_of_the_mechanism():
    """THE LIMIT, pinned so it is a decision rather than a surprise.

    The sweep drops people the GRAPH says have left, and the graph is asked by
    name. A departed person whose only appearance in the whole pool is inside
    free text — no row, no endpoint, no title, no mention list, no manager
    segment — is never asked about, so nothing can find them departed.

    Closing it means extracting person names from prose, which is a different
    mechanism with its own false-positive cost (the existing single-token rule
    in drop_people is the same trade made at the other end). Not attempted here.
    If this test ever starts failing, the mechanism grew — update it, do not
    delete it."""
    def fill(pool):
        pool.add_quotes([ChunkQuote(chunk_id="c-1", doc_title="Charter",
                                    text=f"The charter was written by {GONE}.")])

    ev = _collect(EvidencePool(), strategy=_Lane(fill))
    assert GONE in Evidence.model_dump_json(ev)


def test_the_boundary_holds_for_a_lane_that_does_no_person_work_of_its_own():
    """A lane that pools people straight from a store — no Cypher gate anywhere
    in its path, no identity pass of its own — is still covered on a gated mode,
    because the rule belongs to the collector and not to the lane."""
    def fill(pool):
        pool.add_entities([
            EntityHit(name=GONE, type="people", graph="other",
                      url=f"https://intranet.example/people/{GONE_ALIAS}"),
            EntityHit(name=HERE, type="people", graph="other"),
        ])
        pool.add_sources([SourceDocument(doc_id="p-1", title=GONE,
                                         source_type="people", graph="other")])

    ev = _collect(EvidencePool(), strategy=_Lane(fill))
    assert GONE not in Evidence.model_dump_json(ev)
    assert [e.name for e in ev.entities] == [HERE]


def test_the_boundary_still_runs_when_the_deadline_cancelled_the_lane():
    """The path that made the rule conditional on finishing. Partial evidence
    reaches synthesis (``partial_on_timeout``), so a sweep at the tail of a lane
    is the statement "no departed people, unless we ran out of time"."""
    pool = EvidencePool()
    partial = _CancelledLane(lambda p: p.add_entities([
        EntityHit(name=GONE, type="Person"), EntityHit(name=HERE, type="Person")]))
    ev = _collect(pool, strategy=partial)
    assert GONE not in Evidence.model_dump_json(ev)
    assert [e.name for e in ev.entities] == [HERE]


def test_an_active_person_is_untouched_by_any_of_it():
    """The other half of the requirement, and the one a blunt fix breaks: the
    colleague who is still here keeps their row, their edges, their claims,
    their document and their excerpt — and gains no status flag."""
    ev = _collect(_leaky_pool())
    assert [e.name for e in ev.entities] == [HERE]
    assert next(e for e in ev.entities if e.name == HERE).status is None
    assert [(f.source, f.target) for f in ev.facts] == [(HERE, "Search Platform")]
    assert [c.subject for c in ev.claims] == [HERE]
    assert [s.title for s in ev.sources] == ["Search Platform charter"]
    assert [q.chunk_id for q in ev.quotes] == ["c-3"]


def test_a_departed_person_consumes_no_ranking_capacity_at_the_boundary():
    """"Must not spend a slot" is a claim about ORDER, so it is checked as one:
    the sweep is upstream of the digest's caps, so the active colleague the cap
    would have cut is rendered instead of the departed one."""
    s = make_settings(digest_max_entities=2)

    def fill(pool):
        pool.add_entities([
            EntityHit(name=GONE, type="Person", profile="Staff Engineer"),
            EntityHit(name="Second Person", type="Person", profile="Engineer"),
            EntityHit(name=HERE, type="Person", profile="Engineer"),
        ])

    unswept = EvidencePool()
    fill(unswept)
    before, _, _ = AnswerComposer(s)._build_digest(unswept.snapshot(), strict=False)
    assert HERE not in before                      # cut by the cap

    ev = _collect(EvidencePool(), strategy=_Lane(fill), digest_max_entities=2)
    after, counts, _ = AnswerComposer(s)._build_digest(ev, strict=False)
    assert HERE in after and GONE not in after
    assert counts["entities_rendered"] == 2


def test_both_graphs_are_read_the_same_way_at_the_boundary():
    """Cross-graph consistency: the human is Active on one graph and departed on
    the other, and it is the IDENTITY join — not either graph's own verdict —
    that decides. Whichever graph a label came from, it goes."""
    def fill(pool):
        pool.add_entities([
            EntityHit(name=GONE, type="Person", graph=WIKIPEDIA),
            EntityHit(name=GONE_ALIAS, type="Person", graph=ARXIV),
            EntityHit(name=HERE, type="Person", graph=WIKIPEDIA),
        ])

    ev = _collect(EvidencePool(), strategy=_Lane(fill))
    assert [e.name for e in ev.entities] == [HERE]


def test_the_boundary_never_fails_the_request_when_the_identity_read_is_down():
    """Subtractive by nature, so a failed read leaves exactly the evidence the
    service would have had anyway — never an error the caller sees."""
    class _Broken(_Repo):
        async def person_identities(self, ctx, names):
            raise RuntimeError("neo4j unavailable")

    ev = _collect(_leaky_pool(), repo=_Broken())
    assert len(ev.entities) == 2                   # unchanged, not an exception


def test_the_kill_switch_still_turns_the_whole_boundary_off():
    ev = _collect(_seeded_pool(), inactive_people_filter_enabled=False)
    assert GONE in Evidence.model_dump_json(ev)
