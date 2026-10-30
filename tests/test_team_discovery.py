"""The six P0 retrieval defects behind unanswerable team questions
(retrieval defect report, 2026-08-28).

The through-line is the one the whole people register has: the service is top-K
relevance retrieval, and the org backbone is not a ranking. A team is a NODE the
resolver could not see, its membership is a SET the ranking cut, its edges carry
no `support` so the curve buried them, and the budgets that describe what was
found were spent before the question's own people arrived.

Nothing here names a real team, a real document or a real person: every test
passes arbitrary strings, and a fix that only works for the names in the defect
report fails these.

    uv run pytest tests/test_team_discovery.py -v
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from ai_graph_retrieval.agent.tools import (
    ResearchContext,
    find_groups,
    get_document_text,
    list_group_members,
    research_tools,
)
from ai_graph_retrieval.core.access import AccessContext
from ai_graph_retrieval.core.models import (
    ChunkQuote,
    DocumentText,
    EntityHit,
    Evidence,
    EvidencePool,
    RelationFact,
)
from ai_graph_retrieval.graph import cypher
from ai_graph_retrieval.graph.repository import GraphReadRepository
from ai_graph_retrieval.retrieval.base import Intent, Mode, RetrievalPlan
from ai_graph_retrieval.retrieval.deep_research import (
    DeepResearchStrategy,
    _ordered_names,
)
from ai_graph_retrieval.retrieval.neighbor_scoring import NeighborScorer, ScoringBudget
from ai_graph_retrieval.retrieval.person_status import PersonIdentities
from ai_graph_retrieval.retrieval.profile import profile_for
from tests.support import make_people_settings as make_settings

WIKIPEDIA, ARXIV = "wikipedia", "arxiv"


def _ctx():
    return AccessContext(tenant_id="t", principals=("public",), request_id="r")


def _repo(by_query, settings=None):
    """A repository whose fan-out answers per QUERY, capturing the bound params.

    Per query, not one row list: the reads under test here fan out over several
    named queries in one call, and a stub that answered them all identically
    could not tell the arms apart — which is the thing being asserted.
    """
    repo = GraphReadRepository(None, settings or make_settings(), http=object())
    seen: list[tuple[str, dict]] = []

    async def _stub_rows(query, _ctx, **params):
        name = next((k for k, v in vars(cypher).items()
                     if k.isupper() and v is query), "?")
        seen.append((name, params))
        rows = by_query.get(name, [])
        return rows(**params) if callable(rows) else rows

    repo._rows = _stub_rows
    return repo, seen


class _Step:
    def __init__(self) -> None:
        self.detail: str | None = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


def _rc(repo, pool=None, **settings):
    async def _graph_delta(**_kw):
        return None

    return ResearchContext(
        repo=repo, ctx=SimpleNamespace(log_fields=dict), embedder=None,
        settings=make_settings(**settings), evidence=pool or EvidencePool(),
        progress=SimpleNamespace(step=lambda *a, **k: _Step(),
                                 graph_delta=_graph_delta),
        usage=None, graph_chunk_cap=100,
    )


def _invoke(tool, rc, payload: dict):
    return json.loads(asyncio.run(tool.on_invoke_tool(
        SimpleNamespace(context=rc, tool_name=tool.name, tool_call_id="t1",
                        run_config=None),
        json.dumps(payload),
    )))


# ═══ P0-1 · a structured team must be discoverable ═══════════════════════════
# Entity resolution matches ':Entity'. A :Team node is not one, so a team that
# exists in the graph looked to every read like a team that does not exist.


def test_group_discovery_anchors_on_the_membership_edge_not_on_a_label_list():
    """A group is whatever a Person is a MEMBER_OF — so :Team, :Department and
    whatever ingestion adds next are all covered without naming any of them."""
    q = cypher.GROUPS_RESOLVE
    assert "MATCH (person:Person)-[:MEMBER_OF]->(g)" in q
    assert ":Team" not in q and ":Department" not in q, (
        "a hard-coded label list stops covering the schema the day it grows"
    )
    # Plumbing is excluded the same way the expansion excludes it.
    assert "NOT (g:Chunk OR g:Claim OR g:Community)" in q


def test_group_discovery_is_acl_and_employment_gated_through_its_members():
    """The group node and the MEMBER_OF edge carry no properties at all, so a
    predicate on either evaluates null and drops every row under enforcement.
    The ACL-bearing node is the PERSON — and the person is what is disclosed."""
    prepared = cypher.prepare(cypher.GROUPS_RESOLVE, "allow_access_control")
    assert ("($principals IS NULL OR any(p IN person.allow_access_control "
            "WHERE p IN $principals))") in prepared
    assert "$include_inactive_people" in prepared
    assert "g.allow_access_control" not in prepared, (
        "a group node carries no ACL; gating on it would return nothing"
    )


def test_any_team_resolves_by_its_own_name_whatever_that_name_is():
    """Arbitrary, not the report's: two unrelated names, same behaviour."""
    for name, kind in (("Zeta", "Team"), ("Procurement Ops", "Department")):
        repo, _ = _repo({"GROUPS_RESOLVE": [
            {"name": name, "type": kind, "members": 7, "graph": WIKIPEDIA}]})

        [hit] = asyncio.run(repo.resolve_groups(_ctx(), name))

        assert (hit.name, hit.type, hit.degree) == (name, kind, 7)
        assert hit.via == ["groups"]


def test_discovery_tolerates_the_words_the_question_used():
    """The graphs spell one group "Data-Science" and a caller types "the data
    science team". Discovery is the read that has to bridge that, in both
    directions — an exact-match-only discovery could never find the exact name
    the enumeration needs."""
    q = cypher.GROUPS_RESOLVE
    folded_name = cypher._folded("coalesce(g.name, g.title, '')")
    folded_arg = cypher._folded("$group")
    assert f"{folded_name} CONTAINS {folded_arg}" in q
    assert f"{folded_arg} CONTAINS {folded_name}" in q
    # The fold itself: separators are not identity.
    assert "'-', ' '" in folded_arg and "toLower" in folded_arg


def test_an_empty_phrase_never_reaches_the_graph():
    repo, seen = _repo({})
    assert asyncio.run(repo.resolve_groups(_ctx(), "   ")) == []
    assert seen == []


def test_a_group_in_both_graphs_is_one_row_and_is_never_reported_twice_its_size():
    """The same human is a node in each graph, so summing member counts would
    report a team as twice its size. The largest single graph's count stands."""
    repo, _ = _repo({"GROUPS_RESOLVE": [
        {"name": "Omega", "type": "Team", "members": 9, "graph": WIKIPEDIA},
        {"name": "Omega", "type": "Team", "members": 4, "graph": ARXIV},
    ]})

    hits = asyncio.run(repo.resolve_groups(_ctx(), "omega"))

    assert [(h.name, h.degree) for h in hits] == [("Omega", 9)]


def test_discovery_orders_by_recorded_size_then_name_deterministically():
    repo, _ = _repo({"GROUPS_RESOLVE": [
        {"name": "Beta", "type": "Team", "members": 2, "graph": WIKIPEDIA},
        {"name": "Alpha", "type": "Team", "members": 2, "graph": WIKIPEDIA},
        {"name": "Gamma", "type": "Department", "members": 30, "graph": WIKIPEDIA},
    ]})

    hits = asyncio.run(repo.resolve_groups(_ctx(), "a"))

    assert [h.name for h in hits] == ["Gamma", "Alpha", "Beta"]


def test_the_agent_can_reach_group_discovery_and_the_pool_keeps_what_it_found():
    class _Repo:
        async def resolve_groups(self, ctx, text, *, limit=None):
            inner, _ = _repo({"GROUPS_RESOLVE": [
                {"name": "Tango", "type": "Team", "members": 5, "graph": WIKIPEDIA}]})
            return await inner.resolve_groups(_ctx(), text, limit=limit)

    assert "find_groups" in [t.name for t in research_tools(people=True)]
    pool = EvidencePool()
    out = _invoke(find_groups, _rc(_Repo(), pool), {"query": "tango squad"})

    assert out["groups"] == [
        {"name": "Tango", "kind": "Team", "members": 5, "graph": WIKIPEDIA}]
    assert [e.name for e in pool.snapshot().entities] == ["Tango"]


# ═══ P0-2 · membership is a SET, and the whole set must come back ════════════


def test_enumeration_covers_both_representations_of_membership_in_one_call():
    """Membership is recorded two ways and the caller cannot know which holds the
    group they named: person ATTRIBUTES (the HR taxonomy) and MEMBER_OF edges
    into group NODES (the scrum teams). Covering one arm answers half the
    questions and looks like "no such team" for the other half."""
    repo, seen = _repo({
        "PEOPLE_BY_GROUP": [{"name": "Attr Only", "job_title": "PM",
                             "graph": WIKIPEDIA}],
        "GROUP_MEMBERS": [{"name": "Edge Only", "job_title": "SDE",
                           "graph": ARXIV}],
    })

    people, _docs, _t = asyncio.run(repo.people_by_group(_ctx(), "Whatever Team"))

    assert {q for q, _ in seen} == {"PEOPLE_BY_GROUP", "GROUP_MEMBERS"}
    assert {p.name for p in people} == {"Attr Only", "Edge Only"}
    assert {params["group"] for _, params in seen} == {"Whatever Team"}


def test_a_member_reachable_only_through_the_membership_edge_is_returned():
    """The defect exactly: the attribute arm knows nothing about a scrum team, so
    before the second arm existed this returned nobody at all."""
    repo, _ = _repo({"GROUP_MEMBERS": [
        {"name": f"M{i}", "job_title": "Engineer", "graph": WIKIPEDIA}
        for i in range(12)]})

    people, _docs, truncated = asyncio.run(repo.people_by_group(_ctx(), "Delta"))

    assert len(people) == 12 and truncated is False


def test_enumeration_of_the_membership_edge_is_exact_and_unranked():
    """A set is not a ranking: no score anywhere, and equality rather than
    CONTAINS — a substring match on "Data" selects Data Science AND Data
    Platform, which is a different set than the one asked for."""
    q = cypher.GROUP_MEMBERS
    assert "score" not in q
    assert "CONTAINS" not in q
    name = cypher._folded("coalesce(g.name, g.title, '')")
    assert f"{name} = {cypher._folded('$group')}" in q
    assert "ORDER BY name" in q


def test_one_human_recorded_both_ways_is_one_member_not_two():
    """The union is the only place that duplicate can arise, and an inflated
    roster count is exactly what an enumeration read exists not to produce."""
    row = {"name": "Both", "job_title": "Staff", "graph": WIKIPEDIA}
    repo, _ = _repo({"PEOPLE_BY_GROUP": [row], "GROUP_MEMBERS": [dict(row)]})

    people, _docs, _t = asyncio.run(repo.people_by_group(_ctx(), "Zeta"))

    assert [p.name for p in people] == ["Both"]


def test_a_member_with_no_profile_attributes_is_still_a_member():
    """A arXiv identity stub carries a name and nothing else — the shape
    person_status.py documents as normal. Rendering rules dropped it from the
    enumeration while `truncated` stayed False, so the tool reported a set it
    had silently shrunk as complete. The member comes back as a name-only row."""
    repo, _ = _repo({"GROUP_MEMBERS": [
        {"name": "Stub Person", "graph": ARXIV},
        {"name": "Rich Person", "job_title": "SDE", "department": "Data Science",
         "graph": WIKIPEDIA},
    ]})

    people, _docs, truncated = asyncio.run(repo.people_by_group(_ctx(), "Data Science"))

    assert {p.name for p in people} == {"Stub Person", "Rich Person"}
    assert truncated is False
    assert next(p for p in people if p.name == "Stub Person").profile is None


def test_one_human_spelled_two_ways_across_graphs_is_one_member():
    """The display name is exactly what differs across graphs for one human, so
    a name-keyed dedup counted them twice. The identity key (email) is what says
    they are the same person; the attribute-bearing row speaks for them."""
    repo, _ = _repo({"GROUP_MEMBERS": [
        {"name": "max.fields", "person_email": "a@x.com", "graph": ARXIV},
        {"name": "Max Fields", "person_email": "a@x.com", "job_title": "PM",
         "graph": WIKIPEDIA},
    ]})

    people, _docs, _t = asyncio.run(repo.people_by_group(_ctx(), "Zeta"))

    assert [(p.name, p.profile) for p in people] == [("Max Fields", "title: PM")]


def test_two_humans_sharing_a_name_are_two_members_not_one():
    """A name-keyed dedup collapsed them into one row — an undercount, with one
    human's job attributed to the other. Both stay; the ambiguity rule leaves
    the shared name bare, because attributing either profile would state
    someone else's job."""
    repo, _ = _repo({"GROUP_MEMBERS": [
        {"name": "Sam Carter", "person_email": "sam.1@x.com", "job_title": "SDE",
         "graph": WIKIPEDIA},
        {"name": "Sam Carter", "person_email": "sam.2@x.com", "job_title": "PM",
         "graph": WIKIPEDIA},
        {"name": "Other", "job_title": "Eng", "graph": WIKIPEDIA},
    ]})

    people, _docs, _t = asyncio.run(repo.people_by_group(_ctx(), "X"))

    assert sorted((p.name, p.profile) for p in people) == [
        ("Other", "title: Eng"), ("Sam Carter", None), ("Sam Carter", None),
    ]


def test_a_ceiling_that_bound_on_either_arm_is_reported_not_applied_silently():
    """Limits stay — hub explosion is real — but a truncated set that reads as a
    complete one is the failure this whole register is about."""
    settings = make_settings(people_enumeration_limit=3)
    rows = [{"name": f"M{i}", "job_title": "Engineer", "graph": WIKIPEDIA}
            for i in range(3)]
    for arm in ("PEOPLE_BY_GROUP", "GROUP_MEMBERS"):
        repo, seen = _repo({arm: rows}, settings)

        people, _docs, truncated = asyncio.run(repo.people_by_group(_ctx(), "X"))

        assert len(people) == 3
        assert truncated is True, f"{arm} hit its ceiling and must say so"
        assert all(params["limit"] == 3 for _, params in seen)


def test_the_tool_hands_back_every_member_and_states_whether_the_set_is_whole():
    """No row window on this tool, deliberately: every other people tool shows
    the model `_rows(rc)` items, and doing that here would re-create the top-K
    cut the enumeration exists to remove."""
    class _Repo:
        async def people_by_group(self, ctx, group, *, limit=None):
            inner, _ = _repo({"GROUP_MEMBERS": [
                {"name": f"M{i}", "job_title": "Engineer", "graph": WIKIPEDIA}
                for i in range(40)]})
            return await inner.people_by_group(_ctx(), group, limit=limit)

    rc = _rc(_Repo(), agent_tool_rows=5, agent_tool_rows_research=5)
    out = _invoke(list_group_members, rc, {"group": "Apollo"})

    assert out["returned"] == 40 and len(out["members"]) == 40
    assert out["truncated"] is False


# ═══ P0-3 · structure must not lose to support-heavy extracted edges ═════════
# A schema-asserted edge carries no `support`, so the curve scores every one of
# them at its floor — and they were the first thing every cut removed.


def test_the_expansion_puts_structure_first_and_says_which_rows_are_structural():
    q = cypher.EXPAND_NEIGHBORS
    assert "r.support IS NULL AS structural" in q, (
        "structural is the ABSENCE of support, not a type list — a new "
        "structured verb must be structural on arrival"
    )
    assert "ORDER BY structural DESC, score DESC" in q


def _edge(target, *, structural, score, relationship="R"):
    return RelationFact(source="Group", target=target, relationship=relationship,
                        structural=structural, score=score, support=1)


def test_membership_and_reporting_survive_the_cheap_cut_they_used_to_lose():
    """MEMBER_OF / REPORTS_TO / MANAGES sit at the curve's floor by construction,
    so ranking them on that curve put the org backbone last and truncated it
    first — on a question that is entirely about the org backbone."""
    facts = [_edge(f"Topic{i}", structural=False, score=0.9) for i in range(5)]
    facts += [_edge("Ana", structural=True, score=0.0452, relationship="MEMBER_OF"),
              _edge("Bo", structural=True, score=0.0452, relationship="REPORTS_TO"),
              _edge("Cy", structural=True, score=0.0452, relationship="MANAGES")]

    kept = NeighborScorer._cheap_ranked(facts)[:3]

    assert {f.target for f in kept} == {"Ana", "Bo", "Cy"}


def test_structure_survives_the_llm_tier_too_not_only_the_cheap_one():
    """The re-rank is the other place the floor score could delete them: the
    blend is 90% the model's judgment, and a structural edge is exactly the kind
    of row a relevance model scores low."""
    settings = make_settings(scoring_small_threshold=0,
                             scoring_separation_margin=1.0,
                             scoring_batch_token_budget=10_000)
    facts = [_edge("Topic", structural=False, score=0.9),
             _edge("Ana", structural=True, score=0.0452, relationship="MEMBER_OF")]
    scorer = NeighborScorer(None, settings, None, ScoringBudget(4))

    async def _score_batch(batch, focus):
        return 0, {0: 1.0, 1: 0.0}    # the model prefers the topic outright

    scorer._score_batch = _score_batch
    kept, tier = asyncio.run(scorer.rescore(facts, "who is in this group", keep=1))

    assert tier == "llm"
    assert [f.target for f in kept] == ["Ana"]


def test_the_ranking_is_generic_and_never_reads_the_relationship_name():
    """No query-specific logic: the rule is "carries no support", so an unnamed
    structured verb ranks with MEMBER_OF and an extracted edge never does."""
    ranked = NeighborScorer._cheap_ranked([
        _edge("Extracted", structural=False, score=0.99, relationship="MEMBER_OF"),
        _edge("Structured", structural=True, score=0.0, relationship="WHATEVER"),
    ])
    assert [f.target for f in ranked] == ["Structured", "Extracted"]


# ═══ P0-4 · the enrichment budgets belong to the question's people ═══════════


def _person_fact(target, **kw):
    return RelationFact(source="Group", target=target, relationship="MEMBER_OF", **kw)


def test_concepts_and_documents_lose_the_head_of_the_candidate_list_to_people():
    """The profile and claim caps are spent off the head of this list in
    insertion order, so one broad early read took every slot before the
    question's own people were in the pool (measured: 60 profile slots spent, 43
    filled, none in the group asked about)."""
    ev = Evidence(facts=[
        _person_fact("Sentiment Analysis", target_type="CONCEPT"),
        _person_fact("FAQ TSV file", target_type="Document"),
        _person_fact("Ana", target_type="Person"),
    ])

    ordered = _ordered_names(ev, skip=set(), people_first=True)

    assert ordered[0] == "Ana"
    assert "FAQ TSV file" not in ordered, "a document title describes no person"
    assert "Sentiment Analysis" in ordered, "ordering only — a concept keeps its place"


def test_duplicate_identities_are_collapsed_before_the_budgets_are_spent(monkeypatch):
    """The same human enters the pool under each graph's own label, and every
    label was a separate candidate — so a duplicate took a profile slot, and an
    entire per-entity claim read, from a person nobody had described yet.

    The identity pass is the existing mechanism for that; this pins that it runs
    BEFORE the top-ups rather than only after them.
    """
    import ai_graph_retrieval.agent.runtime as runtime_mod

    order: list[str] = []

    class _Repo:
        sources = ()
        vector_store = None

        def __init__(self) -> None:
            self.profiled: list[str] = []

        async def person_identities(self, ctx, names):
            order.append("identity")
            return PersonIdentities(renames={"mia.fields": "Mia Fields"})

        async def fetch_people_profiles(self, ctx, names):
            order.append("profiles")
            self.profiled = list(names)
            return [], []

        async def fetch_claims(self, ctx, name, since_iso=None):
            order.append("claims")
            return []

        async def expand_entity(self, ctx, name, **_k):
            return []

    class _FakeAgent:
        def __init__(self, *a, **k):
            pass

        async def run(self, sc, scorer):
            return None

    monkeypatch.setattr(runtime_mod, "ResearchAgent", _FakeAgent)
    pool = EvidencePool()
    pool.add_entities([EntityHit(name="mia.fields", type="Person"),
                       EntityHit(name="Mia Fields", type="Person")])
    repo = _Repo()
    settings = make_settings()
    sc = SimpleNamespace(
        ctx=SimpleNamespace(log_fields=lambda: {}), question="who is in X?",
        history=[], settings=settings, repo=repo, llm=None, embedder=None,
        evidence=pool,
        progress=SimpleNamespace(step=lambda *a, **k: _Step(),
                                 graph_delta=_graph_delta_noop, span_parent=None),
        usage=None,
        profile=profile_for(settings, Mode.SUPER_RESEARCH, Intent.RESEARCH),
        chunk_retriever=None, seed_phrases=[], degraded=False, historical=False,
    )

    asyncio.run(DeepResearchStrategy().run(sc, RetrievalPlan(intent=Intent.RESEARCH)))

    assert order.index("identity") < order.index("profiles"), (
        "the duplicate must be gone before the profile budget is spent"
    )
    assert order.index("identity") < order.index("claims")
    # …and the collapse actually reached the candidate list.
    assert repo.profiled == ["Mia Fields"]
    # The pass still runs after the top-ups: they are name-producing reads too.
    assert order.count("identity") == 2


async def _graph_delta_noop(**_kw):
    return None


# ═══ P0-5 · a document read must be a document read ═════════════════════════


def test_the_per_chunk_cap_and_the_document_row_limit_are_separate_budgets():
    """The cap used to be spent twice — bound into `left(c.text, $chars)` AND
    re-used as a running total — so the first chunk exhausted the document and a
    roster came back as one arbitrary chunk described as read start to finish."""
    rows = [{"chunk_id": f"c{i}", "doc_id": "d1", "doc_title": "Roster",
             "text": "x" * 6000, "graph": WIKIPEDIA} for i in range(4)]
    repo, seen = _repo({"DOCUMENT_TEXT": rows},
                       make_settings(document_chunk_limit=50))

    result = asyncio.run(repo.fetch_document_text(_ctx(), "d1", max_chars=6000))

    [(_q, params)] = seen
    assert params["chars"] == 6000 and params["limit"] == 50
    assert [len(c.text) for c in result.chunks] == [6000] * 4
    assert result.truncated is False


def test_a_document_cut_by_the_row_limit_is_not_reported_as_read_in_full():
    settings = make_settings(document_chunk_limit=2)
    rows = [{"chunk_id": f"c{i}", "doc_id": "d1", "text": "body", "graph": WIKIPEDIA}
            for i in range(2)]
    repo, _ = _repo({"DOCUMENT_TEXT": rows}, settings)

    result = asyncio.run(repo.fetch_document_text(_ctx(), "d1", max_chars=6000))

    assert result.truncated is True


def test_the_tool_reports_completeness_per_document_after_the_chunk_budget():
    """A chunk the graph-chunk budget refused is as absent from the answer as one
    the row limit never returned, so completeness is judged after the budget —
    otherwise a partial read is indistinguishable from a whole one."""
    class _Repo:
        async def fetch_document_text(self, ctx, doc_id, *, max_chars):
            return DocumentText(
                chunks=[ChunkQuote(chunk_id=f"{doc_id}-{i}", doc_id=doc_id,
                                   doc_title=doc_id, text=f"{doc_id} part {i}",
                                   graph=WIKIPEDIA) for i in range(3)],
                truncated=False,
            )

    # Budget for 4 chunks; two documents of 3 chunks each want 6.
    rc = _rc(_Repo())
    rc.graph_chunk_cap = 4
    out = _invoke(get_document_text, rc, {"doc_ids": ["dA", "dB"]})

    by_id = {d["doc_id"]: d for d in out["documents"]}
    assert by_id["dA"]["complete"] is True and by_id["dA"]["chunks_read"] == 3
    assert by_id["dB"]["complete"] is False, (
        "the budget dropped two of its chunks; it was not read start to finish"
    )


def test_a_whole_multi_chunk_read_is_still_reported_complete():
    class _Repo:
        async def fetch_document_text(self, ctx, doc_id, *, max_chars):
            return DocumentText(chunks=[
                ChunkQuote(chunk_id=f"c{i}", doc_id=doc_id, doc_title="Roster",
                           text=f"row {i}", graph=WIKIPEDIA) for i in range(5)])

    out = _invoke(get_document_text, _rc(_Repo()), {"doc_ids": ["d1"]})

    [doc] = out["documents"]
    assert doc["complete"] is True and doc["chunks_read"] == 5
    assert doc["text"].splitlines()[0] == "row 0" and "row 4" in doc["text"]


# ═══ P0-6 · the relationship vocabulary must be a cross-graph one ═══════════


def test_the_vocabulary_merges_every_graph_before_the_agent_visible_limit():
    """`$limit` binds PER GRAPH and the fan-out concatenates, so slicing the
    merged list positionally returned the head of the FIRST graph and nothing
    else — measured, 400 rows in and the 60 the agent saw were 100% graph one,
    while the membership verbs an org question needs sat in graph two."""
    rows = ([{"type": f"BK_{i}", "edges": 500 - i, "graph": WIKIPEDIA}
             for i in range(60)]
            + [{"type": "HAS_MEMBER", "edges": 900, "graph": ARXIV}])
    repo, seen = _repo({"RELATIONSHIP_TYPES": rows})

    vocab = asyncio.run(repo.relationship_types(_ctx(), limit=60))

    assert len(vocab) == 60
    assert vocab[0]["type"] == "HAS_MEMBER", "the second graph must be reachable"
    assert seen[0][1]["limit"] == make_settings().rel_vocab_fetch, (
        "over-fetch per graph, then merge — the merge can only pick from what "
        "both graphs actually returned"
    )


def test_a_type_present_in_both_graphs_spends_one_slot_and_keeps_both_counts():
    repo, _ = _repo({"RELATIONSHIP_TYPES": [
        {"type": "MEMBER_OF", "edges": 10, "graph": WIKIPEDIA},
        {"type": "MEMBER_OF", "edges": 7, "graph": ARXIV},
        {"type": "REPORTS_TO", "edges": 12, "graph": WIKIPEDIA},
    ]})

    vocab = asyncio.run(repo.relationship_types(_ctx(), limit=2))

    assert vocab == [{"type": "MEMBER_OF", "edges": 17},
                     {"type": "REPORTS_TO", "edges": 12}]


def test_the_vocabulary_names_the_structured_verbs_expansion_can_filter_on():
    """The list is what the agent is told to consult before choosing
    `relationship_types`, so it must cover every edge EXPAND_NEIGHBORS can
    filter on — not the extracted verbs alone."""
    q = cypher.RELATIONSHIP_TYPES
    assert "coalesce(r.relationship_type, type(r)) AS type" in q
    assert "MATCH (a:Entity)-[r]->(nbr)" in q
    assert "EXTRACTED_REL" not in q


def test_the_merge_is_deterministic_for_identical_input():
    rows = [{"type": "B", "edges": 5, "graph": WIKIPEDIA},
            {"type": "A", "edges": 5, "graph": ARXIV}]
    repo, _ = _repo({"RELATIONSHIP_TYPES": rows})
    first = asyncio.run(repo.relationship_types(_ctx()))
    repo2, _ = _repo({"RELATIONSHIP_TYPES": list(reversed(rows))})
    assert first == asyncio.run(repo2.relationship_types(_ctx()))
    assert [v["type"] for v in first] == ["A", "B"]


def test_no_agent_visible_limit_can_hide_a_graph_entirely():
    """Even at a limit of one, the winning type is the cross-graph winner."""
    repo, _ = _repo({"RELATIONSHIP_TYPES": [
        {"type": "OWNS", "edges": 100, "graph": WIKIPEDIA},
        {"type": "MEMBER_OF", "edges": 101, "graph": ARXIV},
    ]})
    assert [v["type"] for v in
            asyncio.run(repo.relationship_types(_ctx(), limit=1))] == ["MEMBER_OF"]


# ═══ the reads stay inside the service's contracts ══════════════════════════


def test_the_new_reads_bind_only_parameters_the_repository_binds():
    """An unbound parameter is not a degraded read, it is a failed one."""
    import re

    from tests.test_cypher_parameters import ALLOWED

    for name in ("GROUPS_RESOLVE", "GROUP_MEMBERS"):
        prepared = cypher.prepare(getattr(cypher, name), "allow_access_control")
        assert not set(re.findall(r"\$(\w+)", prepared)) - ALLOWED


def test_the_enumeration_read_renders_the_same_profile_row_as_every_people_read():
    """One contract: a member found through a MEMBER_OF edge must carry the
    columns a roster needs, and go through the one identity merge."""
    for field in ("job_title", "department", "employment_status", "d.url AS doc_url"):
        assert field in cypher.GROUP_MEMBERS
