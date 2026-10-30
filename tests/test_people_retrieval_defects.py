"""The retrieval-layer defects behind incomplete and wrongly-populated people
answers (people-retrieval defect register, 2026-08-28).

One theme across all of them: the pipeline is top-K relevance retrieval, and a
question that asks for a SET was answered by whatever the ranking surfaced, cut
by a budget nobody reported. Each test below pins one selection defect — what
gets fetched, and what gets thrown away before any ranking runs.

    uv run pytest tests/test_people_retrieval_defects.py -v
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from ai_graph_retrieval.agent.tools import budget_graph_quotes
from ai_graph_retrieval.core.access import AccessContext
from ai_graph_retrieval.core.models import (
    ChunkQuote,
    EntityHit,
    Evidence,
    EvidencePool,
    RelationFact,
    SourceDocument,
)
from ai_graph_retrieval.graph import cypher
from ai_graph_retrieval.graph.repository import GraphReadRepository
from ai_graph_retrieval.retrieval.deep_research import _ordered_names
from ai_graph_retrieval.retrieval.neighbor_scoring import NeighborScorer, ScoringBudget
from ai_graph_retrieval.synthesis.composer import AnswerComposer
from tests.support import make_settings

WIKIPEDIA, ARXIV = "wikipedia", "arxiv"


def _ctx():
    return AccessContext(tenant_id="t", principals=("public",), request_id="r")


def _repo(rows, settings=None):
    """A repository whose fan-out returns `rows`, capturing the bound params."""
    repo = GraphReadRepository(None, settings or make_settings(), http=object())
    seen: dict = {}

    async def _stub_rows(query, _ctx, **params):
        seen["query"] = query
        seen.update(params)
        return rows(**params) if callable(rows) else rows

    repo._rows = _stub_rows
    return repo, seen


# ── R1 · enumeration needs a read that selects on structure ─────────────────


def test_the_enumeration_read_selects_on_the_attributes_every_profile_projects():
    """`person_department` / `person_team` / `person_manager` are returned by
    every profile read and were selected on by NOTHING, so "all the members of X"
    had no read that could answer it. One query now does — and it is the SAME
    profile row, so an enumerated member renders the columns a roster needs."""
    for attribute in ("person_department", "person_team", "person_manager"):
        assert attribute in cypher.PEOPLE_BY_GROUP
    assert "$group" in cypher.PEOPLE_BY_GROUP
    # The profile contract, shared with the name-keyed read.
    assert "d.url AS doc_url" in cypher.PEOPLE_BY_GROUP


def test_enumeration_is_exact_and_unranked_so_no_member_can_lose_to_another():
    """A set is not a ranking. No score, no ORDER BY score — and equality, never
    CONTAINS: a substring match on "Data" selects Data Science AND Data Platform,
    which is a different set than the one asked for."""
    assert "score" not in cypher.PEOPLE_BY_GROUP
    assert "CONTAINS" not in cypher.PEOPLE_BY_GROUP
    # Equality against the shared folded form (see test_team_discovery.py for why
    # the fold exists) — a widened set of SPELLINGS for one name, never a widened
    # set of names.
    assert f'{cypher._folded("$group")} IN [' in cypher.PEOPLE_BY_GROUP


def test_the_enumeration_read_reports_that_its_ceiling_bound():
    """A truncated set that reads as a complete one is the failure this whole
    register is about, so the ceiling is reported rather than applied silently."""
    settings = make_settings(people_enumeration_limit=2)
    rows = [{"name": f"P{i}", "job_title": "Engineer", "graph": WIKIPEDIA}
            for i in range(2)]
    repo, seen = _repo(rows, settings)

    people, _docs, truncated = asyncio.run(
        repo.people_by_group(_ctx(), "Data Science"))

    assert seen["group"] == "Data Science" and seen["limit"] == 2
    assert [p.name for p in people] == ["P0", "P1"]
    assert truncated is True, "the cap bound and the caller must be able to say so"


def test_an_enumeration_within_its_ceiling_is_reported_complete():
    repo, _ = _repo([{"name": "Solo", "job_title": "Engineer", "graph": WIKIPEDIA}])
    people, _docs, truncated = asyncio.run(repo.people_by_group(_ctx(), "Data Science"))
    assert [p.name for p in people] == ["Solo"] and truncated is False


def test_the_enumeration_read_is_employment_gated_like_every_people_read():
    """A departed colleague must not spend a row of an enumeration either."""
    assert "<<current_person:person>>" in cypher.PEOPLE_BY_GROUP
    prepared = cypher.prepare(cypher.PEOPLE_BY_GROUP, "allow_access_control")
    assert "$include_inactive_people" in prepared
    assert "$principals IS NULL OR" in prepared


# ── R2 · the relationship vocabulary must merge across graphs ───────────────


def test_the_vocabulary_sums_counts_across_graphs_instead_of_head_slicing():
    """`$limit` binds PER GRAPH and the fan-out concatenates, so slicing the
    merged list positionally returned the head of graph one and nothing else.
    Measured live: 400 rows in, and the 60 the agent saw were 100% graph one."""
    rows = (
        [{"type": f"BK_{i}", "edges": 500 - i, "graph": WIKIPEDIA} for i in range(3)]
        + [{"type": "HAS_MEMBER", "edges": 900, "graph": ARXIV}]
    )
    repo, _ = _repo(rows)

    vocab = asyncio.run(repo.relationship_types(_ctx(), limit=2))

    assert [v["type"] for v in vocab] == ["HAS_MEMBER", "BK_0"], (
        "the other graph's busiest type must be reachable within the budget"
    )


def test_a_type_present_in_both_graphs_spends_one_slot_not_two():
    rows = [{"type": "AUTHORED", "edges": 10, "graph": WIKIPEDIA},
            {"type": "AUTHORED", "edges": 7, "graph": ARXIV},
            {"type": "REPORTS_TO", "edges": 12, "graph": WIKIPEDIA}]
    repo, _ = _repo(rows)

    vocab = asyncio.run(repo.relationship_types(_ctx()))

    assert vocab == [{"type": "AUTHORED", "edges": 17},
                     {"type": "REPORTS_TO", "edges": 12}]


def test_the_vocabulary_over_fetches_per_graph_before_merging():
    repo, seen = _repo([])
    asyncio.run(repo.relationship_types(_ctx(), limit=60))
    assert seen["limit"] == make_settings().rel_vocab_fetch


# ── R3 · structure must outrank support frequency ───────────────────────────


def test_the_expansion_ranks_structural_edges_ahead_of_the_support_curve():
    """A schema-asserted edge carries no `support`, so the curve scores every one
    of them at its floor and they all TIE — the ordering then fell to the
    neighbour-degree tiebreak, which prefers a high-degree CONCEPT to a Person.
    Measured on one team entity: 112 edges, all at 0.0452, 4 of them membership."""
    assert "r.support IS NULL AS structural" in cypher.EXPAND_NEIGHBORS
    assert "ORDER BY structural DESC, score DESC" in cypher.EXPAND_NEIGHBORS


def test_the_structural_flag_reaches_the_model_from_the_row():
    rows = [{"source": "DS Team", "target": "Ana", "relationship": "HAS_MEMBER",
             "support": 1, "score": 0.0452, "structural": True, "graph": WIKIPEDIA},
            {"source": "DS Team", "target": "Ranking", "relationship": "ANALYZED",
             "support": 9, "score": 0.15, "structural": False, "graph": WIKIPEDIA}]
    repo, _ = _repo(rows)

    facts = asyncio.run(repo.expand_entity(_ctx(), "DS Team"))

    assert [(f.target, f.structural) for f in facts] == [("Ana", True),
                                                         ("Ranking", False)]


def _edge(target, *, structural, score):
    return RelationFact(source="DS Team", target=target, relationship="R",
                        structural=structural, score=score, support=1)


def test_the_cheap_ranking_keeps_membership_when_the_keep_bound_binds():
    """The floor score is what made membership edges the first thing truncated."""
    facts = [_edge(f"Topic{i}", structural=False, score=0.9) for i in range(3)]
    facts.append(_edge("Ana", structural=True, score=0.0452))

    ranked = NeighborScorer._cheap_ranked(facts)

    assert ranked[0].target == "Ana"


def test_the_digest_renders_structural_facts_before_extracted_ones():
    ev = Evidence(facts=[_edge("Ranking", structural=False, score=0.9),
                         _edge("Ana", structural=True, score=0.0452)])
    digest = AnswerComposer(make_settings())._digest(ev, strict=False)
    facts_section = digest.split("FACTS:")[1]
    assert facts_section.index("Ana") < facts_section.index("Ranking")


# ── R8 · an unscored candidate is not an irrelevant one ─────────────────────


def test_a_candidate_the_model_omitted_falls_back_to_its_cheap_score():
    """`relevance.get(idx, 0.0)` zeroed the term carrying 90% of the weight, so an
    omission from the model's `scores` array silently DELETED the row — and the
    structural edges, sitting at the cheap floor, went first."""
    # A margin nothing can clear, so the escalation gate always fires.
    settings = make_settings(scoring_small_threshold=0, scoring_separation_margin=1.0,
                             scoring_batch_token_budget=10_000)
    facts = [_edge("Scored", structural=False, score=0.9),
             _edge("Omitted", structural=False, score=0.8),
             _edge("Weak", structural=False, score=0.1)]

    scorer = NeighborScorer(None, settings, None, ScoringBudget(4))

    async def _score_batch(batch, focus):
        # The model returns a score for index 0 and for index 2, skipping 1.
        return 0, {0: 0.95, 2: 0.9}

    scorer._score_batch = _score_batch
    kept, tier = asyncio.run(scorer.rescore(facts, "focus", keep=2))

    assert tier == "llm"
    assert "Omitted" in [f.target for f in kept], (
        "an unmentioned candidate must keep its cheap standing, not be zeroed"
    )


# ── R4 · the enrichment budget must be spent on the subject ─────────────────


def _person_fact(target, **kw):
    return RelationFact(source="DS Team", target=target, relationship="HAS_MEMBER",
                        **kw)


def test_document_targets_never_spend_an_enrichment_slot():
    """A fact's target is as often a document as an entity, and a document title
    matches no Person node and no entity — it returned nothing and spent a slot
    doing it ("FAQ TSV file", "HRIS Certification document")."""
    ev = Evidence(facts=[
        _person_fact("FAQ TSV file", target_type="Document"),
        _person_fact("Ana", target_type="Person"),
    ])
    # Candidates are RANK-ordered now (deterministic, not pool order), so the
    # claim here is membership: the document is out, the people are in.
    assert sorted(_ordered_names(ev, skip=set())) == ["Ana", "DS Team"]


def test_a_source_title_that_is_also_an_entity_is_still_a_candidate():
    """The exclusion is about what the row IS, not about the string."""
    ev = Evidence(
        facts=[_person_fact("Embedding Platform")],
        entities=[EntityHit(name="Embedding Platform", type="CONCEPT")],
        sources=[SourceDocument(doc_id="d1", title="Embedding Platform")],
    )
    assert "Embedding Platform" in _ordered_names(ev, skip=set())


def test_a_persons_own_profile_document_never_excludes_that_person():
    """A people-object document's TITLE IS the person's name, so the title filter
    would otherwise drop a person by way of their own profile page."""
    ev = Evidence(
        facts=[_person_fact("Ana", target_type="Person")],
        sources=[SourceDocument(doc_id="u1", title="Ana")],
    )
    assert "Ana" in _ordered_names(ev, skip=set())


def test_people_lead_the_candidates_for_the_passes_that_only_match_people():
    """One broad topic read landing first took all 60 profile slots and all 30
    claim slots before the question's own people were in the pool: 43 profiles
    filled, none in the department asked about."""
    ev = Evidence(facts=[
        _person_fact("Sentiment Analysis", target_type="CONCEPT"),
        _person_fact("S3 Bucket", target_type="CONCEPT"),
        _person_fact("Ana", target_type="Person"),
    ])

    assert _ordered_names(ev, skip=set(), people_first=True)[0] == "Ana"
    # Ordering ONLY — a concept loses priority, never presence.
    assert set(_ordered_names(ev, skip=set(), people_first=True)) == set(
        _ordered_names(ev, skip=set()))


def test_a_person_is_recognised_by_status_when_the_row_carries_no_type():
    ev = Evidence(facts=[
        _person_fact("Concept", target_type="CONCEPT"),
        RelationFact(source="Ana", target="Charter", relationship="AUTHORED",
                     target_type="Document", source_status="Active"),
    ])
    assert _ordered_names(ev, skip=set(), people_first=True)[0] == "Ana"


# ── R5 · a document read must be a document read ────────────────────────────


def test_the_character_budget_is_per_chunk_not_a_total_across_the_document():
    """The cap was spent twice — bound into `left(c.text, $chars)` AND re-used as
    a running total — so the first chunk exhausted the whole document and a
    roster came back as one arbitrary chunk described as read start to finish."""
    rows = [{"chunk_id": f"c{i}", "doc_id": "d1", "doc_title": "Roster",
             "text": "x" * 6000, "graph": WIKIPEDIA} for i in range(3)]
    repo, seen = _repo(rows)

    result = asyncio.run(repo.fetch_document_text(_ctx(), "d1", max_chars=6000))

    assert seen["chars"] == 6000, "the per-chunk cap still binds in Cypher"
    assert [len(c.text) for c in result.chunks] == [6000, 6000, 6000]


def test_a_document_read_says_whether_it_returned_the_whole_document():
    settings = make_settings(document_chunk_limit=2)
    rows = [{"chunk_id": f"c{i}", "doc_id": "d1", "text": "body", "graph": WIKIPEDIA}
            for i in range(2)]
    repo, seen = _repo(rows, settings)

    result = asyncio.run(repo.fetch_document_text(_ctx(), "d1", max_chars=6000))

    assert seen["limit"] == 2
    assert result.truncated is True and len(result.chunks) == 2


def test_a_short_document_is_not_reported_as_truncated():
    rows = [{"chunk_id": "c0", "doc_id": "d1", "text": "body", "graph": WIKIPEDIA}]
    repo, _ = _repo(rows)
    result = asyncio.run(repo.fetch_document_text(_ctx(), "d1", max_chars=6000))
    assert result.truncated is False


def test_the_document_read_still_has_a_total_ceiling_of_its_own():
    """Un-conflating the per-chunk cap from the running total must not remove
    the total: the graph-chunk budget downstream counts CHUNKS, never
    characters, so without a ceiling here one read could hand the agent
    chunk_limit x chars_per_chunk (300k) of context. The ceiling is its own,
    much larger knob — and a read it cuts is a partial read and says so."""
    settings = make_settings(document_total_chars=10_000)
    rows = [{"chunk_id": f"c{i}", "doc_id": "d1", "text": "x" * 6000,
             "graph": WIKIPEDIA} for i in range(3)]
    repo, _ = _repo(rows, settings)

    result = asyncio.run(repo.fetch_document_text(_ctx(), "d1", max_chars=6000))

    assert [len(c.text) for c in result.chunks] == [6000, 4000]
    assert result.truncated is True


def test_a_document_under_the_total_ceiling_is_untouched_by_it():
    rows = [{"chunk_id": f"c{i}", "doc_id": "d1", "text": "x" * 6000,
             "graph": WIKIPEDIA} for i in range(3)]
    repo, _ = _repo(rows)

    result = asyncio.run(repo.fetch_document_text(_ctx(), "d1", max_chars=6000))

    assert [len(c.text) for c in result.chunks] == [6000] * 3
    assert result.truncated is False


# ── R9 · the richer copy of a chunk must survive ────────────────────────────


def _quote(chunk_id, text, graph=WIKIPEDIA):
    return ChunkQuote(chunk_id=chunk_id, doc_id="d1", text=text, graph=graph)


def test_the_pool_keeps_the_longer_copy_of_a_chunk_it_already_holds():
    """`setdefault` was first-wins on chunk_id, and the agent's own documented
    sequence (get_source_documents, then get_document_text) produces exactly the
    losing order — so synthesis answered from a 2,500-char prefix of a roster
    the model had been told it read in full."""
    pool = EvidencePool()
    pool.add_quotes([_quote("c1", "x" * 2500)])
    pool.add_quotes([_quote("c1", "x" * 6000)])

    assert [len(q.text) for q in pool.quotes] == [6000]


def test_a_shorter_re_read_never_displaces_the_richer_copy():
    pool = EvidencePool()
    pool.add_quotes([_quote("c1", "x" * 6000)])
    pool.add_quotes([_quote("c1", "x" * 2500)])
    assert [len(q.text) for q in pool.quotes] == [6000]


def test_an_upgrade_keeps_a_relevance_already_stamped_on_the_pooled_copy():
    pool = EvidencePool()
    scored = _quote("c1", "x" * 2500)
    scored.relevance = 0.71
    pool.add_quotes([scored])
    pool.add_quotes([_quote("c1", "x" * 6000)])
    assert [(len(q.text), q.relevance) for q in pool.quotes] == [(6000, 0.71)]


def test_a_re_read_of_a_pooled_chunk_does_not_spend_a_second_budget_slot():
    """The two widths of one chunk normalise to different text, so the gate scored
    the re-read as brand new and charged a slot for text already pooled."""
    pool = EvidencePool()
    pool.add_quotes([_quote("c1", "x" * 2500)])

    kept = budget_graph_quotes(pool, 2, [_quote("c1", "x" * 6000),
                                         _quote("c2", "new"),
                                         _quote("c3", "newer")])

    assert [q.chunk_id for q in kept] == ["c1", "c2"], (
        "the richer copy passes free; the cap still admits exactly one new chunk"
    )


def test_a_copy_that_adds_nothing_is_dropped_rather_than_re_kept():
    pool = EvidencePool()
    pool.add_quotes([_quote("c1", "x" * 6000)])
    assert budget_graph_quotes(pool, 5, [_quote("c1", "x" * 2500)]) == []


# ── R11 · one spelling per relationship type ────────────────────────────────


def test_contribution_reads_emit_the_graphs_own_relationship_spelling():
    """`.lower()` made `contributed_to` a DIFFERENT fact from the expansion's
    `CONTRIBUTED_TO`, so one edge was stored twice — two rows of the fact window,
    two trace edges, and two enrichment candidate slots."""
    rows = [{"person": "Ana", "role": "CONTRIBUTED_TO", "via": "Ranking",
             "document": "Charter", "contributions": 2, "graph": WIKIPEDIA}]
    settings = make_settings()
    for read in (GraphReadRepository.topic_contributors,
                 GraphReadRepository.document_contributors):
        repo, _ = _repo(rows, settings)
        [fact] = asyncio.run(read(repo, _ctx(), "Ranking"))
        assert fact.relationship == "CONTRIBUTED_TO"


def test_the_two_reads_of_one_edge_collapse_to_a_single_pooled_fact():
    pool = EvidencePool()
    pool.add_facts([RelationFact(source="Ana", target="Charter",
                                 relationship="AUTHORED")])
    added = pool.add_facts([RelationFact(source="Ana", target="Charter",
                                         relationship="AUTHORED")])
    assert added == [] and len(pool.snapshot().facts) == 1


# ── R7 · a clipped list must say it was clipped ─────────────────────────────


def test_find_entity_paths_no_longer_overstates_what_it_returned():
    """`returned` counted every fact while the list rendered `facts[:_rows(rc)]`,
    so the tool's own count was wrong — and the agent, instructed to COVER THE
    WHOLE SET, had no way to tell a complete result from a clipped one."""
    from ai_graph_retrieval.agent import tools

    facts = [RelationFact(source="A", target=f"T{i}", relationship="R")
             for i in range(30)]
    rc = SimpleNamespace(deep=False, settings=make_settings())
    rows = tools._rows(rc)

    payload = tools._listing(facts, [{} for _ in facts[:rows]], key="relationships")

    assert payload["returned"] == rows
    assert payload["total_found"] == 30
    assert payload["truncated"] is True


def test_a_complete_list_is_not_flagged_as_truncated():
    from ai_graph_retrieval.agent import tools

    payload = tools._listing(["a", "b"], [{}, {}], key="people")
    assert payload["truncated"] is False and payload["total_found"] == 2
