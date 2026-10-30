"""Person-anchored authorship: documents BY a person.

The 2026-08-25 RCA: the agent had a tool for "who wrote this document" and none
for "what did this person write". Asked what a manager's direct reports had
published, it returned six posts written by the MANAGER; asked for everything one
person had written, it returned 11k characters of concepts ("Atlassian
arXiv", "EPIC") and named none of their 50 documents.

Both came from the same substitution: with no person-anchored tool, the agent
called the document-anchored one with a person's NAME, which matches documents
TITLED after that person — their profile page, their welcome post — and reports
who wrote those.

These tests pin the two halves that make the failure impossible to reproduce:
the repository maps the row (carrying the publication date, which the bad answer
claimed was unavailable), and the toolset offers the person-anchored direction
with a description that says which is which.
"""
from __future__ import annotations

import asyncio
from datetime import date

from ai_graph_retrieval.core.access import AccessContext
from ai_graph_retrieval.graph.repository import GraphReadRepository
from tests.support import make_settings

_ROWS = [
    {"person": "Ada Lovelace", "job_title": "VP, Business Operations",
     "department": "Business Operations", "role": "AUTHORED",
     "document": "Introducing the Product Quality Dashboard",
     "doc_id": "d1", "url": "https://x/1", "date": "2026-01-13", "graph": "wikipedia"},
    {"person": "Ada Lovelace", "job_title": "VP, Business Operations",
     "department": None, "role": "EDITED", "document": "VPAT 2026 Plan",
     "doc_id": "d2", "url": "https://x/2", "date": "2026-01-06", "graph": "wikipedia"},
]


def _repo(rows):
    repo = GraphReadRepository(None, make_settings(), http=object())
    seen: dict = {}

    async def _stub_rows(_query, _ctx, **params):
        seen.update(params)
        return rows

    repo._rows = _stub_rows
    return repo, seen


def _ctx():
    return AccessContext(tenant_id="t", principals=("public",), request_id="r")


def test_rows_become_person_authored_document_facts_carrying_the_date():
    repo, _ = _repo(_ROWS)
    facts, _ = asyncio.run(repo.documents_by_author(_ctx(), "Ada Lovelace"))

    assert [f.target for f in facts] == [
        "Introducing the Product Quality Dashboard", "VPAT 2026 Plan"
    ]
    # Direction: the PERSON is the source, the document is the target. Reversed,
    # synthesis would describe documents as having written people.
    assert {f.source for f in facts} == {"Ada Lovelace"}
    assert {f.target_type for f in facts} == {"Document"}
    # The edge type survives, so "written" and "edited" stay distinguishable.
    assert [f.relationship for f in facts] == ["AUTHORED", "EDITED"]
    # The date the bad answer said was unavailable.
    assert [f.date for f in facts] == ["2026-01-13", "2026-01-06"]
    assert {f.date_basis for f in facts} == {"event"}
    assert {f.graph for f in facts} == {"wikipedia"}


def test_the_window_reaches_the_query_and_an_absent_window_is_unbounded():
    """`since`/`until` are the classifier's resolved window. Empty string means
    unbounded — the same sentinel RetrievalPlan.time_scope_from/to use — so a
    question with no window must not silently filter."""
    repo, seen = _repo(_ROWS)
    asyncio.run(repo.documents_by_author(
        _ctx(), "Ada Lovelace", since="2026-01-01", until="2026-12-31"))
    assert (seen["since"], seen["until"]) == ("2026-01-01", "2026-12-31")
    assert seen["who"] == "Ada Lovelace"

    repo, seen = _repo(_ROWS)
    asyncio.run(repo.documents_by_author(_ctx(), "Ada Lovelace"))
    assert (seen["since"], seen["until"]) == ("", "")


def test_a_blank_person_reads_nothing_rather_than_matching_everything():
    """`CONTAINS ''` matches every person in the graph, so the guard belongs
    before the read, not in the query."""
    repo, seen = _repo(_ROWS)
    assert asyncio.run(repo.documents_by_author(_ctx(), "   ")) == ([], [])
    assert not seen, "no query should have run"


def test_date_basis_is_absent_when_a_row_carries_no_date():
    """Undated rows still map — the window predicate excludes them, but a row
    that arrives unbounded must not claim an event date it hasn't got."""
    repo, _ = _repo([{**_ROWS[0], "date": None}])
    fact = asyncio.run(repo.documents_by_author(_ctx(), "X"))[0][0]
    assert fact.date is None and fact.date_basis is None


# ── the set is the unit of the read ───────────────────────────────
# A multi-part question ("who reports to X and what have they published") asks
# the same thing of every member of a set. Read one name at a time, the answer's
# completeness depended on how many calls the agent chose to emit — the same
# question named three reports on one run and thirteen on the next.


def test_the_set_wide_read_queries_every_name_and_merges_both_halves():
    calls: list[str] = []
    repo = GraphReadRepository(None, make_settings(), http=object())

    async def _stub_rows(_query, _ctx, **params):
        calls.append(params["who"])
        return [{**_ROWS[0], "doc_id": f"d-{params['who']}",
                 "document": f"doc for {params['who']}"}]

    repo._rows = _stub_rows
    facts, sources = asyncio.run(
        repo.documents_by_authors(_ctx(), ["Alice", "Bob", "Cara"])
    )

    assert calls == ["Alice", "Bob", "Cara"], "every name gets its own read"
    assert len(facts) == 3 and len(sources) == 3, "both halves merge across names"
    assert {s.doc_id for s in sources} == {"d-Alice", "d-Bob", "d-Cara"}


def test_the_row_limit_is_per_person_so_one_prolific_author_cannot_starve_the_set():
    """The reason this fans out instead of passing an IN-list to one query: with a
    single query the LIMIT is spent in ORDER BY order, so the most prolific author
    can consume all of it and leave the rest of the set looking as though it had
    published nothing."""
    repo, seen = _repo(_ROWS)
    asyncio.run(repo.documents_by_authors(_ctx(), ["Alice", "Bob"], limit=25))
    assert seen["limit"] == 25, "each name is read with the FULL limit, not a share of it"


def test_blank_and_duplicate_names_are_dropped_before_the_reads():
    """The set often arrives with a name in it twice, from two directions — and a
    blank name would match every person in the graph (`CONTAINS ''`)."""
    calls: list[str] = []
    repo = GraphReadRepository(None, make_settings(), http=object())

    async def _stub_rows(_query, _ctx, **params):
        calls.append(params["who"])
        return []

    repo._rows = _stub_rows
    asyncio.run(repo.documents_by_authors(
        _ctx(), ["Alice", " alice ", "", "   ", "Bob", "Alice"]))
    assert calls == ["Alice", "Bob"]

    assert asyncio.run(repo.documents_by_authors(_ctx(), [])) == ([], [])
    assert asyncio.run(repo.documents_by_authors(_ctx(), ["  "])) == ([], [])
    assert calls == ["Alice", "Bob"], "no read for an empty set"


def test_both_authorship_directions_are_offered_and_say_which_is_which():
    """The agent picked the wrong direction because only one existed. Both are
    now present, and each description names the other — the steer is the fix, so
    it is worth a test."""
    from ai_graph_retrieval.agent.tools import research_tools

    tools = {t.name: t for t in research_tools(graph_text=True, people=True)}
    assert "find_authored_documents" in tools
    assert "find_contributors" in tools

    by_person = tools["find_authored_documents"].description or ""
    by_title = tools["find_contributors"].description or ""
    # Each points at the other, so the model can tell them apart.
    assert "find_contributors" in by_person
    assert "find_authored_documents" in by_title
    # And the person-anchored tool takes no date argument: the window comes from
    # the classifier's resolution, leaving the model no date to invent. Its one
    # argument is the SET of people — a multi-part question asks the
    # same thing of every member, and one call for the set is what stops the
    # answer's completeness depending on how many calls the agent chose to emit.
    params = tools["find_authored_documents"].params_json_schema["properties"]
    assert set(params) == {"people"}
    assert params["people"]["type"] == "array"


def test_facts_are_recency_scored_so_the_digest_cap_cannot_starve_them():
    """The synthesis digest sorts on (support, score) and truncates at
    digest_max_facts. Authorship carries support 1 by nature — nothing
    corroborates "X wrote Y" beyond the edge — so leaving score at its 0.0
    default put these BELOW every extracted fact of the same support and made
    them first to be dropped from a large pool. Scored with the house recency
    curve instead of an invented support value."""
    rows = [
        {**_ROWS[0], "document": "recent", "date": date.today().isoformat()},
        {**_ROWS[0], "document": "old", "date": "2019-01-01"},
        {**_ROWS[0], "document": "undated", "date": None},
    ]
    repo, _ = _repo(rows)
    facts, _ = asyncio.run(repo.documents_by_author(_ctx(), "X"))
    got = {f.target: f.score for f in facts}

    assert got["recent"] > 0, "a scored fact must not sit at the unranked default"
    assert got["recent"] > got["old"], "newer work ranks first — what the question wants"
    assert got["old"] > 0
    # Undated rows still rank (the house curve gives them a floor), just below
    # anything recent.
    assert 0 < got["undated"] < got["recent"]


def test_the_recency_curve_is_the_service_wide_one_not_a_second_formula():
    """One curve, so a change to the half-life moves every ranking together."""
    from ai_graph_retrieval.retrieval.quality import freshness
    from tests.support import make_settings

    s = make_settings()
    today = date.today().isoformat()
    repo, _ = _repo([{**_ROWS[0], "date": today}])
    fact = asyncio.run(repo.documents_by_author(_ctx(), "X"))[0][0]
    expected = round(
        s.trust_w_recency * freshness(today, date.today(), s.trust_recency_halflife_days), 4
    )
    assert fact.score == expected


# ── the documents have to be CITABLE, not just listed ────────────────────────
# Citations resolve against evidence.sources alone (synthesis/citations.py): a
# marker whose title matches no source is dropped, and a source with no url
# cannot become a citation. So a document known to the request only as a FACT is
# uncitable. Measured on the generalization sweep: a substantive answer about two
# people's contributions carried 0 citations, while a run where the agent
# happened to look each title up separately carried 14.


def test_the_documents_come_back_as_citable_sources_too():
    repo, _ = _repo(_ROWS)
    facts, sources = asyncio.run(repo.documents_by_author(_ctx(), "Ada Lovelace"))

    assert len(sources) == len(facts) == 2
    first = sources[0]
    assert first.title == "Introducing the Product Quality Dashboard"
    assert first.doc_id == "d1"
    # The url is what makes it resolvable — without one the chip is dropped.
    assert first.url == "https://x/1"
    assert first.date == "2026-01-13"
    assert first.graph == "wikipedia"


def test_traversal_reached_documents_claim_no_relevance_or_mentions():
    """These arrive by walking an edge, not by matching a chunk. Inventing either
    signal would push them up the trust ranking on evidence that does not exist —
    and add_sources keeps the max per signal, so a real value from the chunk lane
    still wins."""
    repo, _ = _repo(_ROWS)
    _, sources = asyncio.run(repo.documents_by_author(_ctx(), "X"))
    assert {s.relevance for s in sources} == {0.0}
    assert {s.mentions for s in sources} == {0}


def test_the_tool_registers_both_halves_in_the_evidence_pool():
    """The fix only works end to end if the TOOL adds the sources as well as the
    facts — invoked for real, so a refactor that drops one half is caught."""
    from types import SimpleNamespace

    from ai_graph_retrieval.agent.tools import ResearchContext, find_authored_documents
    from ai_graph_retrieval.core.models import EvidencePool

    class _Repo:
        """Delegates to the REAL set-wide method over a stubbed row reader, so the
        fan-out the tool depends on is exercised rather than faked away."""

        async def documents_by_authors(self, ctx, who, *, since="", until="", limit=None):
            repo, _ = _repo(_ROWS)
            return await repo.documents_by_authors(
                _ctx(), who, since=since, until=until, limit=limit)

    class _Step:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    async def _graph_delta(**_kw):
        return None

    pool = EvidencePool()
    rc = ResearchContext(
        repo=_Repo(), ctx=SimpleNamespace(log_fields=dict), embedder=None,
        settings=make_settings(), evidence=pool,
        progress=SimpleNamespace(step=lambda *a, **k: _Step(),
                                 graph_delta=_graph_delta),
        usage=None,
    )
    asyncio.run(find_authored_documents.on_invoke_tool(
        SimpleNamespace(context=rc, tool_name="find_authored_documents",
                        tool_call_id="t1", run_config=None),
        '{"people": ["Ada Lovelace"]}',
    ))

    snapshot = pool.snapshot()
    assert len(snapshot.facts) == 2, "the authorship edges"
    titles = {d.title for d in snapshot.sources}
    assert "Introducing the Product Quality Dashboard" in titles, "the citable documents"
    assert all(d.url for d in snapshot.sources), "a source with no url cannot be cited"


def test_the_shared_window_is_balanced_per_author_not_swept_by_recency():
    """The model window is budgeted per subject (_rows_per_subject), so one
    prolific recent author must not spend the rows allotted to the others —
    while the date rank still lets the SECOND graph's work into the window
    (the old positional slice kept the first graph's block instead)."""
    import json
    from types import SimpleNamespace

    from ai_graph_retrieval.agent.tools import ResearchContext, find_authored_documents
    from ai_graph_retrieval.core.models import EvidencePool, RelationFact

    facts = [RelationFact(source="Prolific", target=f"new-{i}", relationship="authored",
                          date=f"2026-08-{i + 1:02d}", graph="wikipedia")
             for i in range(6)]
    facts += [RelationFact(source="Quiet", target="cf-recent", relationship="authored",
                           date="2023-05-01", graph="arxiv"),
              RelationFact(source="Quiet", target="cf-early", relationship="authored",
                           date="2022-05-01", graph="arxiv")]

    class _FanRepo:
        async def documents_by_authors(self, ctx, who, *, since="", until="",
                                       limit=None):
            return list(facts), []

    class _Step:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    async def _graph_delta(**_kw):
        return None

    rc = ResearchContext(
        repo=_FanRepo(), ctx=SimpleNamespace(log_fields=dict), embedder=None,
        settings=make_settings(agent_tool_rows=2), evidence=EvidencePool(),
        progress=SimpleNamespace(step=lambda *a, **k: _Step(),
                                 graph_delta=_graph_delta),
        usage=None,
    )
    out = json.loads(asyncio.run(find_authored_documents.on_invoke_tool(
        SimpleNamespace(context=rc, tool_name="find_authored_documents",
                        tool_call_id="t1", run_config=None),
        '{"people": ["Prolific", "Quiet"]}',
    )))

    rows = out["documents"]
    assert len(rows) == 4, "2 rows per subject for 2 subjects"
    counts = {p: sum(r["person"] == p for r in rows) for p in ("Prolific", "Quiet")}
    assert counts == {"Prolific": 2, "Quiet": 2}, "each author keeps their budget"
    docs = [r["document"] for r in rows]
    assert "cf-recent" in docs, "the second graph's work reaches the window"
    assert docs[0] == "new-5", "within the round-robin, newest still leads"


# ── reading a document means reading its chunks, and reading the SET ─
# Verified on both live graphs: Documents carry NO body text of their own (0 of
# 13,008 in arXiv, 0 of 3,263 in the intranet graph) and Chunks carry NO
# doc_id property (0 of 17,850 / 4,747) — the ONLY link is the HAS_CHUNK edge, so
# a doc_id property join would return nothing at all.


def test_document_text_reads_chunks_over_the_edge_not_a_doc_id_join():
    from ai_graph_retrieval.graph import cypher

    q = cypher.DOCUMENT_TEXT
    assert "HAS_CHUNK" in q, "the edge is the only Document->Chunk link that exists"
    assert "c.text" in q, "the text lives on the chunk, never on the document"
    assert "Chunk {doc_id" not in q, (
        "a doc_id property join finds nothing — chunks carry no doc_id"
    )


def test_the_document_read_takes_a_set_of_ids():
    """A question enumerating a series needs every member's document read. One id
    per call spends a round per member, so two of twelve got read and the other ten
    rendered as a single line from whatever calendar listed them — while their
    documents held 24,000-27,000 chars each, as much as the two that came out rich."""
    from ai_graph_retrieval.agent.tools import research_tools

    tools = {t.name: t for t in research_tools(graph_text=True, people=True)}
    params = tools["get_document_text"].params_json_schema["properties"]
    assert set(params) == {"doc_ids"}
    assert params["doc_ids"]["type"] == "array"


# ── one row per HUMAN, not per node ────────────────────────────────
# Ingestion emits the same person as several Person nodes — a canonical one plus
# "(Deactivated)"/"(Unlicensed)" twins and run-together name artefacts — all
# carrying the SAME person_user_id. Measured on the live arXiv graph: 127
# user_ids across 263 nodes, and one engineer appearing as three nodes with
# identical contribution counts took three of the top four rows of a people
# answer. Collapsing on the id is EXACT — no name comparison anywhere.


def test_people_reads_collapse_duplicate_nodes_on_the_identity_field():
    from ai_graph_retrieval.graph import cypher

    for q in (cypher.DOCUMENT_CONTRIBUTORS, cypher.TOPIC_CONTRIBUTORS):
        assert "person.person_user_id" in q, "collapse on the id ingestion supplies"
        assert "coalesce(person.degree, 0) DESC" in q, (
            "the PRIMARY record's name must win the collapse, by degree"
        )


def test_the_collapse_never_compares_names_to_each_other():
    """Rule 8: the fix must not become fuzzy name matching. The only name use is
    picking a representative for a group already decided by the id."""
    from ai_graph_retrieval.graph import cypher

    for q in (cypher.DOCUMENT_CONTRIBUTORS, cypher.TOPIC_CONTRIBUTORS):
        for banned in ("apoc.text", "levenshtein", "STARTS WITH person",
                       "CONTAINS person", "split(person"):
            assert banned not in q, f"{banned} would make this name-similarity matching"


def test_a_person_with_no_id_still_survives_the_collapse():
    """186 of 1,111 Person nodes carry no user_id, so the group key falls back to
    the name — they must not collapse into one anonymous bucket."""
    from ai_graph_retrieval.graph import cypher

    for q in (cypher.DOCUMENT_CONTRIBUTORS, cypher.TOPIC_CONTRIBUTORS):
        assert "coalesce(person.person_user_id,\n" in q
        assert "person.person_full_name, person.user_name, person.name)) AS identity" in q
