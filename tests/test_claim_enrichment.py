"""Post-agent-loop claim top-up (retrieval/deep_research.enrich_claims).

The agent-led redesign removed the old deterministic expansion
(search_expand_top=30 / search_claim_top=10) and with it the per-entity facts
— roles, titles, status — for the people and things an answer ENUMERATES: a
short loop finds the set (one expand call returns a whole team) but has no
rounds left to describe each member, so answers listed bare names (manager
report, 2026-08-10). These tests pin the replacement: after the loop, the lane
claim-fetches the surfaced-but-undescribed entities in one parallel batch,
capped by the profile, additive-only (a failure never costs the answer).

    uv run pytest tests/test_claim_enrichment.py -v
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from ai_graph_retrieval.core.models import (
    ChunkQuote,
    ClaimFact,
    EntityHit,
    EvidencePool,
    RelationFact,
    SourceDocument,
)
from ai_graph_retrieval.retrieval.deep_research import (
    _uncovered_names,
    enrich_claims,
    enrich_graph_chunks,
    enrich_profiles,
)
from ai_graph_retrieval.retrieval.person_status import PersonIdentities
from tests.support import make_settings


class _Step:
    def __init__(self) -> None:
        self.detail: str | None = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class _Repo:
    """Records the names claim-fetched; returns one claim per known name."""

    # The repository contract the graph-chunk relevance pass reads.
    # None ⇒ no vector arm, so the pass returns without scoring.
    sources = ()
    vector_store = None

    async def person_identities(self, ctx, names):
        """The contract the person-identity pass reads. Empty ⇒ every
        name already agrees across graphs and nobody has left, so neither the
        rename nor the exclusion rewrites anything — these tests are about the
        claim and profile top-ups."""
        return PersonIdentities()

    def __init__(self, known: set[str] | None = None, fail: bool = False) -> None:
        self.fetched: list[str] = []
        self._known = known if known is not None else set()
        self._fail = fail

    async def fetch_claims(self, ctx, name, since_iso=None):
        if self._fail:
            raise RuntimeError("graph down")
        self.fetched.append(name)
        if name not in self._known:
            return []
        return [ClaimFact(subject=name, predicate="holds role", object="Engineer")]


def _sc(pool: EvidencePool, repo: _Repo, cap: int) -> SimpleNamespace:
    async def graph_delta(**kwargs):
        return None

    return SimpleNamespace(
        evidence=pool,
        repo=repo,
        ctx=SimpleNamespace(log_fields=lambda: {}),
        profile=SimpleNamespace(claim_enrich_top=cap, profile_enrich_top=0,
                                mention_enrich_top=0),
        settings=SimpleNamespace(agent_claim_enrich_per_entity=5,
                                 person_identity_max_names=400,
                                 inactive_people_filter_enabled=True),
        historical=False,
        progress=SimpleNamespace(step=lambda *a, **k: _Step(), graph_delta=graph_delta,
                                 # The real ProgressReporter always carries this;
                                 # the top-up passes read it to parent their stage
                                 # span.
                                 span_parent=None),
    )


def _team_pool() -> EvidencePool:
    """A manager expanded to three reports; only the manager has claims."""
    pool = EvidencePool()
    pool.add_entities([EntityHit(name="Manager A"), EntityHit(name="Report D")])
    pool.add_facts([
        RelationFact(source="Manager A", target="Report B", relationship="MANAGES"),
        RelationFact(source="Manager A", target="Report C", relationship="MANAGES"),
    ])
    pool.add_claims([
        ClaimFact(subject="Manager A", predicate="holds role", object="Principal"),
    ])
    return pool


def test_traversal_neighbors_are_fetched_before_pool_entities():
    """The names the loop actually walked to lead the batch — they are what the
    answer will enumerate — and a name a claim already DESCRIBES is skipped."""
    pool = _team_pool()
    repo = _Repo(known={"Report B", "Report C"})
    asyncio.run(enrich_claims(_sc(pool, repo, cap=10)))

    assert repo.fetched == ["Report B", "Report C", "Report D"]
    subjects = {c.subject for c in pool.snapshot().claims}
    assert {"Report B", "Report C"} <= subjects


def test_the_cap_bounds_the_batch():
    pool = _team_pool()
    repo = _Repo()
    asyncio.run(enrich_claims(_sc(pool, repo, cap=1)))
    assert repo.fetched == ["Report B"]


def test_zero_cap_disables_the_pass_entirely():
    """Cap 0 means the pass is structurally unreachable —
    not even a progress step is opened."""
    pool = _team_pool()
    repo = _Repo(fail=True)  # would raise if reached
    sc = _sc(pool, repo, cap=0)
    sc.progress = None  # any progress use would crash the test
    asyncio.run(enrich_claims(sc))
    assert repo.fetched == []


def test_being_mentioned_by_a_claim_is_not_being_described():
    """A claim ABOUT the manager that names a report as its object does not
    cover the report — only names appearing as claim SUBJECTS are covered."""
    pool = _team_pool()
    pool.add_claims([
        ClaimFact(subject="Manager A", predicate="manages", object="Report B"),
    ])
    assert _uncovered_names(_sc(pool, _Repo(), 10), cap=10) == [
        "Report B", "Report C", "Report D",
    ]


def test_a_failed_fetch_never_costs_the_answer():
    """Additive-only: the graph going down after a successful loop degrades to
    the loop's evidence, it does not raise into the request."""
    pool = _team_pool()
    before = pool.size()
    asyncio.run(enrich_claims(_sc(pool, _Repo(fail=True), cap=10)))
    assert pool.size() == before


def test_nothing_uncovered_means_no_fetch_and_no_step():
    pool = EvidencePool()
    pool.add_entities([EntityHit(name="Manager A")])
    pool.add_claims([
        ClaimFact(subject="Manager A", predicate="holds role", object="Principal"),
    ])
    repo = _Repo()
    sc = _sc(pool, repo, cap=10)
    sc.progress = None  # any progress use would crash the test
    asyncio.run(enrich_claims(sc))
    assert repo.fetched == []


def test_one_entitys_flood_cannot_fill_the_digest_window():
    """The digest renders the pool's FIRST digest_max_claims claims, so an
    unbounded per-entity return (≤60 per graph) from the first pick would push
    every later member's role claim out of the prompt. Each pick contributes
    only claims whose SUBJECT it is — a claim that merely mentions it describes
    something else — and at most agent_claim_enrich_per_entity of those."""

    class _FloodRepo(_Repo):
        async def fetch_claims(self, ctx, name, since_iso=None):
            self.fetched.append(name)
            about = [ClaimFact(subject=name, predicate=f"p{i}", object=f"o{i}")
                     for i in range(20)]
            mentions = [ClaimFact(subject="Something Else", predicate="mentions",
                                  object=name)]
            return about + mentions

    pool = _team_pool()
    asyncio.run(enrich_claims(_sc(pool, _FloodRepo(), cap=10)))
    added = [c for c in pool.snapshot().claims if c.subject != "Manager A"]
    by_subject: dict[str, int] = {}
    for c in added:
        by_subject[c.subject] = by_subject.get(c.subject, 0) + 1
    assert "Something Else" not in by_subject          # mention-only claims dropped
    assert by_subject == {"Report B": 5, "Report C": 5, "Report D": 5}


def test_the_lane_runs_the_topup_after_the_loop(monkeypatch):
    """DeepResearchStrategy wiring: what the agent surfaced during ITS run is
    what the top-up describes — the pass reads the pool after the loop ends."""
    import ai_graph_retrieval.agent.runtime as runtime
    from ai_graph_retrieval.retrieval.deep_research import DeepResearchStrategy

    class _FakeAgent:
        def __init__(self, settings, *, clarify, graph_text,
                     research=False) -> None:
            pass

        async def run(self, sc, scorer) -> None:
            sc.evidence.add_facts([RelationFact(
                source="Manager A", target="Report B", relationship="MANAGES",
            )])

    monkeypatch.setattr(runtime, "ResearchAgent", _FakeAgent)
    repo = _Repo(known={"Report B"})
    sc = _sc(EvidencePool(), repo, cap=10)
    sc.settings = SimpleNamespace(scoring_enabled=False,
                                  agent_claim_enrich_per_entity=5,
                                  person_identity_max_names=400,
                                  inactive_people_filter_enabled=True,
                                  people_graph_enabled=False)
    sc.profile = SimpleNamespace(
        relationship_enrich_top=0,
        mid_research_clarify=False, graph_text_tools=False, deep_tool_rows=False,
        scoring_budget=0, claim_enrich_top=10, profile_enrich_top=0, mention_enrich_top=0,
    )
    asyncio.run(DeepResearchStrategy().run(sc, plan=None))

    assert repo.fetched == ["Report B", "Manager A"]
    assert {c.subject for c in sc.evidence.snapshot().claims} == {"Report B"}


def test_a_failed_trace_delta_never_costs_the_answer():
    """Additive means the WHOLE pass: the claims are already in the pool when
    the UI trace delta is emitted, so an emitter failure is logged and
    swallowed — never raised into a retrieval that has its evidence."""
    pool = _team_pool()

    async def _boom(**_kw):
        raise RuntimeError("ws transport down")

    sc = _sc(pool, _Repo(known={"Report B"}), cap=10)
    sc.progress = SimpleNamespace(step=lambda *a, **k: _Step(), graph_delta=_boom,
                                  span_parent=None)
    asyncio.run(enrich_claims(sc))  # must not raise
    assert "Report B" in {c.subject for c in pool.snapshot().claims}


# ── profile top-up (structured node fields) ──────────────────────────────────


class _ProfileRepo(_Repo):
    """Records the batch it was asked for; returns profiles for known names."""

    def __init__(self, known: dict | None = None, fail: bool = False) -> None:
        super().__init__()
        self.asked: list[str] | None = None
        self._profiles = known or {}
        self._fail = fail

    async def fetch_people_profiles(self, ctx, names):
        if self._fail:
            raise RuntimeError("graph down")
        self.asked = list(names)
        profiles = [
            EntityHit(name=n, type="Person", profile=self._profiles[n])
            for n in names if n in self._profiles
        ]
        # Real contract: each profiled person's OWN document rides along, so the
        # profile facts have something citable behind them.
        docs = [
            SourceDocument(doc_id=f"uid-{n}", title=n, url=f"https://people/{n}")
            for n in names if n in self._profiles
        ]
        return profiles, docs


def _sc_profile(pool, repo, cap=25):
    sc = _sc(pool, repo, cap=0)          # claim pass off; profile pass under test
    sc.profile = SimpleNamespace(claim_enrich_top=0, profile_enrich_top=cap)
    return sc


def test_profiles_are_fetched_for_traversal_neighbours_in_one_batch():
    """The roster's people arrive as relationship TARGETS — a name and nothing
    else. One batched read gives them their role/location, so the answer's
    columns stop depending on which claims the agent happened to fetch."""
    pool = _team_pool()
    repo = _ProfileRepo({"Report B": "Intern · Data Science",
                         "Report C": "Data Scientist · Delhi, IN"})
    asyncio.run(enrich_profiles(_sc_profile(pool, repo)))

    # one call; per fact the TARGET precedes the SOURCE, so the manager (source
    # of the first edge) lands second — traversal order, not alphabetical
    assert repo.asked == ["Report B", "Manager A", "Report C", "Report D"]
    by_name = {e.name: e.profile for e in pool.snapshot().entities}
    assert by_name["Report B"] == "Intern · Data Science"
    assert by_name["Report C"] == "Data Scientist · Delhi, IN"


def test_a_profile_enriches_a_person_already_in_the_pool():
    """Merge, never replace: the pooled entity keeps its description and gains
    the profile (EvidencePool.add_entities enriches by name)."""
    pool = EvidencePool()
    pool.add_entities([EntityHit(name="Report D", description="mentioned in a post")])
    repo = _ProfileRepo({"Report D": "Senior AI Engineer · Rajasthan, IN"})
    asyncio.run(enrich_profiles(_sc_profile(pool, repo)))

    (e,) = pool.snapshot().entities
    assert e.description == "mentioned in a post"
    assert e.profile == "Senior AI Engineer · Rajasthan, IN"


def test_profiled_people_bring_their_own_document_into_the_sources():
    """A roster assembled from REPORTS_TO edges and person properties has no
    document in the pool that supports it, so synthesis — which can only cite
    documents it was shown — correctly cited nothing (RCA 2026-08-13). The
    profile pass therefore pools each profiled person's OWN people document,
    giving those facts a citable source."""
    pool = _team_pool()
    repo = _ProfileRepo({"Report B": "Intern · Data Science",
                         "Report C": "Data Scientist · Delhi, IN"})
    asyncio.run(enrich_profiles(_sc_profile(pool, repo)))

    by_id = {s.doc_id: s for s in pool.snapshot().sources}
    assert set(by_id) == {"uid-Report B", "uid-Report C"}
    assert by_id["uid-Report B"].title == "Report B"
    assert by_id["uid-Report B"].url == "https://people/Report B"


def test_already_profiled_names_are_not_refetched():
    pool = EvidencePool()
    pool.add_entities([EntityHit(name="Report D", profile="Intern"),
                       EntityHit(name="Report E")])
    repo = _ProfileRepo({})
    asyncio.run(enrich_profiles(_sc_profile(pool, repo)))
    assert repo.asked == ["Report E"]


def test_the_profile_cap_bounds_the_batch_and_zero_disables_the_pass():
    pool = _team_pool()
    repo = _ProfileRepo({})
    asyncio.run(enrich_profiles(_sc_profile(pool, repo, cap=2)))
    assert repo.asked == ["Report B", "Manager A"]

    repo2 = _ProfileRepo({})
    asyncio.run(enrich_profiles(_sc_profile(pool, repo2, cap=0)))
    assert repo2.asked is None


def test_a_failed_profile_fetch_never_costs_the_answer():
    """Additive: the loop's evidence is already pooled, so a graph failure here
    is logged and swallowed — the claim pass and synthesis still run."""
    pool = _team_pool()
    before = pool.size()
    asyncio.run(enrich_profiles(_sc_profile(pool, _ProfileRepo(fail=True))))
    assert pool.size() == before


def test_the_lane_runs_profiles_before_claims():
    """Order matters for cost, not correctness: profiles are one batched read
    that fills the roster's columns; claims then add what node fields cannot
    carry. Both run after the loop, on every agent-mode request."""
    import inspect

    from ai_graph_retrieval.retrieval.deep_research import DeepResearchStrategy

    src = inspect.getsource(DeepResearchStrategy.run)
    assert src.index("enrich_profiles(sc)") < src.index("enrich_claims(sc)")


def test_a_former_employee_is_flagged_at_the_START_of_the_profile_line():
    """Measured (2026-08-12): with the status TRAILING a complete-looking
    profile, the fast answer model skimmed it and listed a departed colleague as
    a roster row in 1 of 3 runs. The first token of the line is what decides
    whether the person belongs in the answer at all, so the flag leads."""
    from ai_graph_retrieval.graph.repository import _profile_line

    line = _profile_line({"job_title": "Data Scientist", "department": "Data Science",
                          "location": "Pune, IN", "employment_status": "Inactive"})
    assert line.startswith("FORMER EMPLOYEE")
    assert "Data Scientist" in line          # the rest of the profile survives

    # "Active" is the unremarkable default and is never stated
    assert _profile_line({"job_title": "Intern", "employment_status": "Active"}) == "title: Intern"
    # nothing populated ⇒ no entity is added at all
    assert _profile_line({"employment_status": "Active"}) == ""


# ── profile line: org position + departure (conditional graph fields) ────────
# Every field below is CONDITIONAL in Wikipedia, so the line must read correctly
# for an IC (no reports), a manager, the CEO (no manager) and a leaver alike.

def _line(**row):
    from ai_graph_retrieval.graph.repository import _profile_line

    return _profile_line(row)


def test_profile_line_states_who_a_person_reports_to():
    line = _line(job_title="Principal Product Manager", department="Product",
                 manager="Carol Hart")
    assert line == ("title: Principal Product Manager · department: Product"
                    " · reports to Carol Hart")


def test_profile_line_counts_reports_and_wider_org_only_when_they_add_something():
    # A manager whose whole org is their direct reports: one count, not two.
    assert _line(job_title="Engineering Manager", direct_reports=23, org_size=23) == (
        "title: Engineering Manager · 23 direct reports"
    )
    # A manager with sub-managers: both counts carry distinct information.
    assert _line(job_title="VP, Marketing", direct_reports=30, org_size=63) == (
        "title: VP, Marketing · 30 direct reports · 63 in org"
    )
    # An individual contributor: a zero count is not a fact worth printing.
    assert _line(job_title="Data Scientist", direct_reports=0, org_size=0) == (
        "title: Data Scientist"
    )
    assert _line(job_title="Support Lead", direct_reports=1, org_size=1) == (
        "title: Support Lead · 1 direct report"
    )


def test_departure_date_leads_the_line_and_survives_a_missing_status():
    """`person_departed_on` is written only for leavers, so a value IS the
    departure — and on these graphs it is the only such signal, since
    person_employment_status is absent. It must lead, like the status flag."""
    line = _line(job_title="Data Scientist", department="Data Science",
                 departed_on="2026-02-14")
    assert line.startswith("FORMER EMPLOYEE")
    assert "left 2026-02-14" in line
    assert line.endswith("title: Data Scientist · department: Data Science")


def test_absent_departure_date_is_not_read_as_proof_of_being_active():
    """NULL means "no departure recorded", not "active" — so it adds nothing."""
    assert _line(job_title="Data Scientist", departed_on=None) == "title: Data Scientist"
    assert "FORMER" not in _line(job_title="Data Scientist", departed_on="")


def test_a_name_matching_several_people_gets_no_profile_rather_than_a_guess():
    """The evidence pool keys entities by NAME, so when one name matches several
    person nodes in a graph there is no way to know which the answer meant.
    Attributing either would state someone else's job — and with org counts in the
    line, someone else's headcount. The name is left bare instead."""
    import asyncio

    from ai_graph_retrieval.core.access import AccessContext
    from ai_graph_retrieval.graph.repository import GraphReadRepository

    rows = [
        # three distinct people sharing a name, in ONE graph → ambiguous
        {"name": "Sam Carter", "job_title": "Engineering Manager", "direct_reports": 23,
         "org_size": 24, "graph": "wikipedia",
         "doc_id": "uid-ak-1", "doc_title": "Sam Carter", "doc_url": "https://p/ak1"},
        {"name": "Sam Carter", "job_title": "Contractor", "department": "Architecture",
         "graph": "wikipedia",
         "doc_id": "uid-ak-2", "doc_title": "Sam Carter", "doc_url": "https://p/ak2"},
        {"name": "Sam Carter", "job_title": "Contractor", "department": "Engineering",
         "graph": "wikipedia"},
        # unambiguous within its graph → kept
        {"name": "Lena Park", "job_title": "VP, Marketing", "direct_reports": 30,
         "org_size": 63, "graph": "wikipedia",
         "doc_id": "uid-kl", "doc_title": "Lena Park", "doc_url": "https://p/kl",
         "doc_source_type": "people"},
        # the SAME human in a second graph is not ambiguity — different graph.
        # arXiv people carry no people document → NULL doc fields.
        {"name": "Lena Park", "job_title": "VP Marketing", "graph": "arxiv"},
    ]
    repo = GraphReadRepository(None, make_settings(), http=object())

    async def _stub_rows(*_args, **_kwargs):
        return rows

    repo._rows = _stub_rows  # stub the one read this exercises

    ctx = AccessContext(tenant_id="t", principals=("public",), request_id="r")
    got, docs = asyncio.run(repo.fetch_people_profiles(ctx, ["Sam Carter", "Lena Park"]))
    names = [p.name for p in got]
    assert "Sam Carter" not in names, "an ambiguous name must not be given a profile"
    assert names.count("Lena Park") == 2, "same person in both graphs is not ambiguity"
    assert "30 direct reports" in next(p.profile for p in got if p.graph == "wikipedia")

    # Documents follow the KEPT profiles only: an ambiguous name's docs would put
    # a citable source behind a fact the read declined to state. The graph with
    # no people documents contributes none, and the kept doc's fields carry over.
    assert [d.doc_id for d in docs] == ["uid-kl"]
    (kim_doc,) = docs
    assert (kim_doc.title, kim_doc.url, kim_doc.source_type) == (
        "Lena Park", "https://p/kl", "people")
    # The profile entity gains the person's own page url from the same row.
    assert next(p.url for p in got if p.graph == "wikipedia") == "https://p/kl"


# ── cross-graph person merge (Wikipedia is the system of record) ──────────────
# A arXiv person node is an identity stub (no HR attributes), and the same
# human's display name differs across graphs for ~a third of people on the dev
# graphs ("(Deactivated)" suffixes, dotted usernames). Rows therefore group by
# IDENTITY (person_user_id, email fallback — never name), fetch attribute donors
# by identity when a group has none, and render every name in the group from the
# group's most attribute-complete row.


def _merge_repo(first_rows, donor_rows=None, donor_fail=False):
    from ai_graph_retrieval.graph import cypher
    from ai_graph_retrieval.graph.repository import GraphReadRepository

    repo = GraphReadRepository(None, make_settings(), http=object())
    calls = {}

    async def _stub_rows(query, ctx, **params):
        if query is cypher.PEOPLE_PROFILES_BY_IDENTITY:
            calls["donor_params"] = params
            if donor_fail:
                raise RuntimeError("donor read down")
            return list(donor_rows or [])
        return list(first_rows)

    repo._rows = _stub_rows  # stub the two reads this exercises
    return repo, calls


def _fetch(repo, names):
    import asyncio

    from ai_graph_retrieval.core.access import AccessContext

    ctx = AccessContext(tenant_id="t", principals=("public",), request_id="r")
    return asyncio.run(repo.fetch_people_profiles(ctx, names))


def test_an_identity_stub_renders_the_wikipedia_attributes_under_its_own_name():
    """The arXiv name 'tanvi.ladkani' stays the entity's name (it is what
    the evidence carries) but the profile line — and the citable people doc —
    come from the same human's attribute-bearing Wikipedia node, joined by
    EMAIL: verified live, each connector mints its own user-id space (Atlassian
    account id vs platform uuid), so the ids never match across graphs and the
    email is the shared key. Never joined by name."""
    stub = {"name": "tanvi.ladkani", "graph": "arxiv",
            "person_user_id": "atlassian-77", "person_email": "tanvi@x.com"}
    donor = {"name": "Tara Lane", "graph": "wikipedia",
             "person_user_id": "uuid-77", "person_email": "tanvi@x.com",
             "job_title": "Data Engineer",
             "department": "Data Engineering", "location": "Pune, IN",
             "doc_id": "uuid-77", "doc_title": "Tara Lane",
             "doc_url": "https://p/tanvi", "doc_source_type": "people"}
    repo, calls = _merge_repo([stub], donor_rows=[donor])
    profiles, docs = _fetch(repo, ["tanvi.ladkani"])

    (p,) = profiles
    assert p.name == "tanvi.ladkani"          # the evidence's name, kept
    assert p.profile == ("title: Data Engineer · department: Data Engineering"
                         " · location: Pune, IN")
    assert p.url == "https://p/tanvi"         # the Wikipedia profile page
    assert [d.doc_id for d in docs] == ["uuid-77"]  # citable people doc rides along
    # The donor read casts both keys: the stub's own id AND its email.
    assert calls["donor_params"]["ids"] == ["atlassian-77"]
    assert calls["donor_params"]["emails"] == ["tanvi@x.com"]


def test_no_donor_read_when_the_group_already_carries_attributes():
    """Same human resolved in both graphs under one name: the Wikipedia row is
    already the group's attribute source — the identity read must not run."""
    rows = [
        {"name": "Lena Park", "graph": "wikipedia", "person_user_id": "u-1",
         "job_title": "VP, Marketing"},
        {"name": "Lena Park", "graph": "arxiv", "person_user_id": "u-1"},
    ]
    repo, calls = _merge_repo(rows, donor_rows=[])
    profiles, _docs = _fetch(repo, ["Lena Park"])
    assert "donor_params" not in calls, "no starved group ⇒ no second read"
    # BOTH rows render the attribute-complete line — the stub gained the merge.
    assert [p.profile for p in profiles] == ["title: VP, Marketing", "title: VP, Marketing"]


def test_rows_without_identity_keys_behave_as_before():
    rows = [{"name": "Lena Park", "graph": "wikipedia", "job_title": "VP, Marketing"},
            {"name": "Bare Stub", "graph": "arxiv"}]
    repo, calls = _merge_repo(rows)
    profiles, _docs = _fetch(repo, ["Lena Park", "Bare Stub"])
    assert "donor_params" not in calls
    assert [p.name for p in profiles] == ["Lena Park"]  # keyless stub stays bare


def test_a_failed_donor_read_costs_the_merge_never_the_profiles():
    stub = {"name": "tanvi.ladkani", "graph": "arxiv", "person_user_id": "u-77"}
    rich = {"name": "Lena Park", "graph": "wikipedia", "person_user_id": "u-1",
            "job_title": "VP, Marketing"}
    repo, _calls = _merge_repo([stub, rich], donor_fail=True)
    profiles, _docs = _fetch(repo, ["tanvi.ladkani", "Lena Park"])
    assert [p.name for p in profiles] == ["Lena Park"]  # stub bare, rest intact


def test_two_humans_behind_one_stub_name_stay_bare():
    """Live case (dev graph): ONE arXiv display name, TWO distinct humans
    (two user ids). Their own lines are identically empty, so a pre-merge
    ambiguity check sees no conflict — but the donors render two DIFFERENT
    people's profiles under that one name. Ambiguity is therefore judged on the
    MERGED lines: the name is left bare rather than gambling on whose job it is."""
    stubs = [
        {"name": "Gary", "graph": "arxiv", "person_user_id": "u-a"},
        {"name": "Gary", "graph": "arxiv", "person_user_id": "u-b"},
    ]
    donors = [
        {"name": "Gary Sims", "graph": "wikipedia", "person_user_id": "u-a",
         "job_title": "SDE II"},
        {"name": "Gary Rowe", "graph": "wikipedia", "person_user_id": "u-b",
         "job_title": "Account Executive"},
    ]
    repo, _calls = _merge_repo(stubs, donor_rows=donors)
    profiles, docs = _fetch(repo, ["Gary"])
    assert profiles == [] and docs == []


def test_donors_never_surface_under_their_own_name():
    """The donor's display name may not be in the evidence at all — it donates
    attributes to the stub's name and is never emitted as its own entity."""
    stub = {"name": "gary", "graph": "arxiv", "person_user_id": "u-9"}
    donor = {"name": "Gary Sims", "graph": "wikipedia",
             "person_user_id": "u-9", "job_title": "SDE II"}
    repo, _calls = _merge_repo([stub], donor_rows=[donor])
    profiles, _docs = _fetch(repo, ["gary"])
    assert [p.name for p in profiles] == ["gary"]
    assert profiles[0].profile == "title: SDE II"


# ── ACL-conditional descriptions (new per-run schema) ────────────────────────
# The plain `description` is ingestion's blend over ALL source docs (no ACL
# tags); `descriptions_json` is the same text per source entry, each ACL-tagged.

def test_description_prefers_the_plain_blend_only_while_acl_is_off():
    import json

    from ai_graph_retrieval.graph.repository import _acl_conditional_description

    dj = json.dumps([
        {"text": "visible sentence.", "acl": ["caud:me"]},
        {"text": "SECRET sentence.", "acl": ["caud:someone-else"]},
    ])
    # ACL off (principals=None): the blend wins; json (unfiltered) is the fallback.
    assert _acl_conditional_description("the blend.", dj, None) == "the blend."
    assert _acl_conditional_description(None, dj, None) == "visible sentence. SECRET sentence."
    # ACL on: ONLY the caller's entries — never the blend, even when present.
    got = _acl_conditional_description("the blend.", dj, ["caud:me"])
    assert got == "visible sentence."
    assert "SECRET" not in got
    # ACL on with no matching entry: fail closed to nothing, not to the blend.
    assert _acl_conditional_description("the blend.", dj, ["caud:stranger"]) is None


# ── post-loop graph-chunk top-up (2026-08-18) ────────────────────────────────
# The agent MAY pull graph chunk text through its tools, but whether it does is
# a per-run tool choice — on Search's short budget the configured allowance
# delivered nothing on most runs. enrich_graph_chunks makes the allowance real:
# one batched supporting-chunks read, drawing from the SAME dedup-then-cap
# budget the tools share, gated on the effective capability (flag AND cap > 0).

class _ChunkRepo(_Repo):
    def __init__(self, quotes=None, fail=False):
        super().__init__()
        self.asked: list[str] | None = None
        self._quotes = quotes or []
        self._chunk_fail = fail

    async def supporting_chunks(self, ctx, names):
        if self._chunk_fail:
            raise RuntimeError("graph down")
        self.asked = list(names)
        return list(self._quotes)


def _quote(cid, text, graph="wikipedia"):
    return ChunkQuote(chunk_id=cid, doc_id="d", doc_title="Doc", text=text, graph=graph)


def _sc_chunks(pool, repo, cap, flag=True):
    sc = _sc(pool, repo, cap=0)  # claim/profile passes off; this pass under test
    sc.profile = SimpleNamespace(claim_enrich_top=0, profile_enrich_top=0,
                                 graph_text_tools=flag, graph_chunk_cap=cap)
    return sc


def test_the_topup_fetches_for_pool_anchors_and_respects_the_cap():
    pool = _team_pool()
    repo = _ChunkRepo([_quote(f"c{i}", f"unique graph text {i}") for i in range(6)])
    asyncio.run(enrich_graph_chunks(_sc_chunks(pool, repo, cap=3)))
    assert repo.asked == ["Report B", "Manager A", "Report C"]  # pool order, cap-bounded
    kept = [q for q in pool.snapshot().quotes if q.graph == "wikipedia"]
    assert len(kept) == 3  # dedup-then-cap: cap bounds what enters the pool


def test_zero_cap_or_disabled_flag_means_no_read_at_all():
    for cap, flag in ((0, True), (5, False)):
        repo = _ChunkRepo([_quote("c1", "text")])
        asyncio.run(enrich_graph_chunks(_sc_chunks(_team_pool(), repo, cap=cap, flag=flag)))
        assert repo.asked is None, (cap, flag)


def test_topup_shares_the_budget_with_tool_fetched_graph_chunks():
    """Agent-fetched graph chunks already in the pool count against the SAME
    cap: the top-up only spends what the tools left."""
    pool = _team_pool()
    pool.add_quotes([_quote("tool1", "tool fetched graph text")])  # graph-sourced
    repo = _ChunkRepo([_quote("g1", "new A"), _quote("g2", "new B")])
    asyncio.run(enrich_graph_chunks(_sc_chunks(pool, repo, cap=2)))
    graph_quotes = [q for q in pool.snapshot().quotes if q.graph == "wikipedia"]
    assert len(graph_quotes) == 2  # tool1 + exactly one more


def test_a_failed_chunk_read_never_costs_the_answer():
    pool = _team_pool()
    before = pool.size()
    asyncio.run(enrich_graph_chunks(_sc_chunks(pool, _ChunkRepo(fail=True), cap=5)))
    assert pool.size() == before


def test_duplicate_texts_never_enter_the_pool_or_spend_budget():
    """A graph chunk whose text the pool already holds is dropped BEFORE the
    count — so duplicates cost nothing and unique text still fills the cap."""
    pool = _team_pool()
    pool.add_quotes([ChunkQuote(chunk_id="o1", doc_id="d", doc_title="Doc",
                                text="Shared   text\nfrom elsewhere", graph="arxiv")])
    repo = _ChunkRepo([
        _quote("g1", "shared text from elsewhere"),   # normalized duplicate → dropped
        _quote("g2", "genuinely new graph text A"),
        _quote("g3", "genuinely new graph text B"),
    ])
    # cap 3 = the pooled arxiv chunk + the two unique wikipedia ones
    asyncio.run(enrich_graph_chunks(_sc_chunks(pool, repo, cap=3)))
    graph_quotes = [q for q in pool.snapshot().quotes if q.graph == "wikipedia"]
    assert [q.chunk_id for q in graph_quotes] == ["g2", "g3"]
