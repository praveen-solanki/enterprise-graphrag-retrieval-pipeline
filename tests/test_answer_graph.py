"""The answer's own subgraph (synthesis/answer_graph.py) — what a client draws.

`evidence` is everything retrieval touched; a canvas centred on its best-connected
node showed investor boilerplate (or a penetration-testing engagement) beside an
answer about interns. The payload carries `answer_graph`: the entities the answer
names, the relationships among them that the run traversed or a cited passage
states, and the bridges that connect them — each saying why it is there.

    uv run pytest tests/test_answer_graph.py -v
"""
from __future__ import annotations

from ai_graph_retrieval.core.models import (
    CitedSubgraph,
    ClaimFact,
    EntityHit,
    Evidence,
    RelationFact,
    SourceDocument,
)
from ai_graph_retrieval.synthesis.answer_graph import build_answer_graph


def _ents(g: dict) -> list[str]:
    """Entity node ids — the provenance layer (documents, passages, claims) is
    tested on its own; the entity tests should not have to read past it."""
    return [n["id"] for n in g["nodes"] if n.get("kind") == "entity"]


def _pool() -> Evidence:
    return Evidence(
        entities=[
            EntityHit(name="Nora Quinn", type="Person", description="Intern, Platform Services",
                      via=["lexical"]),
            EntityHit(name="Platform Services", type="ORG_UNIT"),
            EntityHit(name="Acme", type="ORGANIZATION"),
            EntityHit(name="Snowflake", type="ORGANIZATION"),
            EntityHit(name="Penetration Testing Engagement", type="EVENT"),
            EntityHit(name="UX", type="CONCEPT"),
        ],
        facts=[
            RelationFact(source="Nora Quinn", target="Platform Services",
                         relationship="MEMBER_OF", target_type="ORG_UNIT", structural=True),
            RelationFact(source="Nora Quinn", target="Sam Carter", relationship="REPORTS_TO",
                         target_type="Person", description="Reporting line", origin="expand"),
            RelationFact(source="Acme", target="Snowflake", relationship="TRUSTED_BY",
                         origin="expand"),
            RelationFact(source="Penetration Testing Engagement", target="21 September 2026",
                         relationship="has report delivery date", origin="expand"),
        ],
        claims=[
            ClaimFact(subject="Nora Quinn", predicate="based in", object="Kanpur"),
            ClaimFact(subject="Sam Carter", predicate="manages", object="Nora Quinn"),
        ],
    )


ANSWER = ("**Nora Quinn** (Platform Services) joined in 2025 and reports to "
          "**Sam Carter**. Acme India hosted the cohort in Kanpur.")


def test_only_named_entities_and_the_relationships_among_them():
    g = build_answer_graph(ANSWER, _pool())
    assert _ents(g) == ["Nora Quinn", "Platform Services", "Sam Carter"]
    assert {e["label"] for e in g["edges"]} == {"MEMBER_OF", "REPORTS_TO", "manages"}
    # Boilerplate the answer never mentions stays in `evidence`, not here.
    ids = set(_ents(g))
    assert "Snowflake" not in ids and "Penetration Testing Engagement" not in ids
    assert "21 September 2026" not in ids
    # An edge to something the answer does not name is not an answer edge.
    assert not any(e["target"] == "Snowflake" for e in g["edges"])


def test_a_named_isolate_needs_a_citation_to_stay():
    """"Acme" is named ("Acme India") but nothing drawn connects it and no
    cited source mentions it: a label the answer happened to use, not evidence
    the reader can follow anywhere. With a citation behind it, it stays."""
    assert "Acme" not in set(_ents(build_answer_graph(ANSWER, _pool())))
    cited = CitedSubgraph(entities=[EntityHit(name="Acme", type="ORGANIZATION")],
                          entity_docs={"Acme": ["d1"]})
    g = build_answer_graph(ANSWER, _pool(), [SourceDocument(doc_id="d1", title="Post")], cited)
    acme = next(n for n in g["nodes"] if n["id"] == "Acme")
    assert acme["refs"] == [1]
    assert acme["role"] == "Named in the answer · mentioned in the cited source [1]"


def test_a_claim_that_restates_a_typed_edge_is_not_a_second_edge():
    """The same statement reaches the pool as REPORTS_TO (typed edge) and as the
    extracted claim "reports_to"; measured, that doubled every roster edge."""
    ev = Evidence(
        facts=[RelationFact(source="Max Fields", target="Owen Price",
                            relationship="REPORTS_TO", structural=True)],
        claims=[ClaimFact(subject="Max Fields", predicate="reports_to", object="Owen Price"),
                ClaimFact(subject="Max Fields", predicate="promoted_by", object="Owen Price")],
    )
    g = build_answer_graph("Max Fields reports to Owen Price.", ev)
    assert sorted(e["label"] for e in g["edges"]) == ["REPORTS_TO", "promoted_by"]


def test_a_shared_attribute_is_not_a_bridge():
    """Two named people both BASED_IN the same region: the region has two edges
    INTO it and none out — nothing passes through it, so it is not drawn.
    A → B → C (an edge in and an edge out) is."""
    ev = Evidence(
        facts=[RelationFact(source="Adam Shaw", target="IN", relationship="BASED_IN",
                            origin="expand"),
               RelationFact(source="Peter Stone", target="IN", relationship="BASED_IN",
                            origin="expand"),
               RelationFact(source="Adam Shaw", target="Owen Price",
                            relationship="REPORTS_TO", structural=True),
               RelationFact(source="Owen Price", target="Data Science",
                            relationship="MEMBER_OF", structural=True)],
    )
    g = build_answer_graph("Adam Shaw and Peter Stone sit in Data Science.", ev)
    ids = set(_ents(g))
    assert "IN" not in ids
    assert "Owen Price" in ids  # Adam → Owen → Data Science
    # A chain stated only by a cited passage does not make a bridge either.
    passage = Evidence(facts=[
        RelationFact(source="Ben Fisher", target="Instant teleportation",
                     relationship="WOULD_CHOOSE_AS_SUPERPOWER", origin="cited"),
        RelationFact(source="Instant teleportation", target="Chennai, India",
                     relationship="WOULD_ENABLE_COMMUTE_FROM", origin="cited")])
    g = build_answer_graph("Ben Fisher is based in Chennai, India.", passage)
    assert "Instant teleportation" not in set(_ents(g))


def test_nodes_carry_what_the_pool_knows_about_them_and_why_they_are_drawn():
    g = build_answer_graph(ANSWER, _pool())
    by_id = {n["id"]: n for n in g["nodes"]}
    assert by_id["Nora Quinn"]["type"] == "Person"
    assert by_id["Nora Quinn"]["description"] == "Intern, Platform Services"
    assert by_id["Nora Quinn"]["role"].startswith("Named in the answer")
    assert "exploring" in by_id["Nora Quinn"]["role"]  # via=["lexical"]
    assert by_id["Nora Quinn"]["named"] is True
    # A relationship target the pool never held as an entity still gets the
    # fact's target_type, and appears because the answer names it.
    assert by_id["Sam Carter"]["type"] == "Person"
    by_label = {e["label"]: e for e in g["edges"]}
    assert by_label["REPORTS_TO"]["description"] == "Reporting line"
    assert by_label["REPORTS_TO"]["why"].startswith("Followed by the researcher")
    assert by_label["MEMBER_OF"]["why"].startswith("Recorded relationship")
    assert by_label["manages"]["why"] == "Recorded claim about these entities"


def test_matching_is_whole_word_case_sensitive_and_longest_first():
    pool = _pool()
    # "UX" must not match inside "Linux"; "Kanpur" is a claim OBJECT, not an
    # entity node, so a literal never becomes a node on its own.
    g = build_answer_graph("They run Linux in Kanpur.", pool)
    assert g == {"nodes": [], "edges": []}
    cited = CitedSubgraph(entity_docs={"UX": ["d1"]})
    citations = [SourceDocument(doc_id="d1", title="UX notes")]
    g = build_answer_graph("The UX team.", pool, citations, cited)
    assert _ents(g) == ["UX"]
    assert _ents(build_answer_graph("The ux team.", pool, citations, cited)) == []
    # The longest name wins its span: "Enterprise Search" in the answer is not a
    # hit for the pooled product "Search" (measured: that match put a "Search"
    # hub and nine of its edges under an answer about new hires).
    ev = Evidence(entities=[EntityHit(name="Search", type="PRODUCT"),
                            EntityHit(name="Enterprise Search", type="PRODUCT")])
    both = CitedSubgraph(entity_docs={"Search": ["d1"], "Enterprise Search": ["d1"]})
    g = build_answer_graph("Enterprise Search brought in $600K.", ev, citations, both)
    assert _ents(g) == ["Enterprise Search"]


def test_induced_edges_are_not_drawn_unless_a_cited_passage_states_them():
    """The post-loop fill-in returns every edge that EXISTS between two pooled
    entities. Measured live: 23 parallel `ARR → Acme` edges under a revenue
    answer that used none of them. They are evidence, not the answer's path."""
    ev = Evidence(
        entities=[EntityHit(name="ARR", type="METRIC"), EntityHit(name="Acme")],
        facts=[RelationFact(source="ARR", target="Acme", relationship="PRICED_BY",
                            origin="induced", chunk_ids=["c-far-away"]),
               RelationFact(source="ARR", target="Acme", relationship="RECOGNIZES_AS",
                            origin="induced", chunk_ids=["c-cited"])],
    )
    answer = "Acme ended FY26 at $70.3M ARR."
    assert build_answer_graph(answer, ev)["edges"] == []
    cited = CitedSubgraph(chunk_ids=["c-cited"], chunk_docs={"c-cited": "doc-1"})
    g = build_answer_graph(answer, ev, [SourceDocument(doc_id="doc-1", title="PPL SYNC")], cited)
    assert [e["label"] for e in g["edges"]] == ["RECOGNIZES_AS"]
    assert g["edges"][0]["why"] == "Stated in the cited source [1]"
    assert g["edges"][0]["refs"] == [1]


def test_cited_passages_ground_nodes_and_supply_edges_the_pool_never_had():
    """A search-seeded answer is written from passages that never came from the
    graph, so the pool holds none of the people it names. The cited document's
    own chunk MENTIONS them and carries the edges extracted from it — measured
    on the live graph for the "Meet our new intern" post."""
    pool = Evidence()  # nothing the agent explored is about these people
    citations = [SourceDocument(doc_id="post-1", title="Meet our new intern Ben Fisher!"),
                 SourceDocument(doc_id="post-2", title="Say hi to Vera Shore")]
    cited = CitedSubgraph(
        entities=[EntityHit(name="Ben Fisher", type="PERSON"),
                  EntityHit(name="Chennai, India", type="LOCATION"),
                  EntityHit(name="Acme", type="ORGANIZATION"),
                  EntityHit(name="Breaking Bad", type="PRODUCT")],
        facts=[RelationFact(source="Ben Fisher", target="Acme", relationship="INTERN_AT",
                            origin="cited", chunk_ids=["ch-1"]),
               RelationFact(source="Ben Fisher", target="Chennai, India",
                            relationship="LOCATED_IN", origin="cited", chunk_ids=["ch-1"]),
               RelationFact(source="Ben Fisher", target="Breaking Bad",
                            relationship="REWATCHED", origin="cited", chunk_ids=["ch-1"])],
        entity_docs={"Ben Fisher": ["post-1"], "Chennai, India": ["post-1"],
                     "Acme": ["post-1", "post-2"], "Breaking Bad": ["post-1"]},
        fact_docs={"Ben Fisher|INTERN_AT|Acme": ["post-1"],
                   "Ben Fisher|LOCATED_IN|Chennai, India": ["post-1"],
                   "Ben Fisher|REWATCHED|Breaking Bad": ["post-1"]},
        chunk_ids=["ch-1"], chunk_docs={"ch-1": "post-1"},
    )
    answer = ("| Ben Fisher | Intern | Backend | Chennai, India |\n"
              "All of them joined Acme this quarter.")
    g = build_answer_graph(answer, Evidence(), citations, cited)
    ids = _ents(g)
    assert ids == ["Acme", "Ben Fisher", "Chennai, India"]
    assert "Breaking Bad" not in ids  # in the cited passage, not in the answer
    by_id = {n["id"]: n for n in g["nodes"]}
    assert by_id["Ben Fisher"]["role"] == (
        "Named in the answer · mentioned in the cited source [1]")
    assert by_id["Ben Fisher"]["refs"] == [1]
    assert by_id["Acme"]["refs"] == [1, 2]
    assert {e["label"] for e in g["edges"]} == {"INTERN_AT", "LOCATED_IN"}
    assert all(e["why"] == "Stated in the cited source [1]" for e in g["edges"])
    del pool


def test_a_bridge_entity_is_drawn_when_it_connects_two_named_ones():
    """A → MANAGES → B → WORKS_AT → C with B unnamed: the traversal the answer
    rests on is visible, not two disconnected dots."""
    ev = Evidence(
        entities=[EntityHit(name="Jane Doe", type="Person"),
                  EntityHit(name="Leo Grant", type="Person"),
                  EntityHit(name="Acme", type="ORGANIZATION")],
        facts=[RelationFact(source="Jane Doe", target="Leo Grant", relationship="MANAGES",
                            origin="paths", depth=1),
               RelationFact(source="Leo Grant", target="Acme", relationship="WORKS_AT",
                            origin="paths", depth=2),
               # An unnamed entity adjacent to only ONE named node is not a bridge.
               RelationFact(source="Leo Grant", target="Kafka", relationship="USES",
                            origin="paths", depth=2)],
    )
    g = build_answer_graph("Jane Doe's team sits inside Acme.", ev)
    by_id = {n["id"]: n for n in g["nodes"]}
    assert set(by_id) == {"Jane Doe", "Leo Grant", "Acme"}
    assert by_id["Leo Grant"]["named"] is False
    assert by_id["Leo Grant"]["role"] == "Connects Jane Doe and Acme in the evidence"
    assert [(e["source"], e["label"], e["target"]) for e in g["edges"]] == [
        ("Jane Doe", "MANAGES", "Leo Grant"), ("Leo Grant", "WORKS_AT", "Acme"),
    ]


def test_an_incoming_edge_is_drawn_in_its_true_direction():
    """`direction` is relative to the fact's `source`: an incoming REPORTS_TO on a
    manager reads report → manager, and the drawn arrow must say so."""
    ev = Evidence(facts=[RelationFact(source="Sam Carter", target="Nora Quinn",
                                      relationship="REPORTS_TO", direction="incoming",
                                      structural=True)])
    g = build_answer_graph("Nora Quinn reports to Sam Carter.", ev)
    assert [(e["source"], e["target"]) for e in g["edges"]] == [("Nora Quinn", "Sam Carter")]


def test_empty_answer_or_pool_yields_an_empty_graph():
    assert build_answer_graph("", _pool()) == {"nodes": [], "edges": []}
    assert build_answer_graph(ANSWER, Evidence()) == {"nodes": [], "edges": []}


def test_repository_reads_the_cited_documents_own_subgraph():
    """cited_subgraph: the two batched reads → entities with their citing docs,
    facts stamped origin="cited" with the chunks they were extracted from, and the
    chunk → document map the builder uses to say "stated in [n]"."""
    import asyncio
    from types import SimpleNamespace

    from ai_graph_retrieval.graph import cypher
    from ai_graph_retrieval.graph.repository import GraphReadRepository
    from tests.support import make_settings

    calls: list[str] = []

    async def _rows(query, ctx, **params):
        calls.append(query)
        assert params["doc_ids"] == ["post-1"]
        if query is cypher.CITED_MENTIONS:
            return [{"entity_id": "e1", "name": "Ben Fisher", "type": "PERSON",
                     "description": "Intern", "degree": 3, "employment_status": "Active",
                     "doc_ids": ["post-1"], "chunk_ids": ["ch-1"], "graph": "wikipedia"}]
        if query is cypher.CITED_CHUNKS:
            return [{"chunk_id": "ch-1", "doc_id": "post-1", "doc_title": "Meet our new intern",
                     "text": "Say hi to Ben Fisher, who joins the Backend team in Chennai.",
                     "graph": "wikipedia"}]
        return [{"source": "Ben Fisher", "target": "Acme", "target_type": "ORGANIZATION",
                 "relationship": "INTERN_AT", "direction": "outgoing", "support": 1,
                 "chunk_ids": ["ch-1"], "doc_ids": ["post-1"], "score": 0.3,
                 "graph": "wikipedia"}]

    repo = SimpleNamespace(_rows=_rows, _s=make_settings(), _principals=lambda ctx: None,
                           _neighbor_facts=GraphReadRepository._neighbor_facts)
    repo._neighbor_facts = lambda rows, principals: GraphReadRepository._neighbor_facts(
        repo, rows, principals)
    cited = asyncio.run(
        GraphReadRepository.cited_subgraph(repo, object(), ["post-1", "", "post-1"]))
    assert {cypher.CITED_MENTIONS, cypher.CITED_EDGES, cypher.CITED_CHUNKS} == set(calls)
    assert [e.name for e in cited.entities] == ["Ben Fisher"]
    assert cited.entities[0].via == ["cited"] and cited.entities[0].description == "Intern"
    assert cited.entity_docs == {"Ben Fisher": ["post-1"]}
    assert [f.origin for f in cited.facts] == ["cited"]
    assert cited.facts[0].chunk_ids == ["ch-1"]
    assert cited.fact_docs == {"Ben Fisher|INTERN_AT|Acme": ["post-1"]}
    assert cited.chunk_docs == {"ch-1": "post-1"} and cited.chunk_ids == ["ch-1"]
    assert cited.entity_chunks == {"Ben Fisher": ["ch-1"]}
    assert [q.chunk_id for q in cited.chunks] == ["ch-1"] and cited.chunks[0].doc_id == "post-1"
    # Nothing to cite ⇒ no read at all.
    calls.clear()
    assert asyncio.run(GraphReadRepository.cited_subgraph(repo, object(), [])).entities == []
    assert calls == []


def test_the_provenance_layer_hangs_every_cited_entity_off_its_passage_and_document():
    """Documents, passages and claims are nodes too: an entity the release notes
    merely list (no relationship to draw) is not a standalone dot — it hangs off
    the passage that mentions it, which hangs off the cited document."""
    from ai_graph_retrieval.core.models import ChunkQuote

    citations = [SourceDocument(doc_id="rel-1", title="26.08 - All You Need To Know",
                                url="https://x/rel", date="2026-08-10")]
    cited = CitedSubgraph(
        entities=[EntityHit(name="Landing Pages", type="FEATURE"),
                  EntityHit(name="Release 26.08", type="EVENT")],
        entity_docs={"Landing Pages": ["rel-1"], "Release 26.08": ["rel-1"]},
        entity_chunks={"Landing Pages": ["c-7"], "Release 26.08": ["c-7", "c-8"]},
        chunk_ids=["c-7", "c-8", "c-9"],
        chunk_docs={"c-7": "rel-1", "c-8": "rel-1", "c-9": "rel-1"},
        chunks=[ChunkQuote(chunk_id="c-7", doc_id="rel-1", text="Landing Pages ship GA."),
                ChunkQuote(chunk_id="c-8", doc_id="rel-1", text="Release 26.08 went GA on Aug 10."),
                ChunkQuote(chunk_id="c-9", doc_id="rel-1", text="Unrelated footer.")],
    )
    ev = Evidence(claims=[ClaimFact(subject="Release 26.08", predicate="reached GA on",
                                    object="2026-08-10", valid_from="2026-08-10",
                                    supporting_chunk_ids=["c-8"]),
                          ClaimFact(subject="Release 26.08", predicate="ungrounded", object="x")])
    g = build_answer_graph("Release 26.08 shipped Landing Pages.", Evidence(), citations, cited)
    by_id = {n["id"]: n for n in g["nodes"]}
    assert _ents(g) == ["Landing Pages", "Release 26.08"]
    assert by_id["doc:rel-1"]["kind"] == "document" and by_id["doc:rel-1"]["url"] == "https://x/rel"
    assert by_id["chunk:c-7"]["kind"] == "chunk" and by_id["chunk:c-7"]["refs"] == [1]
    assert "chunk:c-9" not in by_id  # mentions none of the answer's entities
    labels = {(e["source"], e["label"], e["target"]) for e in g["edges"]}
    assert ("doc:rel-1", "HAS_CHUNK", "chunk:c-7") in labels
    assert ("chunk:c-7", "MENTIONS", "Landing Pages") in labels
    assert ("chunk:c-8", "MENTIONS", "Release 26.08") in labels
    # Claims: only the passage-supported one, linked to its subject and its passage.
    g = build_answer_graph("Release 26.08 shipped Landing Pages.", ev, citations, cited)
    by_id = {n["id"]: n for n in g["nodes"]}
    claim = "claim:Release 26.08|reached GA on|2026-08-10"
    assert by_id[claim]["kind"] == "claim" and by_id[claim]["refs"] == [1]
    assert "claim:Release 26.08|ungrounded|x" not in by_id
    labels = {(e["source"], e["label"], e["target"]) for e in g["edges"]}
    assert ("Release 26.08", "ASSERTS", claim) in labels
    assert (claim, "SUPPORTED_BY", "chunk:c-8") in labels
    # Every provenance node is anchored on a citation; nothing dangles.
    ids = set(by_id)
    assert all(e["source"] in ids and e["target"] in ids for e in g["edges"])
