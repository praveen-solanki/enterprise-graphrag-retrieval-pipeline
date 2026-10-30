"""The answer's own subgraph — what a client draws as "the graph behind this answer".

``evidence`` is everything retrieval touched: seed candidates, every neighbourhood
the agent expanded, every post-loop top-up. That is the right thing to SHIP (it is
the audit trail) and the wrong thing to DRAW: on a live people question the pool
held 39 entities and 182 relationships, of which the answer named seven entities
and used none of the relationships, and a client that centred its canvas on the
best-connected node showed the company's investor boilerplate — or, on another
run, a penetration-testing engagement — beside an answer about interns.

WHAT THE GRAPH IS FOR. A reader inspects it to see how the answer was reached:
which entities the evidence is about, which relationships connected them, and
where each came from. So every node and edge here has to be TRACEABLE to the
answer or to the evidence the answer cites, and each says why it is there.

THREE SOURCES OF TRUTH, all of them provenance the service already holds —
nothing is inferred from the question, ranked, or matched fuzzily:

  * THE ANSWER TEXT — a node is drawn when the answer names it, whole-word
    (the same dictionary lookup the citation resolver applies to titles).
  * THE CITED DOCUMENTS' OWN PASSAGES (``CitedSubgraph``, read back from the
    graph after synthesis) — the entities those passages MENTION and the
    relationships EXTRACTED from them, so an answer maps onto graph nodes
    through the documents it cites — e.g. a cited article carrying
    `Ada Lovelace -COLLABORATED_WITH-> Charles Babbage` puts that edge on the
    answer's graph.
  * THE RESEARCHER'S OWN PATH — pool facts carry the read that found them
    (``RelationFact.origin``) and the passages they were extracted from
    (``chunk_ids``). An edge the researcher followed (expand / paths /
    authorship), one recorded by the people directory (structural), or one a
    cited passage states is drawn; an edge the post-loop fill-in merely found
    to EXIST between two pooled entities (``induced``) is not — that fill-in can
    put dozens of parallel edges under an answer that used none of them.

An edge needs BOTH endpoints drawn, with one exception that is the point of a
graph: when two named entities are connected only THROUGH a third pooled entity
(A → MANAGES → B → WORKS_AT → C with B unnamed), B is drawn as the bridge, so the
traversal the answer rests on is visible rather than two disconnected dots.

Every node carries ``role`` and every edge ``why`` — one plain sentence for the
client to show on hover — plus ``refs``, the citation numbers ([n]) that ground it.

Deterministic and query-agnostic: nothing here reads the question, ranks
anything, or drops evidence — ``evidence`` ships unchanged beside it.
"""
from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Sequence

from ai_graph_retrieval.core.models import (
    CitedSubgraph,
    Evidence,
    RelationFact,
    SourceDocument,
)

# Below this an entity name is not a name a whole-word match can stand on.
_MIN_NAME_CHARS = 2

# Reads whose facts are the researcher's own path through the graph. An edge from
# one of these is drawn between two drawn nodes because the run TRAVERSED it.
_TRAVERSED = frozenset({"expand", "paths", "authorship", "traverse", "contributors"})
# The post-loop fill-in of edges among entities already pooled: they exist in the
# graph, but nothing walked them and — unless a cited passage states them — the
# answer did not use them.
_INDUCED = "induced"


def _named_in(answer: str, names: Iterable[str]) -> set[str]:
    """The subset of ``names`` the answer states, whole-word, case-sensitive.

    Whole-word so "UX" does not match "Linux" and "IN" does not match "Interns";
    case-sensitive because the names are the graph's own labels and the model is
    instructed to copy names as given. One compiled alternation, longest first,
    so at any position the longest name wins: "Enterprise Search" is matched as
    itself, never as a hit for the shorter "Search".
    """
    wanted = sorted({n for n in names if n and len(n.strip()) >= _MIN_NAME_CHARS},
                    key=len, reverse=True)
    if not wanted or not answer:
        return set()
    pattern = re.compile(
        r"(?<![\w])(?:" + "|".join(re.escape(n) for n in wanted) + r")(?![\w])"
    )
    return set(pattern.findall(answer))


def _refs(citations: Sequence[SourceDocument]) -> dict[str, int]:
    """doc_id → citation number ([n] is ``citations[n-1]``, the resolver's order)."""
    return {s.doc_id: i for i, s in enumerate(citations, start=1) if s.doc_id}


def _fmt_refs(numbers: Iterable[int]) -> str:
    return ", ".join(f"[{n}]" for n in sorted(set(numbers)))


def _ends(f: RelationFact) -> tuple[str, str]:
    """The edge as it should be DRAWN: ``direction`` is relative to ``source``, so
    an incoming edge reads target → source (a report REPORTS_TO its manager)."""
    return (f.target, f.source) if f.direction == "incoming" else (f.source, f.target)


def build_answer_graph(
    answer: str,
    evidence: Evidence,
    citations: Sequence[SourceDocument] = (),
    cited: CitedSubgraph | None = None,
) -> dict:
    """``{"nodes": [...], "edges": [...]}`` — the entities the answer names, the
    relationships among them that the run traversed or a cited passage states,
    and the bridges that connect them. Node ids are entity names, like the
    trace's, so a client can key the two together. Empty when the answer names
    nothing the evidence holds (a conversational turn).
    """
    cited = cited or CitedSubgraph()
    refs = _refs(citations)
    cited_chunks = set(cited.chunk_ids)

    by_name = {e.name: e for e in evidence.entities}
    for e in cited.entities:
        # The pool's copy wins (it may carry a profile); a cited-only entity
        # still contributes its type and description.
        by_name.setdefault(e.name, e)
    candidates = set(by_name)
    facts = [*evidence.facts, *cited.facts]
    for f in facts:
        candidates.update(n for n in (f.source, f.target) if n)
    # Claim SUBJECTS are nodes; a claim's object is a value ("Kanpur", "21
    # September 2026") and becomes an edge target only when it is already a
    # node in its own right — never a node on the strength of the claim alone.
    candidates.update(c.subject for c in evidence.claims if c.subject)
    named = _named_in(answer, candidates)
    if not named:
        return {"nodes": [], "edges": []}

    # ── which edges count, and why ───────────────────────────────────────────
    def _cited_refs_for(f: RelationFact) -> set[int]:
        key = f"{f.source}|{f.relationship}|{f.target}"
        docs = set(cited.fact_docs.get(key, ()))
        docs.update(cited.chunk_docs.get(c, "") for c in f.chunk_ids if c in cited_chunks)
        return {refs[d] for d in docs if d in refs}

    def _why(f: RelationFact) -> tuple[str, set[int]] | None:
        """The sentence that justifies drawing ``f``, or None when nothing does."""
        cited_in = _cited_refs_for(f)
        if cited_in:
            return f"Stated in the cited source {_fmt_refs(cited_in)}", cited_in
        if f.origin == "cited":
            # A cited-passage edge whose document is not among the numbered
            # citations (the resolver dropped an unreferenced document).
            return "Stated in a passage of a source the answer drew on", set()
        if f.structural:
            return "Recorded relationship (people directory / structured data)", set()
        if f.origin in _TRAVERSED:
            return "Followed by the researcher while exploring the knowledge graph", set()
        if f.origin == _INDUCED:
            return None
        # A read that predates provenance stamping: keep it, and say so honestly.
        return "Found in the knowledge graph while researching the question", set()

    # Directed adjacency over the drawable facts, for bridge discovery.
    drawable: list[tuple[RelationFact, str, set[int]]] = []
    into: dict[str, set[str]] = defaultdict(set)   # node → named nodes with an edge INTO it
    out_of: dict[str, set[str]] = defaultdict(set)  # node → named nodes it has an edge TO
    endpoint_types: dict[str, str] = {}
    for f in facts:
        if not f.source or not f.target or f.source == f.target:
            continue
        if f.target_type:
            endpoint_types.setdefault(f.target, f.target_type)
        judged = _why(f)
        if judged is None:
            continue
        drawable.append((f, *judged))
        # Only the researcher's own path can make a bridge: an edge it followed
        # or one the directory records. A cited PASSAGE also states edges through
        # things the answer never mentions (an intern's "superpower", a shared
        # hobby), and those are the passage's detail, not a step the answer took.
        if not (f.origin in _TRAVERSED or f.structural):
            continue
        a, b = _ends(f)
        if a in named:
            into[b].add(a)
        if b in named:
            out_of[a].add(b)

    # ── nodes: what the answer names, plus the bridges between them ──────────
    drawn: dict[str, str] = {}  # name → role
    for name in named:
        if name in by_name and cited.entity_docs.get(name):
            grounded = {refs[d] for d in cited.entity_docs[name] if d in refs}
            where = (f" · mentioned in the cited source {_fmt_refs(grounded)}"
                     if grounded else " · mentioned in a cited source")
        elif name in by_name and by_name[name].via:
            where = " · reached while exploring the knowledge graph"
        else:
            where = ""
        drawn[name] = f"Named in the answer{where}"
    # Bridges: an unnamed entity the evidence passes THROUGH — an edge in from
    # one named node and an edge out to another (A → B → C). A node two named
    # entities merely share an edge INTO (both BASED_IN the same region) or OUT
    # OF is not a step on any path between them, and drawing it turned a roster
    # into a hairball of shared locations (measured: five such nodes on one
    # eleven-report answer).
    for name in set(into) & set(out_of):
        if name in drawn:
            continue
        upstream = sorted(into[name] - {name})
        downstream = sorted(out_of[name] - {name} - set(upstream[:1]))
        if upstream and downstream:
            drawn[name] = f"Connects {upstream[0]} and {downstream[0]} in the evidence"

    nodes: dict[str, dict] = {}
    for name in sorted(drawn):
        e = by_name.get(name)
        node: dict = {"id": name, "label": name, "kind": "entity", "role": drawn[name],
                      "named": name in named}
        if e is not None:
            if e.type:
                node["type"] = e.type
            if e.description:
                node["description"] = e.description
            if e.profile:
                node["profile"] = e.profile
        if "type" not in node and name in endpoint_types:
            # A fact's target_type is the only type most traversal targets carry.
            node["type"] = endpoint_types[name]
        grounded = {refs[d] for d in cited.entity_docs.get(name, ()) if d in refs}
        if grounded:
            node["refs"] = sorted(grounded)
        nodes[name] = node

    # ── edges: among drawn nodes; a bridge only carries edges to named nodes ──
    edges: dict[str, dict] = {}
    # One edge per (source, RELATIONSHIP, target) however it was spelled: the
    # same statement reaches the pool as a typed edge ("REPORTS_TO") and as an
    # extracted claim ("reports_to"), and drawing both doubled every roster edge.
    seen_edges: dict[tuple[str, str, str], str] = {}
    for f, why, cited_in in drawable:
        a, b = _ends(f)
        if a not in nodes or b not in nodes:
            continue
        if a not in named and b not in named:
            continue
        edge_id = f"{a}|{f.relationship}|{b}"
        folded = (a, f.relationship.casefold(), b)
        edge_id = seen_edges.setdefault(folded, edge_id)
        edge = edges.get(edge_id)
        if edge is None:
            edge = edges[edge_id] = {
                "id": edge_id, "source": a, "target": b, "label": f.relationship,
                "why": why,
                **({"description": f.description} if f.description else {}),
            }
            if cited_in:
                edge["refs"] = sorted(cited_in)
        elif cited_in:
            # The same edge reached twice: a cited statement outranks a traversal
            # as the reason it is shown, and its refs accumulate.
            edge["why"] = why
            edge["refs"] = sorted(set(edge.get("refs", ())) | cited_in)
    for c in evidence.claims:
        if c.subject in nodes and c.object in nodes and c.subject != c.object:
            if c.subject not in named and c.object not in named:
                continue
            folded = (c.subject, c.predicate.casefold(), c.object)
            if folded in seen_edges:
                continue  # the typed edge already says this
            edge_id = seen_edges[folded] = f"{c.subject}|{c.predicate}|{c.object}"
            cited_in = {refs[cited.chunk_docs[cid]] for cid in c.supporting_chunk_ids
                        if cid in cited.chunk_docs and cited.chunk_docs[cid] in refs}
            edges.setdefault(edge_id, {
                "id": edge_id, "source": c.subject, "target": c.object,
                "label": c.predicate,
                "why": (f"Stated in the cited source {_fmt_refs(cited_in)}" if cited_in
                        else "Recorded claim about these entities"),
                **({"refs": sorted(cited_in)} if cited_in else {}),
            })
    # Two prunes, both about what a node can point to as its grounding:
    #  * a bridge that ended up with fewer than two edges is not a bridge after
    #    all (its second link fell outside the drawable set);
    #  * a named node with NO edge and NO cited source is a word the answer used
    #    that happens to be a pool entity's label — "Manager" inside "Product
    #    Manager", "Product", "AI" (measured: eleven such isolates on one roster
    #    answer). Nothing on the canvas would connect it to anything, and no
    #    citation vouches for it, so it is dropped; its `evidence` entry stays.
    degree: dict[str, int] = defaultdict(int)
    for e in edges.values():
        degree[e["source"]] += 1
        degree[e["target"]] += 1
    for name in [n for n, node in nodes.items()
                 if (not node["named"] and degree[n] < 2)
                 or (node["named"] and degree[n] == 0 and not node.get("refs"))]:
        del nodes[name]
        for eid in [eid for eid, e in edges.items() if name in (e["source"], e["target"])]:
            del edges[eid]

    _add_provenance(nodes, edges, evidence, citations, cited, refs)
    return {"nodes": list(nodes.values()), "edges": list(edges.values())}


def _clip(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _add_provenance(
    nodes: dict[str, dict], edges: dict[str, dict], evidence: Evidence,
    citations: Sequence[SourceDocument], cited: CitedSubgraph, refs: dict[str, int],
) -> None:
    """The layer beneath the entities: the cited DOCUMENTS, the PASSAGES of theirs
    the answer's entities were read from, and the CLAIMS those passages support.

    A reader asked "where did this come from?" should be able to walk from an
    entity to the passage that mentions it to the document it was cited from —
    and a cited entity that has no relationship to draw (a feature the release
    notes list) hangs off its passage instead of standing alone. Every node here
    is anchored on a numbered citation; nothing is added for a document the
    answer did not cite or a passage that mentions none of its entities.
    """
    doc_nodes: dict[str, str] = {}  # doc_id → node id
    for s in citations:
        if not s.doc_id or s.doc_id not in refs:
            continue
        n = refs[s.doc_id]
        node_id = doc_nodes[s.doc_id] = f"doc:{s.doc_id}"
        nodes.setdefault(node_id, {
            "id": node_id, "label": s.title or s.doc_id, "kind": "document",
            "type": "DOCUMENT", "named": False, "refs": [n],
            "role": f"Cited source [{n}]" + (f" · {s.date}" if s.date else ""),
            **({"url": s.url} if s.url else {}),
        })

    entity_ids = {n for n, node in nodes.items() if node.get("kind") == "entity"}
    chunk_entities: dict[str, list[str]] = defaultdict(list)
    for name, cids in cited.entity_chunks.items():
        if name in entity_ids:
            for cid in cids:
                chunk_entities[cid].append(name)

    def _chunk_node(q) -> str | None:
        doc_id = q.doc_id or cited.chunk_docs.get(q.chunk_id, "")
        if doc_id not in doc_nodes:
            return None
        n = refs[doc_id]
        node_id = f"chunk:{q.chunk_id}"
        if node_id not in nodes:
            nodes[node_id] = {
                "id": node_id, "label": _clip(q.text, 40) or f"passage of [{n}]",
                "kind": "chunk", "type": "CHUNK", "named": False, "refs": [n],
                "description": _clip(q.text, 240),
                "role": f"Passage of the cited source [{n}] the answer drew on",
            }
            eid = f"{doc_nodes[doc_id]}|HAS_CHUNK|{node_id}"
            edges[eid] = {"id": eid, "source": doc_nodes[doc_id], "target": node_id,
                          "label": "HAS_CHUNK", "why": f"A passage of the cited source [{n}]",
                          "refs": [n]}
        return node_id

    for q in cited.chunks:
        mentioned = sorted(set(chunk_entities.get(q.chunk_id, ())))
        if not mentioned:
            continue
        chunk_id = _chunk_node(q)
        if chunk_id is None:
            continue
        n = nodes[chunk_id]["refs"][0]
        for name in mentioned:
            eid = f"{chunk_id}|MENTIONS|{name}"
            edges.setdefault(eid, {
                "id": eid, "source": chunk_id, "target": name, "label": "MENTIONS",
                "why": f"This passage of the cited source [{n}] mentions {name}",
                "refs": [n],
            })

    by_chunk = {q.chunk_id: q for q in cited.chunks}
    for c in evidence.claims:
        if c.subject not in entity_ids or c.object in entity_ids:
            continue  # an entity→entity claim is already an edge
        support = [cid for cid in c.supporting_chunk_ids if cid in by_chunk]
        if not support:
            continue  # only claims a cited passage supports
        claim_id = f"claim:{c.subject}|{c.predicate}|{c.object}"
        grounded = sorted({refs[cited.chunk_docs[cid]] for cid in support
                           if cited.chunk_docs.get(cid) in refs})
        nodes.setdefault(claim_id, {
            "id": claim_id, "label": _clip(f"{c.subject} {c.predicate} {c.object}", 60),
            "kind": "claim", "type": "CLAIM", "named": False,
            "role": f"Recorded claim about {c.subject}, stated in the cited source "
                    f"{_fmt_refs(grounded)}" if grounded else
                    f"Recorded claim about {c.subject}",
            **({"refs": grounded} if grounded else {}),
            **({"description": f"Valid from {c.valid_from}"} if c.valid_from else {}),
        })
        eid = f"{c.subject}|ASSERTS|{claim_id}"
        edges.setdefault(eid, {"id": eid, "source": c.subject, "target": claim_id,
                               "label": "ASSERTS", "why": f"A recorded claim about {c.subject}",
                               **({"refs": grounded} if grounded else {})})
        for cid in support:
            chunk_id = _chunk_node(by_chunk[cid])
            if chunk_id is None:
                continue
            eid = f"{claim_id}|SUPPORTED_BY|{chunk_id}"
            edges.setdefault(eid, {
                "id": eid, "source": claim_id, "target": chunk_id, "label": "SUPPORTED_BY",
                "why": "The passage this claim was extracted from",
                "refs": nodes[chunk_id]["refs"],
            })
