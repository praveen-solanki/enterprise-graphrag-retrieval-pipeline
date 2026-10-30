"""One human, one name — across graphs.

Only used when ``people_graph_enabled`` is on.

WHY THIS EXISTS. Each graph holds the same person as its own node under its own
system's label, and every label reached the answer. Two symptoms follow:

  * The answer names people by an account handle ("jane.doe") because that is
    what one source system calls the node.
  * The answer reports two people where there is one ("Jane Doe" and "Jane A.
    Doe"): both are real nodes — one per graph, one email.

WHY IT IS ONE PASS OVER THE POOL, NOT A FIX PER READ. The pool keys entities by
name, and facts, claims and chunk mentions all reference a name. Canonicalising in
some reads and not others would produce a NEW split — a profile emitting "Jane
Doe" while entity resolution emitted "jane.doe" — so the rewrite happens once,
over everything, after every name-producing read has run. ``EvidencePool.
rename_entities`` is that single operation.

WHY EMAIL. It is an exact key that source systems populate consistently. Ids do not work
— each connector mints its own id space, so the same human's ids never match across
graphs. No name is ever compared to another name, here or in the query.

Additive and fail-open: any read failure leaves every name exactly as it was, and
the answer is the one the service would have produced anyway.
"""
from __future__ import annotations

import json

import structlog

from ai_graph_retrieval.observability.tracing import (
    record_io,
    set_span_attributes,
    stage_span,
)
from ai_graph_retrieval.retrieval.person_status import MANAGER_SEGMENT, PersonIdentities

logger = structlog.get_logger(__name__)


def candidate_names(ev) -> list[str]:
    """Every name in the pool that could be a person, in a stable order.

    Deliberately NOT filtered to entities typed Person: a person routinely enters
    the pool only as a relationship endpoint ("<name> AUTHORED <doc>") with no
    entity row of their own, and those endpoints are exactly the labels the
    duplicate-person answer was built from. The read itself does the filtering —
    it matches ``:Person`` nodes, so a document title or a concept simply returns
    no row — which keeps the decision about what is a person in the graph rather
    than in a shape test here.

    EVERY name-bearing field of every evidence kind, for that same reason and for
    a second one: this is the resolution the employment sweep runs on
    (person_status.sweep_inactive_people), and the sweep can only drop a person
    the read was asked about. A representation missing here is not a slower
    answer, it is a departed colleague the sweep is structurally unable to see —
    so the set is kept aligned with what ``EvidencePool.drop_people`` cleans.
    """
    names: dict[str, None] = {}          # ordered set
    for e in ev.entities:
        if e.name:
            names.setdefault(e.name, None)
        # The MANAGER named inside this person's profile line. `person_manager`
        # is a string property on the REPORT's node, so a manager reached this
        # way has no entity row and no edge of their own — the one person-name
        # slot that is structured at the source and prose by the time it is
        # pooled. Read back through the segment its writer used
        # (graph/repository._org_parts), never guessed out of the prose.
        for segment in (e.profile or "").split(" · "):
            if segment.startswith(MANAGER_SEGMENT) and (
                    manager := segment[len(MANAGER_SEGMENT):].strip()):
                names.setdefault(manager, None)
    for f in ev.facts:
        for side in (f.source, f.target):
            if side:
                names.setdefault(side, None)
    # CLAIM ends, for exactly the reason the fact endpoints are here: a claim
    # names its subject and object as bare strings, and a person the run reached
    # only through one ("<name> leads Search Platform", "Search Platform is owned
    # by <name>") has no entity row and no edge. The pool sweep is asked to clean
    # claims, so the resolution it runs on has to have been asked about them —
    # a name the identity read never saw is a name nothing can find departed.
    for c in ev.claims:
        for side in (c.subject, c.object):
            if side:
                names.setdefault(side, None)
    # Document TITLES too, for the same reason and with the same safeguard: a
    # people object (a person record ingested as a document) is titled with the
    # person's name, and it can be the ONLY place that person appears — no entity
    # row, no edge.
    # A title that belongs to no person node simply returns no row.
    for d in ev.sources:
        if d.title:
            names.setdefault(d.title, None)
    # A chunk's MENTION list and its document's title: the names traversal
    # recorded the chunk against. The sweep drops a quote on either, so both have
    # to be resolvable — a graph chunk about a person is routinely the only thing
    # a run holds about them.
    for q in ev.quotes:
        if q.doc_title:
            names.setdefault(q.doc_title, None)
        for mention in q.via:
            if mention:
                names.setdefault(mention, None)
    return list(names)


async def canonicalize_person_names(sc) -> PersonIdentities:
    """Give every human in the pool one name, in place.

    Traced as ``stage.person_identity``: the pass rewrites what the reader sees, so
    which names changed has to be answerable from the trace rather than by
    re-running the request.

    Returns the whole resolution, not just what it applied: the same two reads
    also say which of these humans no graph still counts as employed, and the
    caller hands that to ``person_status.exclude_inactive_people`` rather than
    paying for a second identity read. An empty resolution on every skip and
    failure path, so a caller never has to distinguish them.
    """
    with stage_span("stage.person_identity") as span:
        cap = sc.settings.person_identity_max_names
        if cap <= 0:
            set_span_attributes(span, skipped="disabled")
            return PersonIdentities()
        names = candidate_names(sc.evidence.snapshot())
        set_span_attributes(span, candidates=len(names), cap=cap)
        if not names:
            set_span_attributes(span, skipped="no_names")
            return PersonIdentities()
        # The cap keeps this ONE predictable read on a broad roster question. Names
        # are truncated in pool order (seed relevance, then discovery), so the cap
        # drops the least-established names rather than an arbitrary slice.
        try:
            resolution = await sc.repo.person_identities(sc.ctx, names[:cap])
        except Exception as exc:
            logger.warning("person_identity_failed", error=str(exc)[:200],
                           **sc.ctx.log_fields())
            set_span_attributes(span, skipped="read_failed",
                                store_error=str(exc)[:200])
            return PersonIdentities()
        mapping = resolution.renames
        set_span_attributes(span, departed=len(resolution.departed))
        if not mapping:
            set_span_attributes(span, renamed=0)
            return resolution
        before = len(sc.evidence.entity_names)
        renamed = sc.evidence.rename_entities(mapping)
        after = len(sc.evidence.entity_names)
        logger.info("🧑 person_identity " + json.dumps(
            {"candidates": len(names), "renamed": renamed,
             "entities_before": before, "entities_after": after,
             # A sample, not the mapping: the full rewrite goes on the span.
             "sample": dict(list(mapping.items())[:5])}, ensure_ascii=False))
        set_span_attributes(
            span, renamed=renamed, entities_before=before, entities_after=after,
            # Two entities becoming one is the duplicate-person fix landing; zero
            # collapses with renames > 0 means the labels were only cosmetic.
            entities_collapsed=before - after,
        )
        record_io(span, output=json.dumps(
            {"renamed": renamed, "mapping": mapping}, ensure_ascii=False))
        return resolution
