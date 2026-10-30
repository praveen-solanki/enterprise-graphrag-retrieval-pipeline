"""Who is still active — one reading of person status for all of retrieval.

Only used when ``people_graph_enabled`` is on: it reads the person-status
properties an HR-style connector writes, which the public connectors do not.

WHY THIS EXISTS. People who have left were retrieved like
anyone else and reached the answer with a flag on them (``FORMER EMPLOYEE`` on
the profile line, ``[INACTIVE …]`` beside a relationship), leaving the answer
model to decide whether a departed colleague belongs in a current roster. That
decision is not the model's to make and it was not reliably made: a departed
person had already spent a row of every ranked read's ``LIMIT``, a slot of the
digest's entity cap and a place in the answer's people table.

So the exclusion is a RETRIEVAL boundary, in two halves that share this module's
semantics:

  * IN CYPHER — every person-bearing read carries the ``<<current_person:…>>``
    predicate (graph/cypher.py), bound per request from
    ``$include_inactive_people``. A person who has left is never a row, so they
    cannot consume a candidate slot an active colleague would have had.
  * OVER THE POOL — ``sweep_inactive_people`` below, run ONCE per request at the
    RETRIEVAL BOUNDARY (``pipeline/execution.EvidenceCollector``), after the
    lane has filled the pool and before anything is ranked. Cypher can only
    judge ONE graph's node: the same human can be active in one source and
    deactivated in another, and the cross-graph identity read is the only
    thing that can see both. It is also the only pass that can reach a name
    arriving as free text.

    At the COLLECTOR and deliberately not at the tail of a lane. Whether the
    lane ran to completion is not a property the rule may depend on: a lane
    cancelled by the retrieval budget (whose partial evidence still reaches
    synthesis under ``partial_on_timeout``) is a departed person reaching the
    answer if the rule is the lane's last line.

HISTORICAL QUESTIONS KEEP THEIR PEOPLE. ``StrategyContext.historical`` is the
classifier's own reading of the question (a resolved time window, or the
TEMPORAL intent — ``RetrievalPlan.historical``); no query string is inspected
here. On such a question the evidence is kept and MARKED instead of dropped, so
a departed colleague can be named as part of the past without being presented
as a current member.
"""
from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import structlog

from ai_graph_retrieval.observability.tracing import set_span_attributes, stage_span

logger = structlog.get_logger(__name__)

# The ONE ``person_employment_status`` value that means "still here". An
# HR-style connector writes Active/Inactive on person nodes; any OTHER non-empty
# value a source system may write (Deactivated, Terminated, Departed) is read
# the same way rather than enumerated, so a new vocabulary word from a connector
# is excluded on arrival instead of silently passing as current.
ACTIVE = "active"

# What a marked-up person carries once retrieval has decided. Same string the
# digest already renders for a former employee, so the answer model sees one
# vocabulary whichever evidence type carried the person.
INACTIVE = "Inactive"

# What an entity row calls a human: the graph's node label and its plural.
# Matched casefolded, so neither spelling has to be repeated.
PERSON_TYPES = frozenset({"person", "people"})
_PERSON_TYPES = PERSON_TYPES

# The one phrasing a profile line names a person's MANAGER with — written by
# graph/repository._org_parts and read back by retrieval/person_identity.
# ``person_manager`` is a string property on the REPORT's node, so the manager
# routinely appears in the pool ONLY inside their report's profile prose: no
# entity row, no edge, nothing the identity read would otherwise be asked about.
# Shared here rather than spelled twice because the reader is only correct for
# as long as it matches the writer exactly.
MANAGER_SEGMENT = "reports to "


def is_inactive(status: str | None, departed_on: str | None = None) -> bool:
    """Whether these two person-node fields record someone who has left.

    MISSING IS NOT INACTIVE, and absence has an established meaning here that
    makes reading it any other way destructive: some connectors write person
    nodes as identity stubs carrying no attributes at all, and ``person_departed_on`` is
    written ONLY for people who have left — its NULL says "no departure
    recorded", never "this person is active" (graph/repository.py says the same
    of it). Treating either absence as a departure would delete every such
    person from every answer.
    """
    if str(departed_on or "").strip():
        return True
    value = str(status or "").strip()
    return bool(value) and value.casefold() != ACTIVE


def identity_key(row: Mapping) -> str | None:
    """The person's cross-graph identity — NEVER the display name, which is
    exactly what differs across graphs for the same human.

    EMAIL first, user id as the fallback: each connector mints its own id
    space, so the SAME human's ids never match across graphs and email is the
    only shared key. The uid fallback still groups a connector's own rows. Keys
    are namespaced so an id can never collide with an email string."""
    if email := str(row.get("person_email") or "").strip():
        return f"email:{email.casefold()}"
    if uid := str(row.get("person_user_id") or "").strip():
        return f"id:{uid}"
    return None


def departed_person_names(rows: Iterable[Mapping]) -> frozenset[str]:
    """Every display LABEL belonging to a human some graph records as departed.

    Cross-graph by construction, and that is the whole point: one graph marking
    a person Active must not make them current when another holds a departed
    representation of the same human. Rows are grouped on the exact identity key
    (``identity_key``) — no name is ever compared to another name — and one
    departed node condemns every label its group carries, in every graph.

    A row with no identity key can only speak for itself: it contributes its own
    name when that row says departed, and nothing otherwise.
    """
    labels: dict[str, set[str]] = defaultdict(set)
    departed_keys: set[str] = set()
    keyless: set[str] = set()
    for r in rows:
        name = str(r.get("name") or "").strip()
        if not name:
            continue
        gone = is_inactive(r.get("employment_status"), r.get("departed_on"))
        key = identity_key(r)
        if key is None:
            if gone:
                keyless.add(name)
            continue
        labels[key].add(name)
        if gone:
            departed_keys.add(key)
    return frozenset(keyless.union(*(labels[k] for k in departed_keys)) if departed_keys
                     else keyless)


@dataclass(frozen=True)
class PersonIdentities:
    """What ONE identity read resolved about the humans behind the pool's names.

    Two answers off the same two batched reads, because they need the same rows:
    which labels are the same human (``renames``) and which of those humans no
    graph still counts as employed (``departed``, holding EVERY label they carry).
    """

    renames: dict[str, str] = field(default_factory=dict)
    departed: frozenset[str] = frozenset()
    # Whether the graphs actually ANSWERED the employment question, as opposed to
    # answering "nobody has left". The default is False so the only way to claim
    # an answer is to have one: every skip and failure path builds the bare
    # PersonIdentities() and is therefore unresolved by construction.
    #
    # The distinction exists because "no departures found" and "we never got to
    # ask" are the same empty ``departed`` set, and they mean opposite things for
    # evidence no Cypher gate ever saw — see ``exclude_inactive_people``.
    resolved: bool = False

    # The names the read was actually asked about, casefolded. A person outside
    # this set was not judged — the cap truncated them, or they were never a
    # candidate — so a resolved read still says nothing about THEM.
    asked: frozenset[str] = frozenset()

    def judged(self, name: str) -> bool:
        """Whether this resolution carries a real employment verdict for ``name``.

        RENAME-AWARE, because the caller asks AFTER the pool was canonicalized:
        ``asked`` holds the labels the read was given, but the read answers with
        the CANONICAL name, and ``rename_entities`` has already rewritten the
        pool to it by the time the sweep runs. A rename's output is judged by
        construction — it is the read's own answer for an asked name — and
        without the second test an ACTIVE person the read positively resolved
        was dropped as "unverified" the moment their API label differed from
        their canonical one. The departed side never leaned on this method:
        ``departed_labels`` carries both spellings, so a departed person is
        dropped whatever ``judged`` says.
        """
        if not self.resolved:
            return False
        key = name.strip().casefold()
        return key in self.asked or any(
            key == canonical.strip().casefold()
            for canonical in self.renames.values()
        )

    def departed_labels(self) -> set[str]:
        """``departed`` under BOTH spellings — the label each graph gave the human
        AND its post-rename canonical form.

        The renames are applied to the pool first, so the structured rows carry
        the canonical label — but ``rename_entities`` rewrites entities, facts,
        claims and mention lists, never chunk prose or document titles. A chunk
        that says "jane.doe wrote this" still spells the name the source
        system's way, so dropping only the canonical form left exactly the free-text
        copies the sweep exists to reach.
        """
        return {label for n in self.departed
                for label in (n, self.renames.get(n, n))}


def exclude_inactive_people(sc, resolution: PersonIdentities) -> None:
    """Take the people who have left out of the evidence — before it is ranked.

    Runs after ``canonicalize_person_names`` (whose read produced ``resolution``)
    and before the quality gate, so nothing a departed person contributed reaches
    trust ranking, the digest's caps or the answer. Nothing here reads the
    question or any name: the set comes from the graphs' own status fields.

    On a historical question the same set is MARKED instead, so the evidence
    survives and the digest states that the person is no longer active
    wherever it names them.
    """
    with stage_span("stage.inactive_people") as span:
        if not sc.settings.inactive_people_filter_enabled:
            set_span_attributes(span, skipped="disabled")
            return
        names = resolution.departed_labels()
        set_span_attributes(span, departed=len(names), historical=sc.historical)
        if not names:
            return
        if sc.historical:
            flagged = sc.evidence.flag_people(names)
            set_span_attributes(span, flagged=flagged)
            logger.info("🕰️ inactive_people_kept " + json.dumps(
                {"departed": len(names), "flagged": flagged, "reason": "historical"},
                ensure_ascii=False))
            return
        removed = sc.evidence.drop_people(names)
        set_span_attributes(span, **{f"removed_{k}": v for k, v in removed.items()})
        logger.info("🚫 inactive_people_excluded " + json.dumps(
            {"departed": len(names), "removed": removed,
             # A sample, not the set: the full list goes nowhere near a log line
             # that a support ticket might quote.
             "sample": sorted(names)[:5]},
            ensure_ascii=False))


async def sweep_inactive_people(sc) -> None:
    """The boundary pass: resolve who the pool's names are, then take the leavers out.

    The ONE call site is ``pipeline/execution.EvidenceCollector`` — see this
    module's header for why the rule lives there and not at the tail of a lane.

    It resolves the identities itself rather than reusing a lane's, which is what
    makes it lane-independent: the same batched read, idempotent with the rename
    an agent lane already applied (a second run maps canonical labels to
    themselves), and empty on every skip and failure path — so a lane that
    resolved nothing, or none at all, is not a hole.
    """
    from ai_graph_retrieval.retrieval.person_identity import canonicalize_person_names

    exclude_inactive_people(sc, await canonicalize_person_names(sc))
