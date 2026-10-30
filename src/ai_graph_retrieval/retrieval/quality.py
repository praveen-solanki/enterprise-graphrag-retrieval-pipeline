"""Evidence quality — trust-weighted source ranking + contradiction detection.

Both are pure functions over the evidence snapshot (0 LLM, microseconds), run
once between retrieval and synthesis.

Trust ranking is a feature blend led by the engine's own relevance signal
(defaults: relevance 0.40, multisource 0.25, recency 0.15, support 0.12,
authority 0.08 — the ``trust_w_*`` settings): the citations list and the
digest's SOURCES section lead with the documents most worth trusting, not
merely the most-mentioned.

Contradiction detection groups claims by (subject, predicate) and flags
conflicting objects — surfaced to synthesis so disagreements are *presented*,
never silently resolved. Deliberately conservative: substring-related values
("John" vs "John Smith") and DYNAMIC claims with non-overlapping validity
windows (values that legitimately change over time) are not conflicts.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from datetime import date
from typing import TypeVar

from ai_graph_retrieval.core.models import (
    ChunkQuote,
    ClaimFact,
    Contradiction,
    SourceDocument,
)
from ai_graph_retrieval.settings import Settings

_T = TypeVar("_T")

_MAX_CONTRADICTIONS = 5
_AUTHORITY_DEFAULT = 0.5

# source_type → authority prior. Neutral default; extend as source connectors
# with editorial guarantees come online (POC used site_name heuristics).
_AUTHORITY: dict[str, float] = {
    "official": 1.0,
    "policy": 0.9,
    "wiki": 0.6,
}


def rank_quotes(
    quotes: list[ChunkQuote],
    sources: list[SourceDocument],
    settings: Settings,
) -> list[ChunkQuote]:
    """Order excerpts by question relevance, best first.

    Reads the ``relevance`` already on each quote (``retrieval/chunk_relevance``
    stamps it), so nothing is embedded
    or fetched here. Document trust stays in the blend because a cosine knows
    nothing about recency, corroboration or authority; an unscored chunk keeps
    0.0 and is ordered by its document's trust alone.

    Pure and total — returns every quote; the digest's budget decides what drops.
    """
    if not quotes:
        return []
    w_sem, w_doc = settings.quote_w_semantic, settings.quote_w_doc_trust
    # Nothing to rank by. Returning the input untouched matters: the tiebreak
    # below would otherwise sort an all-zero field alphabetically and discard
    # the caller's own document-order ranking.
    if not (w_sem or w_doc) or not any(q.relevance for q in quotes):
        return list(quotes)

    trust_by_key: dict[str, float] = {}
    for s in sources:
        for key in (s.doc_id, s.title):
            if key and key not in trust_by_key:
                trust_by_key[key] = s.trust or 0.0

    def _doc_trust(q: ChunkQuote) -> float:
        for key in (q.doc_id, q.doc_title):
            if key and key in trust_by_key:
                return trust_by_key[key]
        return 0.0

    def _semantic(q: ChunkQuote) -> float:
        return q.relevance or 0.0

    # chunk_id breaks near-ties, so two identical requests order identically.
    return sorted(
        quotes,
        key=lambda q: (-(w_sem * _semantic(q) + w_doc * _doc_trust(q)),
                       q.chunk_id or ""),
    )


def rank_claims_by_recency(claims: list[ClaimFact]) -> list[ClaimFact]:
    """Claims newest first, undated last — one global order across every graph.

    The repository fans out over the graphs and CONCATENATES per-graph lists, each
    already `ORDER BY valid_from DESC` (cypher.FETCH_CLAIMS). The concatenation is
    therefore two descending runs, not one, so a positional slice keeps the first
    graph's tail over the second graph's head. Sorting restores the order the query
    already asked for; `valid_from` is an ISO date and compares across graphs.
    """
    return sorted(claims, key=lambda c: c.valid_from or "", reverse=True)


def order_by_date_desc(rows: Sequence[_T], date_of: Callable[[_T], str | None]) -> list[_T]:
    """Dated rows newest first, undated ones last in their existing order.

    The digest renders a date on FACTS and SOURCES but ordered both by RELEVANCE,
    so the model was handed dated items oldest-first as often as not and wrote
    them out in the order it received them. Selection stays relevance-ranked —
    this reorders only what already survived the cap, which is presentation.

    Undated rows are not sortable against dated ones and are never reordered
    among themselves: a stable sort on a single key keeps their relative order.
    ISO dates compare lexicographically, the same assumption
    ``rank_claims_by_recency`` already makes.
    """
    dated = [r for r in rows if str(date_of(r) or "").strip()]
    undated = [r for r in rows if not str(date_of(r) or "").strip()]
    return sorted(dated, key=lambda r: str(date_of(r)), reverse=True) + undated


def graph_peaks(pairs: Iterable[tuple[str | None, float]]) -> dict[str | None, float]:
    """The largest value each graph contributed — the per-graph normalisation base.

    Absolute per-graph quantities — a Lucene score, a mention count, a degree — are
    functions of the index that produced them: term statistics, corpus size and
    density all differ between graphs. Dividing the merged pool by one global max
    therefore ranks graphs against each other rather than candidates, and the denser
    or better-matching corpus wins every row.

    An untagged row (a single-graph read never carries the tag) shares one
    bucket, which is the previous behaviour exactly.
    """
    peaks: dict[str | None, float] = {}
    for graph, value in pairs:
        peaks[graph] = max(peaks.get(graph, 0.0), value)
    return peaks


def graph_scalers(
    pairs: Iterable[tuple[str | None, float]]
) -> dict[str | None, float]:
    """Damped per-graph divisors: ``sqrt(own peak * global ceiling)``.

    Dividing by the graph's own peak alone makes every graph's best row 1.0,
    so a weak corpus's best reads as strong as a strong corpus's best; dividing
    by the global ceiling alone is the scale bias being corrected. The geometric
    mean sits between the two: the dominant graph keeps the global scale exactly
    (its peak IS the ceiling), a weaker graph is lifted halfway toward it, and a
    single graph divides by its own peak — the pre-split behaviour. Callers
    divide by ``scalers.get(graph) or 1.0``.
    """
    peaks = graph_peaks(pairs)
    ceiling = max(peaks.values(), default=0.0)
    return {g: math.sqrt(p * ceiling) for g, p in peaks.items() if p > 0}


def _source_relevance(s: SourceDocument, by_key: dict[str, float]) -> float:
    """A source's raw query relevance — its own, else its best scored chunk's."""
    return s.relevance or by_key.get(
        s.doc_id or "", by_key.get(s.title or "", 0.0)
    )


def document_relevance(quotes: list[ChunkQuote]) -> dict[str, float]:
    """Per-document relevance — its best chunk's score — keyed by both identifiers.

    Gives a graph-traversed document a query relevance. Max rather than mean: one
    on-topic passage makes a document worth reading. Empty when nothing was scored,
    which leaves trust untouched.
    """
    out: dict[str, float] = {}
    for q in quotes:
        if not q.relevance:
            continue
        for key in (q.doc_id, q.doc_title):
            if key:
                out[key] = max(out.get(key, 0.0), q.relevance)
    return out


def rank_sources(
    sources: list[SourceDocument],
    quotes: list[ChunkQuote],
    settings: Settings,
    *,
    today: date | None = None,
    reorder: bool = True,
    time_scoped: bool = False,
    relevance_by_key: dict[str, float] | None = None,
) -> list[SourceDocument]:
    """Return sources sorted by blended trust (each gets its `trust` filled).

    `relevance` (how well the doc matches THIS query) leads the blend: the fused
    chunk cosine/BM25 the retrieval engine already computed, carried on the source
    (0.0 for docs reached only via graph traversal). Without it a merely well-
    connected doc (high `mentions`) outranks an on-topic match that has few graph
    mentions. Relevance is the ENGINE's signal — there is deliberately no hand-rolled
    term/title matching here; stemming, stopwords and semantics stay the job of the
    embedder + BM25 index, not this ranking file.

    ``reorder=False`` keeps the caller's order and does everything else. A mode whose
    evidence arrives already ranked must not have that order overridden by a score the
    user cannot see — but it still needs the per-source work this function is the only
    place for: the citation hover preview, cleaned and trimmed, and `trust` as an
    observable feature value. Two functions would let those silently diverge from the
    ranked path, which is exactly how one mode's citations end up shaped differently
    from another's.
    """
    if not sources:
        return []
    today = today or date.today()
    # Damped per graph, not one global max: `mentions` counts the chunks of ONE
    # graph that name the anchor, so a larger or denser corpus scores every one of
    # its documents higher and `multisource` (0.25 of trust) became a vote for that
    # corpus rather than for the document. The sqrt-damped divisor removes half the
    # scale gap without making a sparse graph's best count for as much as a dense
    # one's.
    mention_scalers = graph_scalers((s.graph, float(s.mentions)) for s in sources)
    relevance_by_key = relevance_by_key or {}
    # A document is "supported" when a verbatim chunk quote backs it. Sources and
    # quotes may each carry doc_id, title, or both (chunk-lane sources can be
    # title-only), so membership is checked on EITHER identifier — the same
    # identity rule the digest's quote ordering uses.
    supported_keys = {k for q in quotes for k in (q.doc_id, q.doc_title) if k}
    # Citation hover preview: the first verbatim excerpt this request
    # already retrieved for each document. Same either-identifier rule as above.
    preview_by_key: dict[str, str] = {}
    for q in quotes:
        if not (q.text or "").strip():
            continue
        for k in (q.doc_id, q.doc_title):
            if k:
                preview_by_key.setdefault(k, q.text)

    # A question that NAMED its period has already stated its time scope, so
    # decaying documents INSIDE that window double-counts it and pushes the
    # window's earliest items toward the truncation edge. Damp the recency term
    # for those requests; the weight is untouched when no window was asked, and
    # a damping of 1.0 restores the previous behaviour exactly.
    w_recency = settings.trust_w_recency * (
        settings.trust_recency_windowed_damping if time_scoped else 1.0
    )
    total_w = (
        settings.trust_w_relevance + settings.trust_w_multisource
        + w_recency + settings.trust_w_support
        + settings.trust_w_authority
    ) or 1.0

    ranked = []
    for s in sources:
        multisource = s.mentions / (mention_scalers.get(s.graph) or 1.0)
        recency = freshness(s.date, today, settings.trust_recency_halflife_days)
        support = 1.0 if (s.doc_id in supported_keys or s.title in supported_keys) else 0.4
        authority = _AUTHORITY.get((s.source_type or "").lower(), _AUTHORITY_DEFAULT)
        # A graph-traversed document carries no relevance of its own, so the
        # blend's largest weight would contribute nothing and many sources would
        # tie. Where the excerpt ranking scored this document's chunks against
        # the question, its best chunk is that missing signal.
        relevance = _source_relevance(s, relevance_by_key)
        trust = (
            settings.trust_w_relevance * relevance
            + settings.trust_w_multisource * multisource
            + w_recency * recency
            + settings.trust_w_support * support
            + settings.trust_w_authority * authority
        ) / total_w
        # Prefer the document's own opening (stamped by ingestion as
        # `doc_preview_text` and carried on the source) over the retrieved excerpt.
        # The opening is what a hover card should show, and it is the only option for
        # a document reached by traversal alone, which has no excerpt at all. A
        # too-short opening is treated as absent: real ones exist that are a single
        # character, and an excerpt beats a preview the reader cannot use.
        opening = (s.snippet or "").strip()
        preview = (
            (opening if len(opening) >= _MIN_USEFUL_PREVIEW else "")
            or preview_by_key.get(s.doc_id or "")
            or preview_by_key.get(s.title or "")
        )
        ranked.append(s.model_copy(update={
            "trust": round(trust, 4),
            "snippet": _snippet(preview, settings.citation_snippet_chars, title=s.title),
        }))

    if not reorder:
        return ranked  # the caller's order IS the ranking; only annotate
    # Trust descending, then a stable identifier: continuous scores (cosine/RRF)
    # make near-ties common, and Python's stable sort would otherwise preserve
    # whatever order the ANN candidate set happened to arrive in — making citation
    # order vary between identical requests.
    ranked.sort(key=lambda s: (-s.trust, s.doc_id or s.title))
    return ranked


def sources_from_quotes(
    sources: list[SourceDocument], quotes: list[ChunkQuote]
) -> list[SourceDocument]:
    """SourceDocument stubs for quote-backed documents the pool has no source for.

    A kept graph chunk proves its document exists and carries its identity
    (doc_id/doc_title), but only the agent's get_source_documents call
    ever ADDS sources — so a run that answered from a
    quote shipped ``citations: []`` and starved the suggested-actions generator
    of its anchors. If a document's words are in the evidence,
    the document belongs in the citations list.

    Same either-identifier rule as ranking: a source or quote may carry doc_id,
    title, or both. A quote with no title synthesizes nothing — no title means
    nothing to render as a citation chip.
    Relevance stays 0.0 (reached via graph, no engine score); the ranking pass
    fills trust (support=1.0 by construction — the quote IS the support) and
    the hover snippet from the quote's text.
    """
    known = {k for s in sources for k in (s.doc_id, s.title) if k}
    out: list[SourceDocument] = []
    for q in quotes:
        title = (q.doc_title or "").strip()
        if not title:
            continue
        if (q.doc_id and q.doc_id in known) or title in known:
            continue
        known.update(k for k in (q.doc_id, title) if k)
        out.append(SourceDocument(doc_id=q.doc_id, title=title, graph=q.graph))
    return out


# Markdown table pipes and separator rows — scaffolding, not content.
_TABLE_SCAFFOLD = re.compile(r"\|+|(?<![\w-])-{3,}(?![\w-])")
# Below this, a document opening carries no information a reader could use, so the
# retrieved excerpt is the better preview. Real stamped openings run to one character.
_MIN_USEFUL_PREVIEW = 25
# A 1-2 character stub left after a completed sentence — the tail of a word the
# fixed-length stamp cut through, e.g. "…across the country. O".
_ORPHAN_TAIL = re.compile(r"(?<=[.!?])\s+\S{1,2}$")
# A sentence terminator followed by whitespace. The lookbehind skips a period that
# closes an initial or an acronym ("U.S.A. Employee Handbook", "V-1.4"), which would
# otherwise end the preview one word in.
_SENTENCE_END = re.compile(r"(?<![A-Z])[.!?](?=\s)")
# Don't accept a sentence break this early — "Overview." is a technically complete
# sentence and a useless preview. Below this we keep reading to the word-boundary cut.
_MIN_SENTENCE_CUT = 60


def _drop_leading_title(text: str, title: str | None) -> str:
    """Remove a title the preview repeats at its start.

    A hover card already shows the title above the snippet, so an opening that begins
    by restating it spends the reader's ~100 characters saying nothing — measured at
    22% of documents, often with the filename too ("India Employee Handbook V-1.4.pdf"
    followed by "India Employee Handbook V-1.4.docx India Employee Handbook…").

    Presentation only, and conservative: the match must be an exact case-insensitive
    prefix of at least 8 characters, so a short title like "IT" can never eat a real
    opening word.
    """
    for cand in {(title or "").strip(), (title or "").strip().rsplit(".", 1)[0]}:
        if len(cand) >= 8 and text[: len(cand)].casefold() == cand.casefold():
            return text[len(cand):].lstrip(" -–—:|.,–—")
    return text


def _snippet(text: str | None, limit: int, *, title: str | None = None) -> str | None:
    """One-line preview for a citation hover card, or None when there is no excerpt.

    Whitespace is collapsed because chunk text carries newlines, list bullets and
    table pipes that render as ragged noise in a small hover card. The cut lands on a
    word boundary and gets an ellipsis, so a preview never ends mid-word.

    Note what this is: the opening of the excerpt that made the document RELEVANT to
    this question, not the opening of the document. The graph stores no chunk
    ordering, so the document's true first lines are not retrievable — and for a
    citation the relevant chunk is arguably the better preview anyway, since it
    shows the reader why the document was cited.
    """
    # Drop markdown table scaffolding before measuring: a chunk that opens with a
    # table would otherwise spend most of the preview on "| --- | --- |". This is
    # presentation only — the excerpt itself is untouched everywhere else.
    text = _TABLE_SCAFFOLD.sub(" ", text or "")
    text = " ".join(text.split())
    text = _drop_leading_title(text, title)
    if not text:
        return None
    if len(text) <= limit:
        # Ingestion stamps the opening at a fixed character count, so a stored preview
        # can end mid-word — real ones finish "…across the country. O". Our own cut
        # lands on a sentence or word boundary, but a preview SHORTER than the limit is
        # passed through whole, orphan and all. Drop a stub that trails a completed
        # sentence: unambiguous, and it never touches a genuine final word.
        return _ORPHAN_TAIL.sub("…", text)

    # Prefer to end on a completed SENTENCE. A preview that stops mid-thought reads as
    # broken however tidy the character count is, and a whole sentence is what makes a
    # hover card answer "is this the document I want?". `limit` is therefore a ceiling,
    # not a target — most previews land well short of it.
    window = text[:limit]
    ends = [m.end() for m in _SENTENCE_END.finditer(window)]
    if ends and ends[-1] >= _MIN_SENTENCE_CUT:
        return window[: ends[-1]]

    # No usable sentence break — fall back to a word boundary and mark the truncation.
    cut = window.rfind(" ")
    return (window[:cut] if cut > 0 else window).rstrip(" ,.;:—-") + "…"


def freshness(date_iso: str | None, today: date, halflife_days: float) -> float:
    """Exponential decay: 1.0 today, 0.5 at one half-life, → 0 with age.

    Public because the repository scores authorship facts with it too
   : one recency curve for the service, not two.
    """
    if not date_iso:
        return 0.3  # undated — below any recent document, above nothing
    try:
        age_days = max(0, (today - date.fromisoformat(date_iso[:10])).days)
    except ValueError:
        return 0.3
    return 0.5 ** (age_days / max(1.0, halflife_days))


# ── contradictions ──────────────────────────────────────────────────────────


def detect_contradictions(claims: list[ClaimFact]) -> list[Contradiction]:
    groups: dict[tuple[str, str], list[ClaimFact]] = defaultdict(list)
    for c in claims:
        if c.statement_type and c.statement_type.upper() != "FACT":
            continue  # opinions/predictions may legitimately disagree
        groups[(c.subject.strip().lower(), c.predicate.strip().lower())].append(c)

    found: list[Contradiction] = []
    for (_, _), group in sorted(groups.items()):
        if len(group) < 2:
            continue
        conflicting = _conflicting_objects(group)
        if len(conflicting) < 2:
            continue
        found.append(
            Contradiction(
                subject=group[0].subject,
                predicate=group[0].predicate,
                objects=[c.object for c in conflicting],
                detail=_windows_note(conflicting),
            )
        )
        if len(found) >= _MAX_CONTRADICTIONS:
            break
    return found


def _conflicting_objects(group: list[ClaimFact]) -> list[ClaimFact]:
    """Distinct, non-substring-related objects whose validity can overlap."""
    kept: list[ClaimFact] = []
    for claim in group:
        obj = claim.object.strip().lower()
        if not obj:
            continue
        related = False
        for other in kept:
            other_obj = other.object.strip().lower()
            if obj in other_obj or other_obj in obj:
                related = True  # "John" vs "John Smith" — same value, not a conflict
                break
            if not _windows_can_overlap(claim, other):
                related = True  # sequential DYNAMIC values, e.g. successive owners
                break
        if not related:
            kept.append(claim)
    return kept


def _windows_can_overlap(a: ClaimFact, b: ClaimFact) -> bool:
    """False only when both are DYNAMIC and their dated windows are disjoint."""
    if (a.temporal_type or "").upper() != "DYNAMIC" or (
        b.temporal_type or ""
    ).upper() != "DYNAMIC":
        return True
    if not (a.valid_from and b.valid_from):
        return True
    a_end = a.valid_to or "9999-12-31"
    b_end = b.valid_to or "9999-12-31"
    return a.valid_from <= b_end and b.valid_from <= a_end


def _windows_note(claims: list[ClaimFact]) -> str | None:
    dated = [c for c in claims if c.valid_from]
    if not dated:
        return None
    spans = ", ".join(
        f"{c.object}: {c.valid_from}{' → ' + c.valid_to if c.valid_to else ''}"
        for c in dated
    )
    return f"reported validity — {spans}"
