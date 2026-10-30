"""AnswerComposer — one streamed LLM pass over the evidence digest.

The digest is built from the evidence snapshot the orchestrator hands in —
already ACL-filtered at read time, censor-applied, and trust-ranked.
`strict_derived_text` (ACL design doc, Leak 1) controls whether LLM-blended
derived text (entity descriptions) is included; chunk quotes, facts, and claims
are always safe (single-chunk provenance, checked at read time).

Claims outside their validity window are rendered with an explicit
``[expired]`` / ``[future]`` marker so the model presents them as history or
prediction, never as current truth. Detected contradictions get their own
section — the model is instructed to present both sides, not to arbitrate.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime

import structlog

from ai_graph_retrieval.core.models import (
    EntityHit,
    Evidence,
    SourceDocument,
    StepKind,
)
from ai_graph_retrieval.core.narration import reasoning
from ai_graph_retrieval.observability.tracing import (
    annotate_current_span,
    record_io,
    set_span_attributes,
    stage_span,
)
from ai_graph_retrieval.retrieval.base import AnswerShape, Mode, StrategyContext
from ai_graph_retrieval.retrieval.person_status import PERSON_TYPES
from ai_graph_retrieval.retrieval.quality import (
    order_by_date_desc,
    rank_claims_by_recency,
)
from ai_graph_retrieval.synthesis.citations import resolve_citations
from ai_graph_retrieval.synthesis.dedup import (
    content_key,
    interleave_by_document,
    is_near_duplicate,
)
from ai_graph_retrieval.synthesis.prompts import (
    SYNTHESIS_CITATION_ADDENDUM,
    SYNTHESIS_RESEARCH_ADDENDUM,
    SYNTHESIS_SYSTEM_PROMPT,
    shape_addendum,
)

logger = structlog.get_logger("agr.synthesis")

# The one string the digest marks a departed person with, wherever they are
# named. Only a HISTORICAL question can still carry one — retrieval excludes
# them otherwise (retrieval/person_status.py) — and there the flag is what
# keeps a past contributor from reading as a current one.
_INACTIVE_FLAG = " [INACTIVE — no longer with the organization]"

# The marker graph/repository._profile_line leads a departed person's profile
# with. Checked as a substring rather than imported: synthesis must not depend
# on the graph layer, and the profile string is the contract either way.
_FORMER_EMPLOYEE_MARK = "FORMER EMPLOYEE"


@dataclass
class ComposedAnswer:
    """The synthesis result: the answer text plus the documents it actually cited
    (ranked best-first) — the response's Sources list, not the whole evidence pool."""

    text: str
    citations: list[SourceDocument] = field(default_factory=list)


def _alnum_len(text: str) -> int:
    """Length counting alphanumerics only, so markdown scaffolding (#, -, *) and
    whitespace never pass as content."""
    return len(re.sub(r"[^0-9A-Za-z]", "", text))


def _is_degenerate(answer: str, min_alnum: int) -> bool:
    """True when the composed answer has almost no real content — a degenerate model
    output (lone heading / whitespace) rather than a usable answer. The floor is
    `synthesis_min_answer_chars` (config, not a literal); 0 disables the check."""
    return min_alnum > 0 and _alnum_len(answer) < min_alnum

# Digest caps now live in Settings (``digest_*``) so they are tunable per environment
# like every other behaviour knob — they are the caps that most directly shape what the
# answer model can say, and they used to be the only ones with no way to adjust them
# without a code change. See settings.py for each value and its reasoning.


def _truncate_on_boundary(text: str, limit: int) -> str:
    """Truncate to ~``limit`` chars WITHOUT splitting a whitespace-delimited token,
    so a URL (which contains no spaces) is kept whole or dropped — never cut in
    half into an unclickable fragment. Overshoots by at most one token when the
    budget lands mid-token, so a long URL survives intact."""
    text = text.strip()
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    head = text[:limit]
    sp = head.rfind(" ")
    if sp > 0:
        return head[:sp].rstrip()
    nxt = text.find(" ", limit)  # no boundary within budget → finish the token
    return (text if nxt == -1 else text[:nxt]).rstrip()


def _near_duplicate_aliases(quotes, prefix_chars: int) -> dict[str, str]:
    """{document -> the document it is a near-duplicate of}, for the allocator.

    A document is fingerprinted by the opening of its LONGEST chunk — the one most
    likely to carry the shared body — and documents sharing a fingerprint are
    CANDIDATES: a candidate collapses onto the first document of its fingerprint
    only when the two texts agree over the shorter's whole body bar a footer-sized
    tail (``is_near_duplicate``), so a shared boilerplate opening alone never
    groups two substantively different documents. Only confirmed copies appear,
    so the common case returns an empty map and nothing is regrouped.
    """
    if prefix_chars <= 0:
        return {}
    longest: dict[str, str] = {}
    for q in quotes:
        doc = q.doc_id or q.doc_title
        text = (q.text or "").strip()
        if doc and len(text) > len(longest.get(doc, "")):
            longest[doc] = text
    owner: dict[str, str] = {}
    alias: dict[str, str] = {}
    for doc, text in longest.items():
        key = content_key(text, prefix_chars=prefix_chars)
        if len(key) < prefix_chars:
            continue  # too short to fingerprint — never collapse on a partial key
        first = owner.setdefault(key, doc)
        if first != doc and is_near_duplicate(
            longest[first], text, tail_slack=prefix_chars
        ):
            alias[doc] = first
    return alias


def _record_digest(span, digest: str, counts: dict[str, object],
                   rendered: list[dict]) -> None:
    """The selection outcome on ``stage.digest``.

    Documents represented is the headline: an answer can only cite what has text
    behind it, so a digest that spent its budget on eight documents caps the
    answer at eight citations however many sources were ranked.
    """
    set_span_attributes(
        span,
        digest_chars=len(digest),
        excerpt_docs=len({r["doc"] for r in rendered if r["doc"]}),
        excerpt_chars=sum(r["chars"] for r in rendered),
        excerpts_truncated=sum(
            1 for r in rendered if r["chars"] < r["chars_available"]),
        graphs=sorted({r["graph"] for r in rendered if r["graph"]}),
        **counts,
    )
    record_io(span, output=json.dumps(
        {"excerpts": len(rendered), "detail": rendered},
        ensure_ascii=False, default=str))


class AnswerComposer:
    def __init__(self, settings) -> None:
        self._s = settings

    async def compose(
        self,
        sc: StrategyContext,
        evidence: Evidence,
        *,
        mode: Mode,
        research: bool = False,
        answer_shape: AnswerShape | None = None,
        quote_ranked: bool = False,
    ) -> ComposedAnswer:
        """One streamed synthesis pass. ``research`` selects the deeper token
        budget — true for the agent lane Super Research runs. ``mode`` is the
        caller's explicit retrieval mode (required — there is no default mode);
        the mode picks the answer model tier below. ``answer_shape`` is the
        classifier's answer-organisation verdict, read off the question before
        retrieval; None ⇒ no opinion and the prompt's STRUCTURE rules decide. It
        changes the answer's OUTLINE only, never how much evidence the request
        gets — there is one budget per request and no enumeration lane.
        ``quote_ranked`` says the excerpts arrived relevance-ordered, so the
        digest keeps that order.

        Returns the answer text plus the documents the model actually cited, so the
        response's Sources/citations list contains only what supported the answer."""
        # Ground relative-date reasoning in the real current date/time (POC
        # realtime preamble, F8): the model resolves time-sensitive terms
        # ("this year", "latest", "upcoming") from this context, not from Python.
        now = datetime.now(UTC).strftime("%A, %d %B %Y, %H:%M UTC")
        base = SYNTHESIS_SYSTEM_PROMPT
        # SEARCH synthesizes on the fast tier for COST: its digest is large, so it
        # is the one call where the heavy/fast rate gap dominates the request. Its
        # agent rounds stay on the heavy model — tool-driven reasoning is where
        # small models are weakest, and they are few. Super Research keeps the
        # heavy model: its report is the deepest output in the service.
        fast_tier = mode is Mode.SEARCH
        # The asked window, resolved ONCE by the entry classifier, stated as
        # data: the answer model applies it without doing date arithmetic of
        # its own (a fast-tier model computing "is September 2025 within the
        # last six months?" got it wrong; a stated window leaves no math).
        # The window is INPUT for the model's reasoning, never a filter: no
        # evidence is pruned by date here (product direction, 2026-08-10 —
        # deterministic digest pruning excluded every document about a FUTURE
        # window, since anything describing it necessarily predates it; which
        # items fall inside the window is the model's call, per the time-scope
        # rules in the system prompt).
        frm, to = getattr(sc, "time_scope", ("", ""))
        digest = self._digest(
            evidence, strict=self._s.strict_derived_text,
            quote_chars=getattr(getattr(sc, "profile", None), "digest_quote_chars", 0),
            quote_ranked=quote_ranked,
        )
        window = (
            f"\n\nTIME SCOPE ASKED: {frm} to {to} — the question limits itself "
            "to this window. An item whose own date (release, GA, demo cycle, "
            "event) falls outside it must not appear in the answer AT ALL: not "
            "as an item, not as a note, not as an 'outside the window / for "
            "context' mention. Excluded means absent."
            if frm and to else ""
        )
        # ONE model call writes the answer. Enumeration answers used to
        # spend a second, never-shown call first — the "item roster" — which settled
        # the final item list so the date-gating deliberation an autoregressive model
        # cannot un-write never started on the user's page. Its membership rules now
        # live in the base prompt's set-completeness rules (settle the whole set in one pass
        # before the first heading; judge membership by what the evidence shows, not
        # by the question's wording; one entry per item; undated is not excluded), so
        # the same reasoning happens inside the call that writes.
        #
        # Why it had to go rather than be tuned: a second model deciding membership
        # independently disagreed with itself run to run, its forced-function reply
        # could not be continued when it hit the token cap, and a truncated reply
        # parsed to zero items — byte-for-byte identical to "nothing qualifies", which
        # is how an answer came to report no publications while holding them.
        # It also read the whole digest a second time, so every widening
        # of the digest was billed twice.
        # NOTHING per-request goes in here. `now` carries %H:%M, so prepending it
        # changed the first line every minute — and caching is a PREFIX match, so
        # that invalidated the whole 2.2k-4.3k-token block behind it on all but
        # same-minute requests (the agent and the router already moved their
        # timestamp into the user message for this reason).
        # Order is deliberate: shape says what the OUTLINE is, research how DEEP,
        # citations what the output must look like — form rules last, read against
        # a body the content rules already shaped. Keyed on ``research`` and a
        # four-value enum, so the block stays byte-identical between requests
        # (the caching invariant below).
        system = (
            base
            + (shape_addendum(answer_shape) if self._s.answer_shape_enabled else "")
            + (SYNTHESIS_RESEARCH_ADDENDUM if research else "")
            + (SYNTHESIS_CITATION_ADDENDUM
               if (self._s.inline_citations_enabled and evidence.sources) else "")
        )
        messages = [
            {"role": "system", "content": system},
            *sc.history,
            {
                "role": "user",
                # The date/time leads the user message — same placement and same
                # wording shape the agent uses, so the model still resolves
                # "this year" / "latest" from real context rather than guessing.
                "content": (f"CURRENT DATE AND TIME: {now}\n\n"
                            f"QUESTION:\n{sc.question}{window}"
                            f"\n\nEVIDENCE:\n{digest}"),
            },
        ]
        # Per-lane completion cap (F2): research answers more than fast;
        # None ⇒ uncapped.
        max_tokens = (self._s.research_max_tokens if research
                      else self._s.fast_max_tokens)

        async with sc.progress.step(
            StepKind.SYNTHESIZE,
            "Putting the answer together from the evidence, with a source behind each point",
            # User-facing subtitle. Deliberately count-free (product direction,
            # 2026-08-09): the gathering steps already told the user what was
            # found; restating totals here read as an internal metric.
            detail=reasoning(
                "Reviewing the most relevant information from the evidence",
                "writing the answer with a source behind each statement",
                "leaving out anything the material doesn't support",
            ),
        ):
            answer = await self._stream_once(sc, messages, max_tokens, fast=fast_tier)
            # Rare degenerate output (clean stop with ~no content) isn't an error or a
            # length-truncation, so stream_text can't catch it — retry once here rather
            # than ship a lone '#'. A re-issue of the same evidence normally succeeds.
            degenerate_retry = False
            if _is_degenerate(answer, self._s.synthesis_min_answer_chars):
                logger.warning(
                    "degenerate_synthesis_retry",
                    alnum=_alnum_len(answer), preview=answer[:60], **sc.ctx.log_fields(),
                )
                degenerate_retry = True
                retry = await self._stream_once(sc, messages, max_tokens, fast=fast_tier)
                if _alnum_len(retry) > _alnum_len(answer):
                    answer = retry

            # Rewrite the model's chips into stable numeric placeholders ([1],
            # [2], by first appearance) against the ranked sources; citations[n-1]
            # is the document behind [n], for the UI/backend to render. Citations
            # exist ONLY inline — no citation links in the answer text, no
            # trailing Sources section, no guessed fallback list.
            # With citations disabled no chips were requested, so the answer is
            # left untouched (bracket text in it is content, not a chip).
            cited: list[SourceDocument] = []
            if self._s.inline_citations_enabled:
                answer, cited = resolve_citations(answer, evidence.sources)
            # The SYNTHESIZE step already opens a span (core/progress.py), and the
            # generation that writes the answer is its child. What neither carried is
            # what the call RAN UNDER: how big the digest was, which prompt blocks
            # were appended, which model tier, and what came back. Annotating the
            # step's own span keeps that beside the generation instead of adding a
            # sixth box to the waterfall for facts that describe this one.
            annotate_current_span(
                mode=mode.value,
                research=research,
                fast_tier=fast_tier,
                digest_chars=len(digest),
                system_prompt_chars=len(system),
                citation_addendum=bool(
                    self._s.inline_citations_enabled and evidence.sources),
                # The only prompt block decided per QUESTION rather than per mode.
                # The router logs the verdict; this records that it was APPLIED,
                # so "same question, same shape" is checkable without inferring
                # the decision from the answer. Absent ⇒ no verdict.
                answer_shape=answer_shape.value if answer_shape else None,
                shape_applied=bool(answer_shape) and self._s.answer_shape_enabled,
                history_turns=len(sc.history),
                time_scope=bool(frm and to),
                max_completion_tokens=max_tokens or 0,
                answer_chars=len(answer),
                citations=len(cited),
                # One retry means the model returned a lone heading first. It is
                # rare, it doubles the cost of the most expensive call in the
                # request, and it was visible only as a log line.
                degenerate_retry=degenerate_retry,
            )
        return ComposedAnswer(text=answer, citations=cited)

    async def _stream_once(
        self, sc: StrategyContext, messages: list, max_tokens, *, fast: bool = False,
    ) -> str:
        """One streamed synthesis pass — tokens go to the UI live and are joined into
        the returned answer. ``fast`` selects the fast-tier model."""
        parts: list[str] = []
        async for token in sc.llm.stream_text(
            messages, max_tokens=max_tokens, fast=fast, usage=sc.usage,
            # Names the generation that writes the answer — the most expensive
            # observation in the request, previously labelled only by its
            # transport ("llm.chat.stream") rather than its job.
            purpose="synthesis",
        ):
            parts.append(token)
            await sc.progress.answer_delta(token)
        return "".join(parts)

    # ── digest building ────────────────────────────────────────────────────

    def _digest(self, ev: Evidence, *, strict: bool, quote_chars: int = 0,
                entity_urls: bool = False, quote_ranked: bool = False) -> str:
        """The evidence digest, traced as ``stage.digest``.

        The span is where "why wasn't that in the answer?" becomes answerable.
        FOUR gates stand between the evidence pool and the model — document
        trust rank, the count cap, the cumulative character budget and the
        fragment floor — and without this span the only way to see which one
        dropped a given chunk is an offline probe that parses the rendered digest.

        Counts and the budget consumed go on attributes; the
        per-excerpt detail goes on the Output, bounded. Nothing here reads the
        question or the content to decide what to record.
        """
        with stage_span("stage.digest") as span:
            digest, counts, rendered = self._build_digest(
                ev, strict=strict, quote_chars=quote_chars, entity_urls=entity_urls,
                quote_ranked=quote_ranked,
            )
            _record_digest(span, digest, counts, rendered)
            return digest

    def _build_digest(
        self, ev: Evidence, *, strict: bool, quote_chars: int = 0,
        entity_urls: bool = False, quote_ranked: bool = False,
    ) -> tuple[str, dict[str, object], list[dict]]:
        """``quote_chars`` — the per-mode excerpt budget (0 ⇒ the Settings default).

        ``entity_urls`` — render each entity's own page url inline. Off by default:
        document-chip citations are the answer's only links.

        ``quote_ranked`` — the excerpts carry scores, so the graph tier's
        ``relevance`` sort below is meaningful. Recorded (not acted on) so a run
        that took the unscored fallback stays distinguishable.

        Returns the digest text, the SELECTION COUNTS (what each section had
        available against what it actually rendered) and the per-excerpt detail.
        The latter two are for the span only — they never reach the prompt.
        """
        sections: list[str] = []
        counts: dict[str, object] = {}
        rendered: list[dict] = []
        # AVAILABLE vs RENDERED, per section. Only the excerpts are budgeted by
        # characters; entities, facts and claims are count-capped, and a cap that
        # binds is invisible in the digest itself — the section simply looks
        # complete. Recording both numbers is what makes "the roster's own people
        # were dropped by the entity cap" a readable fact rather than a hypothesis.
        counts |= {
            "entities_available": len(ev.entities),
            "facts_available": len(ev.facts),
            "claims_available": len(ev.claims),
            "contradictions": len(ev.contradictions),
            "sources": len(ev.sources),
        }

        if ev.entities:
            lines = []
            # CONTENT TIER when the cap binds — a property of the row, never of
            # the question:
            #   0  a structured profile: the source system's own fields (role,
            #      department, location, employment status). Authoritative, and
            #      the ONLY copy of that data in the whole digest.
            #   1  a blended description: prose ingestion derived from whatever
            #      document happened to mention the entity — useful, but softer,
            #      and often repeating what an EXCERPT already says.
            #   2  neither: renders as a bare "- Name (Person)", telling the
            #      model nothing the FACTS rows do not already carry.
            # Pool order (seed relevance, then discovery order) is preserved
            # WITHIN each tier, so nothing moves until the cap truncates. It
            # binds routinely — a 13-person roster arrives with ~75 entities
            # against a 50 cap, and the profile pass appends last, so without
            # tiering the roster's own people were the first thing dropped.
            # A PERSON is never outranked by its own missing profile. An
            # enumeration returns the complete set and keeps unprofiled members
            # deliberately (repository._profiles_from_rows, keep_unprofiled) —
            # tiering on `profile` alone put exactly those members last and the
            # cap then dropped them, so a roster reported complete came back
            # short. People rank as one tier, richest first inside it; everything
            # else keeps the original content tiering below them.
            # sorted() is stable, so pool order is preserved WITHIN a tier —
            # which is the documented intent (seed relevance first) and is itself
            # deterministic now that the pool no longer depends on tool order.
            def _tier(e: EntityHit) -> tuple[int, int]:
                person = (e.type or "").strip().casefold() in PERSON_TYPES
                content = 0 if e.profile else (1 if e.description else 2)
                return (0 if person else 1, content)

            ranked = sorted(ev.entities, key=_tier)
            for e in ranked[: self._s.digest_max_entities]:
                # Bounded, but on a WORD boundary and at a configurable width. The old
                # hardcoded 280 cut explanations mid-word; the description is separately
                # bounded at assembly (graph/repository.py, _DESCRIPTION_CHARS = 600), so
                # this is a prompt-budget control rather than a safety one.
                desc = "" if strict or not e.description else " — " + _truncate_on_boundary(
                    e.description, self._s.digest_entity_desc_chars
                )
                # The entity's own page (person profile/space, site home) when
                # the source supplied one, rendered only when ``entity_urls`` is
                # on (distinct from citations, which stay document-only).
                url = f" — url: {e.url}" if entity_urls and e.url else ""
                # The source's own structured attributes (role, department,
                # location, former-employee status). Rendered for every mode:
                # unlike the page url this is answer CONTENT, and an enumerated
                # roster needs it to fill its columns.
                profile = f" — {e.profile}" if e.profile else ""
                # The status as its own flag. A profile built by the SAME read
                # that saw the status already leads with FORMER EMPLOYEE — but
                # the cross-graph sweep (person_status, historical questions)
                # stamps `status` on entities whose profile came from the graph
                # that still reads them Active, so suppressing on the profile's
                # mere existence rendered exactly those people as unmarked
                # current colleagues. The profile suppresses the flag only when
                # it actually carries the marker.
                status = _INACTIVE_FLAG if (
                    (e.status or "").lower() == "inactive"
                    and _FORMER_EMPLOYEE_MARK not in (e.profile or "")
                ) else ""
                lines.append(
                    f"- {e.name} ({e.type or 'CONCEPT'}){status}{desc}{profile}{url}")
            # The tier that survived the cap, so a dropped structured profile is
            # distinguishable from a dropped bare name.
            counts["entities_rendered"] = len(lines)
            counts["entities_with_profile"] = sum(
                1 for e in ranked[: self._s.digest_max_entities] if e.profile)
            sections.append("ENTITIES:\n" + "\n".join(lines))

        if ev.facts:
            lines = []
            # STRUCTURE first, then score — never raw support. A schema-asserted
            # edge (REPORTS_TO, MEMBER_OF, WORKS_IN) carries no supporting chunks,
            # so the support curve scores every one of them at its floor and this
            # section rendered the org backbone behind every extracted claim and
            # cut it first when the cap bound. `score` carries the damped
            # cross-graph leveling (one curve, comparable across graphs), while
            # `support` is a per-graph chunk count — leading with it handed the
            # section to the denser corpus. Same key as the neighbour scorer's
            # cheap ranking: support breaks score ties, the target keeps the order
            # deterministic.
            ranked = sorted(
                ev.facts,
                key=lambda f: (not f.structural, -f.score, -f.support, f.target or ""),
            )
            # Selection stays relevance-ranked above; presentation is newest
            # first, because the row RENDERS a date and the model writes dated
            # items out in the order it received them.
            for f in order_by_date_desc(
                ranked[: self._s.digest_max_facts], lambda f: f.date
            ):
                date = f" [{f.date_basis or 'dated'}: {f.date}]" if f.date else ""
                # A relationship description is the sentence that says WHY two things are
                # connected — often the whole answer to "how does X relate to Y" — so it
                # is truncated on a word boundary rather than mid-word, and at a width
                # you can raise per environment.
                desc = (" — " + _truncate_on_boundary(
                    f.description, self._s.digest_fact_desc_chars
                )) if f.description else ""
                # BOTH ends can be a person, and the flag must ride with whichever
                # one has left: an authorship or contribution edge is walked FROM
                # the person, so flagging only the target printed "[INACTIVE]"
                # beside the document or topic they worked on.
                tgt_flag = _INACTIVE_FLAG \
                    if (f.target_status or "").lower() == "inactive" else ""
                src_flag = _INACTIVE_FLAG \
                    if (f.source_status or "").lower() == "inactive" else ""
                # `direction` is relative to source: an incoming edge reads
                # target→source, so render it that way for a truthful statement
                # (e.g. "<report> REPORTS_TO <manager>", not the reverse). Each
                # flag rides with its own end either way.
                if f.direction == "incoming":
                    subj, obj = f"{f.target}{tgt_flag}", f"{f.source}{src_flag}"
                else:
                    subj, obj = f"{f.source}{src_flag}", f"{f.target}{tgt_flag}"
                # The raw extracted forms, when they say more than the canonical
                # type: ingestion normalises several source phrasings into one
                # relationship_type and keeps the originals in raw_types. Mostly
                # they coincide (rendering them again would be noise); where they
                # differ, each raw form is a distinct phrasing the source actually
                # used, which the model can quote where the canonical name is too
                # generic ("RELATED_TO" vs "mentors, approves budget for").
                raw = [t for t in (f.specific_relationships or [])
                       if t and t != f.relationship]
                also = f" [stated as: {', '.join(raw)}]" if raw else ""
                lines.append(
                    f"- {subj} {f.relationship} {obj}"
                    f" (support: {f.support}){also}{date}{desc}"
                )
            counts["facts_rendered"] = len(lines)
            sections.append("FACTS:\n" + "\n".join(lines))

        if ev.claims:
            lines = []
            # Ranked before the cap, like the sections above: the pool holds the
            # fan-out's concatenation (one graph's rows, then the next), so a
            # positional slice kept the first graph's oldest claims over the
            # second's newest. Same order the fetch already asked for
            # (cypher.FETCH_CLAIMS: valid_from DESC), applied across graphs.
            # Expired claims are ordered, never dropped — historical recall is
            # deliberate, and the rendered line flags them.
            ranked = rank_claims_by_recency(ev.claims)
            for c in ranked[: self._s.digest_max_claims]:
                window = f" [{c.date_basis or 'dated'}: {c.valid_from or '?'}"
                window += f" → {c.valid_to}]" if c.valid_to else "]"
                validity = "" if c.is_current else " [expired]" if self._is_past(c) else " [future]"
                lines.append(
                    f"- ({c.statement_type or 'FACT'}) {c.subject} {c.predicate} {c.object}"
                    f" (support: {c.support}){window if c.valid_from else ''}{validity}"
                )
            counts["claims_rendered"] = len(lines)
            sections.append("CLAIMS:\n" + "\n".join(lines))

        if ev.contradictions:
            lines = []
            for con in ev.contradictions:
                values = " vs ".join(f'"{o}"' for o in con.objects)
                detail = f" ({con.detail})" if con.detail else ""
                lines.append(f"- {con.subject} {con.predicate}: {values}{detail}")
            sections.append("CONFLICTING REPORTS:\n" + "\n".join(lines))

        if ev.quotes:
            quotes_in = ev.quotes
            max_quotes = self._s.digest_max_quotes
            # PER-MODE, one budget per request. The old aggregation branch read a
            # global setting and ignored the profile, so a mode's own ceiling applied
            # to only some of its questions; with the lane gone that cannot recur.
            remaining = quote_chars or self._s.digest_quote_char_budget
            budget = remaining
            counts |= {
                "quotes_pool": len(ev.quotes),
                "quote_char_budget": budget,
                "max_quotes": max_quotes,
                "min_excerpt_chars": self._s.digest_min_excerpt_chars,
                "round_robin": self._s.digest_round_robin_enabled,
            }
            # Lead the excerpts with the most trustworthy/relevant source's text:
            # order quotes by their doc's position in the (already trust-ranked)
            # sources list, so the capped/budgeted window keeps the top source's
            # content instead of arbitrary pool-insertion order. Docs with no
            # source row sort last (stable).
            # Match a quote to its source by EITHER identifier. SourceDocument and
            # ChunkQuote both allow title-only records, and the two sides do not always
            # carry the same one — indexing on doc_id alone (or picking one key per
            # side) silently dropped a title-only quote to the tail even when its
            # source ranked first. Index both; look up doc_id first, then title.
            src_rank: dict[str, int] = {}
            for i, s in enumerate(ev.sources):
                for k in (s.doc_id, s.title):
                    if k and k not in src_rank:
                        src_rank[k] = i
            unranked = len(ev.sources)  # sorts after every ranked source

            def _quote_rank(q) -> int:
                for k in (q.doc_id, q.doc_title):
                    if k and k in src_rank:
                        return src_rank[k]
                return unranked

            def _quote_date(q) -> str:
                """The excerpt's document date, day precision — so a
                time-scoped question can be answered from the excerpts
                themselves (the model cannot honor "the last six months" on
                text whose date lives only in the SOURCES list two sections
                away). Empty when the document carries no date."""
                for k in (q.doc_id, q.doc_title):
                    if k and k in src_rank:
                        date = ev.sources[src_rank[k]].date
                        return f", {str(date)[:10]}" if date else ""
                return ""

            # Graph chunks by question-similarity, highest first. A chunk the lane
            # could not score keeps 0.0 and falls back to its document's rank, so
            # an unavailable vector store degrades to document order rather than
            # shuffling the list.
            ranked_quotes = sorted(
                quotes_in, key=lambda q: (-(q.relevance or 0.0), _quote_rank(q))
            )
            # BREADTH FIRST (optional): give every document its first excerpt before
            # any document gets a second. The order above decides WHICH documents
            # come first; this decides how the budget is SPREAD across them. See
            # settings.digest_round_robin_enabled — a greedy fill lets a few
            # documents take most of the budget and cut the rest whole.
            #
            # A republished copy shares its original's group rather than claiming a
            # second document's slot: the allocator spends the budget per GROUP, so
            # near-duplicates compete with each other instead of with the corpus.
            # The group stays the DOCUMENT — the fingerprint only maps two documents
            # onto one key, it never splits a document's own chunks apart.
            alias = _near_duplicate_aliases(
                quotes_in, self._s.digest_near_duplicate_prefix_chars
            )

            def _doc_group(q) -> str | None:
                doc = q.doc_id or q.doc_title
                return alias.get(doc, doc)

            quotes = (
                interleave_by_document(ranked_quotes, doc_of=_doc_group)
                if self._s.digest_round_robin_enabled else ranked_quotes
            )
            counts |= {
                "quotes_graph": len(quotes),
                "quotes_over_count_cap": max(0, len(quotes) - max_quotes),
            }
            # A truncated excerpt shorter than this is a fragment, not evidence: it
            # costs a line and names a document the model may read as support. Only
            # TRUNCATED excerpts are judged — a genuinely short chunk (a people
            # record is often just a name and a role) is real content, kept whole.
            min_excerpt = self._s.digest_min_excerpt_chars
            lines = []
            # (date, rendered line) for each excerpt that survives the budget —
            # the pair lets the KEPT set be presented newest first without
            # changing which excerpts the budget selected.
            kept_lines: list[tuple[str, str]] = []
            fragments_skipped = 0
            kept_ids: set = set()
            budget_exhausted = False
            for q in quotes[:max_quotes]:
                text = _truncate_on_boundary(q.text, remaining)
                if not text:
                    budget_exhausted = True
                    break
                if (
                    min_excerpt
                    and len(text) < min_excerpt
                    and len(text) < len((q.text or "").strip())
                ):
                    fragments_skipped += 1
                    continue      # try the next quote; a shorter one may fit whole
                flag = " [SUBJECT INACTIVE — no longer with the organization]" \
                    if (q.subject_status or "").lower() == "inactive" else ""
                # The DOCUMENT's date — the only date the evidence structures
                # for an excerpt. Passages of one document therefore share a
                # key and keep their relevance order among themselves; ordering
                # by a date stated in the prose itself is deliberately not done
                # in code (no text parsing here) — the synthesis prompt owns
                # that reading ("a date stated only in prose counts like a
                # dated field").
                kept_lines.append((
                    _quote_date(q).lstrip(", "),
                    f'- From "{q.doc_title or "a source document"}"'
                    f'{_quote_date(q)}{flag}: "{text}"',
                ))
                # One row per rendered excerpt, for the span's Output. `chars` against
                # `chars_available` is the truncation: an excerpt cut to a third of
                # its chunk is a different kind of evidence from one rendered whole,
                # and the digest text shows neither.
                rendered.append({
                    "doc": q.doc_title, "graph": q.graph, "chars": len(text),
                    "chars_available": len((q.text or "").strip()),
                    "relevance": round(q.relevance or 0.0, 4),
                })
                kept_ids.add(q.chunk_id)
                remaining -= len(text)
            # PRESENTATION only — the budget above already chose these excerpts.
            # The line carries the document's date (_quote_date), and the model
            # writes dated material out in the order it reads it, so an
            # oldest-first excerpt block produced an oldest-first answer.
            # Undated excerpts keep their relevance order, after the dated ones.
            lines = [line for _date, line in
                     order_by_date_desc(kept_lines, lambda pair: pair[0])]
            counts |= {
                "quotes_rendered": len(lines),
                "quote_chars_used": budget - remaining,
                "fragments_skipped": fragments_skipped,
                # TRUE means the budget ran out mid-pool: everything after this
                # point was retrieved, deduplicated, ranked and never shown. It is
                # the single most useful bit on this span.
                "budget_exhausted": budget_exhausted,
            }
            if len(lines) < len(quotes):
                # The budget discarded retrieved evidence. STARVED DOCUMENTS is the
                # load-bearing number: a document that lost every one of its chunks
                # cannot be quoted at all, so a thin answer is indistinguishable from
                # thin evidence without it. The span carries the counts; this names
                # which store paid for it (multi-graph provenance).
                #
                # From the KEPT SET, not the list's head: the fragment floor skips
                # mid-list (`continue` above), so the rendered quotes are not the
                # first len(lines) of `quotes` — a head slice misattributed both
                # the starved documents and the per-graph counts whenever a
                # fragment was skipped.
                dropped_quotes = [q for q in quotes if q.chunk_id not in kept_ids]
                shown = {q.doc_id or q.doc_title or "?"
                         for q in quotes if q.chunk_id in kept_ids}
                starved = {q.doc_id or q.doc_title or "?"
                           for q in dropped_quotes} - shown
                logger.info(
                    "digest_quotes_truncated",
                    dropped=len(dropped_quotes), kept=len(lines),
                    pooled=len(quotes), starved_documents=len(starved),
                    relevance_ranked=quote_ranked,
                    by_graph=dict(Counter(q.graph or "?" for q in dropped_quotes)),
                )
            if lines:
                sections.append("EXCERPTS:\n" + "\n".join(lines))

        if ev.sources:
            lines = []
            for s in order_by_date_desc(ev.sources, lambda s: s.date):
                line = f"- {s.title}" + (f" ({s.date})" if s.date else "")
                # Hand the permalink to synthesis (when ingestion stamped one) so it can
                # render a clickable link; omitted → the title is cited as plain text.
                if s.url:
                    line += f" — url: {s.url}"
                lines.append(line)
            sections.append("SOURCES:\n" + "\n".join(lines))

        # Exact per-section sizes, from our own labels — never re-parsed out of the
        # rendered text, whose excerpt bodies carry newlines of their own. This is
        # how "excerpts are only 43% of the digest" became a measurement instead of
        # an impression (10 runs, 2026-08-27).
        counts["section_chars"] = json.dumps(
            {s.split(":", 1)[0]: len(s) for s in sections})
        digest = "\n\n".join(sections) if sections else "(no evidence found)"
        return digest, counts, rendered

    @staticmethod
    def _is_past(c) -> bool:
        """A non-current claim is expired when it has an end date; otherwise its
        window simply has not started (future / prediction)."""
        return c.valid_to is not None
