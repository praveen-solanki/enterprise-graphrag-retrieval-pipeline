"""Inline citation resolution (pure; no I/O, no LLM).

Synthesis attaches a chip — ``[Exact Document Title]`` — to each factual statement.
The MODEL decides where each chip goes and which document supports the statement;
this module does the deterministic half: it rewrites every resolved citation into a
stable numeric placeholder — ``[1]``, ``[2]``, numbered by first appearance — and
returns the cited documents in that same order, so ``citations[n-1]`` IS the
document behind ``[n]``. The UI/backend resolves placeholders into clickable
citations from that mapping; the answer text itself carries no citation links.

Resolution rules (each a dictionary hit — no fuzzy/keyword matching):
  * a chip resolves by exact title lookup against the sources the model was shown;
  * a Markdown link the model wrote itself whose url exactly matches a source
    document is that document cited in the wrong form — it is normalized to the
    same placeholder (anchor text kept, link removed); a link to any other url is
    answer content and passes through verbatim;
  * chips that name nothing real are dropped (a hallucinated citation never
    ships), as are chips to documents without a url (nothing for the placeholder
    to resolve to).

There is deliberately NO trailing "Sources" section: it duplicated the inline
citations. Doing this here rather than in a second LLM pass keeps citations exact
and costs zero extra tokens or latency. Nothing in this file is domain-, query-,
or dataset-aware.

Traced as ``stage.citations`` (attributes only — the function stays pure, and the
span is a no-op wherever tracing is unavailable). Every outcome below is a
counter: chips resolved, chips naming nothing real, chips whose document has no
url, and self-written links normalised. Those four numbers are what a "the answer
shows fewer citations than it should" report needs, and none of them were
observable — a dropped chip leaves no trace in the answer, which is the whole
point of dropping it.
"""
from __future__ import annotations

import json
import re

from ai_graph_retrieval.core.models import SourceDocument
from ai_graph_retrieval.observability.tracing import (
    record_io,
    set_span_attributes,
    stage_span,
)

# How many unresolved chip texts the Output lists. A hallucinated citation is
# usually the same handful of invented titles repeated, so a short sample names
# the problem; the count carries the scale.
_TRACE_MAX_UNRESOLVED = 10

# Chip / link text: ordinary characters, plus ONE nested bracket pair.
#
# The nesting is not decoration — real document titles contain brackets, and the
# some connectors produce titles that are file names
# ("[How to] Configure the build server.docx"). Cited, that title
# becomes "[[How to] Configure …docx]". A bracket-free chip pattern matched only the
# INNER "[How to]", dropped it as an unknown document, and left the rest of the
# title — with the outer bracket still attached — sitting in the answer as prose:
#     Sources: [ Configure the build server.docx]
# One level is all a title can produce (brackets in titles are not themselves
# nested), and bounding it keeps the pattern linear: an unbounded nesting rule
# would let one chip's closing bracket be read as another's opening one.
_TEXT = r"(?:[^\[\]\n]|\[[^\[\]\n]*\])"

# One citation mark, either form the model can produce:
#   * a Markdown link "[text](url)" — matched first, so a link is never read as a
#     chip (the chip branch keeps the negative lookahead as a second guard);
#   * OUR OWN chip format "[Some Document Title]".
# This parses declared shapes only and makes no judgement about content.
# The chip branch also swallows an immediately-following parenthetical that is NOT a
# URL. Without that, a chip the model wrote in link SHAPE but filled with the title
# instead of a link — "[Some Doc]( Some Doc)", seen repeatedly in real answers — matched
# NEITHER branch (no scheme for the link branch, `(` blocked by the old lookahead) and
# was emitted verbatim, so a raw title chip reached the reader. The paren must follow
# with no space, which is what keeps ordinary prose like "[Doc A] (see also…)" safe.
# The length caps are unchanged (300 link text / 200 chip): raising them would make
# this pattern DROP bracketed prose it used to leave alone, which is a behaviour
# change well outside the bracketed-title fix. A title long enough to exceed the cap
# fails exactly as it did before.
_MARK = re.compile(
    rf"\[(?P<ltext>{_TEXT}{{1,300}})\]\((?P<url>https?://[^)\s]+)\)"
    rf"|\[(?P<chip>{_TEXT}{{1,200}})\](?:\((?!https?://)[^)\n]{{0,300}}\))?"
)

# Sentinel marking where a dropped chip stood. Whitespace repair happens ONLY
# around these marks (never across the whole answer), so the model's own
# formatting — Markdown hard breaks (two trailing spaces), code blocks,
# deliberate spacing — is preserved untouched.
_GAP = "\x00"


def _close_gaps(text: str) -> str:
    """Erase the drop-site sentinels, repairing spacing locally: a gap before
    punctuation attaches the punctuation to the preceding word, a gap at a line
    edge vanishes with its padding, and a gap between two words becomes exactly
    one space (so removing a mid-text chip never fuses its neighbours). Two
    passes let adjacent dropped chips ("[A] [B]") cascade cleanly."""
    for _ in range(2):
        if _GAP not in text:
            break
        text = re.sub(rf"[ \t]*{_GAP}[ \t]*([.,;:!?])", r"\1", text)
        text = re.sub(rf"(^|\n)[ \t]*{_GAP}[ \t]*", r"\1", text)
        text = re.sub(rf"[ \t]*{_GAP}[ \t]*(?=\n|$)", "", text)
    return re.sub(rf"[ \t]*{_GAP}[ \t]*", " ", text)


# A subsection's consolidated citation line is model-written scaffolding:
# "Sources: [Doc A] [Doc B]". When every chip on it is dropped — unknown title, or a
# document with no url — the label would be left stranded with nothing after it. This
# matches ONLY a label whose content is entirely gone (optionally bold/italic, with an
# optional trailing colon), so a line that still has a real citation is never touched.
_EMPTY_SOURCE_LINE = re.compile(
    r"(?im)^[ \t]*(?:[-*+][ \t]+)?"                              # optional list bullet
    r"[*_]{0,2}(?:sources?|references?)[ \t]*:?[ \t]*[*_]{0,2}"   # label; colon may sit
    r"[ \t]*:?[ \t]*$\n?"                                        # inside or outside emphasis
)


def _drop_empty_source_lines(text: str) -> str:
    """Remove consolidated-citation lines left with no surviving citation."""
    out = _EMPTY_SOURCE_LINE.sub("", text)
    # A label dropped from the LAST line leaves the break that led into it
    # behind; take it too, so the answer ends where its content ends. Only
    # then — an answer that already ended with a newline keeps it.
    if out.endswith("\n") and not text.endswith("\n"):
        out = out.rstrip("\n")
    return out


# A list entry written as "**Name** — detail" whose detail is missing, leaving the
# separator stranded at the end of the line: "- **<Person Name>** —". Reported from
# the pilot UI as a truncated answer, which is exactly how it reads. The prompt now
# forbids it (synthesis/prompts.py), but the model owns the formatting, so this is
# the deterministic backstop for the same reason `_drop_empty_source_lines` exists.
#
# Deliberately narrow: the separator must be the LAST thing on the line, preceded by
# whitespace, and the line must have real content before it. A dash that starts a
# line (a bullet), sits inside a sentence, or opens a Markdown table row or a `---`
# rule is never touched.
# The EN DASH in the class is deliberate — real answers use
# em, en and hyphen as the same separator, so all three must match.
_DANGLING_SEPARATOR = re.compile(r"(?m)(?<=\S)[ \t]+[—–-]+[ \t]*$")  # noqa: RUF001


def _drop_dangling_separators(text: str) -> str:
    """Strip a trailing "— " left where a detail the model had no value for would
    have gone. The name stays; only the orphaned separator goes."""
    return _DANGLING_SEPARATOR.sub("", text)


# The opening of a consolidated citation line, for deciding whether a self-written
# link sits among prose or among markers.
_SOURCE_LABEL = re.compile(
    r"[ \t]*(?:[-*+][ \t]+)?[*_]{0,2}(?:sources?|references?)[ \t]*:", re.IGNORECASE
)


def _on_source_line(text: str, pos: int) -> bool:
    """True when ``pos`` falls on a line that opens with a Sources/References label."""
    return bool(_SOURCE_LABEL.match(text, text.rfind("\n", 0, pos) + 1))


# A resolved placeholder, as `_sub` writes it. Used only AFTER resolution, on the
# rewritten text, where every remaining `[digits]` is one of ours.
_PLACEHOLDER = re.compile(r"\[(\d+)\]")
# An ATX heading opens a new subsection — the unit the citation contract is
# stated in ("each document ONCE per subsection", synthesis/prompts.py).
_HEADING = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]")


def _dedupe_line_markers(line: str, seen: set[str]) -> tuple[str, int]:
    """One citation line: drop the markers already in ``seen`` (adding the rest
    to it). The line is marker scaffolding, so its spacing is ours to tidy — one
    space between what remains, none trailing; a line left with only its label
    is removed afterwards by ``_drop_empty_source_lines``."""
    kept: list[str] = []
    dropped = 0
    cursor = 0
    for m in _PLACEHOLDER.finditer(line):
        kept.append(line[cursor:m.start()])
        if m.group(1) in seen:
            dropped += 1
        else:
            seen.add(m.group(1))
            kept.append(m.group(0))
        cursor = m.end()
    kept.append(line[cursor:])
    return re.sub(r"[ \t]{2,}", " ", "".join(kept)).rstrip(), dropped


def _drop_repeated_line_markers(text: str) -> tuple[str, int]:
    """Enforce the contract's "each document once per subsection" on the
    consolidated citation line.

    The prompt makes a subsection's ``Sources:`` line the ONE place its citations
    appear. The model does not always comply: it cites a document inline where the
    statement is made AND lists it again on the closing line, so the reader sees
    the same citation twice within a few lines — measured on a live answer as a
    closing line of 24 markers, 4 of them already placed inline above it. Whichever
    placement the model wrote FIRST stands (numbering is by first appearance, so
    it is also the one the number was minted for); a repeat of it on a later
    citation line of the same subsection is scaffolding, not a citation, and goes.
    A document cited ONLY on the line is untouched, and so is any line in a
    subsection with no inline citations — the compliant shape passes through
    byte-for-byte. Returns the text and how many repeats were removed.

    Structural, not textual: it reads the marker placeholders this module itself
    wrote and the two syntactic landmarks (heading, citation label) the resolver
    already recognises. Nothing here reads the answer's words.
    """
    out: list[str] = []
    seen: set[str] = set()
    removed = 0
    for line in text.split("\n"):
        if _HEADING.match(line):
            seen = set()
        elif _SOURCE_LABEL.match(line):
            line, dropped = _dedupe_line_markers(line, seen)
            removed += dropped
        else:
            seen.update(_PLACEHOLDER.findall(line))
        out.append(line)
    return "\n".join(out), removed


def _key(title: str) -> str:
    """Lookup key for a title: case- and whitespace-insensitive, so a chip that
    differs only in spacing/casing from the source list still resolves."""
    return " ".join(title.split()).casefold()


def resolve_citations(
    answer: str, sources: list[SourceDocument]
) -> tuple[str, list[SourceDocument]]:
    """Return (answer with citations as numeric placeholders, cited docs in
    placeholder order).

    Every resolved citation becomes ``[n]``, numbered by first appearance in the
    answer; the returned list is in that same order, so entry ``n-1`` is the
    document ``[n]`` refers to — the stable mapping a UI or backend uses to render
    the clickable citation. A document cited several times (or in both chip and
    self-written-link form) keeps one number.
    """
    if not answer or not sources:
        return answer, []
    with stage_span("stage.citations") as span:
        return _resolve(span, answer, sources)


def _resolve(
    span, answer: str, sources: list[SourceDocument]
) -> tuple[str, list[SourceDocument]]:
    by_key: dict[str, int] = {}
    by_url: dict[str, int] = {}
    for i, s in enumerate(sources):
        by_key.setdefault(_key(s.title), i)
        if s.url:
            by_url.setdefault(s.url, i)

    numbers: dict[int, int] = {}  # source index → placeholder number (1-based)
    # Every branch of _sub below increments exactly one of these, so the counters
    # sum to the number of marks the model wrote — which is what makes them
    # trustworthy as a diagnostic rather than a sample.
    tally: dict[str, int] = dict.fromkeys((
        "chips_resolved", "chips_unresolved", "chips_without_url",
        "links_normalised", "links_as_marker", "links_content",
        "links_without_url",
    ), 0)
    unresolved: list[str] = []

    def _number(idx: int) -> int:
        return numbers.setdefault(idx, len(numbers) + 1)

    def _sub(m: re.Match[str]) -> str:
        url = m.group("url")
        if url is not None:
            idx = by_url.get(url)
            if idx is None:
                # The URL names no source we were shown. Before treating it as ordinary
                # content, try the ANCHOR TEXT against the source titles: the model
                # sometimes links a document it is citing to a URL that differs from the
                # stored one (a trailing slash, a query string), and emitting that raw
                # left a bare markdown link sitting in the answer where a citation
                # belonged. A title match means it IS a citation, whatever the href.
                idx = by_key.get(_key(m.group("ltext")))
            if idx is None:
                tally["links_content"] += 1
                return m.group(0)  # content link, not a citation → verbatim
            if not sources[idx].url:
                tally["links_without_url"] += 1
                return _GAP
            # A source cited as a self-written link. Mid-sentence the anchor text is
            # part of the prose ("see the Onboarding Guide [1] for…") so it stays. On a
            # consolidated citation line it is pure redundancy — the line exists to
            # carry markers, and the title is already one click away — so emit the
            # marker alone rather than "Sources: Some Very Long Title [1]".
            if _on_source_line(answer, m.start()):
                tally["links_as_marker"] += 1
                return f"[{_number(idx)}]"
            tally["links_normalised"] += 1
            return f"{m.group('ltext')} [{_number(idx)}]"
        chip = m.group("chip")
        idx = by_key.get(_key(chip))
        if idx is None:
            tally["chips_unresolved"] += 1
            if len(unresolved) < _TRACE_MAX_UNRESOLVED:
                unresolved.append(chip)
            return _GAP  # names no real document → drop the chip entirely
        if not sources[idx].url:
            tally["chips_without_url"] += 1
            return _GAP  # nothing for the placeholder to resolve to → no citation
        tally["chips_resolved"] += 1
        return f"[{_number(idx)}]"

    cleaned = _MARK.sub(_sub, answer)
    # Kept OUT of `tally`: those counters sum to the marks the model wrote, and a
    # repeat removed here was already counted once as the chip it resolved from.
    cleaned, line_repeats = _drop_repeated_line_markers(cleaned)
    cleaned = _drop_dangling_separators(
        _drop_empty_source_lines(_close_gaps(cleaned))
    )

    # Each citation entry carries its marker number explicitly (ref=n ⇒ this is
    # [n] in the answer) — copies, so the shared evidence.sources rows stay
    # ref-less. The list order still matches (citations[n-1] ↔ [n]).
    cited = [
        sources[i].model_copy(update={"ref": n})
        for i, n in sorted(numbers.items(), key=lambda kv: kv[1])
    ]
    set_span_attributes(
        span,
        sources_shown=len(sources),
        marks=sum(tally.values()),
        citations=len(cited),
        answer_chars_in=len(answer),
        answer_chars_out=len(cleaned),
        # A document cited on several statements keeps one number, so marks
        # minus citations is the REUSE — high reuse on a set question means the
        # answer is leaning on a few documents whatever the source count says.
        marks_per_citation=round(
            tally["chips_resolved"] / len(cited), 2) if cited else 0.0,
        # Markers a subsection's citation line repeated from its own prose —
        # how often the model broke "each document once per subsection".
        line_markers_repeated=line_repeats,
        **tally,
    )
    record_io(span, output=json.dumps({
        "citations": [{"ref": s.ref, "title": s.title} for s in cited],
        # Chip texts the model invented. Model-written, so treated as data:
        # sampled and bounded, never matched against anything.
        "unresolved": unresolved,
    }, ensure_ascii=False, default=str))
    return cleaned, cited
