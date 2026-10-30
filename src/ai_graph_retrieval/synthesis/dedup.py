"""Digest-side quote utilities (moved from the deleted retrieval/fusion.py —
the hybrid chunk seed is gone, but the DIGEST still needs both: cross-source
text dedup is the "never pass the same content downstream twice" guarantee of
the two-source design, and the breadth-first allocator decides how the excerpt
budget spreads across documents)."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar

T = TypeVar("T")


def interleave_by_document(
    items: Sequence[T], *, doc_of: Callable[[T], str | None]
) -> list[T]:
    """Round-robin over documents: every document contributes its first item before
    any document contributes its second, and so on.

    Both orders are preserved — documents keep the order they first appear in (so a
    trust-ranked input stays trust-ranked), and each document's own items keep their
    relative order. Pure reordering: the same items come out, never fewer.

    Why the digest needs it: a greedy excerpt loop fills the character budget in
    document order, so the first few documents consume it and the rest render
    nothing at all — the answer then has their titles in SOURCES with no text
    behind them, which reads as a one-line non-answer.

    Round-robin spends the same budget on the same chunks in a different order, so
    breadth of documents wins over depth in the first few. It decides nothing about
    relevance and reads nothing about the query, the domain or the content.
    """
    groups: dict[str, list[T]] = {}
    for it in items:
        groups.setdefault(doc_of(it) or "", []).append(it)
    out: list[T] = []
    # dict preserves insertion order, so this walks documents in rank order.
    for i in range(max((len(v) for v in groups.values()), default=0)):
        for group in groups.values():
            if i < len(group):
                out.append(group[i])
    return out


def _normalized(text: str) -> str:
    """Whitespace-collapsed, casefolded text — the identity used for duplicate
    detection. Two renderings of the same chunk differ in exactly these ways
    (the API preserves original spacing/newlines; our graph chunks are cleaned)."""
    return " ".join(text.split()).casefold()


def content_key(text: str, *, prefix_chars: int) -> str:
    """A document's near-duplicate identity: the opening of its normalized text.

    Exact and containment dedup (``dedupe_quote_texts``) compare whole strings, so
    two copies that differ ANYWHERE inside — a changed heading, a reordered
    section, an emoji in the body — are neither equal nor contained and both
    survive as separate documents. They then take separate round-robin slots and
    separate excerpt budget — two pages of the same content under titles differing
    only by an emoji can take a large share of one query's budget.

    Two documents that open with the same ``prefix_chars`` of normalized text are
    CANDIDATE duplicates for ALLOCATION — the caller confirms a candidate pair
    with ``is_near_duplicate`` before grouping, so shared boilerplate openings
    (templated forms, standard headers) never collapse genuinely different
    documents. Deterministic, content-only (no similarity threshold to tune,
    nothing read from the query). Falls back to the empty string for empty text,
    which the caller keys as "ungrouped".
    """
    return _normalized(text)[:prefix_chars]


def is_near_duplicate(a: str, b: str, *, tail_slack: int) -> bool:
    """Whether two texts are the same document republished: normalized, the
    shorter must be a prefix of the longer over its whole length minus at most
    ``tail_slack`` trailing characters (a footer, a signature, a trailing edit).

    This is what separates a republished copy from two documents sharing a
    boilerplate opening: the copy agrees essentially everywhere, the boilerplate
    pair diverges as soon as the substantive content starts.
    """
    na, nb = _normalized(a), _normalized(b)
    shorter, longer = sorted((na, nb), key=len)
    required = max(len(shorter) - tail_slack, tail_slack)
    return longer.startswith(shorter[:required])


def dedupe_quote_texts(
    items: Sequence[T], *, text_of: Callable[[T], str]
) -> list[T]:
    """Drop items whose text duplicates — or is wholly CONTAINED in — another
    item's text, keeping the longer one. Order-preserving over the survivors.

    Why containment and not just equality: the two chunk sources can hold the
    SAME chunk under different chunk ids, which the pool's id-keyed dedup cannot
    see, and the two copies are rarely byte-equal — one is a prefix of the other.
    The longer copy carries everything the shorter one does, so the shorter is pure
    digest-budget waste.

    WHICH SIDE LOSES has flipped, and the direction is worth knowing. It used to be
    the graph's: its chunks were truncated at 2,500 chars while the API's were not.
    Since ``enrich_chars_per_doc`` rose to 6,000 the graph's copy is usually the
    longer one — measured over six Super Research runs (2026-08-27), dedup drops
    8-16 API chunks per run against 1-4 graph chunks. So graph text already wins the
    overlap, and any starvation of graph chunks downstream is an ordering problem
    rather than a dedup one.

    Deterministic: candidates are considered longest-first (ties by input
    position), so which copy survives never depends on seed-arrival order.
    O(n²) substring checks — fine at this pool's scale (≤ ~90 quotes).
    """
    order = sorted(
        range(len(items)), key=lambda i: (-len(text_of(items[i])), i)
    )
    kept_idx: list[int] = []
    kept_norm: list[str] = []
    for i in order:
        norm = _normalized(text_of(items[i]))
        if not norm or any(norm in existing for existing in kept_norm):
            continue
        kept_idx.append(i)
        kept_norm.append(norm)
    kept_idx.sort()
    return [items[i] for i in kept_idx]
