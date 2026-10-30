"""What the user is told we are doing, in their words — the single vocabulary
for every progress step the UI renders.

Two problems this fixes, both reported from the pilot UI:

  1. **Internal names leaked.** Steps carried the tool that produced them
     (``resolve_entities``, ``expand_entity``, ``find_contributors``) in the
     streamed ``detail`` field. That is our implementation, not the reader's
     concern, and it changes whenever the toolset does. Tool identity now lives
     in the structured LOG line only (``tool=``), where operators want it.

  2. **The wording read like a job queue.** Every lane invented its own label
     ("Mapping the topic", "Searching content", "Collecting sources"), so the
     same work was announced differently depending on the route a question took,
     and none of it read like someone thinking.

Every step therefore carries two things:

  * ``label`` — one of the five PHASES below. A closed set, so a reader sees the
    same five stages on every question and can learn what they mean; a lane
    cannot invent a sixth.
  * ``summary`` — one short, specific, present-tense sentence about THIS step
    ("Following Jane Doe's connections to find who reports to her"). This is
    the part that reads as thought, so it names the thing being worked on
    wherever we know it.

We have no model reasoning stream to show (the gateway does not return one), so
the summaries are written here, from what the code is actually about to do.
They must therefore stay TRUE: a summary describes the work being attempted, and
never claims a result we do not have yet.
"""
from __future__ import annotations

from ai_graph_retrieval.core.models import StepKind

# ── the five phases, in the order a reader meets them ────────────────────────
# Kept as plain constants (not an enum) because they are display strings that
# ship to a UI, and the UI matches on the text.
UNDERSTANDING = "Understanding your request"
SEARCHING = "Searching relevant information"
CONNECTING = "Connecting related information"
ANALYZING = "Analyzing findings"
PREPARING = "Preparing the final response"

PHASES = (UNDERSTANDING, SEARCHING, CONNECTING, ANALYZING, PREPARING)

# The phase a StepKind belongs to. The kind stays on the event (clients that
# already switch on it are unaffected); this is only how it is NAMED.
_PHASE_BY_KIND = {
    StepKind.PLAN: UNDERSTANDING,
    StepKind.SEMANTIC_SEARCH: SEARCHING,
    StepKind.TRAVERSE: CONNECTING,
    StepKind.TOOL_CALL: CONNECTING,
    StepKind.SOURCES: ANALYZING,
    StepKind.SYNTHESIZE: PREPARING,
}


def phase_for(kind: StepKind) -> str:
    """The user-facing phase name for a step kind."""
    return _PHASE_BY_KIND.get(kind, SEARCHING)


def quoted(value: str | None, limit: int = 60) -> str:
    """A subject rendered for a summary sentence — trimmed, quoted, never a
    dangling fragment. Empty input gives an empty string so the caller's
    sentence can fall back to a subject-less phrasing."""
    text = (value or "").strip()
    if not text:
        return ""
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return f"“{text}”"


def with_subject(template: str, subject: str | None, fallback: str) -> str:
    """``template`` filled with a quoted subject, or ``fallback`` when we have
    no subject to name. Keeps every call site from writing the same conditional
    and stops "Following ''s connections" reaching a reader."""
    q = quoted(subject)
    return template.format(subject=q) if q else fallback


def subject_list(names: list[str] | None, limit: int = 2) -> str:
    """Several subjects as ONE plain phrase for a summary sentence — "Alice",
    "Alice and Bob", "Alice, Bob and 3 others".

    Unquoted, unlike ``preview``: this reads inside a sentence the user sees
    ("Collecting the documents written by Alice, Bob and 3 others"), where quote
    marks would make a set of people look like a search string. Empty input gives
    an empty string so the caller can fall back to a subject-less phrasing.
    """
    items = [str(n).strip() for n in (names or []) if n and str(n).strip()]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    shown = items[:limit]
    rest = len(items) - len(shown)
    if not rest:
        return " and ".join((", ".join(shown[:-1]), shown[-1])) if len(shown) > 1 else shown[0]
    return ", ".join(shown) + f" and {rest} other" + ("s" if rest > 1 else "")


def reasoning(*clauses: str | None) -> str:
    """Join 2-3 short reasoning clauses into one linear ``detail`` line.

    This is the step's DYNAMIC subtitle: built
    by the call site from what actually happened on THIS query — real subjects,
    real counts, real decisions — never a canned template and never the model's
    internal chain-of-thought. Clauses must obey the same rules as summaries:
    the reader's language, no tool names, no internal jargon, and nothing
    claimed that did not happen. Empty clauses drop out so a caller can make a
    clause conditional."""
    return " → ".join(c.strip() for c in clauses if c and c.strip())


def preview(names: list[str] | None, limit: int = 3, each: int = 40) -> str:
    """The first few names of a result, for a detail clause — quoted, trimmed,
    with an honest "+N more" tail. Empty input gives an empty string so the
    caller's clause can drop out entirely."""
    items = [quoted(n, limit=each) for n in (names or []) if n and str(n).strip()]
    if not items:
        return ""
    shown = items[:limit]
    more = len(items) - len(shown)
    return ", ".join(shown) + (f" +{more} more" if more > 0 else "")
