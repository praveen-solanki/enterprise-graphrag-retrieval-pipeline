"""Retrieval contracts: lanes, the routing plan, and the per-request context
every strategy receives.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum, StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from ai_graph_retrieval.connectors.llm.client import LlmClient
from ai_graph_retrieval.connectors.llm.embeddings import RequestEmbedder
from ai_graph_retrieval.core.access import AccessContext
from ai_graph_retrieval.core.models import EvidencePool, Usage
from ai_graph_retrieval.core.progress import ProgressReporter
from ai_graph_retrieval.graph.repository import GraphReadRepository
from ai_graph_retrieval.settings import Settings

if TYPE_CHECKING:
    # Imported for typing only: retrieval/profile.py imports Mode/Intent from
    # here, so a runtime import would be circular.
    from ai_graph_retrieval.retrieval.profile import ExecutionProfile


class Mode(str, Enum):
    """Caller-selected retrieval mode (the UI's mode selector).

    Two user-facing modes, and the USER picks one per query — there is no
    automatic mode selection, no default mode, and no routing or escalation
    between modes.

    - SEARCH    — a bounded agent loop over the knowledge graph: it resolves
                  the question's entities, expands them, and reads claims,
                  relationships and a bounded number of graph chunks.
    - SUPER_RESEARCH — the SAME flow, extended: more agent rounds, wide tool
                  rows, a larger graph-chunk allowance, and the mid-research
                  clarification tool. Never a separate retrieval implementation.
    """

    SEARCH = "search"
    SUPER_RESEARCH = "super_research"


# Modes that NEVER pause to ask the user a question: one request in, one answer
# out. Both current modes may pause ONCE, pre-retrieval, when the entry
# classifier itself flagged a genuine answer-changing ambiguity
# (plan.clarification_questions; see pipeline/gates.py). SUPER_RESEARCH may
# additionally pause mid-run via the agent's request_clarification tool
# (profile.mid_research_clarify) — a separate channel from this set, which
# governs the pre-retrieval gate only.
#
# Stated as a SET the gates consult, not as an exclusion list repeated at each
# gate: an exclusion list silently admits new modes the moment the enum grows,
# and adding a mode should not quietly grant it the right to interrupt.
SINGLE_ROUND_TRIP_MODES: frozenset[Mode] = frozenset()


class Intent(str, Enum):
    ENTITY = "entity"
    THEMATIC = "thematic"
    TEMPORAL = "temporal"
    RESEARCH = "research"


class AnswerShape(StrEnum):
    """How the ANSWER should be organised — the classifier's reading of the
    question, decided before anything is retrieved.

    The four members are the four branches of the STRUCTURE precedence list in
    ``synthesis/prompts.py``; a shape added here must be added there in the same
    change, or the two disagree. The VALUES are the wire format (the router's
    tool enum parses the model's reply through them).
    """

    # The items span several periods/releases and the question concerns a period.
    TIMELINE = "timeline"
    # The items share attributes and span no period: one table for the whole set.
    TABLE = "table"
    # Several things asked about each member of a set: one row per member.
    PER_MEMBER = "per_member"
    # Anything else — a single fact, an explanation, a definition, an overview.
    PROSE = "prose"


class ConversationalTurn(BaseModel):
    """The entry classifier's decision that this turn is conversational, not a
    knowledge query: it short-circuits the whole retrieval pipeline and the
    model-written ``reply`` is returned as the answer. The taxonomy is
    prompt-driven (see ``retrieval/router.py``) — extending it is a prompt +
    enum edit, never new branching code."""

    category: str  # greeting | smalltalk | capability | guardrail | out_of_scope
    reply: str


class RetrievalPlan(BaseModel):
    intent: Intent
    anchor_phrases: list[str] = Field(default_factory=list)
    rationale: str = ""
    # The self-contained retrieval query: the question with any conversational
    # references (pronouns, ellipses) resolved from history by the classifier.
    # Empty ⇒ use the question verbatim.
    search_query: str = ""
    # How the answer should be ORGANISED, read off the question alone by the
    # classifier. None ⇒ no opinion, and the prompt's STRUCTURE rules decide —
    # the behaviour before this field existed, so None is always safe.
    # Set by the model, never by code; consumer: synthesis.
    answer_shape: AnswerShape | None = None
    # The time window the QUESTION asks about, resolved by the classifier
    # against the current date ("last six months" → concrete ISO dates); empty
    # when no period is asked. Resolved ONCE by the model that reads the
    # question, then handed downstream as data — so synthesis never has to do
    # date arithmetic (observed failure: a fast-tier answer model computing
    # "September 2025 is within Feb–Aug 2026") and the agent gathers within
    # the same window the answer will be held to. Never set by code.
    time_scope_from: str = ""
    time_scope_to: str = ""
    # The classifier's OWN clarification questions (2026-08-08): written by the
    # same entry call for THIS query — never a template, never a separate
    # planning LLM — and only when a genuine answer-changing ambiguity exists
    # (subject with several distinct referents, genuinely unscoped look-back).
    # Shape: [{"question": str, "options": [str, ...]}]; usually empty. The
    # gates decide whether the mode may pause on them.
    clarification_questions: list[dict] = Field(default_factory=list)

    @property
    def historical(self) -> bool:
        """Whether this question is about the PAST rather than the current state.

        Read off what the classifier already decided, never off the question's
        text: a resolved window means the question named a period, and TEMPORAL
        is the intent for "what changed or happened within a time window". Its
        own prompt states the default the other way round — "a question that
        names none is by default about the CURRENT state" — so absence of both
        is a positive signal, not a missing one.

        Retrieval's one consumer of this is the employment gate
        (retrieval/person_status.py): a current question must not name people who
        have left, and a historical one must still be able to.
        """
        return self.intent is Intent.TEMPORAL or bool(
            self.time_scope_from and self.time_scope_to
        )


@dataclass
class StrategyContext:
    """Everything a strategy needs for one request — all request-scoped."""

    ctx: AccessContext
    question: str
    history: list[dict]
    settings: Settings
    repo: GraphReadRepository
    llm: LlmClient
    embedder: RequestEmbedder
    evidence: EvidencePool
    progress: ProgressReporter
    usage: Usage
    # How hard this request works (the caller's mode → its execution profile).
    # Every depth cap a lane or pass applies comes from here, never from a module
    # constant, so "what does Search do differently" has exactly one answer.
    profile: ExecutionProfile
    # Set by a stage that swallowed a mid-flight failure and continued on the
    # evidence gathered so far (e.g. a provider error inside an agent round).
    # The collector folds it into the response's `partial` flag, so a degraded
    # answer is never presented as a complete one.
    degraded: bool = False
    # The asked time window (plan.time_scope_from/to), carried so the agent's
    # input and the synthesis message can state it as DATA — the window was
    # resolved once by the classifier; downstream models apply it without
    # re-deriving dates. ("", "") ⇒ the question asked no period.
    time_scope: tuple[str, str] = ("", "")
    # The question asks about a PERIOD rather than the current state
    # (``RetrievalPlan.historical``, the classifier's own reading). Carried as
    # data for the same reason the window is: retrieval decides once whether
    # people who have left belong in this answer, and no stage re-reads the
    # question to guess. See retrieval/person_status.py.
    historical: bool = False


class RetrievalStrategy(ABC):
    """A lane fills the evidence pool + trace; synthesis happens outside."""

    name: str

    @abstractmethod
    async def run(self, sc: StrategyContext, plan: RetrievalPlan) -> None: ...


def since_iso(months: int) -> str:
    """ISO date `months` back from today (30-day months, matching stored timestamps).

    UTC, not host-local: every other date in the pipeline (router context, agent
    prompt, synthesis preamble, clarification labels) is UTC, and a host-local date
    here would disagree with them for up to a day around midnight."""
    return (datetime.now(UTC).date() - timedelta(days=30 * months)).isoformat()
