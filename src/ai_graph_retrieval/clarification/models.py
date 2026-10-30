"""Value objects for the clarifying-questions flow (pure; no I/O)."""
from __future__ import annotations

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class ClarificationOption(BaseModel):
    id: str
    label: str


class ClarificationQuestion(BaseModel):
    id: str
    question: str
    options: list[ClarificationOption] = Field(default_factory=list)
    allow_multiple: bool = False
    allow_free_text: bool = True  # the UI always offers a free-text "Other"


class AnswerItem(BaseModel):
    """One answered question on the `/resume` turn."""

    question_id: str
    selected_option_ids: list[str] = Field(default_factory=list)
    free_text: str | None = None


class ClarificationSession(BaseModel):
    """Durable state persisted between the clarify turn and the resume turn.

    Lifecycle state lives in the store, not here: Redis TTL expires abandoned
    sessions and the atomic GETDEL consume makes a status flag redundant.
    """

    conversation_id: str
    tenant_id: str
    user_id: str | None = None
    original_question: str
    history: list[dict] = Field(default_factory=list)
    questions: list[ClarificationQuestion] = Field(default_factory=list)
    # The CHAT thread this pause belongs to (the client's conversation_id on the
    # paused request) — distinct from `conversation_id`, which identifies this
    # one-shot clarification session. Carried across the pause so the resumed
    # turn reaches the same server-side conversation memory; None
    # when the caller had no chat thread.
    chat_conversation_id: str | None = None


# Stand-in choices for a mid-research question the agent asked without usable
# options. Deliberately content-free: we do not know what the agent wanted to
# disambiguate, only that the user must be able to REPLY.
_YES_NO_OPTIONS = ("Yes — go ahead", "No — take a different angle")


def build_question(question: str, options: list[str] | None = None) -> ClarificationQuestion:
    """Build one ClarificationQuestion from plain text + option labels (the shape
    the research agent's request_clarification tool produces). Ids are synthesized;
    the UI adds its own free-text 'Other'.

    A question ALWAYS ships with at least two options. The agent's tool schema asks
    for 2-5, but nothing enforced it, so a model that sent none (or only blank
    labels) produced a pause the user could not answer: the UI has no choice to
    click, and `ResumeRequest.answers` requires at least one entry keyed to a
    question — the run simply stopped. Substituting a yes/no keeps the pause
    answerable, which is the whole point of pausing.
    """
    opts = [
        ClarificationOption(id=f"o{i}", label=label)
        for i, label in enumerate(options or [])
        if label and label.strip()
    ]
    if len(opts) < 2:
        logger.warning(
            "clarification_options_substituted",
            supplied=len(opts), question_chars=len(question or ""),
        )
        opts = [
            ClarificationOption(id=f"o{i}", label=label)
            for i, label in enumerate(_YES_NO_OPTIONS)
        ]
    return ClarificationQuestion(id="q0", question=question.strip(), options=opts)


class MidResearchClarification(Exception):
    """Control-flow signal (NOT an HTTP error): the pipeline needs the user to
    disambiguate before continuing. Two raisers share it — the gates stage
    (Search Mode, on the entry classifier's own questions, pre-retrieval) and
    the research agent's request_clarification tool (mid-run, currently off).
    Carries the questions to present; the route persists a session and returns
    them via the clarify/resume transport. Never reaches the HTTP error
    handler — routes catch it explicitly."""

    def __init__(self, questions: list[ClarificationQuestion]) -> None:
        super().__init__("clarification requested")
        self.questions = questions
