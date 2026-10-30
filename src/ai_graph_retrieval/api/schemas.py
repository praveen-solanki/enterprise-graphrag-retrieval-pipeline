"""Request/response DTOs — all input validation lives at this boundary."""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ai_graph_retrieval.clarification.models import AnswerItem, ClarificationQuestion
from ai_graph_retrieval.core.models import (
    Evidence,
    GraphTrace,
    SourceDocument,
    SuggestedAction,
    UsageReport,
)
from ai_graph_retrieval.retrieval.base import Mode


class HistoryTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


# A connector name as ingestion writes it into run_registry.connector_type
# (e.g. "wikipedia", "arxiv"). Lowercase so a name maps to one database family.
_CONNECTOR_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


class RequestOptions(BaseModel):
    """Per-request depth overrides. Every field optional; omitted ⇒ the mode's
    default. The server CLAMPS each to a settings ceiling (repair-not-reject) and
    echoes the effective values in ``RetrieveResponse.options_applied``. Unknown
    fields are ignored, not 422 — a newer client against an older server degrades
    gracefully."""

    model_config = ConfigDict(extra="ignore")

    agent_rounds: int | None = Field(default=None, ge=1)
    graph_chunk_cap: int | None = Field(default=None, ge=0)
    # Per-request override of ACL enforcement: True ⇒ ACL filtering OFF for this
    # request (reads bypass principal checks), False ⇒ ON. Omitted ⇒ the settings
    # default (acl_enforcement_enabled). SECURITY: a caller choosing its own ACL
    # enforcement is a trust grant — intended for a controlled demo/eval env.
    disabled_acl: bool | None = None
    # Per-request override of the entry-classifier short-circuit (greeting /
    # smalltalk / capability / guardrail). True ⇒ those categories return the
    # model-written reply with no retrieval. False ⇒ every turn retrieves.
    # Omitted ⇒ the settings default (chat_smalltalk_shortcircuit).
    chat_smalltalk_shortcircuit: bool | None = None


def _validate_run_selection(
    connectors: list[str],
    run_id_suffix: str | dict[str, str] | None,
) -> None:
    """Shared cross-field check for the (connectorType, runIdSuffix) pair.

    Both OPTIONAL. Omit connectorType ⇒ the server's default connectors; a
    runIdSuffix then must be a map keyed by connector. With a single connector
    the suffix may be a plain string. A malformed pair is a 422 (the run lookup
    must be unambiguous)."""
    bad = [c for c in connectors if not _CONNECTOR_NAME.match(c)]
    if bad:
        raise ValueError(f"invalid connector name(s): {bad}")
    if run_id_suffix is None:
        return
    if isinstance(run_id_suffix, dict):
        # A value of "" (or an absent key) ⇒ that connector has no explicit run.
        if any(not isinstance(v, str) for v in run_id_suffix.values()):
            raise ValueError("runIdSuffix map values must be strings (\"\" ⇒ no explicit run)")
        if connectors:
            extra = set(run_id_suffix) - set(connectors)
            if extra:
                raise ValueError(
                    f"runIdSuffix map keys must be selected connectors; got {sorted(extra)}"
                )
        return
    if len(connectors) != 1:
        raise ValueError(
            "a string runIdSuffix needs exactly one connectorType; use a map, e.g. "
            '{"wikipedia": "a1b2", "arxiv": ""}'
        )


class _RunConfigMixin(BaseModel):
    """The per-request RunConfig surface shared by /retrieve and /retrieve/resume.

    All optional — omit every field and the request reads the server's default
    connectors (``AGR_DEFAULT_CONNECTORS``) with the mode's default profile.
    ``connectorType`` names WHICH graph(s) are read; ``runIdSuffix`` additionally
    names the ingested run whose Neo4j DB + Qdrant collections should be used,
    resolved from ``run_registry``. A connector with no suffix reads its
    configured source, else its latest COMPLETED run."""

    # One connector name or a list, e.g. "wikipedia" or ["wikipedia", "arxiv"].
    connectorType: str | list[str] | None = Field(
        default=None, json_schema_extra={"example": ["wikipedia", "arxiv"]},
    )
    # The run_registry run_suffix. OPTIONAL. A plain string for a single connector,
    # or a per-connector map. Omit the field, or leave a key "", to use the
    # connector's configured source or latest completed run.
    runIdSuffix: str | dict[str, str] | None = Field(
        default=None,
        json_schema_extra={"example": {"wikipedia": "", "arxiv": ""}},
    )
    # Per-request depth overrides (see RequestOptions).
    options: RequestOptions | None = None

    def connectors(self) -> list[str]:
        """The selected connector names, deduplicated in order ([] ⇒ defaults)."""
        raw = ([self.connectorType] if isinstance(self.connectorType, str)
               else self.connectorType or [])
        out: list[str] = []
        for name in raw:
            name = name.strip().lower()
            if name and name not in out:
                out.append(name)
        return out

    @model_validator(mode="after")
    def _check_run_selection(self):
        _validate_run_selection(self.connectors(), self.runIdSuffix)
        return self


class RetrieveRequest(_RunConfigMixin):
    question: str = Field(min_length=1, max_length=2000)
    # Tenant + ACL principals come from request headers (see
    # api/routes/retrieve._access_context) — not from the request body.
    history: list[HistoryTurn] = Field(default_factory=list, max_length=12)
    # Optional conversation id — when set and ownership enforcement is on, the
    # conversation is bound to its first owner and a mismatched owner is
    # rejected, so one user's history can't feed another's synthesis.
    conversation_id: str | None = Field(default=None, max_length=128)
    # REQUIRED: the user picks the mode per query (search / super_research).
    # There is no default and no automatic mode selection — a request without a
    # mode is a 422, not a guess.
    mode: Mode
    include_trace: bool = True

    @field_validator("question")
    @classmethod
    def _strip_question(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("question must not be blank")
        return stripped


class RetrieveResponse(BaseModel):
    request_id: str
    intent: str
    strategy: str  # the strategy that ultimately produced the evidence
    # The mode this answer was produced under — always the caller's own choice
    # (there is no automatic mode selection and no "continue with a richer
    # mode" affordance).
    mode: str
    # False ⇒ retrieval surfaced no evidence at all: the answer text says so
    # (grounded no-answer, never invented content) and a client may render a
    # distinct empty/no-answer state. Evidence-based (any quote/claim/fact/
    # source ⇒ True) — no judge LLM; consistent across all modes.
    # Conversational replies are True.
    answer_found: bool = True
    escalated: bool = False  # retained for contract stability; always False (no escalation exists)
    partial: bool = False  # answer synthesized from partial evidence (retrieval timed out)
    answer: str
    # The documents the answer actually cited, ranked best first — this is the
    # Sources list, not the whole evidence pool (the pool is under `evidence`).
    citations: list[SourceDocument]
    evidence: Evidence
    trace: GraphTrace | None = None
    usage: UsageReport
    duration_ms: int
    # Context-aware follow-ups this service PROPOSES; it never executes them.
    suggested_actions: list[SuggestedAction] = Field(default_factory=list)
    # The RunConfig actually applied AFTER clamping/defaults: the effective
    # depth knobs plus which graph sources were resolved (and how each was
    # selected). Lets a caller detect silent clamps without breaking on
    # older/newer servers.
    options_applied: dict = Field(default_factory=dict)
    # The contributing subgraph: `{"nodes": [{id,label,type?,description?}],
    # "edges": [{id,source,target,label}]}` — the entities the answer names and
    # the relationships among them (synthesis/answer_graph.py). `evidence` is the
    # whole pool; a client's graph view draws THIS, so an answer is shown with
    # what it rests on, not every unrelated node retrieval touched.
    answer_graph: dict = Field(default_factory=lambda: {"nodes": [], "edges": []})
    # The turn's ids. Present on every path (default None for contract stability
    # with older callers). `conversation_id` is server-minted when the caller omits
    # it; the client reuses it for follow-ups and to poll
    # GET /v1/graph/conversations/{conversation_id}/messages.
    conversation_id: str | None = None
    ai_message_id: str | None = None
    user_message_id: str | None = None


class ClarificationResponse(BaseModel):
    """Turn-1 (`202`) body when the service needs the user to clarify first, or to
    confirm a research plan (D3: `kind="plan_review"` carries understanding/scope)."""

    request_id: str
    conversation_id: str
    status: Literal["awaiting_clarification"] = "awaiting_clarification"
    kind: Literal["clarification", "plan_review"] = "clarification"
    understanding: str | None = None
    scope: list[str] = Field(default_factory=list)
    questions: list[ClarificationQuestion]


class ResumeRequest(_RunConfigMixin):
    """Turn-2 payload: the user's answers to a prior clarification. Carries the
    same RunConfig surface as RetrieveRequest so a clarified request keeps its
    graph/run scope and depth overrides on the resume turn."""

    conversation_id: str = Field(min_length=1, max_length=128)
    answers: list[AnswerItem] = Field(min_length=1, max_length=10)
    history: list[HistoryTurn] = Field(default_factory=list, max_length=12)
    # Required, same as RetrieveRequest.mode: the resume turn reruns under the
    # mode the user chose.
    mode: Mode
    include_trace: bool = True


class ConversationSummary(BaseModel):
    """One row in the conversation list — the session doc's caller-facing fields
    (camelCase, matching the stored document; heavy payloads are never here)."""

    model_config = ConfigDict(extra="ignore")

    conversationId: str
    title: str | None = None
    updatedAt: int | None = None  # Unix ms
    createdAt: int | None = None  # Unix ms
    messageCount: int = 0
    lastAiMessageId: str | None = None


class ConversationMessage(BaseModel):
    """One message in a conversation, in ``seq`` order (camelCase, matching the
    stored doc). A `user` message carries its `question`; an `ai_response` carries
    its lifecycle `status` and — once `completed` — the full result `payload` (the
    same dict the SSE `final` event / one-shot `RetrieveResponse` carry, so a client
    renders it identically). A `failed` one carries `errorCode`/`errorMessage`."""

    model_config = ConfigDict(extra="ignore")

    messageId: str
    role: Literal["user", "ai_response"]
    seq: int
    createdAt: int | None = None  # Unix ms
    question: str | None = None
    answer: str | None = None
    partial: bool | None = None
    # ai_response lifecycle: pending | generating | completed | failed | clarification.
    status: str | None = None
    # Present on a `completed` ai_response: the full build_payload() dict (its own
    # keys stay as the RetrieveResponse contract — not camelCased).
    payload: dict | None = None
    # Present on a `failed` ai_response.
    errorCode: str | None = None
    errorMessage: str | None = None


class ConversationListResponse(BaseModel):
    conversations: list[ConversationSummary] = Field(default_factory=list)
    # Skip-based cursor for the next page; None ⇒ no more rows.
    nextCursor: int | None = None


class ConversationMessagesResponse(BaseModel):
    conversationId: str
    messages: list[ConversationMessage] = Field(default_factory=list)


