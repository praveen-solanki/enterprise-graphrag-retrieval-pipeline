"""ProgressReporter — the one object strategies/agent/synthesis talk to about
"what is happening".

It fans a single call out to both consumers of progress:
  1. the **stream** (SSE events for the live UI), and
  2. the **trace** (the persistent step list + subgraph in the final response).

`emit` is an async callback; the streaming route wires it to an asyncio.Queue,
the non-streaming route wires a no-op. Strategies never know the difference.
"""
from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager

import structlog

from ai_graph_retrieval.core.events import (
    AnswerDeltaEvent,
    StepEvent,
    StreamEvent,
    TraceDeltaEvent,
)
from ai_graph_retrieval.core.models import (
    ChunkQuote,
    ClaimFact,
    EntityHit,
    GraphTraceBuilder,
    RelationFact,
    SourceDocument,
    StepKind,
    StepStatus,
    new_step,
)
from ai_graph_retrieval.core.narration import phase_for
from ai_graph_retrieval.observability.tracing import stage_span

Emit = Callable[[StreamEvent], Awaitable[None]]

logger = structlog.get_logger("agr.trace")


async def null_emit(_: StreamEvent) -> None:
    """Emitter for the non-streaming path."""


# Reserved headroom for the event's own envelope (event name, nodes/edges keys).
_DELTA_OVERHEAD_BYTES = 512


def _delta_chunks(nodes: list, edges: list, budget: int) -> list[tuple[list, list]]:
    """Pack one merge's new nodes+edges into as many (nodes, edges) slices as it
    takes to keep each serialized event under ``budget``.

    One merge used to be one event however large — a tool result carrying
    hundreds of claims (each a node plus its ASSERTS/SUPPORTED_BY edges, with
    the claim text embedded in every id) serialized into one huge event that
    proxies and relays with a frame-size ceiling drop silently. Slices keep
    the stream's order (all of a merge's nodes before its edges), and the UI
    already merges deltas incrementally, holding an edge back until both its
    endpoints exist — so a split is invisible except in frame count.
    """
    ceiling = max(budget - _DELTA_OVERHEAD_BYTES, 1)
    chunks: list[tuple[list, list]] = []
    cur_nodes: list = []
    cur_edges: list = []
    used = 2

    def _flush() -> None:
        nonlocal cur_nodes, cur_edges, used
        if cur_nodes or cur_edges:
            chunks.append((cur_nodes, cur_edges))
            cur_nodes, cur_edges, used = [], [], 2

    for bucket, items in ((0, nodes), (1, edges)):
        for item in items:
            cost = len(item.model_dump_json().encode()) + 1
            if (cur_nodes or cur_edges) and used + cost > ceiling:
                _flush()
            (cur_edges if bucket else cur_nodes).append(item)
            used += cost
    _flush()
    return chunks


class ProgressReporter:
    def __init__(
        self,
        emit: Emit,
        trace: GraphTraceBuilder,
        answer_delta_max_chars: int = 0,
        trace_delta_max_bytes: int = 49_152,
    ) -> None:
        self._emit = emit
        self._trace = trace
        self._trace_delta_max_bytes = trace_delta_max_bytes
        # Upper bound on one answer_delta's text (0 = pass provider chunks
        # through unchanged). Provider stream chunks can be hundreds of
        # characters, which renders as jumpy block-appends in the UI; re-slicing
        # here smooths every consumer (SSE and one-shot) at the one
        # point all answer text flows through.
        self._delta_max = answer_delta_max_chars
        # Explicit parent for the spans `step` opens. Every tool call
        # already flows through `step`, so instrumenting here — rather than in
        # twelve tool bodies — puts every tool execution in the trace. The agent
        # runtime points this at the CURRENT ROUND's span so a tool nests under
        # the model call that chose it; the SDK runs each tool as its own asyncio
        # task, so the parent cannot be inferred from context and must be given.
        # None ⇒ nest under whatever span is current (correct for inline stages).
        self.span_parent: object | None = None
        # Monotonic per-request counter: the ORDER the agent invoked tools, which
        # a waterfall sorted by start time cannot show for concurrent calls.
        self._tool_seq = 0

    @asynccontextmanager
    async def step(
        self,
        kind: StepKind,
        summary: str | None = None,
        *,
        detail: str | None = None,
        tool: str | None = None,
    ):
        """Record + stream + LOG one unit of work: emits `started`, then `completed`
        (with duration) or `error` if the body raises. The structured log lines
        (inheriting the request_id contextvar) mirror the SSE trace into the pod logs,
        so the full stage-by-stage flow is reconstructable for any request — including
        the non-streaming path where SSE is a no-op.

        The user-facing LABEL is derived from ``kind`` (core/narration.phase_for),
        never passed in: a reader gets the same five phase names on every question,
        and no lane can invent a sixth. Callers supply the ``summary`` — the one
        specific sentence about this step.

        ``tool`` is the internal identifier of the work (an agent tool name). It
        goes to the LOG line only and is never streamed: the UI shows what is
        happening, operators get what ran.
        """
        label = phase_for(kind)
        step = new_step(label, kind, detail, summary)
        self._trace.steps.append(step)
        await self._emit(StepEvent(step=step))
        log_fields = {
            "stage": label, "kind": kind.value, "summary": summary,
            "detail": detail, "tool": tool, "step_id": step.id,
        }
        logger.debug("stage_start", **log_fields)
        started = time.monotonic()
        # One span per unit of work. A TOOL step opens no span here: the tool's
        # own span is created around the SDK invocation (agent/tools.py::_traced),
        # which is the only place the arguments and the result are both visible,
        # and it already spans this step plus the pool merge that follows it.
        # Spanning both would put two near-identical boxes in every waterfall.
        span_name = None if tool else f"stage.{kind.value}"
        span_attrs: dict[str, object] = {"step_kind": kind.value, "step_id": step.id}
        if summary:
            span_attrs["summary"] = summary
        with stage_span(span_name, parent=self.span_parent, **span_attrs):
            try:
                yield step
            except Exception as exc:
                step.status = StepStatus.ERROR
                step.duration_ms = int((time.monotonic() - started) * 1000)
                await self._emit(StepEvent(step=step))
                logger.error(
                    "stage_error", **log_fields, duration_ms=step.duration_ms,
                    error=str(exc), error_type=type(exc).__name__,
                )
                raise
        step.status = StepStatus.COMPLETED
        step.duration_ms = int((time.monotonic() - started) * 1000)
        await self._emit(StepEvent(step=step))
        logger.info("stage_done", **log_fields, duration_ms=step.duration_ms)

    async def graph_delta(
        self,
        entities: list[EntityHit] | None = None,
        facts: list[RelationFact] | None = None,
        sources: list[SourceDocument] | None = None,
        quotes: list[ChunkQuote] | None = None,
        claims: list[ClaimFact] | None = None,
    ) -> None:
        """Merge newly-seen graph elements into the trace and stream only the *new*
        nodes/edges so the UI canvas grows incrementally. Covers every node kind —
        entities + relationships (entity↔entity), documents, chunks (+ HAS_CHUNK /
        MENTIONS edges), and claims (+ ASSERTS/ABOUT / SUPPORTED_BY edges). Each
        merge_* dedupes internally, so passing the full set is safe (no double emit)."""
        new_nodes, new_edges = self._trace.merge(
            entities=entities, facts=facts, sources=sources, quotes=quotes, claims=claims,
        )
        if new_nodes or new_edges:
            chunks = _delta_chunks(new_nodes, new_edges, self._trace_delta_max_bytes)
            for chunk_nodes, chunk_edges in chunks:
                await self._emit(TraceDeltaEvent(nodes=chunk_nodes, edges=chunk_edges))
            logger.debug("graph_delta", new_nodes=len(new_nodes),
                         new_edges=len(new_edges), events=len(chunks))

    async def answer_delta(self, delta: str) -> None:
        if not delta:
            return
        if self._delta_max <= 0 or len(delta) <= self._delta_max:
            await self._emit(AnswerDeltaEvent(delta=delta))
            return
        for start in range(0, len(delta), self._delta_max):
            await self._emit(AnswerDeltaEvent(delta=delta[start : start + self._delta_max]))
