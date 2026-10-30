"""The trace is a WATERFALL, not a flat list.

Before this, a Langfuse trace showed only model calls — one root span with every
generation as a sibling — while 18 tool executions, ~21 graph reads and the whole
quality/synthesis pipeline produced no observation at all, and the root's 92.5s
exceeded the sum of its children by ~30s of unattributable time (measured run,
2026-08-20). These tests pin the SHAPE that fixes it: who is whose parent.

Shape is the thing worth testing here. Attribute presence is easy to assert and
easy to keep passing while the hierarchy silently flattens — which is exactly the
failure mode of tracing spans across asyncio tasks, where OTEL's contextvar
propagation does NOT reach a task started elsewhere (the Agents SDK runs each
tool call as its own task). So every test below asserts parent/child links from a
real in-memory exporter, not just that spans exist.

    uv run pytest tests/test_trace_hierarchy.py -v
"""
from __future__ import annotations

import asyncio

import pytest

from ai_graph_retrieval.core.models import GraphTraceBuilder, StepKind
from ai_graph_retrieval.core.progress import ProgressReporter, null_emit
from ai_graph_retrieval.observability import tracing

pytest.importorskip("opentelemetry.sdk", reason="OTEL SDK required for span capture")


def _by_name(exporter) -> dict:
    return {s.name: s for s in exporter.get_finished_spans()}


def _parent_of(exporter, name: str):
    """The span NAME of `name`'s parent, resolved through span ids — the only
    honest way to assert nesting (a span can carry the right attributes and still
    hang off the wrong parent)."""
    spans = exporter.get_finished_spans()
    child = next(s for s in spans if s.name == name)
    if child.parent is None:
        return None
    return next(
        (s.name for s in spans if s.context.span_id == child.parent.span_id), "<unknown>"
    )


# ── stage spans ──────────────────────────────────────────────────────────────


def test_a_stage_nests_under_the_request_root(spans):
    with tracing.retrieve_span("retrieve/super_research"), \
            tracing.stage_span("stage.synthesize"):
        pass
    assert _parent_of(spans, "stage.synthesize") == "retrieve/super_research"


def test_a_stage_records_its_failure_and_re_raises(spans):
    """Tracing observes failures, never swallows them: the exception must reach
    the caller AND leave the span marked ERROR."""
    from opentelemetry.trace import StatusCode

    with pytest.raises(RuntimeError, match="graph down"), tracing.stage_span("stage.traverse"):
        raise RuntimeError("graph down")
    span = _by_name(spans)["stage.traverse"]
    assert span.status.status_code is StatusCode.ERROR
    assert [e.name for e in span.events] == ["exception"]


def test_span_helpers_stay_no_ops_when_tracing_is_off(monkeypatch):
    """The fail-open contract every tracing helper shares: with tracing
    unavailable the block still runs and yields None, so no caller needs a
    branch."""
    monkeypatch.setattr(tracing, "_AVAILABLE", False)
    with tracing.stage_span("stage.x") as span:
        assert span is None
    assert tracing.start_stage_span("stage.y") is None
    tracing.record_error(None, RuntimeError("x"))  # must not raise


# ── the cross-task parenting that motivated the explicit-parent API ──────────


def _fake_tool(name: str, result: str = "[]", *, fail: bool = False):
    """A stand-in for the SDK's FunctionTool: the wrapper only needs a name and
    an ``on_invoke_tool`` taking (ctx, args_json)."""
    from dataclasses import dataclass

    # A dataclass because the wrapper rebuilds the tool with dataclasses.replace,
    # exactly as it does with the SDK's FunctionTool.
    @dataclass
    class _Tool:
        name: str
        on_invoke_tool: object

    async def _invoke(_ctx, _args):
        if fail:
            raise RuntimeError("neo4j refused")
        return result

    return _Tool(name=name, on_invoke_tool=_invoke)


def _ctx_with(reporter: ProgressReporter):
    from types import SimpleNamespace
    return SimpleNamespace(context=SimpleNamespace(progress=reporter))


def test_a_tool_span_nests_under_its_round_even_from_another_task(spans):
    """THE regression the explicit-parent API exists for. The Agents SDK runs
    each tool call as its own asyncio task; OTEL copies the context at task
    creation, so a task started outside the round's scope inherits the ROOT
    context and the tool would silently re-parent there — flattening the
    waterfall while every attribute still looked right."""
    from ai_graph_retrieval.agent.tools import _traced

    reporter = ProgressReporter(null_emit, GraphTraceBuilder())
    tool = _traced(_fake_tool("find_people"))

    async def scenario():
        with tracing.retrieve_span("retrieve/super_research"):
            round_span = tracing.start_stage_span("agent.round.1")
            reporter.span_parent = round_span
            # A separate task, exactly as the SDK invokes a tool:
            await asyncio.create_task(
                tool.on_invoke_tool(_ctx_with(reporter), '{"query": "Owen Price"}')
            )
            tracing.end_span(round_span)

    asyncio.run(scenario())
    assert _parent_of(spans, "tool.find_people") == "agent.round.1"
    assert _parent_of(spans, "agent.round.1") == "retrieve/super_research"


def test_concurrent_tool_calls_in_one_round_all_nest_under_that_round(spans):
    """Independent tool calls in one round run concurrently — each must still
    land under the round, not race into each other or onto the root."""
    from ai_graph_retrieval.agent.tools import _traced

    reporter = ProgressReporter(null_emit, GraphTraceBuilder())
    claims = _traced(_fake_tool("get_claims"))
    expand = _traced(_fake_tool("expand_entity"))

    async def scenario():
        with tracing.retrieve_span("retrieve/super_research"):
            round_span = tracing.start_stage_span("agent.round.2")
            reporter.span_parent = round_span
            ctx = _ctx_with(reporter)
            await asyncio.gather(
                claims.on_invoke_tool(ctx, '{"entity_name": "Max Fields"}'),
                expand.on_invoke_tool(ctx, '{"entity_name": "Owen Price"}'),
            )
            tracing.end_span(round_span)

    asyncio.run(scenario())
    for tool in ("tool.get_claims", "tool.expand_entity"):
        assert _parent_of(spans, tool) == "agent.round.2"


# ── what a tool span carries ─────────────────────────────────────────────────


def test_a_tool_span_carries_the_arguments_and_the_result(spans):
    """The requirement in one test: "which tool did the agent call, with what
    input, and what did it return?" answerable from the span alone — not by
    pairing a round's output with the next round's input."""
    from ai_graph_retrieval.agent.tools import _traced

    reporter = ProgressReporter(null_emit, GraphTraceBuilder())
    tool = _traced(_fake_tool("find_people", result='[{"name": "Max Fields"}]'))
    asyncio.run(tool.on_invoke_tool(_ctx_with(reporter), '{"query": "data science"}'))

    span = _by_name(spans)["tool.find_people"]
    assert span.attributes["tool_name"] == "find_people"
    assert span.attributes["tool_call_order"] == 1
    assert "data science" in span.attributes["langfuse.observation.input"]
    assert "Max Fields" in span.attributes["langfuse.observation.output"]


def test_tool_call_order_is_the_agents_invocation_order(spans):
    """A waterfall sorted by start time cannot show ordering for concurrent
    calls, so the order the agent asked for them is recorded explicitly."""
    from ai_graph_retrieval.agent.tools import _traced

    reporter = ProgressReporter(null_emit, GraphTraceBuilder())
    ctx = _ctx_with(reporter)
    for name in ("resolve_entities", "get_claims", "get_claims"):
        asyncio.run(_traced(_fake_tool(name)).on_invoke_tool(ctx, "{}"))
    orders = [s.attributes["tool_call_order"] for s in spans.get_finished_spans()]
    assert orders == [1, 2, 3]


def test_a_failing_tool_marks_its_span_and_propagates(spans):
    """A tool that raises must be visibly failed in the trace — the case where
    "the trace looked fine" was previously most misleading."""
    from opentelemetry.trace import StatusCode

    from ai_graph_retrieval.agent.tools import _traced

    reporter = ProgressReporter(null_emit, GraphTraceBuilder())
    tool = _traced(_fake_tool("get_claims", fail=True))
    with pytest.raises(RuntimeError, match="neo4j refused"):
        asyncio.run(tool.on_invoke_tool(_ctx_with(reporter), "{}"))
    assert _by_name(spans)["tool.get_claims"].status.status_code is StatusCode.ERROR


def test_a_progress_step_does_not_duplicate_the_tool_span(spans):
    """Each tool opens its own span around the SDK invocation, and its progress
    step sits inside that call — so the step must NOT open a second, nearly
    identical span, or every waterfall shows each tool twice."""
    reporter = ProgressReporter(null_emit, GraphTraceBuilder())

    async def scenario():
        async with reporter.step(StepKind.TOOL_CALL, "Finding people", tool="find_people"):
            pass
        async with reporter.step(StepKind.SYNTHESIZE, "Putting the answer together"):
            pass

    asyncio.run(scenario())
    names = list(_by_name(spans))
    assert names == [f"stage.{StepKind.SYNTHESIZE.value}"]   # the tool step spanned nothing


# ── request-level failure recording ──────────────────────────────────────────


def test_a_clarification_pause_is_not_recorded_as_a_failure(spans):
    """A mid-research pause LEAVES the request by raising, so OTEL's automatic
    exception recording would stamp it ERROR and make failure counts include
    every paused turn. The root span disables that recording precisely so the
    orchestrator can decide which exits are failures."""
    from opentelemetry.trace import StatusCode

    from ai_graph_retrieval.clarification.models import MidResearchClarification

    with pytest.raises(MidResearchClarification):
        with tracing.retrieve_span("retrieve/super_research"):
            raise MidResearchClarification(None)
    root = _by_name(spans)["retrieve/super_research"]
    assert root.status.status_code is not StatusCode.ERROR
    assert not root.events           # no exception event either


def test_a_failed_request_is_marked_failed_on_the_root_span(spans):
    """Previously a 502'd request rendered as a successful trace that merely had
    no output, so failures were uncountable from traces."""
    from opentelemetry.trace import StatusCode

    from ai_graph_retrieval.core.exceptions import GraphQueryError

    exc = GraphQueryError()
    with tracing.retrieve_span("retrieve/search") as span:
        tracing.record_error(span, exc, code=exc.code)
    root = _by_name(spans)["retrieve/search"]
    assert root.status.status_code is StatusCode.ERROR
    assert root.attributes["error.code"] == exc.code


# ── phase 2: store spans ─────────────────────────────────────────────────────


def test_a_store_read_is_its_own_span_under_the_tool_that_asked(spans):
    """The retrieval half was entirely absent: ~21 graph reads, the Qdrant
    searches produced no observation, which is why the
    root span's duration exceeded the sum of its children by ~30s. A store span
    nested under its tool is what makes that time attributable."""
    with tracing.stage_span("tool.get_claims") as tool_span, tracing.stage_span(
        "neo4j.FETCH_CLAIMS", parent=tool_span, store="neo4j",
        query_name="FETCH_CLAIMS", databases=["wiki-2rdv", "arxiv-nw28"],
    ) as read:
        tracing.set_span_attributes(read, rows=138)

    span = _by_name(spans)["neo4j.FETCH_CLAIMS"]
    assert _parent_of(spans, "neo4j.FETCH_CLAIMS") == "tool.get_claims"
    assert span.attributes["store"] == "neo4j"
    assert span.attributes["rows"] == 138
    assert len(span.attributes["databases"]) == 2


# ── phase 3: Langfuse-native identity + scores ───────────────────────────────


def test_the_trace_carries_langfuse_session_user_and_tags(spans):
    """Sent as Langfuse's OWN fields, not generic attributes: the UI groups a
    paused-then-resumed conversation by SESSION and filters the trace list by
    TAGS. As plain attributes the same facts were unusable for either."""
    with tracing.retrieve_span("retrieve/search") as span:
        tracing.record_trace_identity(
            span, session_id="conv-123", user_id="tenant-abc",
            tags=["mode:search", "outcome:ok", "intent:entity"],
        )
    attrs = _by_name(spans)["retrieve/search"].attributes
    assert attrs["langfuse.session.id"] == "conv-123"
    assert attrs["langfuse.user.id"] == "tenant-abc"
    import json as _json
    assert _json.loads(attrs["langfuse.trace.tags"]) == [
        "intent:entity", "mode:search", "outcome:ok",
    ]


def test_quality_signals_become_comparable_scores(spans):
    """Scores are what turn "% of runs shipping zero citations" into a chart.
    Booleans become 0/1 so they aggregate; anything non-numeric is dropped
    rather than stringified — a score that cannot be compared is not a score."""
    with tracing.retrieve_span("retrieve/super_research") as span:
        tracing.record_scores(span, {
            "evidence_found": True, "citations": 15, "partial": False,
            "cost_usd": 0.3583, "strategy": "deep_research",
        })
    attrs = _by_name(spans)["retrieve/super_research"].attributes
    assert attrs["langfuse.score.evidence_found"] == 1
    assert attrs["langfuse.score.citations"] == 15
    assert attrs["langfuse.score.partial"] == 0
    assert attrs["langfuse.score.cost_usd"] == pytest.approx(0.3583)
    assert "langfuse.score.strategy" not in attrs      # not comparable, not a score
    # the score is named for what it MEASURES. `answer_found` claimed a
    # verdict on the prose that the flag cannot make — it read True on 12/12
    # answers that declined — so a dashboard keyed on that name was wrong.
    assert "langfuse.score.answer_found" not in attrs


# ── correlation id + data sources ─────────────────────────────────


def test_every_span_carries_the_requests_correlation_id(spans):
    """One id across the whole waterfall, so any single observation identifies
    the query it belongs to — and joins the trace to the logs, which already
    carry the same request id."""
    tracing.set_correlation_id("req-abc-123")
    try:
        with tracing.retrieve_span("retrieve/search") as root:
            with tracing.stage_span("stage.seed", parent=root) as seed:
                with tracing.stage_span("neo4j.RESOLVE", parent=seed):
                    pass
            gen = tracing.start_llm_span("llm.chat", parent=root)
            tracing.end_span(gen)
    finally:
        tracing.set_correlation_id(None)

    ids = {s.name: s.attributes.get("correlation_id") for s in spans.get_finished_spans()}
    assert set(ids) == {"retrieve/search", "stage.seed", "neo4j.RESOLVE", "llm.chat"}
    assert set(ids.values()) == {"req-abc-123"}   # every span, one id


def test_the_correlation_id_is_per_query_not_global(spans):
    """A new query must get a new id, or two questions asked seconds apart are
    indistinguishable in the trace list."""
    for cid in ("req-1", "req-2"):
        tracing.set_correlation_id(cid)
        with tracing.retrieve_span(f"retrieve/{cid}"):
            pass
    tracing.set_correlation_id(None)
    ids = {s.attributes.get("correlation_id") for s in spans.get_finished_spans()}
    assert ids == {"req-1", "req-2"}


def test_spans_created_in_another_task_inherit_the_correlation_id(spans):
    """Tool calls run as their own asyncio tasks. Unlike the parent SPAN (which
    must be passed explicitly), a contextvar IS copied into a new task — this
    pins that, because losing it would leave exactly the deep spans uncorrelated
    that most need to be."""
    async def scenario():
        tracing.set_correlation_id("req-task")
        with tracing.retrieve_span("retrieve/super_research") as root:
            async def tool():
                with tracing.stage_span("tool.get_claims", parent=root):
                    pass
            await asyncio.create_task(tool())

    asyncio.run(scenario())
    tracing.set_correlation_id(None)
    tool_span = _by_name(spans)["tool.get_claims"]
    assert tool_span.attributes["correlation_id"] == "req-task"


def test_a_tool_span_shows_which_graph_and_which_collection_it_used(spans):
    """The data source belongs NEXT TO the tool, not only on a child span: the
    question is "which tool used what?", and annotate_current_span is how a
    repository that has no reference to the tool span still answers it."""
    with tracing.stage_span("tool.resolve_entities"):
        tracing.annotate_current_span(graphs_queried=["wikipedia", "arxiv"])
        tracing.annotate_current_span(qdrant_collection="wiki-entities-2rdv")
        tracing.annotate_current_span(graphs_with_rows=["wikipedia"])

    attrs = _by_name(spans)["tool.resolve_entities"].attributes
    assert list(attrs["graphs_queried"]) == ["wikipedia", "arxiv"]
    assert attrs["qdrant_collection"] == "wiki-entities-2rdv"
    # Queried both, only one returned rows — a real finding (ACL filter, missing
    # collection, run mismatch) that a row count alone cannot express.
    assert list(attrs["graphs_with_rows"]) == ["wikipedia"]


def test_annotating_outside_any_span_is_harmless(spans):
    """Store code runs in paths with no tool span (boot verification, the
    post-loop top-ups). Annotating there must be a no-op, not a crash."""
    tracing.annotate_current_span(graphs_queried=["wikipedia"])
    assert not spans.get_finished_spans()


def test_an_api_call_span_keeps_input_and_output_even_when_truncated(spans):
    """POST payloads must be visible, not omitted: a truncated body still
    answers "what did we send and what came back?"."""
    with tracing.stage_span("http.post", store="http") as span:
        tracing.record_io(span, input='{"search_term": "who reports to Jane Doe"}',
                          limit=6000)
        tracing.record_io(span, output="x" * 20_000, limit=6000)

    attrs = _by_name(spans)["http.post"].attributes
    assert "Jane Doe" in attrs["langfuse.observation.input"]
    assert len(attrs["langfuse.observation.output"]) == 6000     # truncated, present


# ── post-loop top-ups get a stage ─────────────────────────────────


def test_both_post_loop_topups_nest_their_reads_under_a_stage(spans):
    """A live trace showed `neo4j.SUPPORTING_CHUNKS` — 9.4s, 11% of the request —
    and `neo4j.PEOPLE_PROFILES` hanging straight off the root with no stage
    context. Stage spans come from ProgressReporter.step, and of the three
    post-loop top-ups only enrich_claims opens a step; the other two ran
    unspanned. Asserted through real parent ids, so a span carrying the right
    name under the wrong parent still fails."""
    import asyncio
    from types import SimpleNamespace

    from ai_graph_retrieval.core.models import ChunkQuote, EntityHit, EvidencePool
    from ai_graph_retrieval.retrieval import deep_research
    from ai_graph_retrieval.retrieval.base import Mode
    from tests.support import make_profile, make_settings

    class _Ctx:
        request_id = "req-topup"
        def log_fields(self): return {}

    class _Repo:
        # The graph-chunk top-up now RANKS its candidates before the budget cut
        # (score_graph_chunks), and that pass asks the repository for its vector
        # store; None is the documented skip — the cut falls back to read order.
        sources = ()
        vector_store = None

        async def fetch_people_profiles(self, ctx, names):
            with tracing.stage_span("neo4j.PEOPLE_PROFILES"):
                pass
            return [EntityHit(entity_id="e1", name="Max Fields", type="Person")], []

        async def supporting_chunks(self, ctx, entity_names):
            with tracing.stage_span("neo4j.SUPPORTING_CHUNKS"):
                pass
            return [ChunkQuote(chunk_id="c1", doc_id="d1", doc_title="Doc",
                               text="a supporting excerpt")]

    pool = EvidencePool()
    pool.add_entities([EntityHit(entity_id="e0", name="Owen Price", type="Person")])
    sc = SimpleNamespace(
        ctx=_Ctx(), evidence=pool, repo=_Repo(), settings=make_settings(),
        profile=make_profile(Mode.SUPER_RESEARCH),
        progress=SimpleNamespace(span_parent=None,
                                 graph_delta=lambda **k: asyncio.sleep(0)),
    )
    with tracing.retrieve_span("retrieve/super_research"):
        asyncio.run(deep_research.enrich_profiles(sc))
        asyncio.run(deep_research.enrich_graph_chunks(sc))

    names = [s.name for s in spans.get_finished_spans()]
    assert "stage.profile_topup" in names, names
    assert "stage.graph_chunk_topup" in names, names
    # The reads sit UNDER their stage, not beside it at the root.
    assert _parent_of(spans, "neo4j.PEOPLE_PROFILES") == "stage.profile_topup"
    assert _parent_of(spans, "neo4j.SUPPORTING_CHUNKS") == "stage.graph_chunk_topup"
