"""What reached the answer model, and which gate dropped the rest.

FIVE gates stand between the evidence pool and the model — cross-source dedup,
document trust rank, the count cap, the cumulative character budget and the
fragment floor — and none of them left a trace. The only way to see which one
dropped a given chunk was to attach an offline probe that parses the rendered
digest. Citation
resolution had the same gap: a dropped chip leaves no mark in the answer, which
is the whole point of dropping it.

Each test below pins the fact that makes one gate legible, because a span whose
attributes are wrong is worse than no span — it reads as evidence. Companion
suite: ``test_selection_spans.py``.

    uv run pytest tests/test_digest_spans.py -v
"""
from __future__ import annotations

import json

import pytest

from ai_graph_retrieval.core.models import (
    ChunkQuote,
    EntityHit,
    Evidence,
    SourceDocument,
)
from ai_graph_retrieval.synthesis.citations import resolve_citations
from ai_graph_retrieval.synthesis.composer import AnswerComposer
from tests.support import make_settings, span_attrs, span_output

pytest.importorskip("opentelemetry.sdk", reason="OTEL SDK required for span capture")


def _quote(chunk_id: str, graph: str, text: str, doc: str = "d1",
           title: str = "Doc One") -> ChunkQuote:
    return ChunkQuote(chunk_id=chunk_id, doc_id=doc, doc_title=title,
                      text=text, graph=graph)


def _two_docs() -> list[SourceDocument]:
    return [SourceDocument(doc_id="d1", title="Doc One", url="http://x/1"),
            SourceDocument(doc_id="d2", title="Doc Two", url="http://x/2")]


# ── stage.digest ────────────────────────────────────────────────────────────


def test_the_digest_span_reports_the_budget_it_spent(spans):
    """How much of the character budget the excerpts actually used, and how many
    were truncated to fit, is visible nowhere but this span."""
    settings = make_settings()
    settings.digest_quote_char_budget = 400
    settings.digest_min_excerpt_chars = 0     # measure the budget, not the floor
    # Spaced text on purpose: truncation lands on a whitespace boundary, and a
    # space-free chunk is deliberately kept WHOLE past the budget (so a URL is
    # never cut in half), which would measure the overshoot rather than the split.
    first = "graph " * 50                     # 300 chars
    second = "paper " * 50                    # 300 chars
    ev = Evidence(
        sources=_two_docs(),
        quotes=[
            _quote("g1", "wikipedia", first),
            _quote("a1", "arxiv", second, doc="d2", title="Doc Two"),
        ],
    )
    AnswerComposer(settings)._digest(ev, strict=False)

    at = span_attrs(spans, "stage.digest")
    assert at["quotes_pool"] == 2
    assert at["quote_char_budget"] == 400
    assert at["quote_chars_used"] <= 400
    assert at["excerpt_docs"] == 2
    assert at["excerpts_truncated"] == 1
    assert sorted(at["graphs"]) == ["arxiv", "wikipedia"]


def test_the_digest_span_reports_a_cap_that_bound(spans):
    """A count cap is INVISIBLE in the digest text — the section simply looks
    complete — so available-against-rendered is the only way to see it bind."""
    settings = make_settings()
    settings.digest_max_entities = 2
    ev = Evidence(entities=[EntityHit(name=f"P{i}", type="Person") for i in range(5)])
    AnswerComposer(settings)._digest(ev, strict=False)

    at = span_attrs(spans, "stage.digest")
    assert (at["entities_available"], at["entities_rendered"]) == (5, 2)
    # section sizes come from our own labels, never re-parsed out of the rendered
    # text — whose excerpt bodies carry newlines of their own
    assert set(json.loads(at["section_chars"])) == {"ENTITIES"}


def test_the_digest_span_says_when_the_budget_ran_out(spans):
    """The single most useful bit on the span: everything after the cut was
    retrieved, deduplicated, ranked and never shown."""
    settings = make_settings()
    settings.digest_quote_char_budget = 60
    settings.digest_min_excerpt_chars = 0
    ev = Evidence(
        sources=_two_docs()[:1],
        quotes=[_quote(f"g{i}", "wikipedia", "word " * 30) for i in range(4)],
    )
    AnswerComposer(settings)._digest(ev, strict=False)
    assert span_attrs(spans, "stage.digest")["budget_exhausted"] is True


def test_the_digest_output_carries_per_excerpt_detail(spans):
    """Counts say how much fit; the detail says WHAT fit — and `chars` against
    `chars_available` is the truncation, which the digest text shows for neither."""
    ev = Evidence(
        sources=_two_docs()[:1],
        quotes=[_quote("g1", "wikipedia", "text " * 40)],
    )
    AnswerComposer(make_settings())._digest(ev, strict=False)

    detail = span_output(spans, "stage.digest")["detail"]
    assert [r["doc"] for r in detail] == ["Doc One"]
    assert detail[0]["graph"] == "wikipedia"
    assert detail[0]["chars"] == detail[0]["chars_available"]      # rendered whole


# ── stage.citations ─────────────────────────────────────────────────────────


def test_the_citations_span_separates_the_three_chip_outcomes(spans):
    """Resolved, naming nothing real, and naming a document with no url. All three
    look identical in the shipped answer — the chip is simply gone."""
    sources = [
        SourceDocument(doc_id="d1", title="Leave Policy", url="http://x/leave"),
        SourceDocument(doc_id="d2", title="No Link Doc"),          # no url
    ]
    answer = ("Twenty days. [Leave Policy] Also [Leave Policy] again. "
              "[Invented Title] and [No Link Doc].")
    _, cited = resolve_citations(answer, sources)

    at = span_attrs(spans, "stage.citations")
    assert at["chips_resolved"] == 2            # the same document twice
    assert at["chips_unresolved"] == 1
    assert at["chips_without_url"] == 1
    assert at["citations"] == len(cited) == 1   # one number for the repeated doc
    assert at["marks"] == 4                     # the counters sum to what was written
    assert at["marks_per_citation"] == 2.0
    out = span_output(spans, "stage.citations")
    assert out["citations"] == [{"ref": 1, "title": "Leave Policy"}]
    assert out["unresolved"] == ["Invented Title"]


def test_the_citations_span_counts_self_written_links_apart_from_content_links(spans):
    sources = [SourceDocument(doc_id="d1", title="Leave Policy", url="http://x/leave")]
    answer = ("See the [Leave Policy](http://x/leave) for detail, "
              "and [an outside page](http://elsewhere/x).\n"
              "Sources: [Leave Policy](http://x/leave)")
    resolve_citations(answer, sources)

    at = span_attrs(spans, "stage.citations")
    assert at["links_normalised"] == 1       # mid-sentence: anchor text kept
    assert at["links_as_marker"] == 1        # on a Sources line: marker alone
    assert at["links_content"] == 1          # not a citation → verbatim
    assert at["chips_resolved"] == 0


def test_no_citations_span_when_there_is_nothing_to_resolve(spans):
    """The gate reports zero sources already; a second empty box would be noise."""
    resolve_citations("An answer with no sources.", [])
    assert not [s for s in spans.get_finished_spans() if s.name == "stage.citations"]


# ── the shape verdict is visible where it was APPLIED ──────────────
# The router logs the verdict it chose. That says a shape was decided, not that
# synthesis ran under it — and the field exists precisely so the same question
# comes back the same shape twice, which is a claim about the answer, not about
# the classifier. So the synthesize span carries what the call actually applied,
# beside the answer it produced.


def _shape_sc(captured: dict):
    """A StrategyContext-alike whose llm records the system prompt it was sent."""
    from contextlib import asynccontextmanager
    from types import SimpleNamespace

    class _Llm:
        async def stream_text(self, messages, max_tokens=None, fast=False,
                              usage=None, **_kwargs):
            captured["system"] = messages[0]["content"]
            yield "A long enough answer to clear the degenerate guard."

    class _Progress:
        @asynccontextmanager
        async def step(self, *_a, **_k):
            yield SimpleNamespace()

        async def answer_delta(self, _token):
            pass

    return SimpleNamespace(
        question="What shipped over the last three releases?",
        history=[], progress=_Progress(), llm=_Llm(), usage=SimpleNamespace(),
        ctx=SimpleNamespace(log_fields=lambda: {}),
    )


def test_the_synthesize_span_records_the_shape_it_ran_under(spans):
    """A shape that reached the prompt has to be legible on the span, or checking
    consistency means inferring the decision back out of the finished answer."""
    import asyncio

    from ai_graph_retrieval.observability.tracing import stage_span
    from ai_graph_retrieval.retrieval.base import AnswerShape, Mode

    captured: dict = {}
    with stage_span("stage.synthesize"):
        asyncio.run(AnswerComposer(make_settings()).compose(
            _shape_sc(captured), Evidence(), mode=Mode.SEARCH,
            answer_shape=AnswerShape.TIMELINE,
        ))

    attrs = span_attrs(spans, "stage.synthesize")
    assert attrs["answer_shape"] == "timeline"
    assert attrs["shape_applied"] is True
    # ...and it really did reach the model, so the attribute is not a claim the
    # prompt failed to honour.
    assert "ANSWER SHAPE: this question's shape is TIMELINE" in captured["system"]


def test_the_span_separates_no_verdict_from_a_verdict_switched_off(spans):
    """Both leave the STRUCTURE rules to decide, so both produce the same answer
    shape — and they are different faults. One means the classifier had no
    opinion; the other means the environment is ignoring every opinion it has.
    """
    import asyncio

    from ai_graph_retrieval.observability.tracing import stage_span
    from ai_graph_retrieval.retrieval.base import AnswerShape, Mode

    captured: dict = {}
    with stage_span("stage.synthesize"):
        asyncio.run(AnswerComposer(make_settings()).compose(
            _shape_sc(captured), Evidence(), mode=Mode.SEARCH, answer_shape=None,
        ))
    attrs = span_attrs(spans, "stage.synthesize")
    # OTEL drops a None attribute rather than recording a null, so "no verdict"
    # reads as an ABSENT key — which is why shape_applied is asserted too: absence
    # alone would also be what a span that forgot the field looks like.
    assert "answer_shape" not in attrs
    assert attrs["shape_applied"] is False
    assert "ANSWER SHAPE:" not in captured["system"]

    spans.clear()
    off = make_settings()
    off.answer_shape_enabled = False
    with stage_span("stage.synthesize"):
        asyncio.run(AnswerComposer(off).compose(
            _shape_sc(captured), Evidence(), mode=Mode.SEARCH,
            answer_shape=AnswerShape.TABLE,
        ))
    attrs = span_attrs(spans, "stage.synthesize")
    # The verdict is still recorded — that is what distinguishes the two — but it
    # was not applied, and the prompt confirms it.
    assert attrs["answer_shape"] == "table"
    assert attrs["shape_applied"] is False
    assert "ANSWER SHAPE:" not in captured["system"]
