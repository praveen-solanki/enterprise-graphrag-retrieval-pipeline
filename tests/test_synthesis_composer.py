"""Tests for AnswerComposer's degenerate-output guard (FINDINGS F1).

Rarely the synthesis model stops cleanly after emitting almost nothing (e.g. a lone
'#'). That's neither an exception nor a length-truncation, so stream_text can't catch
it — compose() must detect the near-empty result and retry once.

    uv run pytest tests/test_synthesis_composer.py -v
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

from ai_graph_retrieval.core.models import Evidence, SourceDocument
from ai_graph_retrieval.retrieval.base import Mode
from ai_graph_retrieval.synthesis.composer import AnswerComposer, _is_degenerate
from tests.support import make_settings

_MIN = 12  # mirrors the synthesis_min_answer_chars default


def test_is_degenerate():
    assert _is_degenerate("#", _MIN)
    assert _is_degenerate("##  \n\n-  ", _MIN)   # markdown scaffold, no content
    assert _is_degenerate("", _MIN)
    assert not _is_degenerate("The Data Engineering Team owns it.", _MIN)


def test_zero_threshold_disables_the_guard():
    assert not _is_degenerate("#", 0)


class _Progress:
    @asynccontextmanager
    async def step(self, *_a, **_k):
        yield SimpleNamespace()

    async def answer_delta(self, _token):  # streamed tokens (no-op in the test)
        pass


def _fake_sc(stream_outputs: list[str]):
    """Build a minimal StrategyContext-like object; stream_text returns the next
    output (streamed char-by-char) on each successive call."""
    calls = {"n": 0}

    class _Llm:
        async def stream_text(self, _messages, max_tokens=None, fast=False, usage=None, **_kwargs):
            calls["fast"] = fast
            out = stream_outputs[min(calls["n"], len(stream_outputs) - 1)]
            calls["n"] += 1
            for ch in out:
                yield ch

    sc = SimpleNamespace(
        question="Where do I look for partner info on an account?",
        history=[], progress=_Progress(), llm=_Llm(), usage=SimpleNamespace(),
        ctx=SimpleNamespace(log_fields=lambda: {}),
    )
    return sc, calls


def _settings(**overrides):
    """Real Settings, not a hand-listed stub.

    The stub used to name only the handful of fields the composer read, so promoting a
    digest cap into Settings broke every test here with an AttributeError instead of
    exercising the new default. Real Settings can never drift from what the composer
    reads.

    ``overrides`` reach Settings unchanged, for the tests that exercise a knob's
    OFF path (a kill switch nothing ever flips is a kill switch nobody can trust).
    """
    return make_settings(**overrides)


def test_the_set_completeness_rules_are_always_present_not_lane_selected():
    """There is no enumeration lane. The rules that make a set answer complete used
    to arrive in a separate block chosen by the entry classifier's verdict — and
    that verdict decided the SAME question differently between runs (measured
    2026-08-27: one query took the wide lane on one run and the narrow one on the
    next, rendering 9 excerpts against 3). They are now part of the base prompt on
    every request, and whether the question asks for a set is the answer model's
    reading of it."""
    captured = {}

    class _Llm:
        async def stream_text(self, messages, max_tokens=None, fast=False, usage=None, **_kwargs):
            captured["system"] = messages[0]["content"]
            yield "A long enough answer for the degenerate-answer guard to pass."

    for mode in (Mode.SEARCH, Mode.SUPER_RESEARCH):
        sc, _ = _fake_sc(["unused"])
        sc.llm = _Llm()
        asyncio.run(AnswerComposer(_settings()).compose(sc, Evidence(), mode=mode))
        assert "WHEN THE QUESTION ASKS FOR A SET" in captured["system"], mode
        assert "AGGREGATION MODE" not in captured["system"], mode
        # and the rule that stops it turning a single-fact question into a list
        assert "A question that asks for ONE thing gets one answer" in captured["system"]


def test_compose_retries_on_degenerate_answer():
    sc, calls = _fake_sc([
        "#",  # first pass degenerates
        "There are two primary places to look: the Salesforce Account record and the "
        "Portfolio view.",
    ])
    answer = asyncio.run(AnswerComposer(_settings()).compose(sc, Evidence(), mode=Mode.SEARCH)).text
    assert calls["n"] == 2                 # retried exactly once
    assert "Salesforce" in answer          # kept the good retry


def test_compose_no_retry_when_answer_is_good():
    good = "Partner info lives on the Salesforce Account record under Partner Details."
    sc, calls = _fake_sc([good])
    answer = asyncio.run(AnswerComposer(_settings()).compose(sc, Evidence(), mode=Mode.SEARCH)).text
    assert calls["n"] == 1                 # no wasted retry
    assert answer == good


def test_every_cited_document_becomes_a_placeholder_with_no_links_or_tail():
    # Citations live ONLY inline as [n] placeholders: every chip the model placed
    # gets its number, none are truncated, the answer text carries no citation
    # links, and no trailing Sources section exists.
    srcs = [SourceDocument(doc_id=f"d{i}", title=f"Doc {i}", url=f"https://x/{i}")
            for i in range(12)]
    chips = " ".join(f"Fact {i}. [Doc {i}]" for i in range(12))
    sc, _ = _fake_sc([chips])
    out = asyncio.run(AnswerComposer(_settings()).compose(
        sc, Evidence(sources=srcs), mode=Mode.SEARCH))
    assert len(out.citations) == 12
    for i in range(12):
        assert f"Fact {i}. [{i + 1}]" in out.text        # placeholder, in place
        assert out.citations[i].doc_id == f"d{i}"        # citations[n-1] ↔ [n]
    assert "https://x/" not in out.text                  # urls only in the payload
    assert "## Sources" not in out.text


def test_no_chips_means_no_citations_and_no_fallback_list():
    # When the model cites nothing, nothing is guessed: the payload's citations
    # are empty and the answer text is exactly what the model wrote.
    srcs = [SourceDocument(doc_id=f"d{i}", title=f"Doc {i}", url=f"https://x/{i}")
            for i in range(12)]
    sc, _ = _fake_sc(["An answer with no citation chips at all."])
    out = asyncio.run(AnswerComposer(_settings()).compose(
        sc, Evidence(sources=srcs), mode=Mode.SEARCH))
    assert out.citations == []
    assert out.text == "An answer with no citation chips at all."


def test_chip_to_urlless_document_is_dropped_not_rendered_dead():
    # A cited document with no permalink cannot be resolved — the chip is removed
    # rather than shipped as dead text, and it never reaches citations.
    srcs = [SourceDocument(doc_id="d1", title="Linked Doc", url="https://x/1"),
            SourceDocument(doc_id="d2", title="Unlinked Doc")]
    sc, _ = _fake_sc(["Fact A. [Linked Doc] Fact B. [Unlinked Doc]"])
    out = asyncio.run(AnswerComposer(_settings()).compose(
        sc, Evidence(sources=srcs), mode=Mode.SEARCH))
    assert "Fact A. [1]" in out.text
    assert "Unlinked Doc" not in out.text
    assert [c.doc_id for c in out.citations] == ["d1"]


def test_compose_keeps_first_when_retry_also_degenerate():
    # both passes degenerate → return the (still-degenerate) first; never raise.
    sc, calls = _fake_sc(["#", "-"])
    answer = asyncio.run(AnswerComposer(_settings()).compose(sc, Evidence(), mode=Mode.SEARCH)).text
    assert calls["n"] == 2
    assert answer == "#"


def test_excerpts_lead_with_the_top_ranked_source_even_across_id_forms():
    # A quote carrying only doc_title must still match its source when that source
    # carries a doc_id (and vice-versa) — otherwise a title-only quote sank to the
    # tail and the capped excerpt window dropped the best source's text.
    from ai_graph_retrieval.core.models import ChunkQuote, SourceDocument
    ev = Evidence(
        sources=[SourceDocument(doc_id="d1", title="Top Doc"),
                 SourceDocument(doc_id="d2", title="Other Doc")],
        quotes=[ChunkQuote(chunk_id="c2", doc_id="d2", doc_title="Other Doc", text="second"),
                ChunkQuote(chunk_id="c1", doc_id=None, doc_title="Top Doc", text="first")],
    )
    digest = AnswerComposer(_settings())._digest(ev, strict=False)
    excerpts = digest[digest.index("EXCERPTS:"):]
    assert excerpts.index("first") < excerpts.index("second")


def test_quote_ordering_is_deterministic_and_stable():
    # Quote->source matching must be reproducible across identical requests, and
    # quotes from the SAME source must keep their input order (stable sort).
    from ai_graph_retrieval.core.models import ChunkQuote, SourceDocument
    ev = Evidence(
        sources=[SourceDocument(doc_id="d1", title="First"),
                 SourceDocument(doc_id="d2", title="Second")],
        quotes=[ChunkQuote(chunk_id="c3", doc_id="d2", doc_title="Second", text="beta"),
                ChunkQuote(chunk_id="c1", doc_id="d1", doc_title="First", text="alpha1"),
                ChunkQuote(chunk_id="c2", doc_id="d1", doc_title="First", text="alpha2")],
    )
    c = AnswerComposer(_settings())
    first = c._digest(ev, strict=False)
    assert first == c._digest(ev, strict=False)          # deterministic
    ex = first[first.index("EXCERPTS:"):]
    # The top-ranked source still leads, and a source's own quotes still keep their
    # relative input order — but the budget now goes round-robin over documents, so
    # d2's single quote sits between d1's two rather than after both.
    assert ex.index("alpha1") < ex.index("beta") < ex.index("alpha2")

    # …and with the allocation off, the greedy fill this test originally pinned:
    # every quote of the top-ranked source first.
    greedy = _settings()
    greedy.digest_round_robin_enabled = False
    ex_greedy = AnswerComposer(greedy)._digest(ev, strict=False)
    ex_greedy = ex_greedy[ex_greedy.index("EXCERPTS:"):]
    assert ex_greedy.index("alpha1") < ex_greedy.index("alpha2") < ex_greedy.index("beta")


def test_quote_with_unknown_source_sorts_last_without_dropping():
    from ai_graph_retrieval.core.models import ChunkQuote, SourceDocument
    ev = Evidence(
        sources=[SourceDocument(doc_id="d1", title="Known")],
        quotes=[ChunkQuote(chunk_id="cx", doc_id="zz", doc_title="Orphan", text="orphan"),
                ChunkQuote(chunk_id="c1", doc_id="d1", doc_title="Known", text="known")],
    )
    ex = AnswerComposer(_settings())._digest(ev, strict=False)
    ex = ex[ex.index("EXCERPTS:"):]
    assert ex.index("known") < ex.index("orphan")        # ranked first
    assert "orphan" in ex                                # but never dropped


# ── the answer model per mode: SEARCH is fast-tier ──────────────────────────


def test_search_composes_on_the_fast_tier():
    """SEARCH synthesizes on the fast tier for cost: its digest is the biggest
    prompt in the service. Super Research keeps the heavy model for the deepest
    report in the service."""
    from ai_graph_retrieval.retrieval.base import Mode

    for mode, want in [
        (Mode.SEARCH, True),
        (Mode.SUPER_RESEARCH, False),
    ]:
        sc, calls = _fake_sc(["A long enough answer for the degenerate guard."])
        asyncio.run(AnswerComposer(_settings()).compose(sc, Evidence(), mode=mode))
        assert calls["fast"] is want, mode


def test_the_degenerate_retry_stays_on_the_same_tier():
    from ai_graph_retrieval.retrieval.base import Mode

    sc, calls = _fake_sc(["#", "A real answer, long enough to pass the guard."])
    asyncio.run(AnswerComposer(_settings()).compose(sc, Evidence(), mode=Mode.SEARCH))
    assert calls["n"] == 2 and calls["fast"] is True


# ── cross-source quote dedup at the digest ────────────────────────


def test_graph_only_digests_are_untouched_by_the_dedup():
    """The digest applies no text dedup of its own — even for text that WOULD
    dedupe. Two graph chunks with
    overlapping text are corroboration the pool already vetted by chunk id."""
    from ai_graph_retrieval.core.models import ChunkQuote

    ev = Evidence(quotes=[
        ChunkQuote(chunk_id="c1", text="The policy grants 24 days.", graph="wikipedia"),
        ChunkQuote(chunk_id="c2", text="The policy grants 24 days.", graph="arxiv"),
    ])
    digest = AnswerComposer(_settings())._digest(ev, strict=False)
    assert digest.count("The policy grants 24 days.") == 2


def test_an_asked_window_is_stated_to_the_model_and_prunes_nothing():
    """The asked window is INPUT for the model's reasoning, never a filter
    : deterministic digest pruning excluded
    every document about a FUTURE window — anything describing September is
    necessarily dated before September — so no evidence is dropped by date.
    The classifier-resolved window is still STATED under the question (a
    fast-tier model doing its own date arithmetic got it wrong); which items
    fall inside it is the model's call under the prompt's time-scope rules."""
    from ai_graph_retrieval.core.models import ChunkQuote, SourceDocument

    ev = Evidence(
        sources=[
            SourceDocument(doc_id="cal", title="2026 Holiday Calendar",
                           date="2026-07-29"),
        ],
        quotes=[
            ChunkQuote(chunk_id="q1", doc_id="cal",
                       doc_title="2026 Holiday Calendar",
                       text="Ganesh Chaturthi - September 14, 2026"),
        ],
    )
    captured = {}

    class _Llm:
        async def stream_text(self, messages, max_tokens=None, fast=False, usage=None, **_kwargs):
            captured["user"] = messages[1]["content"]
            yield "A long enough answer for the degenerate-answer guard to pass."

    sc, _ = _fake_sc(["unused"])
    sc.llm = _Llm()
    sc.time_scope = ("2026-09-01", "2026-09-30")  # entirely in the future
    asyncio.run(AnswerComposer(_settings()).compose(sc, ev, mode=Mode.SEARCH))
    # the pre-window document and its quote still reach the model...
    assert "Ganesh Chaturthi - September 14, 2026" in captured["user"]
    assert "2026 Holiday Calendar" in captured["user"]
    # ...alongside the stated window
    assert "TIME SCOPE ASKED: 2026-09-01 to 2026-09-30" in captured["user"]


# ── the answer is ONE model call ───────────────────────────────────
# Enumeration answers used to spend a second, never-shown "item roster" call
# first, which settled the final item list so date-gating deliberation could not
# land on the user's page. It was removed: a second model deciding membership
# disagreed with itself run to run, and its forced-function reply could not be
# continued when it hit the token cap — a truncated reply parsed to zero items,
# which is byte-for-byte "nothing qualifies". Its membership rules now
# live in the aggregation addendum, so the same reasoning happens inside the call
# that writes.


class _OneCallLlm:
    """Fake LLM for the answer path. complete() is a tripwire: nothing on this
    path may make a second, non-streamed model call."""

    def __init__(self, answer: str):
        self.answer = answer
        self.write_messages: list | None = None

    async def complete(self, *_a, **_k):
        raise AssertionError("the answer path must not make a second model call")

    async def stream_text(self, messages, max_tokens=None, fast=False, usage=None, **_kwargs):
        self.write_messages = messages
        for ch in self.answer:
            yield ch


_CLEAN_ANSWER = ("Two features were added in the window: Trending Content and "
                 "Recommended Actions, each with its stakeholders below.")


def test_an_enumeration_answer_is_a_single_streamed_call():
    """The mode that used to cost two calls now costs one, and still receives the
    asked window and the set-completeness rules."""
    llm = _OneCallLlm(_CLEAN_ANSWER)
    sc, _ = _fake_sc(["unused"])
    sc.llm = llm
    sc.time_scope = ("2026-02-10", "2026-08-10")
    out = asyncio.run(AnswerComposer(_settings()).compose(
        sc, Evidence(), mode=Mode.SUPER_RESEARCH))

    assert out.text == _CLEAN_ANSWER
    user = llm.write_messages[-1]["content"]
    assert "TIME SCOPE ASKED: 2026-02-10 to 2026-08-10" in user
    assert "WHEN THE QUESTION ASKS FOR A SET" in llm.write_messages[0]["content"]


def test_no_pre_settled_item_list_rides_the_prompt_any_more():
    """The roster's write-side scaffolding is gone from both halves of the prompt:
    a leftover "these items and no other" instruction with no list behind it would
    contradict the aggregation rules that now do the job."""
    llm = _OneCallLlm(_CLEAN_ANSWER)
    sc, _ = _fake_sc(["unused"])
    sc.llm = llm
    asyncio.run(AnswerComposer(_settings()).compose(
        sc, Evidence(), mode=Mode.SUPER_RESEARCH))

    whole = llm.write_messages[0]["content"] + llm.write_messages[-1]["content"]
    for ghost in ("FINAL ITEM LIST", "final_item_list", "settled exclusion"):
        assert ghost not in whole


def test_the_set_completeness_rules_settle_the_whole_set_before_the_first_heading():
    """The roster existed because an autoregressive model cannot un-write a heading
    it has started, so gating an item mid-sentence leaked "…wait / excluded
    because…" onto the page. With the call gone, the instruction to settle the set
    in ONE pass before writing is the only thing standing between the reader and
    that leak — it has to be explicit about all three parts: one pass, before the
    first heading, and nothing of it on the page."""
    from ai_graph_retrieval.synthesis.prompts import SYNTHESIS_SYSTEM_PROMPT as A

    assert "SETTLE THE WHOLE LIST FIRST, IN ONE PASS, BEFORE THE FIRST HEADING" in A
    assert "That pass is yours alone" in A
    assert "never gate an item mid-sentence" in A
    assert "no visible reconsideration" in A


def test_membership_is_judged_by_conduct_not_the_questions_label():
    """The empty-roster over-refusal (stakeholder question, 2026-08-17): the model
    returned zero items because no document used the question's own category word,
    despite pages of people the evidence showed holding exactly that role. The
    counterweight is shape-based — judge by what the evidence shows an item DOING
    or BEING — and it had to survive the roster's removal,
    so it now lives in the addendum the write call carries."""
    from ai_graph_retrieval.synthesis.prompts import SYNTHESIS_SYSTEM_PROMPT as A

    assert "DOING or BEING" in A
    assert "never that qualifying items lack the question's exact word" in A
    assert "OWN date" in A  # and the date gate the roster really existed for stays


def test_the_set_completeness_rules_keep_the_rosters_other_membership_rules():
    """Alias de-duplication and "undated is not excluded" were the roster prompt's,
    and they are what stop the same person appearing twice and an undated item
    being dropped from a windowed answer."""
    from ai_graph_retrieval.synthesis.prompts import SYNTHESIS_SYSTEM_PROMPT as A

    assert "ONE ENTRY PER ITEM" in A
    assert "undated is not excluded" in A


def test_a_multi_part_question_is_answered_part_by_part():
    """A question can ask several things at once ("who reports to X, what teams are
    they on, and what have they published?"). Naming the set answers only the
    first, and a part with no evidence must not take the answered parts down with
    it — this is the general capability the roster's special-cased enumeration
    flow was standing in for."""
    from ai_graph_retrieval.synthesis.prompts import SYNTHESIS_SYSTEM_PROMPT as P

    assert "EVERY PART OF THE QUESTION GETS ANSWERED" in P
    assert "asking at least two things" in P
    assert "one column per thing asked" in P
    assert "into a refusal of the whole question" in P
    assert "Partial evidence is a partial answer, never no answer." in P


def test_the_digest_cap_spends_itself_on_structured_profiles_first():
    """Measured failure (2026-08-12): a 13-person roster arrives with ~75
    entities against the 50 cap, 58 of them carrying a blended description. The
    profile pass appends last, so the roster's own people were the first thing
    dropped — the answer rendered their Role column as "—" while the data sat in
    the pool. Tier order is profile > description > bare, stable within a tier."""
    from ai_graph_retrieval.core.models import EntityHit

    s = make_settings(digest_max_entities=3)
    # pool order is adversarial on purpose: descriptions first, profiles last
    described = [EntityHit(name=f"Described {i}", description=f"prose {i}")
                 for i in range(5)]
    bare = [EntityHit(name="Bare One")]
    profiled = [EntityHit(name="Kyle Palmer", profile="Intern · Data Science"),
                EntityHit(name="Noah Kent", profile="Intern · Data Science")]
    digest = AnswerComposer(s)._digest(
        Evidence(entities=described + bare + profiled), strict=False)

    assert "Kyle Palmer" in digest and "Noah Kent" in digest
    assert digest.count("Intern · Data Science") == 2
    assert "Bare One" not in digest              # bare loses to both tiers
    assert digest.count("Described ") == 1       # the cap's one remaining slot
    # within a tier, pool order decides — the FIRST described entity is the one kept
    assert "Described 0" in digest


def test_fact_line_shows_raw_relationship_forms_only_when_they_add_information():
    """raw_types keeps the source's own phrasings; ingestion normalises them into
    one relationship_type. Repeating an identical raw form is noise, so the digest
    renders `[stated as: …]` only for forms that differ from the canonical type."""
    from ai_graph_retrieval.core.models import RelationFact

    ev = Evidence(facts=[
        RelationFact(source="Max", target="Enterprise Search", relationship="OWNS",
                     specific_relationships=["OWNS"], support=2),
        RelationFact(source="Kim", target="Brand refresh", relationship="RELATED_TO",
                     specific_relationships=["SPONSORS", "APPROVES_BUDGET_FOR"], support=1),
    ])
    digest = AnswerComposer(_settings())._digest(ev, strict=False)
    assert "Max OWNS Enterprise Search (support: 2)" in digest
    assert "[stated as: OWNS]" not in digest  # identical raw form adds nothing
    assert "[stated as: SPONSORS, APPROVES_BUDGET_FOR]" in digest


def test_the_system_prompt_carries_nothing_per_request_so_it_can_cache():
    """prompt caching is a PREFIX match, so one per-request byte at the
    front of the system block makes the whole 2.2k-4.3k-token prompt behind it
    uncacheable. The date/time used to sit there and changed every minute — the
    same defect fixed for the agent. The date belongs in the USER
    message; the system block must stay byte-identical between requests."""
    captured = {}

    class _Llm:
        async def stream_text(self, messages, max_tokens=None, fast=False, usage=None, **_kwargs):
            captured["system"] = messages[0]["content"]
            captured["user"] = messages[-1]["content"]
            yield "A long enough answer for the degenerate guard."

    sc, _ = _fake_sc(["unused"])
    sc.llm = _Llm()
    asyncio.run(AnswerComposer(_settings()).compose(
        sc, Evidence(), mode=Mode.SEARCH))
    first_system, first_user = captured["system"], captured["user"]

    # The value moved to the user message, where it is still present for the
    # model to resolve "this year" / "latest" against.
    assert "CURRENT DATE AND TIME:" in first_user
    assert "CURRENT DATE AND TIME:" not in first_system
    # No date fragment of any shape survives in the system block. UTC is the
    # giveaway in the old format ("...%d %B %Y, %H:%M UTC").
    assert "UTC" not in first_system
    assert "current date and time is" not in first_system.lower()

    # Byte-identical across composes: the whole point of the fix. A timestamp
    # anywhere in here would only differ once the clock minute rolled over, so
    # asserting equality alone could pass by luck — the checks above are what
    # make this deterministic, and this pins the property they protect.
    asyncio.run(AnswerComposer(_settings()).compose(
        sc, Evidence(), mode=Mode.SEARCH))
    assert captured["system"] == first_system

    # The instruction stays in the (cached) system prompt — only the VALUE is
    # volatile, so moving the value must not take the rule with it.
    assert "current date/time (given with the question)" in first_system


# ── typed records are evidence, and revisions stay out of the answer ──────────
# Both rules already existed in spirit and both lost. The 2026-08-25 RCA, after
# the authorship read was fixed: one run answered "what has X written" with
# "Nothing in the evidence shows any announcements or process docs written by X"
# while holding 52 authorship facts and 47 sources; another opened "No one on
# Dhiraj's direct reports team has any published content", then wrote "Wait —
# that's not right" and listed everything correctly.
#
# The first was the support rule reading as a quality bar: authorship carries
# support 1 by nature and no quoted excerpt, so evidence made entirely of it
# looked weak. The second was deny-then-describe surviving as a visible
# self-correction, which the no-meta-commentary rule did not obviously cover.


def test_typed_records_are_not_treated_as_thinly_sourced_claims():
    from ai_graph_retrieval.synthesis.prompts import SYNTHESIS_SYSTEM_PROMPT as P

    # The support≥2 preference must be scoped to claims drawn from text…
    assert "support ≥ 2" in P
    assert "CLAIMS drawn from text" in P
    # …and typed records exempted from both the caveat and the quote requirement.
    assert "no single-source caveat" in P
    assert "cannot quote its own authorship" in P
    # The conclusion that actually stops the decline.
    assert "an absence of quotes is not an absence of evidence" in P


def test_the_answer_may_not_revise_itself_in_front_of_the_reader():
    from ai_graph_retrieval.synthesis.prompts import SYNTHESIS_SYSTEM_PROMPT as P

    assert "NEVER DENY-THEN-DESCRIBE" in P
    assert "Never correct yourself where the reader can see it" in P
    assert "replaced, not annotated" in P


def test_a_wording_mismatch_is_not_treated_as_a_missing_answer():
    """Observed on the generalization sweep: asked to "list the pages X created
    last year", the answer found the records, stated that X "authored and edited
    numerous pages throughout 2025", and then refused to enumerate them because
    creating is not strictly authoring or editing. It declined while holding the
    answer. Stated as a principle, not a vocabulary — no verb list (rule 8)."""
    from ai_graph_retrieval.synthesis.prompts import SYNTHESIS_SYSTEM_PROMPT as P

    assert "a wording difference a gap" in P
    assert "rather than refusing over the mismatch" in P
    assert "is a worse answer than listing it" in P


# ── why the roster call could not simply be capped higher ──────────────────
# Three dev traces of one enumeration question, same build: the run whose roster
# came in at 882 output tokens answered with 25 citations; the two that hit the
# 1200-token cap EXACTLY answered with none, one of them leaking "the question as
# asked yields no results only because the final item list was returned empty".
# A forced-function reply cut off mid-array parses to zero usable items, which is
# byte-for-byte a model deciding zero items qualify. Raising the cap made that
# rare; it could not make it impossible, because a forced-function JSON array has
# no continuation round available to it — which is why the call is gone rather
# than tuned, and why this file's tripwire fake raises on complete().


def test_the_answer_path_makes_no_forced_function_call_at_all():
    """The failure mode above is only reachable through a non-streamed,
    tool-forced call on the answer path. There is no longer one — for any mode,
    on any mode — so it is unreachable by construction rather than by a
    guard that a later refactor could drop."""
    for mode in (Mode.SEARCH, Mode.SUPER_RESEARCH):
            llm = _OneCallLlm(_CLEAN_ANSWER)   # complete() raises if called
            sc, _ = _fake_sc(["unused"])
            sc.llm = llm
            out = asyncio.run(AnswerComposer(_settings()).compose(
                sc, Evidence(), mode=mode))
            assert out.text == _CLEAN_ANSWER


# ── Super Research owes a deeper answer than Search ────────────────
# The two agent modes shared a byte-identical synthesis instruction, and every
# difference between them lived in RETRIEVAL — rounds, graph chunks, scoring
# budget. So the extra spend bought more EVIDENCE and the same prompt wrote the
# same shape of answer over it. Measured on nine leadership questions: on four,
# Super Research came within a few hundred characters of Search and carried no
# more citations (3,676 vs 3,144 chars at 7 vs 9 on one of them).


def test_super_research_carries_a_depth_contract_search_does_not():
    """The addendum is the mode's answer contract, and it must ride exactly the
    requests that paid for the deeper evidence."""
    seen = {}

    class _Llm:
        async def stream_text(self, messages, **_kw):
            seen[len(seen)] = messages[0]["content"]
            yield "A long enough answer for the degenerate guard."

    for mode, research, expected in (
        (Mode.SUPER_RESEARCH, True, True),
        (Mode.SEARCH, False, False),
    ):
        sc, _ = _fake_sc(["unused"])
        sc.llm = _Llm()
        before = len(seen)
        asyncio.run(AnswerComposer(_settings()).compose(
            sc, Evidence(), mode=mode, research=research))
        system = seen[before]
        assert ("RESEARCH-GRADE ANSWER" in system) is expected, mode


def test_the_depth_contract_states_the_things_a_search_result_would_not_do():
    """Each line answers one of the ways the mode was measurably no better than
    Search: parts silently dropped, facts reported without explanation, related
    material left side by side, and no statement about corroboration."""
    from ai_graph_retrieval.synthesis.prompts import SYNTHESIS_RESEARCH_ADDENDUM as R

    assert "ACCOUNT FOR EVERY PART OF THE REQUEST, VISIBLY" in R
    assert "a silently dropped sub-question is the single worst failure" in R
    assert "EXPLAIN, DO NOT JUST REPORT" in R
    assert "CONNECT WHAT IS SPREAD OUT" in R
    assert "CORROBORATE, AND SAY SO" in R
    assert "BE COMPLETE FIRST, CONCISE SECOND" in R
    # …and depth must not cost the answer-first opening every mode owes the reader
    assert "LEAD WITH THE ANSWER ANYWAY" in R


def test_the_depth_contract_does_not_break_the_cacheable_system_block():
    """It is per-MODE, so the block stays byte-identical between requests — the
    same invariant the base prompt is held to."""
    seen = []

    class _Llm:
        async def stream_text(self, messages, **_kw):
            seen.append(messages[0]["content"])
            yield "A long enough answer for the degenerate guard."

    for _ in range(2):
        sc, _ = _fake_sc(["unused"])
        sc.llm = _Llm()
        asyncio.run(AnswerComposer(_settings()).compose(
            sc, Evidence(), mode=Mode.SUPER_RESEARCH, research=True))
    assert seen[0] == seen[1]
    assert "UTC" not in seen[0]


# ── a refusal may not contain the answer ──────────────────────────
# Measured on five identical Search runs of one two-part leadership question:
# two declined, and one of them declined while QUOTING the answer inside the
# refusal — "I don't have details on the CEO's leadership team", followed by a
# document quote naming the team, its charter and its membership. It held 39
# excerpts, 24 sources, 36 entities and 56 claims at the time.
#
# The cause was the SECOND part being thin (what each member owns), taking the
# solid first part down with it. Super Research on the heavy tier never did this
# (5/5 answered); the fast tier did, so the rule has to name the failure shape
# rather than state the principle and hope.


def test_a_refusal_may_not_contain_the_answer():
    from ai_graph_retrieval.synthesis.prompts import SYNTHESIS_SYSTEM_PROMPT as P

    assert "A REFUSAL MAY NOT CONTAIN THE ANSWER" in P
    assert "that material is the answer" in P
    assert "is self-contradicting, and the reader can see both" in P


def test_a_thin_second_part_does_not_void_a_solid_first_part():
    """The two-part shape specifically: naming the set answers part one the moment
    the evidence names it, whatever the evidence says about part two. And a POSITION
    is a partial answer where the person is not named — the run that refused had the
    roles (CHRO, CPO, CTO…) in hand and treated having no names as having nothing."""
    from ai_graph_retrieval.synthesis.prompts import SYNTHESIS_SYSTEM_PROMPT as P

    assert "A THIN SECOND PART DOES NOT VOID A SOLID FIRST ONE" in P
    assert "first part is answered the moment the evidence names them" in P
    assert "a role is a partial answer; silence is not" in P


# ── answer shape: one grouping decision, made once ───────────────────────────
# Three STRUCTURE rules used to fire on the same question — "use a markdown
# table", "order by time and group under month or release headings", and (in the
# aggregation addendum) "give each item its own ### heading". A release-notes
# question satisfies all three, nothing said which won, and the model resolved
# the tie afresh on every generation: the same question came back grouped by
# month, then by release, then by person. Reported 2026-08-27.
#
# The fix is a precedence list in the prompt (which shape wins) plus the
# classifier stating which branch this question landed on (AnswerShape), so the
# decision is made once from the question rather than per-run from the evidence.


def test_the_answer_shape_is_chosen_once_by_a_stated_precedence():
    """The prompt has to say which rule wins, not list three that all match."""
    from ai_graph_retrieval.synthesis.prompts import SYNTHESIS_SYSTEM_PROMPT as P

    assert "CHOOSE THE ANSWER'S SHAPE ONCE, BEFORE THE FIRST HEADING, AND KEEP IT" in P
    assert "Take the FIRST" in P
    assert "use ONLY that shape" in P
    assert "never change shape partway through" in P
    # The tie the three old rules could not break: a progression shown as one
    # flat table loses the progression, so timeline outranks table explicitly.
    assert "THIS RULE WINS EVEN WHEN RULE 2 ALSO MATCHES" in P


def test_heading_levels_are_stated_so_the_two_rule_sets_cannot_disagree():
    """The citation rules lay out sections as "##" and the completeness rule asked
    for "### <item>" per item. Both sit in the same prompt on an enumeration
    question and nothing said how they nest, so the model invented the hierarchy
    each run — a second, independent source of drift on top of the grouping choice.

    The aggregation addendum was dissolved and its rules moved into the base
    prompt, so both halves of this contradiction now live in one string.
    """
    from ai_graph_retrieval.synthesis.prompts import SYNTHESIS_SYSTEM_PROMPT as P

    assert "HEADING LEVELS ARE FIXED" in P
    assert 'An answer with one level of grouping uses "## "' in P
    # The completeness rule defers to that rather than restating a level of its own.
    assert 'at the level STRUCTURE sets' in P
    assert '- Give each item its own "### <item>" heading' not in P


def test_the_multi_part_shape_no_longer_offers_a_choice():
    """"one row (or one short section) per member" is the same defect in
    miniature — a fork offered to the model and never resolved. The row wins;
    the substring the multi-part test pins has to survive the edit."""
    from ai_graph_retrieval.synthesis.prompts import SYNTHESIS_SYSTEM_PROMPT as P

    assert "one table row per member" in P
    assert "one row (or one short section) per member" not in P
    assert "one column per thing asked" in P  # pinned by the multi-part test


def test_every_shape_the_classifier_may_emit_has_an_addendum():
    """The enum and the addenda are one contract: a member added without a block
    beside it would silently degrade to "no opinion" on every question that
    lands on it, which is the ambiguity this whole change removes."""
    from ai_graph_retrieval.retrieval.base import AnswerShape
    from ai_graph_retrieval.synthesis.prompts import SYNTHESIS_SHAPE_ADDENDA

    assert set(SYNTHESIS_SHAPE_ADDENDA) == set(AnswerShape)
    for shape, block in SYNTHESIS_SHAPE_ADDENDA.items():
        assert block.startswith("\n\nANSWER SHAPE:"), shape


def test_a_shape_addendum_never_points_at_a_rule_its_base_may_not_carry():
    """Each block has to stand on its own text.

    A block that cross-references a numbered rule ("STRUCTURE rule 1") binds to
    nothing once the base prompt is edited. Mode selects DEPTH and BREADTH;
    shape must hold across every mode.
    """
    from ai_graph_retrieval.synthesis.prompts import (
        SYNTHESIS_SHAPE_ADDENDA,
        SYNTHESIS_SYSTEM_PROMPT,
    )

    for shape, block in SYNTHESIS_SHAPE_ADDENDA.items():
        assert "STRUCTURE rule" not in block, shape
        # No cross-reference of any wording: the numbered list is one base's.
        assert "rule 1" not in block and "rule 2" not in block, shape
        # The block states its own shape by name, so it needs no list to bind to.
        assert shape.value.replace("_", "-").upper() in block.upper(), shape

    assert "CHOOSE THE ANSWER'S SHAPE ONCE" in SYNTHESIS_SYSTEM_PROMPT


def test_an_unknown_or_absent_shape_falls_back_to_the_prompts_own_rules():
    """Fail-open, unlike `intent`: a wrong lane silently buys a research question
    a shallow answer and is worth failing the request over; a wrong shape costs
    only consistency, so it degrades to the precedence rules instead."""
    from ai_graph_retrieval.synthesis.prompts import shape_addendum

    assert shape_addendum(None) == ""
    assert shape_addendum("") == ""
    # A value from a drifting provider, or a shape retired from the enum while a
    # cached tool schema still offers it.
    assert shape_addendum("some_retired_shape") == ""


def test_the_stated_shape_reaches_the_answer_model():
    captured = {}

    class _Llm:
        async def stream_text(self, messages, max_tokens=None, fast=False, usage=None, **_kwargs):
            captured["system"] = messages[0]["content"]
            yield "A long enough answer for the degenerate guard."

    from ai_graph_retrieval.retrieval.base import AnswerShape

    sc, _ = _fake_sc(["unused"])
    sc.llm = _Llm()
    asyncio.run(AnswerComposer(_settings()).compose(
        sc, Evidence(), mode=Mode.SEARCH, answer_shape=AnswerShape.TIMELINE))
    assert "ANSWER SHAPE: this question's shape is TIMELINE" in captured["system"]
    assert "TABLE (STRUCTURE rule 2)" not in captured["system"]

    # No shape stated ⇒ nothing appended, and the answer is shaped by the
    # precedence rules alone — the behaviour before this field existed.
    asyncio.run(AnswerComposer(_settings()).compose(
        sc, Evidence(), mode=Mode.SEARCH, answer_shape=None))
    assert "ANSWER SHAPE:" not in captured["system"]


def test_the_shape_can_be_switched_off_per_environment():
    """The kill switch costs stability, never correctness: OFF falls back to the
    STRUCTURE precedence rules, which are unambiguous on their own."""
    captured = {}

    class _Llm:
        async def stream_text(self, messages, max_tokens=None, fast=False, usage=None, **_kwargs):
            captured["system"] = messages[0]["content"]
            yield "A long enough answer for the degenerate guard."

    from ai_graph_retrieval.retrieval.base import AnswerShape

    sc, _ = _fake_sc(["unused"])
    sc.llm = _Llm()
    asyncio.run(AnswerComposer(_settings(answer_shape_enabled=False)).compose(
        sc, Evidence(), mode=Mode.SEARCH, answer_shape=AnswerShape.TABLE))
    assert "ANSWER SHAPE:" not in captured["system"]
    assert "CHOOSE THE ANSWER'S SHAPE ONCE" in captured["system"]


def test_the_shape_addendum_keeps_the_system_block_cacheable():
    """Caching is a PREFIX match. The shape block is keyed on a
    four-value enum, so the prompt takes one of a small number of byte-identical
    forms rather than becoming per-request."""
    captured = []

    class _Llm:
        async def stream_text(self, messages, max_tokens=None, fast=False, usage=None, **_kwargs):
            captured.append(messages[0]["content"])
            yield "A long enough answer for the degenerate guard."

    from ai_graph_retrieval.retrieval.base import AnswerShape

    sc, _ = _fake_sc(["unused"])
    sc.llm = _Llm()
    for _ in range(2):
        asyncio.run(AnswerComposer(_settings()).compose(
            sc, Evidence(), mode=Mode.SEARCH, answer_shape=AnswerShape.PER_MEMBER))
    assert captured[0] == captured[1]
    assert "UTC" not in captured[0]


def test_a_table_is_preferred_wherever_the_items_line_up():
    """tables read better than bullets or repeated
    headings for most enumerations, so the prompt leans that way — a LEAN, not a
    mandate. The guard against over-applying it has to be stated with it, or the
    model starts building one-column tables and grids full of sentences."""
    from ai_graph_retrieval.synthesis.prompts import SYNTHESIS_SYSTEM_PROMPT as P

    # Whitespace-flattened: a phrase that wraps across two prompt lines is the
    # same instruction, and a reflow must not fail a content assertion.
    flat = " ".join(P.split())
    assert "WHERE A TABLE FITS, PREFER IT" in flat
    assert "two or more items share even two attributes" in flat
    # ...and the escape hatch, so the preference cannot turn into a forcing rule.
    assert "A one-column table is not a table" in flat
    assert "prose or bullets is the honest shape" in flat


def test_completeness_no_longer_demands_a_heading_per_item():
    """The completeness rule asked for a heading per item, which foreclosed the
    table the STRUCTURE rules would otherwise choose — the two rules disagreed on
    exactly the enumeration questions where the shape matters most. The rule lives
    in the base prompt since the aggregation lane was dissolved."""
    from ai_graph_retrieval.synthesis.prompts import SYNTHESIS_SYSTEM_PROMPT as P

    assert "COMPLETENESS IS ABOUT COVERAGE, NOT ABOUT HEADINGS" in P
    assert "one ROW per item is that item's place and it needs no heading" in P
    # A headed shape still gets its per-item headings, at the level STRUCTURE sets.
    assert "Only where the shape is headed sections" in P


def test_the_timeline_shape_puts_a_table_inside_each_period():
    """Rule 1 groups by period; the preference decides what goes INSIDE a group.
    Bullets there are the exception, not the default."""
    from ai_graph_retrieval.retrieval.base import AnswerShape
    from ai_graph_retrieval.synthesis.prompts import SYNTHESIS_SHAPE_ADDENDA

    block = SYNTHESIS_SHAPE_ADDENDA[AnswerShape.TIMELINE]
    assert "render each section's items as a TABLE" in block
    assert "bullets only where the items in that section share nothing" in block


def test_the_prose_shape_does_not_forbid_a_table():
    """prose is chosen from the QUESTION, before the evidence is seen. If what
    comes back does line up, a table for that part is still the better answer —
    otherwise the shape verdict would override the preference it was meant to
    coexist with."""
    from ai_graph_retrieval.retrieval.base import AnswerShape
    from ai_graph_retrieval.synthesis.prompts import SYNTHESIS_SHAPE_ADDENDA

    block = SYNTHESIS_SHAPE_ADDENDA[AnswerShape.PROSE]
    assert "a table for that part is welcome" in block
    assert "not a ruling against tables" in block


# ── how the excerpt budget is SPENT (digest allocation) ───────────────────────
# Measured on the a test corpus, 2026-08-27: the greedy fill exhausts the char
# budget every time (the quote-count cap never binds), so WHICH chunks get it is
# decided entirely by order. Four documents took 61% of 90,000 chars on one query
# and the rest of the documents rendered nothing at all.


def _quotes_for_docs(spec):
    """spec: [(doc_id, n_chunks, chars_each, graph)] → ChunkQuotes in that order."""
    from ai_graph_retrieval.core.models import ChunkQuote

    out = []
    for doc, n, chars, graph in spec:
        for i in range(n):
            out.append(ChunkQuote(
                chunk_id=f"{doc}-c{i}", doc_id=doc, doc_title=f"Title {doc}",
                # distinct text per chunk, so cross-source dedup never merges them
                text=f"{doc}#{i} " + ("word " * (chars // 5)), graph=graph,
            ))
    return out


def _digest_excerpts(settings, quotes, sources, *, quote_chars=0,
                     agg_quote_chars=0):
    """The EXCERPTS section the real composer builds, as a list of lines."""
    ev = Evidence(quotes=quotes, sources=sources)
    text = AnswerComposer(settings)._digest(
        ev, strict=False, quote_chars=quote_chars,
    )
    if "EXCERPTS:\n" not in text:
        return []
    section = text.split("EXCERPTS:\n", 1)[1].split("\n\nSOURCES:", 1)[0]
    return [ln for ln in section.split("\n") if ln.startswith('- From "')]


def test_round_robin_gives_every_document_an_excerpt_instead_of_a_few_all_of_it():
    """The release-notes case: with a greedy fill the first documents eat the budget
    and the later ones are cut whole, so their titles reach SOURCES with no text
    behind them — which is what made those answers one-liners."""
    # 8 documents, 3 chunks each, 6,000 chars per chunk = 144,000 against 65,000
    spec = [(f"d{i}", 3, 6000, "wikipedia") for i in range(8)]
    quotes = _quotes_for_docs(spec)
    sources = [SourceDocument(doc_id=f"d{i}", title=f"Title d{i}",
                              url=f"https://x/{i}") for i in range(8)]

    s_greedy = make_settings()
    s_greedy.digest_round_robin_enabled = False
    greedy = _digest_excerpts(s_greedy, quotes, sources)
    rr = _digest_excerpts(make_settings(), quotes, sources)   # the default

    def docs(lines):
        return {ln.split('"')[1] for ln in lines}

    # Same budget, same chunks — but spread across every document instead of a few.
    assert len(docs(greedy)) < len(docs(rr))
    assert docs(rr) == {f"Title d{i}" for i in range(8)}, "every document represented"


def test_round_robin_keeps_the_document_rank_order_it_was_given():
    """Ordering by trust rank still decides WHICH documents come first; round-robin
    only decides how the budget spreads across them."""
    s = make_settings()
    s.digest_round_robin_enabled = True
    spec = [(f"d{i}", 2, 3000, "wikipedia") for i in range(3)]
    sources = [SourceDocument(doc_id=f"d{i}", title=f"Title d{i}",
                              url=f"https://x/{i}") for i in range(3)]
    lines = _digest_excerpts(s, _quotes_for_docs(spec), sources)
    first_three = [ln.split('"')[1] for ln in lines[:3]]
    assert first_three == ["Title d0", "Title d1", "Title d2"]


def test_breadth_first_is_the_default_and_the_flag_still_restores_the_greedy_fill():
    """Enabled by default from the 2026-08-27 A/B (see settings). The flag stays so a
    deployment can fall back with AGR_DIGEST_ROUND_ROBIN_ENABLED=false and nothing
    else."""
    spec = [("d0", 3, 6000, "wikipedia"), ("d1", 3, 6000, "wikipedia")]
    sources = [SourceDocument(doc_id="d0", title="Title d0", url="https://x/0"),
               SourceDocument(doc_id="d1", title="Title d1", url="https://x/1")]
    quotes = _quotes_for_docs(spec)

    assert make_settings().digest_round_robin_enabled is True
    both = [ln.split('"')[1] for ln in
            _digest_excerpts(make_settings(), quotes, sources)[:2]]
    assert both == ["Title d0", "Title d1"], "each document before either repeats"

    off = make_settings()
    off.digest_round_robin_enabled = False
    lines = _digest_excerpts(off, quotes, sources)
    assert [ln.split('"')[1] for ln in lines[:3]] == ["Title d0"] * 3


def test_a_truncated_fragment_is_skipped_but_a_genuinely_short_chunk_is_kept():
    """The tail of a greedy fill was a 1-8 character excerpt — a line naming a
    document with nothing behind it. A chunk that is simply short (a people record
    is often a name and a role) is real content and stays."""
    from ai_graph_retrieval.core.models import ChunkQuote

    s = make_settings()
    s.digest_quote_char_budget = 5000
    s.digest_min_excerpt_chars = 200
    quotes = [
        ChunkQuote(chunk_id="a", doc_id="d0", doc_title="Big Doc",
                   text="alpha " + ("word " * 990), graph="wikipedia"),      # ~5000
        ChunkQuote(chunk_id="b", doc_id="d1", doc_title="Long Second",
                   text="beta " + ("word " * 990), graph="wikipedia"),       # truncates tiny
        ChunkQuote(chunk_id="c", doc_id="d2", doc_title="A Person",
                   text="Jenna Dale — Staff Engineer", graph="wikipedia"), # short, whole
    ]
    sources = [SourceDocument(doc_id=f"d{i}", title=t, url=f"https://x/{i}")
               for i, t in enumerate(("Big Doc", "Long Second", "A Person"))]
    lines = _digest_excerpts(s, quotes, sources)
    titles = [ln.split('"')[1] for ln in lines]
    assert "Big Doc" in titles
    assert "Long Second" not in titles, "a fragment of it is not evidence"
    assert "A Person" in titles, "genuinely short content is kept whole"


def test_interleave_by_document_is_a_pure_reordering():
    from ai_graph_retrieval.synthesis.dedup import interleave_by_document

    items = [("a", 1), ("a", 2), ("a", 3), ("b", 1), ("c", 1), ("c", 2)]
    out = interleave_by_document(items, doc_of=lambda t: t[0])
    # one per document per round, documents in first-appearance order
    assert out == [("a", 1), ("b", 1), ("c", 1), ("a", 2), ("c", 2), ("a", 3)]
    # nothing gained, nothing lost
    assert sorted(out) == sorted(items)


def test_interleave_by_document_handles_empty_and_single_document():
    from ai_graph_retrieval.synthesis.dedup import interleave_by_document

    assert interleave_by_document([], doc_of=lambda t: t) == []
    one = [("a", i) for i in range(3)]
    assert interleave_by_document(one, doc_of=lambda t: t[0]) == one


# ── excerpt order: relevance first, document rank as the tiebreak ────────────
# A chunk whose document was never ranked as a source must not sort behind every
# ranked source; the relevance stamped by the lane fixes the order.


def _mixed(spec):
    """spec: [(chunk_id, graph, doc, chars, relevance)] → ChunkQuotes."""
    from ai_graph_retrieval.core.models import ChunkQuote

    return [
        ChunkQuote(chunk_id=cid, doc_id=doc, doc_title=f"Title {doc}",
                   text=f"{cid} " + ("word " * (chars // 5)), graph=graph,
                   relevance=rel)
        for cid, graph, doc, chars, rel in spec
    ]


def test_graph_chunks_render_before_api_chunks_however_their_documents_rank():

    s = make_settings()
    s.digest_quote_char_budget = 4000            # room for ~2 of these
    quotes = _mixed([
        ("a1", "other", "api", 2000, 0.0),
        ("g1", "wikipedia", "graph", 2000, 0.9),
    ])
    # the API's document is ranked FIRST and carries the engine's score
    sources = [
        SourceDocument(doc_id="api", title="Title api", url="https://x/a",
                       relevance=0.99),
        SourceDocument(doc_id="graph", title="Title graph", url="https://x/g"),
    ]
    titles = [ln.split('"')[1] for ln in _digest_excerpts(s, quotes, sources)]
    assert titles[0] == "Title graph"


def test_within_the_graph_tier_higher_question_similarity_comes_first():
    s = make_settings()
    quotes = _mixed([
        ("g_low", "wikipedia", "d_low", 1000, 0.10),
        ("g_high", "arxiv", "d_high", 1000, 0.90),
        ("g_mid", "wikipedia", "d_mid", 1000, 0.50),
    ])
    sources = [SourceDocument(doc_id=f"d_{k}", title=f"Title d_{k}",
                              url=f"https://x/{k}") for k in ("low", "high", "mid")]
    lines = _digest_excerpts(s, quotes, sources)
    assert [ln.split('"')[1] for ln in lines] == [
        "Title d_high", "Title d_mid", "Title d_low"]


def test_an_unscored_graph_chunk_falls_back_to_document_order_not_randomness():
    """Vector store unavailable ⇒ every relevance is 0.0, and the tier keeps the
    document rank it had before — the fail-open path."""
    s = make_settings()
    quotes = _mixed([
        ("g2", "wikipedia", "second", 1000, 0.0),
        ("g1", "wikipedia", "first", 1000, 0.0),
    ])
    sources = [SourceDocument(doc_id="first", title="Title first", url="https://x/1"),
               SourceDocument(doc_id="second", title="Title second", url="https://x/2")]
    lines = _digest_excerpts(s, quotes, sources)
    assert [ln.split('"')[1] for ln in lines] == ["Title first", "Title second"]


def test_every_api_document_gives_its_first_chunk_before_any_gives_a_second():
    """The asked-for property: walk the documents, take one from each, come back for
    seconds. Nothing is DISCARDED — the allocator reorders, so a long document
    still contributes its tail last."""

    s = make_settings()
    quotes = _mixed(
        [(f"a{i}", "other", "one", 400, 0.0) for i in range(3)]
        + [(f"b{i}", "other", "two", 400, 0.0) for i in range(3)]
    )
    sources = [
        SourceDocument(doc_id="one", title="Title one", url="https://x/1",
                       relevance=0.9),
        SourceDocument(doc_id="two", title="Title two", url="https://x/2",
                       relevance=0.8),
    ]
    titles = [ln.split('"')[1] for ln in _digest_excerpts(s, quotes, sources)]
    assert titles[:2] == ["Title one", "Title two"], (
        "the engine's better document leads, then the next document — not "
        "document one's whole tail first")
    assert len(titles) == 6, "no chunk is thrown away, only reordered"


def test_the_synthesize_step_span_carries_what_the_call_ran_under(spans):
    """compose() adds no span of its own: the SYNTHESIZE step already opens one
    (core/progress.py) and the generation that writes the answer is its child.
    What neither carried was the CONTEXT of that call — how big the digest was,
    which prompt blocks were appended, which model tier, what came back — so a
    cost or quality question had to be answered from the logs. Annotating the
    step's own span keeps those facts beside the generation instead of adding
    another box to the waterfall for facts that describe this one.
    """
    from ai_graph_retrieval.observability import tracing

    sc, _ = _fake_sc(["An answer long enough to pass the degenerate guard."])
    ev = Evidence(sources=[SourceDocument(doc_id="d1", title="Doc",
                                          url="https://x/1")])
    with tracing.stage_span("stage.synthesize"):
        asyncio.run(AnswerComposer(_settings()).compose(sc, ev, mode=Mode.SEARCH))

    span = next(s for s in spans.get_finished_spans() if s.name == "stage.synthesize")
    at = dict(span.attributes)
    assert at["mode"] == "search"
    assert at["fast_tier"] is True            # SEARCH answers on the fast tier
    assert at["research"] is False
    assert at["digest_chars"] > 0
    assert at["system_prompt_chars"] > 0
    assert at["citation_addendum"] is True    # sources present ⇒ chips requested
    assert at["answer_chars"] > 0
    assert at["degenerate_retry"] is False
    assert at["citations"] == 0               # the answer cited nothing


def test_the_synthesize_step_span_reports_the_degenerate_retry(spans):
    """One retry doubles the cost of the most expensive call in the request. It
    was visible only as a log line."""
    from ai_graph_retrieval.observability import tracing

    sc, _ = _fake_sc(["#", "A real answer on the second attempt."])
    with tracing.stage_span("stage.synthesize"):
        asyncio.run(AnswerComposer(_settings()).compose(sc, Evidence(),
                                                        mode=Mode.SUPER_RESEARCH))

    span = next(s for s in spans.get_finished_spans() if s.name == "stage.synthesize")
    assert dict(span.attributes)["degenerate_retry"] is True


def test_the_precedence_list_kept_what_the_two_rules_it_replaced_said():
    """The list replaced "USE A MARKDOWN TABLE" and "ORDER BY TIME", so every
    instruction those carried has to survive somewhere in it — a rewrite that
    resolves an ambiguity must not also drop guidance.

    Two were lost when this landed and are pinned here because nothing else
    would have caught them: the items inside a period section sort by their OWN
    date (the list ordered only the periods, "newest first"), and comparisons and
    rosters are named as the cases that read better as tables.
    """
    from ai_graph_retrieval.synthesis.prompts import SYNTHESIS_SYSTEM_PROMPT as P

    flat = " ".join(P.split())     # wrap-independent, as above
    # From USE A MARKDOWN TABLE.
    for kept in ("people with roles/locations", "features with dates/owners",
                 "releases with statuses", "one column per attribute",
                 "short prose lead-in", "comparisons and rosters"):
        assert kept in flat, kept
    # From ORDER BY TIME — periods AND the items inside them.
    for kept in ("newest first", "newest first by each item's",
                 "progression is visible", "its stakeholders, its status"):
        assert kept in flat, kept
    # Bullets keep their case, reworded into the OTHERWISE rule.
    assert "bullets for items that share no attributes" in flat
