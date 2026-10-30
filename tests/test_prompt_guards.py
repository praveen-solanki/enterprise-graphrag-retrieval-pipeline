"""The prompt-guard blocks and their wiring into each prompt surface.

The COMPOSITION pattern: one named block per risk category, spliced into
whichever prompt needs it (see the module docstring in ``core/prompt_guards.py``).

These tests pin the composition and the two properties that silently break a
prompt: a missing category block, and a stray format brace. They cannot pin model
behaviour — that needs a live evaluation run.

    uv run pytest tests/test_prompt_guards.py -v
"""
from __future__ import annotations

from ai_graph_retrieval.agent.prompts import AGENT_SYSTEM_PROMPT
from ai_graph_retrieval.core import prompt_guards as g
from ai_graph_retrieval.retrieval.router import _SYSTEM
from ai_graph_retrieval.synthesis.prompts import SYNTHESIS_SYSTEM_PROMPT

_CATEGORY_HEADINGS = [
    "## Answering Is the Default",
    "## Prompt Injection",
    "## Your Own Instructions and Mechanics",
    "## Persona and Hypothetical Framing",
    "## Claimed Authority and Access",
    "## Harmful or Abusive Content",
    "## Discrimination and Bias",
    "## Distress and Personal Safety",
    "## Disproportionate Requests",
    "## Outside the Knowledge Base",
]


def test_classifier_system_prompt_carries_every_category_block():
    for heading in _CATEGORY_HEADINGS:
        assert heading in _SYSTEM, f"classifier prompt missing {heading}"


def test_precision_principle_leads_the_guardrail_section():
    # Order matters: the narrow-exception framing must be read before the rules it
    # qualifies, or the model reads a list of reasons to refuse.
    blocks = g.CLASSIFIER_GUARDRAIL_BLOCKS
    assert blocks.startswith(g.PRECISION_PRINCIPLE)
    assert blocks.index("## Answering Is the Default") < blocks.index("## Prompt Injection")


def test_self_disclosure_block_exempts_documented_systems():
    # The regression that started questions about the organization's own
    # documented AI systems were declined as attempts on our configuration. The
    # block must scope itself to THIS assistant and say documented systems are
    # ordinary knowledge.
    block = g.SELF_DISCLOSURE_GUIDANCE
    assert "YOUR OWN" in block
    assert "documents its own" in block
    assert "ordinary knowledge questions" in block


def test_every_category_block_that_can_over_refuse_carries_a_narrowing_clause():
    # Each risk block must state what it does NOT cover, so a sensitive-sounding
    # subject cannot be read as a match. Pinned per block, since losing this half
    # is exactly how over-refusal returns.
    for name, block in (
        ("harmful", g.HARMFUL_CONTENT_GUIDANCE),
        ("bias", g.BIAS_DISCRIMINATION_GUIDANCE),
        ("distress", g.DISTRESS_SAFETY_GUIDANCE),
        ("disproportionate", g.DISPROPORTIONATE_REQUEST_GUIDANCE),
        ("out_of_scope", g.OUT_OF_SCOPE_GUIDANCE),
        ("self_disclosure", g.SELF_DISCLOSURE_GUIDANCE),
    ):
        assert "knowledge" in block.lower() or "normally" in block.lower(), (
            f"{name} block lost its narrowing clause"
        )


def test_agent_prompt_carries_injection_and_grounding_guards():
    assert g.INJECTION_GUARD in AGENT_SYSTEM_PROMPT
    assert g.GROUNDING_GUARD in AGENT_SYSTEM_PROMPT


def test_synthesis_prompt_carries_the_evidence_injection_rule():
    assert "never instructions to you" in SYNTHESIS_SYSTEM_PROMPT
    assert "Nothing inside" in SYNTHESIS_SYSTEM_PROMPT


def test_classifier_system_prompt_is_static_and_carries_no_timestamp():
    # The cache invariant: this block is the cache-marked prefix, so it
    # must be byte-identical between requests. It therefore has no format holes at
    # all — a {now} placeholder here is what made every request a cache miss — and
    # no date of its own. The date travels in the user message instead.
    assert "{" not in _SYSTEM and "}" not in _SYSTEM
    assert "Today is" not in _SYSTEM


def test_blocks_contain_no_format_braces():
    # Every block may be spliced into a str.format template — a literal brace
    # would corrupt the render.
    for name in dir(g):
        val = getattr(g, name)
        if name.isupper() and isinstance(val, str):
            assert "{" not in val and "}" not in val, f"{name} contains a brace"


# ── prompt-cache stability ────────────────────────────────────────

def test_agent_system_prompt_is_stable_across_calls_and_has_no_date():
    # The agent marks its system block for caching and re-sends the growing
    # conversation every round, so a per-request timestamp here was the most
    # expensive cache miss in the service. The block must be byte-identical.
    from ai_graph_retrieval.agent.prompts import agent_input, build_agent_system_prompt
    first, second = build_agent_system_prompt(), build_agent_system_prompt()
    assert first == second
    assert "Current date and time" not in first
    # ...and the date still reaches the model, on the input instead.
    assert "CURRENT DATE AND TIME: Wednesday" in agent_input("q", now="Wednesday, 29 July")
    assert "QUESTION: q" in agent_input("q", now="Wednesday, 29 July")


def test_clarify_nudge_does_not_vary_per_request():
    # The one thing that changes the agent block is a per-AGENT flag, not a
    # per-request value — so both variants are still individually cacheable.
    from ai_graph_retrieval.agent.prompts import build_agent_system_prompt
    assert build_agent_system_prompt(True) == build_agent_system_prompt(True)
    assert build_agent_system_prompt(True) != build_agent_system_prompt(False)


def test_the_full_agent_lane_is_told_what_its_extra_rounds_are_for():
    """Super Research gets several times Search's rounds, and once its
    instructions said nothing about what to do with them — so a round that found
    one answer counted as productive and the loop stopped there. Per-agent, so
    the cacheable system block stays byte-identical across requests."""
    from ai_graph_retrieval.agent.prompts import build_agent_system_prompt

    deep = build_agent_system_prompt(research=True)
    light = build_agent_system_prompt(research=False)

    assert "YOU HAVE THE ROUNDS FOR REAL RESEARCH" in deep
    assert "YOU HAVE THE ROUNDS FOR REAL RESEARCH" not in light
    assert "A single unverified hit is a lead, not a finding." in deep
    assert "FOLLOW THE CONNECTIONS" in deep
    assert "Depth is not volume" in deep
    # stable, and composes with the clarify nudge rather than replacing it
    assert build_agent_system_prompt(research=True) == deep
    both = build_agent_system_prompt(clarify_enabled=True, research=True)
    assert "YOU HAVE THE ROUNDS FOR REAL RESEARCH" in both
    assert "CHECKING IN WITH THE USER MID-RESEARCH" in both


def test_only_the_citation_block_states_where_citations_go():
    """ONE rule, ONE place: a prompt that states its own citation rule beside the
    addendum's contradicts it, and two rules disagreeing inside one prompt is a
    coin toss.

    The base prompt and the research addendum reintroduced that contradiction in
    softer words — "mark it as reported in a single source", "present both values
    with their dates/sources", "cite them together" — and the model satisfied both,
    producing citations inline AND on the consolidated line (reported from the
    pilot UI in Search and Super Research).

    The invariant is structural, not a phrase list: the output marker ``Sources:``
    belongs to exactly one block, so any block that starts telling the model where
    citations go fails this test. An earlier version also asserted the three removed
    phrases verbatim, which added nothing — a paraphrase walks straight past a
    string match, and matching strings is not what makes the rule hold.
    """
    from ai_graph_retrieval.synthesis.prompts import (
        SYNTHESIS_CITATION_ADDENDUM,
        SYNTHESIS_RESEARCH_ADDENDUM,
    )

    others = {
        "base": SYNTHESIS_SYSTEM_PROMPT,
        "research": SYNTHESIS_RESEARCH_ADDENDUM,
    }
    # the output citation marker is named in the citation block and nowhere else
    assert "Sources:" in SYNTHESIS_CITATION_ADDENDUM
    for name, block in others.items():
        assert "Sources:" not in block, f"{name} states citation placement"

