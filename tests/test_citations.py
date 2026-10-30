"""Inline citation resolution — chips become stable numeric placeholders.

Placement is the model's decision; this covers the deterministic half — resolving
the model's chips (and any self-written source links) to real documents, rewriting
each into a ``[n]`` placeholder numbered by first appearance, and returning the
cited documents in that same order so ``citations[n-1]`` is the document behind
``[n]``. The answer text carries no citation links and no Sources section; the
UI/backend renders the clickable citations from the mapping.

    uv run pytest tests/test_citations.py -v
"""
from __future__ import annotations

from ai_graph_retrieval.core.models import SourceDocument
from ai_graph_retrieval.synthesis.citations import resolve_citations


def _srcs():
    return [
        SourceDocument(doc_id="d1", title="Leave Policy", url="https://x/leave"),
        SourceDocument(doc_id="d2", title="HR Handbook", url="https://x/hr"),
        SourceDocument(doc_id="d3", title="Employee Benefits Guide"),  # no URL
    ]


def test_chips_become_numbered_placeholders_in_appearance_order():
    answer = ("Employees get 20 days. [HR Handbook]\n"
              "They can carry forward 5. [Leave Policy]")
    text, cited = resolve_citations(answer, _srcs())
    assert text == ("Employees get 20 days. [1]\n"
                    "They can carry forward 5. [2]")
    # citations[n-1] IS the document behind [n]: appearance order, not rank order
    assert [c.doc_id for c in cited] == ["d2", "d1"]
    # ... and each entry says so explicitly: ref=n ⇒ this document is [n].
    assert [c.ref for c in cited] == [1, 2]


def test_evidence_source_rows_never_gain_a_ref():
    srcs = _srcs()
    resolve_citations("Fact. [Leave Policy]", srcs)
    assert all(s.ref is None for s in srcs)   # citations carry copies


def test_repeat_citations_of_one_document_share_one_number():
    answer = "Fact A. [Leave Policy] Fact B. [Leave Policy] Fact C. [HR Handbook]"
    text, cited = resolve_citations(answer, _srcs())
    assert text == "Fact A. [1] Fact B. [1] Fact C. [2]"
    assert [c.doc_id for c in cited] == ["d1", "d2"]


def test_answer_carries_no_citation_links():
    text, cited = resolve_citations("Fact one. [Leave Policy]", _srcs())
    assert text == "Fact one. [1]"
    assert "https://x/leave" not in text          # url lives ONLY in the payload
    assert "## Sources" not in text
    assert cited[0].url == "https://x/leave"


def test_unknown_chip_is_dropped_so_hallucinated_citations_never_ship():
    text, cited = resolve_citations("Leave is 20 days. [Nonexistent Doc]", _srcs())
    assert "Nonexistent Doc" not in text
    assert cited == []


def test_chip_without_url_is_dropped_so_every_placeholder_can_resolve():
    text, cited = resolve_citations(
        "Benefits are listed. [Employee Benefits Guide]", _srcs()
    )
    assert "Employee Benefits Guide" not in text
    assert text == "Benefits are listed."
    assert cited == []


def test_model_written_source_link_is_normalized_to_a_placeholder():
    # The model is told to use chips only, but if it links a source directly the
    # link is removed and replaced by the same placeholder (anchor text kept).
    answer = "Dashboards are listed in [the K8s page](https://x/leave)."
    text, cited = resolve_citations(answer, _srcs())
    assert text == "Dashboards are listed in the K8s page [1]."
    assert [c.doc_id for c in cited] == ["d1"]


def test_content_links_pass_through_verbatim():
    answer = "Visit [the dev dashboard](https://k8s.dev.example/) directly."
    text, cited = resolve_citations(answer, _srcs())
    assert text == answer                         # content, not a citation
    assert cited == []


def test_chip_and_direct_link_to_same_source_share_one_number():
    answer = "Fact A. [Leave Policy] More at [the policy](https://x/leave)."
    text, cited = resolve_citations(answer, _srcs())
    assert text == "Fact A. [1] More at the policy [1]."
    assert [c.doc_id for c in cited] == ["d1"]


def test_chip_matching_tolerates_case_and_spacing_only():
    text, cited = resolve_citations("x [  leave   policy ]", _srcs())
    assert text == "x [1]"
    assert [c.doc_id for c in cited] == ["d1"]


def test_adjacent_chips_get_their_own_numbers():
    text, cited = resolve_citations("Both agree. [Leave Policy] [HR Handbook]", _srcs())
    assert text == "Both agree. [1] [2]"
    assert [c.doc_id for c in cited] == ["d1", "d2"]


def test_uncited_documents_are_excluded():
    _, cited = resolve_citations("Only one source. [HR Handbook]", _srcs())
    assert [c.title for c in cited] == ["HR Handbook"]     # d1/d3 not listed


def test_no_sources_returns_unchanged():
    assert resolve_citations("text [Anything]", []) == ("text [Anything]", [])


def test_dropping_a_mid_token_chip_never_fuses_the_neighbouring_words():
    text, _ = resolve_citations("foo[Nonexistent Doc]bar", _srcs())
    assert text == "foo bar"


def test_cleanup_is_local_to_drop_sites_and_preserves_the_answers_formatting():
    # A Markdown hard break (two trailing spaces) and deliberate double spacing
    # elsewhere in the answer must survive a chip drop untouched.
    answer = "line one  \nA  spaced   table row\nFact. [Nonexistent Doc]"
    text, _ = resolve_citations(answer, _srcs())
    assert text == "line one  \nA  spaced   table row\nFact."


def test_adjacent_dropped_chips_cascade_cleanly():
    text, _ = resolve_citations(
        "Fact [Nonexistent Doc] [Employee Benefits Guide].", _srcs()
    )
    assert text == "Fact."


# ── one consolidated citation line per subsection ──────────────────

def test_subsection_citation_lines_resolve_in_reading_order():
    answer = (
        "## Reporting Structure\n"
        "Statement A\n"
        "Statement B\n"
        "Sources: [HR Handbook] [Leave Policy]\n"
        "\n"
        "## Leave\n"
        "Statement C\n"
        "Sources: [Leave Policy]\n"
    )
    text, cited = resolve_citations(answer, _srcs())
    assert text == (
        "## Reporting Structure\n"
        "Statement A\n"
        "Statement B\n"
        "Sources: [1] [2]\n"
        "\n"
        "## Leave\n"
        "Statement C\n"
        "Sources: [2]\n"          # same document keeps its number across subsections
    )
    assert [(c.ref, c.doc_id) for c in cited] == [(1, "d2"), (2, "d1")]


def test_a_single_source_subsection_keeps_the_same_line_format():
    text, cited = resolve_citations(
        "## Leave\nEmployees get 20 days.\nSources: [Leave Policy]\n", _srcs()
    )
    assert text == "## Leave\nEmployees get 20 days.\nSources: [1]\n"
    assert [c.ref for c in cited] == [1]


def test_a_source_line_left_empty_by_drops_is_removed():
    # Every chip on the line is unciteable (no url / unknown title), so the label
    # would otherwise strand as a bare "Sources:" under the heading.
    answer = (
        "## Benefits\n"
        "Benefits are listed.\n"
        "Sources: [Employee Benefits Guide] [Nonexistent Doc]\n"
        "\n"
        "## Leave\n"
        "Leave is 20 days.\n"
        "Sources: [Leave Policy]\n"
    )
    text, cited = resolve_citations(answer, _srcs())
    assert "Sources:" in text                      # the citable one survives
    assert text.count("Sources:") == 1
    assert "Benefits are listed.\n\n## Leave" in text
    assert [c.ref for c in cited] == [1]


def test_a_partially_dropped_source_line_keeps_its_survivors():
    text, _ = resolve_citations(
        "## Mixed\nFact.\nSources: [Employee Benefits Guide] [Leave Policy]\n", _srcs()
    )
    assert text == "## Mixed\nFact.\nSources: [1]\n"


def test_bold_and_bulleted_empty_labels_are_removed_too():
    for label in ("**Sources:**", "*Sources*", "- Sources:", "References:"):
        text, _ = resolve_citations(
            f"## X\nFact.\n{label} [Nonexistent Doc]\n", _srcs()
        )
        assert "Sources" not in text and "References" not in text, label
        assert text.rstrip().endswith("Fact.")


def test_inline_chips_still_resolve_for_backward_compatibility():
    # The prompt now asks for subsection lines, but a model that still emits an inline
    # chip must not produce a broken answer.
    text, cited = resolve_citations("Employees get 20 days. [Leave Policy]", _srcs())
    assert text == "Employees get 20 days. [1]"
    assert [c.ref for c in cited] == [1]


def test_the_placement_rule_names_every_inline_position_including_tables():
    # The rule was first written as "not in sentences, statements or bullets", and the
    # model read that narrowly: it kept putting a chip in each TABLE CELL (one per
    # person in a reports-to table). Prompt text can't be behaviour-tested here, so
    # pin that the forbidden positions stay enumerated and the consolidated line is
    # stated as the only permitted place.
    from ai_graph_retrieval.synthesis.prompts import SYNTHESIS_CITATION_ADDENDUM as A
    assert "ONLY place a citation may appear" in A
    for position in ("inside a TABLE", "not in a cell", "bullet or numbered list item",
                     "heading, sub-heading, or caption"):
        assert position in A, position
    # ...and that a per-row/per-item list is explicitly covered, since that is the
    # shape that keeps tempting a citation per line.
    assert "DIFFERENT document" in A


# ── shapes that leaked raw text into real answers ────────

_LONG = "AI in Action: Key Takeaways From Ragan's AI Horizons Conference 2026"


def _src(title=_LONG, url="https://docs.example.com/site/a/page/b"):
    return [SourceDocument(doc_id="d1", title=title, url=url)]


def test_a_chip_written_in_link_shape_with_the_title_instead_of_a_url_resolves():
    """The most common real leak: the model writes "[Title]( Title)".

    The link branch needs a scheme so it declines, and the chip branch used to be
    blocked by a negative lookahead on "(" — so the whole thing was emitted verbatim
    and a raw title chip reached the reader inside the answer.
    """
    out, cited = resolve_citations(f"Sources: [ {_LONG}]( {_LONG})", _src())
    assert out == "Sources: [1]"
    assert len(cited) == 1 and cited[0].ref == 1


def test_a_self_written_link_resolves_even_when_its_url_differs_from_the_source():
    # Same document, href with an extra query string. Matching on the anchor TEXT
    # keeps it a citation instead of leaving a bare markdown link in the answer.
    out, cited = resolve_citations(
        f"Sources: [{_LONG}](https://docs.example.com/site/a/page/b?v=2)", _src()
    )
    assert out == "Sources: [1]"
    assert len(cited) == 1


def test_a_citation_line_carries_markers_only_never_the_title_as_prose():
    out, _ = resolve_citations(f"Sources: [{_LONG}](https://docs.example.com/site/a/page/b)",
                               _src())
    assert out == "Sources: [1]"
    assert _LONG not in out          # the title is one click away, not inline noise


def test_an_anchor_mid_sentence_still_keeps_its_prose():
    # The rule is contextual: on a citation line the title is redundant, but in a
    # sentence it IS the sentence.
    out, _ = resolve_citations(
        "See the [Guide](https://docs.example.com/site/a/page/b) page.",
        _src(title="Guide"),
    )
    assert out == "See the Guide [1] page."


def test_prose_in_parentheses_after_a_chip_survives():
    # The paren must follow with NO space to be treated as part of the chip, which is
    # what keeps ordinary writing intact.
    out, _ = resolve_citations("Sources: [Guide] (see also the appendix)", _src(title="Guide"))
    assert out == "Sources: [1] (see also the appendix)"


def test_a_genuine_content_link_is_never_touched():
    out, cited = resolve_citations(
        "See [the vendor docs](https://vendor.example.com/guide) for setup.", _src()
    )
    assert out == "See [the vendor docs](https://vendor.example.com/guide) for setup."
    assert cited == []


# ── titles that contain brackets (documents from file-based connectors) ───────
#
# Documents from file-based connectors have titles that are file names — and file names
# carry brackets. Cited, such a title nests: "[[How to] Do the thing.docx]".

_BRACKETED = "[How to] Request content promotion in INTRANET NAME.docx"


def _bracketed_src(url="https://x/howto"):
    return [SourceDocument(doc_id="d9", title=_BRACKETED, url=url)]


def test_a_title_containing_brackets_resolves_instead_of_leaking_as_prose():
    """The observed failure: a bracket-free chip pattern matched only the INNER
    "[How to]", dropped it as an unknown document, and left the rest of the title
    with the outer bracket attached — "Sources: [ Request content promotion in
    INTRANET NAME.docx]" reached the reader as text, with no citation."""
    out, cited = resolve_citations(f"Feature requests go to comms.\nSources: [{_BRACKETED}]",
                                   _bracketed_src())
    assert out == "Feature requests go to comms.\nSources: [1]"
    assert [(c.doc_id, c.ref) for c in cited] == [("d9", 1)]


def test_a_bracketed_title_resolves_in_link_shape_too():
    out, cited = resolve_citations(f"Sources: [{_BRACKETED}](https://x/howto)",
                                   _bracketed_src())
    assert out == "Sources: [1]"
    assert len(cited) == 1


def test_a_bracketed_title_with_no_url_drops_whole_leaving_no_stray_brackets():
    out, cited = resolve_citations(
        f"## Promotion\nAsk the comms team.\nSources: [{_BRACKETED}]\n",
        _bracketed_src(url=None),
    )
    assert out == "## Promotion\nAsk the comms team.\n"
    assert cited == []
    assert "[" not in out and "]" not in out       # no half-eaten title survives


def test_a_bracketed_title_alongside_a_plain_one_numbers_both():
    srcs = [
        *_bracketed_src(),
        SourceDocument(doc_id="d8", title="2025 Battlecards: LumApps", url="https://x/bc"),
    ]
    out, cited = resolve_citations(
        f"Sources: [2025 Battlecards: LumApps] [{_BRACKETED}]", srcs
    )
    assert out == "Sources: [1] [2]"
    assert [c.doc_id for c in cited] == ["d8", "d9"]


def test_an_unknown_bracketed_chip_is_dropped_whole():
    out, cited = resolve_citations("Fact. [[How to] Something Else.docx]", _bracketed_src())
    assert out == "Fact."
    assert cited == []


# ── dangling separators (pilot report: "response looks truncated") ───────────
# "- **Seth Ward** —" reads as a cut-off answer. It is not: the model wrote
# "**Name** — role" for a person whose only descriptor was the INACTIVE flag it
# had already used as the section heading, leaving the separator stranded.


def _resolve(answer: str):
    from ai_graph_retrieval.core.models import SourceDocument
    from ai_graph_retrieval.synthesis.citations import resolve_citations

    sources = [SourceDocument(doc_id="d1", title="People Directory", url="https://x/d1")]
    return resolve_citations(answer, sources)[0]


def test_a_stranded_separator_at_end_of_line_is_removed():
    out = _resolve("### No Longer Active\n\n- **Seth Ward** —\n\nSources: [People Directory]")
    assert "- **Seth Ward**\n" in out
    assert "Sharma** —" not in out


def test_en_dash_and_hyphen_are_treated_the_same_way():
    # all three are used interchangeably as a separator in real answers
    for sep in ("\u2014", "\u2013", "-"):
        assert _resolve(f"- **A Name** {sep}\n") .rstrip() == "- **A Name**"


def test_a_separator_inside_a_line_is_untouched():
    text = "- **Adam Shaw** — Senior Test Engineer\n"
    assert _resolve(text) == text


def test_a_list_bullet_is_not_mistaken_for_a_dangling_separator():
    text = "- item one\n- item two\n"
    assert _resolve(text) == text


def test_a_markdown_rule_survives():
    text = "Some text\n\n---\n\nMore text\n"
    assert _resolve(text) == text


def test_a_trailing_separator_with_a_citation_after_it_is_kept():
    """The dash is only stranded if nothing follows on the line."""
    out = _resolve("- **A Name** — [People Directory]\n")
    assert "— [1]" in out


# ── "each document once per subsection" — the consolidated line vs inline ────


def test_a_marker_repeated_on_the_subsection_citation_line_is_dropped():
    """The model cited two documents inline and then listed them again on the
    closing Sources line (measured live: 24 markers, 4 already inline). The
    inline placement — where the number was minted — stands; the repeat goes,
    and a document cited ONLY on the line stays."""
    answer = ("Fact one. [Leave Policy] Fact two. [HR Handbook]\n"
              "More prose.\n"
              "Sources: [Leave Policy] [HR Handbook] [Employee Benefits Guide]")
    srcs = _srcs()
    srcs[2] = SourceDocument(doc_id="d3", title="Employee Benefits Guide", url="https://x/b")
    text, cited = resolve_citations(answer, srcs)
    assert text == ("Fact one. [1] Fact two. [2]\n"
                    "More prose.\n"
                    "Sources: [3]")
    assert [c.doc_id for c in cited] == ["d1", "d2", "d3"]


def test_a_citation_line_left_empty_by_repeats_is_removed():
    answer = "Fact one. [Leave Policy]\nSources: [Leave Policy]"
    text, cited = resolve_citations(answer, _srcs())
    assert text == "Fact one. [1]"
    assert [c.doc_id for c in cited] == ["d1"]


def test_the_compliant_shape_passes_through_untouched():
    """No inline citations ⇒ the per-subsection lines are exactly what ships."""
    answer = ("## First\nContent.\nSources: [Leave Policy] [HR Handbook]\n"
              "## Second\nMore.\nSources: [Leave Policy]")
    text, _ = resolve_citations(answer, _srcs())
    assert text == ("## First\nContent.\nSources: [1] [2]\n"
                    "## Second\nMore.\nSources: [1]")


def test_repeats_are_judged_per_subsection_not_per_answer():
    """A heading opens a new subsection: a document cited inline under the FIRST
    heading may legitimately close the SECOND subsection's line."""
    answer = ("## First\nFact. [Leave Policy]\n"
              "## Second\nMore.\nSources: [Leave Policy]")
    text, _ = resolve_citations(answer, _srcs())
    assert text == "## First\nFact. [1]\n## Second\nMore.\nSources: [1]"
