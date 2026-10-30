"""Contribution counts must rank people, not corpora.

``_involvement`` turns a contribution count into a score with an absolute curve
(``support_weight * log10(1+n) / hops``). The count is of things in ONE graph, and
the two graphs differ by 3.76x in chunk volume (4,747 wikipedia / 17,850 arxiv), so
the denser corpus handed its people the higher count on every row and swept the
digest's FACTS section on density alone — measured end to end: with the cap at 4, all
four rendered contributors came from the larger graph.

``_level_scores_per_graph`` lifts each graph by ``sqrt(ceiling/peak)`` — halfway
toward the largest graph peak. Three things it deliberately does NOT do:

  * normalise to 1.0 — that would put both people-reads' top row at the same score
    and erase the ``hops`` divisor, which exists precisely to stop the two-hop read
    (whose unit is far more plentiful) outranking the direct one on every row;
  * equate the graphs' peaks — a sparse graph's most-involved person is its best,
    not automatically as involved as the dense graph's best;
  * touch the spacing WITHIN a graph — the involvement curve's own ordering is the
    ranking intent and survives untouched.

    uv run pytest tests/test_contribution_density_bias.py -v
"""
from __future__ import annotations

import asyncio
import math

from ai_graph_retrieval.core.access import AccessContext
from ai_graph_retrieval.graph.repository import GraphReadRepository
from tests.support import make_settings

WIKIPEDIA, ARXIV = "wikipedia", "arxiv"


def _ctx():
    return AccessContext(tenant_id="t", principals=("public",), request_id="r")


def _repo(rows, settings=None):
    repo = GraphReadRepository(None, settings or make_settings(), http=object())

    async def _stub_rows(_query, _ctx, **_params):
        return rows

    repo._rows = _stub_rows
    return repo


def _doc_row(person: str, contributions: int, graph: str) -> dict:
    return {"person": person, "role": "AUTHORED", "document": "D",
            "contributions": contributions, "graph": graph}


def _topic_row(person: str, contributions: int, graph: str) -> dict:
    return {"person": person, "role": "CONTRIBUTED_TO", "via": "V",
            "contributions": contributions, "graph": graph}


def _raw(n: int, hops: int = 1) -> float:
    """The involvement curve, unlevelled — what a single graph still produces."""
    return round(make_settings().support_weight * math.log10(1.0 + n) / hops, 4)


def _damped(n: int, own_peak: int, ceiling: int, hops: int = 1) -> float:
    """The curve after leveling: lifted by sqrt(ceiling peak / own graph peak)."""
    return round(
        _raw(n, hops) * math.sqrt(_raw(ceiling, hops) / _raw(own_peak, hops)), 4
    )


def _scores(facts) -> dict[str, float]:
    return {f.source: f.score for f in facts}


# ── 1. density no longer decides the ranking ────────────────────────────────


def test_a_denser_graph_no_longer_outranks_on_count_alone():
    """The regression: arXiv holds ~3.76x the material, so its people carry
    ~3.76x the contributions and took every top row regardless of standing. The
    damped lift closes half the gap: the smaller graph's most-involved person now
    outranks the dense graph's middling one — without being equated to its best."""
    rows = [_doc_row("BK Best", 5, WIKIPEDIA)]
    rows += [_doc_row("CF Best", 19, ARXIV), _doc_row("CF Mid", 7, ARXIV)]

    facts = asyncio.run(_repo(rows).document_contributors(_ctx(), "topic"))
    scores = _scores(facts)

    assert scores["BK Best"] > _raw(5), "lifted toward the denser graph's scale"
    assert scores["BK Best"] > scores["CF Mid"], "raw, 5 lost to 7 on density alone"
    assert scores["BK Best"] < scores["CF Best"], "never equated to the best"
    assert scores["BK Best"] == _damped(5, own_peak=5, ceiling=19)


def test_the_denser_graph_cannot_sweep_the_digest_facts_section():
    """End to end, the shape the verification measured: 8 contributors offered,
    4 rendered — the smaller graph's experts must not be shut out."""
    from ai_graph_retrieval.core.models import Evidence
    from ai_graph_retrieval.synthesis.composer import AnswerComposer

    settings = make_settings()
    settings.digest_max_facts = 4
    rows = [_doc_row(f"BK Expert {i}", 5, WIKIPEDIA) for i in range(4)]
    rows += [_doc_row("CF Person 0", 19, ARXIV)]
    rows += [_doc_row(f"CF Person {i}", 7, ARXIV) for i in range(1, 4)]

    facts = asyncio.run(_repo(rows, settings).document_contributors(_ctx(), "t"))
    digest, counts, _ = AnswerComposer(settings)._build_digest(
        Evidence(facts=facts), strict=False)

    assert counts["facts_rendered"] == 4
    assert "BK Expert" in digest, "the smaller graph reaches the model"


def test_the_facts_section_ranks_by_corrected_score_not_raw_support():
    """The digest's FACTS cap must follow the leveled cross-graph score. Sorting
    raw `support` first re-ranked by per-graph chunk counts and handed the
    section back to the denser corpus, undoing the leveling upstream."""
    from ai_graph_retrieval.core.models import Evidence, RelationFact
    from ai_graph_retrieval.synthesis.composer import AnswerComposer

    settings = make_settings()
    settings.digest_max_facts = 2
    # Pool state: scores already leveled. The sparse graph's best edge outscores
    # the dense graph's minor edge, but carries the smaller raw support count.
    facts = [
        RelationFact(source="A", target="dense-top", relationship="OWNS",
                     support=100, score=0.30, graph=ARXIV),
        RelationFact(source="A", target="dense-minor", relationship="USES",
                     support=5, score=0.117, graph=ARXIV),
        RelationFact(source="A", target="sparse-top", relationship="OWNS",
                     support=3, score=0.134, graph=WIKIPEDIA),
    ]
    digest, counts, _ = AnswerComposer(settings)._build_digest(
        Evidence(facts=facts), strict=False)

    assert counts["facts_rendered"] == 2
    assert "sparse-top" in digest, "the leveled score decides the cap"
    assert "dense-minor" not in digest, "raw support no longer outranks it"


def test_facts_score_ties_still_break_on_support():
    """Support keeps its meaning as the evidence count — it orders rows the
    score cannot separate, exactly like the neighbour scorer's cheap rank."""
    from ai_graph_retrieval.core.models import Evidence, RelationFact
    from ai_graph_retrieval.synthesis.composer import AnswerComposer

    settings = make_settings()
    settings.digest_max_facts = 1
    facts = [
        RelationFact(source="A", target="thin", relationship="USES",
                     support=1, score=0.2, graph=WIKIPEDIA),
        RelationFact(source="A", target="corroborated", relationship="USES",
                     support=6, score=0.2, graph=WIKIPEDIA),
    ]
    digest, _, _ = AnswerComposer(settings)._build_digest(
        Evidence(facts=facts), strict=False)

    assert "corroborated" in digest and "thin" not in digest


def test_leveling_is_proportional_not_a_preference_for_either_graph():
    """Neither graph is ranked above the other: swap which one is denser and the
    outcome mirrors exactly."""
    a = asyncio.run(_repo([_doc_row("X", 5, WIKIPEDIA),
                           _doc_row("Y", 19, ARXIV)]
                          ).document_contributors(_ctx(), "t"))
    b = asyncio.run(_repo([_doc_row("X", 19, WIKIPEDIA),
                           _doc_row("Y", 5, ARXIV)]
                          ).document_contributors(_ctx(), "t"))

    lifted = _damped(5, own_peak=5, ceiling=19)
    assert _scores(a) == {"X": lifted, "Y": _raw(19)}
    assert _scores(b) == {"X": _raw(19), "Y": lifted}


def test_the_topic_read_is_levelled_the_same_way():
    """Both people-reads carry a per-graph count, so both need the correction."""
    rows = [_topic_row("BK", 5, WIKIPEDIA), _topic_row("CF", 19, ARXIV)]

    facts = asyncio.run(_repo(rows).topic_contributors(_ctx(), "t"))

    assert _scores(facts) == {
        "BK": _damped(5, own_peak=5, ceiling=19, hops=2), "CF": _raw(19, hops=2)
    }


# ── 2. what the fix must NOT disturb ────────────────────────────────────────


def test_the_hops_correction_between_the_two_reads_survives():
    """The load-bearing property: the agent calls both reads in the SAME turn, so
    their scores are compared in one pool. Normalising each read to 1.0 would put
    both top rows at the same score and undo the divisor that stops the plentiful
    two-hop unit outranking the direct one."""
    direct = asyncio.run(_repo([_doc_row("Direct", 5, WIKIPEDIA)]
                               ).document_contributors(_ctx(), "t"))
    indirect = asyncio.run(_repo([_topic_row("Indirect", 40, WIKIPEDIA)]
                                 ).topic_contributors(_ctx(), "t"))

    assert direct[0].score == _raw(5, hops=1)
    assert indirect[0].score == _raw(40, hops=2)
    assert direct[0].score != 1.0 and indirect[0].score != 1.0


def test_single_graph_scores_are_byte_identical_to_the_raw_curve():
    """One graph ⇒ its peak IS the ceiling ⇒ the multiplier is exactly 1."""
    rows = [_doc_row("Heavy", 50, WIKIPEDIA), _doc_row("Mid", 10, WIKIPEDIA),
            _doc_row("Light", 1, WIKIPEDIA)]

    facts = asyncio.run(_repo(rows).document_contributors(_ctx(), "t"))

    assert _scores(facts) == {
        "Heavy": _raw(50), "Mid": _raw(10), "Light": _raw(1),
    }


def test_within_graph_order_and_spacing_are_preserved():
    """Levelling is a positive scalar per graph, so it is monotonic — the curve's
    own ranking inside each graph is the intent and must not move."""
    rows = [_doc_row("BK Heavy", 8, WIKIPEDIA), _doc_row("BK Light", 2, WIKIPEDIA),
            _doc_row("CF Heavy", 30, ARXIV), _doc_row("CF Light", 8, ARXIV)]

    facts = asyncio.run(_repo(rows).document_contributors(_ctx(), "t"))
    scores = _scores(facts)

    assert scores["BK Heavy"] > scores["BK Light"]
    assert scores["CF Heavy"] > scores["CF Light"]
    # The ratio inside a graph is untouched by a per-graph scalar.
    assert round(scores["BK Heavy"] / scores["BK Light"], 3) == round(
        _raw(8) / _raw(2), 3)


def test_an_untagged_row_keeps_its_own_score():
    """A single-graph deployment never carries the tag; those rows share one
    bucket, which is the previous behaviour exactly."""
    rows = [{"person": "P", "role": "AUTHORED", "document": "D",
             "contributions": 7}]

    facts = asyncio.run(_repo(rows).document_contributors(_ctx(), "t"))

    assert _scores(facts) == {"P": _raw(7)}


def test_zero_contributions_do_not_divide_by_zero():
    """A graph whose every row scores 0 has no peak to level against."""
    rows = [_doc_row("A", 0, WIKIPEDIA), _doc_row("B", 0, ARXIV)]

    facts = asyncio.run(_repo(rows).document_contributors(_ctx(), "t"))

    assert _scores(facts) == {"A": 0.0, "B": 0.0}


def test_one_graph_scoring_zero_leaves_the_other_untouched():
    rows = [_doc_row("Zero", 0, WIKIPEDIA), _doc_row("Real", 9, ARXIV)]

    facts = asyncio.run(_repo(rows).document_contributors(_ctx(), "t"))

    assert _scores(facts) == {"Zero": 0.0, "Real": _raw(9)}


def test_contributor_retrieval_semantics_are_unchanged():
    """Only `score` moves: the fact's shape, support convention and provenance
    are what the digest and the tool payloads read."""
    facts = asyncio.run(_repo([_doc_row("P", 5, WIKIPEDIA)]
                              ).document_contributors(_ctx(), "t"))

    f = facts[0]
    assert (f.source, f.target, f.target_type) == ("P", "D", "Document")
    assert f.relationship == "AUTHORED"
    assert f.support == 1, "authorship is a typed edge — involvement lives in score"
    assert f.graph == WIKIPEDIA


def test_the_read_is_deterministic():
    """Two identical requests score identically — no dependence on row order."""
    rows = [_doc_row("A", 5, WIKIPEDIA), _doc_row("B", 19, ARXIV)]

    first = asyncio.run(_repo(rows).document_contributors(_ctx(), "t"))
    second = asyncio.run(_repo(list(reversed(rows))).document_contributors(_ctx(), "t"))

    assert _scores(first) == _scores(second)
