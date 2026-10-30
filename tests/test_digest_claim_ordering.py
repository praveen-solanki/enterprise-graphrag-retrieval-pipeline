"""The digest's CLAIMS cap follows the claim ranking, not arrival order.

The pool holds the fan-out's concatenation — one graph's rows, then the next,
each block already `valid_from DESC` from cypher.FETCH_CLAIMS — so slicing it
positionally kept the first graph's OLDEST claims and dropped the second
graph's newest. Measured shape: five expired 2017-2021 claims rendered while
the current, best-supported 2026 claim never reached the model.

    uv run pytest tests/test_digest_claim_ordering.py -v
"""
from __future__ import annotations

from ai_graph_retrieval.core.models import ClaimFact, Evidence
from ai_graph_retrieval.synthesis.composer import AnswerComposer
from tests.support import make_settings

WIKIPEDIA, ARXIV = "wikipedia", "arxiv"


def _claim(obj: str, valid_from: str | None, graph: str, *,
           support: int = 1, is_current: bool = True) -> ClaimFact:
    return ClaimFact(subject="Search", predicate="owns", object=obj,
                     valid_from=valid_from, graph=graph, support=support,
                     is_current=is_current)


def _rendered(ev: Evidence, **overrides) -> list[str]:
    """The CLAIMS section's objects, in rendered order."""
    digest = AnswerComposer(make_settings(**overrides))._digest(ev, strict=False)
    if "CLAIMS:" not in digest:
        return []
    section = digest[digest.index("CLAIMS:"):]
    section = section[: section.index("\n\n")] if "\n\n" in section else section
    return [line.split(" owns ")[1].split(" (support")[0]
            for line in section.splitlines() if line.startswith("- (")]


def _two_graph_pool() -> Evidence:
    """Wikipedia's block (older) first, then arXiv's — the fan-out shape."""
    return Evidence(claims=[
        _claim("bk-2021", "2021-01-01", WIKIPEDIA, is_current=False),
        _claim("bk-2020", "2020-01-01", WIKIPEDIA, is_current=False),
        _claim("bk-2019", "2019-01-01", WIKIPEDIA, is_current=False),
        _claim("cf-2026", "2026-08-01", ARXIV, support=9),
        _claim("cf-2025", "2025-01-01", ARXIV, support=7),
    ])


def test_a_later_graphs_current_claim_is_not_dropped_by_the_cap():
    kept = _rendered(_two_graph_pool(), digest_max_claims=2)

    assert kept == ["cf-2026", "cf-2025"]


def test_the_whole_section_is_ordered_not_just_the_survivors():
    kept = _rendered(_two_graph_pool(), digest_max_claims=5)

    assert kept == ["cf-2026", "cf-2025", "bk-2021", "bk-2020", "bk-2019"]


def test_expired_claims_are_ordered_never_dropped():
    """Historical recall is deliberate (cypher.FETCH_CLAIMS keeps expired rows
    and the rendered line flags them) — ranking must not filter them out."""
    ev = Evidence(claims=[
        _claim("current", "2020-01-01", WIKIPEDIA),
        _claim("expired", "2026-01-01", ARXIV, is_current=False),
    ])

    assert _rendered(ev) == ["expired", "current"]


def test_undated_claims_sort_last_but_still_render():
    ev = Evidence(claims=[
        _claim("undated", None, WIKIPEDIA),
        _claim("dated", "2024-01-01", ARXIV),
    ])

    assert _rendered(ev) == ["dated", "undated"]


def test_single_graph_order_is_unchanged():
    """One graph arrives already `valid_from DESC`; ranking must be a no-op."""
    ev = Evidence(claims=[
        _claim("newest", "2026-01-01", WIKIPEDIA),
        _claim("middle", "2025-01-01", WIKIPEDIA),
        _claim("oldest", "2024-01-01", WIKIPEDIA),
    ])

    assert _rendered(ev) == ["newest", "middle", "oldest"]
    assert _rendered(ev, digest_max_claims=2) == ["newest", "middle"]


def test_the_cap_still_bounds_the_section():
    ev = Evidence(claims=[
        _claim(f"c{i}", f"20{i:02d}-01-01", WIKIPEDIA) for i in range(10, 30)
    ])

    assert len(_rendered(ev, digest_max_claims=4)) == 4


def test_ordering_is_deterministic_across_identical_requests():
    ev = _two_graph_pool()

    assert _rendered(ev, digest_max_claims=3) == _rendered(ev, digest_max_claims=3)


def test_the_available_count_still_reports_the_whole_pool():
    """Ranking reorders; it must not change what the span says was available."""
    ev = _two_graph_pool()
    _digest, counts, _rows = AnswerComposer(
        make_settings(digest_max_claims=2)
    )._build_digest(ev, strict=False)

    assert counts["claims_available"] == 5
    assert counts["claims_rendered"] == 2


def test_other_digest_sections_are_untouched():
    """The change is scoped to CLAIMS: entity and fact ordering keep their own
    ranking keys."""
    from ai_graph_retrieval.core.models import EntityHit, RelationFact

    ev = Evidence(
        entities=[EntityHit(name="Bare"), EntityHit(name="Profiled", profile="VP")],
        facts=[RelationFact(source="A", target="Weak", relationship="R", support=1),
               RelationFact(source="A", target="Strong", relationship="R", support=9)],
        claims=[_claim("only", "2024-01-01", WIKIPEDIA)],
    )
    digest = AnswerComposer(make_settings())._digest(ev, strict=False)

    assert digest.index("Profiled") < digest.index("Bare")
    assert digest.index("Strong") < digest.index("Weak")
