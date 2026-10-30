"""get_claims ranks what it hands the model (agent-led redesign, 2026-08-08).

The raw fetch fans out over the graphs and CONCATENATES per-graph lists, so
its order was not even globally recency-sorted — and the tool used to return
every row (up to 60 per graph), dominating the round's tokens. Now the model
sees a bounded, ordered slice: embedding relevance to the agent's stated
``focus`` when one is given (rule 8 — semantics are the embedder's job, no
LLM spent), newest-first otherwise; the pool still keeps everything fetched.

    uv run pytest tests/test_claim_ranking.py -v
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from ai_graph_retrieval.agent.tools import _claim_text, _rank_claims
from ai_graph_retrieval.core.models import ClaimFact


def _claim(subject: str, predicate: str, obj: str, valid_from: str | None = None):
    return ClaimFact(subject=subject, predicate=predicate, object=obj,
                     valid_from=valid_from)


def _rc(embedder=None):
    return SimpleNamespace(embedder=embedder)


class _Embedder:
    """Real-shaped: get() -> one vector, embed_batch() -> {text: vector}."""

    def __init__(self, query_vec, batch):
        self._q = query_vec
        self._b = batch

    async def get(self, _q):
        return self._q

    async def embed_batch(self, _texts):
        return self._b


def test_without_a_focus_claims_are_globally_newest_first():
    # The defect this fixes: two graphs' recency-ordered lists concatenated is
    # not a recency order — Wikipedia's 2024 rows sat above arXiv's 2026.
    claims = [
        _claim("A", "released", "v1", "2024-01-01"),
        _claim("A", "released", "v3", "2026-03-01"),
        _claim("A", "released", "v2", "2025-06-01"),
        _claim("A", "mentioned", "x", None),  # undated sorts last
    ]
    out = asyncio.run(_rank_claims(_rc(), claims, focus=None))
    assert [c.object for c in out] == ["v3", "v2", "v1", "x"]


def test_a_focus_ranks_by_embedding_relevance():
    near = _claim("A", "owns", "the payroll system", "2024-01-01")
    far = _claim("A", "attended", "the summer party", "2026-01-01")
    embedder = _Embedder(
        query_vec=[1.0, 0.0],
        batch={_claim_text(near): [1.0, 0.0], _claim_text(far): [0.0, 1.0]},
    )
    out = asyncio.run(_rank_claims(_rc(embedder), [far, near], focus="payroll ownership"))
    # Relevance beats recency when the agent said what it is looking for.
    assert out[0] is near


def test_an_embedding_failure_falls_back_to_newest_first():
    class _Broken:
        async def get(self, _q):
            raise RuntimeError("embeddings down")

    claims = [_claim("A", "p", "old", "2020-01-01"), _claim("A", "p", "new", "2026-01-01")]
    out = asyncio.run(_rank_claims(_rc(_Broken()), claims, focus="anything"))
    assert [c.object for c in out] == ["new", "old"], \
        "a ranking detail must not cost the whole tool call"


def test_the_tool_offers_focus_to_the_model():
    """The parameter must be in the tool's schema, or the capability exists in
    code and not for the agent — the same silent-failure shape as anchors that
    never reached the prompt."""
    from ai_graph_retrieval.agent.tools import get_claims

    params = get_claims.params_json_schema["properties"]
    assert "focus" in params
    assert "months" in params
