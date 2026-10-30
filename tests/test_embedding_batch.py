"""RequestEmbedder.embed_batch — one batched embedding call, deliberately cache-free.

Guards the properties that matter on the hot path: exactly one API call per batch,
deduped and blank-free input, no coupling to the ``_futures`` cache (so it can never
install a pending future another caller awaits), and a fail-open empty mapping.

    uv run pytest tests/test_embedding_batch.py -v
"""
from __future__ import annotations

import asyncio

from ai_graph_retrieval.connectors.llm.embeddings import RequestEmbedder
from ai_graph_retrieval.core.models import Usage


class _Llm:
    """Records each embed() batch so we can assert on call count and contents."""

    def __init__(self, fail: bool = False, short: bool = False):
        self.batches: list[list[str]] = []
        self._fail = fail
        self._short = short

    async def embed(self, texts, *, dimensions=None, usage=None):
        self.batches.append(list(texts))
        if self._fail:
            raise RuntimeError("gateway down")
        vecs = [[float(len(t)), 0.0] for t in texts]
        return vecs[:-1] if self._short and vecs else vecs


def _emb(llm):
    return RequestEmbedder(llm, Usage())


def test_batches_all_texts_into_one_call_and_dedupes():
    llm = _Llm()
    out = asyncio.run(_emb(llm).embed_batch(["A", "B", "A", "  ", ""]))
    assert len(llm.batches) == 1                  # ONE API call
    assert sorted(llm.batches[0]) == ["A", "B"]   # deduped, blanks dropped
    assert set(out) == {"A", "B"}                 # blanks absent from the result


def test_blank_only_input_makes_no_call():
    llm = _Llm()
    assert asyncio.run(_emb(llm).embed_batch(["", "   "])) == {}
    assert llm.batches == []


def test_batch_failure_returns_an_empty_mapping():
    llm = _Llm(fail=True)
    assert asyncio.run(_emb(llm).embed_batch(["A", "B"])) == {}   # missing == no signal


def test_batch_and_get_stay_independent():
    # Public-behavior contract: a batch call neither blocks nor pre-populates the
    # per-request get() path — get() for the same text still makes its own call,
    # and both return the same vector. (No assertion on internals.)
    llm = _Llm()
    e = _emb(llm)

    async def scenario():
        batch = await e.embed_batch(["A"])
        single = await e.get("A")
        return batch, single

    batch, single = asyncio.run(scenario())
    assert len(llm.batches) == 2          # independent calls, no hidden coupling
    assert batch["A"] == single           # and an identical result either way


def test_short_response_omits_the_missing_text():
    llm = _Llm(short=True)
    out = asyncio.run(_emb(llm).embed_batch(["A", "B"]))
    assert set(out) <= {"A", "B"}
    assert all(v for v in out.values())           # never a None/empty vector


