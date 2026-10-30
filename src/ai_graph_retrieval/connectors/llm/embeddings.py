"""Per-request embedding cache — embed each distinct text once per request.

The POC re-embedded the question inside every tool call; here router,
strategies, and all agent tools share one cache. Futures de-duplicate
concurrent requests for the same text (two parallel tools embedding the same
sub-goal trigger a single API call).

Failure contract: callers treat a `None` embedding as "semantic tier
unavailable" and degrade to lexical retrieval (POC behavior, kept).
"""
from __future__ import annotations

import asyncio

import structlog

from ai_graph_retrieval.connectors.llm.client import LlmClient
from ai_graph_retrieval.core.models import Usage
from ai_graph_retrieval.core.run_config import EmbeddingSpace

logger = structlog.get_logger(__name__)


class RequestEmbedder:
    def __init__(
        self, llm: LlmClient, usage: Usage, *, space: EmbeddingSpace | None = None
    ) -> None:
        self._llm = llm
        self._usage = usage
        # Per-request run selection: the embedding space a resolved run's vectors
        # were built in. Fills the DEFAULT dimension + deployment so query vectors
        # match the run's collection; an explicit ``dimensions=`` at a call site
        # still wins (a caller pinning the chunk space). None ⇒ configured defaults.
        self._space = space
        # Keyed by (dimension, text): the same text embedded into the 3072-d graph
        # space and the 1536-d chunk space are distinct vectors and must not collide.
        self._futures: dict[tuple[int | None, str], asyncio.Future[list[float] | None]] = {}

    def _dims(self, dimensions: int | None) -> int | None:
        if dimensions is not None:
            return dimensions
        return self._space.dimensions if self._space else None

    def _extra(self) -> dict:
        """Deployment override, ONLY when a run space names one — so the default
        path calls ``embed`` with exactly today's kwargs (keeps callers/mocks that
        predate the ``model`` param working)."""
        deployment = self._space.deployment if self._space else None
        return {"model": deployment} if deployment else {}

    async def get(self, text: str, *, dimensions: int | None = None) -> list[float] | None:
        key = text.strip()
        if not key:
            return None
        dims = self._dims(dimensions)
        cache_key = (dims, key)
        existing = self._futures.get(cache_key)
        if existing is not None:
            return await asyncio.shield(existing)

        future: asyncio.Future[list[float] | None] = asyncio.get_running_loop().create_future()
        self._futures[cache_key] = future
        try:
            vectors = await self._llm.embed(
                [key], dimensions=dims, usage=self._usage, **self._extra()
            )
            future.set_result(vectors[0])
        except Exception as exc:  # degrade to lexical-only retrieval
            logger.warning("embedding_unavailable", error=str(exc))
            future.set_result(None)
        except BaseException:  # cancellation — never leave an unresolvable future cached
            self._futures.pop(cache_key, None)
            future.cancel()
            raise
        return future.result()

    async def embed_batch(
        self, texts: list[str], *, dimensions: int | None = None
    ) -> dict[str, list[float]]:
        """Embed several short texts in ONE call, returning {text: vector}.

        Deliberately simple and cache-free. It exists for one-off batches — the
        document titles of a single request — that ``get``'s per-request cache would
        never be asked for again. Staying out of ``_futures`` means this never
        installs a pending future another caller could end up awaiting, which is the
        behaviour that matters on the hot path; at worst a text is embedded twice.

        Blank texts are dropped. On failure the mapping is empty, so a caller treats
        a missing vector as "no signal" — the same degrade contract as ``get``.
        """
        keys = sorted({t.strip() for t in texts if t and t.strip()})
        if not keys:
            return {}
        try:
            vectors = await self._llm.embed(
                keys, dimensions=self._dims(dimensions), usage=self._usage, **self._extra()
            )
        except Exception as exc:  # semantic tier unavailable — caller degrades
            logger.warning("embedding_batch_unavailable", error=str(exc), n=len(keys))
            return {}
        return {k: v for k, v in zip(keys, vectors, strict=False) if v}

