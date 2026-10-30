"""In-process cumulative usage/cost aggregation for the `/usage` endpoint (F3).

The JSONL usage ledger is the durable, cross-replica record; this is a cheap
per-replica running total for at-a-glance operator visibility (cumulative spend,
per-mode volume, average cost/query, recent rows) without an offline log crunch.
Thread-safety is not needed — it is updated only from the request task in
`orchestrator._record`, one request at a time per event loop.
"""
from __future__ import annotations

from collections import deque
from typing import Any


class UsageAggregator:
    def __init__(self, recent_size: int = 50) -> None:
        self._queries = 0
        self._cost_usd = 0.0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._cached_prompt_tokens = 0
        self._embedding_tokens = 0
        self._llm_calls = 0
        self._by_mode: dict[str, dict[str, float]] = {}
        self._recent: deque[dict[str, Any]] = deque(maxlen=recent_size)

    def add(self, record: dict[str, Any]) -> None:
        """Fold one per-request usage record (the dict written to the usage ledger)."""
        self._queries += 1
        self._cost_usd += float(record.get("cost_usd", 0.0) or 0.0)
        self._prompt_tokens += int(record.get("prompt_tokens", 0) or 0)
        self._completion_tokens += int(record.get("completion_tokens", 0) or 0)
        self._cached_prompt_tokens += int(record.get("cached_prompt_tokens", 0) or 0)
        self._embedding_tokens += int(record.get("embedding_tokens", 0) or 0)
        self._llm_calls += int(record.get("llm_calls", 0) or 0)

        key = str(record.get("intent") or record.get("strategy") or "unknown")
        bucket = self._by_mode.setdefault(key, {"queries": 0, "cost_usd": 0.0})
        bucket["queries"] += 1
        bucket["cost_usd"] += float(record.get("cost_usd", 0.0) or 0.0)

        self._recent.append(record)

    def snapshot(self) -> dict[str, Any]:
        avg = round(self._cost_usd / self._queries, 6) if self._queries else 0.0
        return {
            "cumulative": {
                "queries": self._queries,
                "cost_usd": round(self._cost_usd, 6),
                "prompt_tokens": self._prompt_tokens,
                "completion_tokens": self._completion_tokens,
                "cached_prompt_tokens": self._cached_prompt_tokens,
                "embedding_tokens": self._embedding_tokens,
                "llm_calls": self._llm_calls,
            },
            "avg_cost_usd": avg,
            "by_mode": {
                k: {"queries": v["queries"], "cost_usd": round(v["cost_usd"], 6)}
                for k, v in self._by_mode.items()
            },
            "recent": list(self._recent),
            "note": "per-replica running total; the JSONL usage ledger is the durable record",
        }
