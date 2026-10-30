"""JsonlLedger — a non-blocking, bounded JSONL sink for usage/audit entries.

Design constraints (this sits on the request hot path):
  * `record()` never blocks and never raises — a full queue drops the entry
    and counts the drop (visibility over backpressure; the structured log
    stream is the durable channel, the file ledger is a convenience).
  * one writer task per ledger; file I/O is offloaded to a worker thread so
    the event loop never touches disk;
  * entries are drained in batches (one write + flush per wakeup, not per
    entry) to keep syscall volume flat under load;
  * shutdown drains whatever is queued before closing.
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

_SENTINEL = None


class JsonlLedger:
    def __init__(self, path: str | None, *, name: str, queue_size: int = 2048) -> None:
        self._path = path
        self._name = name
        self._queue: asyncio.Queue[dict | None] = asyncio.Queue(maxsize=queue_size)
        self._writer: asyncio.Task | None = None
        self._dropped = 0

    @property
    def enabled(self) -> bool:
        return self._path is not None

    async def start(self) -> None:
        if not self.enabled or self._writer is not None:
            return
        directory = os.path.dirname(self._path or "")
        if directory:
            await asyncio.to_thread(os.makedirs, directory, exist_ok=True)
        self._writer = asyncio.create_task(self._run(), name=f"ledger:{self._name}")
        logger.info("ledger_started", ledger=self._name, path=self._path)

    def record(self, entry: dict[str, Any]) -> None:
        """Queue one entry; drop (and count) when the sink cannot keep up."""
        if not self.enabled or self._writer is None:
            return
        try:
            self._queue.put_nowait(entry)
        except asyncio.QueueFull:
            self._dropped += 1
            if self._dropped % 100 == 1:  # log the first drop and every 100th
                logger.warning("ledger_dropping_entries", ledger=self._name,
                               dropped=self._dropped)

    async def aclose(self) -> None:
        if self._writer is None:
            return
        await self._queue.put(_SENTINEL)
        try:
            await asyncio.wait_for(self._writer, timeout=5.0)
        except (TimeoutError, asyncio.CancelledError):
            self._writer.cancel()
        self._writer = None
        logger.info("ledger_stopped", ledger=self._name, dropped=self._dropped)

    # ── writer task ────────────────────────────────────────────────────────

    async def _run(self) -> None:
        while True:
            entry = await self._queue.get()
            if entry is _SENTINEL:
                return
            batch = [entry]
            while not self._queue.empty():  # drain what accumulated
                nxt = self._queue.get_nowait()
                if nxt is _SENTINEL:
                    await self._write(batch)
                    return
                batch.append(nxt)
            await self._write(batch)

    async def _write(self, batch: list[dict]) -> None:
        lines = "".join(
            json.dumps(entry, ensure_ascii=False, default=str) + "\n" for entry in batch
        )
        try:
            await asyncio.to_thread(self._append, lines)
        except OSError as exc:  # disk trouble must never surface to requests
            self._dropped += len(batch)
            logger.error("ledger_write_failed", ledger=self._name, error=str(exc))

    def _append(self, lines: str) -> None:
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(lines)
