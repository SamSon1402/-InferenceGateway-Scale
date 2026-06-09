"""In-process result store.

When a job is submitted, the request handler needs to block until the
worker finishes processing it. The :class:`ResultStore` mediates that
hand-off using one :class:`asyncio.Future` per outstanding job.

This implementation is single-process: it assumes the worker that picks
up the job runs in the same process as the request that submitted it.
In a multi-pod deployment that assumption breaks — you'd replace this
with Redis pub/sub or a per-job result key with ``BLPOP``. The
interface (``wait_for``, ``set_result``, ``set_error``) stays the same.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass


@dataclass
class JobResult:
    content: str | None
    error: str | None
    model: str | None
    process_ms: int
    attempts: int


class ResultStore:
    def __init__(self) -> None:
        self._waiters: dict[str, asyncio.Future[JobResult]] = {}

    def register(self, job_id: str) -> asyncio.Future[JobResult]:
        """Reserve a future before enqueueing the job. Returning it lets
        the request handler ``await`` directly without re-looking it up.
        """
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[JobResult] = loop.create_future()
        self._waiters[job_id] = fut
        return fut

    def set_result(self, job_id: str, result: JobResult) -> None:
        fut = self._waiters.pop(job_id, None)
        if fut and not fut.done():
            fut.set_result(result)

    def set_error(self, job_id: str, exc: Exception) -> None:
        fut = self._waiters.pop(job_id, None)
        if fut and not fut.done():
            fut.set_exception(exc)

    def cancel(self, job_id: str) -> None:
        fut = self._waiters.pop(job_id, None)
        if fut and not fut.done():
            fut.cancel()

    def outstanding(self) -> int:
        return len(self._waiters)
