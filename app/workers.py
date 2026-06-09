"""Async worker pool.

Each :class:`Worker` is a coroutine that pulls jobs off the queue,
acquires tokens from the per-provider rate limiter, calls the provider
with retry/backoff on 429s and 5xx, and writes the result back to the
:class:`ResultStore` so the original request handler can return.

The :class:`WorkerPool` owns the workers as tasks. The autoscaler calls
``pool.scale_to(n)`` to grow or shrink. Scale-down is cooperative — we
flip the stop flag on workers we want to retire, then drain.
"""
from __future__ import annotations

import asyncio
import logging
import time

from .providers import (
    Provider,
    ProviderError,
    ProviderRegistry,
    RateLimitedError,
    RetryPolicy,
)
from .queue import JobQueue
from .ratelimit import RateLimiter
from .schemas import Job, JobState
from .store import JobResult, ResultStore

logger = logging.getLogger(__name__)


class Worker:
    """One consumer loop. Stops cleanly when ``stop()`` is called."""

    _next_id = 0

    def __init__(
        self,
        *,
        queue: JobQueue,
        registry: ProviderRegistry,
        limiter: RateLimiter,
        store: ResultStore,
        retry: RetryPolicy,
        on_429: callable,  # type: ignore[type-arg]
    ) -> None:
        Worker._next_id += 1
        self.id = Worker._next_id
        self._queue = queue
        self._registry = registry
        self._limiter = limiter
        self._store = store
        self._retry = retry
        self._on_429 = on_429
        self._stop = asyncio.Event()
        self._inflight = False

    @property
    def inflight(self) -> bool:
        return self._inflight

    def stop(self) -> None:
        self._stop.set()

    async def run(self) -> None:
        logger.info("worker_started id=%d", self.id)
        try:
            while not self._stop.is_set():
                # Short queue timeout so the loop checks the stop flag
                # promptly when the autoscaler trims the pool.
                job = await self._queue.dequeue(timeout=1.0)
                if job is None:
                    continue
                self._inflight = True
                try:
                    await self._process(job)
                finally:
                    self._inflight = False
        finally:
            logger.info("worker_stopped id=%d", self.id)

    async def _process(self, job: Job) -> None:
        provider_name = job.request.provider
        try:
            provider: Provider = self._registry.get(provider_name)
        except ProviderError as exc:
            self._store.set_error(job.id, exc)
            return

        bucket = self._limiter.bucket(provider_name)
        # Token bucket gate — sized to the *estimated* token usage so a
        # 4k-token job doesn't get treated the same as a 100-token one.
        await bucket.acquire(job.request.estimated_tokens)

        start = time.perf_counter()
        content: str | None = None
        last_exc: Exception | None = None

        for attempt in range(self._retry.max_attempts):
            job.attempts = attempt + 1
            try:
                content = await provider.call(
                    job.request.messages,
                    max_tokens=job.request.max_tokens,
                    temperature=job.request.temperature,
                )
                break
            except RateLimitedError as exc:
                self._on_429()
                last_exc = exc
                # Honour Retry-After if present, else exponential backoff.
                wait_s = exc.retry_after_s if exc.retry_after_s is not None else self._retry.backoff(attempt)
                logger.warning(
                    "worker=%d retry attempt=%d wait=%.2fs reason=429",
                    self.id, attempt + 1, wait_s,
                )
                await asyncio.sleep(wait_s)
                continue
            except ProviderError as exc:
                last_exc = exc
                break  # non-retryable

        process_ms = int((time.perf_counter() - start) * 1000)

        if content is None:
            err = str(last_exc or "exhausted retries")
            self._store.set_result(job.id, JobResult(
                content=None, error=err, model=None,
                process_ms=process_ms, attempts=job.attempts,
            ))
            return

        self._store.set_result(job.id, JobResult(
            content=content, error=None, model=provider.model,
            process_ms=process_ms, attempts=job.attempts,
        ))


class WorkerPool:
    """Owns a dynamic set of workers; the autoscaler resizes via ``scale_to``."""

    def __init__(
        self,
        *,
        queue: JobQueue,
        registry: ProviderRegistry,
        limiter: RateLimiter,
        store: ResultStore,
        retry: RetryPolicy,
        min_size: int,
        max_size: int,
    ) -> None:
        if min_size < 1 or max_size < min_size:
            raise ValueError("require 1 <= min_size <= max_size")
        self._queue = queue
        self._registry = registry
        self._limiter = limiter
        self._store = store
        self._retry = retry
        self._min = min_size
        self._max = max_size
        self._workers: list[Worker] = []
        self._tasks: list[asyncio.Task] = []
        # Stats
        self._429_total = 0

    @property
    def size(self) -> int:
        return len(self._workers)

    @property
    def inflight(self) -> int:
        return sum(1 for w in self._workers if w.inflight)

    @property
    def total_429(self) -> int:
        return self._429_total

    def _spawn(self) -> None:
        w = Worker(
            queue=self._queue,
            registry=self._registry,
            limiter=self._limiter,
            store=self._store,
            retry=self._retry,
            on_429=self._record_429,
        )
        self._workers.append(w)
        self._tasks.append(asyncio.create_task(w.run(), name=f"worker-{w.id}"))

    def _record_429(self) -> None:
        self._429_total += 1

    async def start(self, initial: int) -> None:
        initial = max(self._min, min(self._max, initial))
        for _ in range(initial):
            self._spawn()

    async def scale_to(self, target: int) -> None:
        target = max(self._min, min(self._max, target))
        current = self.size
        if target > current:
            for _ in range(target - current):
                self._spawn()
            logger.info("pool_scaled_up from=%d to=%d", current, target)
        elif target < current:
            # Cooperative shutdown: signal the last (target..current)
            # workers to stop after their current job. They drain on
            # their own timeline.
            for w in self._workers[target:]:
                w.stop()
            self._workers = self._workers[:target]
            logger.info("pool_scaled_down from=%d to=%d", current, target)

    async def shutdown(self) -> None:
        for w in self._workers:
            w.stop()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._workers.clear()
        self._tasks.clear()
