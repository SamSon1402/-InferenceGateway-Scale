"""Priority job queue — Redis-backed in prod, in-memory for tests.

The queue exposes a tiny interface so it can be swapped without touching
workers or the autoscaler:

    enqueue(job, priority)  → None
    dequeue(timeout)        → Job | None
    size_by_priority()      → dict[Priority, int]

The Redis implementation uses one list per priority lane and a blocking
``BLPOP`` across all three keys. BLPOP scans keys in argument order, so
high-priority items are always served before normal/low — no separate
scheduler needed.
"""
from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from contextlib import suppress

import redis.asyncio as redis_async

from .schemas import Job, Priority

logger = logging.getLogger(__name__)


class JobQueue(ABC):
    @abstractmethod
    async def enqueue(self, job: Job, priority: Priority) -> None: ...

    @abstractmethod
    async def dequeue(self, *, timeout: float = 1.0) -> Job | None: ...

    @abstractmethod
    async def size_by_priority(self) -> dict[Priority, int]: ...

    async def total_size(self) -> int:
        return sum((await self.size_by_priority()).values())


# --------------------------------------------------------------------------- #
# In-memory — for tests, CI, and local dev when Redis isn't available         #
# --------------------------------------------------------------------------- #

class InMemoryQueue(JobQueue):
    def __init__(self) -> None:
        # One ``asyncio.Queue`` per priority. Dequeue scans them in order.
        self._lanes: dict[Priority, asyncio.Queue[Job]] = {
            p: asyncio.Queue() for p in Priority
        }

    async def enqueue(self, job: Job, priority: Priority) -> None:
        await self._lanes[priority].put(job)

    async def dequeue(self, *, timeout: float = 1.0) -> Job | None:
        # Cheap path: any lane already has an item → return it.
        for p in Priority:  # already in priority order
            if not self._lanes[p].empty():
                return self._lanes[p].get_nowait()

        # Slow path: wait on any lane. Use ``asyncio.wait`` so we don't
        # tie ourselves to the lowest-priority lane's blocking get.
        getters = [asyncio.create_task(self._lanes[p].get()) for p in Priority]
        try:
            done, pending = await asyncio.wait(
                getters, timeout=timeout, return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await t
            if not done:
                return None
            # Re-enqueue anything we accidentally won that wasn't the highest
            # priority available (rare but possible under contention).
            results = sorted(
                (t.result() for t in done if not t.cancelled()),
                key=lambda j: j.request.priority,
            )
            winner = results[0]
            for extra in results[1:]:
                await self._lanes[extra.request.priority].put(extra)
            return winner
        finally:
            for t in getters:
                if not t.done():
                    t.cancel()

    async def size_by_priority(self) -> dict[Priority, int]:
        return {p: self._lanes[p].qsize() for p in Priority}


# --------------------------------------------------------------------------- #
# Redis — production backend                                                  #
# --------------------------------------------------------------------------- #

class RedisQueue(JobQueue):
    """Redis-backed priority queue using one list per lane.

    BLPOP takes multiple keys and pops from the *first non-empty* one,
    which gives us priority ordering for free across whatever set of
    keys we pass — no Lua script needed.
    """

    def __init__(self, client: redis_async.Redis, namespace: str = "iqs") -> None:
        self._r = client
        self._keys = {p: f"{namespace}:queue:{p.name.lower()}" for p in Priority}

    async def enqueue(self, job: Job, priority: Priority) -> None:
        await self._r.rpush(self._keys[priority], job.model_dump_json())

    async def dequeue(self, *, timeout: float = 1.0) -> Job | None:
        # BLPOP keys... timeout — scans keys in order, returns (key, value)
        # for the first non-empty one. Timeout=0 would block forever; we
        # use a finite timeout so the worker can check its stop flag.
        key_order = [self._keys[p] for p in Priority]
        result = await self._r.blpop(key_order, timeout=timeout)
        if result is None:
            return None
        _key, payload = result
        return Job.model_validate_json(payload)

    async def size_by_priority(self) -> dict[Priority, int]:
        pipe = self._r.pipeline()
        for p in Priority:
            pipe.llen(self._keys[p])
        sizes = await pipe.execute()
        return {p: int(n) for p, n in zip(Priority, sizes)}
