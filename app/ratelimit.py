"""Per-provider token bucket rate limiter.

Classic token bucket:

* Each bucket has a *capacity* (max burst) and a *refill rate* (tokens
  per second). Tokens regenerate continuously up to capacity.
* ``acquire(n)`` waits until ``n`` tokens are available, then deducts
  them. Workers call this before sending a request, sized to the
  estimated token count of the job.

Why per-token, not per-request?
    A 4 000-token completion consumes the same provider budget as forty
    100-token ones. A request-counting limiter under-protects the
    provider for big jobs and over-protects it for small ones. Token-
    weighted limits track what providers actually meter on.

Concurrency note: a single ``asyncio.Lock`` per bucket serialises
``acquire``. With one lock per provider, that's three locks total —
not a hot path.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from .schemas import BucketState, ProviderName


@dataclass
class _BucketSpec:
    capacity: int
    refill_per_s: float


class TokenBucket:
    """One bucket per provider."""

    def __init__(self, name: ProviderName, *, capacity: int, refill_per_s: float) -> None:
        if capacity <= 0 or refill_per_s <= 0:
            raise ValueError("capacity and refill must be positive")
        self.name = name
        self._capacity = capacity
        self._refill_per_s = refill_per_s
        self._tokens = float(capacity)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()
        # Stats — read by /v1/metrics
        self._throttle_events = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def refill_per_s(self) -> float:
        return self._refill_per_s

    @property
    def throttle_events(self) -> int:
        return self._throttle_events

    def _refill_unlocked(self) -> None:
        """Top up tokens based on time elapsed since last refill.

        Called under the lock so the read-modify-write of ``_tokens``
        and ``_last_refill`` is atomic w.r.t. other ``acquire`` calls.
        """
        now = time.monotonic()
        elapsed = now - self._last_refill
        if elapsed > 0:
            self._tokens = min(self._capacity, self._tokens + elapsed * self._refill_per_s)
            self._last_refill = now

    async def acquire(self, tokens: int) -> None:
        """Block until ``tokens`` are available, then deduct them.

        Implementation: compute the deficit, sleep for the time it
        would take to refill that deficit, then re-check. Avoids busy
        spinning; the worst-case extra wait is one refill tick.
        """
        if tokens <= 0:
            return
        if tokens > self._capacity:
            # Caller is asking for more than the bucket can ever hold;
            # cap the request at capacity so we don't deadlock.
            tokens = self._capacity

        while True:
            async with self._lock:
                self._refill_unlocked()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                deficit = tokens - self._tokens
                wait_s = deficit / self._refill_per_s

            # We were throttled — record it and sleep outside the lock.
            self._throttle_events += 1
            await asyncio.sleep(wait_s)

    async def try_acquire(self, tokens: int) -> bool:
        """Non-blocking variant — returns True if granted immediately."""
        if tokens <= 0:
            return True
        async with self._lock:
            self._refill_unlocked()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    def snapshot(self) -> BucketState:
        # Read without locking — values are advisory for /metrics.
        return BucketState(
            provider=self.name,
            capacity=self._capacity,
            available=int(self._tokens),
            refill_per_s=self._refill_per_s,
            throttled=self._tokens < self._capacity * 0.05,
        )


class RateLimiter:
    """Container that owns one bucket per provider."""

    def __init__(self, specs: dict[ProviderName, _BucketSpec]) -> None:
        self._buckets: dict[ProviderName, TokenBucket] = {
            name: TokenBucket(name, capacity=spec.capacity, refill_per_s=spec.refill_per_s)
            for name, spec in specs.items()
        }

    def bucket(self, provider: ProviderName) -> TokenBucket:
        return self._buckets[provider]

    def snapshots(self) -> list[BucketState]:
        return [b.snapshot() for b in self._buckets.values()]

    def total_throttle_events(self) -> int:
        return sum(b.throttle_events for b in self._buckets.values())


# Helper to build the limiter from Settings without leaking Pydantic
# field names into the limiter itself.
def build_rate_limiter(
    *,
    openai: tuple[int, float],
    anthropic: tuple[int, float],
    mistral: tuple[int, float],
) -> RateLimiter:
    return RateLimiter({
        ProviderName.OPENAI:    _BucketSpec(*openai),
        ProviderName.ANTHROPIC: _BucketSpec(*anthropic),
        ProviderName.MISTRAL:   _BucketSpec(*mistral),
    })
