"""HPA-style autoscaler for the worker pool.

Uses the same proportional formula Kubernetes' Horizontal Pod Autoscaler
applies::

    desired_replicas = ceil(current_replicas * (current_metric / target_metric))

The metric here is ``queue_depth / current_workers``. When queue depth
rises, the formula scales up proportionally; when it falls, it scales
down. Independent cooldown timers for up and down (down is slower, to
avoid flap).

A small tolerance band (≈10%) prevents thrash when the metric hovers
around the target — matches K8s HPA behavior.
"""
from __future__ import annotations

import asyncio
import logging
import math
import time

from .queue import JobQueue
from .workers import WorkerPool

logger = logging.getLogger(__name__)


class Autoscaler:
    # Same tolerance HPA uses by default.
    _TOLERANCE = 0.10

    def __init__(
        self,
        *,
        pool: WorkerPool,
        queue: JobQueue,
        target_queue_per_worker: float = 10.0,
        check_interval_s: float = 5.0,
        scale_up_cooldown_s: float = 30.0,
        scale_down_cooldown_s: float = 60.0,
    ) -> None:
        self._pool = pool
        self._queue = queue
        self._target = target_queue_per_worker
        self._interval = check_interval_s
        self._cooldown_up = scale_up_cooldown_s
        self._cooldown_down = scale_down_cooldown_s
        self._last_scale_up = 0.0
        self._last_scale_down = 0.0
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    async def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run(), name="autoscaler")
        logger.info(
            "autoscaler_started target_qpw=%.1f interval=%.1fs",
            self._target, self._interval,
        )

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except asyncio.TimeoutError:
                self._task.cancel()

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                await self._tick()
            except Exception:
                logger.exception("autoscaler tick failed")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._interval)
            except asyncio.TimeoutError:
                pass

    async def _tick(self) -> None:
        current = self._pool.size
        depth = await self._queue.total_size()
        # HPA proportional formula. Use current QPW as the "current metric".
        current_metric = depth / max(current, 1)
        ratio = current_metric / self._target

        # Tolerance band — don't act on tiny deviations.
        if abs(ratio - 1.0) < self._TOLERANCE:
            return

        desired = math.ceil(current * ratio)
        now = time.monotonic()

        if desired > current:
            if now - self._last_scale_up < self._cooldown_up:
                return
            await self._pool.scale_to(desired)
            self._last_scale_up = now
            logger.info(
                "scale_up depth=%d workers=%d→%d qpw=%.1f target=%.1f",
                depth, current, self._pool.size, current_metric, self._target,
            )
        elif desired < current:
            if now - self._last_scale_down < self._cooldown_down:
                return
            await self._pool.scale_to(desired)
            self._last_scale_down = now
            logger.info(
                "scale_down depth=%d workers=%d→%d qpw=%.1f target=%.1f",
                depth, current, self._pool.size, current_metric, self._target,
            )
