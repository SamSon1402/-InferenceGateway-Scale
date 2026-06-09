"""FastAPI application — wires queue, workers, autoscaler, and rate limiter.

Endpoints:

    POST /v1/completions    enqueue + wait (sync-feeling API; queue inside)
    GET  /v1/metrics        operational metrics — queue depth, workers, 429s, buckets
    GET  /healthz           liveness probe for K8s
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

import httpx
import redis.asyncio as redis_async
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .autoscaler import Autoscaler
from .config import get_settings
from .providers import (
    AnthropicProvider,
    MistralProvider,
    OpenAIProvider,
    Provider,
    ProviderRegistry,
    RetryPolicy,
)
from .queue import InMemoryQueue, JobQueue, RedisQueue
from .ratelimit import RateLimiter, build_rate_limiter
from .schemas import (
    CompletionRequest,
    CompletionResponse,
    Job,
    JobState,
    Metrics,
    Priority,
    ProviderName,
)
from .store import ResultStore
from .workers import WorkerPool

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Wiring                                                                      #
# --------------------------------------------------------------------------- #

def _build_providers(http: httpx.AsyncClient) -> ProviderRegistry:
    s = get_settings()
    providers: dict[ProviderName, Provider] = {}
    if s.openai_api_key:
        providers[ProviderName.OPENAI] = OpenAIProvider(s.openai_api_key, http)
    if s.anthropic_api_key:
        providers[ProviderName.ANTHROPIC] = AnthropicProvider(s.anthropic_api_key, http)
    if s.mistral_api_key:
        providers[ProviderName.MISTRAL] = MistralProvider(s.mistral_api_key, http)
    if not providers:
        raise RuntimeError(
            "no providers configured — set at least one of "
            "OPENAI_API_KEY, ANTHROPIC_API_KEY, MISTRAL_API_KEY"
        )
    return ProviderRegistry(providers)


async def _build_queue() -> tuple[JobQueue, redis_async.Redis | None]:
    s = get_settings()
    if s.queue_backend == "memory":
        return InMemoryQueue(), None
    client = redis_async.from_url(s.redis_url, decode_responses=True)
    # Validate the connection upfront — better to fail at startup than on
    # the first request.
    await client.ping()
    return RedisQueue(client, namespace=s.queue_namespace), client


# --------------------------------------------------------------------------- #
# Lifespan                                                                    #
# --------------------------------------------------------------------------- #

@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    logging.basicConfig(
        level=s.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    http = httpx.AsyncClient(
        timeout=httpx.Timeout(s.request_timeout_s, connect=s.connect_timeout_s),
        limits=httpx.Limits(max_keepalive_connections=64, max_connections=128),
    )
    registry = _build_providers(http)
    limiter = build_rate_limiter(
        openai=(s.openai_capacity, s.openai_refill_per_s),
        anthropic=(s.anthropic_capacity, s.anthropic_refill_per_s),
        mistral=(s.mistral_capacity, s.mistral_refill_per_s),
    )
    queue, redis_client = await _build_queue()
    store = ResultStore()
    retry = RetryPolicy(
        max_attempts=s.retry_max_attempts,
        initial_backoff_s=s.retry_initial_backoff_s,
        max_backoff_s=s.retry_max_backoff_s,
    )
    pool = WorkerPool(
        queue=queue, registry=registry, limiter=limiter, store=store,
        retry=retry, min_size=s.min_workers, max_size=s.max_workers,
    )
    await pool.start(initial=s.initial_workers)
    autoscaler = Autoscaler(
        pool=pool, queue=queue,
        target_queue_per_worker=s.target_queue_per_worker,
        check_interval_s=s.autoscaler_interval_s,
        scale_up_cooldown_s=s.scale_up_cooldown_s,
        scale_down_cooldown_s=s.scale_down_cooldown_s,
    )
    await autoscaler.start()

    app.state.http = http
    app.state.queue = queue
    app.state.store = store
    app.state.pool = pool
    app.state.limiter = limiter
    app.state.autoscaler = autoscaler
    app.state.redis_client = redis_client

    logger.info(
        "startup_complete providers=%s workers=%d queue=%s",
        [n.value for n in registry.names()], pool.size, s.queue_backend,
    )
    try:
        yield
    finally:
        await autoscaler.stop()
        await pool.shutdown()
        await http.aclose()
        if redis_client is not None:
            await redis_client.aclose()
        logger.info("shutdown_complete")


app = FastAPI(
    title="InferenceGateway-Scale",
    version="0.1.0",
    description=(
        "Scalable LLM inference gateway with Redis-backed priority queue, "
        "per-provider token-bucket rate limiting, and HPA-style worker autoscaling."
    ),
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# Routes                                                                      #
# --------------------------------------------------------------------------- #

@app.get("/healthz", tags=["ops"])
async def healthz(request: Request) -> JSONResponse:
    queue: JobQueue = request.app.state.queue
    pool: WorkerPool = request.app.state.pool
    try:
        depth = await queue.total_size()
        ok = True
    except Exception as exc:
        depth = -1
        ok = False
        logger.exception("healthz queue check failed")
    status_code = 200 if ok else 503
    return JSONResponse(
        status_code=status_code,
        content={"ok": ok, "workers": pool.size, "queue_depth": depth,
                 "inflight": pool.inflight},
    )


@app.get("/v1/metrics", response_model=Metrics, tags=["ops"])
async def metrics(request: Request) -> Metrics:
    queue: JobQueue = request.app.state.queue
    pool: WorkerPool = request.app.state.pool
    limiter: RateLimiter = request.app.state.limiter
    s = get_settings()
    by_pri = await queue.size_by_priority()
    return Metrics(
        workers=pool.size,
        min_workers=s.min_workers,
        max_workers=s.max_workers,
        queue_depth_total=sum(by_pri.values()),
        queue_depth_by_priority={p.name.lower(): n for p, n in by_pri.items()},
        rate_limit_429_total=pool.total_429,
        inflight=pool.inflight,
        buckets=limiter.snapshots(),
    )


@app.post("/v1/completions", response_model=CompletionResponse, tags=["completions"])
async def completions(req: CompletionRequest, request: Request) -> CompletionResponse:
    """Enqueue the job, wait for a worker to pick it up, return the result.

    Sync-feeling API from the caller's perspective; under the hood it's
    a queue + worker pool. Sync timeout caps how long we wait — a 504
    is returned if no worker completes the job in time.
    """
    s = get_settings()
    queue: JobQueue = request.app.state.queue
    store: ResultStore = request.app.state.store

    job = Job(request=req)
    fut = store.register(job.id)
    enqueue_t = asyncio.get_running_loop().time()
    await queue.enqueue(job, req.priority)

    try:
        result = await asyncio.wait_for(fut, timeout=s.sync_timeout_s)
    except asyncio.TimeoutError:
        store.cancel(job.id)
        raise HTTPException(
            status_code=504,
            detail=f"job {job.id} did not complete within {s.sync_timeout_s}s",
        )

    queue_wait_ms = int((asyncio.get_running_loop().time() - enqueue_t) * 1000) - result.process_ms
    state = JobState.SUCCEEDED if result.content is not None else JobState.FAILED

    return CompletionResponse(
        job_id=job.id,
        state=state,
        content=result.content,
        error=result.error,
        provider=req.provider,
        model=result.model,
        queue_wait_ms=max(0, queue_wait_ms),
        process_ms=result.process_ms,
        attempts=result.attempts,
    )
