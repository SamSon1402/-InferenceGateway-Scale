# InferenceGateway-Scale

Scalable LLM inference gateway. A FastAPI service that fronts OpenAI / Anthropic / Mistral with a Redis-backed priority queue, a per-provider token-bucket rate limiter, and a Kubernetes-HPA-style worker autoscaler.

Companion to the live demo at `samson1402.github.io/inference-gateway-scale`.

## What's in the box

```
app/
├── main.py         FastAPI app, routes, lifespan wiring
├── config.py       pydantic-settings
├── schemas.py      Pydantic models — request, job, metrics
├── queue.py        JobQueue interface + Redis & in-memory implementations
├── ratelimit.py    Per-provider token bucket + RateLimiter container
├── providers.py    OpenAI / Anthropic / Mistral adapters with 429 retry
├── store.py        In-process result store (futures for sync API hand-off)
├── workers.py      Worker + WorkerPool (scale_to, drain)
└── autoscaler.py   HPA-style proportional autoscaler
```

## Endpoints

| Method | Path                | Purpose                                          |
|--------|---------------------|--------------------------------------------------|
| POST   | `/v1/completions`   | Sync API; enqueues, waits for worker, returns    |
| GET    | `/v1/metrics`       | Queue depth, worker count, 429s, bucket state    |
| GET    | `/healthz`          | Liveness probe for K8s                           |

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env       # set at least one provider API key

# Production-shaped: Redis-backed queue
docker compose up -d redis
uvicorn app.main:app --reload

# Or no-Redis mode for quick smoke tests
QUEUE_BACKEND=memory uvicorn app.main:app --reload
```

Smoke test:

```bash
curl -sX POST http://localhost:8000/v1/completions \
  -H 'content-type: application/json' \
  -d '{
    "messages": [{"role":"user","content":"say hi in 5 words"}],
    "provider": "openai",
    "priority": 0,
    "estimated_tokens": 100
  }' | jq
```

```bash
curl http://localhost:8000/v1/metrics | jq
```

## Design notes worth pointing at

### Why Redis lists + BLPOP for the queue

BLPOP accepts multiple keys and pops from the *first non-empty* one. We use one list per priority lane (`iqs:queue:high`, `:normal`, `:low`) and ask BLPOP to scan them in that order. Priority for free — no Lua script, no separate scheduler thread.

### Why a swappable queue interface

`JobQueue` is an ABC; `RedisQueue` and `InMemoryQueue` implement it. The pool, workers, and autoscaler depend on the interface, never on Redis directly. Two payoffs:
1. CI runs the full engine with `QUEUE_BACKEND=memory` — no Redis container needed.
2. The same code can move to a different queue (SQS, NATS, Pub/Sub) by writing one new adapter.

### Why token buckets are token-weighted, not request-weighted

Provider rate limits are usually in **tokens-per-minute**, not requests. A 4 000-token completion consumes 40× a 100-token one. The worker passes `job.estimated_tokens` to `bucket.acquire(n)` so we throttle on what the provider actually meters on. A request-count limiter would over-protect for short prompts and under-protect for long ones.

### Why the autoscaler uses the K8s HPA formula

```
desired = ceil(current_replicas * (current_metric / target_metric))
```

It's proportional, well-understood, and battle-tested. We add the same ±10% tolerance band HPA uses by default to prevent thrash, plus separate cooldowns (up: 30 s, down: 60 s). Scale-down is intentionally slower than scale-up — burning a few extra workers for an extra minute is cheap; cold-starting them during a fresh spike is expensive.

### Why scale-down is cooperative, not preemptive

When the autoscaler shrinks the pool, it flips the stop flag on the last N workers and lets them finish whatever they're processing. No torn-up jobs, no orphan retries. The drained workers exit on their next queue-poll timeout (1 s by default).

### Why an in-process result store

The sync `/v1/completions` handler needs to block on a result. The store holds one `asyncio.Future` per outstanding job; the worker resolves it. **This works because submitter and worker share a process.** In a multi-pod deployment you'd swap this for Redis Pub/Sub or a per-job result key with `BLPOP`. The interface (`register`, `set_result`, `set_error`) doesn't change — only the implementation does.

### 429 handling: AWS-style full jitter

```python
backoff = uniform(0, min(max_backoff, base * 2 ** attempt))
```

Full jitter (random in `[0, exp]`) gives better spread under thundering-herd retries than equal jitter or no jitter — important when many workers hit the same rate limit at the same instant. `Retry-After` is honoured when present.

## Deliberately out of scope

- **Multi-pod result delivery** — current store is single-process. README notes the swap.
- **Persistent job records** — jobs vanish on restart; production would persist to Postgres or extend the Redis schema.
- **Auth / multi-tenant quotas** — gateway is open; an upstream API gateway or middleware would enforce these.
- **Provider routing strategies** (cheapest / fastest / quality) — that's `AgentOrchestrator-RT`. This gateway respects the provider the caller asked for.
- **Latency benchmarking & percentile telemetry** — that's `ConvoStream-Bench`. The metrics endpoint here is operational, not analytical.
