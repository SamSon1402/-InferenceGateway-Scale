"""Pydantic models — requests, responses, job lifecycle, metrics."""
from __future__ import annotations

from datetime import datetime
from enum import Enum, IntEnum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class Priority(IntEnum):
    """Lower value = higher priority. The queue dequeues HIGH first.

    Using an IntEnum (not a StrEnum) so comparison is trivial and the
    Redis sorted set / lane-ordered BLPOP both fall out naturally.
    """
    HIGH = 0
    NORMAL = 1
    LOW = 2


class ProviderName(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    MISTRAL = "mistral"


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class CompletionRequest(BaseModel):
    """Inbound completion request — what the API sees."""
    messages: list[Message] = Field(min_length=1)
    provider: ProviderName
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    priority: Priority = Priority.NORMAL
    # Tokens estimated for rate-limiter accounting. The gateway is
    # token-aware so a 4k-token job doesn't consume the same budget
    # as a 100-token one.
    estimated_tokens: int = Field(default=256, ge=1, le=8192)


class Job(BaseModel):
    """Internal queue item. Serialised to JSON when stored in Redis."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    request: CompletionRequest
    enqueued_at: datetime = Field(default_factory=datetime.utcnow)
    attempts: int = 0


class CompletionResponse(BaseModel):
    job_id: str
    state: JobState
    content: str | None = None
    error: str | None = None
    provider: ProviderName
    model: str | None = None
    queue_wait_ms: int
    process_ms: int
    attempts: int


class BucketState(BaseModel):
    provider: ProviderName
    capacity: int
    available: int
    refill_per_s: float
    throttled: bool


class Metrics(BaseModel):
    """Snapshot of operational state — useful for /healthz and dashboards."""
    workers: int
    min_workers: int
    max_workers: int
    queue_depth_total: int
    queue_depth_by_priority: dict[str, int]
    rate_limit_429_total: int
    inflight: int
    buckets: list[BucketState]
