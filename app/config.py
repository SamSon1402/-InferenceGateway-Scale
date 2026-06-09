"""Application settings."""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "inference-gateway-scale"
    log_level: str = "INFO"

    # Queue backend. ``redis`` is the production default; ``memory`` is for
    # local dev / CI when Redis isn't running.
    queue_backend: Literal["redis", "memory"] = "redis"
    redis_url: str = "redis://localhost:6379"
    queue_namespace: str = "iqs"  # prefix for all Redis keys

    # Worker pool
    min_workers: int = 1
    max_workers: int = 8
    initial_workers: int = 2

    # Autoscaler — same HPA-style proportional formula K8s uses.
    autoscaler_interval_s: float = 5.0
    target_queue_per_worker: float = 10.0
    scale_up_cooldown_s: float = 30.0
    scale_down_cooldown_s: float = 60.0  # slower scale-down to avoid flap

    # Per-provider token bucket rate limits — (capacity, refill_per_sec)
    openai_capacity: int = 10_000
    openai_refill_per_s: float = 167.0       # ≈ 10k tpm
    anthropic_capacity: int = 8_000
    anthropic_refill_per_s: float = 133.0    # ≈ 8k tpm
    mistral_capacity: int = 12_000
    mistral_refill_per_s: float = 200.0      # ≈ 12k tpm

    # Provider credentials (optional — if unset, that provider is unavailable)
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    mistral_api_key: str | None = None

    # Retry on 429 / 5xx
    retry_max_attempts: int = 4
    retry_initial_backoff_s: float = 0.5
    retry_max_backoff_s: float = 8.0

    # Sync request timeout — how long /v1/completions waits for a result
    sync_timeout_s: float = 30.0

    # HTTP
    request_timeout_s: float = 30.0
    connect_timeout_s: float = 5.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
