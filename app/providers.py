"""Provider HTTP adapters.

Distinct from AgentOrchestrator-RT's providers in two ways:

1. There's no circuit breaker here — the gateway protects providers
   via the token-bucket rate limiter, not via failure-counting. Providers
   that return 429 are retried with backoff (see :class:`RetryPolicy`),
   not taken out of rotation.
2. These adapters return ``(content, model)``; they don't track stats
   themselves. The worker does the timing and accounting.

If a provider returns a ``Retry-After`` header on 429, we honour it
instead of using our exponential schedule.
"""
from __future__ import annotations

import asyncio
import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass

import httpx

from .schemas import Message, ProviderName

logger = logging.getLogger(__name__)


class ProviderError(Exception):
    """Non-retryable provider failure."""


class RateLimitedError(ProviderError):
    """Provider returned 429. Worker should retry with backoff."""

    def __init__(self, retry_after_s: float | None = None) -> None:
        super().__init__("rate limited (429)")
        self.retry_after_s = retry_after_s


# --------------------------------------------------------------------------- #
# Retry policy                                                                #
# --------------------------------------------------------------------------- #

@dataclass
class RetryPolicy:
    max_attempts: int = 4
    initial_backoff_s: float = 0.5
    max_backoff_s: float = 8.0

    def backoff(self, attempt: int) -> float:
        """Exponential backoff with full jitter (AWS-style).

        Full jitter — random in [0, exp_backoff] — gives better spread
        under thundering-herd retries than equal jitter or no jitter.
        """
        base = min(self.max_backoff_s, self.initial_backoff_s * (2 ** attempt))
        return random.uniform(0, base)


# --------------------------------------------------------------------------- #
# Provider adapters                                                           #
# --------------------------------------------------------------------------- #

class Provider(ABC):
    name: ProviderName
    model: str

    @abstractmethod
    async def call(
        self,
        messages: list[Message],
        *,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Return the assistant message content. Raise on failure."""


class _HttpProvider(Provider):
    """Shared HTTP handling. Subclasses define ``url``, ``headers``,
    ``build_body``, and ``extract_content``."""

    def __init__(self, api_key: str, http: httpx.AsyncClient) -> None:
        self._api_key = api_key
        self._http = http

    async def call(self, messages, *, max_tokens, temperature) -> str:
        body = self.build_body(messages, max_tokens, temperature)
        try:
            resp = await self._http.post(self.url, headers=self.headers, json=body)
        except httpx.HTTPError as exc:
            raise ProviderError(f"{self.name.value} transport: {exc}") from exc

        if resp.status_code == 429:
            ra = resp.headers.get("Retry-After")
            retry_after = float(ra) if ra and ra.replace(".", "", 1).isdigit() else None
            raise RateLimitedError(retry_after_s=retry_after)

        if resp.status_code >= 500:
            # 5xx is retryable; surface as RateLimitedError so the worker
            # uses the same backoff path. (Some teams use a distinct
            # ``RetryableError`` — equivalent.)
            raise RateLimitedError()

        if resp.status_code >= 400:
            raise ProviderError(f"{self.name.value} {resp.status_code}: {resp.text[:200]}")

        return self.extract_content(resp.json())

    # Subclasses fill these in.
    @property
    @abstractmethod
    def url(self) -> str: ...

    @property
    @abstractmethod
    def headers(self) -> dict: ...

    @abstractmethod
    def build_body(self, messages, max_tokens, temperature) -> dict: ...

    @abstractmethod
    def extract_content(self, data: dict) -> str: ...


class OpenAIProvider(_HttpProvider):
    name = ProviderName.OPENAI
    model = "gpt-4o-mini"
    url = "https://api.openai.com/v1/chat/completions"

    @property
    def headers(self) -> dict:
        return {"Authorization": f"Bearer {self._api_key}"}

    def build_body(self, messages, max_tokens, temperature) -> dict:
        return {
            "model": self.model,
            "messages": [m.model_dump() for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

    def extract_content(self, data: dict) -> str:
        return data["choices"][0]["message"]["content"]


class AnthropicProvider(_HttpProvider):
    name = ProviderName.ANTHROPIC
    model = "claude-3-5-haiku-20241022"
    url = "https://api.anthropic.com/v1/messages"

    @property
    def headers(self) -> dict:
        return {"x-api-key": self._api_key, "anthropic-version": "2023-06-01"}

    def build_body(self, messages, max_tokens, temperature) -> dict:
        system = "\n".join(m.content for m in messages if m.role == "system") or None
        turns = [m.model_dump() for m in messages if m.role != "system"]
        body = {
            "model": self.model,
            "messages": turns,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system:
            body["system"] = system
        return body

    def extract_content(self, data: dict) -> str:
        return data["content"][0]["text"]


class MistralProvider(_HttpProvider):
    name = ProviderName.MISTRAL
    model = "mistral-small-latest"
    url = "https://api.mistral.ai/v1/chat/completions"

    @property
    def headers(self) -> dict:
        return {"Authorization": f"Bearer {self._api_key}"}

    def build_body(self, messages, max_tokens, temperature) -> dict:
        return {
            "model": self.model,
            "messages": [m.model_dump() for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

    def extract_content(self, data: dict) -> str:
        return data["choices"][0]["message"]["content"]


class ProviderRegistry:
    """Holds whichever provider adapters were configured at startup."""

    def __init__(self, providers: dict[ProviderName, Provider]) -> None:
        self._providers = providers

    def get(self, name: ProviderName) -> Provider:
        if name not in self._providers:
            raise ProviderError(f"provider {name.value} not configured")
        return self._providers[name]

    def names(self) -> list[ProviderName]:
        return list(self._providers.keys())
