from __future__ import annotations

import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol

from .token_estimation import estimate_tokens


RETRYABLE_HTTP_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class AgentLLMClient(Protocol):
    def complete(self, messages: list[dict[str, str]], *, trace_id: str, timeout_seconds: float) -> str:
        ...


@dataclass(slots=True)
class AgentLLMConfig:
    api_key: str
    model: str
    base_url: str
    provider: str = "openai_compatible"
    temperature: float = 0.2
    max_tokens: int = 1600
    response_format: str = "json_object"
    retry_attempts: int = 3
    retry_base_delay_seconds: float = 0.5
    retry_max_delay_seconds: float = 4.0
    max_estimated_tokens_per_day: int = 0

    @classmethod
    def from_env(cls) -> "AgentLLMConfig | None":
        api_key = (
            os.getenv("MAHJONG_LLM_API_KEY")
            or os.getenv("MAHJONG_DEEPSEEK_API_KEY")
            or os.getenv("DEEPSEEK_API_KEY")
        )
        model = os.getenv("MAHJONG_LLM_MODEL")
        if not api_key or not model:
            return None
        provider = (os.getenv("MAHJONG_LLM_PROVIDER") or "openai_compatible").strip().lower()
        return cls(
            api_key=api_key,
            model=model,
            provider=provider,
            base_url=(os.getenv("MAHJONG_LLM_BASE_URL") or default_base_url(provider)).rstrip("/"),
            temperature=env_float("MAHJONG_LLM_TEMPERATURE", 0.2),
            max_tokens=env_int("MAHJONG_LLM_MAX_COMPLETION_TOKENS", 1600),
            response_format=os.getenv("MAHJONG_LLM_RESPONSE_FORMAT", "json_object"),
            retry_attempts=max(1, env_int("MAHJONG_LLM_RETRY_ATTEMPTS", 3)),
            retry_base_delay_seconds=max(0.0, env_float("MAHJONG_LLM_RETRY_BASE_DELAY_SECONDS", 0.5)),
            retry_max_delay_seconds=max(0.0, env_float("MAHJONG_LLM_RETRY_MAX_DELAY_SECONDS", 4.0)),
            max_estimated_tokens_per_day=max(0, env_int("MAHJONG_LLM_MAX_ESTIMATED_TOKENS_PER_DAY", 0)),
        )


class DailyTokenLedger:
    """Process-wide conservative daily budget shared by all model clients."""

    _lock = threading.Lock()
    _day = date.today()
    _usage: dict[str, int] = {}

    @classmethod
    def reserve(cls, key: str, estimated_tokens: int, limit: int) -> None:
        if limit <= 0:
            return
        with cls._lock:
            today = date.today()
            if cls._day != today:
                cls._day = today
                cls._usage.clear()
            current = cls._usage.get(key, 0)
            if current + estimated_tokens > limit:
                raise RuntimeError(
                    f"daily LLM token budget exceeded for {key}: "
                    f"{current}+{estimated_tokens}>{limit}"
                )
            cls._usage[key] = current + estimated_tokens


@dataclass(slots=True)
class OpenAICompatibleAgentClient:
    config: AgentLLMConfig
    urlopen: Any = urllib.request.urlopen
    fallback_client: AgentLLMClient | None = None
    sleep_fn: Any = time.sleep
    monotonic_fn: Any = time.monotonic
    random_fn: Any = random.random

    @classmethod
    def from_env(cls) -> "OpenAICompatibleAgentClient | None":
        config = AgentLLMConfig.from_env()
        if config is None:
            return None
        fallback = fallback_client_from_env()
        return cls(config=config, fallback_client=fallback)

    def complete(self, messages: list[dict[str, str]], *, trace_id: str, timeout_seconds: float) -> str:
        estimated = sum(estimate_tokens(item.get("content", "")) for item in messages) + self.config.max_tokens
        DailyTokenLedger.reserve(
            f"{self.config.provider}:{self.config.model}",
            estimated,
            self.config.max_estimated_tokens_per_day,
        )
        try:
            return self._complete_with_retry(messages, trace_id=trace_id, timeout_seconds=timeout_seconds)
        except Exception as primary_error:
            if self.fallback_client is None:
                raise
            try:
                return self.fallback_client.complete(messages, trace_id=trace_id, timeout_seconds=timeout_seconds)
            except Exception as fallback_error:
                raise RuntimeError(
                    f"primary and fallback LLM calls failed; primary={primary_error}; fallback={fallback_error}"
                ) from fallback_error

    def _complete_with_retry(
        self,
        messages: list[dict[str, str]],
        *,
        trace_id: str,
        timeout_seconds: float,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if self.config.response_format:
            payload["response_format"] = {"type": self.config.response_format}
        if self.config.provider == "deepseek":
            payload["thinking"] = {"type": "disabled"}
        deadline = self.monotonic_fn() + max(0.1, timeout_seconds)
        last_error: Exception | None = None
        attempts = max(1, self.config.retry_attempts)
        for attempt in range(1, attempts + 1):
            remaining = deadline - self.monotonic_fn()
            if remaining <= 0:
                break
            request = urllib.request.Request(
                f"{self.config.base_url}/chat/completions",
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json; charset=utf-8",
                    "X-Trace-Id": trace_id,
                },
                method="POST",
            )
            try:
                with self.urlopen(request, timeout=remaining) as response:
                    raw = json.loads(response.read().decode("utf-8"))
                return content_from_response(raw)
            except urllib.error.HTTPError as exc:
                last_error = RuntimeError(http_error_note(exc))
                if exc.code not in RETRYABLE_HTTP_STATUS:
                    raise last_error from exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = RuntimeError(f"LLM request failed: {type(exc).__name__}: {exc}")

            if attempt >= attempts:
                break
            delay = min(
                self.config.retry_max_delay_seconds,
                self.config.retry_base_delay_seconds * (2 ** (attempt - 1)),
            )
            delay *= 1.0 + max(0.0, min(1.0, float(self.random_fn()))) * 0.1
            remaining = deadline - self.monotonic_fn()
            if remaining <= delay:
                break
            if delay > 0:
                self.sleep_fn(delay)
        if last_error is not None:
            raise last_error
        raise RuntimeError(f"LLM request timed out after {timeout_seconds:.1f}s")


@dataclass(slots=True)
class StaticAgentClient:
    outputs: list[str]
    calls: list[dict[str, Any]] = field(default_factory=list)

    def complete(self, messages: list[dict[str, str]], *, trace_id: str, timeout_seconds: float) -> str:
        self.calls.append({"trace_id": trace_id, "timeout_seconds": timeout_seconds, "messages": messages})
        if not self.outputs:
            raise RuntimeError("StaticAgentClient has no output left")
        return self.outputs.pop(0)


def content_from_response(raw: dict[str, Any]) -> str:
    choices = raw.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("LLM response has no choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise RuntimeError("LLM response has no text content")
    return str(message["content"])


def http_error_note(exc: urllib.error.HTTPError) -> str:
    body = exc.read().decode("utf-8", errors="replace")
    return f"LLM HTTP error {exc.code} {exc.reason}: {body[:300]}"


def default_base_url(provider: str) -> str:
    if provider == "deepseek":
        return "https://api.deepseek.com"
    if provider in {"zai", "glm", "bigmodel"}:
        return "https://api.z.ai/api/paas/v4"
    return "https://api.openai.com/v1"


def fallback_client_from_env() -> "OpenAICompatibleAgentClient | None":
    """Build an optional independent fallback provider from environment."""

    model = os.getenv("MAHJONG_LLM_FALLBACK_MODEL")
    api_key = os.getenv("MAHJONG_LLM_FALLBACK_API_KEY")
    if not model or not api_key:
        return None
    provider = (os.getenv("MAHJONG_LLM_FALLBACK_PROVIDER") or "openai_compatible").strip().lower()
    config = AgentLLMConfig(
        api_key=api_key,
        model=model,
        provider=provider,
        base_url=(os.getenv("MAHJONG_LLM_FALLBACK_BASE_URL") or default_base_url(provider)).rstrip("/"),
        temperature=env_float("MAHJONG_LLM_FALLBACK_TEMPERATURE", env_float("MAHJONG_LLM_TEMPERATURE", 0.2)),
        max_tokens=env_int(
            "MAHJONG_LLM_FALLBACK_MAX_COMPLETION_TOKENS",
            env_int("MAHJONG_LLM_MAX_COMPLETION_TOKENS", 1600),
        ),
        response_format=os.getenv("MAHJONG_LLM_FALLBACK_RESPONSE_FORMAT", "json_object"),
        retry_attempts=max(1, env_int("MAHJONG_LLM_FALLBACK_RETRY_ATTEMPTS", 2)),
        retry_base_delay_seconds=max(0.0, env_float("MAHJONG_LLM_FALLBACK_RETRY_BASE_DELAY_SECONDS", 0.5)),
        retry_max_delay_seconds=max(0.0, env_float("MAHJONG_LLM_FALLBACK_RETRY_MAX_DELAY_SECONDS", 2.0)),
        max_estimated_tokens_per_day=max(
            0,
            env_int("MAHJONG_LLM_FALLBACK_MAX_ESTIMATED_TOKENS_PER_DAY", 0),
        ),
    )
    return OpenAICompatibleAgentClient(config=config)


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default
