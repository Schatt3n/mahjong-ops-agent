from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Protocol


class AgentLLMClientV2(Protocol):
    def complete(self, messages: list[dict[str, str]], *, trace_id: str, timeout_seconds: float) -> str:
        ...


class UrlOpenResponseV2(Protocol):
    def __enter__(self) -> "UrlOpenResponseV2":
        ...

    def __exit__(self, exc_type, exc, tb) -> None:
        ...

    def read(self) -> bytes:
        ...


UrlOpenV2 = Callable[..., UrlOpenResponseV2]


@dataclass(slots=True)
class AgentLLMConfigV2:
    api_key: str
    model: str
    base_url: str
    provider: str = "openai_compatible"
    temperature: float = 0.2
    max_tokens: int = 4096
    response_format: str = "json_object"

    @classmethod
    def from_env(cls) -> "AgentLLMConfigV2 | None":
        api_key = os.getenv("MAHJONG_LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        model = os.getenv("MAHJONG_LLM_MODEL")
        if not api_key or not model:
            return None
        provider = (os.getenv("MAHJONG_LLM_PROVIDER") or "openai_compatible").strip().lower()
        base_url = os.getenv("MAHJONG_LLM_BASE_URL") or _default_base_url(provider)
        return cls(
            api_key=api_key,
            model=model,
            provider=provider,
            base_url=base_url.rstrip("/"),
            temperature=_env_float("MAHJONG_LLM_TEMPERATURE", 0.2),
            max_tokens=_env_int("MAHJONG_LLM_MAX_COMPLETION_TOKENS", 4096),
        )


@dataclass(slots=True)
class OpenAICompatibleAgentClientV2:
    config: AgentLLMConfigV2
    urlopen: UrlOpenV2 = urllib.request.urlopen

    @classmethod
    def from_env(cls) -> "OpenAICompatibleAgentClientV2 | None":
        config = AgentLLMConfigV2.from_env()
        return cls(config=config) if config else None

    def complete(self, messages: list[dict[str, str]], *, trace_id: str, timeout_seconds: float) -> str:
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
        request = urllib.request.Request(
            f"{self.config.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json; charset=utf-8",
            },
            method="POST",
        )
        try:
            with self.urlopen(request, timeout=timeout_seconds) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(_http_error_note(exc)) from exc
        except (urllib.error.URLError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"LLM request failed: {type(exc).__name__}: {exc}") from exc
        return _content_from_response(raw)


@dataclass(slots=True)
class StaticAgentClientV2:
    outputs: list[str]
    calls: list[dict[str, Any]]

    def complete(self, messages: list[dict[str, str]], *, trace_id: str, timeout_seconds: float) -> str:
        self.calls.append({"messages": messages, "trace_id": trace_id, "timeout_seconds": timeout_seconds})
        if not self.outputs:
            raise RuntimeError("StaticAgentClientV2 has no output left")
        return self.outputs.pop(0)


def _content_from_response(raw: dict[str, Any]) -> str:
    choices = raw.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("LLM response has no choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise RuntimeError("LLM response has no text content")
    return str(message["content"])


def _http_error_note(exc: urllib.error.HTTPError) -> str:
    body = exc.read().decode("utf-8", errors="replace")
    return f"LLM HTTP error {exc.code} {exc.reason}: {body[:300]}"


def _default_base_url(provider: str) -> str:
    if provider == "deepseek":
        return "https://api.deepseek.com"
    if provider == "qwen":
        return "https://dashscope.aliyuncs.com/compatible-mode/v1"
    if provider in {"zai", "glm", "bigmodel"}:
        return "https://api.z.ai/api/paas/v4"
    return "https://api.openai.com/v1"


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default
