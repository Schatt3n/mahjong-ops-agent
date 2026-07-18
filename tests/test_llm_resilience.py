from __future__ import annotations

import io
import json
import urllib.error

import pytest

from mahjong_agent_runtime import AgentLLMConfig, OpenAICompatibleAgentClient, StaticAgentClient
from mahjong_agent_runtime.token_estimation import estimate_tokens


class FakeResponse:
    def __init__(self, content: str) -> None:
        self.body = json.dumps({"choices": [{"message": {"content": content}}]}).encode()

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


def config(**overrides: object) -> AgentLLMConfig:
    values = {
        "api_key": "test",
        "model": "test-model",
        "base_url": "https://example.invalid/v1",
        "retry_attempts": 3,
        "retry_base_delay_seconds": 0.0,
        "retry_max_delay_seconds": 0.0,
    }
    values.update(overrides)
    return AgentLLMConfig(**values)


def test_chinese_token_estimation_is_not_four_times_undercounted() -> None:
    text = "帮我组一个下午四点的无烟麻将局"
    assert estimate_tokens(text) >= len(text)


def test_llm_retries_retryable_http_error() -> None:
    calls = 0

    def urlopen(*args: object, **kwargs: object) -> FakeResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise urllib.error.HTTPError(
                "https://example.invalid",
                503,
                "busy",
                {},
                io.BytesIO(b'{"error":"busy"}'),
            )
        return FakeResponse('{"ok":true}')

    client = OpenAICompatibleAgentClient(config=config(), urlopen=urlopen)
    assert client.complete([{"role": "user", "content": "test"}], trace_id="trace_1", timeout_seconds=2) == '{"ok":true}'
    assert calls == 2


def test_llm_does_not_retry_non_retryable_auth_error() -> None:
    calls = 0

    def urlopen(*args: object, **kwargs: object) -> FakeResponse:
        nonlocal calls
        calls += 1
        raise urllib.error.HTTPError(
            "https://example.invalid",
            401,
            "unauthorized",
            {},
            io.BytesIO(b'{"error":"bad key"}'),
        )

    client = OpenAICompatibleAgentClient(config=config(), urlopen=urlopen)
    with pytest.raises(RuntimeError, match="401"):
        client.complete([{"role": "user", "content": "test"}], trace_id="trace_2", timeout_seconds=2)
    assert calls == 1


def test_llm_uses_fallback_after_primary_exhausted() -> None:
    fallback = StaticAgentClient(outputs=['{"fallback":true}'])

    def urlopen(*args: object, **kwargs: object) -> FakeResponse:
        raise urllib.error.URLError("offline")

    client = OpenAICompatibleAgentClient(
        config=config(retry_attempts=1),
        urlopen=urlopen,
        fallback_client=fallback,
    )
    assert client.complete([{"role": "user", "content": "test"}], trace_id="trace_3", timeout_seconds=2) == '{"fallback":true}'
    assert len(fallback.calls) == 1
