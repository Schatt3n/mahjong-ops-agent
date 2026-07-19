from __future__ import annotations

import io
import json
import threading
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor

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


def test_default_completion_budget_can_hold_structured_agent_contract() -> None:
    assert config().max_tokens == 3200


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


def test_llm_limits_concurrent_provider_requests() -> None:
    lock = threading.Lock()
    active = 0
    max_active = 0

    def urlopen(*args: object, **kwargs: object) -> FakeResponse:
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            time.sleep(0.03)
            return FakeResponse('{"ok":true}')
        finally:
            with lock:
                active -= 1

    client = OpenAICompatibleAgentClient(config=config(max_concurrency=2), urlopen=urlopen)
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [
            pool.submit(
                client.complete,
                [{"role": "user", "content": "test"}],
                trace_id=f"trace_concurrency_{index}",
                timeout_seconds=2,
            )
            for index in range(6)
        ]
        assert [future.result() for future in futures] == ['{"ok":true}'] * 6

    metrics = client.concurrency_metrics()
    assert max_active == 2
    assert metrics["configured_max_concurrency"] == 2
    assert metrics["max_active_provider_requests"] == 2
    assert metrics["max_waiting_provider_requests"] >= 2
    assert metrics["active_provider_requests"] == 0
    assert metrics["waiting_provider_requests"] == 0


def test_llm_concurrency_queue_respects_end_to_end_timeout() -> None:
    first_request_started = threading.Event()
    release_first_request = threading.Event()

    def urlopen(*args: object, **kwargs: object) -> FakeResponse:
        first_request_started.set()
        release_first_request.wait(timeout=2)
        return FakeResponse('{"ok":true}')

    client = OpenAICompatibleAgentClient(config=config(max_concurrency=1), urlopen=urlopen)
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(
            client.complete,
            [{"role": "user", "content": "first"}],
            trace_id="trace_first",
            timeout_seconds=2,
        )
        assert first_request_started.wait(timeout=1)
        with pytest.raises(RuntimeError, match="concurrency queue timed out"):
            client.complete(
                [{"role": "user", "content": "second"}],
                trace_id="trace_second",
                timeout_seconds=0.05,
            )
        release_first_request.set()
        assert first.result() == '{"ok":true}'
