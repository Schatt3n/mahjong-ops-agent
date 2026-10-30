from __future__ import annotations

import io
import json
from http.client import IncompleteRead
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import pytest

from mahjong_agent_runtime import AgentLLMConfig, OpenAICompatibleAgentClient, StaticAgentClient
from mahjong_agent_runtime.token_estimation import estimate_tokens
from mahjong_agent_runtime.llm import fallback_client_from_env


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


def test_truncated_response_retries_within_the_existing_transport_budget() -> None:
    calls = []

    class TruncatedResponse(FakeResponse):
        def read(self):
            raise IncompleteRead(b'{"choices":', 100)

    def urlopen(*args, **kwargs):
        calls.append(kwargs["timeout"])
        return TruncatedResponse("") if len(calls) == 1 else FakeResponse('{"ok":true}')

    client = OpenAICompatibleAgentClient(config=config(), urlopen=urlopen)
    assert client.complete([{"role": "user", "content": "test"}], trace_id="transport-truncation", timeout_seconds=2) == '{"ok":true}'
    assert len(calls) == 2 and 0 < calls[1] <= calls[0] <= 2


def test_repeated_truncation_keeps_failure_reason_and_stops_at_attempt_limit() -> None:
    calls = 0

    class TruncatedResponse(FakeResponse):
        def read(self):
            raise IncompleteRead(b'{"choices":', 100)

    def urlopen(*args, **kwargs):
        nonlocal calls
        calls += 1
        return TruncatedResponse("")

    client = OpenAICompatibleAgentClient(config=config(retry_attempts=2), urlopen=urlopen)
    with pytest.raises(RuntimeError, match="IncompleteRead"):
        client.complete([{"role": "user", "content": "test"}], trace_id="transport-truncation-limit", timeout_seconds=2)
    assert calls == 2


def test_bigmodel_requests_disable_default_thinking_for_contract_tasks() -> None:
    captured: dict = {}

    def urlopen(request, **_kwargs: object) -> FakeResponse:
        captured.update(json.loads(request.data.decode("utf-8")))
        return FakeResponse('{"ok":true}')

    client = OpenAICompatibleAgentClient(
        config=config(provider="bigmodel"),
        urlopen=urlopen,
    )

    client.complete([{"role": "user", "content": "test"}], trace_id="trace_glm", timeout_seconds=2)

    assert captured["thinking"] == {"type": "disabled"}


@pytest.mark.parametrize("effort", [None, "low", "high", "max"])
def test_deepseek_enabled_thinking_sends_only_explicit_effort_without_temperature(effort) -> None:
    """推理开关和强度实际进入请求，温度及未配置强度不进入 DeepSeek 推理请求。"""
    captured: dict = {}

    def urlopen(request, **_kwargs: object) -> FakeResponse:
        captured.update(json.loads(request.data.decode("utf-8")))
        return FakeResponse('{"ok":true}')

    client = OpenAICompatibleAgentClient(
        config=config(provider="deepseek", thinking_mode="enabled", reasoning_effort=effort),
        urlopen=urlopen,
    )
    result = client.complete([{"role": "user", "content": "test"}], trace_id="thinking-on", timeout_seconds=2)
    assert result == '{"ok":true}'
    assert captured["thinking"] == {"type": "enabled"}
    assert "temperature" not in captured
    assert captured["max_tokens"] == 3200
    assert captured["response_format"] == {"type": "json_object"}
    if effort is None:
        assert "reasoning_effort" not in captured
    else:
        assert captured["reasoning_effort"] == effort


@pytest.mark.parametrize("overrides", [{}, {"thinking_mode": "disabled", "reasoning_effort": "high"}])
def test_deepseek_default_or_disabled_thinking_retains_existing_request_contract(overrides) -> None:
    """未启用时保留既有关闭模式、温度和输出上限，不把强度单独发送。"""
    captured: dict = {}

    def urlopen(request, **_kwargs: object) -> FakeResponse:
        captured.update(json.loads(request.data.decode("utf-8")))
        return FakeResponse('{"ok":true}')

    client = OpenAICompatibleAgentClient(config=config(provider="deepseek", **overrides), urlopen=urlopen)
    client.complete([{"role": "user", "content": "test"}], trace_id="thinking-off", timeout_seconds=2)
    assert captured["thinking"] == {"type": "disabled"}
    assert captured["temperature"] == 0.2
    assert captured["max_tokens"] == 3200
    assert "reasoning_effort" not in captured


@pytest.mark.parametrize("provider", ["zai", "glm", "bigmodel", "openai_compatible"])
def test_deepseek_thinking_settings_do_not_change_other_provider_payloads(provider) -> None:
    """GLM 保留关闭行为，普通兼容端不接收 DeepSeek 专用字段。"""
    captured: dict = {}

    def urlopen(request, **_kwargs: object) -> FakeResponse:
        captured.update(json.loads(request.data.decode("utf-8")))
        return FakeResponse('{"ok":true}')

    client = OpenAICompatibleAgentClient(
        config=config(provider=provider, thinking_mode="enabled", reasoning_effort="low"),
        urlopen=urlopen,
    )
    client.complete([{"role": "user", "content": "test"}], trace_id="other-provider", timeout_seconds=2)
    assert captured["temperature"] == 0.2
    assert "reasoning_effort" not in captured
    if provider == "openai_compatible":
        assert "thinking" not in captured
    else:
        assert captured["thinking"] == {"type": "disabled"}


def test_thinking_environment_settings_are_optional_and_explicit(monkeypatch) -> None:
    """无环境设置时关闭，显式启用及强度由配置对象准确保留。"""
    monkeypatch.setenv("MAHJONG_LLM_API_KEY", "test")
    monkeypatch.setenv("MAHJONG_LLM_MODEL", "test-model")
    monkeypatch.delenv("MAHJONG_LLM_THINKING_MODE", raising=False)
    monkeypatch.delenv("MAHJONG_LLM_REASONING_EFFORT", raising=False)
    default = AgentLLMConfig.from_env()
    assert default.thinking_mode == "disabled"
    assert default.reasoning_effort is None
    monkeypatch.setenv("MAHJONG_LLM_THINKING_MODE", "enabled")
    monkeypatch.setenv("MAHJONG_LLM_REASONING_EFFORT", "low")
    enabled = AgentLLMConfig.from_env()
    assert enabled.thinking_mode == "enabled"
    assert enabled.reasoning_effort == "low"


@pytest.mark.parametrize("name,value,field", [
    ("MAHJONG_LLM_THINKING_MODE", "", "thinking_mode"),
    ("MAHJONG_LLM_THINKING_MODE", "auto", "thinking_mode"),
    ("MAHJONG_LLM_REASONING_EFFORT", "", "reasoning_effort"),
    ("MAHJONG_LLM_REASONING_EFFORT", "medium", "reasoning_effort"),
])
def test_invalid_explicit_thinking_environment_setting_fails_without_fallback(monkeypatch, name, value, field) -> None:
    """空值和不支持的枚举显式失败，不能悄悄回落默认推理行为。"""
    monkeypatch.setenv("MAHJONG_LLM_API_KEY", "test")
    monkeypatch.setenv("MAHJONG_LLM_MODEL", "test-model")
    monkeypatch.delenv("MAHJONG_LLM_THINKING_MODE", raising=False)
    monkeypatch.delenv("MAHJONG_LLM_REASONING_EFFORT", raising=False)
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=field):
        AgentLLMConfig.from_env()


@pytest.mark.parametrize("overrides,field", [
    ({"thinking_mode": None}, "thinking_mode"),
    ({"thinking_mode": "auto"}, "thinking_mode"),
    ({"reasoning_effort": "medium"}, "reasoning_effort"),
    ({"reasoning_effort": False}, "reasoning_effort"),
])
def test_invalid_direct_thinking_configuration_is_rejected(overrides, field) -> None:
    """直接构造客户端与环境配置使用相同枚举边界。"""
    with pytest.raises(ValueError, match=field):
        config(**overrides)


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


class FakeClock:
    """只有测试动作推进的单调时钟，用来精确核对尝试、退避及回退共用预算。"""
    def __init__(self):
        self.elapsed = 0.0

    def __call__(self):
        return self.elapsed

    def advance(self, seconds):
        self.elapsed += seconds


class IncrementalResponse(FakeResponse):
    """逐次返回字节片段并推进假时钟；关闭状态独立记录，不能走无界 read。"""
    def __init__(self, clock, fragments):
        self.clock = clock
        self.fragments = iter(fragments)
        self.closed = False

    def read1(self, size):
        delay, body = next(self.fragments, (0, b""))
        self.clock.advance(delay)
        return body

    def read(self):
        raise AssertionError("incremental response must use read1")

    def __exit__(self, *args):
        self.closed = True


def encoded_response(content='{"ok":true}'):
    return json.dumps({"choices": [{"message": {"content": content}}]}, ensure_ascii=False).encode()


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan"), True, "60"])
def test_invalid_attempt_timeout_direct_configuration_is_rejected(value):
    with pytest.raises(ValueError, match="attempt_timeout_seconds"):
        config(attempt_timeout_seconds=value)


@pytest.mark.parametrize("name", ["MAHJONG_LLM_ATTEMPT_TIMEOUT_SECONDS", "MAHJONG_LLM_FALLBACK_ATTEMPT_TIMEOUT_SECONDS"])
@pytest.mark.parametrize("value", ["", "bad", "0", "-1", "inf", "nan"])
def test_explicit_invalid_attempt_timeout_env_is_rejected(monkeypatch, name, value):
    monkeypatch.setenv("MAHJONG_LLM_API_KEY", "unit-test-only")
    monkeypatch.setenv("MAHJONG_LLM_MODEL", "unit-test-model")
    monkeypatch.setenv("MAHJONG_LLM_FALLBACK_API_KEY", "unit-test-only")
    monkeypatch.setenv("MAHJONG_LLM_FALLBACK_MODEL", "unit-test-model")
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        (fallback_client_from_env() if "FALLBACK" in name else AgentLLMConfig.from_env())


def test_attempt_timeout_is_optional_and_primary_fallback_env_are_separate(monkeypatch):
    monkeypatch.setenv("MAHJONG_LLM_API_KEY", "unit-test-only")
    monkeypatch.setenv("MAHJONG_LLM_MODEL", "unit-test-model")
    monkeypatch.setenv("MAHJONG_LLM_FALLBACK_API_KEY", "unit-test-only")
    monkeypatch.setenv("MAHJONG_LLM_FALLBACK_MODEL", "unit-test-model")
    monkeypatch.delenv("MAHJONG_LLM_ATTEMPT_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("MAHJONG_LLM_FALLBACK_ATTEMPT_TIMEOUT_SECONDS", raising=False)
    assert AgentLLMConfig.from_env().attempt_timeout_seconds is None
    assert fallback_client_from_env().config.attempt_timeout_seconds is None
    monkeypatch.setenv("MAHJONG_LLM_ATTEMPT_TIMEOUT_SECONDS", "60")
    monkeypatch.setenv("MAHJONG_LLM_FALLBACK_ATTEMPT_TIMEOUT_SECONDS", "12.5")
    assert AgentLLMConfig.from_env().attempt_timeout_seconds == 60
    assert fallback_client_from_env().config.attempt_timeout_seconds == 12.5


def test_incremental_json_decodes_only_after_full_body_and_closes_response():
    clock = FakeClock()
    content = '{"reply":"中文回复"}'
    body = encoded_response(content)
    split = body.index("中".encode()) + 1
    response = IncrementalResponse(clock, [(0.1, body[:split]), (0.1, body[split:])])
    client = OpenAICompatibleAgentClient(config=config(), urlopen=lambda *a, **kw: response, monotonic_fn=clock)
    assert client.complete([], trace_id="split-utf8", timeout_seconds=1) == content
    assert response.closed
    assert client.concurrency_metrics()["http_attempts"] == 1


def test_attempt_timeouts_retries_and_backoff_share_one_total_deadline():
    clock = FakeClock()
    responses, timeouts = [], []

    def urlopen(*args, **kwargs):
        timeouts.append(kwargs["timeout"])
        response = IncrementalResponse(clock, [(kwargs["timeout"], encoded_response())])
        responses.append(response)
        return response

    client = OpenAICompatibleAgentClient(config=config(attempt_timeout_seconds=1, retry_attempts=9,
        retry_base_delay_seconds=.25, retry_max_delay_seconds=.25), urlopen=urlopen,
        monotonic_fn=clock, sleep_fn=clock.advance, random_fn=lambda: 0)
    with pytest.raises(RuntimeError, match="TimeoutError"):
        client.complete([], trace_id="one-total-budget", timeout_seconds=3)
    assert timeouts == [1, 1, .5]
    assert clock.elapsed == 3 and all(response.closed for response in responses)
    metrics = client.concurrency_metrics()
    assert (metrics["http_attempts"], metrics["http_retries"], metrics["http_timeouts"], metrics["http_recoveries"]) == (3, 2, 3, 0)
    assert metrics["active_provider_requests"] == metrics["waiting_provider_requests"] == 0


def test_transport_recovery_has_its_own_metrics_and_preserves_partial_body_boundary():
    clock = FakeClock()
    responses = [IncrementalResponse(clock, [(1, b'{"choices":')]),
                 IncrementalResponse(clock, [(.1, encoded_response())])]
    opened = []

    def urlopen(*args, **kwargs):
        opened.append(kwargs["timeout"])
        return responses[len(opened) - 1]

    client = OpenAICompatibleAgentClient(config=config(attempt_timeout_seconds=1), urlopen=urlopen, monotonic_fn=clock)
    assert client.complete([], trace_id="bounded-recovery", timeout_seconds=2) == '{"ok":true}'
    assert opened == [1, 1] and all(response.closed for response in responses)
    metrics = client.concurrency_metrics()
    assert (metrics["http_attempts"], metrics["http_retries"], metrics["http_timeouts"], metrics["http_recoveries"]) == (2, 1, 1, 1)
    assert all("key" not in key and "content" not in key for key in metrics)


def test_legacy_read_only_injected_response_is_checked_after_return_and_closed():
    clock = FakeClock()

    class LegacyResponse(FakeResponse):
        closed = False
        def read(self):
            clock.advance(1)
            return self.body
        def __exit__(self, *args):
            self.closed = True

    response = LegacyResponse('{"too_late":true}')
    client = OpenAICompatibleAgentClient(config=config(retry_attempts=1), urlopen=lambda *a, **kw: response, monotonic_fn=clock)
    with pytest.raises(RuntimeError, match="TimeoutError"):
        client.complete([], trace_id="legacy-injection", timeout_seconds=.5)
    assert response.closed


def test_short_fallback_remainder_cannot_be_expanded_or_return_late_content():
    clock = FakeClock()
    primary_timeouts, fallback_timeouts = [], []

    def primary_open(*args, **kwargs):
        primary_timeouts.append(kwargs["timeout"])
        clock.advance(.06)
        raise urllib.error.URLError("temporary network failure")

    response = IncrementalResponse(clock, [(.08, encoded_response())])
    def fallback_open(*args, **kwargs):
        fallback_timeouts.append(kwargs["timeout"])
        return response

    fallback = OpenAICompatibleAgentClient(config=config(retry_attempts=1), urlopen=fallback_open, monotonic_fn=clock)
    client = OpenAICompatibleAgentClient(config=config(retry_attempts=1), urlopen=primary_open,
        fallback_client=fallback, monotonic_fn=clock)
    with pytest.raises(RuntimeError, match="primary and fallback"):
        client.complete([], trace_id="shared-fallback-budget", timeout_seconds=.1)
    assert primary_timeouts == [.1]
    assert fallback_timeouts == [pytest.approx(.04)]
    assert response.closed


@contextmanager
def local_http_server(action):
    """只监听回环地址；所有处理线程在退出时停止并回收，不访问外部服务。"""
    stop = threading.Event()
    requests = []

    class Server(ThreadingHTTPServer):
        daemon_threads = False
        block_on_close = True

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(payload)
            try:
                action(self, len(requests), stop)
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                self.close_connection = True

    server = Server(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01})
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        stop.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        assert not thread.is_alive()


def tracked_local_opener():
    """使用真实 urllib、禁用代理，只捕获连接关闭证据，不包装或模拟响应读取。"""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    sources = []

    def open_response(*args, **kwargs):
        try:
            response = opener.open(*args, **kwargs)
        except urllib.error.HTTPError as error:
            response = error.fp
            sources.append((response.fp, response.fp.raw, response.fp.raw._sock))
            raise
        sources.append((response.fp, response.fp.raw, response.fp.raw._sock))
        return response

    return open_response, sources


def assert_connections_closed(sources):
    assert sources
    for source, raw, sock in sources:
        assert source.closed and raw.closed
        assert raw._sock is None and sock.fileno() == -1


@pytest.mark.parametrize("mode", ["body", "error_body", "chunk_header", "chunk_trailer"])
def test_real_slow_drip_body_and_chunk_metadata_obey_deadline(mode):
    """连续小字节可绕过闲置timeout；真实 socket 接收边界仍须在总预算内停止。"""
    def action(handler, attempt, stop):
        handler.send_response(503 if mode == "error_body" else 200)
        handler.send_header("Transfer-Encoding" if mode.startswith("chunk") else "Content-Length",
                            "chunked" if mode.startswith("chunk") else "1000000")
        handler.end_headers()
        if mode == "chunk_header":
            handler.wfile.write(b"1;")
        elif mode == "chunk_trailer":
            handler.wfile.write(b"1\r\n \r\n0\r\nX-Slow: ")
        while not stop.is_set():
            handler.wfile.write(b"x" if mode.startswith("chunk") else b" ")
            handler.wfile.flush()
            stop.wait(.01)

    open_response, sources = tracked_local_opener()
    with local_http_server(action) as (url, requests):
        client = OpenAICompatibleAgentClient(config=config(base_url=url, retry_attempts=1), urlopen=open_response)
        started = time.monotonic()
        with pytest.raises(RuntimeError, match="TimeoutError"):
            client.complete([], trace_id="local-drip", timeout_seconds=.18)
        elapsed = time.monotonic() - started
        assert .10 <= elapsed < .8
        assert len(requests) == 1
        assert_connections_closed(sources)
        assert client.concurrency_metrics()["http_timeouts"] == 1


@pytest.mark.parametrize("mode", ["normal_chunked", "truncated_length", "truncated_chunk", "stalled_first", "slow_error_first"])
def test_real_chunked_json_and_transport_failures_recover_within_attempt_budget(mode):
    body = encoded_response('{"reply":"可以"}')

    def action(handler, attempt, stop):
        bad_first = attempt == 1 and mode != "normal_chunked"
        handler.send_response(503 if bad_first and mode == "slow_error_first" else 200)
        chunked = mode in {"normal_chunked", "truncated_chunk"}
        handler.send_header("Transfer-Encoding" if chunked else "Content-Length",
                            "chunked" if chunked else str(len(body) + (10 if bad_first else 0)))
        handler.end_headers()
        if bad_first and mode == "stalled_first":
            stop.wait(2)
        elif bad_first and mode == "slow_error_first":
            while not stop.is_set():
                handler.wfile.write(b" ")
                handler.wfile.flush()
                stop.wait(.01)
        elif bad_first and mode == "truncated_length":
            handler.wfile.write(body)  # 可解析 JSON 也不能掩盖 Content-Length 未交付完整。
        elif bad_first and mode == "truncated_chunk":
            handler.wfile.write(f"{len(body) + 10:x}\r\n".encode() + body)
        elif chunked:
            for fragment in (body[:11], body[11:]):
                handler.wfile.write(f"{len(fragment):x}\r\n".encode() + fragment + b"\r\n")
                handler.wfile.flush()
            handler.wfile.write(b"0\r\n\r\n")
        else:
            handler.wfile.write(body)

    open_response, sources = tracked_local_opener()
    with local_http_server(action) as (url, requests):
        client = OpenAICompatibleAgentClient(config=config(base_url=url, provider="deepseek", thinking_mode="enabled",
            reasoning_effort="low", attempt_timeout_seconds=.16), urlopen=open_response)
        started = time.monotonic()
        assert client.complete([], trace_id="local-recovery", timeout_seconds=.7) == '{"reply":"可以"}'
        assert time.monotonic() - started < .7
        expected_attempts = 1 if mode == "normal_chunked" else 2
        assert len(requests) == expected_attempts
        assert_connections_closed(sources)
        assert all(request["thinking"] == {"type": "enabled"} and request["reasoning_effort"] == "low"
                   and request["max_tokens"] == 3200 and "temperature" not in request for request in requests)
        metrics = client.concurrency_metrics()
        assert metrics["http_attempts"] == expected_attempts
        assert metrics["http_recoveries"] == expected_attempts - 1
        assert metrics["http_timeouts"] == (1 if mode in {"stalled_first", "slow_error_first"} else 0)
