"""封装兼容 OpenAI 协议的模型调用、配置、用量记账与测试替身。"""

from __future__ import annotations

import json
from http.client import HTTPResponse, IncompleteRead
import math
import os
import random
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import date
from socket import SocketIO
from typing import Any, Protocol

from .token_estimation import estimate_tokens


RETRYABLE_HTTP_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class AgentLLMClient(Protocol):
    """定义智能体llmclient需要遵守的抽象接口契约。"""
    def complete(self, messages: list[dict[str, str]], *, trace_id: str, timeout_seconds: float) -> str:
        ...


@dataclass(slots=True)
class AgentLLMConfig:
    """配置模型认证、请求预算和推理模式；供应商专用参数由客户端按提供方发送。"""
    api_key: str
    model: str
    base_url: str
    provider: str = "openai_compatible"
    temperature: float = 0.2
    max_tokens: int = 3200
    response_format: str = "json_object"
    retry_attempts: int = 3
    retry_base_delay_seconds: float = 0.5
    retry_max_delay_seconds: float = 4.0
    max_estimated_tokens_per_day: int = 0
    max_concurrency: int = 3
    thinking_mode: str = "disabled"
    reasoning_effort: str | None = None
    attempt_timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        """校验显式推理和传输预算，避免非法设置静默改变模型或重试行为。"""
        if self.thinking_mode not in ("enabled", "disabled"):
            raise ValueError("thinking_mode must be enabled or disabled")
        if self.reasoning_effort is not None and self.reasoning_effort not in ("low", "high", "max"):
            raise ValueError("reasoning_effort must be None, low, high or max")
        if self.attempt_timeout_seconds is not None and (
            isinstance(self.attempt_timeout_seconds, bool)
            or not isinstance(self.attempt_timeout_seconds, (int, float))
            or not math.isfinite(self.attempt_timeout_seconds)
            or self.attempt_timeout_seconds <= 0
        ):
            raise ValueError("attempt_timeout_seconds must be None or a finite positive number")

    @classmethod
    def from_env(cls) -> "AgentLLMConfig | None":
        """读取本地调用配置；缺少认证或模型时不启用远端，非法显式推理配置则报错。"""
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
            max_tokens=env_int("MAHJONG_LLM_MAX_COMPLETION_TOKENS", 3200),
            response_format=os.getenv("MAHJONG_LLM_RESPONSE_FORMAT", "json_object"),
            retry_attempts=max(1, env_int("MAHJONG_LLM_RETRY_ATTEMPTS", 3)),
            retry_base_delay_seconds=max(0.0, env_float("MAHJONG_LLM_RETRY_BASE_DELAY_SECONDS", 0.5)),
            retry_max_delay_seconds=max(0.0, env_float("MAHJONG_LLM_RETRY_MAX_DELAY_SECONDS", 4.0)),
            max_estimated_tokens_per_day=max(0, env_int("MAHJONG_LLM_MAX_ESTIMATED_TOKENS_PER_DAY", 0)),
            max_concurrency=max(1, env_int("MAHJONG_LLM_MAX_CONCURRENCY", 3)),
            thinking_mode=os.getenv("MAHJONG_LLM_THINKING_MODE", "disabled"),
            reasoning_effort=os.getenv("MAHJONG_LLM_REASONING_EFFORT"),
            attempt_timeout_seconds=optional_positive_env_float("MAHJONG_LLM_ATTEMPT_TIMEOUT_SECONDS"),
        )


def _remaining_response_seconds(deadline: float, monotonic_fn: Any) -> float:
    """统一检查本次读取截止时间，持续到达的数据不能重新获得一份预算。"""
    remaining = deadline - monotonic_fn()
    if remaining <= 0:
        raise TimeoutError("LLM response deadline exceeded")
    return remaining


class _DeadlineSocket:
    """仅代理本次响应的底层接收，保留 CPython SocketIO 的引用释放协议。

    HTTPResponse.read1 的 chunk 头及 trailer 仍会内部循环 readline；在 recv_into
    边界检查期限才能约束这些读取，同时保留 BufferedReader 已经预读的正文。
    """

    def __init__(self, socket: Any, deadline: float, monotonic_fn: Any) -> None:
        """借用现有连接，不创建 socket、线程或定时器，也不改变 HTTP 解析器。"""
        self._socket = socket
        self._deadline = deadline
        self._monotonic_fn = monotonic_fn

    def recv_into(self, *args: Any, **kwargs: Any) -> int:
        """每次实际接收前收紧 socket 等待时间，并拒绝超时后才到达的数据。"""
        self._socket.settimeout(_remaining_response_seconds(self._deadline, self._monotonic_fn))
        count = self._socket.recv_into(*args, **kwargs)
        _remaining_response_seconds(self._deadline, self._monotonic_fn)
        return count

    def __getattr__(self, name: str) -> Any:
        """委托关闭及 _decref_socketios 等原协议，避免代理延长底层连接生命周期。"""
        return getattr(self._socket, name)


class DailyTokenLedger:
    """进程级保守的每日令牌预算，供所有模型客户端共享。"""

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
    """调用兼容 OpenAI Chat Completions 协议的远程模型客户端。"""

    config: AgentLLMConfig
    urlopen: Any = urllib.request.urlopen
    fallback_client: AgentLLMClient | None = None
    sleep_fn: Any = time.sleep
    monotonic_fn: Any = time.monotonic
    random_fn: Any = random.random
    _concurrency_gate: threading.BoundedSemaphore = field(init=False, repr=False)
    _concurrency_metrics_lock: threading.Lock = field(init=False, repr=False)
    _active_provider_requests: int = field(init=False, default=0, repr=False)
    _waiting_provider_requests: int = field(init=False, default=0, repr=False)
    _max_active_provider_requests: int = field(init=False, default=0, repr=False)
    _max_waiting_provider_requests: int = field(init=False, default=0, repr=False)
    _http_metrics: dict[str, int] = field(init=False, default_factory=lambda: {
        "http_attempts": 0, "http_retries": 0, "http_recoveries": 0, "http_timeouts": 0,
    }, repr=False)

    def __post_init__(self) -> None:
        """完成数据类初始化后的依赖组装和不变量校验。"""
        self._concurrency_gate = threading.BoundedSemaphore(max(1, int(self.config.max_concurrency)))
        self._concurrency_metrics_lock = threading.Lock()

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
        deadline = self.monotonic_fn() + max(0.0, timeout_seconds)
        try:
            return self._complete_primary_with_concurrency_limit(
                messages,
                trace_id=trace_id,
                deadline=deadline,
            )
        except Exception as primary_error:
            if self.fallback_client is None:
                raise
            remaining = deadline - self.monotonic_fn()
            if remaining <= 0:
                raise primary_error
            try:
                content = self.fallback_client.complete(messages, trace_id=trace_id, timeout_seconds=remaining)
                _remaining_response_seconds(deadline, self.monotonic_fn)
                return content
            except Exception as fallback_error:
                raise RuntimeError(
                    f"primary and fallback LLM calls failed; primary={primary_error}; fallback={fallback_error}"
                ) from fallback_error

    def _complete_primary_with_concurrency_limit(
        self,
        messages: list[dict[str, str]],
        *,
        trace_id: str,
        deadline: float,
    ) -> str:
        """在启动重试循环前应用提供方背压。

        队列等待会消耗与 HTTP 请求相同的端到端超时。这可以防止主代理、文案和审查者调用的突发压垮单个提供方，同时防止无界本地队列。
        """

        with self._concurrency_metrics_lock:
            self._waiting_provider_requests += 1
            self._max_waiting_provider_requests = max(
                self._max_waiting_provider_requests,
                self._waiting_provider_requests,
            )
        remaining = deadline - self.monotonic_fn()
        acquired = False
        try:
            if remaining > 0:
                acquired = self._concurrency_gate.acquire(timeout=remaining)
        finally:
            with self._concurrency_metrics_lock:
                self._waiting_provider_requests -= 1
        if not acquired:
            raise RuntimeError("LLM concurrency queue timed out before a provider slot became available")

        try:
            with self._concurrency_metrics_lock:
                self._active_provider_requests += 1
                self._max_active_provider_requests = max(
                    self._max_active_provider_requests,
                    self._active_provider_requests,
                )
            remaining = deadline - self.monotonic_fn()
            if remaining <= 0:
                raise RuntimeError("LLM request timed out while waiting for a provider slot")
            return self._complete_with_retry(messages, trace_id=trace_id, deadline=deadline)
        finally:
            with self._concurrency_metrics_lock:
                self._active_provider_requests -= 1
            self._concurrency_gate.release()

    def concurrency_metrics(self) -> dict[str, int | float | None]:
        """只公开背压、HTTP 尝试与传输恢复计数，不记录认证、请求正文或模型推理内容。"""
        with self._concurrency_metrics_lock:
            return {
                "configured_max_concurrency": max(1, int(self.config.max_concurrency)),
                "active_provider_requests": self._active_provider_requests,
                "waiting_provider_requests": self._waiting_provider_requests,
                "max_active_provider_requests": self._max_active_provider_requests,
                "max_waiting_provider_requests": self._max_waiting_provider_requests,
                "configured_attempt_timeout_seconds": self.config.attempt_timeout_seconds,
                **self._http_metrics,
            }

    def _record_http_metric(self, name: str) -> None:
        """在既有指标锁内累计传输事实，与模型动作合同纠错次数分开。"""
        with self._concurrency_metrics_lock:
            self._http_metrics[name] += 1

    def _record_http_timeout(self, error: Exception) -> None:
        """统计原生超时及 urllib 保留原异常的封装，不按错误文本猜测类型。"""
        if isinstance(error, TimeoutError) or (
            isinstance(error, urllib.error.URLError) and isinstance(error.reason, TimeoutError)
        ):
            self._record_http_metric("http_timeouts")

    def _read_response_body(self, response: Any, *, deadline: float) -> bytes:
        """在同一截止时间内读取完整正文；成功和 HTTP 错误响应共用此边界。

        CPython 的成功响应直接是 HTTPResponse，HTTPError.fp 才包装该对象。
        经核验的 HTTPResponse.fp.raw 是 SocketIO；局部替换其 _sock 不丢预读数据。
        这约束正文与 chunk 元数据，不更换 urllib 的连接、TLS 或响应头读取实现。
        仅有 read() 的注入测试响应保持单次读取，并在返回后核对期限。
        """
        actual = response.fp if isinstance(response, urllib.error.HTTPError) else response
        socket_io = proxy = original_socket = None
        _remaining_response_seconds(deadline, self.monotonic_fn)
        if isinstance(actual, HTTPResponse) and actual.fp is not None:
            socket_io = getattr(actual.fp, "raw", None)
            if not isinstance(socket_io, SocketIO) or getattr(socket_io, "_sock", None) is None:
                raise RuntimeError("HTTPResponse has no deadline-capable SocketIO")
            original_socket = socket_io._sock
            proxy = _DeadlineSocket(original_socket, deadline, self.monotonic_fn)
            socket_io._sock = proxy
        try:
            read1 = getattr(actual, "read1", None)
            if not callable(read1):
                body = actual.read()
                _remaining_response_seconds(deadline, self.monotonic_fn)
                return body
            expected_length = actual.length if isinstance(actual, HTTPResponse) and not actual.chunked else None
            chunks: list[bytes] = []
            received = 0
            while True:
                _remaining_response_seconds(deadline, self.monotonic_fn)
                chunk = read1(64 * 1024)
                _remaining_response_seconds(deadline, self.monotonic_fn)
                if not chunk:
                    break
                chunks.append(chunk)
                received += len(chunk)
            # read1 对 Content-Length 提前 EOF 不主动报错，必须保留原 read 的完整性保障。
            if expected_length is not None and received < expected_length:
                raise IncompleteRead(b"", expected_length - received)
            return b"".join(chunks)
        finally:
            # EOF/关闭时 SocketIO 已把 _sock 设为 None；不能恢复引用而复活已释放的连接。
            if socket_io is not None and socket_io._sock is proxy:
                socket_io._sock = original_socket

    def _complete_with_retry(
        self,
        messages: list[dict[str, str]],
        *,
        trace_id: str,
        deadline: float,
    ) -> str:
        """在同一总超时和次数预算内重试传输失败，完整返回后不替业务判断重跑模型。"""
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if self.config.response_format:
            payload["response_format"] = {"type": self.config.response_format}
        if self.config.provider == "deepseek":
            payload["thinking"] = {"type": self.config.thinking_mode}
            if self.config.thinking_mode == "enabled":
                # DeepSeek 推理模式不使用采样温度；强度只在明确启用且配置时发送。
                # 其他提供方继续遵守各自原合同，不透传这些专用参数。
                payload.pop("temperature")
                if self.config.reasoning_effort is not None:
                    payload["reasoning_effort"] = self.config.reasoning_effort
        elif self.config.provider in {"zai", "glm", "bigmodel"}:
            payload["thinking"] = {"type": "disabled"}
        last_error: Exception | None = None
        attempts = max(1, self.config.retry_attempts)
        for attempt in range(1, attempts + 1):
            remaining = deadline - self.monotonic_fn()
            if remaining <= 0:
                break
            attempt_deadline = deadline
            if self.config.attempt_timeout_seconds is not None:
                attempt_deadline = min(deadline, self.monotonic_fn() + self.config.attempt_timeout_seconds)
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
                request_timeout = _remaining_response_seconds(attempt_deadline, self.monotonic_fn)
                self._record_http_metric("http_attempts")
                if attempt > 1:
                    self._record_http_metric("http_retries")
                with self.urlopen(request, timeout=request_timeout) as response:
                    raw = json.loads(self._read_response_body(response, deadline=attempt_deadline).decode("utf-8"))
                content = content_from_response(raw)
                _remaining_response_seconds(attempt_deadline, self.monotonic_fn)
                if attempt > 1:
                    self._record_http_metric("http_recoveries")
                return content
            except urllib.error.HTTPError as exc:
                try:
                    with exc:
                        note = http_error_note(exc, self._read_response_body(exc, deadline=attempt_deadline))
                except (urllib.error.URLError, TimeoutError, IncompleteRead, ConnectionError) as body_error:
                    self._record_http_timeout(body_error)
                    note = f"LLM HTTP error {exc.code} {exc.reason}: response body failed: {type(body_error).__name__}: {body_error}"
                last_error = RuntimeError(note)
                if exc.code not in RETRYABLE_HTTP_STATUS:
                    raise last_error from exc
            # 响应体截断发生在JSON解析前；与既有网络失败共用次数与总超时预算，
            # 不把不完整文本交给模型动作解析，更不据语义判定结果重试。
            except (urllib.error.URLError, TimeoutError, IncompleteRead, ConnectionError, json.JSONDecodeError) as exc:
                self._record_http_timeout(exc)
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
        raise RuntimeError("LLM request timed out before response completed")


@dataclass(slots=True)
class StaticAgentClient:
    """按预设顺序返回模型输出的确定性测试客户端。"""

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


def http_error_note(exc: urllib.error.HTTPError, body: bytes) -> str:
    """只格式化已经受截止时间约束且完整读完的错误体，不能在错误处理里重新无界读取。"""
    return f"LLM HTTP error {exc.code} {exc.reason}: {body.decode('utf-8', errors='replace')[:300]}"


def default_base_url(provider: str) -> str:
    """构建默认baseurl实现，供未显式注入依赖的本地运行使用。"""
    if provider == "deepseek":
        return "https://api.deepseek.com"
    if provider in {"zai", "glm", "bigmodel"}:
        return "https://api.z.ai/api/paas/v4"
    return "https://api.openai.com/v1"


def fallback_client_from_env() -> "OpenAICompatibleAgentClient | None":
    """根据环境构建可选的独立回退提供方。"""

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
            env_int("MAHJONG_LLM_MAX_COMPLETION_TOKENS", 3200),
        ),
        response_format=os.getenv("MAHJONG_LLM_FALLBACK_RESPONSE_FORMAT", "json_object"),
        retry_attempts=max(1, env_int("MAHJONG_LLM_FALLBACK_RETRY_ATTEMPTS", 2)),
        retry_base_delay_seconds=max(0.0, env_float("MAHJONG_LLM_FALLBACK_RETRY_BASE_DELAY_SECONDS", 0.5)),
        retry_max_delay_seconds=max(0.0, env_float("MAHJONG_LLM_FALLBACK_RETRY_MAX_DELAY_SECONDS", 2.0)),
        max_estimated_tokens_per_day=max(
            0,
            env_int("MAHJONG_LLM_FALLBACK_MAX_ESTIMATED_TOKENS_PER_DAY", 0),
        ),
        max_concurrency=max(1, env_int("MAHJONG_LLM_FALLBACK_MAX_CONCURRENCY", 2)),
        attempt_timeout_seconds=optional_positive_env_float("MAHJONG_LLM_FALLBACK_ATTEMPT_TIMEOUT_SECONDS"),
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


def optional_positive_env_float(name: str) -> float | None:
    """未设置时维持原预算；显式空值、非数值或无限/非正预算必须报错。"""
    raw = os.getenv(name)
    if raw is None:
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a finite positive number") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return value
