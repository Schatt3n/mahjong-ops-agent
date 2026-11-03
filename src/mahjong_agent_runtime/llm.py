"""封装兼容 OpenAI 协议的模型调用、配置、用量记账与测试替身。"""

from __future__ import annotations

import json
import hashlib
from copy import deepcopy
from http.client import HTTPResponse, IncompleteRead
import math
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from datetime import date
from socket import SocketIO
from typing import Any, Protocol

from .token_estimation import estimate_tokens


RETRYABLE_HTTP_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
DIAGNOSTIC_REQUEST_ID_HEADERS = ("x-request-id", "request-id", "x-amzn-requestid", "x-ms-request-id", "x-bd-request-id")
DIAGNOSTIC_USAGE_FIELDS = (
    "prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens",
    "prompt_cache_hit_tokens", "prompt_cache_miss_tokens",
)


class LLMCompletionText(str):
    """保持字符串返回合同，同时把安全诊断绑定到本次结果，避免并发共享最后响应。"""

    def __new__(cls, content: str, diagnostics: dict[str, Any]) -> "LLMCompletionText":
        """复制已投影的诊断；JSON解析、判空与原有字符串行为保持一致。"""
        result = super().__new__(cls, content)
        result._completion_diagnostics = deepcopy(diagnostics)
        return result

    def __getnewargs__(self) -> tuple[str, dict[str, Any]]:
        """支持既有缓存与检查点的深复制，诊断不会使字符串结果无法持久化。"""
        return str(self), self._completion_diagnostics


class LLMResponseFormatError(RuntimeError):
    """响应缺少文本时保留原异常语义及诊断，不把协议错误伪装成模型动作。"""

    def __init__(self, message: str, diagnostics: dict[str, Any]) -> None:
        """异常只携带允许列表元数据，不保存供应商正文、请求头或推理内容。"""
        super().__init__(message)
        self._completion_diagnostics = deepcopy(diagnostics)


def completion_diagnostics(value: Any) -> dict[str, Any]:
    """读取本次返回值或异常的独立诊断快照；普通字符串及无诊断异常返回空字典。"""
    diagnostics = getattr(value, "_completion_diagnostics", None)
    return deepcopy(diagnostics) if isinstance(diagnostics, dict) else {}


_RESPONSE_SOURCE_KEYS = (
    "schema_version", "trace_id", "step_index", "raw_response_sha256", "call_id",
    "terminal_attempt_id", "route", "response_route", "http_response_complete",
)


def _source_generated_id(value: Any, prefix: str) -> str | None:
    """只允许客户端生成的有界不透明身份，不能把供应商字符串或测试标记当作来源。"""
    return value if isinstance(value, str) and re.fullmatch(prefix + r"[0-9a-f]{32}", value) else None


def _source_attempt_ids(attempts: Any) -> list[str] | None:
    """校验单次调用的有序且唯一attempt身份；空列表仅表示尚未开始HTTP的本地失败。"""
    if not isinstance(attempts, list):
        return None
    identities: list[str] = []
    for index, attempt in enumerate(attempts, start=1):
        identity = _source_generated_id(attempt.get("attempt_id"), "llm_attempt_") if isinstance(attempt, dict) else None
        if (identity is None or type(attempt.get("attempt_index")) is not int
                or attempt["attempt_index"] != index or identity in identities):
            return None
        identities.append(identity)
    return identities


def _returned_diagnostic_leaf(diagnostics: Any) -> tuple[dict[str, Any] | None, list[str] | None, bool]:
    """逐层验证失败primary与实际返回fallback的绑定，不借另一调用的成功正文证明来源。"""
    current, path, identities_valid = diagnostics, [], True
    seen: set[int] = set()
    call_ids: set[str] = set()
    failed_attempt_ids: set[str] = set()
    for _ in range(9):
        if (not isinstance(current, dict) or id(current) in seen
                or type(current.get("schema_version")) is not int or current["schema_version"] != 1):
            return None, None, False
        seen.add(id(current))
        current_call = _source_generated_id(current.get("call_id"), "llm_call_")
        if current_call is not None and current_call in call_ids:
            return None, None, False
        if current_call is not None:
            call_ids.add(current_call)
        identities_valid = identities_valid and current_call is not None
        if current.get("route") == "primary":
            terminal_ids = _source_attempt_ids(current.get("attempts"))
            return current, path, bool(identities_valid and terminal_ids is not None
                                       and not failed_attempt_ids.intersection(terminal_ids))
        if current.get("route") != "fallback":
            return None, None, False
        primary = current.get("primary")
        # complete()仅在本地primary抛错后进入fallback；成功wrapper不能缺失、错绑或伪造该失败分支。
        if (current.get("error_type") is not None or not isinstance(primary, dict)
                or type(primary.get("schema_version")) is not int or primary["schema_version"] != 1
                or primary.get("route") != "primary" or current_call is None
                or primary.get("call_id") != current_call):
            return None, None, False
        error_type = primary.get("error_type")
        if not isinstance(error_type, str) or not 0 < len(error_type) <= 160 or not error_type.isidentifier():
            return None, None, False
        primary_ids = _source_attempt_ids(primary.get("attempts"))
        if primary_ids is None or failed_attempt_ids.intersection(primary_ids):
            return None, None, False
        failed_attempt_ids.update(primary_ids)
        current = current.get("fallback")
        path.append("fallback")
    return None, None, False


def _model_response_source(raw_response: Any, *, trace_id: Any, step_index: Any, diagnostics: Any) -> dict[str, Any]:
    """构建来源值且不改变正文；HTTP完整性必须与实际返回叶节点和原始长度共同吻合。"""
    is_text = isinstance(raw_response, str)
    leaf, path, identities_valid = _returned_diagnostic_leaf(diagnostics)
    call_id = _source_generated_id(leaf.get("call_id"), "llm_call_") if leaf is not None else None
    terminal_id = None
    http_complete = False
    attempts = leaf.get("attempts") if leaf is not None else None
    if isinstance(attempts, list) and attempts:
        attempt_ids = _source_attempt_ids(attempts)
        terminal = attempts[-1] if isinstance(attempts[-1], dict) else {}
        if attempt_ids:
            terminal_id = attempt_ids[-1]
        content = terminal.get("content")
        expected_content = _diagnostic_text_field({"content": raw_response}, "content") if is_text else None
        # 诊断来自完整HTTP JSON，缺失/null/非字符串不能伪装成真正的空字符串。
        content_matches = isinstance(content, dict) and expected_content is not None and all(
            key in content and type(content[key]) is type(value) and content[key] == value
            for key, value in expected_content.items()
        )
        http_complete = bool(
            identities_valid and call_id is not None and terminal_id is not None and is_text
            and leaf.get("error_type") is None and terminal.get("error_type") is None
            and type(terminal.get("http_status")) is int and terminal["http_status"] == 200
            and terminal.get("body_complete") is True and terminal.get("response_json_parsed") is True
            and type(terminal.get("choice_count")) is int and terminal["choice_count"] >= 1 and content_matches
        )
    safe_trace = trace_id if isinstance(trace_id, str) and 0 < len(trace_id) <= 160 and trace_id.isprintable() else None
    return {
        "schema_version": 1, "trace_id": safe_trace,
        "step_index": step_index if type(step_index) is int and step_index > 0 else None,
        # surrogatepass只保证异常Unicode仍可审计，不清洗或替换模型正文。
        "raw_response_sha256": hashlib.sha256(raw_response.encode("utf-8", errors="surrogatepass")).hexdigest() if is_text else None,
        "call_id": call_id, "terminal_attempt_id": terminal_id,
        "route": diagnostics.get("route") if path is not None else None,
        "response_route": path, "http_response_complete": http_complete,
    }


def model_response_source(raw_response: Any, *, trace_id: str, step_index: int) -> dict[str, Any]:
    """把本次返回值的独立诊断绑定到trace/步骤/正文哈希，不保留正文、推理或认证。

    call_id和terminal_attempt_id指向实际返回的HTTP叶节点；response_route记录从外层
    调用到该叶节点的fallback路径。普通字符串仅有文本身份，不能为空响应提供HTTP证明。
    """
    return _model_response_source(raw_response, trace_id=trace_id, step_index=step_index,
                                  diagnostics=completion_diagnostics(raw_response))


def compact_model_response_source(source: Any) -> dict[str, Any] | None:
    """按严格类型和字段允许列表投影来源；额外诊断、正文与推理字段一律不进入上下文。"""
    if not isinstance(source, dict) or not all(key in source for key in _RESPONSE_SOURCE_KEYS):
        return None
    if (type(source["schema_version"]) is not int or source["schema_version"] != 1
            or not isinstance(source["trace_id"], str) or not 0 < len(source["trace_id"]) <= 160
            or not source["trace_id"].isprintable() or type(source["step_index"]) is not int or source["step_index"] <= 0
            or not isinstance(source["raw_response_sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", source["raw_response_sha256"]) is None
            or type(source["http_response_complete"]) is not bool):
        return None
    for key, prefix in (("call_id", "llm_call_"), ("terminal_attempt_id", "llm_attempt_")):
        if source[key] is not None and _source_generated_id(source[key], prefix) is None:
            return None
    route, path = source["route"], source["response_route"]
    if route is not None and (not isinstance(route, str) or route not in {"primary", "fallback"}):
        return None
    if (route is None and path is not None or route == "primary" and path != []
            or route == "fallback" and (not isinstance(path, list) or not 1 <= len(path) <= 8
                                        or any(item != "fallback" for item in path))):
        return None
    if source["http_response_complete"] and (source["call_id"] is None or source["terminal_attempt_id"] is None or route is None):
        return None
    return {key: deepcopy(source[key]) for key in _RESPONSE_SOURCE_KEYS}


def validate_model_response_source(
    source: Any, *, raw_response: str, trace_id: str, step_index: int,
    diagnostics: dict[str, Any] | None = None, require_http: bool = False,
) -> bool:
    """严格核对来源与本次原始响应；空串永远要求完整HTTP证明，不靠空哈希放行。

    评测从已保存trace传入diagnostics；生产返回值也可携带它。缺失诊断只能验证普通非空
    文本身份，require_http=True时绝不把普通字符串、缺失或错误attempt升级为HTTP来源。
    """
    compacted = compact_model_response_source(source)
    if compacted is None or compacted != source or not isinstance(raw_response, str):
        return False
    if diagnostics is not None and not isinstance(diagnostics, dict):
        return False
    actual_diagnostics = completion_diagnostics(raw_response) if diagnostics is None else diagnostics
    expected = _model_response_source(raw_response, trace_id=trace_id, step_index=step_index, diagnostics=actual_diagnostics)
    # 已提供调用诊断时不能在不一致后退化成普通文本来源；非空真实响应也须长度/末次attempt吻合。
    proof_required = require_http or raw_response == "" or bool(actual_diagnostics)
    return source == expected and (expected["http_response_complete"] or not proof_required)


def _diagnostic_string(value: Any, *, secrets: tuple[str, ...] = ()) -> str | None:
    """仅保留有界标识文本，并移除当前凭证的精确回显；不处理模型正文。"""
    if not isinstance(value, str):
        return None
    for secret in secrets:
        if secret:
            value = value.replace(secret, "[redacted]")
    return "".join(character if character.isprintable() else " " for character in value)[:160]


def _diagnostic_text_field(message: dict[str, Any], field_name: str) -> dict[str, Any]:
    """文本诊断只保留存在性、类型及长度，尤其不复制推理原文。"""
    value = message.get(field_name)
    is_text = isinstance(value, str)
    return {
        "present": field_name in message,
        "kind": "string" if is_text else "null" if value is None else "non_string",
        "length": len(value) if is_text else None,
        "empty": value == "" if is_text else None,
        "whitespace_only": bool(value) and not value.strip() if is_text else None,
    }


def _response_diagnostics(raw: Any, response: Any, *, secrets: tuple[str, ...]) -> dict[str, Any]:
    """完整JSON读取后立即投影协议元数据，文本缺失时也保留供应商终止与用量证据。"""
    payload = raw if isinstance(raw, dict) else {}
    choices = payload.get("choices")
    first = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first.get("message"), dict) else {}
    status = getattr(response, "status", None)
    headers = getattr(response, "headers", None)
    request_ids = {}
    if headers is not None and callable(getattr(headers, "get", None)):
        for name in DIAGNOSTIC_REQUEST_ID_HEADERS:
            value = _diagnostic_string(headers.get(name), secrets=secrets)
            if value is not None:
                request_ids[name] = value
    usage = {}
    raw_usage = payload.get("usage")
    if isinstance(raw_usage, dict):
        for name in DIAGNOSTIC_USAGE_FIELDS:
            value = raw_usage.get(name)
            if type(value) is int and 0 <= value <= 10**12:
                usage[name] = value
        for name, fields in (("completion_tokens_details", ("reasoning_tokens", "accepted_prediction_tokens", "rejected_prediction_tokens", "audio_tokens")),
                             ("prompt_tokens_details", ("cached_tokens", "audio_tokens"))):
            detail = raw_usage.get(name)
            if isinstance(detail, dict):
                projected = {key: detail[key] for key in fields if type(detail.get(key)) is int and 0 <= detail[key] <= 10**12}
                if projected:
                    usage[name] = projected
    return {
        "http_status": status if type(status) is int and 100 <= status <= 599 else None,
        "request_ids": request_ids,
        "response_id": _diagnostic_string(payload.get("id"), secrets=secrets),
        "returned_model": _diagnostic_string(payload.get("model"), secrets=secrets),
        "finish_reason": _diagnostic_string(first.get("finish_reason"), secrets=secrets),
        "usage": usage,
        "choice_count": len(choices) if isinstance(choices, list) else None,
        "content": _diagnostic_text_field(message, "content"),
        "reasoning_fields": {name: _diagnostic_text_field(message, name) for name in ("reasoning_content", "reasoning")},
        "body_complete": True,
        "response_json_parsed": True,
    }


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
        """保持原调用、重试与返回文本语义，仅把允许列表诊断绑定到单次结果或异常。"""
        started = time.perf_counter()
        diagnostics: dict[str, Any] = {
            "schema_version": 1, "call_id": f"llm_call_{uuid.uuid4().hex}", "route": "primary",
            "provider": _diagnostic_string(self.config.provider, secrets=(self.config.api_key,)),
            "requested_model": _diagnostic_string(self.config.model, secrets=(self.config.api_key,)),
            "attempts": [],
        }
        estimated = sum(estimate_tokens(item.get("content", "")) for item in messages) + self.config.max_tokens
        DailyTokenLedger.reserve(
            f"{self.config.provider}:{self.config.model}",
            estimated,
            self.config.max_estimated_tokens_per_day,
        )
        deadline = self.monotonic_fn() + max(0.0, timeout_seconds)
        try:
            content = self._complete_primary_with_concurrency_limit(
                messages,
                trace_id=trace_id,
                deadline=deadline,
                diagnostics=diagnostics,
            )
            diagnostics["total_elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
            return LLMCompletionText(content, diagnostics)
        except Exception as primary_error:
            diagnostics["error_type"] = type(primary_error).__name__
            diagnostics["total_elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
            primary_error._completion_diagnostics = deepcopy(diagnostics)
            if self.fallback_client is None:
                raise
            remaining = deadline - self.monotonic_fn()
            if remaining <= 0:
                raise primary_error
            try:
                content = self.fallback_client.complete(messages, trace_id=trace_id, timeout_seconds=remaining)
                _remaining_response_seconds(deadline, self.monotonic_fn)
                return LLMCompletionText(content, {
                    "schema_version": 1, "call_id": diagnostics["call_id"], "route": "fallback",
                    "primary": diagnostics, "fallback": completion_diagnostics(content),
                    "total_elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                })
            except Exception as fallback_error:
                combined = RuntimeError(
                    f"primary and fallback LLM calls failed; primary={primary_error}; fallback={fallback_error}"
                )
                combined._completion_diagnostics = {
                    "schema_version": 1, "call_id": diagnostics["call_id"], "route": "fallback",
                    "primary": diagnostics, "fallback": completion_diagnostics(fallback_error),
                    "error_type": type(fallback_error).__name__,
                    "total_elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                }
                raise combined from fallback_error

    def _complete_primary_with_concurrency_limit(
        self,
        messages: list[dict[str, str]],
        *,
        trace_id: str,
        deadline: float,
        diagnostics: dict[str, Any],
    ) -> str:
        """在启动重试循环前应用提供方背压。

        队列等待会消耗与 HTTP 请求相同的端到端超时。这可以防止主代理、文案和审查者调用的突发压垮单个提供方，同时防止无界本地队列。
        """

        queue_started = time.perf_counter()
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
            diagnostics["queue_wait_ms"] = round((time.perf_counter() - queue_started) * 1000, 3)
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
            return self._complete_with_retry(messages, trace_id=trace_id, deadline=deadline, diagnostics=diagnostics)
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
        diagnostics: dict[str, Any],
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
        # 记录真正发出的参数；未发送的推理选项保持未知，不能拿配置默认值冒充请求事实。
        diagnostics["request"] = {
            "max_tokens": payload["max_tokens"],
            "response_format": _diagnostic_string((payload.get("response_format") or {}).get("type"), secrets=(self.config.api_key,)),
            "thinking_mode": (payload.get("thinking") or {}).get("type"),
            "reasoning_effort": payload.get("reasoning_effort"),
        }
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
            attempt_diagnostics = {
                "attempt_id": f"llm_attempt_{uuid.uuid4().hex}", "attempt_index": attempt,
                "body_complete": False, "response_json_parsed": False,
            }
            diagnostics["attempts"].append(attempt_diagnostics)
            http_started = time.perf_counter()
            try:
                request_timeout = _remaining_response_seconds(attempt_deadline, self.monotonic_fn)
                self._record_http_metric("http_attempts")
                if attempt > 1:
                    self._record_http_metric("http_retries")
                with self.urlopen(request, timeout=request_timeout) as response:
                    attempt_diagnostics.update(_response_diagnostics({}, response, secrets=(self.config.api_key,)))
                    attempt_diagnostics.update(body_complete=False, response_json_parsed=False)
                    body = self._read_response_body(response, deadline=attempt_deadline)
                    attempt_diagnostics["body_complete"] = True
                    raw = json.loads(body.decode("utf-8"))
                    attempt_diagnostics.update(_response_diagnostics(raw, response, secrets=(self.config.api_key,)))
                content = content_from_response(raw, diagnostics=diagnostics)
                _remaining_response_seconds(attempt_deadline, self.monotonic_fn)
                if attempt > 1:
                    self._record_http_metric("http_recoveries")
                return content
            except urllib.error.HTTPError as exc:
                attempt_diagnostics.update(_response_diagnostics({}, exc, secrets=(self.config.api_key,)))
                attempt_diagnostics.update(http_status=exc.code, error_type=type(exc).__name__, body_complete=False, response_json_parsed=False)
                try:
                    with exc:
                        body = self._read_response_body(exc, deadline=attempt_deadline)
                        attempt_diagnostics["body_complete"] = True
                        note = http_error_note(exc, body)
                except (urllib.error.URLError, TimeoutError, IncompleteRead, ConnectionError) as body_error:
                    self._record_http_timeout(body_error)
                    note = f"LLM HTTP error {exc.code} {exc.reason}: response body failed: {type(body_error).__name__}: {body_error}"
                last_error = RuntimeError(note)
                if exc.code not in RETRYABLE_HTTP_STATUS:
                    raise last_error from exc
            # 响应体截断发生在JSON解析前；与既有网络失败共用次数与总超时预算，
            # 不把不完整文本交给模型动作解析，更不据语义判定结果重试。
            except (urllib.error.URLError, TimeoutError, IncompleteRead, ConnectionError, json.JSONDecodeError) as exc:
                attempt_diagnostics["error_type"] = type(exc).__name__
                self._record_http_timeout(exc)
                last_error = RuntimeError(f"LLM request failed: {type(exc).__name__}: {exc}")
            finally:
                attempt_diagnostics["http_elapsed_ms"] = round((time.perf_counter() - http_started) * 1000, 3)

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


def content_from_response(raw: dict[str, Any], *, diagnostics: dict[str, Any] | None = None) -> str:
    """仅提取协议中的原始文本；空串交给既有动作合同拒绝，缺失文本保留安全诊断。"""
    choices = raw.get("choices") if isinstance(raw, dict) else None
    if not isinstance(choices, list) or not choices:
        raise LLMResponseFormatError("LLM response has no choices", diagnostics or {})
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise LLMResponseFormatError("LLM response has no text content", diagnostics or {})
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
