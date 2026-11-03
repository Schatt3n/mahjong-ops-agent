"""逐次调用诊断须可解释空响应，并保持原文本、重试、异常与并发隔离合同。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from email.message import Message
import io
import json
import threading
import urllib.error

import pytest

from mahjong_agent_runtime import AgentLLMConfig, OpenAICompatibleAgentClient, StaticAgentClient
from mahjong_agent_runtime.action_contract import parse_action_with_repairs
from mahjong_agent_runtime.llm import LLMCompletionText, completion_diagnostics


class Response:
    """仅替代HTTP传输，允许测试真实客户端的协议与允许列表投影。"""

    def __init__(self, payload, *, status=200, headers=None):
        self.body = json.dumps(payload).encode()
        self.status = status
        self.headers = Message()
        for name, value in (headers or {}).items():
            self.headers[name] = value

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return self.body


def config(**updates):
    """使用不可联网的测试配置，不读取实际环境凭证。"""
    values = dict(api_key="unit-only-secret-key", model="requested-model", provider="deepseek",
        base_url="https://example.invalid/v1", retry_attempts=3,
        retry_base_delay_seconds=0, retry_max_delay_seconds=0)
    values.update(updates)
    return AgentLLMConfig(**values)


def response_payload(content="", **updates):
    """空文本与推理文本同时存在，诊断只能保留推理长度和计数。"""
    result = {"id": "completion-id", "model": "returned-model", "choices": [{
        "finish_reason": "length", "message": {"content": content, "reasoning_content": "PRIVATE_REASONING"}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 32, "total_tokens": 132,
                  "completion_tokens_details": {"reasoning_tokens": 32}}}
    result.update(updates)
    return result


def complete(client, *, trace_id="diagnostic-test"):
    return client.complete([{"role": "user", "content": "原请求正文不进入诊断"}], trace_id=trace_id, timeout_seconds=2)


def test_empty_content_keeps_original_rejection_and_diagnostics_without_retry():
    calls = []
    payload = response_payload()
    client = OpenAICompatibleAgentClient(config=config(max_tokens=32, thinking_mode="enabled", reasoning_effort="high"),
        urlopen=lambda request, **kwargs: calls.append(request) or Response(payload, headers={"X-Request-ID": "provider-req"}))
    content = complete(client)
    assert isinstance(content, str) and content == "" and not content
    assert len(calls) == 1 and client.concurrency_metrics()["http_retries"] == 0
    _, errors, _ = parse_action_with_repairs(content)
    assert errors and "not valid JSON" in errors[0]
    diagnostics = completion_diagnostics(content)
    assert diagnostics["provider"] == "deepseek"
    assert diagnostics["requested_model"] == "requested-model"
    assert diagnostics["request"] == {"max_tokens": 32, "response_format": "json_object", "thinking_mode": "enabled", "reasoning_effort": "high"}
    assert diagnostics["route"] == "primary" and diagnostics["queue_wait_ms"] >= 0
    attempt = diagnostics["attempts"][0]
    assert attempt["http_status"] == 200 and attempt["request_ids"] == {"x-request-id": "provider-req"}
    assert attempt["returned_model"] == "returned-model" and attempt["finish_reason"] == "length"
    assert attempt["usage"]["completion_tokens_details"] == {"reasoning_tokens": 32}
    assert attempt["content"] == {"present": True, "kind": "string", "length": 0, "empty": True, "whitespace_only": False}
    assert attempt["reasoning_fields"]["reasoning_content"]["length"] == len("PRIVATE_REASONING")
    assert attempt["body_complete"] and attempt["response_json_parsed"] and attempt["http_elapsed_ms"] >= 0
    assert "PRIVATE_REASONING" not in json.dumps(diagnostics)


@pytest.mark.parametrize("content", ["  \n\t", '{"ok":true}', "答复"])
def test_original_text_and_json_serialization_are_preserved(content):
    client = OpenAICompatibleAgentClient(config=config(), urlopen=lambda *a, **k: Response(response_payload(content)))
    result = complete(client)
    assert str(result) == content and json.loads(json.dumps(result)) == content
    info = completion_diagnostics(result)["attempts"][0]["content"]
    assert info["length"] == len(content) and info["whitespace_only"] is (not content.strip())
    assert completion_diagnostics(deepcopy(result)) == completion_diagnostics(result)
    copied = completion_diagnostics(result)
    copied["attempts"][0]["content"]["length"] = 999
    assert completion_diagnostics(result)["attempts"][0]["content"]["length"] == len(content)


@pytest.mark.parametrize("payload,expected", [
    (response_payload(None), "no text content"),
    ({"model": "returned-model", "usage": {"completion_tokens": 0}, "choices": []}, "no choices"),
    ({"choices": [{"finish_reason": "stop", "message": {"reasoning_content": "ONLY_REASONING"}}]}, "no text content"),
])
def test_missing_text_exception_keeps_response_diagnostics(payload, expected):
    calls = []
    client = OpenAICompatibleAgentClient(config=config(), urlopen=lambda *a, **k: calls.append(1) or Response(payload))
    with pytest.raises(RuntimeError, match=expected) as caught:
        complete(client)
    assert len(calls) == 1
    attempt = completion_diagnostics(caught.value)["attempts"][0]
    assert attempt["body_complete"] is True and attempt["response_json_parsed"] is True
    assert attempt["http_status"] == 200
    assert "ONLY_REASONING" not in json.dumps(completion_diagnostics(caught.value))


def test_diagnostics_drop_unknown_headers_usage_and_bound_and_redact_identifiers():
    secret = "unit-only-secret-key"
    payload = response_payload("reply", id=f"{secret}-id", model=f"{secret}" + "X" * 400)
    payload["choices"][0]["finish_reason"] = "stop" + "Y" * 400
    payload["usage"] = {"prompt_tokens": True, "completion_tokens": "32", "total_tokens": -1,
        "reasoning_tokens": 5, "unknown_secret": secret,
        "completion_tokens_details": {"reasoning_tokens": 5, "debug_text": secret}}
    client = OpenAICompatibleAgentClient(config=config(), urlopen=lambda *a, **k: Response(payload, headers={
        "X-Request-ID": secret + "Q" * 400, "Authorization": "Bearer " + secret,
        "Set-Cookie": "cookie-secret", "X-Debug-Private": "header-private"}))
    diagnostics = completion_diagnostics(complete(client))
    text = json.dumps(diagnostics)
    assert all(value not in text for value in [secret, "cookie-secret", "header-private", "debug_text", "unknown_secret", "PRIVATE_REASONING"])
    attempt = diagnostics["attempts"][0]
    assert attempt["usage"] == {"reasoning_tokens": 5, "completion_tokens_details": {"reasoning_tokens": 5}}
    assert set(attempt["request_ids"]) == {"x-request-id"}
    assert len(attempt["returned_model"]) <= 160 and len(attempt["finish_reason"]) <= 160
    assert len(attempt["request_ids"]["x-request-id"]) <= 160


def test_retry_history_has_unique_attempt_ids_and_keeps_existing_retry_count():
    count = 0
    def transport(request, **kwargs):
        nonlocal count
        count += 1
        if count == 1:
            raise urllib.error.HTTPError("https://example.invalid", 503, "busy", {"x-request-id": "failed-request"}, io.BytesIO(b"busy"))
        return Response(response_payload("reply"))
    client = OpenAICompatibleAgentClient(config=config(), urlopen=transport)
    diagnostics = completion_diagnostics(complete(client))
    assert count == 2
    first, second = diagnostics["attempts"]
    assert first["http_status"] == 503 and first["request_ids"] == {"x-request-id": "failed-request"}
    assert first["attempt_index"] == 1 and second["attempt_index"] == 2
    assert first["attempt_id"] != second["attempt_id"]
    assert client.concurrency_metrics()["http_retries"] == 1


def test_fallback_diagnostics_name_actual_route_and_both_calls():
    fallback = OpenAICompatibleAgentClient(config=config(provider="other", model="fallback-model"),
        urlopen=lambda *a, **k: Response(response_payload("fallback reply", model="actual-fallback")))
    def failed_primary(*args, **kwargs):
        raise urllib.error.HTTPError("https://example.invalid", 401, "denied", {}, io.BytesIO(b"denied"))
    client = OpenAICompatibleAgentClient(config=config(), urlopen=failed_primary, fallback_client=fallback)
    result = complete(client)
    diagnostics = completion_diagnostics(result)
    assert result == "fallback reply" and diagnostics["route"] == "fallback"
    assert diagnostics["primary"]["provider"] == "deepseek"
    assert diagnostics["primary"]["attempts"][0]["http_status"] == 401
    assert diagnostics["fallback"]["provider"] == "other"
    assert diagnostics["fallback"]["requested_model"] == "fallback-model"
    assert diagnostics["fallback"]["attempts"][0]["returned_model"] == "actual-fallback"
    assert diagnostics["call_id"] != diagnostics["fallback"]["call_id"]
    assert diagnostics["fallback"]["request"]["thinking_mode"] is None


def test_opaque_fallback_is_marked_without_inventing_provider_evidence():
    client = OpenAICompatibleAgentClient(config=config(retry_attempts=1), fallback_client=StaticAgentClient(["fallback"]),
        urlopen=lambda *a, **k: (_ for _ in ()).throw(ConnectionError("offline")))
    diagnostics = completion_diagnostics(complete(client))
    assert diagnostics["route"] == "fallback" and diagnostics["fallback"] == {}
    assert completion_diagnostics("plain") == {} and completion_diagnostics(RuntimeError("plain")) == {}


def test_concurrent_results_keep_distinct_call_and_attempt_identity():
    barrier = threading.Barrier(4)
    def transport(request, **kwargs):
        index = json.loads(request.data)["messages"][0]["content"]
        barrier.wait(2)
        return Response(response_payload(index, id=f"response-{index}"), headers={"x-request-id": f"request-{index}"})
    client = OpenAICompatibleAgentClient(config=config(max_concurrency=4), urlopen=transport)
    def run(index):
        result = client.complete([{"role": "user", "content": str(index)}], trace_id="same-trace", timeout_seconds=3)
        return str(result), completion_diagnostics(result)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run, range(4)))
    assert len({diagnostics["call_id"] for _, diagnostics in results}) == 4
    assert len({diagnostics["attempts"][0]["attempt_id"] for _, diagnostics in results}) == 4
    for text, diagnostics in results:
        assert diagnostics["attempts"][0]["response_id"] == f"response-{text}"
        assert diagnostics["attempts"][0]["request_ids"] == {"x-request-id": f"request-{text}"}


def test_constructed_completion_text_does_not_share_mutable_diagnostics():
    source = {"attempts": [{"finish_reason": "stop"}]}
    result = LLMCompletionText("result", source)
    source["attempts"][0]["finish_reason"] = "changed"
    assert completion_diagnostics(result)["attempts"][0]["finish_reason"] == "stop"
