"""逐次调用诊断须可解释空响应，并保持原文本、重试、异常与并发隔离合同。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from email.message import Message
import io
import json
import socket
import ssl
import threading
import time
import urllib.error

import pytest

from mahjong_agent_runtime import AgentLLMConfig, OpenAICompatibleAgentClient, StaticAgentClient
from mahjong_agent_runtime.action_contract import parse_action_with_repairs
from mahjong_agent_runtime.llm import (
    ExternalCallBudgetExhausted, LLMCompletionText, completion_diagnostics, eval_transport_failure, external_call_deadline,
)


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
    assert eval_transport_failure(caught.value) == {
        "category": "response_protocol_error", "retryable": False, "reason": "response_format_invalid",
    }
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
    assert first["failure_reason"] == "retryable_http_status" and eval_transport_failure(diagnostics) is None
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


@pytest.mark.parametrize("failure,category,retryable,reason", [
    (TimeoutError("slow"), "unattributed_timeout", True, "timeout_waiting_for_response_headers"),
    (urllib.error.URLError(TimeoutError("slow")), "unattributed_timeout", True, "timeout_waiting_for_response_headers"),
    (ConnectionResetError("reset"), "network_error", True, "connection_error"),
    (urllib.error.URLError(socket.gaierror(socket.EAI_AGAIN, "temporary")), "network_error", True, "temporary_dns_failure"),
    (urllib.error.URLError(socket.gaierror(socket.EAI_NONAME, "not found")), "network_error", False, "dns_lookup_failed"),
    (urllib.error.URLError(ssl.SSLCertVerificationError("untrusted")), "network_error", False, "tls_error"),
    (urllib.error.URLError("timeout"), "unattributed_transport_failure", False, "unclassified_transport_error"),
])
def test_eval_transport_failure_preserves_typed_evidence_without_guessing_from_text(failure, category, retryable, reason):
    def transport(*args, **kwargs):
        raise failure

    client = OpenAICompatibleAgentClient(config=config(retry_attempts=1), urlopen=transport)
    with pytest.raises(RuntimeError) as caught:
        complete(client)
    expected = {"category": category, "retryable": retryable, "reason": reason}
    assert eval_transport_failure(caught.value) == expected
    diagnostics = completion_diagnostics(caught.value)
    assert eval_transport_failure(diagnostics) == expected
    assert diagnostics["timeout_budget_seconds"] == 2
    assert eval_transport_failure(failure) is None


@pytest.mark.parametrize("status,retryable", [(429, True), (504, True), (401, False), (409, False)])
def test_eval_http_status_is_peer_evidence_with_separate_eval_retry_eligibility(status, retryable):
    def transport(*args, **kwargs):
        raise urllib.error.HTTPError("https://example.invalid", status, "error", {}, io.BytesIO(b"body"))

    client = OpenAICompatibleAgentClient(config=config(retry_attempts=1), urlopen=transport)
    with pytest.raises(RuntimeError) as caught:
        complete(client)
    info = eval_transport_failure(caught.value)
    assert info["category"] == "provider_or_gateway" and info["retryable"] is retryable
    assert completion_diagnostics(caught.value)["attempts"][0]["http_status"] == status


def test_eval_timeout_reading_http_200_with_incomplete_body_keeps_origin_unknown():
    class StalledResponse(Response):
        def read(self):
            raise TimeoutError("test stalled body")

    client = OpenAICompatibleAgentClient(config=config(retry_attempts=1), urlopen=lambda *a, **k: StalledResponse({}))
    with pytest.raises(RuntimeError) as caught:
        complete(client)
    attempt = completion_diagnostics(caught.value)["attempts"][0]
    assert attempt["http_status"] == 200 and attempt["body_complete"] is False
    assert eval_transport_failure(caught.value) == {
        "category": "unattributed_timeout", "retryable": True, "reason": "timeout_reading_response_body",
    }


def test_local_queue_and_budget_failures_are_visible_and_do_not_get_eval_retries():
    class Busy:
        def acquire(self, *, timeout):
            return False

    for settings, category, reason in [
        ({}, "local_queue", "provider_slot_timeout"),
        ({"max_estimated_tokens_per_day": 1}, "local_budget", "daily_token_budget"),
    ]:
        def unexpected_transport(*args, **kwargs):
            pytest.fail("local failure before HTTP")

        client = OpenAICompatibleAgentClient(config=config(**settings), urlopen=unexpected_transport)
        client._concurrency_gate = Busy()
        with pytest.raises(RuntimeError) as caught:
            complete(client)
        assert eval_transport_failure(caught.value) == {"category": category, "retryable": False, "reason": reason}
        assert completion_diagnostics(caught.value)["attempts"] == []


def test_mixed_fallback_failure_keeps_permanent_primary_error():
    def permanent_primary(*args, **kwargs):
        raise urllib.error.HTTPError("https://example.invalid", 401, "denied", {}, io.BytesIO(b"denied"))

    def transient_fallback(*args, **kwargs):
        raise TimeoutError("fallback timed out")

    fallback = OpenAICompatibleAgentClient(config=config(retry_attempts=1), urlopen=transient_fallback)
    client = OpenAICompatibleAgentClient(config=config(retry_attempts=1), urlopen=permanent_primary, fallback_client=fallback)
    with pytest.raises(RuntimeError) as caught:
        complete(client)
    assert eval_transport_failure(caught.value) == {
        "category": "provider_or_gateway", "retryable": False, "reason": "other_http_status",
    }


def test_eval_context_caps_http_budget_and_preserves_requested_value():
    http_timeouts = []

    def transport(*args, timeout):
        http_timeouts.append(timeout)
        return Response(response_payload("reply"))

    client = OpenAICompatibleAgentClient(config=config(retry_attempts=1), urlopen=transport)
    token = external_call_deadline.set(time.monotonic() + 1)
    try:
        diagnostics = completion_diagnostics(complete(client))
        assert diagnostics["requested_timeout_budget_seconds"] == 2
        assert 0 < http_timeouts[0] <= diagnostics["timeout_budget_seconds"] <= 1
    finally:
        external_call_deadline.reset(token)
    diagnostics = completion_diagnostics(complete(client))
    assert diagnostics["timeout_budget_seconds"] == 2 and "requested_timeout_budget_seconds" not in diagnostics


def test_exhausted_eval_context_stops_before_http_and_fallback():
    def unexpected_transport(*args, **kwargs):
        pytest.fail("external deadline expired before HTTP")

    fallback = StaticAgentClient(["fallback"])
    client = OpenAICompatibleAgentClient(config=config(retry_attempts=1), urlopen=unexpected_transport, fallback_client=fallback)
    token = external_call_deadline.set(time.monotonic() - 1)
    try:
        with pytest.raises(ExternalCallBudgetExhausted) as caught:
            complete(client)
        assert completion_diagnostics(caught.value)["attempts"] == []
        assert not fallback.calls
        assert eval_transport_failure(caught.value) == {
            "category": "local_budget", "retryable": False, "reason": "external_call_budget_exhausted",
        }
    finally:
        external_call_deadline.reset(token)


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
