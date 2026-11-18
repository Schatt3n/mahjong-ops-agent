from __future__ import annotations

import hashlib
import io
import json
import re

import pytest

from mahjong_agent_runtime.llm import (
    AgentLLMConfig,
    LLMCompletionText,
    LLMResponseFormatError,
    OPENCODE_GO_BASE_URL,
    OpenAICompatibleAgentClient,
    completion_diagnostics,
    model_response_source,
    validate_model_response_source,
)


MESSAGES = [
    {"role": "system", "content": "Return a JSON object."},
    {"role": "user", "content": "This is an isolated transport test."},
]
TRACE_ID = "trace_test_responses"
SYNTHETIC_KEY = "test-opencode-go-key"
REASONING_MARKER = "synthetic reasoning text must stay out of diagnostics"
REFUSAL_MARKER = "synthetic refusal text must stay out of diagnostics"


class JsonResponse(io.BytesIO):
    status = 200
    headers = {"x-opencode-log-id": "req_test_responses"}

    def __init__(self, document: dict) -> None:
        super().__init__(json.dumps(document).encode("utf-8"))


def api_response(parts: list[dict], **overrides: object) -> dict:
    return {
        "id": "resp_test_01",
        "object": "response",
        "status": "completed",
        "model": "gpt-6-luna",
        "output": [
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": REASONING_MARKER}]},
            {"type": "message", "role": "assistant", "status": "completed", "content": parts},
        ],
        "usage": {
            "input_tokens": 47,
            "output_tokens": 21,
            "total_tokens": 68,
            "input_tokens_details": {"cached_tokens": 16},
            "output_tokens_details": {"reasoning_tokens": 9},
        },
        **overrides,
    }


def client_with_response(document: dict, *, response_format: str = "json_object", thinking_mode: str = "enabled"):
    requests = []

    def urlopen(request, *, timeout):
        requests.append(request)
        return JsonResponse(document)

    config = AgentLLMConfig(
        api_key=SYNTHETIC_KEY, provider="opencode_go", api_protocol="responses",
        base_url=OPENCODE_GO_BASE_URL, model="gpt-6-luna", max_tokens=7200,
        response_format=response_format, thinking_mode=thinking_mode, reasoning_effort="low",
    )
    return OpenAICompatibleAgentClient(config=config, urlopen=urlopen), requests


@pytest.mark.parametrize("response_format,thinking_mode", [("json_object", "enabled"), ("text", "enabled"), ("", "disabled")])
def test_opencode_go_responses_uses_responses_wire_contract(response_format, thinking_mode):
    client, requests = client_with_response(
        api_response([{"type": "output_text", "text": '{"ok":true}'}]),
        response_format=response_format, thinking_mode=thinking_mode,
    )

    result = client.complete(MESSAGES, trace_id=TRACE_ID, timeout_seconds=2)

    assert result == '{"ok":true}'
    (request,) = requests
    assert request.full_url == "https://opencode.ai/inference/go/openai/v1/responses"
    assert request.get_method() == "POST"
    headers = {name.lower(): value for name, value in request.header_items()}
    assert headers["authorization"] == f"Bearer {SYNTHETIC_KEY}"
    assert headers["user-agent"].startswith("mahjong-ops-agent/")
    assert re.fullmatch(r"ses_[a-f0-9]{32}", headers["x-opencode-session"])
    payload = json.loads(request.data)
    assert payload["model"] == "gpt-6-luna"
    assert payload["input"] == MESSAGES
    assert payload["max_output_tokens"] == 7200
    assert payload["store"] is False
    if thinking_mode == "enabled":
        assert payload["reasoning"] == {"effort": "low"}
    else:
        assert "reasoning" not in payload
    if response_format:
        assert payload["text"]["format"] == {"type": response_format}
    else:
        assert "format" not in payload.get("text", {})
    assert {"messages", "max_tokens", "max_completion_tokens", "temperature", "thinking", "reasoning_effort", "response_format"}.isdisjoint(payload)


def test_responses_combines_exact_visible_parts_and_keeps_http_evidence():
    parts = ('  {"reply":', '"测试"}\n')
    document = api_response([{"type": "output_text", "text": text} for text in parts])
    document["output"].insert(1, {
        "type": "message", "role": "assistant", "status": "completed", "phase": "commentary",
        "content": [{"type": "output_text", "text": "This preliminary note is not the answer."}],
    })
    client, _ = client_with_response(document)

    result = client.complete(MESSAGES, trace_id=TRACE_ID, timeout_seconds=2)

    assert result == "".join(parts)
    diagnostics = completion_diagnostics(result)
    (attempt,) = diagnostics["attempts"]
    assert attempt["api"] == "responses"
    assert attempt["response_status"] == "completed"
    assert attempt["output_count"] == 3
    assert attempt.get("choice_count") is None
    assert attempt["response_id"] == "resp_test_01"
    assert attempt["request_ids"] == {"x-opencode-log-id": "req_test_responses"}
    assert attempt["http_status"] == 200
    assert attempt["response_json_parsed"] is True and attempt["body_complete"] is True
    assert attempt["content"]["kind"] == "output_text_blocks"
    assert attempt["content"]["text_block_count"] == 2
    assert attempt["content"]["length"] == len(result)
    assert attempt["content"]["sha256"] == hashlib.sha256(result.encode("utf-8")).hexdigest()
    assert attempt["usage"]["prompt_tokens"] == 47
    assert attempt["usage"]["completion_tokens"] == 21
    assert attempt["usage"]["total_tokens"] == 68
    assert attempt["usage"]["prompt_tokens_details"] == {"cached_tokens": 16}
    assert attempt["usage"]["completion_tokens_details"] == {"reasoning_tokens": 9}
    encoded = json.dumps(diagnostics)
    assert REASONING_MARKER not in encoded and SYNTHETIC_KEY not in encoded
    source = model_response_source(result, trace_id=TRACE_ID, step_index=1)
    assert source["http_response_complete"] is True
    assert validate_model_response_source(source, raw_response=result, trace_id=TRACE_ID, step_index=1, require_http=True)
    replaced_text = LLMCompletionText(result.replace("测试", "替换"), diagnostics)
    assert model_response_source(replaced_text, trace_id=TRACE_ID, step_index=1)["http_response_complete"] is False


def test_responses_distinguishes_explicit_empty_text_from_missing_text():
    empty, _ = client_with_response(api_response([{"type": "output_text", "text": ""}]))
    missing, _ = client_with_response(api_response([]))

    result = empty.complete(MESSAGES, trace_id=TRACE_ID, timeout_seconds=2)

    assert result == ""
    source = model_response_source(result, trace_id=TRACE_ID, step_index=1)
    assert validate_model_response_source(source, raw_response=result, trace_id=TRACE_ID, step_index=1, require_http=True)
    with pytest.raises(LLMResponseFormatError, match="no text content"):
        missing.complete(MESSAGES, trace_id=TRACE_ID, timeout_seconds=2)


@pytest.mark.parametrize(
    "parts,overrides,message",
    [
        ([{"type": "refusal", "refusal": REFUSAL_MARKER}], {}, "refused"),
        (
            [{"type": "output_text", "text": '{"ok":true}'}],
            {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}},
            "not completed",
        ),
    ],
    ids=["refusal", "incomplete-with-valid-partial-json"],
)
def test_responses_failure_never_turns_partial_output_into_agent_actions(parts, overrides, message):
    client, requests = client_with_response(api_response(parts, **overrides))

    with pytest.raises(LLMResponseFormatError, match=message) as exc:
        client.complete(MESSAGES, trace_id=TRACE_ID, timeout_seconds=2)

    assert len(requests) == 1
    diagnostics = completion_diagnostics(exc.value)
    (attempt,) = diagnostics["attempts"]
    assert attempt["api"] == "responses"
    assert attempt["response_status"] == overrides.get("status", "completed")
    if overrides:
        assert attempt["incomplete_reason"] == "max_output_tokens"
    else:
        assert attempt["refusal_count"] == 1
    assert REFUSAL_MARKER not in str(exc.value) and REFUSAL_MARKER not in json.dumps(diagnostics)
