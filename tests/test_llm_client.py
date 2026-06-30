from __future__ import annotations

import json

import pytest

from mahjong_agent.budget import LLMBudgetLimits, LLMBudgetManager
from mahjong_agent.llm import LLMConfig
from mahjong_agent.llm_client import OpenAICompatibleSemanticLLMClient


class FakeHTTPResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload, ensure_ascii=False).encode("utf-8")


def test_openai_compatible_semantic_client_posts_chat_completion_and_audits() -> None:
    calls = []
    audits = []

    def fake_urlopen(request, timeout):
        calls.append({"request": request, "timeout": timeout})
        return FakeHTTPResponse(
            {
                "choices": [{"message": {"content": "{\"intent\":\"find_players\"}"}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
            }
        )

    client = OpenAICompatibleSemanticLLMClient(
        config=LLMConfig(
            api_key="secret-key",
            model="test-model",
            base_url="https://example.test/v1",
            max_completion_tokens=64,
            response_format="json_object",
            thinking_enabled=False,
        ),
        budget_manager=LLMBudgetManager(LLMBudgetLimits(max_calls_per_day=3, max_tokens_per_day=10_000)),
        audit_logger=lambda trace_id, event, payload: audits.append((trace_id, event, payload)),
        urlopen=fake_urlopen,
    )

    content = client.complete(
        [{"role": "system", "content": "只输出 JSON"}, {"role": "user", "content": "组局"}],
        trace_id="trace_llm_client",
        timeout_seconds=3.5,
    )

    assert content == "{\"intent\":\"find_players\"}"
    assert calls[0]["timeout"] == 3.5
    request = calls[0]["request"]
    assert request.full_url == "https://example.test/v1/chat/completions"
    payload = json.loads(request.data.decode("utf-8"))
    assert payload["model"] == "test-model"
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["messages"][1]["content"] == "组局"
    assert request.headers["Authorization"] == "Bearer secret-key"

    audit_text = json.dumps(audits, ensure_ascii=False)
    assert "secret-key" not in audit_text
    assert [item[1] for item in audits] == [
        "semantic_llm_budget",
        "semantic_llm_request",
        "semantic_llm_response",
    ]


def test_openai_compatible_semantic_client_fails_closed_when_budget_denied() -> None:
    called = False

    def fake_urlopen(request, timeout):
        nonlocal called
        called = True
        return FakeHTTPResponse({})

    client = OpenAICompatibleSemanticLLMClient(
        config=LLMConfig(api_key="secret-key", model="test-model"),
        budget_manager=LLMBudgetManager(LLMBudgetLimits(max_calls_per_day=0)),
        urlopen=fake_urlopen,
    )

    with pytest.raises(RuntimeError, match="LLM budget denied"):
        client.complete([{"role": "user", "content": "组局"}], trace_id="trace_budget", timeout_seconds=1)

    assert called is False
