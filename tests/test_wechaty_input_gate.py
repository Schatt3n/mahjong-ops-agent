from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from mahjong_agent_runtime import AgentRuntime, StaticAgentClient, UserMessage


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "scripts" / "agent_runtime_app.py"
spec = importlib.util.spec_from_file_location("agent_runtime_app_for_test", APP_PATH)
assert spec is not None and spec.loader is not None
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)


def test_parse_wechaty_input_gate_response_routes_operational_message() -> None:
    decision, errors = app.parse_wechaty_input_gate_response(
        json.dumps(
            {
                "should_route": True,
                "category": "operational",
                "confidence": 0.92,
                "reasoning_summary": "用户在问 0.5 的局。",
                "evidence": ["有没有 0.5"],
            },
            ensure_ascii=False,
        )
    )

    assert errors == []
    assert decision["should_route"] is True
    assert decision["category"] == "operational"


def test_parse_wechaty_input_gate_response_blocks_casual_chat() -> None:
    decision, errors = app.parse_wechaty_input_gate_response(
        json.dumps(
            {
                "should_route": False,
                "category": "casual_chat",
                "confidence": 0.88,
                "reasoning_summary": "朋友日常闲聊，不涉及麻将馆运营。",
                "evidence": ["晚上吃什么"],
            },
            ensure_ascii=False,
        )
    )

    assert errors == []
    assert decision["should_route"] is False
    assert decision["category"] == "casual_chat"


def test_run_wechaty_input_gate_fails_closed_on_invalid_contract(monkeypatch) -> None:
    monkeypatch.delenv("MAHJONG_WECHATY_INPUT_GATE_LLM_MODEL", raising=False)
    monkeypatch.setenv("MAHJONG_WECHATY_INPUT_GATE_ENABLED", "true")
    monkeypatch.setenv("MAHJONG_WECHATY_INPUT_GATE_FAIL_OPEN", "false")
    runtime = AgentRuntime(llm_client=StaticAgentClient(outputs=["不是 JSON"]))
    message = UserMessage(
        conversation_id="wechaty:contact:test",
        sender_id="friend",
        sender_name="朋友",
        text="晚上吃啥",
        message_id="msg_gate_invalid",
    )

    decision = app.run_wechaty_input_gate(message, trace_id="trace_gate_invalid", runtime=runtime)

    assert decision["enabled"] is True
    assert decision["should_route"] is False
    assert decision["errors"]


def test_run_wechaty_input_gate_uses_recent_context_for_short_answer(monkeypatch) -> None:
    monkeypatch.delenv("MAHJONG_WECHATY_INPUT_GATE_LLM_MODEL", raising=False)
    monkeypatch.setenv("MAHJONG_WECHATY_INPUT_GATE_ENABLED", "true")
    runtime = AgentRuntime(
        llm_client=StaticAgentClient(
            outputs=[
                json.dumps(
                    {
                        "should_route": True,
                        "category": "followup_answer",
                        "confidence": 0.91,
                        "reasoning_summary": "上一轮在问是否要组局，用户短答可以。",
                        "evidence": ["recent_conversation 中有麻将运营追问", "当前消息是可以"],
                    },
                    ensure_ascii=False,
                )
            ]
        )
    )
    runtime.store.append_assistant_turn(
        "wechaty:contact:test",
        "现在没有，要组一个吗？",
        "trace_prev",
        metadata={"delivery_status": "sent"},
    )
    message = UserMessage(
        conversation_id="wechaty:contact:test",
        sender_id="friend",
        sender_name="朋友",
        text="可以",
        message_id="msg_gate_followup",
    )

    decision = app.run_wechaty_input_gate(message, trace_id="trace_gate_followup", runtime=runtime)

    assert decision["should_route"] is True
    assert decision["category"] == "followup_answer"
