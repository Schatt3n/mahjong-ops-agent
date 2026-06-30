from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo

from mahjong_agent.trial_tool_planning import (
    TrialToolCallNormalizer,
    TrialToolPlanPromptBuilder,
    TrialToolPlanPromptInput,
)


TZ = ZoneInfo("Asia/Shanghai")


def make_prompt_input() -> TrialToolPlanPromptInput:
    return TrialToolPlanPromptInput(
        stage="after_open_game_search",
        now=datetime(2026, 6, 28, 22, 55, tzinfo=TZ),
        sender_id="zhang",
        sender_name="张哥",
        customer_profile={"display_name": "张哥", "preferred_levels": ["0.5", "1"]},
        source_text="可以",
        effective_text="通宵0.5有人吗\n可以",
        workflow_followup_context={"previous_system_suggested_reply": "0.5的暂时没有诶。要组一个吗？"},
        text_normalization={"normalized_text": "通宵0.5有人吗\n可以", "changed": False},
        decision_action="ask_clarification",
        parsed_game={"level": "0.5", "missing_count": 3},
        missing_fields=["start_time", "known_players"],
        critical_fields={"start_time", "known_players", "stake", "smoke", "duration"},
        available_tools=[
            {"name": "search_candidate_customers", "risk_level": "low"},
            {"name": "send_message", "risk_level": "high", "allowed_execution_modes": ["create_pending_outbox"]},
        ],
        tool_registry_version="tool_registry.v1",
        existing_tool_results={"search_current_open_games": {"called": True, "result_count": 0}},
        active_skills=[{"id": "multi_turn_slot_filling", "instructions": ["结合上一轮回复"]}],
    )


def test_trial_tool_plan_prompt_builder_builds_payload_contract() -> None:
    builder = TrialToolPlanPromptBuilder()

    payload = builder.build_payload(
        make_prompt_input(),
        model="deepseek-v4-flash",
        temperature=0.1,
        max_tokens=260,
        thinking_enabled=False,
        response_format="json_object",
    )

    assert payload["model"] == "deepseek-v4-flash"
    assert payload["temperature"] == 0.1
    assert payload["max_tokens"] == 260
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["response_format"] == {"type": "json_object"}
    assert "工具规划器" in payload["messages"][0]["content"]

    prompt = json.loads(payload["messages"][1]["content"])
    assert prompt["stage"] == "after_open_game_search"
    assert prompt["now"] == "2026-06-28 22:55:00"
    assert prompt["sender"] == {"id": "zhang", "name": "张哥"}
    assert prompt["workflow_followup_context"]["previous_system_suggested_reply"].endswith("要组一个吗？")
    assert prompt["critical_missing_fields"] == ["known_players", "start_time"]
    assert prompt["available_tools"][1]["name"] == "send_message"
    assert prompt["existing_tool_results"]["search_current_open_games"]["result_count"] == 0
    assert any("ToolGateway" in item for item in prompt["rules"])


def test_trial_tool_plan_prompt_builder_omits_optional_response_controls() -> None:
    payload = TrialToolPlanPromptBuilder().build_payload(
        make_prompt_input(),
        model="test-model",
        temperature=0.2,
        max_tokens=128,
    )

    assert "thinking" not in payload
    assert "response_format" not in payload


def test_trial_tool_call_normalizer_filters_unknown_duplicates_and_normalizes_send_mode() -> None:
    raw_calls = [
        {"tool_name": "unknown_tool", "arguments": {}, "reason": "不要执行"},
        {"tool_name": "search_candidate_customers", "arguments": {"limit": 5}, "reason": "找候选人"},
        {"name": "search_candidate_customers", "arguments": {"limit": 99}, "reason": "重复"},
        {
            "tool_name": "send_message",
            "arguments": {"execution_mode": "direct_send", "extra": "keep"},
            "call_reason": "创建邀约草稿",
        },
        "bad-call",
    ]
    available_tools = [
        {"name": "search_candidate_customers"},
        {"name": "send_message", "allowed_execution_modes": ["create_pending_outbox"]},
    ]

    normalized = TrialToolCallNormalizer().normalize(raw_calls, available_tools)

    assert [item["tool_name"] for item in normalized] == ["search_candidate_customers", "send_message"]
    assert normalized[0] == {
        "tool_name": "search_candidate_customers",
        "arguments": {"limit": 5},
        "reason": "找候选人",
        "requested_by": "llm",
    }
    assert normalized[1]["arguments"] == {
        "execution_mode": "create_pending_outbox",
        "requested_execution_mode": "direct_send",
        "extra": "keep",
    }
    assert normalized[1]["reason"] == "创建邀约草稿"


def test_trial_tool_call_normalizer_handles_non_list_and_bad_arguments() -> None:
    normalizer = TrialToolCallNormalizer()

    assert normalizer.normalize({"tool_name": "send_message"}, [{"name": "send_message"}]) == []
    assert normalizer.normalize(
        [{"tool_name": "send_message", "arguments": "bad"}],
        [{"name": "send_message"}],
    ) == [
        {
            "tool_name": "send_message",
            "arguments": {"execution_mode": "create_pending_outbox"},
            "reason": "LLM 请求调用工具。",
            "requested_by": "llm",
        }
    ]
