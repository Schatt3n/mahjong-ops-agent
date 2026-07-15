from __future__ import annotations

import json

from mahjong_agent_runtime import (
    AgentAction,
    AgentRuntime,
    ProgressMonitor,
    StaticAgentClient,
    ToolCall,
    ToolResult,
    UserMessage,
)
from mahjong_agent_runtime.models import StateTransition


def test_progress_monitor_requests_replan_then_aborts_repeated_observation() -> None:
    monitor = ProgressMonitor(
        repeated_observation_limit=2,
        consecutive_no_progress_limit=2,
        max_replan_attempts=1,
    )
    action = tool_action("search_current_games", {"requirement": {"stake": "0.5"}})
    result = successful_result("search_current_games", {"matches": []})

    first = monitor.observe_action(action, [result], step_index=1)
    second = monitor.observe_action(action, [result], step_index=2)
    third = monitor.observe_action(action, [result], step_index=3)

    assert first.progress_made is True
    assert first.detected is False
    assert second.should_replan is True
    assert "repeated_observation" in second.detection_reasons
    assert third.should_abort is True


def test_progress_monitor_treats_different_validated_query_as_new_information() -> None:
    monitor = ProgressMonitor(max_replan_attempts=1)
    hangzhou = tool_action("search_current_games", {"requirement": {"game_type": "hangzhou_mahjong"}})
    sichuan = tool_action("search_current_games", {"requirement": {"game_type": "sichuan_mahjong"}})
    empty_result = successful_result("search_current_games", {"matches": []})

    monitor.observe_action(hangzhou, [empty_result], step_index=1)
    stalled = monitor.observe_action(hangzhou, [empty_result], step_index=2)
    replanned = monitor.observe_action(sichuan, [empty_result], step_index=3)

    assert stalled.should_replan is True
    assert replanned.progress_made is True
    assert replanned.detected is False


def test_progress_monitor_detects_short_action_cycle() -> None:
    monitor = ProgressMonitor(
        repeated_observation_limit=99,
        consecutive_no_progress_limit=99,
        max_replan_attempts=1,
        max_cycle_period=2,
    )
    action_a = tool_action("search_current_games", {"requirement": {"stake": "0.5"}})
    action_b = tool_action("search_customers", {"requirement": {"stake": "0.5"}})
    result_a = successful_result("search_current_games", {"matches": []})
    result_b = successful_result("search_customers", {"candidates": []})

    monitor.observe_action(action_a, [result_a], step_index=1)
    monitor.observe_action(action_b, [result_b], step_index=2)
    monitor.observe_action(action_a, [result_a], step_index=3)
    decision = monitor.observe_action(action_b, [result_b], step_index=4)

    assert decision.cycle_period == 2
    assert decision.should_replan is True
    assert "short_cycle" in decision.detection_reasons


def test_progress_monitor_does_not_treat_nonconsecutive_return_as_direct_repeat() -> None:
    monitor = ProgressMonitor(
        repeated_observation_limit=2,
        consecutive_no_progress_limit=99,
        max_cycle_period=1,
    )
    action_a = tool_action("search_current_games", {"requirement": {"stake": "0.5"}})
    action_b = tool_action("search_current_games", {"requirement": {"stake": "1"}})
    result = successful_result("search_current_games", {"matches": []})

    monitor.observe_action(action_a, [result], step_index=1)
    monitor.observe_action(action_b, [result], step_index=2)
    returned = monitor.observe_action(action_a, [result], step_index=3)

    assert returned.progress_made is False
    assert returned.repeated_observation_count == 1
    assert returned.detected is False


def test_progress_monitor_resets_stall_epoch_after_state_transition() -> None:
    monitor = ProgressMonitor(max_replan_attempts=1)
    read_action = tool_action("search_current_games", {"requirement": {"stake": "0.5"}})
    read_result = successful_result("search_current_games", {"matches": []})
    monitor.observe_action(read_action, [read_result], step_index=1)
    assert monitor.observe_action(read_action, [read_result], step_index=2).should_replan is True

    transition = StateTransition(
        entity_type="game",
        entity_id="game_1",
        from_status=None,
        to_status="forming",
        reason="created",
        trace_id="trace_progress_transition",
    )
    write_action = tool_action("create_game", {"requirement": {"stake": "0.5"}})
    write_result = successful_result("create_game", {"game_id": "game_1"}, transitions=[transition])
    write_decision = monitor.observe_action(write_action, [write_result], step_index=3)
    read_after_write = monitor.observe_action(read_action, [read_result], step_index=4)

    assert write_decision.progress_reasons == ["state_transition", "new_tool_result"]
    assert write_decision.detected is False
    assert read_after_write.progress_made is True
    assert read_after_write.detected is False


def test_progress_monitor_detects_repeated_invalid_contract_feedback() -> None:
    monitor = ProgressMonitor(max_replan_attempts=1)
    payload = {"errors": ["objective_status=needs_tool requires at least one tool_call"]}

    first = monitor.observe_runtime_feedback("action_contract_error", payload, step_index=1)
    second = monitor.observe_runtime_feedback("action_contract_error", payload, step_index=2)
    third = monitor.observe_runtime_feedback("action_contract_error", payload, step_index=3)

    assert first.detected is False
    assert second.should_replan is True
    assert third.should_abort is True


def test_runtime_feeds_progress_guard_back_to_model_and_aborts_repeated_loop() -> None:
    repeated_action = action_json(
        objective_status="needs_tool",
        tool_calls=[
            {
                "name": "search_current_games",
                "arguments": {"requirement": {"game_type": "hangzhou_mahjong"}, "limit": 1},
                "reason": "查询当前局。",
            }
        ],
    )
    client = StaticAgentClient(
        [
            repeated_action,
            repeated_action,
            repeated_action,
            action_json(objective_status="completed", reply_to_user="不应执行到这里。"),
        ]
    )
    runtime = AgentRuntime(
        llm_client=client,
        max_steps=8,
        repeated_observation_limit=2,
        consecutive_no_progress_limit=2,
        max_progress_replans=1,
    )

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="progress_loop_runtime",
            sender_id="zhang",
            sender_name="张哥",
            text="现在有局吗",
            message_id="msg_progress_loop_runtime",
        ),
        trace_id="trace_progress_loop_runtime",
    )

    assert result.final_reply == "这个我先转人工确认一下。"
    assert len(client.calls) == 3
    third_payload = json.loads(client.calls[2]["messages"][1]["content"])
    assert any(item["name"] == "agent_progress_guard" for item in third_payload["previous_tool_results"])

    events = runtime.trace_recorder.get_trace("trace_progress_loop_runtime")
    steps = [event.step for event in events]
    assert steps.count("agent_replan_requested") == 1
    assert steps.count("agent_loop_aborted") == 1
    assert "agent_loop_detected" in steps
    assert any(
        event.step == "final_output" and event.content.get("reason") == "agent_loop_no_progress"
        for event in events
    )


def tool_action(name: str, arguments: dict) -> AgentAction:
    return AgentAction(
        goal="测试进展监测",
        objective_status="needs_tool",
        reasoning_summary="执行测试工具。",
        tool_calls=[ToolCall(name=name, arguments=arguments, reason="测试。")],
        stop_reason={
            "can_stop": False,
            "why": "需要工具结果。",
            "pending_work": [name],
            "depends_on_tool_results": False,
        },
    )


def successful_result(
    name: str,
    result: dict,
    *,
    transitions: list[StateTransition] | None = None,
) -> ToolResult:
    return ToolResult(
        name=name,
        called=True,
        allowed=True,
        result=result,
        state_transitions=transitions or [],
    )


def action_json(
    *,
    objective_status: str,
    reply_to_user: str = "",
    tool_calls: list[dict] | None = None,
) -> str:
    calls = tool_calls or []
    return json.dumps(
        {
            "goal": "测试 Agent 无进展检测",
            "objective_status": objective_status,
            "reasoning_summary": "根据当前状态继续处理。",
            "objective_state": {
                "current_phase": "test",
                "known_facts": {},
                "missing_facts": [],
                "blockers": [],
            },
            "objective_plan": [
                {
                    "step_id": "step_1",
                    "title": "测试步骤",
                    "status": "in_progress" if objective_status == "needs_tool" else "done",
                    "tool": calls[0]["name"] if calls else None,
                    "depends_on": [],
                    "decision_rule": "根据工具结果决定下一步。",
                }
            ],
            "plan_revision_reason": "测试。",
            "reply_to_user": reply_to_user,
            "tool_calls": calls,
            "needs_human": False,
            "stop_reason": {
                "can_stop": objective_status != "needs_tool",
                "why": "工具完成后才能停止。" if calls else "已经得到最终回复。",
                "pending_work": [call["name"] for call in calls],
                "depends_on_tool_results": False,
            },
            "badcase": None,
        },
        ensure_ascii=False,
    )
