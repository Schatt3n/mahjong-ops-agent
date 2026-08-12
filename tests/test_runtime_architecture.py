from __future__ import annotations

import json
from pathlib import Path

from mahjong_agent_runtime import (
    ActionProcessor,
    AgentLoop,
    AgentRuntime,
    ContextLifecycleManager,
    HookEvent,
    HookManager,
    StaticAgentClient,
    ToolExecutionService,
    UserMessage,
)


ROOT = Path(__file__).resolve().parents[1]


def test_runtime_no_longer_depends_on_private_compatibility_mixin() -> None:
    runtime_source = (ROOT / "src" / "mahjong_agent_runtime" / "runtime.py").read_text(encoding="utf-8")

    assert "runtime_compat" not in runtime_source
    assert AgentRuntime.__bases__ == (object,)
    assert not (ROOT / "src" / "mahjong_agent_runtime" / "runtime_compat.py").exists()


def test_runtime_is_composed_from_thin_loop_services_and_emits_hooks() -> None:
    observed: list[HookEvent] = []
    hooks = HookManager()
    for name in [
        "message_received",
        "before_agent_loop",
        "after_context_built",
        "before_llm_call",
        "after_llm_response",
        "after_action_proposed",
        "before_tool_execute",
        "after_tool_execute",
        "before_reply_send",
        "after_agent_loop",
        "after_turn_finished",
    ]:
        hooks.register(name, observed.append)

    runtime = AgentRuntime(
        llm_client=StaticAgentClient(
            [
                action_json(
                    objective_status="needs_tool",
                    tool_calls=[
                        {
                            "name": "search_current_games",
                            "arguments": {"requirement": {"game_type": "hangzhou_mahjong"}, "limit": 1},
                            "reason": "先查询现有局。",
                        }
                    ],
                ),
                action_json(objective_status="completed", reply_to_user="现在没有诶。"),
            ]
        ),
        hook_manager=hooks,
    )

    assert isinstance(runtime.agent_loop, AgentLoop)
    assert isinstance(runtime.action_processor, ActionProcessor)
    assert isinstance(runtime.tool_execution_service, ToolExecutionService)
    assert isinstance(runtime.context_lifecycle, ContextLifecycleManager)

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="hook_architecture",
            sender_id="zhang",
            sender_name="张哥",
            text="现在有局吗",
            message_id="msg_hook_architecture",
        ),
        trace_id="trace_hook_architecture",
    )

    observed_names = [event.name for event in observed]
    assert result.final_reply == "现在没有诶。"
    assert observed_names.index("message_received") < observed_names.index("before_agent_loop")
    assert observed_names.index("before_agent_loop") < observed_names.index("after_agent_loop")
    assert observed_names.index("after_context_built") < observed_names.index("before_llm_call")
    assert "after_llm_response" in observed_names
    assert "after_action_proposed" in observed_names
    assert "before_tool_execute" in observed_names
    assert "after_tool_execute" in observed_names
    assert "before_reply_send" in observed_names
    assert observed_names[-1] == "after_turn_finished"


def action_json(
    *,
    objective_status: str,
    reply_to_user: str = "",
    tool_calls: list[dict] | None = None,
) -> str:
    return json.dumps(
        {
            "goal": "测试组件化主链路",
            "objective_status": objective_status,
            "reasoning_summary": "测试。",
            "objective_state": {"current_phase": "test", "known_facts": {}, "missing_facts": [], "blockers": []},
            "objective_plan": [
                {
                    "step_id": "step_1",
                    "title": "测试步骤",
                    "status": "in_progress" if objective_status == "needs_tool" else "done",
                    "tool": (tool_calls or [{}])[0].get("name") if tool_calls else None,
                    "depends_on": [],
                    "decision_rule": "测试规则。",
                }
            ],
            "plan_revision_reason": "测试。",
            "reply_to_user": reply_to_user,
            "tool_calls": tool_calls or [],
            "needs_human": False,
            "stop_reason": {
                "can_stop": objective_status != "needs_tool",
                "why": "测试停止条件。",
                "pending_work": [call.get("name", "tool") for call in tool_calls or []],
                "depends_on_tool_results": False,
            },
            "badcase": None,
        },
        ensure_ascii=False,
    )
