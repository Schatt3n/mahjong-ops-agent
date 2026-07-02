from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from mahjong_agent.controlled_runtime import ControlledRuntimeConfig, build_controlled_runtime
from mahjong_agent.models import ChannelType, Message
from mahjong_agent.workflow_models import ActionName, GameWorkflowStatus, ToolName


TZ = ZoneInfo("Asia/Shanghai")
NOW = datetime(2026, 7, 2, 16, 0, tzinfo=TZ)


class SequenceAgentClient:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def complete(self, messages, *, trace_id, timeout_seconds):
        self.calls.append({"messages": messages, "trace_id": trace_id, "timeout_seconds": timeout_seconds})
        if not self.outputs:
            raise AssertionError("No fake agent output left")
        return self.outputs.pop(0)


def test_autonomous_agent_can_wait_for_user_without_semantic_workflow(tmp_path) -> None:
    client = SequenceAgentClient(
        [
            {
                "decision": "wait_user",
                "goal_status": "waiting_user",
                "intent": "find_players",
                "reasoning_summary": "用户要组局，但还缺人数。",
                "requirement": {
                    "slots": {
                        "game_type": slot("hangzhou_mahjong"),
                        "stake": slot("1"),
                        "smoke": slot("no_smoke"),
                        "start_time_mode": slot("asap_when_full"),
                    }
                },
                "reply_text": "可以，人齐开。你这边几个人？",
            }
        ]
    )
    runtime = build_controlled_runtime(
        llm_client=client,
        config=ControlledRuntimeConfig(
            trace_jsonl_path=tmp_path / "trace.jsonl",
            autonomous_agent_enabled=True,
        ),
    )

    result = runtime.service.handle_message(message("杭麻，1块，无烟的，人齐开"), now=NOW, trace_id="trace_auto_wait")

    assert result.final_text == "可以，人齐开。你这边几个人？"
    assert result.run.semantic_resolution.raw_response["runtime"] == "autonomous_agent.v1"
    assert result.run.validated_action.effective_action == ActionName.ASK_CLARIFICATION
    assert len(client.calls) == 1


def test_autonomous_agent_calls_tool_then_replies(tmp_path) -> None:
    client = SequenceAgentClient(
        [
            {
                "decision": "tool_call",
                "goal_status": "in_progress",
                "intent": "inquire_existing_game",
                "reasoning_summary": "用户询问是否有现成局，先查当前局池。",
                "requirement": {"slots": {"stake": slot("0.5"), "game_type": slot("hangzhou_mahjong")}},
                "tool_call": {"tool_name": "search_current_open_games", "arguments": {}},
                "reply_text": "",
            },
            {
                "decision": "final_reply",
                "goal_status": "completed",
                "intent": "inquire_existing_game",
                "reasoning_summary": "当前没有匹配局，询问是否要新组。",
                "requirement": {"slots": {"stake": slot("0.5"), "game_type": slot("hangzhou_mahjong")}},
                "reply_text": "现在没有0.5的局，要组一个吗？",
            },
        ]
    )
    runtime = build_controlled_runtime(
        llm_client=client,
        config=ControlledRuntimeConfig(
            trace_jsonl_path=tmp_path / "trace.jsonl",
            autonomous_agent_enabled=True,
        ),
    )

    result = runtime.service.handle_message(message("通宵0.5有人吗"), now=NOW, trace_id="trace_auto_search")

    assert result.final_text == "现在没有0.5的局，要组一个吗？"
    assert result.tool_orchestration.result_for(ToolName.SEARCH_CURRENT_OPEN_GAMES) is not None
    assert result.run.validated_action.effective_action == ActionName.SEARCH_EXISTING_GAMES
    assert len(client.calls) == 2


def test_autonomous_agent_create_game_tool_goes_through_state_machine(tmp_path) -> None:
    requirement = {
        "slots": {
            "game_type": slot("hangzhou_mahjong"),
            "stake": slot("1"),
            "smoke": slot("no_smoke"),
            "start_time_mode": slot("asap_when_full"),
            "duration_mode": slot("overnight"),
            "party_size": slot(1),
        }
    }
    client = SequenceAgentClient(
        [
            {
                "decision": "tool_call",
                "goal_status": "in_progress",
                "intent": "find_players",
                "reasoning_summary": "信息足够，创建待组局。",
                "requirement": requirement,
                "tool_call": {"tool_name": "create_game", "arguments": {}},
                "reply_text": "",
            },
            {
                "decision": "final_reply",
                "goal_status": "in_progress",
                "intent": "find_players",
                "reasoning_summary": "已经创建局，下一步可继续找候选人。",
                "requirement": requirement,
                "reply_text": "好的，我先建这个局。",
            },
        ]
    )
    runtime = build_controlled_runtime(
        llm_client=client,
        config=ControlledRuntimeConfig(
            trace_jsonl_path=tmp_path / "trace.jsonl",
            autonomous_agent_enabled=True,
        ),
    )

    result = runtime.service.handle_message(message("我一个人，杭麻1块无烟，人齐开通宵"), now=NOW, trace_id="trace_auto_create")

    assert result.run.validated_action.effective_action == ActionName.QUEUE_INVITES
    assert result.tool_orchestration.result_for(ToolName.CREATE_GAME) is not None
    assert result.run.state_transitions
    assert result.run.state_transitions[-1].to_status == GameWorkflowStatus.OPEN.value


def slot(value):
    return {
        "value": value,
        "source": "explicit",
        "confidence": 0.92,
        "confirmed": True,
        "needs_confirmation": False,
        "evidence": "测试消息",
    }


def message(text: str) -> Message:
    return Message(
        text=text,
        sender_id="zhang",
        sender_name="张哥",
        channel_id="boss_trial",
        channel_type=ChannelType.WEB_CONSOLE,
        sent_at=NOW,
        metadata={"conversation_id": "boss_trial"},
    )
