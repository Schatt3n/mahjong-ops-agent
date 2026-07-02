from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from mahjong_agent.core import AgentCore
from mahjong_agent.controlled_runtime import ControlledRuntimeConfig, build_controlled_runtime
from mahjong_agent.models import ChannelType, CustomerProfile, Message, PlayPreference
from mahjong_agent.workflow_models import ActionName, GameWorkflowStatus, ToolName
from mahjong_agent.autonomous_agent import DEFAULT_AGENT_PROMPT_PATH


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


def test_autonomous_agent_forces_current_game_search_before_availability_reply(tmp_path) -> None:
    client = SequenceAgentClient(
        [
            {
                "decision": "wait_user",
                "goal_status": "waiting_user",
                "intent": "inquire_existing_game",
                "reasoning_summary": "用户问通宵有没有人，但未提供档位和人数。",
                "requirement": {"slots": {}},
                "reply_text": "通宵有的，你几个人？打什么档位？",
            },
            {
                "decision": "final_reply",
                "goal_status": "completed",
                "intent": "inquire_existing_game",
                "reasoning_summary": "当前局池工具返回没有匹配局。",
                "requirement": {"slots": {}},
                "reply_text": "现在没有通宵局，要组一个吗？",
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

    result = runtime.service.handle_message(message("通宵有人吗"), now=NOW, trace_id="trace_auto_evidence")

    assert result.final_text == "现在没有通宵局，要组一个吗？"
    assert result.tool_orchestration.result_for(ToolName.SEARCH_CURRENT_OPEN_GAMES) is not None
    first_step = result.run.reply_draft.metadata["agent_steps"][0]
    assert first_step["decision"] == "tool_call"
    assert first_step["tool_name"] == ToolName.SEARCH_CURRENT_OPEN_GAMES.value
    assert len(client.calls) == 2
    second_payload = client.calls[1]["messages"][1]["content"]
    assert "visibility_contract" in second_payload
    assert '"has_matches": false' in second_payload


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


def test_autonomous_agent_executes_batch_tool_plan_without_extra_llm(tmp_path) -> None:
    requirement = complete_requirement_payload()
    client = SequenceAgentClient(
        [
            {
                "decision": "tool_call",
                "goal_status": "in_progress",
                "intent": "find_players",
                "reasoning_summary": "信息足够，创建局、搜索候选人并生成待审批邀约草稿。",
                "requirement": requirement,
                "tool_calls": [
                    {"tool_name": "create_game", "arguments": {}},
                    {"tool_name": "search_candidate_customers", "arguments": {}},
                    {"tool_name": "create_pending_outbox", "arguments": {}},
                ],
                "reply_text": "好的，我帮你问问。",
            }
        ]
    )
    runtime = build_controlled_runtime(
        core=core_with_customers(8),
        llm_client=client,
        config=ControlledRuntimeConfig(
            trace_jsonl_path=tmp_path / "trace.jsonl",
            autonomous_agent_enabled=True,
        ),
    )

    result = runtime.service.handle_message(
        message("一个人，1块的"),
        now=NOW,
        trace_id="trace_auto_batch_tools",
    )

    assert result.final_text == "好的，我帮你问问。"
    assert result.tool_orchestration.result_for(ToolName.CREATE_GAME) is not None
    assert result.tool_orchestration.result_for(ToolName.SEARCH_CANDIDATE_CUSTOMERS) is not None
    assert result.tool_orchestration.result_for(ToolName.CREATE_PENDING_OUTBOX) is not None
    assert result.run.validated_action.effective_action == ActionName.QUEUE_INVITES
    assert result.run.reply_draft.metadata["agent_steps"][0]["tool_calls"] == [
        {"tool_name": "create_game", "arguments": {}},
        {"tool_name": "search_candidate_customers", "arguments": {}},
        {"tool_name": "create_pending_outbox", "arguments": {}},
    ]
    assert len(client.calls) == 1


def test_autonomous_agent_does_not_create_outbox_when_smoke_is_unconfirmed(tmp_path) -> None:
    requirement = incomplete_smoke_requirement_payload()
    client = SequenceAgentClient(
        [
            {
                "decision": "tool_call",
                "goal_status": "in_progress",
                "intent": "find_players",
                "reasoning_summary": "用户补充了人数和钱数，尝试找人。",
                "requirement": requirement,
                "tool_calls": [
                    {"tool_name": "create_game", "arguments": {}},
                    {"tool_name": "search_candidate_customers", "arguments": {}},
                    {"tool_name": "create_pending_outbox", "arguments": {}},
                ],
                "reply_text": "好的，我帮你问问。",
            },
            {
                "decision": "wait_user",
                "goal_status": "waiting_user",
                "intent": "find_players",
                "reasoning_summary": "烟况还没确认，先问用户。",
                "requirement": requirement,
                "reply_text": "有烟无烟都行吗？",
            },
        ]
    )
    runtime = build_controlled_runtime(
        core=core_with_customers(8),
        llm_client=client,
        config=ControlledRuntimeConfig(
            trace_jsonl_path=tmp_path / "trace.jsonl",
            autonomous_agent_enabled=True,
        ),
    )

    result = runtime.service.handle_message(
        message("一个人，1块的"),
        now=NOW,
        trace_id="trace_auto_missing_smoke",
    )

    outbox_result = result.tool_orchestration.result_for(ToolName.CREATE_PENDING_OUTBOX)
    assert outbox_result is not None
    assert outbox_result.called is False
    assert outbox_result.allowed is False
    assert "smoke" in outbox_result.error
    assert result.final_text == "有烟无烟都行吗？"
    assert len(client.calls) == 2


def test_autonomous_agent_prompt_keeps_tool_arguments_compact() -> None:
    prompt = DEFAULT_AGENT_PROMPT_PATH.read_text(encoding="utf-8")

    assert "禁止在 `tool_call.arguments` 或 `tool_calls[].arguments` 里重复" in prompt
    assert "大多数工具调用的 `arguments` 应该是 `{}`" in prompt
    assert "输出要尽量小" in prompt


def test_autonomous_agent_does_not_feed_private_outbox_counts_to_model(tmp_path) -> None:
    requirement = complete_requirement_payload()
    client = SequenceAgentClient(
        [
            {
                "decision": "tool_call",
                "goal_status": "in_progress",
                "intent": "find_players",
                "reasoning_summary": "信息足够，先创建局。",
                "requirement": requirement,
                "tool_call": {"tool_name": "create_game", "arguments": {}},
            },
            {
                "decision": "tool_call",
                "goal_status": "in_progress",
                "intent": "find_players",
                "reasoning_summary": "局已创建，搜索候选人。",
                "requirement": requirement,
                "tool_call": {"tool_name": "search_candidate_customers", "arguments": {}},
            },
            {
                "decision": "tool_call",
                "goal_status": "in_progress",
                "intent": "find_players",
                "reasoning_summary": "候选人已找到，创建待审批邀约草稿。",
                "requirement": requirement,
                "tool_call": {"tool_name": "create_pending_outbox", "arguments": {}},
            },
            {
                "decision": "final_reply",
                "goal_status": "completed",
                "intent": "find_players",
                "reasoning_summary": "已生成待审批邀约草稿。",
                "requirement": requirement,
                "reply_text": "好的，按这个要求帮你问了，有消息跟你说。",
            },
        ]
    )
    runtime = build_controlled_runtime(
        core=core_with_customers(8),
        llm_client=client,
        config=ControlledRuntimeConfig(
            trace_jsonl_path=tmp_path / "trace.jsonl",
            autonomous_agent_enabled=True,
        ),
    )

    result = runtime.service.handle_message(
        message("杭麻，1块，无烟也可以，人齐开，我一个人"),
        now=NOW,
        trace_id="trace_auto_visibility",
    )

    assert result.tool_orchestration.result_for(ToolName.CREATE_PENDING_OUTBOX) is not None
    assert result.final_text == "好的，按这个要求帮你问了，有消息跟你说。"
    fourth_payload = client.calls[3]["messages"][1]["content"]
    assert "visibility_contract" in fourth_payload
    assert "customer_visible_facts" in fourth_payload
    assert "已按要求帮忙问了" in fourth_payload
    assert "draft_count" not in fourth_payload
    assert "outbox_count" not in fourth_payload
    assert "stored_count" not in fourth_payload
    assert "result_count" not in fourth_payload
    assert "候选0" not in fourth_payload


def test_autonomous_agent_tool_result_prompt_marks_internal_counts(tmp_path) -> None:
    requirement = complete_requirement_payload()
    client = SequenceAgentClient(
        [
            {
                "decision": "tool_call",
                "goal_status": "in_progress",
                "intent": "find_players",
                "reasoning_summary": "搜索候选人。",
                "requirement": requirement,
                "tool_call": {"tool_name": "search_candidate_customers", "arguments": {}},
            },
            {
                "decision": "final_reply",
                "goal_status": "completed",
                "intent": "find_players",
                "reasoning_summary": "测试工具结果摘要。",
                "requirement": requirement,
                "reply_text": "我先确认一下。",
            },
        ]
    )
    runtime = build_controlled_runtime(
        core=core_with_customers(2),
        llm_client=client,
        config=ControlledRuntimeConfig(
            trace_jsonl_path=tmp_path / "trace.jsonl",
            autonomous_agent_enabled=True,
        ),
    )

    runtime.service.handle_message(message("杭麻1块无烟，人齐开，我一个人"), now=NOW, trace_id="trace_auto_summary")

    second_payload = client.calls[1]["messages"][1]["content"]
    assert "visibility_contract" in second_payload
    assert "private_facts_not_for_customer" in second_payload
    assert "已找到可邀约候选人" in second_payload
    assert "candidate_count" not in second_payload
    assert "result_summary" not in second_payload
    assert "候选0" not in second_payload
    assert '"result":' not in second_payload


def complete_requirement_payload():
    return {
        "slots": {
            "game_type": slot("hangzhou_mahjong"),
            "stake": slot("1"),
            "smoke": slot("no_smoke"),
            "start_time_mode": slot("asap_when_full"),
            "duration_mode": slot("normal"),
            "party_size": slot(1),
        }
    }


def incomplete_smoke_requirement_payload():
    payload = complete_requirement_payload()
    payload["slots"]["start_time_mode"] = slot("asap_when_full")
    payload["slots"]["duration_mode"] = slot("overnight")
    payload["slots"]["smoke"] = {
        "value": "any",
        "source": "inferred",
        "confidence": 0.5,
        "confirmed": False,
        "needs_confirmation": False,
        "evidence": "未提及烟况",
    }
    return payload


def core_with_customers(count: int) -> AgentCore:
    core = AgentCore()
    for index in range(count):
        core.upsert_customer(
            CustomerProfile(
                id=f"candidate_{index}",
                display_name=f"候选{index}",
                preferred_levels=["1"],
                smoke_free_preference=True,
                play_preferences=[
                    PlayPreference(
                        game_type="hangzhou_mahjong",
                        preferred_levels=["1"],
                        preferred_variants=["caiqiao"],
                    )
                ],
                usual_start_hours=[16, 17, 18],
            )
        )
    return core


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
