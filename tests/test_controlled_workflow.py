from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from mahjong_agent.context_builder import WorkflowContextBuilder
from mahjong_agent.controlled_workflow import ControlledWorkflowService
from mahjong_agent.core import AgentCore
from mahjong_agent.memory import InMemoryShortTermMemoryStore, ShortTermMemoryRecord
from mahjong_agent.models import ChannelType, CustomerProfile, Message, PlayPreference
from mahjong_agent.observability import InMemoryTraceRecorder, TraceStep, validate_controlled_trace_completeness
from mahjong_agent.reply_policy import ReplyPolicy
from mahjong_agent.semantic_resolver import SemanticResolver
from mahjong_agent.state_machine import InMemoryWorkflowStateStore
from mahjong_agent.workflow_models import ActionName, GameRequirement, GameWorkflowStatus, SlotSource, SlotValue, ToolName, UserMessage


TZ = ZoneInfo("Asia/Shanghai")
NOW = datetime(2026, 6, 30, 16, 0, tzinfo=TZ)


class FakeSemanticLLMClient:
    def __init__(self, output: str | dict[str, Any]) -> None:
        self.output = output
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        trace_id: str,
        timeout_seconds: float,
    ) -> str | dict[str, Any]:
        self.calls.append(
            {
                "messages": messages,
                "trace_id": trace_id,
                "timeout_seconds": timeout_seconds,
            }
        )
        return self.output


class FakeReplyLLMClient:
    def __init__(self, output: str | dict[str, Any]) -> None:
        self.output = output
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        trace_id: str,
        timeout_seconds: float,
    ) -> str | dict[str, Any]:
        self.calls.append(
            {
                "messages": messages,
                "trace_id": trace_id,
                "timeout_seconds": timeout_seconds,
            }
        )
        return self.output


def make_message(text: str = "人齐开吧，有烟无烟都行") -> Message:
    return Message(
        text=text,
        sender_id="zhang",
        sender_name="张哥",
        channel_id="boss_trial",
        channel_type=ChannelType.WEB_CONSOLE,
        sent_at=NOW,
        id="msg_controlled",
        metadata={"conversation_id": "boss_trial"},
    )


def seed_customers(core: AgentCore) -> None:
    core.upsert_customer(
        CustomerProfile(
            id="ran",
            display_name="冉姐",
            preferred_levels=["0.5"],
            smoke_free_preference=True,
            play_preferences=[
                PlayPreference(
                    game_type="hangzhou_mahjong",
                    preferred_levels=["0.5"],
                    preferred_variants=["caiqiao"],
                )
            ],
            usual_start_hours=[16, 17],
        )
    )
    core.upsert_customer(
        CustomerProfile(
            id="liu",
            display_name="刘姐",
            preferred_levels=["0.5"],
            smoke_free_preference=False,
            play_preferences=[PlayPreference(game_type="hangzhou_mahjong", preferred_levels=["0.5"])],
            usual_start_hours=[16, 18],
        )
    )


def complete_create_game_contract() -> dict[str, Any]:
    return {
        "intent": "find_players",
        "proposed_action": "create_game",
        "confidence": 0.91,
        "needs_human_review": False,
        "reasoning_summary": "用户确认要新组局，槽位来自当前消息、上下文和画像。",
        "slots": {
            "game_type": {
                "value": "hangzhou_mahjong",
                "source": "context",
                "confidence": 0.86,
                "confirmed": True,
                "needs_confirmation": False,
            },
            "stake": {
                "value": "0.5",
                "source": "context",
                "confidence": 0.84,
                "confirmed": True,
                "needs_confirmation": False,
            },
            "start_time_mode": {
                "value": "people_ready",
                "source": "explicit",
                "confidence": 0.92,
                "confirmed": True,
                "needs_confirmation": False,
            },
            "missing_count": {
                "value": 3,
                "source": "context",
                "confidence": 0.82,
                "confirmed": True,
                "needs_confirmation": False,
            },
            "smoke": {
                "value": "any",
                "source": "explicit",
                "confidence": 0.9,
                "confirmed": True,
                "needs_confirmation": False,
            },
            "duration_hours": {
                "value": 4,
                "source": "profile",
                "confidence": 0.78,
                "confirmed": True,
                "needs_confirmation": False,
            },
        },
    }


def create_game_contract_with_profile_observation() -> dict[str, Any]:
    output = complete_create_game_contract()
    output["profile_observations"] = [
        {
            "field": "smoke_preference",
            "value": "any",
            "confidence": 0.82,
            "source": "current_message",
            "evidence": "用户说有烟无烟都行",
            "risk": "low",
        }
    ]
    return output


def test_controlled_workflow_records_full_trace_and_queues_pending_invites() -> None:
    core = AgentCore()
    seed_customers(core)
    memory = InMemoryShortTermMemoryStore()
    trace = InMemoryTraceRecorder()
    state_store = InMemoryWorkflowStateStore()
    llm_client = FakeSemanticLLMClient(complete_create_game_contract())
    service = ControlledWorkflowService(
        core=core,
        context_builder=WorkflowContextBuilder(core, memory),
        semantic_resolver=SemanticResolver(llm_client),
        state_store=state_store,
        memory_store=memory,
        trace_recorder=trace,
    )

    result = service.handle_message(make_message(), now=NOW, trace_id="trace_controlled")

    assert result.run.semantic_resolution is not None
    assert result.run.semantic_resolution.proposed_action.name == ActionName.CREATE_GAME
    assert result.run.validated_action is not None
    assert result.run.validated_action.effective_action == ActionName.QUEUE_INVITES
    assert result.final_text == "好的，我帮你问问。"

    tool_names = [item.request.tool_name for item in result.tool_orchestration.tool_results]
    assert tool_names == [
        ToolName.SEARCH_CANDIDATE_CUSTOMERS,
        ToolName.CREATE_PENDING_OUTBOX,
        ToolName.CREATE_GAME,
    ]
    assert result.tool_orchestration.result_for(ToolName.CREATE_PENDING_OUTBOX).result["drafts"]
    create_game_result = result.tool_orchestration.result_for(ToolName.CREATE_GAME)
    assert create_game_result.called is True
    assert create_game_result.allowed is True
    assert create_game_result.result["state_write_intent"]["kind"] == "create_game"
    assert create_game_result.result["state_write_intent"]["target_status"] == GameWorkflowStatus.NEGOTIATING.value

    assert [transition.to_status for transition in result.run.state_transitions] == [
        GameWorkflowStatus.OPEN.value,
        GameWorkflowStatus.NEGOTIATING.value,
    ]
    assert all(transition.allowed for transition in result.run.state_transitions)
    game_id = result.run.state_transitions[-1].entity_id
    assert state_store.current_status("game", game_id) == GameWorkflowStatus.NEGOTIATING.value
    assert len(state_store.transition_history(entity_type="game", entity_id=game_id)) == 2
    assert result.run.state_transitions[-1].metadata["store_applied"] is True

    steps = [event.step for event in result.trace_events]
    assert TraceStep.USER_INPUT in steps
    assert TraceStep.CONTEXT_BUILT in steps
    assert TraceStep.LLM_PROMPT in steps
    assert TraceStep.LLM_RESPONSE in steps
    assert TraceStep.ACTION_PROPOSED in steps
    assert TraceStep.ACTION_VALIDATED in steps
    assert TraceStep.TOOL_CALLED in steps
    assert TraceStep.STATE_TRANSITION in steps
    assert TraceStep.REPLY_DRAFTED in steps
    assert TraceStep.REPLY_GUARDED in steps
    assert TraceStep.MEMORY_WRITTEN in steps
    assert TraceStep.FINAL_OUTPUT in steps
    completeness = validate_controlled_trace_completeness(result.trace_events)
    assert completeness.complete is True
    final_event = next(event for event in result.trace_events if event.step == TraceStep.FINAL_OUTPUT)
    assert final_event.content["trace_completeness"]["complete"] is True
    assert final_event.content["trace_completeness"]["missing_steps"] == []

    prompt_event = next(event for event in result.trace_events if event.step == TraceStep.LLM_PROMPT)
    assert "semantic_resolution_contract_v1" in prompt_event.content["messages"][1]["content"]
    assert llm_client.calls[0]["trace_id"] == "trace_controlled"

    memory_records = memory.load("boss_trial", "zhang", now=NOW)
    assert len(memory_records) == 1
    assert memory_records[0].system_reply == "好的，我帮你问问。"
    assert memory_records[0].game_requirement.slot("start_time_mode").value == "people_ready"


def test_controlled_workflow_applies_profile_update_after_semantic_observation() -> None:
    core = AgentCore()
    seed_customers(core)
    memory = InMemoryShortTermMemoryStore()
    trace = InMemoryTraceRecorder()
    service = ControlledWorkflowService(
        core=core,
        context_builder=WorkflowContextBuilder(core, memory),
        semantic_resolver=SemanticResolver(FakeSemanticLLMClient(create_game_contract_with_profile_observation())),
        memory_store=memory,
        trace_recorder=trace,
    )

    result = service.handle_message(
        make_message("人齐开吧，有烟无烟都行"),
        now=NOW,
        trace_id="trace_profile_update",
    )

    tool_names = [item.request.tool_name for item in result.tool_orchestration.tool_results]
    assert tool_names == [
        ToolName.SEARCH_CANDIDATE_CUSTOMERS,
        ToolName.CREATE_PENDING_OUTBOX,
        ToolName.CREATE_GAME,
        ToolName.PROFILE_UPDATE,
    ]
    profile_update = result.tool_orchestration.result_for(ToolName.PROFILE_UPDATE)
    assert profile_update is not None
    assert profile_update.called is True
    assert profile_update.allowed is True
    assert profile_update.result["applied_count"] == 1
    observations = core.store.customers["zhang"].metadata["controlled_profile_observations"]
    assert observations[0]["field"] == "smoke_preference"
    assert observations[0]["evidence"] == "用户说有烟无烟都行"


def test_controlled_workflow_prompt_and_trace_include_structured_followup_context() -> None:
    core = AgentCore()
    seed_customers(core)
    memory = InMemoryShortTermMemoryStore()
    previous_requirement = GameRequirement()
    previous_requirement.set_slot(
        SlotValue(
            name="stake",
            value="0.5",
            source=SlotSource.EXPLICIT,
            confidence=0.9,
            confirmed=True,
            needs_confirmation=False,
        )
    )
    memory.append(
        ShortTermMemoryRecord(
            conversation_id="boss_trial",
            sender_id="zhang",
            user_message=UserMessage(
                text="老板，今天下班有人打麻将吗？0.5或者1都行，烟也都可",
                sender_id="zhang",
                sender_name="张哥",
                conversation_id="boss_trial",
                trace_id="trace_prev",
                message_id="msg_prev",
            ),
            system_reply="可以，我先确认下：大概几点能到？你这边几个人？",
            game_requirement=previous_requirement,
            created_at=NOW,
        ),
        now=NOW,
    )
    trace = InMemoryTraceRecorder()
    llm_client = FakeSemanticLLMClient(complete_create_game_contract())
    service = ControlledWorkflowService(
        core=core,
        context_builder=WorkflowContextBuilder(core, memory),
        semantic_resolver=SemanticResolver(llm_client),
        memory_store=memory,
        trace_recorder=trace,
    )

    result = service.handle_message(
        make_message("六点，我这边两个人"),
        now=NOW,
        trace_id="trace_followup",
    )

    context_event = next(event for event in result.trace_events if event.step == TraceStep.CONTEXT_BUILT)
    followup = context_event.content["followup_context"]
    assert followup["schema_version"] == "followup_context.v1"
    assert followup["unresolved_questions"] == ["start_time", "party_size"]
    assert followup["current_message_response_type"] == "slot_fill"
    assert followup["should_treat_current_message_as_followup"] is True

    prompt_event = next(event for event in result.trace_events if event.step == TraceStep.LLM_PROMPT)
    prompt_text = prompt_event.content["messages"][1]["content"]
    assert '"schema_version": "followup_context.v1"' in prompt_text
    assert '"current_message_response_type": "slot_fill"' in prompt_text
    assert '"previous_game_requirement"' in prompt_text


def test_controlled_workflow_trace_line_uses_required_format() -> None:
    core = AgentCore()
    seed_customers(core)
    service = ControlledWorkflowService(
        core=core,
        context_builder=WorkflowContextBuilder(core),
        semantic_resolver=SemanticResolver(FakeSemanticLLMClient(complete_create_game_contract())),
    )

    result = service.handle_message(make_message(), now=NOW, trace_id="trace_format")

    line = result.trace_events[0].format_log_line()
    assert line.startswith("trace_format-2026-06-30 16:00:00-INFO: ")
    assert '"text": "人齐开吧，有烟无烟都行"' in line


def test_controlled_workflow_traces_rejected_reply_llm_contract() -> None:
    core = AgentCore()
    seed_customers(core)
    reply_llm = FakeReplyLLMClient('建议如下：{"text":"好的，我帮你问问。"}')
    service = ControlledWorkflowService(
        core=core,
        context_builder=WorkflowContextBuilder(core),
        semantic_resolver=SemanticResolver(FakeSemanticLLMClient(complete_create_game_contract())),
        reply_policy=ReplyPolicy(reply_llm),
    )

    result = service.handle_message(make_message(), now=NOW, trace_id="trace_reply_contract")

    assert result.final_text == "好的，我帮你问问。"
    reply_event = next(event for event in result.trace_events if event.step == TraceStep.REPLY_DRAFTED)
    assert reply_event.level == "WARN"
    contract = reply_event.content["reply_draft"]["metadata"]["llm_contract"]
    assert contract["accepted"] is False
    assert contract["strict_json"] is True
    assert contract["raw_output"].startswith("建议如下")
    assert "single JSON object" in contract["parse_error"]
    assert reply_llm.calls[0]["trace_id"] == "trace_reply_contract"


def test_controlled_workflow_traces_rejected_semantic_llm_contract() -> None:
    core = AgentCore()
    seed_customers(core)
    service = ControlledWorkflowService(
        core=core,
        context_builder=WorkflowContextBuilder(core),
        semantic_resolver=SemanticResolver(
            FakeSemanticLLMClient('解析如下：{"intent":"find_players","proposed_action":"create_game"}')
        ),
    )

    result = service.handle_message(make_message(), now=NOW, trace_id="trace_semantic_contract")

    assert result.run.semantic_resolution is not None
    assert result.run.semantic_resolution.needs_human_review is True
    assert result.run.validated_action is not None
    assert result.run.validated_action.effective_action == ActionName.HUMAN_REVIEW
    assert result.final_text == "这个我先转人工确认一下。"
    llm_event = next(event for event in result.trace_events if event.step == TraceStep.LLM_RESPONSE)
    assert llm_event.level == "WARN"
    contract = llm_event.content["raw_response"]["llm_contract"]
    assert contract["accepted"] is False
    assert contract["strict_json"] is True
    assert contract["raw_output"].startswith("解析如下")
    assert "single JSON object" in contract["parse_error"]
