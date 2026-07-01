from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from mahjong_agent.context_builder import WorkflowContextBuilder
from mahjong_agent.controlled_workflow import ControlledWorkflowService
from mahjong_agent.core import AgentCore
from mahjong_agent.input_gate import InMemoryInputGate
from mahjong_agent.memory import InMemoryShortTermMemoryStore, ShortTermMemoryRecord
from mahjong_agent.models import ChannelType, CustomerProfile, Message, PlayPreference
from mahjong_agent.observability import InMemoryTraceRecorder, TraceStep, validate_controlled_trace_completeness
from mahjong_agent.reply_policy import ReplyPolicy
from mahjong_agent.semantic_resolver import SemanticResolver
from mahjong_agent.state_machine import InMemoryWorkflowStateStore, StateMachine
from mahjong_agent.tool_orchestrator import ToolOrchestrationResult
from mahjong_agent.workflow_models import (
    ActionName,
    GameRequirement,
    GameWorkflowStatus,
    RiskLevel,
    SlotSource,
    SlotValue,
    ToolCallRequest,
    ToolExecutionMode,
    ToolName,
    ToolResult,
    UserMessage,
)


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


class FailingThenOkContextBuilder:
    def __init__(self, delegate: WorkflowContextBuilder) -> None:
        self.delegate = delegate
        self.calls = 0

    def build(
        self,
        message: Message,
        *,
        now: datetime | None = None,
        trace_id: str | None = None,
    ):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("temporary context failure")
        return self.delegate.build(message, now=now, trace_id=trace_id)


class OpenOnlyCreateGameToolOrchestrator:
    def run(self, *, context, semantic_resolution, validated_action, now=None) -> ToolOrchestrationResult:
        outbox_request = ToolCallRequest(
            tool_name=ToolName.CREATE_PENDING_OUTBOX,
            execution_mode=ToolExecutionMode.CREATE_PENDING,
            idempotency_key=f"{validated_action.idempotency_key}:create_pending_outbox",
            reason="fake outbox created",
        )
        create_request = ToolCallRequest(
            tool_name=ToolName.CREATE_GAME,
            execution_mode=ToolExecutionMode.STATE_WRITE,
            idempotency_key=f"{validated_action.idempotency_key}:create_game",
            reason="fake create game",
            risk_level=RiskLevel.MEDIUM,
        )
        return ToolOrchestrationResult(
            tool_results=[
                ToolResult(
                    request=outbox_request,
                    called=True,
                    allowed=True,
                    result={"drafts": [{"id": "draft_fake"}], "result_count": 1},
                ),
                ToolResult(
                    request=create_request,
                    called=True,
                    allowed=True,
                    result={
                        "state_write_intent": {
                            "kind": "create_game",
                            "entity_type": "game",
                            "entity_id": "game_open_only",
                            "target_status": GameWorkflowStatus.OPEN.value,
                            "reason": "工具意图只要求创建 open 局",
                            "requirement": semantic_resolution.game_requirement.to_prompt_dict(),
                        },
                        "game_id": "game_open_only",
                    },
                ),
            ]
        )


class ExpireCloseGameToolOrchestrator:
    def run(self, *, context, semantic_resolution, validated_action, now=None) -> ToolOrchestrationResult:
        close_request = ToolCallRequest(
            tool_name=ToolName.CLOSE_GAME,
            execution_mode=ToolExecutionMode.STATE_WRITE,
            idempotency_key=f"{validated_action.idempotency_key}:close_game",
            reason="fake close game",
            risk_level=RiskLevel.MEDIUM,
        )
        return ToolOrchestrationResult(
            tool_results=[
                ToolResult(
                    request=close_request,
                    called=True,
                    allowed=True,
                    result={
                        "state_write_intent": {
                            "kind": "close_game",
                            "entity_type": "game",
                            "entity_id": "game_expire_only",
                            "target_status": GameWorkflowStatus.EXPIRED.value,
                            "reason": "工具意图要求归档为超时",
                            "requirement": semantic_resolution.game_requirement.to_prompt_dict(),
                        },
                        "game_id": "game_expire_only",
                    },
                )
            ]
        )


class MissingStateIntentToolOrchestrator:
    def run(self, *, context, semantic_resolution, validated_action, now=None) -> ToolOrchestrationResult:
        create_request = ToolCallRequest(
            tool_name=ToolName.CREATE_GAME,
            execution_mode=ToolExecutionMode.STATE_WRITE,
            idempotency_key=f"{validated_action.idempotency_key}:create_game",
            reason="fake create game without intent",
            risk_level=RiskLevel.MEDIUM,
        )
        return ToolOrchestrationResult(
            tool_results=[
                ToolResult(
                    request=create_request,
                    called=True,
                    allowed=True,
                    result={"game_id": "game_without_intent"},
                )
            ]
        )


class InvalidStateIntentToolOrchestrator:
    def run(self, *, context, semantic_resolution, validated_action, now=None) -> ToolOrchestrationResult:
        create_request = ToolCallRequest(
            tool_name=ToolName.CREATE_GAME,
            execution_mode=ToolExecutionMode.STATE_WRITE,
            idempotency_key=f"{validated_action.idempotency_key}:create_game",
            reason="fake create game with invalid intent",
            risk_level=RiskLevel.MEDIUM,
        )
        return ToolOrchestrationResult(
            tool_results=[
                ToolResult(
                    request=create_request,
                    called=True,
                    allowed=True,
                    result={
                        "state_write_intent": {
                            "kind": "create_game",
                            "entity_type": "game",
                            "entity_id": "game_invalid_state",
                            "target_status": GameWorkflowStatus.CANCELLED.value,
                            "reason": "非法工具意图：新建局不能直接取消",
                            "requirement": semantic_resolution.game_requirement.to_prompt_dict(),
                            "target_status_reason": "unexpected extra key",
                        },
                        "game_id": "game_invalid_state",
                    },
                )
            ]
        )


def make_message(
    text: str = "人齐开吧，有烟无烟都行",
    *,
    message_id: str = "msg_controlled",
    sender_id: str = "zhang",
    sender_name: str = "张哥",
    metadata: dict[str, Any] | None = None,
) -> Message:
    message_metadata = {"conversation_id": "boss_trial"}
    if metadata:
        message_metadata.update(metadata)
    return Message(
        text=text,
        sender_id=sender_id,
        sender_name=sender_name,
        channel_id="boss_trial",
        channel_type=ChannelType.WEB_CONSOLE,
        sent_at=NOW,
        id=message_id,
        metadata=message_metadata,
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


def candidate_accept_contract(game_id: str) -> dict[str, Any]:
    return {
        "intent": "candidate_reply",
        "proposed_action": "accept_seat",
        "confidence": 0.93,
        "needs_human_review": False,
        "reasoning_summary": "候选人明确回复可以来，属于确认加入当前局。",
        "action_arguments": {"game_id": game_id},
        "slots": {},
    }


def cancel_game_contract() -> dict[str, Any]:
    return {
        "intent": "cancel_game",
        "proposed_action": "cancel_game",
        "confidence": 0.9,
        "needs_human_review": False,
        "reasoning_summary": "用户表示这桌不打了，需要关闭相关局。",
        "slots": {},
    }


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
        ToolName.SEARCH_CURRENT_OPEN_GAMES,
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
    assert TraceStep.REPLY_APPROVAL in steps
    assert TraceStep.MEMORY_WRITTEN in steps
    assert TraceStep.FINAL_OUTPUT in steps
    completeness = validate_controlled_trace_completeness(result.trace_events)
    assert completeness.complete is True
    final_event = next(event for event in result.trace_events if event.step == TraceStep.FINAL_OUTPUT)
    assert final_event.content["trace_completeness"]["complete"] is True
    assert final_event.content["trace_completeness"]["missing_steps"] == []
    assert final_event.content["reply_approval"]["queued"] is False
    assert final_event.content["reply_approval"]["reason"] == "reply_approval_queue_not_configured"
    state_event = next(event for event in result.trace_events if event.step == TraceStep.STATE_TRANSITION)
    assert state_event.content["rejected_state_write_intents"] == []

    prompt_event = next(event for event in result.trace_events if event.step == TraceStep.LLM_PROMPT)
    assert "semantic_resolution_contract_v1" in prompt_event.content["messages"][1]["content"]
    assert llm_client.calls[0]["trace_id"] == "trace_controlled"

    memory_records = memory.load("boss_trial", "zhang", now=NOW)
    assert len(memory_records) == 1
    assert memory_records[0].system_reply == "好的，我帮你问问。"
    assert memory_records[0].game_requirement.slot("start_time_mode").value == "people_ready"


def test_controlled_workflow_uses_tool_state_write_intent_target_status() -> None:
    core = AgentCore()
    seed_customers(core)
    memory = InMemoryShortTermMemoryStore()
    state_store = InMemoryWorkflowStateStore()
    service = ControlledWorkflowService(
        core=core,
        context_builder=WorkflowContextBuilder(core, memory),
        semantic_resolver=SemanticResolver(FakeSemanticLLMClient(complete_create_game_contract())),
        tool_orchestrator=OpenOnlyCreateGameToolOrchestrator(),
        state_store=state_store,
        memory_store=memory,
    )

    result = service.handle_message(make_message(), now=NOW, trace_id="trace_open_only")

    assert result.run.validated_action is not None
    assert result.run.validated_action.effective_action == ActionName.QUEUE_INVITES
    assert [transition.to_status for transition in result.run.state_transitions] == [
        GameWorkflowStatus.OPEN.value
    ]
    transition = result.run.state_transitions[0]
    assert transition.entity_id == "game_open_only"
    assert transition.metadata["tool_intent_kind"] == "create_game"
    assert transition.metadata["state_write_intent_contract"] == "state_write_intent.v1"
    assert state_store.current_status("game", "game_open_only") == GameWorkflowStatus.OPEN.value


def test_controlled_workflow_does_not_fallback_when_state_write_intent_missing() -> None:
    core = AgentCore()
    memory = InMemoryShortTermMemoryStore()
    state_store = InMemoryWorkflowStateStore()
    service = ControlledWorkflowService(
        core=core,
        context_builder=WorkflowContextBuilder(core, memory),
        semantic_resolver=SemanticResolver(FakeSemanticLLMClient(complete_create_game_contract())),
        tool_orchestrator=MissingStateIntentToolOrchestrator(),
        state_store=state_store,
        memory_store=memory,
    )

    result = service.handle_message(make_message(), now=NOW, trace_id="trace_missing_state_intent")

    assert result.run.validated_action is not None
    assert result.run.validated_action.effective_action == ActionName.QUEUE_INVITES
    assert result.run.state_transitions == []
    assert state_store.current_status("game", "game_without_intent") is None


def test_controlled_workflow_rejects_invalid_state_write_intent_contract() -> None:
    core = AgentCore()
    memory = InMemoryShortTermMemoryStore()
    state_store = InMemoryWorkflowStateStore()
    service = ControlledWorkflowService(
        core=core,
        context_builder=WorkflowContextBuilder(core, memory),
        semantic_resolver=SemanticResolver(FakeSemanticLLMClient(complete_create_game_contract())),
        tool_orchestrator=InvalidStateIntentToolOrchestrator(),
        state_store=state_store,
        memory_store=memory,
    )

    result = service.handle_message(make_message(), now=NOW, trace_id="trace_invalid_state_intent")

    assert result.run.validated_action is not None
    assert result.run.validated_action.effective_action == ActionName.QUEUE_INVITES
    assert result.run.state_transitions == []
    assert state_store.current_status("game", "game_invalid_state") is None
    state_event = next(event for event in result.trace_events if event.step == TraceStep.STATE_TRANSITION)
    rejected = state_event.content["rejected_state_write_intents"]
    assert rejected
    assert rejected[0]["schema"] == "state_write_intent.v1"
    assert rejected[0]["tool_name"] == ToolName.CREATE_GAME
    assert "state_write_intent.target_status 'cancelled' is not allowed for create_game" in rejected[0]["errors"]
    assert "state_write_intent.target_status_reason is not allowed for create_game" in rejected[0]["errors"]
    assert rejected[0]["raw_intent"]["entity_id"] == "game_invalid_state"


def test_controlled_workflow_close_uses_tool_state_write_intent_target_status() -> None:
    core = AgentCore()
    memory = InMemoryShortTermMemoryStore()
    state_store = InMemoryWorkflowStateStore()
    state_store.apply_transition(
        StateMachine().validate_game_transition(
            entity_id="game_expire_only",
            from_status=None,
            to_status=GameWorkflowStatus.OPEN,
            reason="seed existing open game",
        )
    )
    service = ControlledWorkflowService(
        core=core,
        context_builder=WorkflowContextBuilder(core, memory),
        semantic_resolver=SemanticResolver(FakeSemanticLLMClient(cancel_game_contract())),
        tool_orchestrator=ExpireCloseGameToolOrchestrator(),
        state_store=state_store,
        memory_store=memory,
    )

    result = service.handle_message(make_message("这桌不打了"), now=NOW, trace_id="trace_expire_only")

    assert result.run.validated_action is not None
    assert result.run.validated_action.effective_action == ActionName.CLOSE_GAME
    assert [transition.to_status for transition in result.run.state_transitions] == [
        GameWorkflowStatus.EXPIRED.value
    ]
    transition = result.run.state_transitions[0]
    assert transition.entity_id == "game_expire_only"
    assert transition.metadata["tool_intent_kind"] == "close_game"
    assert state_store.current_status("game", "game_expire_only") == GameWorkflowStatus.EXPIRED.value


def test_controlled_workflow_records_candidate_acceptance_and_updates_open_game_snapshot() -> None:
    core = AgentCore()
    seed_customers(core)
    memory = InMemoryShortTermMemoryStore()
    state_store = InMemoryWorkflowStateStore()

    create_service = ControlledWorkflowService(
        core=core,
        context_builder=WorkflowContextBuilder(core, memory, state_store=state_store),
        semantic_resolver=SemanticResolver(FakeSemanticLLMClient(complete_create_game_contract())),
        state_store=state_store,
        memory_store=memory,
    )
    create_result = create_service.handle_message(
        make_message("人齐开吧，有烟无烟都行"),
        now=NOW,
        trace_id="trace_create_before_accept",
    )
    game_id = create_result.run.state_transitions[-1].entity_id
    assert state_store.current_status("game", game_id) == GameWorkflowStatus.NEGOTIATING.value

    accept_service = ControlledWorkflowService(
        core=core,
        context_builder=WorkflowContextBuilder(core, memory, state_store=state_store),
        semantic_resolver=SemanticResolver(FakeSemanticLLMClient(candidate_accept_contract(game_id))),
        state_store=state_store,
        memory_store=memory,
    )
    accept_result = accept_service.handle_message(
        make_message(
            "可以",
            message_id="msg_candidate_accept",
            sender_id="ran",
            sender_name="冉姐",
        ),
        now=NOW,
        trace_id="trace_candidate_accept",
    )

    assert accept_result.run.validated_action is not None
    assert accept_result.run.validated_action.effective_action == ActionName.ACCEPT_SEAT
    assert accept_result.run.validated_action.required_tools == [ToolName.RECORD_SEAT_ACCEPTANCE]
    accept_tool = accept_result.tool_orchestration.result_for(ToolName.RECORD_SEAT_ACCEPTANCE)
    assert accept_tool is not None
    assert accept_tool.called is True
    assert accept_tool.allowed is True
    assert accept_tool.result["state_write_intent"]["kind"] == "record_seat_acceptance"
    assert accept_result.final_text == "好的，加你272了。"

    assert len(accept_result.run.state_transitions) == 1
    applied_transition = accept_result.run.state_transitions[0]
    assert applied_transition.allowed is True
    assert applied_transition.to_status == GameWorkflowStatus.NEGOTIATING.value
    assert applied_transition.metadata["store_applied"] is True
    assert applied_transition.metadata["participant"]["customer_id"] == "ran"
    assert applied_transition.metadata["seat_delta"] == {
        "previous_current_player_count": 1,
        "previous_missing_count": 3,
        "current_player_count": 2,
        "missing_count": 2,
        "seats_total": 4,
    }

    latest = state_store.transition_history(entity_type="game", entity_id=game_id)[-1]
    slots = latest.metadata["requirement"]["slots"]
    assert slots["current_player_count"]["value"] == 2
    assert slots["missing_count"]["value"] == 2

    rebuilt_context = WorkflowContextBuilder(core, memory, state_store=state_store).build(
        make_message("还有人吗", message_id="msg_after_accept"),
        now=NOW,
        trace_id="trace_after_accept",
    )
    rebuilt_game = rebuilt_context.context.open_games[0]
    assert rebuilt_game.slot("current_player_count").value == 2
    assert rebuilt_game.slot("missing_count").value == 2


def test_controlled_workflow_input_gate_deduplicates_source_message_id() -> None:
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
        input_gate=InMemoryInputGate(),
    )

    first = service.handle_message(
        make_message(
            "人齐开吧，有烟无烟都行",
            message_id="msg_duplicate_first",
            metadata={"source_message_id": "wechat_msg_001", "sequence": 1},
        ),
        now=NOW,
        trace_id="trace_duplicate_first",
    )
    second = service.handle_message(
        make_message(
            "人齐开吧，有烟无烟都行",
            message_id="msg_duplicate_retry",
            metadata={"source_message_id": "wechat_msg_001", "sequence": 1},
        ),
        now=NOW,
        trace_id="trace_duplicate_retry",
    )

    assert len(llm_client.calls) == 1
    assert first.final_text == "好的，我帮你问问。"
    assert second.final_text == first.final_text
    assert second.run.validated_action is not None
    assert second.run.validated_action.code == "input_gate_duplicate"
    assert second.run.state_transitions == []
    assert second.tool_orchestration.tool_results == []
    assert second.reply_approval is not None
    assert second.reply_approval.queued is False
    assert second.reply_approval.reason == "input_gate_short_circuit"
    game_id = first.run.state_transitions[-1].entity_id
    assert len(state_store.transition_history(entity_type="game", entity_id=game_id)) == 2
    assert len(memory.load("boss_trial", "zhang", now=NOW)) == 1

    gate_event = next(event for event in second.trace_events if event.step == "input_gate")
    assert gate_event.level == "WARN"
    assert gate_event.content["duplicate"] is True
    assert gate_event.content["has_cached_result"] is True
    final_event = next(event for event in second.trace_events if event.step == TraceStep.FINAL_OUTPUT)
    assert final_event.content["short_circuited"] is True
    assert final_event.content["input_gate"]["source_message_id"] == "wechat_msg_001"
    assert final_event.content["reply_approval"]["queued"] is False
    assert final_event.content["reply_approval"]["reason"] == "input_gate_short_circuit"
    assert final_event.content["trace_completeness"]["complete"] is True


def test_controlled_workflow_input_gate_waits_for_missing_sequence() -> None:
    core = AgentCore()
    seed_customers(core)
    memory = InMemoryShortTermMemoryStore()
    trace = InMemoryTraceRecorder()
    llm_client = FakeSemanticLLMClient(complete_create_game_contract())
    service = ControlledWorkflowService(
        core=core,
        context_builder=WorkflowContextBuilder(core, memory),
        semantic_resolver=SemanticResolver(llm_client),
        memory_store=memory,
        trace_recorder=trace,
        input_gate=InMemoryInputGate(),
    )

    blocked = service.handle_message(
        make_message(
            "第二条先到了",
            message_id="msg_sequence_2_first_arrival",
            metadata={"source_message_id": "wechat_msg_seq_2", "sequence": 2},
        ),
        now=NOW,
        trace_id="trace_sequence_blocked",
    )

    assert len(llm_client.calls) == 0
    assert blocked.run.validated_action is not None
    assert blocked.run.validated_action.code == "input_gate_waiting_for_sequence"
    assert blocked.final_text == "这条消息顺序有点乱，我先等第 1 条消息处理完再继续。"

    first = service.handle_message(
        make_message(
            "第一条到了",
            message_id="msg_sequence_1",
            metadata={"source_message_id": "wechat_msg_seq_1", "sequence": 1},
        ),
        now=NOW,
        trace_id="trace_sequence_1",
    )
    retried_second = service.handle_message(
        make_message(
            "第二条先到了",
            message_id="msg_sequence_2_retry",
            metadata={"source_message_id": "wechat_msg_seq_2", "sequence": 2},
        ),
        now=NOW,
        trace_id="trace_sequence_2_retry",
    )

    assert first.run.semantic_resolution is not None
    assert retried_second.run.semantic_resolution is not None
    assert len(llm_client.calls) == 2
    assert retried_second.run.validated_action is not None
    assert retried_second.run.validated_action.effective_action == ActionName.QUEUE_INVITES
    retry_gate_event = next(event for event in retried_second.trace_events if event.step == "input_gate")
    assert retry_gate_event.content["accepted"] is True


def test_controlled_workflow_input_gate_releases_inflight_on_exception() -> None:
    core = AgentCore()
    seed_customers(core)
    memory = InMemoryShortTermMemoryStore()
    context_builder = FailingThenOkContextBuilder(WorkflowContextBuilder(core, memory))
    llm_client = FakeSemanticLLMClient(complete_create_game_contract())
    service = ControlledWorkflowService(
        core=core,
        context_builder=context_builder,
        semantic_resolver=SemanticResolver(llm_client),
        memory_store=memory,
        input_gate=InMemoryInputGate(),
    )
    message = make_message(
        "人齐开吧，有烟无烟都行",
        message_id="msg_exception_retry",
        metadata={"source_message_id": "wechat_msg_retry_after_exception", "sequence": 1},
    )

    with pytest.raises(RuntimeError, match="temporary context failure"):
        service.handle_message(message, now=NOW, trace_id="trace_exception_first")

    retried = service.handle_message(message, now=NOW, trace_id="trace_exception_retry")

    assert context_builder.calls == 2
    assert len(llm_client.calls) == 1
    assert retried.run.validated_action is not None
    assert retried.run.validated_action.effective_action == ActionName.QUEUE_INVITES
    gate_event = next(event for event in retried.trace_events if event.step == "input_gate")
    assert gate_event.content["accepted"] is True


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
        ToolName.SEARCH_CURRENT_OPEN_GAMES,
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
