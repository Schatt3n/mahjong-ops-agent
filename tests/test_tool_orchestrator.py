from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from mahjong_agent.core import AgentCore
from mahjong_agent.models import CustomerProfile, PlayPreference
from mahjong_agent.tool_orchestrator import ToolOrchestrator, ToolOrchestratorConfig
from mahjong_agent.workflow_models import (
    ActionName,
    ActionSource,
    ConversationContext,
    GameRequirement,
    ProposedAction,
    RiskLevel,
    SemanticResolution,
    SlotSource,
    SlotValue,
    ToolExecutionMode,
    ToolName,
    UserIntent,
    UserMessage,
    ValidatedAction,
)


TZ = ZoneInfo("Asia/Shanghai")
NOW = datetime(2026, 6, 30, 16, 0, tzinfo=TZ)


def confirmed_slot(name: str, value, source: SlotSource = SlotSource.EXPLICIT) -> SlotValue:
    return SlotValue(
        name=name,
        value=value,
        source=source,
        confidence=0.9,
        confirmed=True,
        needs_confirmation=False,
    )


def complete_requirement() -> GameRequirement:
    requirement = GameRequirement(organizer_id="zhang", organizer_name="张哥")
    requirement.set_slot(confirmed_slot("game_type", "hangzhou_mahjong"))
    requirement.set_slot(confirmed_slot("stake", "0.5"))
    requirement.set_slot(confirmed_slot("start_time_mode", "people_ready"))
    requirement.set_slot(confirmed_slot("missing_count", 3))
    requirement.set_slot(confirmed_slot("smoke", "no_smoke"))
    requirement.set_slot(confirmed_slot("duration_hours", 4))
    requirement.set_slot(confirmed_slot("duration_mode", "fixed"))
    return requirement


def make_context(open_games: list[GameRequirement] | None = None) -> ConversationContext:
    return ConversationContext(
        current_message=UserMessage(
            text="帮我组一桌",
            sender_id="zhang",
            sender_name="张哥",
            conversation_id="group_a",
            trace_id="trace_tool",
            message_id="msg_tool",
        ),
        open_games=open_games or [],
    )


def make_resolution(requirement: GameRequirement | None = None) -> SemanticResolution:
    return SemanticResolution(
        intent=UserIntent.FIND_PLAYERS,
        proposed_action=ProposedAction(
            name=ActionName.CREATE_GAME,
            source=ActionSource.LLM,
            confidence=0.9,
            reason="用户明确要求组局",
        ),
        game_requirement=requirement or complete_requirement(),
    )


def make_validated(required_tools: list[ToolName], *, risk_level: RiskLevel = RiskLevel.LOW) -> ValidatedAction:
    return ValidatedAction(
        proposed_action=ProposedAction(
            name=ActionName.CREATE_GAME,
            source=ActionSource.LLM,
            confidence=0.9,
            reason="test",
        ),
        effective_action=ActionName.QUEUE_INVITES,
        allowed=True,
        code="allowed",
        reason="test allowed",
        risk_level=risk_level,
        approval_required=True,
        idempotency_key="action_test",
        required_tools=required_tools,
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
            smoke_free_preference=True,
            play_preferences=[PlayPreference(game_type="hangzhou_mahjong", preferred_levels=["0.5"])],
        )
    )


def test_orchestrator_runs_current_game_search_as_read_only_tool() -> None:
    open_game = complete_requirement()
    open_game.set_slot(confirmed_slot("missing_count", 1, source=SlotSource.TOOL))
    orchestrator = ToolOrchestrator(AgentCore())

    result = orchestrator.run(
        context=make_context(open_games=[open_game]),
        semantic_resolution=make_resolution(),
        validated_action=make_validated([ToolName.SEARCH_CURRENT_OPEN_GAMES]),
        now=NOW,
    )

    assert len(result.tool_results) == 1
    tool_result = result.tool_results[0]
    assert tool_result.called is True
    assert tool_result.allowed is True
    assert tool_result.request.tool_name == ToolName.SEARCH_CURRENT_OPEN_GAMES
    assert tool_result.request.execution_mode == ToolExecutionMode.READ_ONLY
    assert tool_result.request.idempotency_key == "action_test:search_current_open_games"
    assert tool_result.result["result_count"] == 1


def test_orchestrator_searches_candidates_then_creates_pending_outbox() -> None:
    core = AgentCore()
    seed_customers(core)
    orchestrator = ToolOrchestrator(core)

    result = orchestrator.run(
        context=make_context(),
        semantic_resolution=make_resolution(),
        validated_action=make_validated([ToolName.SEARCH_CANDIDATE_CUSTOMERS, ToolName.CREATE_PENDING_OUTBOX]),
        now=NOW,
    )

    assert [item.request.tool_name for item in result.tool_results] == [
        ToolName.SEARCH_CANDIDATE_CUSTOMERS,
        ToolName.CREATE_PENDING_OUTBOX,
    ]
    candidate_result = result.tool_results[0]
    outbox_result = result.tool_results[1]
    assert candidate_result.called is True
    assert candidate_result.result["result_count"] >= 1
    assert outbox_result.called is True
    assert outbox_result.request.execution_mode == ToolExecutionMode.CREATE_PENDING
    assert outbox_result.result["policy"] == "只创建待审批草稿，不自动发送。"
    assert outbox_result.result["drafts"]
    assert outbox_result.result["drafts"][0]["status"] == "pending_approval"
    assert "打吗" in outbox_result.result["drafts"][0]["message_text"]


def test_orchestrator_blocks_pending_outbox_without_candidate_result() -> None:
    result = ToolOrchestrator(AgentCore()).run(
        context=make_context(),
        semantic_resolution=make_resolution(),
        validated_action=make_validated([ToolName.CREATE_PENDING_OUTBOX]),
        now=NOW,
    )

    assert len(result.tool_results) == 1
    assert result.tool_results[0].called is False
    assert result.tool_results[0].allowed is False
    assert "requires candidate search" in result.tool_results[0].error


def test_orchestrator_blocks_direct_send_tool_by_default() -> None:
    result = ToolOrchestrator(AgentCore()).run(
        context=make_context(),
        semantic_resolution=make_resolution(),
        validated_action=make_validated([ToolName.SEND_MESSAGE]),
        now=NOW,
    )

    assert result.tool_results[0].called is False
    assert result.tool_results[0].allowed is False
    assert result.tool_results[0].request.execution_mode == ToolExecutionMode.DIRECT_SEND
    assert "High risk" in result.tool_results[0].error


def test_orchestrator_blocks_state_write_when_disabled() -> None:
    result = ToolOrchestrator(
        AgentCore(),
        config=ToolOrchestratorConfig(allow_state_write=False),
    ).run(
        context=make_context(),
        semantic_resolution=make_resolution(),
        validated_action=make_validated([ToolName.CLOSE_GAME]),
        now=NOW,
    )

    assert result.tool_results[0].called is False
    assert result.tool_results[0].allowed is False
    assert result.tool_results[0].request.execution_mode == ToolExecutionMode.STATE_WRITE
    assert "State-write tools are disabled" in result.tool_results[0].error
