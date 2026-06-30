from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from mahjong_agent.core import AgentCore
from mahjong_agent.models import CustomerProfile, PlayPreference
from mahjong_agent.tool_orchestrator import (
    InMemoryToolExecutionLedger,
    SQLiteToolExecutionLedger,
    ToolOrchestrator,
    ToolOrchestratorConfig,
)
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


def make_resolution_with_profile_observations() -> SemanticResolution:
    resolution = make_resolution()
    resolution.raw_response = {
        "model_output": {
            "profile_observations": [
                {
                    "field": "smoke_preference",
                    "value": "any",
                    "confidence": 0.82,
                    "source": "current_message",
                    "evidence": "用户说有烟无烟都行",
                    "risk": "low",
                },
                {
                    "field": "private_health_note",
                    "value": "敏感信息",
                    "confidence": 0.91,
                    "source": "current_message",
                    "evidence": "不应写入",
                    "risk": "high",
                },
            ]
        }
    }
    return resolution


def make_validated(
    required_tools: list[ToolName],
    *,
    effective_action: ActionName = ActionName.QUEUE_INVITES,
    risk_level: RiskLevel = RiskLevel.LOW,
) -> ValidatedAction:
    return ValidatedAction(
        proposed_action=ProposedAction(
            name=ActionName.CREATE_GAME,
            source=ActionSource.LLM,
            confidence=0.9,
            reason="test",
        ),
        effective_action=effective_action,
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


def test_orchestrator_creates_game_state_write_intent_after_outbox() -> None:
    core = AgentCore()
    seed_customers(core)
    result = ToolOrchestrator(
        core,
        config=ToolOrchestratorConfig(allow_state_write=True),
    ).run(
        context=make_context(),
        semantic_resolution=make_resolution(),
        validated_action=make_validated(
            [ToolName.SEARCH_CANDIDATE_CUSTOMERS, ToolName.CREATE_PENDING_OUTBOX, ToolName.CREATE_GAME]
        ),
        now=NOW,
    )

    create_result = result.result_for(ToolName.CREATE_GAME)
    assert create_result is not None
    assert create_result.called is True
    assert create_result.allowed is True
    assert create_result.request.execution_mode == ToolExecutionMode.STATE_WRITE
    assert create_result.result["policy"] == "只生成状态写入意图，由 StateMachine 校验并由 StateStore 落库。"
    intent = create_result.result["state_write_intent"]
    assert intent["kind"] == "create_game"
    assert intent["entity_type"] == "game"
    assert intent["entity_id"] == "action_test"
    assert intent["target_status"] == "negotiating"
    assert intent["enter_negotiating_if_outbox_created"] is True


def test_orchestrator_creates_close_game_state_write_intent_when_enabled() -> None:
    result = ToolOrchestrator(
        AgentCore(),
        config=ToolOrchestratorConfig(allow_state_write=True),
    ).run(
        context=make_context(),
        semantic_resolution=make_resolution(),
        validated_action=make_validated(
            [ToolName.CLOSE_GAME],
            effective_action=ActionName.CLOSE_GAME,
            risk_level=RiskLevel.MEDIUM,
        ),
        now=NOW,
    )

    close_result = result.result_for(ToolName.CLOSE_GAME)
    assert close_result is not None
    assert close_result.called is True
    assert close_result.allowed is True
    assert close_result.request.execution_mode == ToolExecutionMode.STATE_WRITE
    assert close_result.result["state_write_intent"] == {
        "kind": "close_game",
        "entity_type": "game",
        "entity_id": "action_test",
        "target_status": "cancelled",
        "reason": "test allowed",
        "requirement": make_resolution().game_requirement.to_prompt_dict(),
    }


def test_orchestrator_applies_allowed_profile_observations_only() -> None:
    core = AgentCore()
    result = ToolOrchestrator(
        core,
        config=ToolOrchestratorConfig(allow_state_write=True),
    ).run(
        context=make_context(),
        semantic_resolution=make_resolution_with_profile_observations(),
        validated_action=make_validated(
            [ToolName.PROFILE_UPDATE],
            effective_action=ActionName.ASK_CLARIFICATION,
            risk_level=RiskLevel.LOW,
        ),
        now=NOW,
    )

    profile_result = result.result_for(ToolName.PROFILE_UPDATE)
    assert profile_result is not None
    assert profile_result.called is True
    assert profile_result.allowed is True
    assert profile_result.request.execution_mode == ToolExecutionMode.STATE_WRITE
    assert profile_result.result["applied_count"] == 1
    assert profile_result.result["rejected_count"] == 1
    assert profile_result.result["applied"][0]["field"] == "smoke_preference"
    assert "field_not_allowed" in profile_result.result["rejected"][0]["reason"]

    profile = core.store.customers["zhang"]
    observations = profile.metadata["controlled_profile_observations"]
    assert observations[0]["value"] == "any"
    assert observations[0]["evidence"] == "用户说有烟无烟都行"


def test_orchestrator_deduplicates_profile_observations_in_customer_metadata() -> None:
    core = AgentCore()
    orchestrator = ToolOrchestrator(
        core,
        config=ToolOrchestratorConfig(allow_state_write=True),
    )
    validated = make_validated(
        [ToolName.PROFILE_UPDATE],
        effective_action=ActionName.ASK_CLARIFICATION,
        risk_level=RiskLevel.LOW,
    )

    first = orchestrator.run(
        context=make_context(),
        semantic_resolution=make_resolution_with_profile_observations(),
        validated_action=validated,
        now=NOW,
    )
    second = orchestrator.run(
        context=make_context(),
        semantic_resolution=make_resolution_with_profile_observations(),
        validated_action=validated,
        now=NOW,
    )

    assert first.result_for(ToolName.PROFILE_UPDATE).result["applied_count"] == 1
    assert second.result_for(ToolName.PROFILE_UPDATE).deduplicated is True
    observations = core.store.customers["zhang"].metadata["controlled_profile_observations"]
    assert len(observations) == 1


def test_orchestrator_deduplicates_pending_outbox_by_idempotency_key() -> None:
    core = AgentCore()
    seed_customers(core)
    ledger = InMemoryToolExecutionLedger()
    orchestrator = ToolOrchestrator(core, execution_ledger=ledger)
    required_tools = [ToolName.SEARCH_CANDIDATE_CUSTOMERS, ToolName.CREATE_PENDING_OUTBOX]

    first = orchestrator.run(
        context=make_context(),
        semantic_resolution=make_resolution(),
        validated_action=make_validated(required_tools),
        now=NOW,
    )
    second = orchestrator.run(
        context=make_context(),
        semantic_resolution=make_resolution(),
        validated_action=make_validated(required_tools),
        now=NOW,
    )

    first_outbox = first.result_for(ToolName.CREATE_PENDING_OUTBOX)
    second_candidate_search = second.result_for(ToolName.SEARCH_CANDIDATE_CUSTOMERS)
    second_outbox = second.result_for(ToolName.CREATE_PENDING_OUTBOX)
    assert first_outbox is not None
    assert second_candidate_search is not None
    assert second_outbox is not None
    assert second_candidate_search.deduplicated is False
    assert second_outbox.deduplicated is True
    assert second_outbox.called is True
    assert second_outbox.allowed is True
    assert second_outbox.result["drafts"][0]["id"] == first_outbox.result["drafts"][0]["id"]
    assert len(ledger.history(tool_name=ToolName.CREATE_PENDING_OUTBOX)) == 2
    assert ledger.history(tool_name=ToolName.CREATE_PENDING_OUTBOX)[1].deduplicated is True


def test_orchestrator_deduplicates_pending_outbox_after_sqlite_ledger_reload(tmp_path) -> None:
    core = AgentCore()
    seed_customers(core)
    ledger_path = tmp_path / "tool_ledger.sqlite3"
    required_tools = [ToolName.SEARCH_CANDIDATE_CUSTOMERS, ToolName.CREATE_PENDING_OUTBOX]

    first = ToolOrchestrator(
        core,
        execution_ledger=SQLiteToolExecutionLedger(ledger_path),
    ).run(
        context=make_context(),
        semantic_resolution=make_resolution(),
        validated_action=make_validated(required_tools),
        now=NOW,
    )
    second = ToolOrchestrator(
        core,
        execution_ledger=SQLiteToolExecutionLedger(ledger_path),
    ).run(
        context=make_context(),
        semantic_resolution=make_resolution(),
        validated_action=make_validated(required_tools),
        now=NOW,
    )

    first_outbox = first.result_for(ToolName.CREATE_PENDING_OUTBOX)
    second_outbox = second.result_for(ToolName.CREATE_PENDING_OUTBOX)
    assert first_outbox is not None
    assert second_outbox is not None
    assert first_outbox.deduplicated is False
    assert second_outbox.deduplicated is True
    assert second_outbox.result["drafts"][0]["id"] == first_outbox.result["drafts"][0]["id"]
    reloaded_history = SQLiteToolExecutionLedger(ledger_path).history(tool_name=ToolName.CREATE_PENDING_OUTBOX)
    assert len(reloaded_history) == 2
    assert reloaded_history[0].deduplicated is False
    assert reloaded_history[1].deduplicated is True


def test_orchestrator_does_not_cache_denied_side_effect_tool_result() -> None:
    core = AgentCore()
    seed_customers(core)
    ledger = InMemoryToolExecutionLedger()
    denied = ToolOrchestrator(
        core,
        config=ToolOrchestratorConfig(allow_create_pending=False),
        execution_ledger=ledger,
    ).run(
        context=make_context(),
        semantic_resolution=make_resolution(),
        validated_action=make_validated([ToolName.CREATE_PENDING_OUTBOX]),
        now=NOW,
    )

    assert denied.result_for(ToolName.CREATE_PENDING_OUTBOX).allowed is False
    assert ledger.lookup("action_test:create_pending_outbox") is None

    allowed = ToolOrchestrator(core, execution_ledger=ledger).run(
        context=make_context(),
        semantic_resolution=make_resolution(),
        validated_action=make_validated([ToolName.SEARCH_CANDIDATE_CUSTOMERS, ToolName.CREATE_PENDING_OUTBOX]),
        now=NOW,
    )

    outbox = allowed.result_for(ToolName.CREATE_PENDING_OUTBOX)
    assert outbox is not None
    assert outbox.called is True
    assert outbox.allowed is True
    assert outbox.deduplicated is False
    assert ledger.lookup("action_test:create_pending_outbox") is outbox


def test_sqlite_tool_ledger_does_not_cache_denied_side_effect_tool_result(tmp_path) -> None:
    core = AgentCore()
    seed_customers(core)
    ledger_path = tmp_path / "tool_ledger.sqlite3"
    denied = ToolOrchestrator(
        core,
        config=ToolOrchestratorConfig(allow_create_pending=False),
        execution_ledger=SQLiteToolExecutionLedger(ledger_path),
    ).run(
        context=make_context(),
        semantic_resolution=make_resolution(),
        validated_action=make_validated([ToolName.CREATE_PENDING_OUTBOX]),
        now=NOW,
    )

    assert denied.result_for(ToolName.CREATE_PENDING_OUTBOX).allowed is False
    assert SQLiteToolExecutionLedger(ledger_path).lookup("action_test:create_pending_outbox") is None

    allowed = ToolOrchestrator(
        core,
        execution_ledger=SQLiteToolExecutionLedger(ledger_path),
    ).run(
        context=make_context(),
        semantic_resolution=make_resolution(),
        validated_action=make_validated([ToolName.SEARCH_CANDIDATE_CUSTOMERS, ToolName.CREATE_PENDING_OUTBOX]),
        now=NOW,
    )

    outbox = allowed.result_for(ToolName.CREATE_PENDING_OUTBOX)
    assert outbox is not None
    assert outbox.called is True
    assert outbox.allowed is True
    assert outbox.deduplicated is False
    assert SQLiteToolExecutionLedger(ledger_path).lookup("action_test:create_pending_outbox") is not None
