from __future__ import annotations

from mahjong_agent.action_validator import ActionValidationInput, ActionValidator
from mahjong_agent.state_machine import InMemoryWorkflowStateStore, SQLiteWorkflowStateStore, StateMachine
from mahjong_agent.tool_permissions import tool_allowed_for_action, validate_required_tools_for_action
from mahjong_agent.workflow_models import (
    ActionName,
    ActionSource,
    ConversationContext,
    GameRequirement,
    GameWorkflowStatus,
    ProposedAction,
    RiskLevel,
    SemanticResolution,
    SlotSource,
    SlotValue,
    ToolName,
    UserIntent,
    UserMessage,
)


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
    requirement = GameRequirement()
    requirement.set_slot(confirmed_slot("game_type", "hangzhou_mahjong"))
    requirement.set_slot(confirmed_slot("stake", "0.5"))
    requirement.set_slot(confirmed_slot("start_time_mode", "people_ready"))
    requirement.set_slot(confirmed_slot("missing_count", 3))
    requirement.set_slot(confirmed_slot("smoke", "any"))
    requirement.set_slot(confirmed_slot("duration_mode", "overnight"))
    return requirement


def incomplete_requirement() -> GameRequirement:
    requirement = GameRequirement()
    requirement.set_slot(confirmed_slot("duration_mode", "overnight", SlotSource.CONTEXT))
    return requirement


def make_context(
    *,
    text: str = "组",
    open_games: list[GameRequirement] | None = None,
    followup: bool = True,
) -> ConversationContext:
    return ConversationContext(
        current_message=UserMessage(
            text=text,
            sender_id="zhang",
            sender_name="张哥",
            conversation_id="group_a",
            trace_id="trace_current",
            message_id="msg_current",
        ),
        open_games=open_games or [],
        followup_context={
            "current_message_may_answer_previous_reply": followup,
            "previous_reply_asked_create_confirmation": followup,
            "signals": {
                "current_message_is_short_ack": text in {"组", "可以", "好", "要"},
                "previous_reply_asked_create_confirmation": followup,
            },
        },
    )


def make_resolution(
    action: ActionName,
    requirement: GameRequirement | None = None,
    *,
    intent: UserIntent = UserIntent.FIND_PLAYERS,
    confidence: float = 0.86,
    needs_human_review: bool = False,
    risk_level: RiskLevel = RiskLevel.LOW,
) -> SemanticResolution:
    return SemanticResolution(
        intent=intent,
        proposed_action=ProposedAction(
            name=action,
            source=ActionSource.LLM,
            confidence=confidence,
            reason="test proposal",
            risk_level=risk_level,
        ),
        game_requirement=requirement or complete_requirement(),
        needs_human_review=needs_human_review,
    )


def make_resolution_with_profile_observation(action: ActionName) -> SemanticResolution:
    resolution = make_resolution(action, complete_requirement())
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
                }
            ]
        }
    }
    return resolution


def test_create_game_with_complete_slots_queues_invites_not_final_reply() -> None:
    validator = ActionValidator()

    result = validator.validate(make_context(), make_resolution(ActionName.CREATE_GAME))

    assert result.effective_action == ActionName.QUEUE_INVITES
    assert result.allowed is True
    assert result.approval_required is True
    assert result.required_tools == [
        ToolName.SEARCH_CURRENT_OPEN_GAMES,
        ToolName.SEARCH_CANDIDATE_CUSTOMERS,
        ToolName.CREATE_PENDING_OUTBOX,
        ToolName.CREATE_GAME,
    ]
    assert result.missing_slots == []
    assert result.idempotency_key.startswith("action_")


def test_confirmed_create_confirmation_with_missing_slots_becomes_clarification() -> None:
    result = ActionValidator().validate(
        make_context(text="组"),
        make_resolution(
            ActionName.ASK_CREATE_CONFIRMATION,
            incomplete_requirement(),
            intent=UserIntent.FIND_PLAYERS,
        ),
    )

    assert result.effective_action == ActionName.ASK_CLARIFICATION
    assert result.allowed is True
    assert result.code == "confirmed_create_missing_slots"
    assert "stake" in result.missing_slots
    assert "party_size" in result.missing_slots


def test_profile_observations_append_profile_update_tool_for_allowed_actions() -> None:
    result = ActionValidator().validate(
        make_context(text="有烟无烟都行"),
        make_resolution_with_profile_observation(ActionName.CREATE_GAME),
    )

    assert result.allowed is True
    assert result.effective_action == ActionName.QUEUE_INVITES
    assert result.required_tools == [
        ToolName.SEARCH_CURRENT_OPEN_GAMES,
        ToolName.SEARCH_CANDIDATE_CUSTOMERS,
        ToolName.CREATE_PENDING_OUTBOX,
        ToolName.CREATE_GAME,
        ToolName.PROFILE_UPDATE,
    ]


def test_high_risk_action_does_not_append_profile_update_tool() -> None:
    resolution = make_resolution_with_profile_observation(
        ActionName.CREATE_GAME,
    )
    resolution.needs_human_review = True
    resolution.proposed_action.risk_level = RiskLevel.HIGH

    result = ActionValidator().validate(make_context(), resolution)

    assert result.allowed is False
    assert result.effective_action == ActionName.HUMAN_REVIEW
    assert ToolName.PROFILE_UPDATE not in result.required_tools


def test_invalid_action_arguments_are_rejected_before_tool_orchestration() -> None:
    resolution = make_resolution(ActionName.CREATE_GAME, complete_requirement())
    resolution.proposed_action.arguments = {"game_id": "llm_generated_game"}

    result = ActionValidator().validate(make_context(), resolution)

    assert result.allowed is False
    assert result.effective_action == ActionName.HUMAN_REVIEW
    assert result.code == "action_arguments_contract_invalid"
    assert result.required_tools == []
    assert "action_arguments.game_id is not allowed for create_game" in result.reason


def test_tool_permission_contract_allows_only_tools_for_effective_action() -> None:
    assert tool_allowed_for_action(ToolName.SEARCH_CANDIDATE_CUSTOMERS, ActionName.QUEUE_INVITES)
    assert tool_allowed_for_action(ToolName.PROFILE_UPDATE, ActionName.ASK_CLARIFICATION)
    assert not tool_allowed_for_action(ToolName.SEND_MESSAGE, ActionName.QUEUE_INVITES)
    assert validate_required_tools_for_action(
        ActionName.MATCH_EXISTING_GAME,
        [ToolName.SEARCH_CURRENT_OPEN_GAMES, ToolName.CREATE_PENDING_OUTBOX],
    ) == ["tool create_pending_outbox is not allowed for action match_existing_game"]


def test_action_validator_rejects_required_tools_outside_effective_action_permission() -> None:
    validator = ActionValidator()
    context = make_context()
    resolution = make_resolution(ActionName.CREATE_GAME, complete_requirement())

    result = validator._validated(
        ActionValidationInput(context=context, semantic_resolution=resolution),
        effective_action=ActionName.QUEUE_INVITES,
        allowed=True,
        code="test_invalid_tool",
        reason="test invalid tool",
        required_tools=[ToolName.SEND_MESSAGE],
    )

    assert result.allowed is False
    assert result.effective_action == ActionName.HUMAN_REVIEW
    assert result.code == "tool_permission_denied"
    assert result.risk_level == RiskLevel.HIGH
    assert result.required_tools == []
    assert "tool send_message is not allowed for action queue_invites" in result.reason


def test_create_game_missing_critical_slots_downgrades_to_clarification() -> None:
    requirement = GameRequirement()
    requirement.set_slot(confirmed_slot("game_type", "hangzhou_mahjong"))
    requirement.set_slot(confirmed_slot("stake", "0.5"))

    result = ActionValidator().validate(
        make_context(text="帮我组一桌"),
        make_resolution(ActionName.CREATE_GAME, requirement),
    )

    assert result.allowed is False
    assert result.effective_action == ActionName.ASK_CLARIFICATION
    assert result.code == "critical_slots_missing"
    assert {"start_time_mode", "party_size", "smoke", "duration_mode"}.issubset(set(result.missing_slots))


def test_search_existing_no_match_with_followup_signal_does_not_queue_invites() -> None:
    result = ActionValidator().validate(
        make_context(text="可以", followup=True),
        make_resolution(
            ActionName.SEARCH_EXISTING_GAMES,
            complete_requirement(),
            intent=UserIntent.INQUIRE_EXISTING_GAME,
            confidence=0.9,
        ),
    )

    assert result.allowed is True
    assert result.effective_action == ActionName.ASK_CREATE_CONFIRMATION
    assert result.code == "no_existing_match_ask_create"
    assert result.required_tools == [ToolName.SEARCH_CURRENT_OPEN_GAMES]


def test_followup_create_contract_queues_invites_when_llm_proposes_create() -> None:
    result = ActionValidator().validate(
        make_context(text="可以", followup=True),
        make_resolution(
            ActionName.CREATE_GAME,
            complete_requirement(),
            intent=UserIntent.FIND_PLAYERS,
            confidence=0.9,
        ),
    )

    assert result.allowed is True
    assert result.effective_action == ActionName.QUEUE_INVITES
    assert result.code == "queue_invites_after_create_validation"
    assert result.required_tools == [
        ToolName.SEARCH_CURRENT_OPEN_GAMES,
        ToolName.SEARCH_CANDIDATE_CUSTOMERS,
        ToolName.CREATE_PENDING_OUTBOX,
        ToolName.CREATE_GAME,
    ]


def test_search_existing_no_match_without_create_confirmation_asks_create_confirmation() -> None:
    result = ActionValidator().validate(
        make_context(text="现在有0.5吗", followup=False),
        make_resolution(
            ActionName.SEARCH_EXISTING_GAMES,
            complete_requirement(),
            intent=UserIntent.INQUIRE_EXISTING_GAME,
            confidence=0.9,
        ),
    )

    assert result.allowed is True
    assert result.effective_action == ActionName.ASK_CREATE_CONFIRMATION
    assert result.code == "no_existing_match_ask_create"


def test_existing_game_preferred_over_new_create() -> None:
    open_game = complete_requirement()
    open_game.set_slot(confirmed_slot("missing_count", 1, source=SlotSource.TOOL))

    result = ActionValidator().validate(
        make_context(text="帮我组一个", open_games=[open_game]),
        make_resolution(ActionName.CREATE_GAME, complete_requirement()),
    )

    assert result.allowed is True
    assert result.effective_action == ActionName.MATCH_EXISTING_GAME
    assert result.code == "existing_game_preferred"
    assert result.required_tools == [ToolName.SEARCH_CURRENT_OPEN_GAMES]


def test_existing_game_preferred_when_request_has_acceptable_ranges() -> None:
    open_game = complete_requirement()
    open_game.set_slot(confirmed_slot("stake", "0.5", source=SlotSource.TOOL))
    open_game.set_slot(confirmed_slot("smoke", "no_smoke", source=SlotSource.TOOL))
    open_game.set_slot(confirmed_slot("missing_count", 1, source=SlotSource.TOOL))

    requested = complete_requirement()
    requested.set_slot(confirmed_slot("stake", ["0.5", "1"]))
    requested.set_slot(confirmed_slot("smoke", "any"))

    result = ActionValidator().validate(
        make_context(text="0.5或者1都行，有烟无烟都可", open_games=[open_game]),
        make_resolution(ActionName.CREATE_GAME, requested),
    )

    assert result.allowed is True
    assert result.effective_action == ActionName.MATCH_EXISTING_GAME
    assert result.code == "existing_game_preferred"


def test_low_confidence_state_action_downgrades_to_clarification() -> None:
    result = ActionValidator().validate(
        make_context(),
        make_resolution(ActionName.CREATE_GAME, complete_requirement(), confidence=0.4),
    )

    assert result.allowed is False
    assert result.effective_action == ActionName.ASK_CLARIFICATION
    assert result.code == "low_confidence_downgrade"


def test_human_review_risk_blocks_automatic_action() -> None:
    result = ActionValidator().validate(
        make_context(),
        make_resolution(
            ActionName.CREATE_GAME,
            complete_requirement(),
            needs_human_review=True,
            risk_level=RiskLevel.HIGH,
        ),
    )

    assert result.allowed is False
    assert result.effective_action == ActionName.HUMAN_REVIEW
    assert result.approval_required is True
    assert result.risk_level == RiskLevel.HIGH


def test_state_machine_blocks_terminal_reopen() -> None:
    machine = StateMachine()

    assert machine.can_transition_game(None, GameWorkflowStatus.OPEN)
    assert machine.can_transition_game(GameWorkflowStatus.OPEN, GameWorkflowStatus.NEGOTIATING)
    assert not machine.can_transition_game(GameWorkflowStatus.CANCELLED, GameWorkflowStatus.OPEN)
    transition = machine.validate_game_transition(
        entity_id="game_1",
        from_status=GameWorkflowStatus.CANCELLED,
        to_status=GameWorkflowStatus.OPEN,
        reason="terminal status cannot reopen",
    )
    assert transition.allowed is False
    assert transition.metadata["state_machine_version"] == "controlled_state_machine.v1"


def test_workflow_state_store_applies_and_audits_transitions() -> None:
    machine = StateMachine()
    store = InMemoryWorkflowStateStore()

    opened = store.apply_transition(
        machine.validate_game_transition(
            entity_id="game_1",
            from_status=None,
            to_status=GameWorkflowStatus.OPEN,
            reason="create pending game",
        )
    )
    negotiating = store.apply_transition(
        machine.validate_game_transition(
            entity_id="game_1",
            from_status=GameWorkflowStatus.OPEN,
            to_status=GameWorkflowStatus.NEGOTIATING,
            reason="pending outbox created",
        )
    )
    stale = store.apply_transition(
        machine.validate_game_transition(
            entity_id="game_1",
            from_status=GameWorkflowStatus.OPEN,
            to_status=GameWorkflowStatus.CANCELLED,
            reason="stale close attempt",
        )
    )

    assert opened.allowed is True
    assert opened.metadata["store_applied"] is True
    assert negotiating.allowed is True
    assert store.current_status("game", "game_1") == GameWorkflowStatus.NEGOTIATING.value
    assert stale.allowed is False
    assert stale.metadata["store_rejected_reason"] == "state_store_status_mismatch"
    assert len(store.transition_history(entity_type="game", entity_id="game_1")) == 3


def test_sqlite_workflow_state_store_persists_status_and_history(tmp_path) -> None:
    machine = StateMachine()
    path = tmp_path / "workflow_state.sqlite3"
    store = SQLiteWorkflowStateStore(path)

    opened = store.apply_transition(
        machine.validate_game_transition(
            entity_id="game_sqlite",
            from_status=None,
            to_status=GameWorkflowStatus.OPEN,
            reason="create pending game",
        )
    )
    negotiating = store.apply_transition(
        machine.validate_game_transition(
            entity_id="game_sqlite",
            from_status=GameWorkflowStatus.OPEN,
            to_status=GameWorkflowStatus.NEGOTIATING,
            reason="pending outbox created",
        )
    )

    reloaded = SQLiteWorkflowStateStore(path)
    stale = reloaded.apply_transition(
        machine.validate_game_transition(
            entity_id="game_sqlite",
            from_status=GameWorkflowStatus.OPEN,
            to_status=GameWorkflowStatus.CANCELLED,
            reason="stale close attempt",
        )
    )
    history = reloaded.transition_history(entity_type="game", entity_id="game_sqlite")

    assert opened.metadata["store_backend"] == "sqlite"
    assert negotiating.metadata["store_applied"] is True
    assert reloaded.current_status("game", "game_sqlite") == GameWorkflowStatus.NEGOTIATING.value
    assert stale.allowed is False
    assert stale.metadata["store_rejected_reason"] == "state_store_status_mismatch"
    assert [item.to_status for item in history] == [
        GameWorkflowStatus.OPEN.value,
        GameWorkflowStatus.NEGOTIATING.value,
        GameWorkflowStatus.CANCELLED.value,
    ]
    assert history[-1].allowed is False
