from __future__ import annotations

from mahjong_agent.action_validator import ActionValidator
from mahjong_agent.state_machine import StateMachine
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


def test_create_game_with_complete_slots_queues_invites_not_final_reply() -> None:
    validator = ActionValidator()

    result = validator.validate(make_context(), make_resolution(ActionName.CREATE_GAME))

    assert result.effective_action == ActionName.QUEUE_INVITES
    assert result.allowed is True
    assert result.approval_required is True
    assert result.required_tools == [ToolName.SEARCH_CANDIDATE_CUSTOMERS, ToolName.CREATE_PENDING_OUTBOX]
    assert result.missing_slots == []
    assert result.idempotency_key.startswith("action_")


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


def test_search_existing_no_match_but_followup_confirmed_create_queues_invites() -> None:
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
    assert result.effective_action == ActionName.QUEUE_INVITES
    assert result.code == "confirmed_create_after_no_match"
    assert result.required_tools == [ToolName.SEARCH_CANDIDATE_CUSTOMERS, ToolName.CREATE_PENDING_OUTBOX]


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
