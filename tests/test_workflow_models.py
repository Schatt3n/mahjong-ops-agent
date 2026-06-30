from __future__ import annotations

from mahjong_agent.workflow_models import (
    ActionName,
    ActionSource,
    ConversationContext,
    CustomerProfile,
    GameRequirement,
    ProposedAction,
    SlotSource,
    SlotValue,
    ToolCallRequest,
    ToolName,
    UserIntent,
    UserMessage,
    WorkflowTurn,
)


def confirmed_slot(
    name: str,
    value,
    source: SlotSource = SlotSource.EXPLICIT,
    confidence: float = 0.9,
) -> SlotValue:
    return SlotValue(
        name=name,
        value=value,
        source=source,
        confidence=confidence,
        confirmed=True,
        needs_confirmation=False,
    )


def test_slot_value_keeps_source_confidence_and_confirmation_contract() -> None:
    profile_slot = confirmed_slot("stake", "0.5", source=SlotSource.PROFILE)
    explicit_slot = confirmed_slot("stake", "0.5", source=SlotSource.EXPLICIT)
    inferred_slot = SlotValue(
        name="stake",
        value="0.5",
        source="not_a_known_source",
        confidence=2,
        confirmed=False,
        needs_confirmation=True,
    )

    assert profile_slot.usable
    assert not profile_slot.trusted_for_state
    assert explicit_slot.trusted_for_state
    assert inferred_slot.source == SlotSource.UNKNOWN
    assert inferred_slot.confidence == 1.0

    needs_confirmation = explicit_slot.require_confirmation("用户输入存在歧义")

    assert not needs_confirmation.usable
    assert needs_confirmation.metadata["confirmation_reason"] == "用户输入存在歧义"


def test_game_requirement_missing_slots_accepts_equivalent_structured_slots() -> None:
    requirement = GameRequirement()
    requirement.set_slot(confirmed_slot("game_type", "hangzhou_mahjong"))
    requirement.set_slot(confirmed_slot("stake", "0.5"))
    requirement.set_slot(confirmed_slot("start_at", "2026-06-30T16:00:00+08:00"))
    requirement.set_slot(confirmed_slot("missing_count", 2))
    requirement.set_slot(confirmed_slot("smoke", "any"))
    requirement.set_slot(confirmed_slot("duration_hours", 4))

    assert requirement.missing_required_slots() == []
    assert requirement.is_complete()


def test_game_requirement_inherits_confirmed_context_without_overwriting_explicit_slots() -> None:
    previous = GameRequirement()
    previous.set_slot(confirmed_slot("smoke", "any"))
    previous.set_slot(confirmed_slot("stake", "0.5"))

    current = GameRequirement()
    current.set_slot(confirmed_slot("stake", "1"))
    current.set_slot(
        SlotValue(
            name="smoke",
            value="unknown",
            source=SlotSource.INFERRED,
            confidence=0.99,
            confirmed=False,
            needs_confirmation=True,
        )
    )

    current.inherit_confirmed_context(previous)

    assert current.slot("stake").value == "1"
    assert current.slot("stake").source == SlotSource.EXPLICIT
    assert current.slot("smoke").value == "any"
    assert current.slot("smoke").source == SlotSource.CONTEXT
    assert current.slot("smoke").metadata["inherited_from_context"] is True


def test_conversation_context_prompt_contract_includes_previous_reply_and_requirement() -> None:
    previous_requirement = GameRequirement()
    previous_requirement.set_slot(confirmed_slot("stake", "0.5"))
    previous_requirement.set_slot(confirmed_slot("smoke", "any"))
    previous_turn = WorkflowTurn(
        user_message=UserMessage(
            text="通宵0.5有人吗",
            sender_id="zhang",
            sender_name="张哥",
            conversation_id="test01",
            trace_id="trace_prev",
        ),
        system_reply="0.5的暂时没有诶。要组一个吗？",
        game_requirement=previous_requirement,
    )
    current_message = UserMessage(
        text="组",
        sender_id="zhang",
        sender_name="张哥",
        conversation_id="test01",
        trace_id="trace_current",
    )
    profile = CustomerProfile(
        customer_id="zhang",
        display_name="张哥",
        preferred_slots={"stake": confirmed_slot("stake", "0.5", source=SlotSource.PROFILE)},
    )

    context = ConversationContext(
        current_message=current_message,
        customer_profile=profile,
        recent_turns=[previous_turn],
        memory_summary="上一轮用户在问有没有 0.5 通宵局，老板问是否要新组。",
    )
    prompt_dict = context.to_prompt_dict()

    assert prompt_dict["current_message"]["text"] == "组"
    assert prompt_dict["previous_system_reply"] == "0.5的暂时没有诶。要组一个吗？"
    assert prompt_dict["previous_game_requirement"]["slots"]["smoke"]["value"] == "any"
    assert prompt_dict["customer_profile"]["preferred_slots"]["stake"]["source"] == "profile"
    assert "要新组" in prompt_dict["memory_summary"]


def test_contract_coerces_unknown_llm_values_instead_of_crashing() -> None:
    action = ProposedAction(
        name="invented_action",
        source="llm",
        confidence=-1,
        reason="模型输出了未知动作",
    )
    tool_request = ToolCallRequest(tool_name="invented_tool")

    assert action.name == ActionName.UNKNOWN
    assert action.source == ActionSource.LLM
    assert action.confidence == 0.0
    assert tool_request.tool_name == ToolName.UNKNOWN


def test_semantic_resolution_accepts_string_intent_contract() -> None:
    from mahjong_agent.workflow_models import SemanticResolution

    resolution = SemanticResolution(
        intent="find_players",
        proposed_action=ProposedAction(
            name="create_game",
            source="llm",
            confidence=0.88,
            reason="用户明确说帮我组一桌",
        ),
    )

    assert resolution.intent == UserIntent.FIND_PLAYERS
    assert resolution.proposed_action.name == ActionName.CREATE_GAME
