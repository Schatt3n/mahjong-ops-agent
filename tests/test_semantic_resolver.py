from __future__ import annotations

from typing import Any

from mahjong_agent.semantic_resolver import SemanticResolver, SemanticResolverConfig
from mahjong_agent.workflow_models import (
    ActionName,
    ConversationContext,
    CustomerProfile,
    GameRequirement,
    RiskLevel,
    SlotSource,
    SlotValue,
    UserIntent,
    UserMessage,
    WorkflowTurn,
)


class FakeSemanticLLMClient:
    def __init__(self, output: str | dict[str, Any] | BaseException) -> None:
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
        if isinstance(self.output, BaseException):
            raise self.output
        return self.output


def confirmed_slot(name: str, value, source: SlotSource = SlotSource.EXPLICIT) -> SlotValue:
    return SlotValue(
        name=name,
        value=value,
        source=source,
        confidence=0.9,
        confirmed=True,
        needs_confirmation=False,
    )


def make_context() -> ConversationContext:
    previous_requirement = GameRequirement()
    previous_requirement.set_slot(confirmed_slot("stake", "0.5"))
    previous_requirement.set_slot(confirmed_slot("duration_mode", "overnight"))
    previous_turn = WorkflowTurn(
        user_message=UserMessage(
            text="通宵0.5有人吗",
            sender_id="zhang",
            sender_name="张哥",
            conversation_id="group_a",
            trace_id="trace_prev",
            message_id="msg_prev",
        ),
        system_reply="0.5的暂时没有诶。要组一个吗？",
        game_requirement=previous_requirement,
    )
    return ConversationContext(
        current_message=UserMessage(
            text="组",
            sender_id="zhang",
            sender_name="张哥",
            conversation_id="group_a",
            trace_id="trace_current",
            message_id="msg_current",
        ),
        customer_profile=CustomerProfile(
            customer_id="zhang",
            display_name="张哥",
            preferred_slots={
                "stake_preferences": SlotValue(
                    name="stake_preferences",
                    value=["0.5", "1"],
                    source=SlotSource.PROFILE,
                    confidence=0.75,
                    confirmed=False,
                    needs_confirmation=True,
                )
            },
        ),
        recent_turns=[previous_turn],
        memory_summary="上一轮用户问通宵0.5有没有局，老板问是否要组一个。",
        followup_context={
            "has_previous_system_reply": True,
            "previous_system_reply": "0.5的暂时没有诶。要组一个吗？",
            "current_message_may_answer_previous_reply": True,
        },
    )


def test_semantic_resolver_builds_prompt_from_conversation_context() -> None:
    client = FakeSemanticLLMClient(
        {
            "intent": "find_players",
            "proposed_action": "create_game",
            "confidence": 0.86,
            "needs_human_review": False,
            "reasoning_summary": "用户在确认上一轮是否要新组局。",
            "slots": {
                "stake": {
                    "value": "0.5",
                    "source": "context",
                    "confidence": 0.9,
                    "confirmed": True,
                    "needs_confirmation": False,
                    "evidence": "上一轮已确认通宵0.5",
                },
                "party_size": {
                    "value": 1,
                    "source": "profile",
                    "confidence": 0.7,
                    "evidence": "画像显示张哥通常一人",
                },
            },
        }
    )
    resolver = SemanticResolver(client)

    resolution = resolver.resolve(make_context())

    assert client.calls
    messages = client.calls[0]["messages"]
    assert "语义解析器" in messages[0]["content"]
    assert "不能生成最终老板回复" in messages[0]["content"]
    assert "previous_system_reply" in messages[1]["content"]
    assert "要组一个吗" in messages[1]["content"]

    assert resolution.intent == UserIntent.FIND_PLAYERS
    assert resolution.proposed_action.name == ActionName.CREATE_GAME
    assert resolution.proposed_action.source == "llm"
    assert resolution.proposed_action.confidence == 0.86
    assert resolution.proposed_action.risk_level == RiskLevel.MEDIUM
    assert resolution.reasoning_summary == "用户在确认上一轮是否要新组局。"
    assert resolution.raw_response["llm_contract"]["accepted"] is True
    assert resolution.raw_response["llm_contract"]["strict_json"] is True
    assert resolution.raw_response["llm_contract"]["raw_output"]["intent"] == "find_players"

    stake = resolution.game_requirement.slot("stake")
    party_size = resolution.game_requirement.slot("party_size")
    assert stake.value == "0.5"
    assert stake.source == SlotSource.CONTEXT
    assert stake.trusted_for_state
    assert party_size.value == 1
    assert party_size.source == SlotSource.PROFILE
    assert not party_size.usable


def test_semantic_resolver_maps_action_from_intent_when_action_missing() -> None:
    client = FakeSemanticLLMClient(
        {
            "intent": "inquire_existing_game",
            "confidence": 0.81,
            "reasoning_summary": "用户只是问有没有现成局。",
            "slots": {"stake": {"value": "0.5", "source": "explicit", "confidence": 0.9}},
        }
    )

    resolution = SemanticResolver(client).resolve(make_context())

    assert resolution.intent == UserIntent.INQUIRE_EXISTING_GAME
    assert resolution.proposed_action.name == ActionName.SEARCH_EXISTING_GAMES
    assert resolution.game_requirement.slot("stake").value == "0.5"


def test_semantic_resolver_bad_json_goes_to_human_review() -> None:
    client = FakeSemanticLLMClient("不是 JSON")

    resolution = SemanticResolver(client).resolve(make_context())

    assert resolution.needs_human_review is True
    assert resolution.intent == UserIntent.UNKNOWN
    assert resolution.proposed_action.name == ActionName.HUMAN_REVIEW
    assert resolution.proposed_action.risk_level == RiskLevel.HIGH
    assert "single JSON object" in resolution.reasoning_summary
    assert resolution.raw_response["llm_contract"]["accepted"] is False
    assert resolution.raw_response["llm_contract"]["parse_error"] == (
        "LLM semantic resolver output must be a single JSON object with no surrounding text."
    )
    assert resolution.raw_response["llm_contract"]["raw_output"] == "不是 JSON"


def test_semantic_resolver_rejects_json_fragment_by_default() -> None:
    client = FakeSemanticLLMClient(
        '好的，解析如下：{"intent":"find_players","proposed_action":"create_game","confidence":0.9}'
    )

    resolution = SemanticResolver(client).resolve(make_context())

    assert resolution.needs_human_review is True
    assert resolution.proposed_action.name == ActionName.HUMAN_REVIEW
    assert "single JSON object" in resolution.reasoning_summary
    assert resolution.raw_response["llm_contract"]["accepted"] is False
    assert resolution.raw_response["llm_contract"]["raw_output"].startswith("好的")


def test_semantic_resolver_can_opt_into_legacy_json_fragment_extraction() -> None:
    client = FakeSemanticLLMClient(
        '好的，解析如下：{"intent":"find_players","proposed_action":"create_game","confidence":0.9}'
    )

    resolution = SemanticResolver(
        client,
        SemanticResolverConfig(allow_json_fragment_extraction=True),
    ).resolve(make_context())

    assert resolution.intent == UserIntent.FIND_PLAYERS
    assert resolution.proposed_action.name == ActionName.CREATE_GAME


def test_semantic_resolver_timeout_goes_to_human_review() -> None:
    client = FakeSemanticLLMClient(TimeoutError("model timeout"))

    resolution = SemanticResolver(client).resolve(make_context())

    assert resolution.needs_human_review is True
    assert resolution.proposed_action.name == ActionName.HUMAN_REVIEW
    assert "timeout" in resolution.reasoning_summary
