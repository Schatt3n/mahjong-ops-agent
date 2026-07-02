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
                    "confirmed": False,
                    "needs_confirmation": True,
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


def test_semantic_resolver_rejects_missing_required_action_contract() -> None:
    client = FakeSemanticLLMClient(
        {
            "intent": "inquire_existing_game",
            "confidence": 0.81,
            "reasoning_summary": "用户只是问有没有现成局。",
            "slots": {
                "stake": {
                    "value": "0.5",
                    "source": "explicit",
                    "confidence": 0.9,
                    "confirmed": True,
                    "needs_confirmation": False,
                }
            },
        }
    )

    resolution = SemanticResolver(client).resolve(make_context())

    assert resolution.needs_human_review is True
    assert resolution.intent == UserIntent.UNKNOWN
    assert resolution.proposed_action.name == ActionName.HUMAN_REVIEW
    assert "missing required field 'proposed_action'" in resolution.reasoning_summary
    assert resolution.raw_response["llm_contract"]["accepted"] is False
    assert resolution.raw_response["llm_contract"]["contract_errors"] == [
        "missing required field 'proposed_action'"
    ]


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
        (
            '好的，解析如下：{"intent":"find_players","proposed_action":"create_game",'
            '"confidence":0.9,"reasoning_summary":"用户确认要组局。","slots":{}}'
        )
    )

    resolution = SemanticResolver(
        client,
        SemanticResolverConfig(allow_json_fragment_extraction=True),
    ).resolve(make_context())

    assert resolution.intent == UserIntent.FIND_PLAYERS
    assert resolution.proposed_action.name == ActionName.CREATE_GAME


def test_semantic_resolver_accepts_reference_action_arguments() -> None:
    client = FakeSemanticLLMClient(
        {
            "intent": "candidate_reply",
            "proposed_action": "accept_seat",
            "confidence": 0.93,
            "needs_human_review": False,
            "reasoning_summary": "候选人确认加入上下文中的局。",
            "action_arguments": {"game_id": "game_001", "outbox_id": "outbox_001"},
            "slots": {},
        }
    )

    resolution = SemanticResolver(client).resolve(make_context())

    assert resolution.needs_human_review is False
    assert resolution.intent == UserIntent.CANDIDATE_REPLY
    assert resolution.proposed_action.name == ActionName.ACCEPT_SEAT
    assert resolution.proposed_action.arguments == {
        "game_id": "game_001",
        "outbox_id": "outbox_001",
    }
    assert resolution.raw_response["llm_contract"]["accepted"] is True


def test_semantic_resolver_rejects_action_arguments_that_cross_backend_boundary() -> None:
    client = FakeSemanticLLMClient(
        {
            "intent": "find_players",
            "proposed_action": "create_game",
            "confidence": 0.86,
            "needs_human_review": False,
            "reasoning_summary": "用户确认要组局，但模型自造了新局 ID。",
            "action_arguments": {"game_id": "llm_game_001"},
            "slots": {},
        }
    )

    resolution = SemanticResolver(client).resolve(make_context())

    assert resolution.needs_human_review is True
    assert resolution.proposed_action.name == ActionName.HUMAN_REVIEW
    assert resolution.raw_response["llm_contract"]["contract_errors"] == [
        "action_arguments.game_id is not allowed for create_game"
    ]


def test_semantic_resolver_rejects_state_write_action_arguments() -> None:
    client = FakeSemanticLLMClient(
        {
            "intent": "cancel_game",
            "proposed_action": "close_game",
            "confidence": 0.9,
            "needs_human_review": False,
            "reasoning_summary": "用户说这桌不打了，但模型试图指定状态。",
            "action_arguments": {
                "game_id": ["game_001"],
                "reason_code": "refund",
                "target_status": "cancelled",
            },
            "slots": {},
        }
    )

    resolution = SemanticResolver(client).resolve(make_context())

    assert resolution.needs_human_review is True
    errors = resolution.raw_response["llm_contract"]["contract_errors"]
    assert "action_arguments.target_status is not allowed for close_game" in errors
    assert "action_arguments.game_id must be a non-empty string" in errors
    assert "action_arguments.reason_code invalid 'refund'" in errors


def test_semantic_resolver_rejects_invalid_contract_types() -> None:
    client = FakeSemanticLLMClient(
        {
            "intent": "find_players",
            "proposed_action": "create_game",
            "confidence": "high",
            "needs_human_review": "no",
            "reasoning_summary": "",
            "slots": [],
            "action_arguments": [],
            "profile_observations": {},
        }
    )

    resolution = SemanticResolver(client).resolve(make_context())

    assert resolution.needs_human_review is True
    assert resolution.proposed_action.name == ActionName.HUMAN_REVIEW
    errors = resolution.raw_response["llm_contract"]["contract_errors"]
    assert "invalid confidence 'high'" in errors
    assert "reasoning_summary must be a non-empty string" in errors
    assert "slots must be an object" in errors
    assert "needs_human_review must be a boolean when provided" in errors
    assert "action_arguments must be an object when provided" in errors
    warnings = resolution.raw_response["llm_contract"]["nonfatal_contract_warnings"]
    assert "profile_observations must be an array when provided" in warnings


def test_semantic_resolver_rejects_incompatible_intent_and_action_contract() -> None:
    client = FakeSemanticLLMClient(
        {
            "intent": "find_players",
            "proposed_action": "ignore",
            "confidence": 0.82,
            "needs_human_review": False,
            "reasoning_summary": "用户想让老板帮忙组局，但模型又选择静默。",
            "slots": {},
        }
    )

    resolution = SemanticResolver(client).resolve(make_context())

    assert resolution.needs_human_review is True
    assert resolution.proposed_action.name == ActionName.HUMAN_REVIEW
    assert resolution.raw_response["llm_contract"]["contract_errors"] == [
        "proposed_action 'ignore' is incompatible with intent 'find_players'"
    ]


def test_semantic_resolver_allows_unknown_intent_to_ignore() -> None:
    client = FakeSemanticLLMClient(
        {
            "intent": "unknown",
            "proposed_action": "ignore",
            "confidence": 0.7,
            "needs_human_review": False,
            "reasoning_summary": "当前消息看不出麻将运营任务。",
            "slots": {},
        }
    )

    resolution = SemanticResolver(client).resolve(make_context())

    assert resolution.needs_human_review is False
    assert resolution.intent == UserIntent.UNKNOWN
    assert resolution.proposed_action.name == ActionName.IGNORE
    assert resolution.raw_response["llm_contract"]["accepted"] is True


def test_semantic_resolver_normalizes_duration_alias_slot() -> None:
    client = FakeSemanticLLMClient(
        {
            "intent": "find_players",
            "proposed_action": "create_game",
            "confidence": 0.95,
            "needs_human_review": False,
            "reasoning_summary": "用户补充了通宵局信息。",
            "slots": {
                "duration": {
                    "value": "overnight",
                    "source": "context",
                    "confidence": 0.95,
                    "confirmed": True,
                    "needs_confirmation": False,
                    "evidence": "用户上一轮说通宵局",
                },
                "party_size": {
                    "value": 1,
                    "source": "explicit",
                    "confidence": 0.98,
                    "confirmed": True,
                    "needs_confirmation": False,
                    "evidence": "用户说我一个人",
                },
            },
        }
    )

    resolution = SemanticResolver(client).resolve(make_context())

    assert resolution.game_requirement.slot("duration") is None
    assert resolution.game_requirement.slot("duration_mode").value == "overnight"
    assert resolution.game_requirement.slot("duration_mode").metadata["normalized_from_slot"] == "duration"
    assert "duration_mode" not in resolution.game_requirement.missing_required_slots(("duration_mode",))
    assert "party_size" not in resolution.game_requirement.missing_required_slots(("party_size",))


def test_semantic_resolver_rejects_invalid_slot_contracts() -> None:
    client = FakeSemanticLLMClient(
        {
            "intent": "find_players",
            "proposed_action": "create_game",
            "confidence": 0.86,
            "needs_human_review": False,
            "reasoning_summary": "用户确认要组局。",
            "slots": {
                "stake": "0.5",
                "smoke": {
                    "value": "",
                    "source": "explicit",
                    "confidence": 0.9,
                },
                "duration_mode": {
                    "value": "overnight",
                    "source": "guessed",
                    "confidence": 0.8,
                    "confirmed": True,
                    "needs_confirmation": False,
                    "metadata": [],
                },
                "party_size": {
                    "value": 1,
                    "source": "profile",
                    "confidence": "likely",
                    "confirmed": "yes",
                    "needs_confirmation": "no",
                },
                "start_time_mode": {
                    "value": "people_ready",
                    "source": "explicit",
                    "confidence": 0.9,
                    "confirmed": True,
                    "needs_confirmation": True,
                },
            },
        }
    )

    resolution = SemanticResolver(client).resolve(make_context())

    assert resolution.needs_human_review is True
    assert resolution.proposed_action.name == ActionName.HUMAN_REVIEW
    errors = resolution.raw_response["llm_contract"]["contract_errors"]
    assert "slot 'stake' must be an object" in errors
    assert "slot 'smoke' value must be non-empty" in errors
    assert "slot 'smoke' missing required field 'confirmed'" in errors
    assert "slot 'smoke' missing required field 'needs_confirmation'" in errors
    assert "slot 'duration_mode' invalid source 'guessed'" in errors
    assert "slot 'duration_mode' metadata must be an object when provided" in errors
    assert "slot 'party_size' invalid confidence 'likely'" in errors
    assert "slot 'party_size' confirmed must be a boolean" in errors
    assert "slot 'party_size' needs_confirmation must be a boolean" in errors
    assert "slot 'start_time_mode' confirmed and needs_confirmation are inconsistent" in errors


def test_semantic_resolver_treats_invalid_profile_observation_contracts_as_nonfatal() -> None:
    client = FakeSemanticLLMClient(
        {
            "intent": "find_players",
            "proposed_action": "create_game",
            "confidence": 0.86,
            "needs_human_review": False,
            "reasoning_summary": "用户确认要组局，同时模型输出了不合法画像观察。",
            "slots": {},
            "profile_observations": [
                "smoke any",
                {
                    "field": "private_health",
                    "value": "敏感内容",
                    "confidence": 0.8,
                    "source": "current_message",
                    "evidence": "用户没有说",
                    "risk": "low",
                },
                {
                    "field": "smoke_preference",
                    "value": "",
                    "confidence": 0.5,
                    "source": "guess",
                    "evidence": "",
                    "risk": "high",
                },
            ],
        }
    )

    resolution = SemanticResolver(client).resolve(make_context())

    assert resolution.needs_human_review is False
    assert resolution.proposed_action.name == ActionName.CREATE_GAME
    contract = resolution.raw_response["llm_contract"]
    assert contract["accepted"] is True
    warnings = contract["nonfatal_contract_warnings"]
    assert "profile_observations[0] must be an object" in warnings
    assert "profile_observations[1].field invalid 'private_health'" in warnings
    assert "profile_observations[2].value must be non-empty" in warnings
    assert "profile_observations[2].confidence below writable threshold 0.5" in warnings
    assert "profile_observations[2].source invalid 'guess'" in warnings
    assert "profile_observations[2].evidence must be non-empty" in warnings
    assert "profile_observations[2].risk invalid 'high'" in warnings


def test_semantic_resolver_accepts_profile_observation_field_aliases() -> None:
    client = FakeSemanticLLMClient(
        {
            "intent": "find_players",
            "proposed_action": "ask_clarification",
            "confidence": 0.86,
            "needs_human_review": False,
            "reasoning_summary": "用户补充了档位偏好。",
            "slots": {},
            "profile_observations": [
                {
                    "field": "stake_preferences",
                    "value": "1",
                    "confidence": 0.8,
                    "source": "current_message",
                    "evidence": "用户说1块",
                    "risk": "low",
                }
            ],
        }
    )

    resolution = SemanticResolver(client).resolve(make_context())

    assert resolution.needs_human_review is False
    assert resolution.raw_response["llm_contract"]["accepted"] is True
    assert "nonfatal_contract_warnings" not in resolution.raw_response["llm_contract"]


def test_semantic_resolver_timeout_goes_to_human_review() -> None:
    client = FakeSemanticLLMClient(TimeoutError("model timeout"))

    resolution = SemanticResolver(client).resolve(make_context())

    assert resolution.needs_human_review is True
    assert resolution.proposed_action.name == ActionName.HUMAN_REVIEW
    assert "timeout" in resolution.reasoning_summary
