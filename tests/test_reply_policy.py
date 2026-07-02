from __future__ import annotations

import json

from mahjong_agent.reply_guard import ReplyGuard
from mahjong_agent.reply_policy import ReplyPolicy, ReplyPolicyConfig
from mahjong_agent.tool_orchestrator import ToolOrchestrationResult
from mahjong_agent.workflow_models import (
    ActionName,
    ActionSource,
    ConversationContext,
    ProposedAction,
    ReplyDraft,
    ReplyStatus,
    RiskLevel,
    SemanticResolution,
    StateTransition,
    ToolCallRequest,
    ToolExecutionMode,
    ToolName,
    ToolResult,
    UserIntent,
    UserMessage,
    ValidatedAction,
)


class FixedReplyLLMClient:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def complete(self, messages, *, trace_id: str, timeout_seconds: float):
        self.calls.append(
            {
                "messages": messages,
                "trace_id": trace_id,
                "timeout_seconds": timeout_seconds,
            }
        )
        return self.outputs.pop(0)


def make_context() -> ConversationContext:
    return ConversationContext(
        current_message=UserMessage(
            text="帮我组一桌",
            sender_id="zhang",
            sender_name="张哥",
            conversation_id="group_a",
            trace_id="trace_reply",
            message_id="msg_reply",
        )
    )


def make_resolution() -> SemanticResolution:
    return SemanticResolution(
        intent=UserIntent.FIND_PLAYERS,
        proposed_action=ProposedAction(
            name=ActionName.CREATE_GAME,
            source=ActionSource.LLM,
            confidence=0.9,
            reason="用户明确组局",
        ),
    )


def make_validated(
    action: ActionName,
    *,
    missing_slots: list[str] | None = None,
    risk_level: RiskLevel = RiskLevel.LOW,
    allowed: bool = True,
) -> ValidatedAction:
    return ValidatedAction(
        proposed_action=ProposedAction(
            name=action,
            source=ActionSource.LLM,
            confidence=0.9,
            reason="test",
            risk_level=risk_level,
        ),
        effective_action=action,
        allowed=allowed,
        code="test_code",
        reason="test reason",
        missing_slots=missing_slots or [],
        risk_level=risk_level,
    )


def tool_result(
    tool_name: ToolName,
    result: dict,
    *,
    called: bool = True,
    allowed: bool = True,
) -> ToolResult:
    return ToolResult(
        request=ToolCallRequest(
            tool_name=tool_name,
            execution_mode=ToolExecutionMode.CREATE_PENDING
            if tool_name == ToolName.CREATE_PENDING_OUTBOX
            else ToolExecutionMode.READ_ONLY,
        ),
        called=called,
        allowed=allowed,
        result=result,
    )


def test_reply_policy_queues_invite_only_after_outbox_created() -> None:
    orchestration = ToolOrchestrationResult(
        tool_results=[
            tool_result(
                ToolName.CREATE_PENDING_OUTBOX,
                {"drafts": [{"message_text": "冉姐，16:00，0.5无烟，打吗？"}]},
            )
        ]
    )

    draft = ReplyPolicy().draft(
        context=make_context(),
        semantic_resolution=make_resolution(),
        validated_action=make_validated(ActionName.QUEUE_INVITES, risk_level=RiskLevel.MEDIUM),
        tool_result=orchestration,
    )
    guarded = ReplyGuard().guard(
        draft=draft,
        validated_action=make_validated(ActionName.QUEUE_INVITES, risk_level=RiskLevel.MEDIUM),
        tool_result=orchestration,
    )

    assert draft.text == "好的，我帮你问问。"
    assert guarded.changed is False
    assert guarded.final_text == "好的，我帮你问问。"


def test_reply_guard_blocks_invite_promise_without_outbox() -> None:
    draft = ReplyDraft(text="好的，我帮你问问。", risk_level=RiskLevel.MEDIUM)
    orchestration = ToolOrchestrationResult(tool_results=[])
    validated = make_validated(ActionName.QUEUE_INVITES, risk_level=RiskLevel.MEDIUM)

    guarded = ReplyGuard().guard(draft=draft, validated_action=validated, tool_result=orchestration)

    assert guarded.changed is True
    assert guarded.final_text == "我先确认一下。"
    assert "不能承诺" in guarded.guard_reasons[0]


def test_reply_policy_existing_game_uses_search_tool_result() -> None:
    orchestration = ToolOrchestrationResult(
        tool_results=[
            tool_result(
                ToolName.SEARCH_CURRENT_OPEN_GAMES,
                {"matches": [{"summary": "18:00 0.5无烟 三缺一"}]},
            )
        ]
    )

    draft = ReplyPolicy().draft(
        context=make_context(),
        semantic_resolution=make_resolution(),
        validated_action=make_validated(ActionName.MATCH_EXISTING_GAME),
        tool_result=orchestration,
    )

    assert draft.text == "18:00 0.5无烟 三缺一，要不要加？"


def test_reply_policy_clarification_asks_missing_slots_only() -> None:
    draft = ReplyPolicy().draft(
        context=make_context(),
        semantic_resolution=make_resolution(),
        validated_action=make_validated(
            ActionName.ASK_CLARIFICATION,
            missing_slots=["stake", "party_size", "duration_mode", "smoke"],
            allowed=False,
        ),
        tool_result=ToolOrchestrationResult(),
    )

    assert draft.text == "打多大？ 你这边几个人？ 大概要打多久？"
    assert "烟况" not in draft.text


def test_reply_guard_does_not_generate_business_clarification_text() -> None:
    guarded = ReplyGuard().guard(
        draft=ReplyDraft(text="现在没有合适的，要组一个吗？"),
        validated_action=make_validated(
            ActionName.ASK_CLARIFICATION,
            missing_slots=["stake", "party_size"],
            allowed=False,
        ),
        tool_result=ToolOrchestrationResult(),
    )

    assert guarded.changed is False
    assert guarded.final_text == "现在没有合适的，要组一个吗？"


def test_reply_guard_replaces_room_promise() -> None:
    guarded = ReplyGuard().guard(
        draft=ReplyDraft(text="好的，我给你留着。"),
        validated_action=make_validated(ActionName.ACCEPT_SEAT),
        tool_result=ToolOrchestrationResult(),
    )

    assert guarded.changed is True
    assert guarded.final_text == "我先确认一下房间情况。"


def test_reply_guard_high_risk_goes_to_human_review() -> None:
    guarded = ReplyGuard().guard(
        draft=ReplyDraft(text="可以，我处理。", risk_level=RiskLevel.HIGH),
        validated_action=make_validated(ActionName.HUMAN_REVIEW, risk_level=RiskLevel.HIGH, allowed=False),
        tool_result=ToolOrchestrationResult(),
    )

    assert guarded.changed is True
    assert guarded.final_text == "这个我先转人工确认一下。"


def test_reply_policy_ask_create_confirmation_is_not_invite_promise() -> None:
    draft = ReplyPolicy().draft(
        context=make_context(),
        semantic_resolution=make_resolution(),
        validated_action=make_validated(ActionName.ASK_CREATE_CONFIRMATION),
        tool_result=ToolOrchestrationResult(),
    )

    assert draft.text == "现在没有合适的，要组一个吗？"


def test_reply_policy_accept_seat_uses_state_transition_seat_delta() -> None:
    transition = StateTransition(
        entity_type="game",
        entity_id="game_accept_001",
        from_status="negotiating",
        to_status="negotiating",
        reason="候选人确认加入",
        allowed=True,
        metadata={
            "seat_delta": {
                "previous_current_player_count": 1,
                "previous_missing_count": 3,
                "current_player_count": 2,
                "missing_count": 2,
                "seats_total": 4,
            }
        },
    )

    draft = ReplyPolicy().draft(
        context=make_context(),
        semantic_resolution=make_resolution(),
        validated_action=make_validated(ActionName.ACCEPT_SEAT, risk_level=RiskLevel.MEDIUM),
        tool_result=ToolOrchestrationResult(),
        state_transitions=[transition],
    )

    assert draft.text == "好的，加你272了。"
    assert draft.metadata["state_transitions"][0]["metadata"]["seat_delta"]["missing_count"] == 2


def test_reply_policy_accept_seat_says_full_when_missing_count_zero() -> None:
    transition = StateTransition(
        entity_type="game",
        entity_id="game_accept_002",
        from_status="negotiating",
        to_status="confirmed",
        reason="最后一位候选人确认加入",
        allowed=True,
        metadata={"seat_delta": {"current_player_count": 4, "missing_count": 0, "seats_total": 4}},
    )

    draft = ReplyPolicy().draft(
        context=make_context(),
        semantic_resolution=make_resolution(),
        validated_action=make_validated(ActionName.ACCEPT_SEAT, risk_level=RiskLevel.MEDIUM),
        tool_result=ToolOrchestrationResult(),
        state_transitions=[transition],
    )

    assert draft.text == "好的，加你了，人齐了。"


def test_reply_policy_can_use_llm_contract_after_tool_results() -> None:
    client = FixedReplyLLMClient(
        [
            {
                "text": "好，我来问问。",
                "reasoning_summary": "后端已创建待审批邀约草稿。",
                "risk_level": "medium",
            }
        ]
    )
    orchestration = ToolOrchestrationResult(
        tool_results=[
            tool_result(
                ToolName.CREATE_PENDING_OUTBOX,
                {"drafts": [{"message_text": "冉姐，16:00，0.5无烟，打吗？"}]},
            )
        ]
    )

    draft = ReplyPolicy(client).draft(
        context=make_context(),
        semantic_resolution=make_resolution(),
        validated_action=make_validated(ActionName.QUEUE_INVITES, risk_level=RiskLevel.MEDIUM),
        tool_result=orchestration,
    )

    assert draft.source == ActionSource.LLM
    assert draft.text == "好，我来问问。"
    assert draft.metadata["schema"] == "reply_draft_contract_v1"
    assert draft.metadata["llm_contract"]["accepted"] is True
    assert draft.metadata["llm_contract"]["strict_json"] is True
    assert draft.metadata["llm_contract"]["raw_output"]["text"] == "好，我来问问。"
    payload = json.loads(client.calls[0]["messages"][1]["content"])
    assert payload["task"] == "reply_draft_contract_v1"
    assert payload["input"]["validated_action"]["effective_action"] == "queue_invites"
    assert payload["input"]["tool_results"][0]["tool_name"] == "create_pending_outbox"
    assert payload["input"]["tool_results"][0]["result"]["drafts"][0]["message_text"] == "冉姐，16:00，0.5无烟，打吗？"


def test_reply_policy_llm_prompt_includes_state_transition_metadata() -> None:
    client = FixedReplyLLMClient(
        [
            {
                "text": "好的，加你272了。",
                "reasoning_summary": "候选人确认后还缺两人。",
                "risk_level": "medium",
            }
        ]
    )
    transition = StateTransition(
        entity_type="game",
        entity_id="game_accept_003",
        from_status="negotiating",
        to_status="negotiating",
        reason="候选人确认加入",
        allowed=True,
        metadata={
            "participant": {"customer_id": "ran", "display_name": "冉姐"},
            "seat_delta": {"current_player_count": 2, "missing_count": 2, "seats_total": 4},
        },
    )

    draft = ReplyPolicy(client).draft(
        context=make_context(),
        semantic_resolution=make_resolution(),
        validated_action=make_validated(ActionName.ACCEPT_SEAT, risk_level=RiskLevel.MEDIUM),
        tool_result=ToolOrchestrationResult(),
        state_transitions=[transition],
    )

    payload = json.loads(client.calls[0]["messages"][1]["content"])
    state_payload = payload["input"]["state_transitions"][0]
    assert draft.text == "好的，加你272了。"
    assert state_payload["metadata"]["seat_delta"]["current_player_count"] == 2
    assert state_payload["metadata"]["participant"]["customer_id"] == "ran"


def test_reply_policy_falls_back_when_llm_contract_is_invalid() -> None:
    client = FixedReplyLLMClient(["不是 JSON"])
    orchestration = ToolOrchestrationResult(
        tool_results=[
            tool_result(
                ToolName.CREATE_PENDING_OUTBOX,
                {"drafts": [{"message_text": "冉姐，16:00，0.5无烟，打吗？"}]},
            )
        ]
    )

    draft = ReplyPolicy(client).draft(
        context=make_context(),
        semantic_resolution=make_resolution(),
        validated_action=make_validated(ActionName.QUEUE_INVITES, risk_level=RiskLevel.MEDIUM),
        tool_result=orchestration,
    )

    assert draft.source == ActionSource.RULES
    assert draft.text == "好的，我帮你问问。"
    assert draft.metadata["llm_contract"]["accepted"] is False
    assert draft.metadata["llm_contract"]["parse_error"] == (
        "reply draft LLM output must be a single JSON object with no surrounding text."
    )
    assert draft.metadata["llm_contract"]["raw_output"] == "不是 JSON"


def test_reply_policy_falls_back_when_llm_contract_missing_required_fields() -> None:
    client = FixedReplyLLMClient(
        [
            {
                "text": "好，我来问问。",
                "risk_level": "medium",
            }
        ]
    )
    orchestration = ToolOrchestrationResult(
        tool_results=[
            tool_result(
                ToolName.CREATE_PENDING_OUTBOX,
                {"drafts": [{"message_text": "冉姐，16:00，0.5无烟，打吗？"}]},
            )
        ]
    )

    draft = ReplyPolicy(client).draft(
        context=make_context(),
        semantic_resolution=make_resolution(),
        validated_action=make_validated(ActionName.QUEUE_INVITES, risk_level=RiskLevel.MEDIUM),
        tool_result=orchestration,
    )

    assert draft.source == ActionSource.RULES
    assert draft.text == "好的，我帮你问问。"
    assert draft.metadata["llm_contract"]["accepted"] is False
    assert draft.metadata["llm_contract"]["contract_errors"] == [
        "missing required field 'reasoning_summary'"
    ]


def test_reply_policy_rejects_invite_promise_without_pending_outbox() -> None:
    client = FixedReplyLLMClient(
        [
            {
                "text": "好的，我帮你问问。",
                "reasoning_summary": "用户确认组局。",
                "risk_level": "low",
            }
        ]
    )

    draft = ReplyPolicy(client).draft(
        context=make_context(),
        semantic_resolution=make_resolution(),
        validated_action=make_validated(
            ActionName.ASK_CLARIFICATION,
            missing_slots=["stake", "party_size", "smoke"],
        ),
        tool_result=ToolOrchestrationResult(),
    )

    assert draft.source == ActionSource.RULES
    assert draft.text == "打多大？ 你这边几个人？ 烟况有要求吗？"
    assert draft.metadata["llm_contract"]["accepted"] is False
    assert draft.metadata["llm_contract"]["contract_errors"] == [
        "reply promises inviting players before create_pending_outbox succeeded"
    ]


def test_reply_policy_accepts_empty_text_when_contract_is_complete() -> None:
    client = FixedReplyLLMClient(
        [
            {
                "text": "",
                "reasoning_summary": "本轮无需回复。",
                "risk_level": "low",
            }
        ]
    )

    draft = ReplyPolicy(client).draft(
        context=make_context(),
        semantic_resolution=make_resolution(),
        validated_action=make_validated(ActionName.IGNORE, risk_level=RiskLevel.LOW),
        tool_result=ToolOrchestrationResult(),
    )

    assert draft.source == ActionSource.LLM
    assert draft.text == ""
    assert draft.status == ReplyStatus.DRAFT
    assert draft.metadata["llm_contract"]["accepted"] is True


def test_reply_policy_falls_back_when_llm_contract_has_invalid_types() -> None:
    client = FixedReplyLLMClient(
        [
            {
                "text": ["好"],
                "reasoning_summary": "",
                "risk_level": "urgent",
            }
        ]
    )

    draft = ReplyPolicy(client).draft(
        context=make_context(),
        semantic_resolution=make_resolution(),
        validated_action=make_validated(ActionName.ASK_CREATE_CONFIRMATION, risk_level=RiskLevel.LOW),
        tool_result=ToolOrchestrationResult(),
    )

    assert draft.source == ActionSource.RULES
    errors = draft.metadata["llm_contract"]["contract_errors"]
    assert "text must be a string" in errors
    assert "reasoning_summary must be a non-empty string" in errors
    assert "invalid risk_level 'urgent'" in errors


def test_reply_policy_rejects_json_fragment_by_default() -> None:
    client = FixedReplyLLMClient(['建议如下：{"text":"好，我来问问。","risk_level":"medium"}'])
    orchestration = ToolOrchestrationResult(
        tool_results=[
            tool_result(
                ToolName.CREATE_PENDING_OUTBOX,
                {"drafts": [{"message_text": "冉姐，16:00，0.5无烟，打吗？"}]},
            )
        ]
    )

    draft = ReplyPolicy(client).draft(
        context=make_context(),
        semantic_resolution=make_resolution(),
        validated_action=make_validated(ActionName.QUEUE_INVITES, risk_level=RiskLevel.MEDIUM),
        tool_result=orchestration,
    )

    assert draft.source == ActionSource.RULES
    assert draft.text == "好的，我帮你问问。"
    assert draft.metadata["llm_contract"]["accepted"] is False
    assert draft.metadata["llm_contract"]["raw_output"].startswith("建议如下")


def test_reply_policy_can_opt_into_legacy_json_fragment_extraction() -> None:
    client = FixedReplyLLMClient(
        [
            (
                '建议如下：{"text":"好，我来问问。","reasoning_summary":"已创建待审批草稿。",'
                '"risk_level":"medium"}'
            )
        ]
    )
    orchestration = ToolOrchestrationResult(
        tool_results=[
            tool_result(
                ToolName.CREATE_PENDING_OUTBOX,
                {"drafts": [{"message_text": "冉姐，16:00，0.5无烟，打吗？"}]},
            )
        ]
    )

    draft = ReplyPolicy(
        client,
        ReplyPolicyConfig(allow_json_fragment_extraction=True),
    ).draft(
        context=make_context(),
        semantic_resolution=make_resolution(),
        validated_action=make_validated(ActionName.QUEUE_INVITES, risk_level=RiskLevel.MEDIUM),
        tool_result=orchestration,
    )

    assert draft.source == ActionSource.LLM
    assert draft.text == "好，我来问问。"
