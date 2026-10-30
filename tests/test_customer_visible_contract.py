from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import importlib.util
import sys
from pathlib import Path

import pytest

from mahjong_agent_runtime.copywriting import build_customer_visible_text_generation_payload
from mahjong_agent_runtime.customer_visible_contract import (
    FORBIDDEN_CUSTOMER_SERVICE_PHRASES,
    PREFERRED_CANDIDATE_INVITE_PHRASES,
    PREFERRED_OPERATION_ACK_PHRASES,
    PREFERRED_REQUESTER_CURRENT_GAME_PHRASES,
    compact_customer_visible_text,
    customer_visible_action_claim_violations,
    customer_visible_contract_snapshot,
    customer_visible_text_contract_violations,
)
from mahjong_agent_runtime.copywriting import parse_customer_visible_text_generation
from mahjong_agent_runtime.customer_visible_review import build_reply_self_review_payload
from mahjong_agent_runtime.models import AgentAction, UserMessage


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_customer_visible_contract.py"


def load_contract_verifier_module():
    spec = importlib.util.spec_from_file_location("verify_customer_visible_contract_for_test", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_customer_visible_contract_verifier_passes_current_prompts() -> None:
    module = load_contract_verifier_module()

    assert module.verify_prompts() == []


def test_customer_visible_copywriting_payload_uses_shared_contract_terms() -> None:
    payload = build_customer_visible_text_generation_payload(
        message=UserMessage(conversation_id="c1", sender_id="zhang", sender_name="张哥", text="现在有人吗"),
        action=AgentAction(
            goal="测试客户可见话术合同",
            objective_status="completed",
            reasoning_summary="",
            reply_to_user="七点三缺一，打吗？",
            stop_reason={"can_stop": True, "why": "test", "pending_work": [], "depends_on_tool_results": False},
        ),
        items=[{"item_id": "reply_to_user", "source": "reply_to_user", "text": "七点三缺一，打吗？"}],
        context_payload={},
        generation_scope="reply_to_user",
    )

    style_contract = payload["style_quality_contract"]
    assert style_contract["forbidden_customer_service_phrases"] == list(FORBIDDEN_CUSTOMER_SERVICE_PHRASES)
    assert style_contract["preferred_short_phrases"] == [
        *PREFERRED_REQUESTER_CURRENT_GAME_PHRASES,
        *PREFERRED_CANDIDATE_INVITE_PHRASES,
        *PREFERRED_OPERATION_ACK_PHRASES,
    ]
    style_examples_text = "\n".join(item["good"] for item in payload["style_examples"])
    assert "七点三缺一，可以不？" in style_examples_text
    assert "还没有，还差俩" in style_examples_text
    assert "两个人，18.30 星月的局，371 她，打吗？" in style_examples_text
    assert payload["current_request"] == {"text": "现在有人吗", "quoted_text": ""}
    assert "Only items[].text is allowed as factual source" in payload["semantic_source_of_truth"]
    assert "current_request is only used to decide" in payload["semantic_source_of_truth"]
    assert payload["reply_relevance_contract"]["applies_when"] == "generation_scope=reply_to_user"
    assert "shortest text that fully answers current_request" in payload["reply_relevance_contract"]["rule"]
    assert "Preserving a coherent customer decision summary overrides brevity" in payload["reply_relevance_contract"]["priority"]
    assert "treat those clauses as one coherent option summary" in payload["reply_relevance_contract"]["coherent_option_summary_rule"]
    assert "tone only" in payload["style_examples_boundary"]
    assert "Never copy facts from examples" in payload["style_examples_boundary"]


def test_customer_visible_contract_snapshot_is_grouped_for_eval_and_docs() -> None:
    snapshot = customer_visible_contract_snapshot()

    assert snapshot["forbidden_customer_service_phrases"] == FORBIDDEN_CUSTOMER_SERVICE_PHRASES
    assert snapshot["preferred_requester_current_game_phrases"] == PREFERRED_REQUESTER_CURRENT_GAME_PHRASES
    assert snapshot["preferred_candidate_invite_phrases"] == PREFERRED_CANDIDATE_INVITE_PHRASES
    assert snapshot["preferred_operation_ack_phrases"] == PREFERRED_OPERATION_ACK_PHRASES


def test_customer_visible_text_contract_violations_group_shared_terms() -> None:
    violations = customer_visible_text_contract_violations("我是智能助手，已经生成草稿，asap_when_full 后等审批。")

    assert "implementation_identity_term:智能助手" in violations
    assert "internal_process_term:草稿" in violations
    assert "internal_process_term:审批" in violations
    assert "internal_enum:asap_when_full" in violations


def test_customer_visible_text_contract_normalizes_spacing_width_and_case() -> None:
    assert compact_customer_visible_text("Ａ I / 智能 助手 / asap when full") == "ai智能助手asapwhenfull"

    violations = customer_visible_text_contract_violations("我是Ａ I，不要写成智能 助手，asap when full 后等审 批。")

    assert "implementation_identity_term:AI" in violations
    assert "implementation_identity_term:智能助手" in violations
    assert "internal_process_term:审批" in violations
    assert "internal_enum:asap_when_full" in violations


def test_customer_visible_action_claim_requires_backend_delivery_evidence() -> None:
    pending_only = {
        "contact_started": False,
        "draft_statuses": ["pending_approval", "pending_approval"],
    }

    assert customer_visible_action_claim_violations("还差一个，在问了，人齐开。", pending_only) == [
        "unverified_external_action:问了"
    ]
    assert customer_visible_action_claim_violations("还差一个，我帮你问问。", pending_only) == []
    assert customer_visible_action_claim_violations(
        "还差一个，在问了，人齐开。",
        {"contact_started": True, "draft_statuses": ["sent"]},
    ) == []


@pytest.mark.parametrize("text", [
    "要我帮你问问有没有人吗？",
    "目前没有现成局，要不要我帮你问问？",
    "我帮你问问有没有人。",
])
def test_service_offer_and_future_intent_do_not_claim_contact_has_happened(text: str) -> None:
    assert customer_visible_action_claim_violations(
        text, {"contact_started": False, "draft_statuses": []}
    ) == []


@pytest.mark.parametrize("text, expected_phrase", [
    ("已经问了，有人会告诉你。", "问了"),
    ("正在问其他人。", "在问"),
    ("邀请已发送。", "已发送"),
])
def test_completed_or_ongoing_contact_still_requires_delivery_evidence(text: str, expected_phrase: str) -> None:
    assert customer_visible_action_claim_violations(
        text, {"contact_started": False, "draft_statuses": ["pending_approval"]}
    ) == [f"unverified_external_action:{expected_phrase}"]
    assert customer_visible_action_claim_violations(
        text, {"contact_started": True, "draft_statuses": ["sent"]}
    ) == []


def test_review_payload_preserves_latest_constraints_and_their_conversation_evidence() -> None:
    message = UserMessage(
        conversation_id="time-flexible-chat", sender_id="customer", sender_name="客户",
        text="1块的杭麻，时间都行", message_id="latest-message",
    )
    known_facts = {"game_type": "hangzhou_mahjong", "stake": "1", "time_preference": "any"}
    fact_assertions = [{
        "field": "time_preference", "value": "any", "role": "current",
        "source_message_id": "latest-message",
    }]
    recent_conversation = [
        {"role": "user", "content": "下午有人吗", "message_id": "earlier-message"},
        {"role": "assistant", "content": "想打多少的？"},
    ]
    earlier_query = {
        "name": "search_current_games", "called": True, "allowed": True,
        "result": {"requirement": {"stake": "1", "time_preference": "下午"}, "matches": []},
    }
    action = AgentAction(
        goal="沿用一块杭麻，按客户最新表达放宽时间并询问是否需要找人",
        objective_status="waiting_user", reasoning_summary="当前时间偏好由最新消息说明",
        objective_state={
            "known_facts": known_facts, "missing_facts": ["当前人数"],
            "fact_assertions": fact_assertions, "unrelated_debug_notes": "not-for-review",
        },
        reply_to_user="目前没有现成局，要我帮你问问有没有人吗？",
        stop_reason={"can_stop": True, "why": "等客户决定", "pending_work": [], "depends_on_tool_results": True},
    )
    payload = build_reply_self_review_payload(
        message=message, action=action,
        review_items=[{
            "item_id": "reply_to_user", "recipient_id": "customer",
            "source_text": action.reply_to_user, "text": action.reply_to_user,
        }],
        context_payload={"recent_conversation": recent_conversation, "turn_tool_evidence": [earlier_query]},
        review_scope="reply_to_user",
    )
    assert payload["current_message"]["text"] == message.text
    assert payload["recent_conversation_tail"] == recent_conversation
    assert payload["action_boundary"]["known_facts"] == known_facts
    assert payload["action_boundary"]["missing_facts"] == ["当前人数"]
    assert payload["action_boundary"]["fact_assertions"] == fact_assertions
    assert "unrelated_debug_notes" not in payload["action_boundary"]
    assert payload["turn_tool_evidence"] == [earlier_query]
    assert payload["turn_tool_evidence"][0]["result"]["requirement"]["time_preference"] == "下午"
    service_contract = payload["customer_service_review_contract"]
    assert service_contract["offer_is_not_execution"] is True
    assert service_contract["proposed_next_steps_are_not_authoritative"] is True
    assert service_contract["latest_user_constraints_override_query_filters"] is True
    assert service_contract["allow_known_condition_omission_without_game_option"] is True


@pytest.mark.parametrize("review_scope, recipient_id", [
    ("tool_calls", "candidate"), ("reply_to_user", "another-customer"),
])
def test_review_contract_does_not_allow_candidate_invitation_condition_omission(review_scope: str, recipient_id: str) -> None:
    message = UserMessage(conversation_id="c1", sender_id="customer", sender_name="客户", text="一块杭麻")
    action = AgentAction(
        goal="保留被邀约客户做决定所需的条件", objective_status="waiting_user", reasoning_summary="",
        reply_to_user="1块杭麻，人齐开，打吗？",
        stop_reason={"can_stop": True, "why": "等回应", "pending_work": [], "depends_on_tool_results": False},
    )
    payload = build_reply_self_review_payload(
        message=message, action=action,
        review_items=[{"item_id": "invitation", "recipient_id": recipient_id, "text": action.reply_to_user}],
        context_payload={}, review_scope=review_scope,
    )
    assert payload["customer_service_review_contract"]["allow_known_condition_omission_without_game_option"] is False
    assert payload["action_boundary"]["known_facts"] == {}
    assert payload["action_boundary"]["missing_facts"] == []
    assert payload["action_boundary"]["fact_assertions"] == []


def test_review_boundary_rejects_wrong_fact_container_types_without_interpreting_text() -> None:
    message = UserMessage(conversation_id="c1", sender_id="customer", sender_name="客户", text="时间都行")
    action = AgentAction(
        goal="审查载荷只接受模型事实容器的既定类型", objective_status="waiting_user", reasoning_summary="",
        objective_state={"known_facts": ["时间都行"], "missing_facts": "人数", "fact_assertions": {"stake": "1"}},
        reply_to_user="要我帮你问问吗？",
    )
    payload = build_reply_self_review_payload(
        message=message, action=action,
        review_items=[{"item_id": "reply_to_user", "recipient_id": "customer", "text": action.reply_to_user}],
        context_payload={}, review_scope="reply_to_user",
    )
    assert payload["action_boundary"]["known_facts"] == {}
    assert payload["action_boundary"]["missing_facts"] == []
    assert payload["action_boundary"]["fact_assertions"] == []


@pytest.mark.parametrize("text,goal,missing,query", [
    ("可以，帮我问问", "先问玩法", ["玩法"], None),
    ("只看全店有哪些在组局，先不找人", "用画像做全店浏览", [], {"game_type": "national_standard_mahjong"}),
    ("一致", "承接闲聊并引导客户约局", [], None),
])
def test_review_receives_original_scope_and_profile_despite_conflicting_model_interpretation(text, goal, missing, query):
    """保留原话、权威画像与实际窄查询供审查核对，不能被主模型自述替换。"""
    message = UserMessage(conversation_id="review-source", sender_id="customer", sender_name="客户", text=text)
    profile = {"customer_id": "customer", "preferred_games": ["national_standard_mahjong"]}
    evidence = [] if query is None else [{"name": "search_current_games", "called": True, "allowed": True,
        "result": {"requirement": query, "matches": [], "customer_reply_contract": {
            "evidence_scope": {"kind": "submitted_filters_only"}}}}]
    action = AgentAction(goal=goal, objective_status="waiting_user", reasoning_summary="模型解释需外部证据核对",
        objective_state={"missing_facts": missing, "reply_scope": {"requested_information": goal}}, reply_to_user="待审文本")
    payload = build_reply_self_review_payload(message=message, action=action,
        review_items=[{"item_id": "reply_to_user", "recipient_id": "customer", "text": action.reply_to_user}],
        context_payload={"sender_profile": profile, "turn_tool_evidence": evidence}, review_scope="reply_to_user")
    assert payload["current_message"]["text"] == text
    assert payload["sender_profile"] == profile
    assert payload["turn_tool_evidence"] == evidence
    contract = payload["customer_service_review_contract"]
    assert contract["model_missing_facts_are_not_authority"] is True
    assert contract["known_constraints_require_source_recheck"] is True
    assert contract["profile_does_not_override_inventory_scope"] is True
    assert contract["reply_scope_requires_actual_request_support"] is True


def _follow_up_review_payload(context, *, recipient="customer", scope="reply_to_user"):
    """构造同一客户的服务提议，单独核对复查证据的权限与目标投影。"""
    message = UserMessage(conversation_id="c1", sender_id="customer", sender_name="客户", text="现在有人吗")
    action = AgentAction(goal="回应查局", objective_status="waiting_user", reasoning_summary="", reply_to_user="要我帮你问问吗？")
    return build_reply_self_review_payload(
        message=message, action=action,
        review_items=[{"item_id": "reply_to_user", "recipient_id": recipient, "text": action.reply_to_user}],
        context_payload=context, review_scope=scope,
    )


@pytest.fixture
def current_follow_up_context(monkeypatch):
    """固定服务器时钟，让过期任务与仍有效的目标证据可确定重放。"""
    monkeypatch.setattr("mahjong_agent_runtime.customer_visible_review.now", lambda: datetime.fromisoformat("2030-01-01T12:00:00+00:00"))
    return {
        "task_context_window": {"task_context_id": "goal-1"},
        "active_follow_ups": [{
            "task_id": "follow-1", "task_type": "conversation_follow_up", "status": "pending",
            "conversation_id": "c1", "subject_id": "customer", "subject_name": "私有备注",
            "due_at": "2030-01-01T12:05:00+00:00", "aggregate_type": "conversation_goal", "aggregate_id": "goal-1",
            "idempotency_key": "内部幂等标识", "last_error": "不应投影的日志",
            "payload": {"task_context_id": "goal-1", "goal": "继续查看合适局", "waiting_for": "局况变化",
                        "expires_at": "2030-01-01T13:00:00+00:00", "private_notes": "不可复制的私有内容"},
        }],
    }


def test_review_follow_up_evidence_is_current_and_minimal(current_follow_up_context):
    payload = _follow_up_review_payload(current_follow_up_context)
    assert payload["active_follow_up_summaries"] == [{
        "task_id": "follow-1", "task_type": "conversation_follow_up", "status": "pending",
        "due_at": "2030-01-01T12:05:00+00:00", "aggregate_type": "conversation_goal", "aggregate_id": "goal-1",
        "task_context_id": "goal-1", "goal": "继续查看合适局", "waiting_for": "局况变化",
        "expires_at": "2030-01-01T13:00:00+00:00",
    }]
    assert "private_notes" not in str(payload) and "内部幂等标识" not in str(payload)
    assert payload["customer_service_review_contract"]["read_only_lookup_is_not_durable_follow_up"] is True
    assert payload["customer_service_review_contract"]["current_clock_is_not_availability_evidence"] is True


@pytest.mark.parametrize("field,value", [
    ("conversation_id", "another-conversation"), ("subject_id", "another-customer"),
    ("task_type", "game_recruitment"), ("status", "cancelled"), ("status", "completed"),
    ("task_id", ""), ("payload", []),
    ("payload.task_context_id", "old-goal"), ("payload.task_context_id", None),
    ("payload.expires_at", "2030-01-01T11:59:59+00:00"),
    ("payload.expires_at", "2030-01-01T13:00:00"), ("payload.expires_at", "invalid"),
])
def test_unrelated_or_expired_follow_up_cannot_support_current_promise(current_follow_up_context, field, value):
    context = deepcopy(current_follow_up_context)
    task = context["active_follow_ups"][0]
    if field.startswith("payload."):
        task["payload"][field.split(".", 1)[1]] = value
    else:
        task[field] = value
    assert _follow_up_review_payload(context)["active_follow_up_summaries"] == []


@pytest.mark.parametrize("recipient,scope", [("other", "reply_to_user"), ("customer", "tool_calls")])
def test_follow_up_goal_is_not_projected_into_another_recipient_review(current_follow_up_context, recipient, scope):
    assert _follow_up_review_payload(current_follow_up_context, recipient=recipient, scope=scope)["active_follow_up_summaries"] == []


def test_customer_visible_text_generation_rejects_rewrite_with_contract_terms() -> None:
    _, errors = parse_customer_visible_text_generation(
        """
        {
          "reasoning_summary": "模型误保留了内部词。",
          "item_rewrites": [
            {
              "item_id": "reply_to_user",
              "final_text": "我是智能助手，已经生成草稿，等审批。",
              "semantic_preserved": true,
              "used_facts": [],
              "withheld_facts": [],
              "style_checks": ["误判安全"],
              "change_summary": "错误改写"
            }
          ]
        }
        """,
        [{"item_id": "reply_to_user", "source": "reply_to_user", "text": "好，我帮你看看。"}],
    )

    assert any("violates customer-visible contract" in error for error in errors)
    assert any("智能助手" in error and "草稿" in error and "审批" in error for error in errors)
