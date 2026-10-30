"""用真实双 Store 工具结果验证跟进证据边界；不以模型替身冒充真实语义评测。"""

from __future__ import annotations

from copy import deepcopy
from datetime import timedelta
from pathlib import Path

import pytest

from mahjong_agent_runtime import (
    AgentAction, CustomerProfile, InMemoryAgentStore, SQLiteAgentStore, ToolCall, ToolGateway, UserMessage,
)
from mahjong_agent_runtime.customer_visible_review import build_reply_self_review_payload
from mahjong_agent_runtime.domains.context_builders.builder import AgentContextBuilder
from mahjong_agent_runtime.models import now
from mahjong_agent_runtime.services.objective_continuation import blocking_continuation
from mahjong_agent_runtime.task_context import TaskContextManager


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(params=["memory", "sqlite"])
def open_game_with_empty_candidates(request, tmp_path):
    """通过网关完成真实建局和空候选检索，保持无复查、候补和外发安排。"""
    store = InMemoryAgentStore() if request.param == "memory" else SQLiteAgentStore(tmp_path / "followup.sqlite3")
    store.upsert_customer(CustomerProfile(customer_id="requester", display_name="客户", preferred_games=["club_custom_mahjong"]))
    message = UserMessage(conversation_id="followup-chat", sender_id="requester", sender_name="客户",
                          text="帮我组个局", message_id="followup-input")
    TaskContextManager(store).prepare(message, trace_id="followup-prepare")
    store.append_user_turn(message, "followup-input")
    gateway = ToolGateway(store)
    requirement = {"game_type": "club_custom_mahjong", "stake": "1", "cap_score": 256,
                   "smoke_preference": "any", "start_time_kind": "asap_when_full"}

    def execute(name, arguments, step):
        return gateway.execute(ToolCall(name=name, arguments=arguments), trace_id=f"followup-{step}",
            conversation_id=message.conversation_id, sender_id=message.sender_id, sender_name=message.sender_name,
            step_index=step, source_message_id=message.message_id)

    created = execute("create_game", {"organizer_id": "requester", "organizer_name": "客户",
        "requirement": requirement, "known_players": [{"customer_id": "requester", "display_name": "客户", "seat_count": 1}]}, 1)
    assert created.called and created.allowed and not created.error
    searched = execute("search_customers", {"game_id": created.result["game"]["game_id"],
        "requirement": requirement, "exclude_customer_ids": ["requester"], "limit": 8}, 2)
    assert searched.called and searched.allowed and not searched.error
    assert searched.result["candidates"] == []
    yield store, gateway, message, [created, searched], execute
    if isinstance(store, SQLiteAgentStore):
        store._connection.close()


def _review_payload(fixture, text, *, recipient_id="requester", review_scope="reply_to_user"):
    """保持待审文本原样，只投影真实状态和本轮工具证据，不生成批准结论。"""
    store, gateway, message, evidence, _ = fixture
    context = AgentContextBuilder(store, gateway).build(message, trace_id="review-context", turn_tool_evidence=evidence).payload
    action = AgentAction(goal="处理当前组局委托", objective_status="completed", reasoning_summary="", reply_to_user=text)
    payload = build_reply_self_review_payload(message=message, action=action,
        review_items=[{"item_id": "reply_to_user", "recipient_id": recipient_id, "source_text": text, "text": text}],
        context_payload=context, review_scope=review_scope)
    return payload


def test_empty_candidates_may_finish_without_manufacturing_future_tasks(open_game_with_empty_candidates):
    """同步阶段可结束，存量开放局不能被合同改写成持续匹配或通知安排。"""
    store, _, _, evidence, _ = open_game_with_empty_candidates
    continuation = evidence[-1].result["continuation"]
    assert continuation["can_stop"] is True
    assert continuation["allowed_terminal_statuses"] == ["completed", "waiting_user"]
    assert continuation["pending_capabilities"] == continuation["suggested_tools"] == []
    assert "does not schedule another search or a notification" in continuation["reason"]
    assert "an open game alone is not such an arrangement" in continuation["instruction"]
    assert "short acknowledgement" in continuation["instruction"]
    assert "will keep being matched" not in str(continuation)
    for status in ["completed", "waiting_user"]:
        action = AgentAction(goal="本轮空候选后短承接", objective_status=status, reasoning_summary="", reply_to_user="好。")
        assert blocking_continuation(action, evidence) is None
    assert len(store.active_games()) == 1
    assert store.active_games()[0].remaining_seats() == 3
    assert not store.scheduled_tasks and not store.waiting_demands
    assert not store.invite_drafts and not store.outbound_message_drafts


@pytest.mark.parametrize("text", ["好。", "要我帮你留意着，有合适的叫你吗？", "行，我帮你留意着，有合适的叫你。"])
def test_open_game_only_payload_does_not_fabricate_commitment_evidence(open_game_with_empty_candidates, text):
    """短承接、服务提议和持续承诺都保持原文；语义归类由审查模型完成，不按关键词伪造任务。"""
    payload = _review_payload(open_game_with_empty_candidates, text)
    assert payload["review_items"][0]["text"] == text
    assert len(payload["active_games"]) == 1
    assert payload["active_follow_up_summaries"] == []
    assert [item["name"] for item in payload["turn_tool_evidence"]] == ["create_game", "search_customers"]
    assert payload["turn_tool_evidence"][-1]["result"]["candidates"] == []
    contract = payload["customer_service_review_contract"]
    assert contract["open_game_is_not_durable_follow_up"] is True
    assert contract["short_acknowledgement_is_not_future_notification"] is True
    assert contract["commitment_requires_matching_mechanism_scope"] is True
    assert contract["proposal_scope_is_whole_utterance"] is True
    assert contract["commitment_evidence_requires_accepted_responsibility"] is True
    assert "expected_approved" not in str(payload)
    store = open_game_with_empty_candidates[0]
    assert not store.scheduled_tasks and not store.waiting_demands
    assert not store.invite_drafts and not store.outbound_message_drafts


def test_real_follow_up_is_projected_without_claiming_delivery(open_game_with_empty_candidates):
    """真实持久复查可证明范围内的后续重评，仍不能证明已外发或无限通知。"""
    fixture = open_game_with_empty_candidates
    store, _, _, evidence, execute = fixture
    stamp = now()
    scheduled = execute("schedule_follow_up", {"goal": "有限复查本次组局是否有新进展", "waiting_for": "players",
        "due_at": (stamp + timedelta(minutes=5)).isoformat(), "expires_at": (stamp + timedelta(minutes=30)).isoformat(),
        "review_budget": 1}, 3)
    assert scheduled.called and scheduled.allowed and not scheduled.error
    evidence.append(scheduled)
    payload = _review_payload(fixture, "我过会儿再看看，有进展跟你说。")
    task = scheduled.result["follow_up"]
    assert task["task_id"] in store.scheduled_tasks
    summaries = payload["active_follow_up_summaries"]
    assert len(summaries) == 1 and summaries[0]["task_id"] == task["task_id"]
    assert summaries[0]["status"] == "pending"
    assert summaries[0]["goal"] == "有限复查本次组局是否有新进展"
    assert summaries[0]["expires_at"] == task["payload"]["expires_at"]
    assert not store.outbound_message_drafts
    capability = payload["customer_service_review_contract"]["conversation_follow_up_capability"]
    assert "按实际进展准备给同一客户的反馈草稿" in capability
    assert "无必要进展可静默" in capability
    assert "不证明已发送、保证有结果或无限跟进" in capability
    rules = payload["semantic_fidelity_contract"]["rules"]
    assert any("覆盖该范围内有进展再反馈的承接" in rule for rule in rules)
    assert any("有限安排不证明无限跟进、一定凑齐或已经外发" in rule for rule in rules)
    assert any("招募窗口任务只支持届时开启招募，不自动支持通知发起人" in rule for rule in rules)


@pytest.mark.parametrize("change", ["cancelled", "expired", "wrong_goal", "wrong_customer"])
def test_unusable_follow_up_cannot_become_current_responsibility_evidence(open_game_with_empty_candidates, monkeypatch, change):
    """取消、过期和其他目标/客户的安排不为当前持续承诺作证。"""
    fixture = open_game_with_empty_candidates
    store, gateway, message, evidence, execute = fixture
    stamp = now()
    result = execute("schedule_follow_up", {"goal": "复查本次进展", "waiting_for": "players",
        "due_at": (stamp + timedelta(minutes=5)).isoformat(), "expires_at": (stamp + timedelta(minutes=30)).isoformat()}, 3)
    assert result.called and result.allowed and not result.error
    evidence.append(result)
    if change == "cancelled":
        cancelled = execute("cancel_follow_up", {"reason": "本次不再跟进"}, 4)
        assert cancelled.called and cancelled.allowed and not cancelled.error
        evidence.append(cancelled)
    context = AgentContextBuilder(store, gateway).build(message, trace_id="followup-invalid-context", turn_tool_evidence=evidence).payload
    if change == "expired":
        monkeypatch.setattr("mahjong_agent_runtime.customer_visible_review.now", lambda: stamp + timedelta(hours=1))
    elif change in {"wrong_goal", "wrong_customer"}:
        context = deepcopy(context)
        task = context["active_follow_ups"][0]
        if change == "wrong_goal":
            task["payload"]["task_context_id"] = "another-goal"
        else:
            task["subject_id"] = "another-customer"
    action = AgentAction(goal="本次持续跟进", objective_status="completed", reasoning_summary="", reply_to_user="有进展跟你说。")
    payload = build_reply_self_review_payload(message=message, action=action,
        review_items=[{"item_id": "reply_to_user", "recipient_id": message.sender_id, "text": action.reply_to_user}],
        context_payload=context, review_scope="reply_to_user")
    assert payload["active_follow_up_summaries"] == []


def test_prompt_and_mirror_separate_acknowledgement_proposal_and_future_obligation():
    """合同明确保持提议正例，同时不把开放局或任意任务视为未来回信凭证。"""
    prompt = (ROOT / "src/mahjong_agent_runtime/prompts/agent_runtime_reply_self_review.md").read_text()
    assert "对本轮真实完成的建局、找候选人或准备邀约可以正常短承接" in prompt
    assert "这与持续关注、再次检索、后续回信是不同责任" in prompt
    assert "开放局、一次查询、已有草稿或任意其他任务都不能单独证明" in prompt
    assert "无任务证据不能反向把第一类服务提议判成第二类承诺" in prompt
    assert "服务中将要做的打听、留意、回信或通知仍属于待同意的内容" in prompt
    assert "不能因已有开放局就补上该承诺" in prompt
    assert "active_follow_up_summaries 中的有效同目标任务支持有限复查" in prompt
    assert "并可按实际进展准备给同一客户的反馈草稿" in prompt
