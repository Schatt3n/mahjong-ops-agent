"""私聊后续承诺必须引用本人当前有效安排，离线桩只替代语义识别。"""

from datetime import datetime, timedelta, timezone
import json

import pytest

from mahjong_agent_runtime import AgentAction, InMemoryAgentStore, InMemoryTraceRecorder, SQLiteAgentStore, ToolGateway, UserMessage
from mahjong_agent_runtime.budget import TokenBudget
from mahjong_agent_runtime.domains.context_builders.builder import AgentContextBuilder
from mahjong_agent_runtime.models import WaitingDemand, WaitingDemandStatus, using_clock
from mahjong_agent_runtime.visibility import CustomerVisibleProcessor


_NOW = datetime(2026, 11, 25, 10, 35, tzinfo=timezone.utc)
_TEXT = "今晚先不排你，以后有杭麻2底的局再喊你。"
_OFFER = "要我给你留意吗，有杭麻2底的局再喊你？"
_WAITING = "我给你留意到今晚十点，有杭麻2底的局就喊你。"
_RECHECK = "我五分钟后再看看这桌，有进展跟你说。"


@pytest.fixture(params=(InMemoryAgentStore, SQLiteAgentStore))
def store(request, tmp_path):
    """同一隐私与有效期规则覆盖实时内存和 SQLite 权威记录。"""
    if request.param is InMemoryAgentStore:
        yield InMemoryAgentStore()
    else:
        instance = SQLiteAgentStore(tmp_path / "agent.sqlite3")
        try:
            yield instance
        finally:
            instance._connection.close()


def _demand(demand_id, *, sender="customer", conversation="private", expired=False, status=WaitingDemandStatus.ACTIVE):
    """准备各类干扰记录，只有本人同会话且有效的一条可以支持回复。"""
    return WaitingDemand(
        demand_id=demand_id, conversation_id=conversation, sender_id=sender, sender_name=sender, status=status,
        demand={"game_type": "hangzhou_mahjong", "stake": "2", "time_preference": "今晚十点前",
                "extra_constraints": ["不可外发的私有条目"]},
        created_at=_NOW - timedelta(hours=1),
        expires_at=(_NOW - timedelta(minutes=1) if expired else _NOW + timedelta(hours=3, minutes=25)),
    )


def _claim(quote, kind, arrangement_ref):
    """声明模型抽取的一个责任及其实际引用。"""
    return {"quote": quote, "kind": kind, "arrangement_ref": arrangement_ref}


def _audit(message, text, context, claims):
    """经正式审查入口核对每条原稿。测试不调用真实模型。"""
    class ReviewClient:
        def complete(self, messages, **_opts):
            self.payload = json.loads(messages[-1]["content"])
            item = {"item_id": "reply_to_user", "approved": True, "suggested_safe_text": text}
            if claims is not None:
                item["follow_up_commitments"] = claims
            return json.dumps({"approved": True, "needs_human": False, "reasoning_summary": "离线替身",
                               "violations": [], "item_reviews": [item]}, ensure_ascii=False)

    client = ReviewClient()
    processor = CustomerVisibleProcessor(client, InMemoryTraceRecorder(), timeout_seconds=1, review_enabled=True)
    result = processor.run_content_review(
        message=message, action=AgentAction(goal="回复", objective_status="completed", reasoning_summary="离线检查",
                                          reply_to_user=text),
        trace_id="binding", context_payload=context, turn_budget=TokenBudget(), review_scope="reply_to_user",
        review_items=[{"item_id": "reply_to_user", "recipient_id": message.sender_id, "text": text}],
    )
    return result.result, client.payload


@pytest.mark.parametrize("text,claims,expected_label", [
    (_TEXT, [_claim("以后有杭麻2底的局再喊你", "match_notification", None)], "unsupported_follow_up_commitment"),
    (_TEXT, [_claim("以后有杭麻2底的局再喊你", "match_notification", "waiting:made_up")], "unsupported_follow_up_commitment"),
    (_TEXT, None, "unverified_follow_up_commitment_assessment"),
    (_TEXT, [_claim("那我就记下了", "match_notification", None)], "unverified_follow_up_commitment_assessment"),
    (_OFFER, [], None),
])
def test_private_review_requires_exact_commitment_quote_and_real_arrangement(text, claims, expected_label):
    """原邀约拒绝、头像记忆和未支撑的审查批准都不能成为未来安排凭证。"""
    message = UserMessage("private", "customer", "牌友", "今晚去不了，下次有杭麻2底的局喊我。")
    context = {"active_games": [{"game_id": "open", "status": "forming"}],
               "turn_tool_evidence": [
                   {"name": "record_candidate_reply", "called": True, "allowed": True,
                    "result": {"recorded_status": "declined"}},
                   {"name": "record_user_memory", "called": True, "allowed": True, "result": {}},
               ]}
    result, payload = _audit(message, text, context, claims)
    assert payload["follow_up_commitment_evidence"] == []
    assert result["raw_approved"] is True
    assert result["approved"] is (expected_label is None)
    assert result["item_reviews"][0]["violations"] == ([expected_label] if expected_label else [])


def test_existing_waiting_demand_is_scoped_to_recipient_active_and_matching_type(store):
    """读权限由 Store 限制，候补能力只支持真实匹配后的通知，不能作复查证据。"""
    with using_clock(lambda: _NOW):
        for record in (
            _demand("mine"), _demand("other", sender="other"), _demand("other_room", conversation="elsewhere"),
            _demand("expired", expired=True), _demand("cancelled", status=WaitingDemandStatus.CANCELLED),
        ):
            store.insert_waiting_demand(record)
        message = UserMessage("private", "customer", "牌友", "今晚十点前有杭麻2底的局喊我。")
        builder = AgentContextBuilder(store, ToolGateway(store))
        context = builder.build(message, trace_id="own-waiting-preview").payload
        good, payload = _audit(message, _WAITING, context, [
            _claim("我给你留意到今晚十点，有杭麻2底的局就喊你", "match_notification", "waiting:mine"),
        ])
        assert good["approved"] is True
        assert [row["ref"] for row in payload["follow_up_commitment_evidence"]] == ["waiting:mine"]
        assert "不可外发的私有条目" not in json.dumps(payload["follow_up_commitment_evidence"], ensure_ascii=False)
        for wrong in ("waiting:other", "waiting:other_room", "waiting:expired", "waiting:cancelled"):
            bad, _ = _audit(message, _WAITING, context, [_claim("有杭麻2底的局就喊你", "match_notification", wrong)])
            assert bad["approved"] is False
        wrong_type, _ = _audit(message, _RECHECK, context, [
            _claim("我五分钟后再看看这桌，有进展跟你说", "finite_recheck", "waiting:mine"),
        ])
        assert wrong_type["approved"] is False


def test_same_goal_finite_recheck_does_not_prove_open_ended_match_notification():
    """同目标有效复查可以支持相应范围的反馈，不能替代候补匹配机制。"""
    with using_clock(lambda: _NOW):
        message = UserMessage("private", "customer", "牌友", "这桌五分钟后再看看？")
        context = {"task_context_window": {"task_context_id": "here"}, "active_follow_ups": [{
            "task_id": "task", "task_type": "conversation_follow_up", "conversation_id": "private",
            "subject_id": "customer", "status": "pending", "due_at": (_NOW + timedelta(minutes=5)).isoformat(),
            "payload": {"task_context_id": "here", "goal": "复查这桌人数", "waiting_for": "players",
                        "expires_at": (_NOW + timedelta(minutes=10)).isoformat()},
        }]}
        approved, _ = _audit(message, _RECHECK, context, [
            _claim("我五分钟后再看看这桌，有进展跟你说", "finite_recheck", "follow_up:task"),
        ])
        assert approved["approved"] is True
        rejected, _ = _audit(message, _TEXT, context, [
            _claim("以后有杭麻2底的局再喊你", "match_notification", "follow_up:task"),
        ])
        assert rejected["approved"] is False
