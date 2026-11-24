"""以真实网关、门店库和可见证据核验仿真前置授权及失败行为。"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from threading import Event

import pytest

from mahjong_agent_runtime import SQLiteAgentStore
from mahjong_agent_runtime.models import ToolCall, using_clock

from .venue_channel import AGENT_ID, MessageEnvelope
from .venue_write_guard import VenueWriteGuard

NOW = datetime.fromisoformat("2026-11-25T18:00:00+08:00")
CUSTOMER, CONVERSATION = "mock-guest", "sandbox-dialogue"


def inquiry(text: str, *, message_id: str = "m-current", sender_id: str = CUSTOMER) -> MessageEnvelope:
    """构造前置授权判断收到的同结构原始消息。"""
    return MessageEnvelope(message_id, sender_id, text, NOW, "private", AGENT_ID)


def archived(message_id: str, text: str, *, sender: str = AGENT_ID, recipient: str = CUSTOMER,
             sequence: int = 1, conversation: str = CONVERSATION) -> dict:
    """构造已落账的私聊行，允许测试换人、换会话和时序。"""
    return {
        "message_id": message_id, "text": text, "sender_id": sender, "recipient_id": recipient,
        "kind": "private", "direction": "outbound" if sender == AGENT_ID else "inbound",
        "sequence": sequence, "conversation_id": conversation, "sent_at": NOW.isoformat(),
    }


def permission(scope: str, evidence: str, offer: str | None = None, offer_text: str | None = None) -> str:
    """使用与真实模型相同的四字段动作证据对象。"""
    return json.dumps({"permissions": [{
        "scope": scope, "evidence": evidence, "offer_message_id": offer, "offer_evidence": offer_text,
    }]}, ensure_ascii=False)


class StubAuthorizationClient:
    """只控制测试判定结果，随后仍走正式授权和业务工具。"""

    def __init__(self, payload: str | Exception):
        self.payload = payload
        self.calls = []

    def complete(self, messages, *, trace_id, timeout_seconds):
        self.calls.append(json.loads(messages[-1]["content"]))
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


def execute(gateway, call: ToolCall, *, sender_id: str = CUSTOMER, source: str = "m-current", conversation: str = CONVERSATION):
    """用与 Runtime 相同的已认证客户资料调用网关。"""
    with using_clock(lambda: NOW):
        return gateway.execute(call, trace_id="guard-test", conversation_id=conversation, sender_id=sender_id,
                               sender_name="模拟客人", step_index=1, source_message_id=source)


def new_game(organizer: str = CUSTOMER) -> ToolCall:
    """可被真实 Store 写入的新局调用。"""
    return ToolCall("create_game", {
        "organizer_id": organizer, "organizer_name": "模拟客人",
        "requirement": {"game_type": "hangzhou_mahjong", "stake": "1"},
    }, call_id="create-one")


def test_authorized_creation_is_bound_to_current_customer_and_normal_gateway_checks(tmp_path):
    """明确委托只能作用于本消息、本会话、本客户，且原生权限仍生效。"""
    model = StubAuthorizationClient(permission("organize", "麻烦新组一桌"))
    guard = VenueWriteGuard(model, tmp_path / "authorization.jsonl")
    store = SQLiteAgentStore(tmp_path / "agent.sqlite3")
    gateway = guard.guarded_gateway(store)
    message = inquiry("麻烦新组一桌。我今晚参加。")
    result = guard.authorize(message, CONVERSATION, ())
    assert result.status == "authorized" and result.permissions[0].scope == "organize"
    assert guard.authorize(message, CONVERSATION, ()) is result and len(model.calls) == 1
    assert guard.healthy()
    with pytest.raises(ValueError, match="identity or text changed"):
        guard.authorize(inquiry("另一条正文"), CONVERSATION, ())

    for options in ({"source": "later"}, {"conversation": "foreign"}, {"sender_id": "foreign"}):
        denied = execute(gateway, new_game(), **options)
        assert not denied.called and not denied.allowed and denied.call_id == "create-one"
        assert denied.result["validation_error"]["code"] == "venue_write_not_authorized"
    assert not store.games
    original_denial = execute(gateway, new_game("outside"))
    assert not original_denial.called and "authenticated sender_id" in original_denial.error
    assert original_denial.result == {}

    allowed = execute(gateway, new_game())
    assert allowed.called and allowed.allowed and len(store.games) == 1
    separately_denied = execute(gateway, ToolCall("reserve_room", {}))
    assert separately_denied.result["validation_error"]["scope"] == "reserve_room"
    assert guard.summary() == {
        "decisions": {"authorized": 1}, "authorized_scopes": {"organize": 1},
        "blocked_writes": 4, "blocked_by_tool": {"create_game": 3, "reserve_room": 1},
        "blocked_by_reason": {"missing_preflight": 2, "sender_mismatch": 1, "scope_not_authorized": 1},
        "audit_errors": 0,
    }
    records = [json.loads(line) for line in (tmp_path / "authorization.jsonl").read_text().splitlines()]
    assert records[0]["raw"] == model.payload and records[0]["input"]["current"]["text"] == message.text
    assert [row["event"] for row in records] == ["authorization"] + ["write_blocked"] * 4


@pytest.mark.parametrize(("case", "expected"), [
    ("no-permission", "no_permission"), ("unavailable", "unavailable"), ("fabricated", "invalid"),
    ("bare", "invalid"), ("wrong-audience", "invalid"), ("future-offer", "invalid"),
    ("stale-offer", "invalid"), ("actual-offer", "authorized"), ("current-customer-not-an-offer", "invalid"),
])
def test_offer_and_current_text_are_mandatory_for_bare_assent(tmp_path, case, expected):
    """只能引用实际前序送达的店家具体提议；故障不影响只读查局。"""
    message = inquiry("可以。")
    concrete_offer = "今晚七点的杭州麻将给你留一个位置，确认报名吗？"
    older = archived("old-offer", concrete_offer)
    actual = archived("actual-offer", concrete_offer, sequence=2)
    current = archived(message.message_id, message.text, sender=CUSTOMER, recipient=AGENT_ID, sequence=3)
    wrong_audience = archived("foreign-offer", "要不要代别人确定？", recipient="different-customer", sequence=2)
    wrong_conversation = archived("foreign-thread", "外人私聊秘密", conversation="other", sequence=1)
    future = archived("future", concrete_offer, sequence=4)
    customer = archived("customer-offer", concrete_offer, sender=CUSTOMER, recipient=AGENT_ID, sequence=1)
    model = StubAuthorizationClient({
        "no-permission": '{"permissions":[]}',
        "unavailable": TimeoutError("should never appear in audit"),
        "fabricated": permission("participate", "给我报名", "actual-offer", concrete_offer),
        "bare": permission("participate", "可以"),
        "wrong-audience": permission("participate", "可以", "foreign-offer", "要不要代别人确定？"),
        "future-offer": permission("participate", "可以", "future", concrete_offer),
        "stale-offer": permission("participate", "可以", "old-offer", concrete_offer),
        "actual-offer": permission("participate", "可以", "actual-offer", concrete_offer),
        "current-customer-not-an-offer": permission("participate", "可以", "customer-offer", concrete_offer),
    }[case])
    guard = VenueWriteGuard(model, tmp_path / "authorization.jsonl")
    decision = guard.authorize(message, CONVERSATION, (older, customer, wrong_conversation, wrong_audience, actual, current, future))
    assert decision.status == expected
    assert [row["message_id"] for row in model.calls[0]["history"]] == ["old-offer", "customer-offer", "actual-offer"]
    if expected == "authorized":
        assert decision.permissions[0].offer_message_id == "actual-offer"
    else:
        assert not decision.permissions
    assert guard.healthy() == (expected in {"authorized", "no_permission"})
    gateway = guard.guarded_gateway(SQLiteAgentStore(tmp_path / "agent.sqlite3"))
    query = execute(gateway, ToolCall("search_current_games", {"requirement": {}, "limit": 1}))
    assert query.called and query.allowed and query.result["matches"] == []
    confirmed_status = execute(gateway, ToolCall("record_candidate_reply", {"status": "accepted"}))
    if expected == "authorized":
        assert "scope" not in confirmed_status.result.get("validation_error", {})
    else:
        assert confirmed_status.result["validation_error"]["scope"] == "participate"
    assert "should never appear" not in (tmp_path / "authorization.jsonl").read_text()


def test_gateway_never_waits_for_preflight_or_invokes_network(tmp_path):
    """即使入站前置模型仍在等待，网关立即拒绝写入，不在工具锁里调用模型。"""
    started, allow_finish = Event(), Event()

    class InFlightClient(StubAuthorizationClient):
        def complete(self, messages, *, trace_id, timeout_seconds):
            started.set()
            assert allow_finish.wait(2)
            return super().complete(messages, trace_id=trace_id, timeout_seconds=timeout_seconds)

    body = "帮我新组一桌"
    model = InFlightClient(permission("organize", body))
    guard = VenueWriteGuard(model, tmp_path / "authorization.jsonl")
    gateway = guard.guarded_gateway(SQLiteAgentStore(tmp_path / "agent.sqlite3"))
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = pool.submit(guard.authorize, inquiry(body), CONVERSATION, ())
        assert started.wait(2)
        try:
            denial = pool.submit(execute, gateway, new_game()).result(timeout=1)
            assert not denial.called and guard.summary()["blocked_by_reason"] == {"missing_preflight": 1}
        finally:
            allow_finish.set()
        assert pending.result(timeout=2).status == "authorized"
    assert len(model.calls) == 1


def test_native_group_authorization_still_applies_without_private_preflight(tmp_path):
    """群的正式权限合同保持原样，私聊缺失前置判定时仍拒绝。"""
    model = StubAuthorizationClient('{"permissions":[]}')
    guard = VenueWriteGuard(model, tmp_path / "authorization.jsonl")
    store = SQLiteAgentStore(tmp_path / "agent.sqlite3")
    gateway = guard.guarded_gateway(store)
    assert not guard.healthy()
    group_context = "im:group:venue-group"
    wrong_actor = execute(gateway, new_game("outside"), conversation=group_context)
    assert not wrong_actor.called and "authenticated sender_id" in wrong_actor.error
    allowed = execute(gateway, new_game(), conversation=group_context)
    assert allowed.called and allowed.allowed
    unrelated_private = execute(gateway, new_game(), conversation="im:dm:venue-group:mock-guest")
    assert unrelated_private.result["validation_error"]["code"] == "venue_write_not_authorized"
    assert len(model.calls) == 0


def test_history_projection_and_audit_failures_complete_the_cached_decision(tmp_path):
    """输入或本地审计故障都关闭写入，同时重放和读操作仍可继续。"""
    model = StubAuthorizationClient(permission("organize", "帮我新组一桌"))
    msg = inquiry("帮我新组一桌")
    bad_history = VenueWriteGuard(model, tmp_path / "history.jsonl")
    failed_read = bad_history.authorize(msg, CONVERSATION, [None])
    assert failed_read.status == "unavailable" and not bad_history.healthy()
    assert bad_history.authorize(msg, CONVERSATION, []) is failed_read and not model.calls
    missing_parent = VenueWriteGuard(model, tmp_path / "missing" / "auth.jsonl")
    failed_audit = missing_parent.authorize(msg, CONVERSATION, [])
    assert failed_audit.status == "unavailable" and not missing_parent.healthy()
    assert missing_parent.authorize(msg, CONVERSATION, []) is failed_audit and len(model.calls) == 1
    assert missing_parent.summary()["audit_errors"] == 1
