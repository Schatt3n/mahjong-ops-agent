"""以真实网关、门店库和可见证据核验仿真前置授权及失败行为。"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime
from hashlib import sha256
import json
from threading import Event
from unittest.mock import Mock

import pytest

from mahjong_agent_runtime import SQLiteAgentStore
from mahjong_agent_runtime.llm import LLMCompletionText
from mahjong_agent_runtime.models import ToolCall, using_clock

from . import venue_write_guard
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


def test_interest_question_cannot_authorize_creation_without_separate_new_table_request(tmp_path):
    """即使模型误判探询有无牌友为委托，门店也只有在另有开桌证据时才写入。"""
    interest = "麻烦问问有没有人打川麻，一块底。"
    explicit = "没有现成桌的话，请你组一桌，我也上。"
    model = StubAuthorizationClient(permission("organize", interest))
    guard = VenueWriteGuard(model, tmp_path / "authorization.jsonl")
    store = SQLiteAgentStore(tmp_path / "agent.sqlite3")
    gateway = guard.guarded_gateway(store)
    assert guard.authorize(inquiry(interest), CONVERSATION, ()).status == "no_permission"
    report = execute(gateway, new_game())
    assert not report.called and not store.games
    logged = json.loads(guard.log_path.read_text().splitlines()[0])
    assert logged["rejected_scopes"] == {"organize": "interest_question_without_new_table_request"}

    model.payload = permission("organize", explicit)
    assert guard.authorize(inquiry(interest + explicit, message_id="explicit"), CONVERSATION, ()).status == "authorized"
    assert execute(gateway, new_game(), source="explicit").allowed


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
    guard = VenueWriteGuard(model, tmp_path / "authorization.jsonl", repair_stale_citation=True)
    decision = guard.authorize(message, CONVERSATION, (older, customer, wrong_conversation, wrong_audience, actual, current, future))
    assert decision.status == expected
    assert [row["message_id"] for row in model.calls[0]["history"]] == ["old-offer", "customer-offer", "actual-offer"]
    assert len(model.calls) == (2 if case in {"fabricated", "stale-offer"} else 1)
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


@pytest.mark.parametrize(("reply", "status", "visible_writes"), [
    (permission("organize", "请新组一桌"), "authorized", {"create_game", "create_invite_drafts"}),
    (permission("participate", "给我留个位"), "authorized", {"join_game"}),
    (permission("reserve_room", "请订个包间"), "authorized", {"reserve_room"}),
    (permission("waitlist", "请给我登记候补"), "authorized", {"register_waiting_demand"}),
    ('{"permissions":[]}', "no_permission", set()),
    (permission("organize", "并不存在的委托"), "invalid", set()),
    (TimeoutError("authorization timed out"), "unavailable", set()),
])
def test_planner_scope_filters_only_the_dedicated_writes_supported_by_verified_evidence(tmp_path, reply, status, visible_writes):
    """模型看到本轮获准的专用工具及原有查询、拒绝和撤回入口。"""
    model = StubAuthorizationClient(reply)
    guard = VenueWriteGuard(model, tmp_path / "authorization.jsonl")
    gateway = guard.guarded_gateway(SQLiteAgentStore(tmp_path / "agent.sqlite3"))
    body = "请新组一桌，给我留个位，请订个包间，请给我登记候补。"
    authorization = guard.authorize(inquiry(body), CONVERSATION, ())
    assert authorization.status == status

    baseline = gateway.tool_specs_for_prompt()
    dedicated = {"create_game", "create_invite_drafts", "join_game", "reserve_room", "register_waiting_demand"}
    baseline_names = {spec["name"] for spec in baseline}
    assert dedicated | {"search_current_games", "record_candidate_reply", "record_registered_member_withdrawal",
                        "cancel_waiting_demand"} <= baseline_names
    with guard.planner_scope(authorization):
        assert gateway.tool_specs_for_prompt() == [spec for spec in baseline if spec["name"] not in dedicated - visible_writes]
        if "join_game" not in visible_writes:
            accepted = execute(gateway, ToolCall("record_candidate_reply", {"status": "accepted"}))
            assert accepted.result["validation_error"]["scope"] == "participate"
        if "create_game" not in visible_writes:
            created = execute(gateway, new_game())
            assert created.result["validation_error"]["scope"] == "organize"
    assert gateway.tool_specs_for_prompt() == baseline
    assert len(model.calls) == 1


def test_planner_scope_is_thread_local_and_restored_after_nested_failure(tmp_path):
    """并行回合互不共享权限，嵌套范围出错后仍恢复原有工具。"""
    body = "请新组一桌"
    model = StubAuthorizationClient(permission("organize", body))
    guard = VenueWriteGuard(model, tmp_path / "authorization.jsonl")
    store = SQLiteAgentStore(tmp_path / "agent.sqlite3")
    gateway = guard.guarded_gateway(store)
    independent_guard = VenueWriteGuard(model, tmp_path / "independent.jsonl")
    independent_gateway = independent_guard.guarded_gateway(store)
    organize = guard.authorize(inquiry(body), CONVERSATION, ())
    model.payload = permission("reserve_room", "请订个包间")
    reserve = guard.authorize(inquiry("请订个包间", message_id="reserve"), CONVERSATION, ())
    baseline = gateway.tool_specs_for_prompt()

    def worker_specs():
        before = gateway.tool_specs_for_prompt()
        with guard.planner_scope(reserve):
            inside = {spec["name"] for spec in gateway.tool_specs_for_prompt()}
        return before, inside, gateway.tool_specs_for_prompt()

    with guard.planner_scope(organize), ThreadPoolExecutor(max_workers=1) as pool:
        organizer_specs = gateway.tool_specs_for_prompt()
        assert independent_gateway.tool_specs_for_prompt() == baseline
        before, inside, after = pool.submit(worker_specs).result(timeout=2)
        assert before == after == baseline
        assert "create_game" not in inside and "reserve_room" in inside
        assert gateway.tool_specs_for_prompt() == organizer_specs
        assert {"create_game", "create_invite_drafts"} <= {spec["name"] for spec in organizer_specs}
        with pytest.raises(RuntimeError, match="abort nested scope"):
            with guard.planner_scope(reserve):
                assert {spec["name"] for spec in gateway.tool_specs_for_prompt()} == inside
                raise RuntimeError("abort nested scope")
        assert gateway.tool_specs_for_prompt() == organizer_specs
    assert gateway.tool_specs_for_prompt() == baseline


def test_native_group_authorization_still_applies_without_private_preflight(tmp_path):
    """群的正式权限合同保持原样，私聊缺失前置判定时仍拒绝。"""
    model = StubAuthorizationClient('{"permissions":[]}')
    guard = VenueWriteGuard(model, tmp_path / "authorization.jsonl")
    store = SQLiteAgentStore(tmp_path / "agent.sqlite3")
    gateway = guard.guarded_gateway(store)
    assert not guard.healthy()
    group_context = "im:group:venue-group"
    assert "create_game" in {spec["name"] for spec in gateway.tool_specs_for_prompt()}
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


def test_resume_audit_counts_prior_and_new_records_without_reusing_permission(tmp_path):
    """同一审计文件跨进程累加计数，执行时每条入站仍需本进程完成前置授权。"""
    path = tmp_path / "authorization.jsonl"
    body = "帮我新组一桌"
    previous = VenueWriteGuard(StubAuthorizationClient(permission("organize", body)), path)
    model = StubAuthorizationClient(permission("organize", body))
    resumed = VenueWriteGuard(model, path, resume_audit=True)
    assert not resumed.healthy()

    # 续跑对象可能在上一个调度器释放锁之前创建，汇总仍应看到其随后落盘的审计。
    assert previous.authorize(inquiry(body), CONVERSATION, ()).status == "authorized"
    store = SQLiteAgentStore(tmp_path / "agent.sqlite3")
    prior_denied = execute(previous.guarded_gateway(store), ToolCall("reserve_room", {}))
    assert not prior_denied.called
    assert resumed.healthy() and resumed.summary() == previous.summary()
    assert not model.calls

    gateway = resumed.guarded_gateway(store)
    denied = execute(gateway, new_game())
    assert not denied.called and not store.games
    assert resumed.authorize(inquiry(body, message_id="fresh"), CONVERSATION, ()).status == "authorized"
    assert execute(gateway, new_game(), source="fresh").allowed
    assert len(model.calls) == 1
    assert resumed.summary() == {
        "decisions": {"authorized": 2}, "authorized_scopes": {"organize": 2},
        "blocked_writes": 2, "blocked_by_tool": {"reserve_room": 1, "create_game": 1},
        "blocked_by_reason": {"scope_not_authorized": 1, "missing_preflight": 1}, "audit_errors": 0,
    }


@pytest.mark.parametrize(("damaged", "separator"), [
    ("{", ""),
    ('{"event":"authorization","status":"authorized","permissions":[{"scope":"unrecognized"}]}', "\n"),
    ('{"event":"write_blocked","tool":"create_game","reason":[]}', "\n"),
])
def test_resume_audit_invalid_row_fails_aggregate_check_without_granting_permission(tmp_path, damaged, separator):
    """旧记录破损使总验收失败，已持久化的有效计数和新的真实权限判断仍独立。"""
    path = tmp_path / "authorization.jsonl"
    body = "帮我新组一桌"
    model = StubAuthorizationClient(permission("organize", body))
    previous = VenueWriteGuard(model, path)
    assert previous.authorize(inquiry(body), CONVERSATION, ()).status == "authorized"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(damaged + separator)
    resumed = VenueWriteGuard(model, path, resume_audit=True)
    assert resumed.summary()["audit_errors"] == 1
    assert resumed.summary()["decisions"] == {"authorized": 1}
    assert not resumed.healthy()

    store = SQLiteAgentStore(tmp_path / "agent.sqlite3")
    gateway = resumed.guarded_gateway(store)
    assert not execute(gateway, new_game()).called and not store.games
    assert resumed.authorize(inquiry(body, message_id="fresh"), CONVERSATION, ()).status == "authorized"
    assert execute(gateway, new_game(), source="fresh").allowed
    assert resumed.summary()["decisions"] == {"authorized": 2}
    assert resumed.summary()["audit_errors"] == 1
    assert not resumed.healthy()


def test_resume_audit_requires_each_archived_input_exactly_once(tmp_path):
    """有一条有效审计不足以覆盖两条来话，重复旧行不能填平缺失。"""
    path = tmp_path / "authorization.jsonl"
    model = StubAuthorizationClient('{"permissions":[]}')
    previous = VenueWriteGuard(model, path)
    previous.authorize(inquiry("还有位吗"), CONVERSATION, ())
    resumed = VenueWriteGuard(model, path, resume_audit=True)
    assert resumed.healthy(expected_inputs=1)
    assert not resumed.healthy(expected_inputs=2)
    assert not resumed.healthy(expected_inputs=0)

    duplicate = path.read_text(encoding="utf-8")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(duplicate)
    assert resumed.summary()["audit_errors"] == 1
    assert resumed.summary()["decisions"] == {"no_permission": 1}
    assert not resumed.healthy(expected_inputs=1) and not resumed.healthy(expected_inputs=2)


def test_current_citation_repair_keeps_both_sources_and_shares_the_budget(tmp_path, monkeypatch):
    """旧委托不能为查进度授权，修正只读取本回合凭据并保存两次真实应答来源。"""
    earlier = "麻烦新组一桌"
    current = inquiry("现在还差几个人？")
    first_raw = permission("organize", earlier)
    final_raw = '{"permissions":[]}'
    model = Mock()
    model.complete.side_effect = [
        LLMCompletionText(first_raw, {"attempt": "first"}),
        LLMCompletionText(final_raw, {"attempt": "repair"}),
    ]
    clock = iter([100.0, 103.0])
    monkeypatch.setattr(venue_write_guard, "monotonic", lambda: next(clock))
    log = tmp_path / "authorization.jsonl"
    guard = VenueWriteGuard(model, log, timeout_seconds=10, repair_stale_citation=True)
    history = [archived("older-request", earlier, sender=CUSTOMER, recipient=AGENT_ID)]

    result = guard.authorize(current, CONVERSATION, history)

    assert result.status == "no_permission" and not result.permissions and guard.healthy()
    assert guard.authorize(current, CONVERSATION, history) is result
    first, repair = model.complete.call_args_list
    assert [first.kwargs["timeout_seconds"], repair.kwargs["timeout_seconds"]] == [10, 7]
    initial_input = json.loads(first.args[0][-1]["content"])
    repair_input = json.loads(repair.args[0][-1]["content"])
    assert list(initial_input) == ["history", "current"]
    assert list(repair_input) == ["rejected_output", "history", "current"]
    assert repair_input == {"rejected_output": first_raw, **initial_input}
    audit = json.loads(log.read_text())
    prior, = audit["prior_attempts"]
    for index, row, raw in [(1, prior, first_raw), (2, audit, final_raw)]:
        assert row["raw"] == raw
        assert row["response_source"]["step_index"] == index
        assert row["response_source"]["raw_response_sha256"] == sha256(raw.encode()).hexdigest()
    assert prior["diagnostics"] == {"attempt": "first"} and audit["diagnostics"] == {"attempt": "repair"}
    gateway = guard.guarded_gateway(SQLiteAgentStore(tmp_path / "agent.sqlite3"))
    assert not execute(gateway, new_game()).allowed


def test_repaired_current_citation_can_preserve_the_originally_proposed_scope(tmp_path):
    """只有本轮确有同一请求，修正后的授权才能送入原生网关。"""
    current = "今晚帮我新组一桌。"
    model = Mock()
    model.complete.side_effect = [permission("organize", "上周找人组一桌"), permission("organize", current)]
    guard = VenueWriteGuard(model, tmp_path / "authorization.jsonl", repair_stale_citation=True)
    result = guard.authorize(inquiry(current), CONVERSATION, ())
    assert result.status == "authorized" and result.permissions[0].evidence == current and guard.healthy()
    store = SQLiteAgentStore(tmp_path / "agent.sqlite3")
    assert execute(guard.guarded_gateway(store), new_game()).allowed and len(store.games) == 1


@pytest.mark.parametrize(("current", "repair_raw", "expected"), [
    ("正好，那就给我留一个，我晚点过来。", permission("participate", "那就给我留一个"), "authorized"),
    ("好。", permission("participate", "好。"), "invalid"),
    ("好。", '{"permissions":[]}', "no_permission"),
    ("给我留一个，也帮我订个包间。", permission("reserve_room", "也帮我订个包间。"), "invalid"),
])
def test_visible_offer_citation_repair_requires_current_consent_or_a_valid_offer(tmp_path, current, repair_raw, expected):
    """店家后来只答桌况时，明确留位可凭当前请求补判，简答仍不能丢弃具体提议。"""
    offer = "七点的杭麻目前三缺一，你打吗？"
    history = [archived("earlier-offer", offer), archived("latest-reply", "七点打到十点。", sequence=2)]
    model = Mock()
    model.complete.side_effect = [permission("participate", current, "earlier-offer", offer), repair_raw]
    log = tmp_path / "authorization.jsonl"
    guard = VenueWriteGuard(model, log, repair_stale_citation=True)

    result = guard.authorize(inquiry(current), CONVERSATION, history)

    assert result.status == expected
    assert model.complete.call_count == 2
    audit = json.loads(log.read_text())
    prior, = audit["prior_attempts"]
    assert prior["error"] == "offer evidence is not current, exact and customer-visible"
    if expected == "authorized":
        assert result.permissions == (venue_write_guard.WritePermission("participate", "那就给我留一个", None, None),)
    else:
        assert not result.permissions


def test_repair_can_restore_exact_text_for_the_explicitly_quoted_offer(tmp_path):
    """有渠道引用时允许用户接受相应前文，但引文必须和实际送达的文字相符。"""
    offer = "七点的杭麻目前三缺一，你打吗？"
    message = replace(inquiry("好。"), quoted_message_id="earlier-offer")
    history = [archived("earlier-offer", offer), archived("latest-reply", "七点打到十点。", sequence=2)]
    model = Mock()
    model.complete.side_effect = [permission("participate", message.text, "earlier-offer", "你要打吗？"),
                                  permission("participate", message.text, "earlier-offer", offer)]
    guard = VenueWriteGuard(model, tmp_path / "authorization.jsonl", repair_stale_citation=True)

    result = guard.authorize(message, CONVERSATION, history)

    assert result.status == "authorized" and model.complete.call_count == 2
    assert result.permissions == (venue_write_guard.WritePermission("participate", message.text, "earlier-offer", offer),)


@pytest.mark.parametrize(("enable", "first_raw", "remaining_clock"), [
    (False, permission("organize", "历史消息要求开一桌"), 101.0),
    (True, permission("organize", "历史消息要求开一桌"), 110.0),
    (True, permission("organize", "历史消息要求开一桌", "foreign-offer", "猜的提议"), 101.0),
    (True, permission("invalid-scope", "历史消息要求开一桌"), 101.0),
    (True, "not-json", 101.0),
])
def test_citation_repair_does_not_retry_other_failures_or_exhausted_budget(tmp_path, monkeypatch, enable, first_raw, remaining_clock):
    """仅可见范围内的错引可以补判，预算用尽或其他合同错误仍拒绝。"""
    model = Mock()
    model.complete.return_value = first_raw
    clock = iter([100.0, remaining_clock])
    monkeypatch.setattr(venue_write_guard, "monotonic", lambda: next(clock))
    log = tmp_path / "authorization.jsonl"
    guard = VenueWriteGuard(model, log, timeout_seconds=10, repair_stale_citation=enable)
    assert guard.authorize(inquiry("现在还差几个人？"), CONVERSATION, ()).status == "invalid"
    assert model.complete.call_count == 1 and "prior_attempts" not in json.loads(log.read_text())


@pytest.mark.parametrize("repair_raw", [
    permission("organize", "历史消息要求开一桌"),
    permission("reserve_room", "帮我订个包间"),
    TimeoutError("不应记录供应商异常正文"),
])
def test_unsafe_citation_repair_stays_invalid_and_never_retries_twice(tmp_path, repair_raw):
    """再次引用旧消息、扩展授权范围或重试超时，都保留无效判定。"""
    model = Mock()
    model.complete.side_effect = [permission("organize", "历史消息要求开一桌"), repair_raw]
    log = tmp_path / "authorization.jsonl"
    guard = VenueWriteGuard(model, log, repair_stale_citation=True)
    result = guard.authorize(inquiry("帮我订个包间"), CONVERSATION, ())
    assert result.status == "invalid" and not result.permissions and not guard.healthy()
    assert model.complete.call_count == 2
    audit = json.loads(log.read_text())
    assert len(audit["prior_attempts"]) == 1 and audit["status"] == "invalid"
    assert not execute(guard.guarded_gateway(SQLiteAgentStore(tmp_path / "agent.sqlite3")), new_game()).allowed
    assert "不应记录供应商异常正文" not in log.read_text()
