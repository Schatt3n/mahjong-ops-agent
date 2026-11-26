"""以真实网关、门店库和可见证据核验仿真前置授权及失败行为。"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime
from hashlib import sha256
import json
from threading import Event
from unittest.mock import Mock

import pytest

from mahjong_agent_runtime import AgentAction, AgentRuntime, InMemoryTraceRecorder, SQLiteAgentStore, StaticAgentClient
from mahjong_agent_runtime.domains.context_builders.builder import read_main_agent_system_contract
from mahjong_agent_runtime.llm import LLMCompletionText
from mahjong_agent_runtime.models import ToolCall, using_clock

from . import venue_write_guard
from .venue_channel import AGENT_ID, MessageEnvelope, SandboxChannel
from .venue_fixture import _call, _reply
from .venue_models import configure_venue_preflight_prompts
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


@pytest.mark.parametrize(("body", "scope_evidence"), [
    ("老板 凑一桌杭州麻将 1块档 我七点半后能到 帮我留意下 有信回我", "凑一桌杭州麻将 1块档"),
    ("老板 那帮我问问吧 杭州麻将1块档 我七点半后能到 有合适的说一声", None),
])
def test_current_direct_table_request_keeps_its_scope_after_a_read_only_offer(tmp_path, body, scope_evidence):
    """上一轮只提议查询时，本轮仍以独立委托组桌的原话作为写入证据。"""
    model = Mock()
    model.complete.return_value = permission("organize", scope_evidence) if scope_evidence else '{"permissions":[]}'
    guard = VenueWriteGuard(model, tmp_path / "authorization.jsonl")
    store = SQLiteAgentStore(tmp_path / "agent.sqlite3")
    history = [archived("interest-offer", "要我帮你问问有没有人凑一桌不？")]

    decision = guard.authorize(inquiry(body), CONVERSATION, history)
    messages = model.complete.call_args.args[0]
    assert "先判断当前原话是否独立委托组桌" in messages[0]["content"]
    assert json.loads(messages[1]["content"])["history"][0]["message_id"] == "interest-offer"
    assert decision.verified_scopes == (frozenset({"organize"}) if scope_evidence else frozenset())
    if scope_evidence:
        assert decision.permissions[0].evidence == scope_evidence
        assert decision.permissions[0].offer_message_id is None
    result = execute(guard.guarded_gateway(store), new_game())
    assert result.called is bool(scope_evidence)
    assert bool(store.games) is bool(scope_evidence)


def test_conditional_arrival_without_table_agreement_cannot_claim_a_seat(tmp_path):
    """同一桌的待确认开场不能代替顾客报名的前提，直接留位仍走普通授权。"""
    model = Mock()
    model.complete.return_value = '{"permissions":[]}'
    guard = VenueWriteGuard(model, tmp_path / "authorization.jsonl")
    store = SQLiteAgentStore(tmp_path / "agent.sqlite3")
    with using_clock(lambda: NOW):
        game, _ = store.create_game(
            conversation_id=CONVERSATION, organizer_id="organizer", organizer_name="发起人",
            requirement={"game_type": "hangzhou_mahjong", "stake": "1"},
            known_players=[{"customer_id": "organizer", "display_name": "发起人"}],
            trace_id="existing-table",
        )
    gateway = guard.guarded_gateway(store)
    offer = archived("scheduled-table", "七点半开的桌，八点多到得看大家等不等。")
    conditional = "我八点零九能到，八点一刻前能坐就给我留位，太久别让他们等。"
    assert guard.authorize(inquiry(conditional), CONVERSATION, [offer]).status == "no_permission"
    assert "实际可见前文必须肯定" in model.complete.call_args.args[0][0]["content"]
    for name, extra in (("join_game", {}), ("record_candidate_reply", {"status": "accepted"})):
        denied = execute(gateway, ToolCall(name, {"game_id": game.game_id, "customer_id": CUSTOMER, **extra}))
        assert not denied.called and denied.result["validation_error"]["scope"] == "participate"
    assert [row.customer_id for row in store.require_game(game.game_id).participants] == ["organizer"]

    direct = "给我留一位。我想顺便问一下几点散？"
    model.complete.return_value = permission("participate", "给我留一位")
    assert guard.authorize(inquiry(direct, message_id="direct"), CONVERSATION, [offer]).status == "authorized"
    accepted = execute(gateway, ToolCall("join_game", {
        "game_id": game.game_id, "customer_id": CUSTOMER, "display_name": "模拟客人",
    }), source="direct")
    assert accepted.called and accepted.allowed
    assert any(row.customer_id == CUSTOMER for row in store.require_game(game.game_id).participants)


@pytest.mark.parametrize(("signup_text", "expected"), [
    ("我想报名，可以留我一个吗？", True),
    ("我想先问下，现在还有位置、还能报名吗？", False),
    ("我只能打到十点，能提前走的话麻烦给我留一个。", False),
])
def test_private_request_about_visible_board_distinguishes_polite_signup_from_inquiry(tmp_path, signup_text, expected):
    """首次私聊的礼貌报名以本人原话授权，群看板不成为私聊里已送达的留位提议。"""
    client = Mock()
    client.complete.return_value = permission("participate", signup_text) if expected else '{"permissions":[]}'
    guard = VenueWriteGuard(client, tmp_path / "authorization.jsonl")
    store = SQLiteAgentStore(tmp_path / "agent.sqlite3")
    board = dict(archived("published-board", '当前缺人局：\n1、2026-11-25 19:00 杭麻 1 无烟 173\n\n回复编号即可认领，如"2来"'),
                 kind="group", conversation_id="other-group")
    text = "老板好，看到群里晚上七点那桌杭州麻将，底注一块、无烟、打四个小时，" + signup_text
    decision = guard.authorize(inquiry(text), CONVERSATION, [board])
    prompt = client.complete.call_args.args[0]
    assert "先分清礼貌请求和询问可能性" in prompt[0]["content"]
    assert json.loads(prompt[1]["content"])["history"] == []
    assert decision.verified_scopes == ({"participate"} if expected else set())
    if expected:
        assert decision.permissions[0].offer_message_id is None
    with using_clock(lambda: NOW):
        game, _ = store.create_game(conversation_id=CONVERSATION, organizer_id="organizer", organizer_name="发起人",
            requirement={"game_type": "hangzhou_mahjong", "stake": "1", "smoke_preference": "non_smoking",
                         "planned_start_at": "2026-11-25T19:00:00+08:00", "duration_hours": 4},
            known_players=[{"customer_id": "organizer", "display_name": "发起人"}], trace_id="published-game")
        result = execute(guard.guarded_gateway(store), ToolCall("record_candidate_reply", {
            "game_id": game.game_id, "customer_id": CUSTOMER, "display_name": "模拟客人", "status": "accepted",
        }))
    assert (result.called, result.allowed) == (expected, expected)
    assert any(row.customer_id == CUSTOMER for row in store.require_game(game.game_id).participants) is expected


@pytest.mark.parametrize(("venue_reply", "current", "agreed_duration", "allowed"), [
    ("20:45那桌时长还没定。你能打到几点？合适的话我再帮你确认留位。",
     "老板，能打到十一点。20:45那桌1块位的，合适就留我一个。", None, False),
    ("20:45这桌确认22:45散，能打到十一点的话时间对得上。",
     "老板，能打到十一点。20:45那桌1块位的，合适就留我一个。", 2, True),
    ("20:45那桌时长还没定，你十一点先走已经跟他们说好了。",
     "老板，能打到十一点。20:45那桌1块位的，合适就留我一个。", None, True),
    ("20:45那桌时长还没定。你能打到几点？",
     "给我留一位。我想顺便问一下几点散？", None, True),
])
def test_leave_by_time_is_checked_only_for_conditional_reservation(tmp_path, venue_reply, current, agreed_duration, allowed):
    """确认离场条件凭实际送达的桌况；合法无条件报名和明确可先走的桌照常入席。"""
    model = Mock()
    model.complete.return_value = permission("participate", current) if allowed else '{"permissions":[]}'
    guard = VenueWriteGuard(model, tmp_path / "authorization.jsonl")
    store = SQLiteAgentStore(tmp_path / "agent.sqlite3")
    requirement = {"game_type": "hangzhou_mahjong", "stake": "1", "start_time": "20:45"}
    if agreed_duration is not None:
        requirement["duration_hours"] = agreed_duration
    with using_clock(lambda: NOW):
        game, _ = store.create_game(conversation_id=CONVERSATION, organizer_id="organizer", organizer_name="发起人",
            requirement=requirement, known_players=[{"customer_id": "organizer", "display_name": "发起人"}],
            trace_id="existing-duration-table")
        authorization = guard.authorize(inquiry(current), CONVERSATION, [archived("duration-reply", venue_reply)])
        result = execute(guard.guarded_gateway(store), ToolCall("join_game", {
            "game_id": game.game_id, "customer_id": CUSTOMER, "display_name": "模拟客人",
        }))
    request = model.complete.call_args.args[0]
    assert "离场时限" in request[0]["content"] and "明确允许本人到点先走" in request[0]["content"]
    assert json.loads(request[1]["content"])["history"][0]["text"] == venue_reply
    assert authorization.verified_scopes == ({"participate"} if allowed else set())
    assert result.called == allowed and result.allowed == allowed
    assert any(p.customer_id == CUSTOMER for p in store.require_game(game.game_id).participants) == allowed


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
    unresolved_quote = {"quoted_message_present": True, "business_reference_resolved": False}
    participation_tools = {"join_game", "record_candidate_reply", "record_registered_member_withdrawal"}
    with guard.planner_scope(authorization):
        scoped = [spec for spec in baseline if spec["name"] not in dedicated - visible_writes]
        assert gateway.tool_specs_for_prompt() == scoped
        assert gateway.tool_specs_for_prompt(message_reference_contract=unresolved_quote) == [
            spec for spec in scoped if spec["name"] not in participation_tools
        ]
        if "join_game" not in visible_writes:
            accepted = execute(gateway, ToolCall("record_candidate_reply", {"status": "accepted"}))
            assert accepted.result["validation_error"]["scope"] == "participate"
        if "create_game" not in visible_writes:
            created = execute(gateway, new_game())
            assert created.result["validation_error"]["scope"] == "organize"
    assert gateway.tool_specs_for_prompt() == baseline
    assert gateway.tool_specs_for_prompt(message_reference_contract=unresolved_quote) == [
        spec for spec in baseline if spec["name"] not in participation_tools
    ]
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
        assert guard.current_planner_authorization is None
        with guard.planner_scope(reserve):
            inside = {spec["name"] for spec in gateway.tool_specs_for_prompt()}
            assert guard.current_planner_authorization is reserve
        assert guard.current_planner_authorization is None
        return before, inside, gateway.tool_specs_for_prompt()

    with guard.planner_scope(organize), ThreadPoolExecutor(max_workers=1) as pool:
        organizer_specs = gateway.tool_specs_for_prompt()
        assert guard.current_planner_authorization is organize
        assert independent_guard.current_planner_authorization is None
        assert independent_gateway.tool_specs_for_prompt() == baseline
        before, inside, after = pool.submit(worker_specs).result(timeout=2)
        assert before == after == baseline
        assert "create_game" not in inside and "reserve_room" in inside
        assert gateway.tool_specs_for_prompt() == organizer_specs
        assert {"create_game", "create_invite_drafts"} <= {spec["name"] for spec in organizer_specs}
        with pytest.raises(RuntimeError, match="abort nested scope"):
            with guard.planner_scope(reserve):
                assert {spec["name"] for spec in gateway.tool_specs_for_prompt()} == inside
                assert guard.current_planner_authorization is reserve
                raise RuntimeError("abort nested scope")
        assert gateway.tool_specs_for_prompt() == organizer_specs
        assert guard.current_planner_authorization is organize
    assert gateway.tool_specs_for_prompt() == baseline
    assert guard.current_planner_authorization is None


@pytest.mark.parametrize("authorized", [False, True])
def test_private_turn_traces_same_preflight_to_planner_and_reply_review(tmp_path, authorized):
    """用同一正式私聊链路验证未授权写入被挡、许可写入仍执行及最终回复通过审查。"""
    body = "请新组一桌杭州麻将，11月27日晚上8点开，一块底。" if authorized else "有现成的杭麻吗？"
    status = "completed" if authorized else "waiting_user"
    reply = "帮你登记了，还没凑齐。" if authorized else "现在没查到现成的杭麻。需要我帮你新组一桌吗？"
    call = replace(new_game(), reason="尝试按本轮需求登记一桌")
    if authorized:
        call.arguments["requirement"].update(start_time_kind="fixed_time", start_time="20:00",
                                             planned_start_at="2026-11-27T20:00:00+08:00")
    create = AgentAction(goal="处理找局请求", objective_status="needs_tool", reasoning_summary="尝试登记一桌",
        tool_calls=[call], stop_reason={"can_stop": False, "why": "等待工具结果",
            "pending_work": ["工具结果"], "depends_on_tool_results": True})
    search = AgentAction(goal="查现成杭麻", objective_status="needs_tool", reasoning_summary="按本轮权限只读查询",
        tool_calls=[ToolCall("search_current_games", {"requirement": {"game_type": "hangzhou_mahjong"}},
            call_id="search-current", reason="核对是否有现成杭麻")],
        stop_reason={"can_stop": False, "why": "等待查询结果", "pending_work": ["查询"], "depends_on_tool_results": True})
    final = AgentAction(goal="回答本轮请求", objective_status=status, reasoning_summary="依据本轮执行情况回复",
        reply_to_user=reply, stop_reason={"can_stop": True, "why": "回答已知事实或等客户决定",
            "pending_work": [] if authorized else ["客户选择"], "depends_on_tool_results": True})
    actions = (create, final) if authorized else (create, search, final)
    main = StaticAgentClient([json.dumps(action.to_dict(), ensure_ascii=False) for action in actions])
    review = StaticAgentClient([json.dumps({"approved": True, "needs_human": False, "reasoning_summary": "适当处理",
        "violations": [], "item_reviews": [{"item_id": "reply_to_user", "approved": True,
            "suggested_safe_text": reply, "reasoning_summary": "符合当前权限及证据", "follow_up_commitments": [],
            "violations": []}]}, ensure_ascii=False)])
    authorizer = StubAuthorizationClient(permission("organize", body) if authorized else '{"permissions":[]}')
    store = SQLiteAgentStore(tmp_path / "agent.sqlite3")
    trace = InMemoryTraceRecorder()
    guard = VenueWriteGuard(authorizer, tmp_path / "authorization.jsonl")
    runtime = AgentRuntime(llm_client=main, store=store, tool_gateway=guard.guarded_gateway(store), trace_recorder=trace,
        reply_self_review_enabled=True, reply_self_review_client=review)
    configure_venue_preflight_prompts(runtime, tmp_path, guard)
    channel = SandboxChannel(runtime, room_id="venue", identities={CUSTOMER: "模拟客人"},
        archive_path=tmp_path / "chat.sqlite3", write_guard=guard, show_verified_write_scope=True)

    result = channel.send(inquiry(body))

    events = trace.get_trace(result["trace_id"])
    assert result["status"] == status and result["final_reply"] == reply, [
        event.content for event in events if event.step in {"action_contract_error", "tool_result", "final_output"}
    ]
    assert [row["text"] for row in result["archive_outputs"]] == [reply]
    assert len(main.calls) == len(actions) and len(review.calls) == 1 and len(authorizer.calls) == 1
    assert not [event.content["errors"] for event in events if event.step == "action_contract_error"]
    assert bool(store.games) is authorized, result["tool_results"]
    assert guard.summary()["blocked_writes"] == (0 if authorized else 1), result["tool_results"]
    transport = [event for event in events if event.step in {"llm_prompt", "customer_visible_content_review_prompt"}]
    assert [event.step for event in transport] == ["llm_prompt"] * len(actions) + ["customer_visible_content_review_prompt"]
    for event, call in zip(transport, main.calls + review.calls, strict=True):
        prompt = event.content["messages"][0]["content"]
        assert prompt == call["messages"][0]["content"]
        overlay, = [line for line in prompt.splitlines() if line.startswith("sandbox_private_write_preflight = ")]
        assert json.loads(overlay.split(" = ", 1)[1]) == {
            "status": "authorized" if authorized else "no_permission", "verified_scopes": ["organize"] if authorized else [],
        }
    planner_contract = read_main_agent_system_contract(main.calls[0]["messages"][0])
    assert ("create_game" in {spec["name"] for spec in planner_contract["available_tools"]}) is authorized
    review_payload = json.loads(review.calls[0]["messages"][-1]["content"])
    evidence = {entry["name"]: entry for entry in review_payload["turn_tool_evidence"]}
    assert evidence["create_game"]["called"] is authorized and evidence["create_game"]["allowed"] is authorized
    if not authorized:
        assert evidence["create_game"]["error"]
        assert evidence["search_current_games"]["called"] and not evidence["search_current_games"]["result"]["matches"]


@pytest.mark.parametrize("path", ["direct", "after_lookup", "known_opening"])
def test_v7_0402_opening_clarification_uses_delivered_delegation_and_actual_scope(tmp_path, path):
    """到店更正能推进问开场；确有可行开场和权限时保留实际登记与查验。"""
    at = datetime.fromisoformat("2026-11-25T19:35:00+08:00")
    planned = "2026-11-25T20:15:00+08:00"
    known = path == "known_opening"
    looked_up = path == "after_lookup"
    body = ("杭州麻将。我最早20:10才能到。就20:15开吧，帮我组一桌。" if known
            else "杭州麻将。我七点四十到不了，最早八点过一点。")
    question = "这桌几点开合适？" if known else "你最早八点过一点到。这桌几点开合适？"
    if looked_up:
        question = "八点过一点以后没有能搭的现成杭麻桌。你想这桌几点开？"
    receipt_text = "八点一刻开场的杭麻给你记上了，还差三位。" if known else question
    prior = [
        ("prior-inquiry", "老板，一底两底的局有合适的吗？我晚上七点四十能到。"),
        ("prior-offer", "七点四十以后现在没有一底两底能搭的现成桌诶，要不我帮你组一桌？"),
        ("prior-delegation", "老板，组一桌吧。一底两底的，我七点四十能到。"),
        ("prior-question", "行，一底两底的我帮你组。你想打什么玩法？"),
    ]
    authority = StubAuthorizationClient(permission("organize", "帮我组一桌。") if known else '{"permissions":[]}')
    store = SQLiteAgentStore(tmp_path / "v7-0402.sqlite3")
    guard = VenueWriteGuard(authority, tmp_path / "authorization.jsonl")
    trace = InMemoryTraceRecorder()
    terms = {"game_type": "hangzhou_mahjong", "stake_range": {"min": 1, "max": 2}}
    steps = [_reply(prior[1][1], status="waiting_user"), _reply(prior[3][1], status="waiting_user")]
    if looked_up:
        steps.append(_call("search_current_games", {"lookup_scope": "personal", "requirement": {**terms,
            "start_time_range": {"start_at": "2026-11-25T20:05:00+08:00", "end_at": "2026-11-25T23:59:59+08:00"}}}))
    steps.append(_reply(question, status="waiting_user"))
    if known:
        steps.extend([_call("create_game", {"organizer_id": CUSTOMER, "organizer_name": "顾客", "known_players": [],
            "requirement": {**terms, "start_time_kind": "scheduled", "planned_start_at": planned}}), _reply(receipt_text)])
    class ShopClient(StaticAgentClient):
        """成功建局后用真实局号查当前可以联系的候选。"""

        def complete(self, messages, **options):
            content = json.loads(messages[-1]["content"])
            recent_tools = content.get("turn_tool_evidence") or []
            if known and recent_tools and recent_tools[-1].get("name") == "create_game" and recent_tools[-1].get("allowed"):
                game_id = recent_tools[-1]["result"]["game"]["game_id"]
                self.outputs.insert(0, _call("search_customers", {"game_id": game_id}))
            return super().complete(messages, **options)

    main = ShopClient(steps)

    def review_result(text, *, violations):
        """为离线回放提供原稿及有据审查结果。"""
        return json.dumps({"approved": not violations, "needs_human": False, "violations": violations,
            "reasoning_summary": "核对是否仍缺真正的开场和可执行范围",
            "item_reviews": [{"item_id": "reply_to_user", "approved": not violations, "violations": violations,
                "suggested_safe_text": "" if violations else text, "reasoning_summary": "比对原话及实际回执",
                "follow_up_commitments": []}]}, ensure_ascii=False)

    reviewer = StaticAgentClient(
        [review_result(prior[index][1], violations=[]) for index in (1, 3)]
        + ([review_result(question, violations=["known_constraint_reask", "delegation_not_advanced"])] if known else [])
        + [review_result(receipt_text, violations=[])]
    )
    clock_client = (StaticAgentClient([json.dumps({
        "clock_role": "other", "clock_roles": [{"clock_ref": 1, "role": "arrival_not_before"},
            {"clock_ref": 2, "role": "opening"}], "lookup_scope": "personal",
        "opening_window": {"start_at": planned, "end_at": planned}, "accepted_stake_range": {"min": 1, "max": 2},
        "known_cap_score": None, "has_fixed_opening_authority": True, "opening_customer_ref": 5,
        "current_ref": 5, "supporting_customer_refs": [3, 5],
    })]) if known else StubAuthorizationClient(TimeoutError("no complete clock response")))
    runtime = AgentRuntime(llm_client=main, store=store, tool_gateway=guard.guarded_gateway(store),
        trace_recorder=trace, table_availability_client=clock_client,
        reply_self_review_enabled=True, reply_self_review_client=reviewer)
    configure_venue_preflight_prompts(runtime, tmp_path, guard)
    channel = SandboxChannel(runtime, room_id="venue", identities={CUSTOMER: "顾客"},
        archive_path=tmp_path / "chat.sqlite3", write_guard=guard, show_verified_write_scope=True)
    current_permission = authority.payload
    for index, minute in ((0, 0), (2, 2)):
        identifier, text = prior[index]
        history_at = at.replace(minute=minute)
        authority.payload = '{"permissions":[]}' if index == 0 else permission("organize", "组一桌吧。")
        with using_clock(lambda: history_at):
            delivered = channel.send(MessageEnvelope(identifier, CUSTOMER, text, history_at, "private", AGENT_ID))
        assert delivered["final_reply"] == prior[index + 1][1]
    authority.payload = current_permission
    with using_clock(lambda: at):
        result = channel.send(MessageEnvelope("current-later-arrival", CUSTOMER, body, at, "private", AGENT_ID))

    audits = [json.loads(line) for line in (tmp_path / "authorization.jsonl").read_text().splitlines()
              if json.loads(line).get("event") == "authorization"]
    assert [row["status"] for row in audits] == ["no_permission", "authorized", "authorized" if known else "no_permission"]
    assert [row["text"] for row in authority.calls[-1]["history"]] == [text for _, text in prior]
    prompted = reviewer.calls[2]["messages"]
    payload = json.loads(prompted[-1]["content"])
    assert payload["current_message"]["text"] == body
    assert prior[2][1] in [turn["content"] for turn in payload["recent_conversation_tail"]]
    assert json.loads(prompted[0]["content"].split("sandbox_private_write_preflight = ", 1)[1])["verified_scopes"] == (
        ["organize"] if known else [])
    rule = payload["game_query_strategy_contract"]["delegation_acceptance"]["progress_review"]["opening_clarification"]
    assert "最早到店时间与牌局开场是两件事" in rule and "当前获准办理时，应按原委托推进" in rule
    assert "只问仍缺的可行开场属有效承接" in "".join(payload["semantic_fidelity_contract"]["rules"])
    assert "沙盒未给 `organize` 时" in prompted[0]["content"]
    events = trace.get_trace(result["trace_id"])
    receipt, = result["archive_outputs"]
    assert receipt["text"] == receipt_text and receipt["source_message_id"] == "current-later-arrival", [
        event.content for event in events if event.step in {"action_contract_error", "tool_result", "final_output",
                                                           "table_availability_authorization_invalid"}
    ]
    assert result["status"] == ("completed" if known else "waiting_user")
    tools = [event.content for event in events
             if event.step == "tool_result" and event.content["name"] in {"search_current_games", "create_game"}]
    if known:
        decision, = [event.content["decision"] for event in events
                     if event.step == "table_availability_authorization_result"]
        assert decision["has_fixed_opening_authority"] is True
        created, = tools
        assert created["name"] == "create_game" and created["called"] and created["allowed"]
        assert any(event.step == "tool_result" and event.content["name"] == "search_customers"
                   and event.content["called"] and event.content["allowed"] for event in events)
        game, = store.games.values()
        assert game.planned_start_at.isoformat() == planned and game.organizer_id == CUSTOMER
    elif looked_up:
        assert any(event.step == "table_availability_authorization_unavailable" for event in events)
        lookup, = tools
        assert lookup["name"] == "search_current_games" and lookup["called"] and lookup["result"]["matches"] == []
    else:
        assert not tools
    if not known:
        assert not store.games and not store.invite_drafts and not store.room_reservations
        assert "create_game" not in {spec["name"] for spec in read_main_agent_system_contract(
            main.calls[2]["messages"][0])["available_tools"]}
    store._connection.close()


@pytest.mark.parametrize("unconditional", [False, True])
def test_v4_conditional_seat_denial_remains_truthful_without_reasking_clear_registration(tmp_path, unconditional):
    """原桌已知，是否不等也报名不明时只澄清这项；真独立报名沿用真实占席回执。"""
    at = datetime.fromisoformat("2026-11-26T19:37:00+08:00")
    prior = "老板好 看板那桌人齐开的杭麻1块底还有位吗 我20:03能到 可以等我到再开吗"
    offer = ("有位的，那桌杭麻1块底人齐开，现在一个人，还差三个。20:03能不能赶上不好说，"
             "要等的话得桌上大家同意。你打的话我先给你记上？")
    conditional = "要的 麻烦先给我记上 我这边尽量20:05前到 要是凑齐了大家愿意等就等我一下 谢谢"
    separate = "我确认参加这桌，先给我留一个。等不了也照样报名，能等到20:05我只是顺便问问。"
    body = separate if unconditional else conditional
    first_denied_reply = "这桌杭麻人齐开，现在还没给你确认位子。你20:05前到，大家能不能等还没定。"
    error_suggestion = "好的，先给你记上了。这桌人齐开，不一定能等到你。"
    clarified = "现在还没给你留位。这桌人齐就开，要是大家等不了，你也想先报名吗？"
    confirmed = "好，位子给你留上了。等不等到20:05还没定。"
    final_text = confirmed if unconditional else clarified

    def response(text, status, call=None):
        return json.dumps(AgentAction(goal="核实这桌报名", objective_status=status, reasoning_summary="按同一桌实际情况办理",
            tool_calls=[call] if call else [], reply_to_user=text, stop_reason={
                "can_stop": call is None, "why": "按已知信息处理",
                "pending_work": [] if status == "completed" else ["确认本桌"], "depends_on_tool_results": bool(call),
            }).to_dict(), ensure_ascii=False)

    def reviewed(text, *, approved=True):
        flags = [] if approved else ["semantic_fact_loss"]
        return json.dumps({"approved": approved, "needs_human": False, "violations": flags, "reasoning_summary": "核对是否占席",
            "item_reviews": [{"item_id": "reply_to_user", "approved": approved, "violations": flags,
                "suggested_safe_text": text, "reasoning_summary": "核对是否占席", "follow_up_commitments": []}]}, ensure_ascii=False)

    store = SQLiteAgentStore(tmp_path / "participation.sqlite3")
    traces = InMemoryTraceRecorder()
    authorization = StubAuthorizationClient(permission("participate", "我确认参加这桌，先给我留一个。") if unconditional
        else permission("waitlist", "麻烦先给我记上"))
    guard = VenueWriteGuard(authorization, tmp_path / "authorization.jsonl")
    reviewer = StaticAgentClient([reviewed(offer)] + ([] if unconditional else [reviewed(error_suggestion, approved=False)])
                                 + [reviewed(final_text)])
    main = StaticAgentClient([])
    runtime = AgentRuntime(llm_client=main, store=store, tool_gateway=guard.guarded_gateway(store), trace_recorder=traces,
                           reply_self_review_enabled=True, reply_self_review_client=reviewer)
    configure_venue_preflight_prompts(runtime, tmp_path, guard)
    channel = SandboxChannel(runtime, room_id="venue", identities={CUSTOMER: "顾客", "host": "桌主"},
                            archive_path=tmp_path / "chat.sqlite3", write_guard=guard, show_verified_write_scope=True)
    terms = {"game_type": "hangzhou_mahjong", "stake": 1, "start_time_kind": "asap_when_full"}
    lookup = ToolCall("search_current_games", {"lookup_scope": "specific_table", "requirement": terms},
                      reason="按原条件查桌", call_id="lookup")
    with using_clock(lambda: at.replace(minute=35)):
        game, _ = store.create_game(conversation_id=channel.private_conversation_id("host"), organizer_id="host", organizer_name="桌主",
            requirement=terms | {"duration_hours": 4}, known_players=[], trace_id="seed")
        main.outputs.extend([
            response("", "needs_tool", lookup), response(offer, "waiting_user"),
            response("", "needs_tool", ToolCall("join_game", {"game_id": game.game_id, "customer_id": CUSTOMER,
                "display_name": "顾客"}, reason="处理本人报名", call_id="join")),
        ])
        if not unconditional:
            main.outputs.append(response(first_denied_reply, "completed"))
        main.outputs.append(response(final_text, "completed" if unconditional else "waiting_user"))
        old = channel.send(MessageEnvelope("prior", CUSTOMER, prior, at.replace(minute=35), "private", AGENT_ID))
        assert old["final_reply"] == offer

    current = MessageEnvelope("conditional-seat", CUSTOMER, body, at, "private", AGENT_ID)
    with using_clock(lambda: at):
        result = channel.send(current)

    events = traces.get_trace(result["trace_id"])
    receipt, = result["archive_outputs"]
    assert result["final_reply"] == receipt["text"] == final_text
    assert receipt["source_message_id"] == current.message_id
    writes = [event.content["call"]["name"] for event in events if event.step == "tool_called"]
    assert writes == ["join_game"], [(event.step, event.content.get("name"), event.content.get("errors"),
                                      event.content.get("error")) for event in events if "tool" in event.step or "contract" in event.step]
    joined, = [event.content for event in events if event.step == "tool_result" and event.content.get("name") == "join_game"]
    assert joined["called"] is joined["allowed"] is unconditional
    assert any(person.customer_id == CUSTOMER and person.status == "confirmed"
               for person in store.require_game(game.game_id).participants) is unconditional
    final_audit = json.loads(reviewer.calls[-1]["messages"][-1]["content"])["semantic_fidelity_contract"]
    if unconditional:
        assert "seat_registration_evidence" not in final_audit
        assert result["status"] == "completed" and "？" not in final_text
    else:
        assert final_audit["seat_registration_evidence"] == {
            "game_id": game.game_id, "write_status": "not_executed", "authorization_status": "not_verified",
        }
        assert any("已经回应占席请求" in rule for rule in final_audit["rules"])
        rule = "customer_visible_contract:seat_registration.not_recorded"
        feedback, = [event.content["result"] for event in events if event.step == "tool_result"
                     and any(rule in item["violations"] for item in event.content.get("result", {}).get("item_reviews", []))]
        assert feedback["item_reviews"][0]["suggested_safe_text"] == ""
        assert all(row["text"] != error_suggestion for row in channel.events_for(CUSTOMER))
    store._connection.close()


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
