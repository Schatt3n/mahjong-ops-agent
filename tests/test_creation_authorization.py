"""验证建局前使用已送达的本任务授权以及拒绝后无副作用的工具边界。"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from mahjong_agent_runtime import (
    AgentAction, AgentRuntime, CustomerProfile, InMemoryAgentStore, InMemoryTraceRecorder, SQLiteAgentStore,
    ToolCall, ToolGateway, ToolResult, UserMessage,
)
from mahjong_agent_runtime.domains.creation_authorization import (
    _parse_decision, check_creation_authorization, creation_clock_conflict, customer_authorization_evidence,
    delegated_private_creation_needs_gameplay,
)
from mahjong_agent_runtime.models import now, using_clock
from mahjong_agent_runtime.services import ToolExecutionService
from tests.simulation.venue_channel import AGENT_ID, MessageEnvelope, SandboxChannel


_FIXTURE = json.loads((Path(__file__).resolve().parents[1] / "eval" / "golden" /
                       "venue_creation_authorization_replay.json").read_text(encoding="utf-8"))


class Client:
    """按次记录模型请求，给运行时返回明确的测试响应。"""
    def __init__(self, answers):
        self.answers = iter(answers)
        self.calls = []

    def complete(self, messages, **_):
        self.calls.append(messages)
        return json.dumps(next(self.answers), ensure_ascii=False)


def test_creation_authorization_sees_actual_local_timestamp_and_proposed_expiry_before_write():
    """独立授权模型能区分到店钟点与默认牌局有效期，没有授权时也保留判决幂等。"""
    store = InMemoryAgentStore()
    client = Client([{"scope": "unconfirmed", "customer_ref": 1, "shop_offer_ref": None}])
    trace = InMemoryTraceRecorder()
    sent = datetime.fromisoformat("2026-11-26T11:07:00+08:00")
    incoming = UserMessage("dm", "guest", "顾客", "帮我凑一桌，我七点一刻左右能到", message_id="source", sent_at=sent)
    proposed = ToolCall("create_game", {"requirement": {"start_time_kind": "asap_when_full"}})
    with using_clock(lambda: sent):
        assert check_creation_authorization(store, client, proposed, incoming,
            trace_id="first", step_index=1, timeout_seconds=1, trace_recorder=trace) == "unconfirmed"
    with using_clock(lambda: sent + timedelta(seconds=30)):
        assert check_creation_authorization(store, client, proposed, incoming,
            trace_id="retry", step_index=1, timeout_seconds=1, trace_recorder=trace) == "unconfirmed"
    assert len(client.calls) == 1
    payload = json.loads(client.calls[0][1]["content"])
    assert payload["customer_message_local_at"] == "2026-11-26T11:07:00+08:00"
    assert payload["current_clock_observation"]["local_candidate_at"] == "2026-11-26T19:15:00+08:00"
    assert payload["current_clock_observation"]["qualifier"] == "左右"
    assert payload["proposed_game_lifecycle"] == {
        "planned_start_at": None, "planned_end_at": None, "expires_at": "2026-11-26T15:07:00+08:00",
    }
    assert not store.games


def test_creation_authorization_receives_latest_successful_lookup_without_table_members():
    """写前审查能辨认真实已有桌，失败查局和后台默认时长均不覆盖有效证据。"""
    store = InMemoryAgentStore()
    client = Client([{"scope": "unconfirmed", "customer_ref": 1, "shop_offer_ref": None}])
    sent = datetime.fromisoformat("2026-11-26T09:07:00+08:00")
    incoming = UserMessage("dm", "guest", "顾客", "帮我找一桌", message_id="source", sent_at=sent)
    lookup = ToolResult("search_current_games", True, True, result={
        "requirement": {"game_type": "red_center_mahjong", "requesting_party": {"contact_id": "internal-guest"}},
        "matches": [{"game": {"planned_start_at": "2026-11-26T12:15:00+08:00",
                              "planned_end_at": "2026-11-26T16:15:00+08:00",
                              "requirement": {"game_type": "red_center_mahjong"},
                              "participants": [{"customer_id": "other-customer"}]}}],
    })
    failed = ToolResult("search_current_games", False, False, error="not executed", result={"matches": []})
    assert check_creation_authorization(store, client, ToolCall("create_game", {"requirement": {}}), incoming,
        trace_id="trace", step_index=1, timeout_seconds=1, trace_recorder=InMemoryTraceRecorder(),
        observed_tool_results=[lookup, failed]) == "unconfirmed"
    review = json.loads(client.calls[0][1]["content"])["latest_existing_game_lookup"]
    assert review == {"query": {"game_type": "red_center_mahjong"}, "returned_matches": 1,
                      "verified_options": [{"game_type": "red_center_mahjong", "planned_start_at": "2026-11-26T12:15:00+08:00"}]}


@pytest.mark.parametrize(("text", "proposed", "blocked"), [
    ("帮我凑一桌 我九点半前能到 能打到晚上", "2026-11-26T21:30:00+08:00", True),
    ("帮我凑一桌 我九点半前能到 能打到晚上", "2026-11-26T13:30:00Z", True),
    ("我九点半前能到 想打红中", "2026-11-26T10:00:00+08:00", False),
    ("我晚上九点半前能到", "2026-11-26T21:30:00+08:00", False),
    ("我九点半前能到 这场晚上九点半开", "2026-11-26T21:30:00+08:00", False),
    ("我九点半前参加会议，下午四点打麻将", "2026-11-26T16:00:00+08:00", False),
    ("我九点半前能到", "2026-11-27T21:30:00+08:00", False),
    ("我九点半前能到", "2026-11-26T09:30:00+08:00", False),
    ("九点半开吧", "2026-11-26T21:30:00+08:00", False),
])
def test_only_opposite_half_day_of_same_single_bounded_clock_is_blocked(text, proposed, blocked):
    """同数字的半天翻转须澄清，其他正式开场仍交给语义判断及原授权。"""
    sent = datetime.fromisoformat("2026-11-26T09:07:00+08:00")
    message = UserMessage("dm", "guest", "顾客", text, message_id="source", sent_at=sent)
    call = ToolCall("create_game", {"requirement": {"start_time_kind": "scheduled", "planned_start_at": proposed}})
    assert bool(creation_clock_conflict(InMemoryAgentStore(), call, message)) is blocked


def test_earlier_explicit_own_clock_in_same_task_keeps_semantic_approval_available():
    """靠谱的本任务本人早前开场线索不应被最新一句到店时刻覆盖。"""
    stamp = datetime.fromisoformat("2026-11-26T09:07:00+08:00")
    earlier = stamp - timedelta(minutes=2)
    store = InMemoryAgentStore()
    store.activate_task_context(conversation_id="dm", customer_id="guest", trace_id="setup", activity_at=earlier,
        started_at=earlier, reason="test", force_new=True, archive_previous=False)
    caller = UserMessage("dm", "guest", "顾客", "想约今晚九点半那桌", message_id="older", sent_at=earlier)
    current = UserMessage("dm", "guest", "顾客", "我九点半前能到", message_id="source", sent_at=stamp)
    call = ToolCall("create_game", {"requirement": {"start_time_kind": "scheduled", "planned_start_at": "2026-11-26T21:30:00+08:00"}})
    assert creation_clock_conflict(store, call, current)
    store.append_user_turn(caller, "trace-older")
    assert creation_clock_conflict(store, call, current) is None


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    if request.param == "memory":
        yield InMemoryAgentStore()
    else:
        db = SQLiteAgentStore(tmp_path / "creation-authorization.sqlite3")
        yield db
        db._connection.close()


def _action(*, tool=None, reply="好", status="completed"):
    return {
        "goal": "按客户已确认的范围处理",
        "objective_status": status,
        "reasoning_summary": "核对本轮实际授权",
        "reply_to_user": reply if tool is None else "",
        "tool_calls": [] if tool is None else [tool],
        "needs_human": False,
        "stop_reason": {"can_stop": tool is None, "depends_on_tool_results": tool is not None,
                        "pending_work": [] if tool is None else ["核对工具结果"], "why": "按工具结果决定"},
    }


def _create():
    start = now() + timedelta(hours=2)
    return {
        "name": "create_game",
        "arguments": {
            "organizer_id": "guest", "organizer_name": "牌友甲",
            "requirement": {"game_type": "hangzhou_mahjong", "stake": "1", "smoke_preference": "no_smoking",
                            "start_time": start.strftime("%H:%M"), "planned_start_at": start.isoformat(),
                            "duration_hours": 4, "start_time_kind": "scheduled", "duration_kind": "fixed"},
            "known_players": [{"customer_id": "guest", "display_name": "牌友甲"}],
        },
        "reason": "主模型提出建局",
    }


def test_v6_ordinary_private_delegation_waits_for_gameplay_before_creating(tmp_path):
    """复现真实七点半委托，先送达唯一缺项，补玩法后保留原开场和离店上限写入。"""
    at = datetime.fromisoformat("2026-11-25T19:00:00+08:00")
    start = "2026-11-25T19:30:00+08:00"
    end = "2026-11-25T22:00:00+08:00"
    original = "老板晚上好 七点半这桌还有位吗 0.5的底"
    offer = "七点半0.5底这桌现在没查到诶，要不我帮你组一桌？"
    delegate = "老板 那就麻烦你帮我组一桌吧 七点半 0.5底 无烟最好 我打到10点前"
    clarify = "好，你想打哪种麻将？"
    confirm = "七点半国标给你记上了，0.5底无烟，还差三位。"
    search = {"name": "search_current_games", "reason": "查先前问的七点半桌", "call_id": "lookup",
              "arguments": {"lookup_scope": "specific_table", "requirement": {"stake": 0.5, "planned_start_at": start}}}
    terms = {"stake": 0.5, "smoke_preference": "no_smoking", "start_time_kind": "scheduled",
             "start_time": "19:30", "planned_start_at": start, "max_duration_hours": 2.5}
    create = {"name": "create_game", "reason": "按核实开场与最晚离店建待组局", "call_id": "create_v6",
              "arguments": {"requirement": terms, "organizer_id": "guest", "organizer_name": "顾客",
                            "requesting_party": {"contact_id": "guest", "contact_name": "顾客", "seat_count": 1}}}
    wrong_duration_terms = {key: value for key, value in terms.items() if key != "max_duration_hours"}
    wrong_duration = {**create, "call_id": "create_first", "arguments": {
        **create["arguments"], "requirement": {**wrong_duration_terms, "duration_hours": 2.5}}}
    complete_create = {**create, "arguments": {**create["arguments"], "requirement": {
        **terms, "game_type": "national_standard_mahjong"}}}

    class ReplayPlanner(Client):
        """候选查询使用本次真实建局回执中的牌局标识。"""

        def complete(self, messages, **options):
            response = json.loads(super().complete(messages, **options))
            if (calls := response["tool_calls"]) and calls[0]["name"] == "search_customers":
                evidence = json.loads(messages[-1]["content"])["turn_tool_evidence"]
                game, = [item["result"]["game"] for item in evidence if item["name"] == "create_game" and item["called"]]
                calls[0]["arguments"]["game_id"] = game["game_id"]
            return json.dumps(response, ensure_ascii=False)

    discover = {"name": "search_customers", "reason": "查已登记同桌的潜在牌友", "arguments": {"exclude_customer_ids": ["guest"]}}
    planner = ReplayPlanner([_action(tool=search, status="needs_tool"), _action(reply=offer, status="waiting_user"),
                            _action(tool=wrong_duration, status="needs_tool"), _action(tool=create, status="needs_tool"),
                            _action(reply=clarify, status="waiting_user"), _action(tool=complete_create, status="needs_tool"),
                            _action(tool=discover, status="needs_tool"), _action(reply=confirm)])
    authorizer = Client([{"scope": "create", "customer_ref": 3, "shop_offer_ref": 2},
                         {"scope": "create", "customer_ref": 5, "shop_offer_ref": None}])
    verify = {"clock_role": "opening", "lookup_scope": "specific_table", "opening_window": None,
              "accepted_stake_range": None, "known_cap_score": None, "has_fixed_opening_authority": False,
              "current_ref": 1, "supporting_customer_refs": [1]}
    availability = Client([verify, {**verify, "clock_role": "other", "lookup_scope": "personal",
        "clock_roles": [{"clock_ref": 1, "role": "opening"}, {"clock_ref": 2, "role": "departure_deadline"}],
        "opening_window": {"start_at": start, "end_at": start}, "accepted_stake_range": {"min": 0.5, "max": 0.5},
        "has_fixed_opening_authority": True, "opening_customer_ref": 3, "current_ref": 3,
        "supporting_customer_refs": [1, 3], "agreed_duration_hours": None, "max_duration_hours": None}])
    store, trace = SQLiteAgentStore(tmp_path / "v6.sqlite3"), InMemoryTraceRecorder()
    runtime = AgentRuntime(llm_client=planner, store=store, trace_recorder=trace, reply_self_review_enabled=False,
        table_availability_client=availability, creation_authorization_client=authorizer)
    channel = SandboxChannel(runtime, room_id="board", identities={"guest": "顾客"}, archive_path=tmp_path / "chat.sqlite3")

    def inbound(source, body, when):
        with using_clock(lambda: when):
            return channel.send(MessageEnvelope(source, "guest", body, when, "private", AGENT_ID))

    first = inbound("initial", original, at)
    assert first["final_reply"] == offer
    lookup, = [item.content for item in trace.events if item.step == "tool_result"
               and item.content["name"] == "search_current_games"]
    assert lookup["called"] and lookup["allowed"] and not lookup["result"]["matches"]
    second = inbound("v6-delegation", delegate, at)
    history = trace.get_trace(second["trace_id"])
    rejected = [event.content for event in history if event.step == "tool_result" and event.content["name"] == "create_game"]
    assert [item["result"]["validation_error"]["code"] for item in rejected] == [
        "verified_table_availability_conflict", "create_game_gameplay_required",
    ]
    assert all(not item["called"] and not item["allowed"] and not item["state_transitions"] for item in rejected)
    assert not any(event.step == "tool_gateway_received" for event in history)
    scope, = [event.content["decision"] for event in history if event.step == "creation_authorization_result"]
    assert scope == {"scope": "create", "customer_ref": 3, "shop_offer_ref": 2}
    verified, = [event.content["decision"] for event in history if event.step == "table_availability_authorization_result"]
    assert verified["opening_window"] == {"start_at": start, "end_at": start} and verified["must_end_by"] == end
    assert not store.games and not store.invite_drafts and not store.scheduled_tasks
    assert second["status"] == "waiting_user" and second["final_reply"] == clarify
    assert second["archive_outputs"][0]["source_message_id"] == "v6-delegation"
    assert [row["text"] for row in channel.events_for("guest")] == [original, offer, delegate, clarify]

    third = inbound("concrete-gameplay", "打国标", at + timedelta(minutes=1))
    assert third["final_reply"] == confirm, [(event.step, event.content) for event in trace.get_trace(third["trace_id"])
        if event.step in {"tool_result", "action_contract_error", "creation_authorization_result", "creation_authorization_error"}]
    game, = store.games.values()
    assert game.requirement["game_type"] == "national_standard_mahjong" and game.requirement["max_duration_hours"] == 2.5
    assert "duration_hours" not in game.requirement
    assert game.planned_start_at.isoformat() == start and game.planned_end_at.isoformat() == end
    assert game.remaining_seats() == 3


@pytest.mark.parametrize(("case", "expected_source"), [
    ("generic_offer", None),
    ("own_current_gameplay", "customer"),
    ("own_prior_gameplay", "customer"),
    ("unique_profile", "unique_customer_profile"),
    ("accepted_delivered_offer", "accepted_delivered_offer"),
    ("undelivered_offer", None),
    ("foreign_gameplay", None),
    ("conflicting_current_gameplay", None),
    ("delegates_choice", "customer_delegated_gameplay_choice"),
    ("accepts_any_gameplay", "customer_delegated_gameplay_choice"),
    ("forged_choice", None),
    ("clipped_negation", None),
    ("delegates_only_time", None),
])
def test_v9_direct_delegation_requires_gameplay_provenance(tmp_path, case, expected_source):
    """冻结直接建桌的真实来话和授权，合法玩法来源与模型提案分别验收。"""
    evidence = Path(__file__).resolve().parents[1] / "docs/venue_simulation_evidence"
    archive = json.loads((evidence / "2026-11-27-v9-0146-direct-creation-gameplay.json").read_text(encoding="utf-8"))
    first, delivered, current = archive["ledger"][:3]
    at = datetime.fromisoformat(archive["business_time"])
    earlier = at - timedelta(minutes=2)
    store, trace = SQLiteAgentStore(tmp_path / "direct-gameplay.sqlite3"), InMemoryTraceRecorder()
    topic, _ = store.activate_task_context(conversation_id="dm", customer_id="guest", trace_id="fixture", activity_at=earlier,
        started_at=earlier, reason="回放", force_new=True, archive_previous=False)
    earlier_text = first["text"] + ("，红中麻将。" if case == "own_prior_gameplay" else "")
    modifiers = {
        "own_current_gameplay": "打红中", "conflicting_current_gameplay": "打杭麻", "delegates_choice": "玩法你定",
        "accepts_any_gameplay": "打什么都行", "clipped_negation": "别说玩法你定", "delegates_only_time": "开场时间你定",
    }
    body = current["text"] + (f"，{modifiers[case]}。" if case in modifiers else "")
    message = UserMessage("dm", "guest", "顾客", body, current["message_id"], at)
    decision = dict(archive["creation_authorizer"]["decision"])
    choices = {"delegates_choice": "玩法你定", "accepts_any_gameplay": "打什么都行", "forged_choice": "玩法你定",
               "clipped_negation": "玩法你定", "delegates_only_time": "开场时间你定"}
    if case in choices:
        decision["gameplay_choice"] = {"customer_ref": 3, "quote": choices[case]}
    client = Client([decision])
    service = ToolExecutionService(store, ToolGateway(store, trace_recorder=trace), trace, creation_authorization_client=client)
    proposal = ToolCall("create_game", {"organizer_id": "guest", "organizer_name": "顾客", **archive["proposal"]})
    offer = ("要不要我帮你组一桌0.5块底红中麻将？" if case == "accepted_delivered_offer" else delivered["text"])
    with using_clock(lambda: earlier):
        store.append_user_turn(UserMessage("dm", "guest", "顾客", earlier_text, first["message_id"], earlier), "first")
        store.append_assistant_turn("dm", offer, "delivered",
            metadata={"delivery_status": "delivered", "task_context_id": topic.task_context_id})
        if case == "undelivered_offer":
            store.append_assistant_turn("dm", "要不帮你组一桌红中？", "pending",
                metadata={"delivery_status": "pending_operator_send", "task_context_id": topic.task_context_id})
        if case == "foreign_gameplay":
            store.append_user_turn(UserMessage("dm", "another-customer", "他人", "我要打红中", "foreign", earlier), "foreign")
    with using_clock(lambda: at):
        if case in {"unique_profile", "conflicting_current_gameplay"}:
            store.upsert_customer(CustomerProfile("guest", "顾客", preferred_games=["red_center_mahjong"]))
        store.append_user_turn(message, "current")
        action = AgentAction("按本人委托组桌", "needs_tool", "核对本轮玩法", tool_calls=[proposal])
        result, = service.execute_tool_calls(action,
            message=message, trace_id="direct", step_index=1, previous_step_tool_results=[], run_id="direct-run",
            run_version=store.conversation_version(message.conversation_id)).tool_results
        if case == "generic_offer":
            repeated, = service.execute_tool_calls(action,
                message=message, trace_id="direct", step_index=2, previous_step_tool_results=[result], run_id="direct-run",
                run_version=store.conversation_version(message.conversation_id)).tool_results
            assert not repeated.called and repeated.result == result.result
    allowed = expected_source is not None
    assert result.called is result.allowed is allowed, result.error
    authorization, = [event.content for event in trace.events if event.step == "creation_authorization_result"]
    assert authorization["decision"] == decision
    assert len(client.calls) == 1 and "gameplay_choice" in client.calls[0][0]["content"]
    if allowed:
        game, = store.games.values()
        assert game.requirement["game_type"] == "red_center_mahjong" and game.remaining_seats() == 3
        assert authorization["gameplay_source"]["source"] == expected_source
    else:
        assert not store.games and not store.scheduled_tasks
        assert not any(event.step == "tool_gateway_received" for event in trace.events)
        assert result.result["validation_error"]["code"] == "create_game_gameplay_required"
    assert not store.invite_drafts and not store.outbound_message_drafts
    store._connection.close()


@pytest.mark.parametrize(("gameplay", "source", "scope", "blocked"), [
    ({}, {}, "create", True),
    ({"game_type": "  "}, {}, "create", True),
    ({"game_type": "any"}, {}, "create", True),
    ({"game_type": "unknown"}, {}, "create", True),
    ({"requested_game": "sichuan_mahjong"}, {}, "create", False),
    ({"game_type": "club_custom_mahjong"}, {}, "create", False),
    ({}, {"source": "group", "is_room": True}, "create", False),
    ({}, {"source": "group_private_continuation"}, "create", False),
    ({}, {"input_source": "system_trigger"}, "create", False),
    ({}, {}, "inquire", False),
    ({}, {}, None, False),
])
def test_gameplay_write_boundary_preserves_custom_types_and_separate_sources(gameplay, source, scope, blocked):
    """只拦确实获准的本人普通私聊条件缺失，具体开放玩法和其他业务入口按原合同处理。"""
    call = ToolCall("create_game", {"organizer_id": "guest", "requirement": {"stake": 0.5, **gameplay}})
    message = UserMessage("dm", "guest", "顾客", "帮我凑一桌", metadata=source)
    assert delegated_private_creation_needs_gameplay(call, message, scope) is blocked


def test_same_half_day_miswrite_retries_never_reach_gateway_or_change_state(store):
    """同一误写用另一时区表示或重复提案均不允许执行，无需依赖审查模型。"""
    stamp = datetime.fromisoformat("2026-11-26T09:07:00+08:00")
    msg = UserMessage("dm", "guest", "牌友甲", "我九点半前能到，能打到晚上", message_id="morning", sent_at=stamp)
    trace = InMemoryTraceRecorder()
    service = ToolExecutionService(store=store, tool_gateway=ToolGateway(store, trace_recorder=trace), trace_recorder=trace)
    with using_clock(lambda: stamp):
        raw = _create()
        requirements = [{**raw["arguments"]["requirement"], "start_time": "21:30", "planned_start_at": value}
                        for value in ("2026-11-26T21:30:00+08:00", "2026-11-26T13:30:00Z")]
        for step, requirement in enumerate(requirements, start=1):
            proposal = ToolCall("create_game", {**raw["arguments"], "requirement": requirement})
            outcome = service.execute_tool_calls(AgentAction("组桌", "needs_tool", "拟定开始时间", tool_calls=[proposal]),
                message=msg, trace_id="clock-conflict", step_index=step, previous_step_tool_results=[],
                run_id="clock-run", run_version=0, context_payload={})
            result, = outcome.tool_results
            assert not result.called and not result.allowed and not result.state_transitions
            assert result.result["validation_error"]["code"] == "create_game_half_day_time_conflict"
            assert result.result["reference_clock_observation"]["source_message_id"] == msg.message_id
    assert not any(event.step == "tool_gateway_received" for event in trace.events)
    assert not store.games and not store.invite_drafts


def test_unconfirmed_proposal_feedback_preserves_delegation_and_only_inquire_boundary(store):
    """拟写时间有歧义不能虚称没委托；只请打听仍按原权限拒绝建局。"""
    raw = _create()
    gateway = ToolGateway(store)
    call = ToolCall(raw["name"], raw["arguments"])
    def check(scope):
        return gateway.execute(call, trace_id=f"scope:{scope}", step_index=1, conversation_id="dm",
            sender_id="guest", sender_name="牌友甲", message_reference_contract={"game_creation_authorization": scope})
    uncertain = check("unconfirmed")
    inquire = check("inquire")
    assert not uncertain.called and not inquire.called and not store.games
    assert "已有组桌委托要保留" in uncertain.result["instruction"] and "直接询问该缺项" in uncertain.result["instruction"]
    assert inquire.error == "顾客只委托了解有没有人愿意打，未授权新建牌局。"
    assert inquire.result["instruction"] == "此建局未执行。按顾客实际授权处理，必要时只读查局或查候选意向，不要重复提交同一写入。"


def test_interest_only_model_proposal_is_blocked_before_any_business_write(store):
    tool = _create()
    proposed = _action(tool=tool, status="needs_tool")
    planner = Client([proposed, proposed, _action(reply="那目前先不安排。")])
    authorizer = Client([{"scope": "inquire", "customer_ref": 2, "shop_offer_ref": 1}])
    runtime = AgentRuntime(llm_client=planner, creation_authorization_client=authorizer, store=store)
    message = UserMessage(
        conversation_id="dm", sender_id="guest", sender_name="牌友甲", text=_FIXTURE["customer_inquiry"],
        message_id="interest",
        metadata={"game_creation_authorization": "create"},
    )
    task = runtime.task_context_manager.prepare(message, trace_id="before-inquiry")
    with using_clock(lambda: message.sent_at - timedelta(seconds=1)):
        store.append_assistant_turn("dm", _FIXTURE["delivered_shop_offer"], "interest-offer",
                                    metadata={"delivery_status": "delivered", "task_context_id": task.context.task_context_id})
    result = runtime.handle_user_message(message)
    assert len(authorizer.calls) == 1
    evidence = json.loads(authorizer.calls[0][-1]["content"])["conversation"]
    assert [item["text"] for item in evidence] == [_FIXTURE["delivered_shop_offer"], _FIXTURE["customer_inquiry"]]
    attempts = [item for item in result.tool_results if item.name == "create_game"]
    assert len(attempts) == 2
    assert all(not item.called and not item.allowed and not item.state_transitions for item in attempts)
    assert not store.games and not store.invite_drafts and not store.room_reservations
    assert "create_game_authorization_required" in planner.calls[1][-1]["content"]


def test_explicit_org_consent_still_creates_the_proposed_table(store):
    tool = _create()
    candidate_query = {"name": "search_customers", "reason": "核对组局候选",
                       "arguments": {"game_id": "dynamic", "limit": 6}}
    authorizer = Client([{"scope": "create", "customer_ref": 1, "shop_offer_ref": None}])

    class Planner(Client):
        def complete(self, messages, **kwargs):
            if not self.calls:
                return super().complete(messages, **kwargs)
            context = json.loads(messages[-1]["content"])
            if len(self.calls) == 1:
                created, = [row for row in context["turn_tool_evidence"] if row["name"] == "create_game"]
                candidate_query["arguments"]["game_id"] = created["result"]["game"]["game_id"]
            return super().complete(messages, **kwargs)

    planner = Planner([_action(tool=tool, status="needs_tool"), _action(tool=candidate_query, status="needs_tool"), _action()])
    result = AgentRuntime(llm_client=planner, creation_authorization_client=authorizer, store=store).handle_user_message(
        UserMessage(conversation_id="dm", sender_id="guest", sender_name="牌友甲",
                    text=_FIXTURE["explicit_creation_control"], message_id="direct-create"),
    )
    writes = [item for item in result.tool_results if item.name == "create_game"]
    assert len(writes) == 1 and writes[0].allowed and writes[0].called and writes[0].error is None
    game, = store.games.values()
    assert game.organizer_id == "guest" and game.remaining_seats() == 3
    assert game.planned_end_at - game.planned_start_at == timedelta(hours=4)


def test_evidence_only_contains_same_customer_and_delivered_earlier_shop_offer(store):
    original = "要我帮你开一桌凑人吗？"
    store.append_assistant_turn("dm", original, "sent", metadata={"delivery_status": "delivered"})
    store.append_assistant_turn("dm", "私下草稿", "pending", metadata={"delivery_status": "pending_operator_send"})
    store.append_assistant_turn("other", "另一位顾客的回复", "unrelated", metadata={"delivery_status": "delivered"})
    store.append_user_turn(UserMessage(conversation_id="dm", sender_id="other", sender_name="他人", text="非本人"), "foreign")
    current = UserMessage(conversation_id="dm", sender_id="guest", sender_name="牌友甲", text="行，按这桌组", message_id="accept")
    evidence = customer_authorization_evidence(store, current)
    assert evidence == [{"speaker": "shop", "text": original, "ref": 1},
                        {"speaker": "customer", "text": current.text, "current": True, "ref": 2}]
    assert _parse_decision('{"scope":"create","customer_ref":2,"shop_offer_ref":1}', evidence)["scope"] == "create"
    for bad in (
        '{"scope":"create","customer_ref":null,"shop_offer_ref":null}',
        '{"scope":"create","customer_ref":1,"shop_offer_ref":null}',
        '{"scope":"create","customer_ref":2,"shop_offer_ref":3}',
        '{"scope":"inquire","scope":"create","customer_ref":2,"shop_offer_ref":1}',
    ):
        with pytest.raises(ValueError):
            _parse_decision(bad, evidence)


def test_creation_citation_repairs_visible_reversed_offer_once_without_serving_stale_permission(store):
    at = now()
    current = UserMessage(conversation_id="dm", sender_id="guest", sender_name="牌友甲",
                          text="就我一个先，七点无烟的你帮我组着，我能打满四小时", sent_at=at)
    earlier = UserMessage(conversation_id="dm", sender_id="guest", sender_name="牌友甲",
                          text="七点有无烟吗？", sent_at=at - timedelta(minutes=2))
    store.append_user_turn(earlier, "first")
    with using_clock(lambda: at - timedelta(minutes=1)):
        store.append_assistant_turn("dm", "我帮你组一桌，就你一个人来打吗？", "effective-offer",
                                    metadata={"delivery_status": "delivered"})
    initial = {"scope": "create", "customer_ref": 1, "shop_offer_ref": 2}
    corrected = {"scope": "create", "customer_ref": 3, "shop_offer_ref": 2}
    client = Client([initial, corrected])
    trace = InMemoryTraceRecorder()
    tool = ToolCall(**_create())
    options = {"trace_id": "repair", "step_index": 101, "timeout_seconds": 45, "trace_recorder": trace}
    assert check_creation_authorization(store, client, tool, current, **options) == "create"
    assert len(client.calls) == 2 and client.calls[0][-1] == client.calls[1][-1]
    assert "current=true" in client.calls[1][0]["content"]
    assert [event.step for event in trace.events] == [
        "creation_authorization_prompt", "creation_authorization_invalid",
        "creation_authorization_prompt", "creation_authorization_result",
    ]
    assert not store.games
    assert check_creation_authorization(store, client, tool, current, **options) == "create"
    assert len(client.calls) == 2


@pytest.mark.parametrize("explicit_smoke,visible", [(None, True), ("smoking", False)])
def test_creation_authorizer_only_receives_compatible_same_customer_attendance_as_terms(
    store, monkeypatch, explicit_smoke, visible,
):
    """历史补的是实际操作条件，独立审核仍须引用当次明确组局来话。"""
    history = {"game_type": "hangzhou_mahjong", "stake": "0.5", "smoke_preference": "no_smoking", "completed_games": 3}
    store.upsert_customer(CustomerProfile("guest", "牌友甲", smoke_preference=explicit_smoke, private_remark="不应发送的备注"))
    requested = []

    def observed(_store, customer_id):
        requested.append(customer_id)
        return history

    monkeypatch.setattr(type(store), "completed_game_preference", observed)
    current = UserMessage(conversation_id="dm", sender_id="guest", sender_name="牌友甲", message_id="returning",
                          text="帮我组个19:00的局呗", metadata={"attendance_preference": {"stake": "99"}})
    fixture = _create()
    fixture["arguments"]["requirement"]["stake"] = "0.5"
    tool = ToolCall(**fixture)
    client = Client([{"scope": "create", "customer_ref": None, "shop_offer_ref": None}])
    status = check_creation_authorization(store, client, tool, current, trace_id="learned",
        step_index=1, timeout_seconds=1, trace_recorder=InMemoryTraceRecorder())
    assert requested == ["guest"]
    assert status == "unconfirmed" and not store.games
    sent = client.calls[0][-1]["content"]
    payload = json.loads(sent)
    assert payload.get("attendance_preference") == (history if visible else None)
    assert payload["conversation"] == [{"speaker": "customer", "current": True, "ref": 1, "text": current.text}]
    assert "不应发送的备注" not in sent and '"99"' not in sent


@pytest.mark.parametrize("text,seats", [("帮我组个19:00的局呗", 1), ("帮我组个19:00的局呗，我和搭子两个人。", 2)])
def test_returning_personal_organization_uses_default_or_stated_party(store, monkeypatch, text, seats):
    """直接委托组个人局沿用通常条件和一席，顾客主动说出同行时以实际人数为准。"""
    store.upsert_customer(CustomerProfile("guest", "牌友甲"))
    monkeypatch.setattr(type(store), "completed_game_preference", lambda *_: {
        "game_type": "hangzhou_mahjong", "stake": "0.5", "smoke_preference": "no_smoking", "completed_games": 3,
    })
    call = _create()
    call["arguments"]["requirement"]["stake"] = "0.5"
    call["arguments"]["known_players"] = []
    if seats > 1:
        call["arguments"]["requesting_party"] = {"contact_id": "guest", "contact_name": "牌友甲", "seat_count": seats}
    proposal = _action(tool=call, status="needs_tool")
    candidate_query = {"name": "search_customers", "reason": "按已建的桌查牌友", "arguments": {"game_id": "dynamic"}}

    class Planner(Client):
        def complete(self, messages, **kwargs):
            if len(self.calls) == 1:
                context = json.loads(messages[-1]["content"])
                created, = [row for row in context["turn_tool_evidence"] if row["name"] == "create_game"]
                candidate_query["arguments"]["game_id"] = created["result"]["game"]["game_id"]
            return super().complete(messages, **kwargs)

    planner = Planner([proposal, _action(tool=candidate_query, status="needs_tool"),
                       _action(reply="好，这桌先给你组着。")])
    authorizer = Client([{"scope": "create", "customer_ref": 1, "shop_offer_ref": None}])
    runtime = AgentRuntime(llm_client=planner, creation_authorization_client=authorizer, store=store)
    result = runtime.handle_user_message(UserMessage("dm", "guest", "牌友甲", text, "opening"))
    confirmed, search = result.tool_results
    assert confirmed.called and confirmed.allowed
    assert search.name == "search_customers" and search.called and search.allowed
    game, = store.games.values()
    assert game.remaining_seats() == 4 - seats
    assert game.participants[0].customer_id == "guest" and game.participants[0].seat_count == seats
    assert result.final_reply == "好，这桌先给你组着。"
