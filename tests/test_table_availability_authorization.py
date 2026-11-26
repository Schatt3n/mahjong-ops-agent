"""验证独立开场核对只能引用同一任务本人来话，且不会重复消耗或放大权限。"""

from dataclasses import replace
from datetime import datetime, timedelta
import json

import pytest

from mahjong_agent_runtime import (
    AgentAction, InMemoryAgentStore, InMemoryTraceRecorder, SQLiteAgentStore, ToolCall, ToolGateway, UserMessage,
)
from mahjong_agent_runtime.domains.table_availability_authorization import check_table_availability, qualified_table_clock
from mahjong_agent_runtime.models import using_clock
from mahjong_agent_runtime.runtime import message_idempotency_key
from mahjong_agent_runtime.services import ToolExecutionService
from mahjong_agent_runtime.tool_consistency import validate_table_availability


SENT = datetime.fromisoformat("2026-11-26T09:07:00+08:00")
ORIGINAL = "行 帮我凑一桌 红中2块3块的 我九点半前能到 能打到晚上"
SAME_DAY_WINDOW = {"start_at": "2026-11-26T09:07:00+08:00", "end_at": "2026-11-26T18:00:00+08:00"}
MULTI_SENT = SENT.replace(hour=19, minute=37)
MULTI_ORIGINAL = "老板，帮我组一桌吧。1 到 3 档，我 8 点 10 分左右到，打到 10 点。"


class ReviewClient:
    """只模拟独立模型的结构化结果，捕获原样发出的证据和调用次数。"""

    def __init__(self, answers):
        self.answers = iter(answers)
        self.calls = []

    def complete(self, messages, **options):
        self.calls.append((messages, options))
        answer = next(self.answers)
        if isinstance(answer, Exception):
            raise answer
        return answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)


def _incoming(text=ORIGINAL, *, sent=SENT, message_id="current", sender_id="guest"):
    """创建给同一顾客的输入，时间和版本后续盖章由 Store 完成。"""
    return UserMessage("dm", sender_id, "顾客", text, message_id=message_id, sent_at=sent)


def _activate(store, *, started=SENT, reset=False):
    """打开本人任务，使历史过滤基于真实任务身份。"""
    return store.activate_task_context(conversation_id="dm", customer_id="guest", trace_id="task",
                                       activity_at=started, started_at=started, reason="test",
                                       force_new=reset, archive_previous=reset)


def _persist(store, message):
    """使用真实接收事务记录本轮消息标识与版本。"""
    store.accept_user_input(message, trace_id=message.message_id, message_key=message_idempotency_key(message))


def _decision(current=1, **patch):
    """模拟正常拒绝固定开场的客户意思，个别测试修改待验证字段。"""
    return {
        "clock_role": "arrival_by", "lookup_scope": "personal", "opening_window": SAME_DAY_WINDOW,
        "accepted_stake_range": {"min": 2, "max": 3}, "known_cap_score": None,
        "has_fixed_opening_authority": False, "current_ref": current, "supporting_customer_refs": [current],
        **patch,
    }


def _check(store, client, message, trace=None, *, step=1):
    """贴近执行入口调用，不给独立模型传待审工具参数。"""
    return check_table_availability(store, client, message, trace_id=message.message_id, step_index=step,
                                    timeout_seconds=12.0, trace_recorder=trace or InMemoryTraceRecorder())


@pytest.mark.parametrize("text,qualifier", [
    (ORIGINAL, "前"),
    ("晚上九点半以后才有空", "以后"),
    ("开场八点左右行吗", "左右"),
    ("九点半到", None),
    ("有现成局吗", None),
])
def test_single_clock_observes_qualifiers_without_inferring_role(text, qualifier):
    """资格探测仅识别钟点结构，不判断钟点的业务作用。"""
    clock = qualified_table_clock(_incoming(text))
    assert (clock["qualifier"] if clock else None) == qualifier
    assert clock is None or clock["clock_count"] == 1
    if text == "九点半到":
        assert clock["clock_text"] == "九点半"


def test_multi_clock_eligibility_preserves_order_exact_minutes_and_each_qualifier():
    """词面证据保留每个钟点，带空格分钟和未限定的离店时刻各自有可核对编号。"""
    observation = qualified_table_clock(_incoming(MULTI_ORIGINAL, sent=MULTI_SENT))
    assert observation["clock_count"] == 2
    assert [(item["clock_ref"], item["clock_text"], item["local_candidate_at"], item["qualifier"])
            for item in observation["candidates"]] == [
        (1, "8 点 10 分", "2026-11-26T20:10:00+08:00", "左右"),
        (2, "10 点", "2026-11-26T22:00:00+08:00", None),
    ]
    mixed = qualified_table_clock(_incoming("九点半前能到，11:00开打吗"))
    assert [(item["clock_ref"], item["qualifier"]) for item in mixed["candidates"]] == [(1, "前"), (2, None)]


@pytest.mark.parametrize("text,role,scope,hard", [
    ("这会儿有合适的吗 打到十一点", "departure_deadline", "personal", True),
    ("现在有桌吗，想打到十一点", "play_until", "personal", False),
    ("十一点左右要走，有合适的吗", "departure_deadline", "personal", False),
    ("十一点那桌还有位吗", "opening", "specific_table", False),
])
def test_departure_filter_requires_exact_hard_personal_constraint(text, role, scope, hard):
    """同一钟点可以是硬离店、柔性希望或原桌开场，仅有证据的硬离店成为个人查询合同。"""
    store = InMemoryAgentStore()
    sent_at = datetime.fromisoformat("2026-11-25T19:00:00+08:00")
    _activate(store, started=sent_at)
    message = _incoming(text, sent=sent_at)
    _persist(store, message)
    raw = _decision(lookup_scope=scope, opening_window=None, accepted_stake_range=None, clock_role=role)
    checked = _check(store, ReviewClient([raw]), message)
    deadline = "2026-11-25T23:00:00+08:00"
    assert checked.get("must_end_by") == (deadline if hard else None)
    missing = ToolCall("search_current_games", {"requirement": {}})
    filtered = ToolCall("search_current_games", {"requirement": {"must_end_by": deadline}})
    assert bool(validate_table_availability(missing, checked)) is hard
    assert bool(validate_table_availability(filtered, checked)) is not hard


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_named_unscheduled_table_still_found_after_rejecting_personal_arrival_filter(store_kind, tmp_path):
    """最早到店不可抹掉原桌；真实查局、回复反馈与持久座位分别验证。"""
    at = datetime.fromisoformat("2026-11-26T18:39:30+08:00")
    arrival = "2026-11-26T18:48:00+08:00"
    store = InMemoryAgentStore() if store_kind == "memory" else SQLiteAgentStore(tmp_path / "named-table.sqlite3")
    message = _incoming("老板好 那个川麻2-3底无烟的还有位吗 我这边最快18:48到", sent=at)
    review = ReviewClient([_decision(lookup_scope="specific_table", clock_role="arrival_not_before",
                                      opening_window=None, accepted_stake_range=None)])
    trace = InMemoryTraceRecorder()
    service = ToolExecutionService(store=store, tool_gateway=ToolGateway(store), trace_recorder=trace,
                                   table_availability_client=review)
    terms = {"game_type": "sichuan_mahjong", "smoke_preference": "no_smoking", "stake_range": {"min": 2, "max": 3}}
    with using_clock(lambda: at):
        _activate(store, started=at)
        _persist(store, message)
        original, _ = store.create_game(conversation_id="other-owner-dm", organizer_id="table-host", organizer_name="桌主",
            requirement=terms | {"start_time_kind": "asap_when_full"}, known_players=[{"customer_id": "table-host"}],
            trace_id="named-board-table")
        assert original.planned_start_at is None and original.remaining_seats() == 3
        original_state = original.to_dict()

        def submit(requirement, step):
            call = ToolCall("search_current_games", {"requirement": requirement})
            action = AgentAction(goal="查看指定桌的余位", objective_status="needs_tool", reasoning_summary="查原桌", tool_calls=[call])
            return service.execute_tool_calls(action, message=message, trace_id=message.message_id,
                previous_step_tool_results=[], step_index=step, run_id="named-table-run",
                run_version=store.conversation_version(message.conversation_id)).tool_results[0]

        rejected = submit(terms | {"start_time_range": {"start_at": arrival, "end_at": "2026-11-26T23:59:00+08:00"}}, 1)
        assert not rejected.called and not rejected.allowed and "原桌" in rejected.result["instruction"]
        evidence = rejected.result["table_availability_evidence"]
        assert evidence["personal_clocks_not_opening"] == [arrival]
        for fixed in ({"start_time": "18:48"}, {"planned_start_at": "2026-11-26T10:48:00Z"}):
            assert validate_table_availability(ToolCall("search_current_games", {"requirement": terms | fixed}), evidence)
        outcome = submit(terms, 2)
        assert outcome.called and outcome.allowed and outcome.error is None
        assert [(item["game"]["game_id"], item["game"]["planned_start_at"]) for item in outcome.result["matches"]] == [(original.game_id, None)]
        assert len(review.calls) == 1
        assert set(store.games) == {original.game_id}
        assert store.require_game(original.game_id).to_dict() == original_state
        assert store.room_reservations == {}


@pytest.mark.parametrize("text,also_personal,opens", [
    ("18:30开那桌还缺人吗？我最快18:48到", False, "18:30"),
    ("18:48开那桌还缺人吗？我最快18:48到", False, "18:48"),
    ("18:30开那桌还缺人吗？我最快18:48到，别的也帮我看看", True, "18:30"),
])
def test_named_table_keeps_its_own_opening_and_separate_alternative_search(text, also_personal, opens):
    """用每个钟点的实际角色校验，保留原桌开场及另查适合顾客的选项。"""
    at = datetime.fromisoformat("2026-11-26T18:15:00+08:00")
    store = InMemoryAgentStore()
    message = _incoming(text, sent=at)
    _activate(store, started=at)
    _persist(store, message)
    raw = _decision(lookup_scope="specific_table", clock_role="other", opening_window=None, accepted_stake_range=None,
                    also_requests_personal_options=also_personal, clock_roles=[
                        {"clock_ref": 1, "role": "opening"}, {"clock_ref": 2, "role": "arrival_not_before"},
                    ])
    evidence = _check(store, ReviewClient([raw]), message)
    assert evidence is not None
    scope = {"lookup_scope": "specific_table"} if also_personal else {}
    point = ToolCall("search_current_games", {"requirement": {"start_time": opens}, **scope})
    assert validate_table_availability(point, evidence) is None
    arrived = ToolCall("search_current_games", {"requirement": {"start_time": "18:48"}, **scope})
    assert bool(validate_table_availability(arrived, evidence)) is (opens != "18:48")
    if also_personal:
        no_purpose = ToolCall("search_current_games", {"requirement": {"start_time": opens}})
        assert "lookup_scope" in validate_table_availability(no_purpose, evidence)
        window = {"start_at": "2026-11-26T18:48:00+08:00", "end_at": "2026-11-26T23:59:00+08:00"}
        personal = ToolCall("search_current_games", {"requirement": {"start_time_range": window},
                                                      "lookup_scope": "personal", "include_nearby": False})
        assert validate_table_availability(personal, evidence) is None


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_mixed_named_and_personal_reads_keep_separate_verified_conditions_and_results(store_kind, tmp_path):
    """同轮两次读取共享原话核实，独立查、独立回灌，个人空范围不能覆盖存在的原桌。"""
    from mahjong_agent_runtime.customer_visible_review import _review_game_availability_evidence
    from mahjong_agent_runtime.tool_consistency import executed_requirement_contract

    at = datetime.fromisoformat("2026-11-26T17:09:31+08:00")
    arrival, departure = "2026-11-26T19:00:00+08:00", "2026-11-26T21:30:00+08:00"
    message = _incoming("现在凑人的那桌是川麻？也想看看红中1块的，我最早七点到，九点半必须走。", sent=at)
    store = InMemoryAgentStore() if store_kind == "memory" else SQLiteAgentStore(tmp_path / "mixed-lookup.sqlite3")
    trace = InMemoryTraceRecorder()
    review = ReviewClient([_decision(lookup_scope="specific_table", clock_role="other", opening_window=None,
        accepted_stake_range=None, also_requests_personal_options=True, clock_roles=[
            {"clock_ref": 1, "role": "arrival_not_before"}, {"clock_ref": 2, "role": "departure_deadline"},
        ])])
    service = ToolExecutionService(store=store, tool_gateway=ToolGateway(store), trace_recorder=trace,
                                   table_availability_client=review)
    with using_clock(lambda: at):
        _activate(store, started=at)
        _persist(store, message)
        original, _ = store.create_game(conversation_id="board-dm", organizer_id="host", organizer_name="桌主",
            requirement={"game_type": "sichuan_mahjong", "stake": "1", "start_time_kind": "asap_when_full"},
            known_players=[{"customer_id": "host"}], trace_id="board-dm")
        before = original.to_dict()
        window = {"start_at": arrival, "end_at": "2026-11-26T23:59:00+08:00"}

        def execute(calls, step):
            action = AgentAction(goal="原桌与另查", objective_status="needs_tool", reasoning_summary="分开查",
                                 tool_calls=calls)
            return service.execute_tool_calls(action, message=message, trace_id=message.message_id,
                previous_step_tool_results=[], step_index=step, run_id="mixed-table-run",
                run_version=store.conversation_version(message.conversation_id)).tool_results

        drift, missing = execute([
            ToolCall("search_current_games", {"requirement": {"start_time_range": {**window, "start_at": at.isoformat()}},
                                              "lookup_scope": "specific_table"}, call_id="original-bad", depends_on=[]),
            ToolCall("search_current_games", {"requirement": {}}, call_id="no-purpose", depends_on=[]),
        ], 1)
        assert not drift.called and "原桌查找不按开场区间" in drift.error
        assert not missing.called and "lookup_scope" in missing.error
        personal, named = execute([
            ToolCall("search_current_games", {"requirement": {"game_type": "red_center_mahjong", "stake": "1",
                "start_time_range": window, "must_end_by": departure}, "lookup_scope": "personal", "include_nearby": False},
                call_id="red-center", depends_on=[]),
            ToolCall("search_current_games", {"requirement": {}, "lookup_scope": "specific_table"}, call_id="board", depends_on=[]),
        ], 2)
        assert personal.called and personal.allowed and personal.result["matches"] == []
        assert personal.result["requirement"]["must_end_by"] == departure
        assert named.called and named.allowed and [row["game"]["game_id"] for row in named.result["matches"]] == [original.game_id]
        assert named.result["requirement"] == {}
        assert len(review.calls) == 1
        outcome = _review_game_availability_evidence([personal.to_dict(), named.to_dict()], [])
        assert outcome["searches_by_scope"] == {
            "personal": {"status": "completed", "matched_count": 0, "call_id": "red-center"},
            "specific_table": {"status": "completed", "matched_count": 1, "call_id": "board"},
        }
        continuity = executed_requirement_contract([personal, named])
        assert continuity["source_call_id"] == "red-center"
        assert continuity["preserve_requirement"]["game_type"] == "red_center_mahjong"
        assert set(store.games) == {original.game_id} and store.require_game(original.game_id).to_dict() == before


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
@pytest.mark.parametrize("verified_cap", [None, 16])
def test_verified_single_stake_allows_exact_search_without_widening_or_changing_cap(store_kind, verified_cap, tmp_path):
    """固定底注和同点区间都能正式查局，区间扩大和封顶偏离仍在执行前被拒。"""
    at = SENT.replace(hour=18)
    store = InMemoryAgentStore() if store_kind == "memory" else SQLiteAgentStore(tmp_path / "single-stake.sqlite3")
    cap_text = "，封顶16" if verified_cap is not None else ""
    message = _incoming(f"晚上七点半开，有杭麻1块底的现成桌吗{cap_text}？", sent=at)
    opening = "2026-11-26T19:30:00+08:00"
    single_stake = {"min": 1, "max": 1}
    verified = _decision(clock_role="opening", opening_window={"start_at": opening, "end_at": opening},
                         has_fixed_opening_authority=True, accepted_stake_range=single_stake,
                         known_cap_score=verified_cap)
    client, trace = ReviewClient([verified]), InMemoryTraceRecorder()
    service = ToolExecutionService(store=store, tool_gateway=ToolGateway(store), trace_recorder=trace,
                                   table_availability_client=client)
    cap_terms = {"cap_score": verified_cap} if verified_cap is not None else {}
    table_terms = {"game_type": "hangzhou_mahjong", "planned_start_at": opening}
    with using_clock(lambda: at):
        _activate(store, started=at)
        _persist(store, message)
        game, _ = store.create_game(conversation_id="one-stake-host", organizer_id="host", organizer_name="桌主",
            known_players=[{"customer_id": "host"}], trace_id="one-stake-host",
            requirement=table_terms | {"stake": "1"} | cap_terms)
        initial_game = game.to_dict()

        def search(terms, step):
            action = AgentAction(goal="查同一底注的现成桌", objective_status="needs_tool", reasoning_summary="核实牌桌",
                                 tool_calls=[ToolCall("search_current_games", {"requirement": table_terms | terms})])
            return service.execute_tool_calls(action, message=message, trace_id=message.message_id,
                previous_step_tool_results=[], step_index=step, run_id="single-stake-run",
                run_version=store.conversation_version(message.conversation_id)).tool_results[0]

        broader = search({"stake_range": {"min": 1, "max": 2}} | cap_terms, 1)
        changed_cap = search({"stake": "1", "cap_score": 24}, 2)
        for rejected in (broader, changed_cap):
            assert not rejected.called and not rejected.allowed
            assert rejected.result["validation_error"]["code"] == "verified_table_availability_conflict"
            evidence = rejected.result["table_availability_evidence"]
            assert evidence["accepted_stake_range"] == single_stake and evidence["known_cap_score"] == verified_cap
        for step, terms in ((3, {"stake": "1"}), (4, {"stake_range": single_stake})):
            result = search(terms | cap_terms, step)
            assert result.called and result.allowed and result.error is None
            assert [row["game"]["game_id"] for row in result.result["matches"]] == [game.game_id]
        assert len(client.calls) == 1
        assert set(store.games) == {game.game_id} and store.require_game(game.game_id).to_dict() == initial_game
        assert not store.room_reservations


def test_verified_personal_departure_is_required_on_executed_read_instead_of_invented_hours():
    """通过正常工具服务核实当前无后缀钟点，保留绝对结束界限并拒绝用统一小时上限替代。"""
    at = datetime.fromisoformat("2026-11-25T19:00:00+08:00")
    store = InMemoryAgentStore()
    message = _incoming("这会儿有合适的吗 打到十一点", sent=at)
    bridge = ReviewClient([_decision(clock_role="departure_deadline", opening_window=None, accepted_stake_range=None)])
    trace = InMemoryTraceRecorder()
    service = ToolExecutionService(store=store, tool_gateway=ToolGateway(store), trace_recorder=trace,
                                   table_availability_client=bridge)
    deadline = "2026-11-25T23:00:00+08:00"
    with using_clock(lambda: at):
        _activate(store, started=at)
        _persist(store, message)
        for label, start, length in (("too-long", "19:30", 4), ("fits", "19:45", 2)):
            store.create_game(conversation_id=label, organizer_id=label, organizer_name="桌主",
                requirement={"planned_start_at": f"2026-11-25T{start}:00+08:00", "duration_hours": length},
                known_players=[{"customer_id": label, "display_name": "桌主"}], trace_id=label)

        def submit(terms, step):
            call = ToolCall("search_current_games", {"requirement": terms})
            action = AgentAction(goal="查十一点前能结束的桌", objective_status="needs_tool", reasoning_summary="先查",
                                 tool_calls=[call])
            return service.execute_tool_calls(action, message=message, trace_id=message.message_id,
                previous_step_tool_results=[], step_index=step, run_id="deadline-run",
                run_version=store.conversation_version(message.conversation_id)).tool_results[0]

        bad = submit({"max_duration_hours": 4}, 1)
        assert not bad.called and f"must_end_by={deadline}" in bad.error
        invented_cap = submit({"must_end_by": deadline, "max_duration_hours": 4}, 2)
        assert not invented_cap.called and "小时上限" in invented_cap.error
        bad_date = submit({"must_end_by": "2026-11-25T23:30:00+08:00"}, 3)
        assert not bad_date.called and f"must_end_by={deadline}" in bad_date.error
        good = submit({"must_end_by": deadline}, 4)
    assert good.called and good.allowed and good.result["requirement"]["must_end_by"] == deadline
    only_match, = good.result["matches"]
    assert only_match["game"]["planned_start_at"] == "2026-11-25T19:45:00+08:00"
    assert not good.result["alternatives"] and len(bridge.calls) == 1
    assert len([event for event in trace.get_trace(message.message_id) if event.step == "tool_called"]) == 1


def _multi_decision(*roles, **patch):
    """约束独立模型按词面编号写语义判断，模型没有劳作提案的参数。"""
    return _decision(clock_role="other", opening_window=None, accepted_stake_range=None, **{
        "clock_roles": [{"clock_ref": ref, "role": role} for ref, role in enumerate(roles, start=1)], **patch,
    })


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_archived_two_clocks_block_fixed_create_and_invented_duration_before_any_write(backend, tmp_path):
    """原来话通过实际共享工具服务，保留到店和离店，拒绝把 20:10 写成约定开场。"""
    store = SQLiteAgentStore(tmp_path / "multi-clock.sqlite3") if backend == "sqlite" else InMemoryAgentStore()
    message = _incoming(MULTI_ORIGINAL, sent=MULTI_SENT)
    decision = _multi_decision("arrival_estimate", "play_until", opening_customer_ref=None)
    review, trace = ReviewClient([decision]), InMemoryTraceRecorder()
    authorization = ReviewClient([{"scope": "create", "customer_ref": 1, "shop_offer_ref": None}])
    service = ToolExecutionService(store=store, tool_gateway=ToolGateway(store), trace_recorder=trace,
                                   table_availability_client=review, creation_authorization_client=authorization)
    opening, leaving = "2026-11-26T20:10:00+08:00", "2026-11-26T22:00:00+08:00"
    with using_clock(lambda: MULTI_SENT):
        _activate(store, started=MULTI_SENT)
        _persist(store, message)

        def submit(tool, step):
            action = AgentAction(goal="帮本人找一桌", objective_status="needs_tool", reasoning_summary="核对条件",
                                 tool_calls=[tool])
            return service.execute_tool_calls(action, message=message, trace_id=message.message_id,
                previous_step_tool_results=[], step_index=step, run_id="multi-clock-run",
                run_version=store.conversation_version(message.conversation_id)).tool_results[0]

        # 补全合法档位表示，使新建提案能走到本例专门核验的时间守卫。
        wrong = submit(ToolCall("create_game", {"organizer_id": "guest", "organizer_name": "顾客", "known_players": [],
            "requirement": {"stake": "0.5", "start_time_kind": "scheduled", "planned_start_at": opening, "duration_hours": 1.83}}), 1)
        assert not wrong.called and not wrong.allowed and "还没选定" in wrong.error
        proof = wrong.result["table_availability_evidence"]
        assert proof["arrival_lookup_floor_at"] == opening and proof["play_until_at"] == leaving
        assert proof["opening_window"] is None and proof["has_fixed_opening_authority"] is False
        query = {"start_time_range": {"start_at": opening, "end_at": "2026-11-26T21:59:00+08:00"}}
        rejected_read = submit(ToolCall("search_current_games", {"requirement": query | {"duration_hours": 1.83}}), 2)
        assert not rejected_read.called and "固定时长" in rejected_read.error
        allowed_read = submit(ToolCall("search_current_games", {"requirement": query}), 3)
        assert allowed_read.called and allowed_read.allowed and not allowed_read.result["matches"]

    assert not store.games and not store.invite_drafts and not store.room_reservations
    assert not authorization.calls and len(review.calls) == 1
    assert [event.content["call"]["name"] for event in trace.get_trace("current") if event.step == "tool_called"] == [
        "search_current_games",
    ]
    if backend == "sqlite":
        store._connection.close()


def test_explicit_opening_and_departure_can_still_define_a_fixed_game():
    """明确选定的开场与离店边界可以组成合法安排，绝不把离店端写成开场。"""
    at = SENT.replace(hour=17)
    store = InMemoryAgentStore()
    _activate(store, started=at)
    message = _incoming("帮我组一桌，19:00开，21:00我得离开。", sent=at)
    _persist(store, message)
    start = "2026-11-26T19:00:00+08:00"
    end = "2026-11-26T21:00:00+08:00"
    decision = _multi_decision("opening", "departure_deadline") | {
        "opening_window": {"start_at": start, "end_at": start},
        "opening_customer_ref": 1, "has_fixed_opening_authority": True,
    }
    checked = _check(store, ReviewClient([decision]), message)
    assert checked["play_until_at"] == checked["must_end_by"] == end and "arrival_lookup_floor_at" not in checked
    requirement = {"planned_start_at": start, "start_time_kind": "scheduled"}
    with using_clock(lambda: at):
        assert "max_duration_hours=2.0" in validate_table_availability(
            ToolCall("create_game", {"requirement": requirement}), checked,
        )
        assert "max_duration_hours=2.0" in validate_table_availability(
            ToolCall("create_game", {"requirement": requirement | {"duration_hours": 2}}), checked,
        )
        assert validate_table_availability(
            ToolCall("create_game", {"requirement": requirement | {"max_duration_hours": 2}}), checked,
        ) is None
        assert "must_end_by" in validate_table_availability(ToolCall("search_current_games", {"requirement": requirement}), checked)
        assert validate_table_availability(ToolCall("search_current_games", {
            "requirement": requirement | {"must_end_by": end},
        }), checked) is None
        assert "时段" in validate_table_availability(ToolCall("create_game", {
            "requirement": requirement | {"planned_start_at": end},
        }), checked)
        confirmed_duration = checked | {"agreed_duration_hours": 1}
        assert validate_table_availability(
            ToolCall("create_game", {"requirement": requirement | {"duration_hours": 1}}), confirmed_duration,
        ) is None
        assert "已约定的 duration_hours=1" in validate_table_availability(
            ToolCall("create_game", {"requirement": requirement | {"max_duration_hours": 2}}), confirmed_duration,
        )


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_explicit_archived_control_retries_with_verified_cap_and_persists_same_start(backend, tmp_path):
    """已核实开场的建局遇到默认时长过长，回灌可执行上限后按同一开场实际落库。"""
    store = SQLiteAgentStore(tmp_path / "explicit-clock.sqlite3") if backend == "sqlite" else InMemoryAgentStore()
    message = _incoming("老板，帮我组一桌吧。1 到 3 档，这桌就晚上八点半开，我十点要走。", sent=MULTI_SENT)
    open_at, depart_at = "2026-11-26T20:30:00+08:00", "2026-11-26T22:00:00+08:00"
    stakes = {"min": 1, "max": 3}
    decision = _multi_decision("opening", "departure_deadline") | {
        "accepted_stake_range": stakes, "opening_window": {"start_at": open_at, "end_at": open_at},
        "has_fixed_opening_authority": True, "opening_customer_ref": 1,
    }
    review, trace = ReviewClient([decision]), InMemoryTraceRecorder()
    authorization = ReviewClient([{"scope": "create", "customer_ref": 1, "shop_offer_ref": None}])
    service = ToolExecutionService(store=store, tool_gateway=ToolGateway(store), trace_recorder=trace,
                                   table_availability_client=review, creation_authorization_client=authorization)
    requirement = {"planned_start_at": open_at, "start_time_kind": "scheduled", "stake_range": stakes}
    with using_clock(lambda: MULTI_SENT):
        _activate(store, started=MULTI_SENT)
        _persist(store, message)

        def submit(terms, step):
            call = ToolCall("create_game", {"organizer_id": "guest", "organizer_name": "顾客", "known_players": [],
                                           "requirement": terms})
            action = AgentAction(goal="照已核实的开场组一桌", objective_status="needs_tool", reasoning_summary="组桌",
                                 tool_calls=[call])
            return service.execute_tool_calls(action, message=message, trace_id=message.message_id,
                previous_step_tool_results=[], step_index=step, run_id="explicit-clock-run",
                run_version=store.conversation_version(message.conversation_id)).tool_results[0]

        default_overrun = submit(requirement, 1)
        assert not default_overrun.called and not store.games and not authorization.calls
        assert all(value in default_overrun.error for value in (open_at, "max_duration_hours=1.5", "重试建局", "无需再问"))
        corrected = submit(requirement | {"max_duration_hours": 1.5}, 2)
        assert corrected.called and corrected.allowed and not corrected.error
        game, = store.games.values()
        assert game.planned_start_at.isoformat() == open_at and game.planned_end_at.isoformat() == depart_at
        assert game.requirement["max_duration_hours"] == 1.5 and "duration_hours" not in game.requirement
    assert len(authorization.calls) == len(review.calls) == 1
    if backend == "sqlite":
        store._connection.close()


@pytest.mark.parametrize("patch", [
    {"clock_roles": None},
    {"clock_roles": [{"clock_ref": 1, "role": "arrival_estimate"}, {"clock_ref": 1, "role": "play_until"}]},
    {"clock_roles": [{"clock_ref": 1, "role": "arrival_estimate"}, {"clock_ref": 3, "role": "play_until"}]},
    {"has_fixed_opening_authority": True, "opening_customer_ref": 1,
     "opening_window": {"start_at": "2026-11-26T20:10:00+08:00", "end_at": "2026-11-26T20:10:00+08:00"}},
])
def test_multi_clock_authority_cannot_omit_duplicate_or_relabel_the_opening_source(patch):
    """每个词面候选需独立回传，固定开场不能引用只有到店和离店角色的当前材料。"""
    store = InMemoryAgentStore()
    _activate(store, started=MULTI_SENT)
    message = _incoming(MULTI_ORIGINAL, sent=MULTI_SENT)
    _persist(store, message)
    decision = _multi_decision("arrival_estimate", "play_until") | patch
    assert _check(store, ReviewClient([decision]), message) is None


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_same_goal_history_and_durable_source_version_cache(backend, tmp_path):
    """原 morning 来话的本人时段仍可引用，过期任务、未交付回复和其他顾客不会串入。"""
    path = tmp_path / "table-availability.sqlite3"
    store = SQLiteAgentStore(path) if backend == "sqlite" else InMemoryAgentStore()
    _activate(store, started=SENT - timedelta(hours=8))
    store.append_user_turn(_incoming("我想打旧任务今晚九点的", sent=SENT - timedelta(hours=8), message_id="old"), "old")
    _activate(store, started=SENT - timedelta(minutes=2), reset=True)
    store.append_user_turn(_incoming("老板今天有合适的桌吗 红中2块3块的 我时长能调 上午下午都行",
                                     sent=SENT - timedelta(minutes=2), message_id="prior"), "prior")
    store.append_user_turn(_incoming("另一客户晚上开", sent=SENT - timedelta(minutes=2),
                                     message_id="other", sender_id="other"), "other")
    with using_clock(lambda: SENT - timedelta(minutes=1)):
        store.append_assistant_turn("dm", "要不要我帮你问问有没有人", "shop", metadata={"delivery_status": "delivered"})
        store.append_assistant_turn("dm", "还没发出的假提议", "unsent", metadata={"delivery_status": "pending"})
    message = _incoming()
    _persist(store, message)
    decision = _decision(3, supporting_customer_refs=[1, 3])
    client = ReviewClient([decision])
    trace = InMemoryTraceRecorder()
    checked = _check(store, client, message, trace)
    assert checked == decision and checked["has_fixed_opening_authority"] is False
    checked["accepted_stake_range"]["min"] = 100
    if backend == "sqlite":
        store._connection.close()
        store = SQLiteAgentStore(path)
    assert _check(store, client, message, trace, step=2) == decision
    assert len(client.calls) == 1
    prompt, options = client.calls[0]
    payload = json.loads(prompt[1]["content"])
    assert set(payload) == {"conversation", "customer_message_local_at", "current_clock_observation"}
    assert [row["speaker"] for row in payload["conversation"]] == ["customer", "shop", "customer"]
    assert "上午下午都行" in payload["conversation"][0]["text"]
    assert payload["conversation"][2]["text"] == ORIGINAL
    assert payload["current_clock_observation"]["local_candidate_at"].endswith("09:30:00+08:00")
    assert options["trace_id"] == "current:table-availability:1" and options["timeout_seconds"] == 12.0
    assert [event.step for event in trace.get_trace("current")] == [
        "table_availability_authorization_prompt", "table_availability_authorization_result",
        "table_availability_authorization_reused",
    ]
    assert not store.games
    if backend == "sqlite":
        store._connection.close()


@pytest.mark.parametrize("text,role,start_time", [
    (ORIGINAL, "arrival_by", "09:07"),
    ("我上午九点半以后才到，能帮我凑一桌不", "arrival_not_before", "09:30"),
    ("我上午九点半左右能到，能帮我凑一桌不", "arrival_estimate", "09:30"),
    ("我今天18点半以后才到", "arrival_not_before", None),
    ("我打到晚上九点半左右就得走", "play_until", "09:07"),
])
def test_verified_opening_dayparts_fill_missing_exact_window(text, role, start_time):
    """上午和下午开都行时能查完整开场范围，到店上界和打到几点不切掉原时间。"""
    store = InMemoryAgentStore()
    prior_at = SENT - timedelta(minutes=2)
    _activate(store, started=prior_at)
    _persist(store, _incoming("老板今天有合适的桌吗 红中2块3块的 我时长能调 上午下午都行",
                              sent=prior_at, message_id="prior"))
    message = _incoming(text)
    _persist(store, message)
    dayparts = {"date": "2026-11-26", "parts": ["morning", "afternoon"], "source_ref": 1}
    decision = _decision(2, clock_role=role, opening_window=None, opening_dayparts=dayparts,
                         supporting_customer_refs=[1, 2])
    trace = InMemoryTraceRecorder()
    checked = _check(store, ReviewClient([decision]), message, trace)
    expected = ({**SAME_DAY_WINDOW, "start_at": f"2026-11-26T{start_time}:00+08:00"} if start_time else None)
    expected_floor = {"arrival_not_before": "18:30" if start_time is None else "09:30",
                      "arrival_estimate": "09:30"}.get(role)
    known_times = {"arrival_lookup_floor_at": f"2026-11-26T{expected_floor}:00+08:00"} if expected_floor else {}
    if role == "play_until":
        known_times["play_until_at"] = "2026-11-26T21:30:00+08:00"
    assert checked == {**decision, "opening_window": expected, **known_times}
    assert trace.get_trace("current")[-1].content["decision"]["opening_window"] == expected


@pytest.mark.parametrize("scope,parts,exact,end_hour,expected_hour", [
    ("personal", ["afternoon", "morning"], "12:15", "16:00", "16:00"),
    ("personal", ["afternoon", "morning"], "12:15", "21:30", "18:00"),
    ("personal", ["morning", "evening"], "20:00", "21:30", "21:30"),
    ("personal", ["morning", "evening"], "09:00", "21:30", None),
    ("personal", ["morning", "evening"], None, None, None),
    ("specific_table", ["morning", "afternoon"], None, None, None),
    ("other", ["morning", "afternoon"], None, None, None),
])
def test_exact_opening_window_preserved_only_inside_applicable_dayparts(scope, parts, exact, end_hour, expected_hour):
    """已有精确开场条件优先，分开的时段、具体桌和其他目标不自动补成完整窗口。"""
    store = InMemoryAgentStore()
    prior_at = SENT - timedelta(minutes=2)
    _activate(store, started=prior_at)
    prior_text = "上午晚上开都行" if "evening" in parts else "上午下午开都行"
    _persist(store, _incoming(prior_text, sent=prior_at, message_id="prior"))
    message = _incoming("晚上九点半前能打完就行")
    _persist(store, message)
    opening_window = {"start_at": f"2026-11-26T{exact}:00+08:00",
                      "end_at": f"2026-11-26T{end_hour}:00+08:00"} if exact else None
    dayparts = {"date": "2026-11-26", "parts": parts, "source_ref": 1}
    decision = _decision(2, lookup_scope=scope, clock_role="play_until", opening_window=opening_window,
                         opening_dayparts=dayparts, supporting_customer_refs=[1, 2])
    checked = _check(store, ReviewClient([decision]), message)
    expected = {**opening_window, "end_at": f"2026-11-26T{expected_hour}:00+08:00"} if expected_hour else None
    assert checked == {**decision, "opening_window": expected,
                       "opening_dayparts": dayparts if scope == "personal" else None,
                       **({"play_until_at": "2026-11-26T21:30:00+08:00"} if scope == "personal" else {}),
                       **({"personal_clocks_not_opening": ["2026-11-26T21:30:00+08:00"]} if scope == "specific_table" else {})}


@pytest.mark.parametrize("patch", [
    {"date": "2026-11-27"},
    {"date": "20261126"},
    {"source_ref": 2},
    {"source_ref": 3},
    {"parts": ["morning", "morning"]},
    {"parts": ["daylight"]},
])
def test_opening_dayparts_require_verified_citation_and_local_date(patch):
    """时段的专门引用必须是本判断所引用的本人来话，禁止引用店家、跨日猜测或随意扩展枚举。"""
    store = InMemoryAgentStore()
    prior_at = SENT - timedelta(minutes=2)
    _activate(store, started=prior_at)
    _persist(store, _incoming("今天上午下午有合适的桌吗", sent=prior_at, message_id="prior"))
    with using_clock(lambda: SENT - timedelta(minutes=1)):
        store.append_assistant_turn("dm", "下午也有", "shop", metadata={"delivery_status": "delivered"})
    message = _incoming()
    _persist(store, message)
    dayparts = {"date": "2026-11-26", "parts": ["morning", "afternoon"], "source_ref": 1, **patch}
    invalid = _decision(3, opening_window=None, opening_dayparts=dayparts, supporting_customer_refs=[1])
    without_dayparts = _decision(3, opening_window=None, opening_dayparts=None, supporting_customer_refs=[1, 3])
    client = ReviewClient([invalid, without_dayparts])
    assert _check(store, client, message) is None
    assert _check(store, client, message) == without_dayparts


def test_explicit_future_opening_daypart_uses_actual_date_and_local_midnight():
    """有本人明确指定的开场日期才采用未来时段，并保持晚上查询在当天结束。"""
    store = InMemoryAgentStore()
    prior_at = SENT - timedelta(minutes=2)
    _activate(store, started=prior_at)
    _persist(store, _incoming("下周三下午或者晚上开都行", sent=prior_at, message_id="prior"))
    message = _incoming("晚上九点半前我能到")
    _persist(store, message)
    dayparts = {"date": "2026-12-02", "parts": ["afternoon", "evening"], "source_ref": 1}
    decision = _decision(2, opening_window=None, opening_dayparts=dayparts, supporting_customer_refs=[1, 2])
    expected = {"start_at": "2026-12-02T12:00:00+08:00", "end_at": "2026-12-02T23:59:59.999999+08:00"}
    assert _check(store, ReviewClient([decision]), message) == {**decision, "opening_window": expected}


def test_source_bound_evening_arrival_estimate_and_maximum_are_distinct_from_opening():
    """本人凑桌续句核实到店估计和最长时长，但不产生已选定的新局开场。"""
    at = SENT.replace(hour=11)
    store = InMemoryAgentStore()
    prior_at = at - timedelta(minutes=2)
    _activate(store, started=prior_at)
    _persist(store, _incoming("老板好 现在还有位吗 杭州麻将1到3块底 我七点一刻左右能到",
                              sent=prior_at, message_id="prior"))
    message = _incoming("老板好 帮我凑一桌吧 杭州麻将1到3块底 我七点一刻左右能到 最多打三小时", sent=at)
    _persist(store, message)
    decision = _decision(2, clock_role="arrival_estimate", opening_window=None, opening_dayparts=None,
                         accepted_stake_range={"min": 1, "max": 3}, max_duration_hours=3,
                         supporting_customer_refs=[1, 2])
    assert _check(store, ReviewClient([decision]), message) == {
        **decision, "arrival_lookup_floor_at": "2026-11-26T19:15:00+08:00",
    }


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_evening_search_separates_earlier_nearby_and_excludes_explicit_duration_conflict(backend, tmp_path):
    """真实查局前先核对到店用途与时长，近时备选不混入可直接匹配的牌局。"""
    at = SENT.replace(hour=11)
    store = SQLiteAgentStore(tmp_path / "evening.sqlite3") if backend == "sqlite" else InMemoryAgentStore()
    message = _incoming("老板好 帮我凑一桌吧 杭州麻将1到3块底 我七点一刻左右能到 最多打三小时", sent=at)
    stakes = {"min": 1, "max": 3}
    decision = _decision(clock_role="arrival_estimate", opening_window=None, accepted_stake_range=stakes,
                         max_duration_hours=3)
    client, trace = ReviewClient([decision]), InMemoryTraceRecorder()
    service = ToolExecutionService(store=store, tool_gateway=ToolGateway(store), trace_recorder=trace,
                                   table_availability_client=client)
    games = {}
    with using_clock(lambda: at):
        _activate(store, started=at)
        _persist(store, message)
        for opening, hours in (("18:00", 2), ("19:08", 2.5), ("19:30", 3), ("19:45", 4), ("20:20", 2)):
            host = f"host-{opening}"
            game, _ = store.create_game(
                conversation_id=host, organizer_id=host, organizer_name="桌主",
                known_players=[{"customer_id": host, "display_name": "桌主"}],
                requirement={"game_type": "hangzhou_mahjong", "stake_range": stakes,
                             "planned_start_at": f"2026-11-26T{opening}:00+08:00", "duration_hours": hours},
                trace_id=host,
            )
            games[opening] = game.game_id

        def query(step, start, **extra):
            tool = ToolCall("search_current_games", {"requirement": {
                "game_type": "hangzhou_mahjong", "stake_range": stakes,
                "start_time_range": {"start_at": f"2026-11-26T{start}:00+08:00",
                                     "end_at": "2026-11-26T23:59:00+08:00"},
                **extra,
            }})
            action = AgentAction(goal="帮顾客凑局", objective_status="needs_tool", reasoning_summary="先查现成桌",
                                 tool_calls=[tool])
            return service.execute_tool_calls(action, message=message, trace_id=message.message_id,
                previous_step_tool_results=[], step_index=step, run_id="evening-run",
                run_version=store.conversation_version(message.conversation_id)).tool_results[0]

        assert "最多可打时长" in query(1, "19:00").error
        assert "19:15:00+08:00" in query(2, "19:00", max_duration_hours=3).error
        result = query(3, "19:15", max_duration_hours=3)

    assert result.called and result.allowed
    assert {row["game"]["game_id"] for row in result.result["matches"]} == {games["19:30"], games["20:20"]}
    nearby, = result.result["nearby_options"]
    assert nearby["game"]["game_id"] == games["19:08"]
    assert nearby["relative_to_window"] == "before" and nearby["requires_timing_confirmation"] is True
    assert not result.result["alternatives"]
    assert len([event for event in trace.get_trace(message.message_id) if event.step == "tool_called"]) == 1
    assert len(client.calls) == 1
    if backend == "sqlite":
        store._connection.close()


@pytest.mark.parametrize("bad", [
    {"current_ref": 2},
    {"current_ref": True},
    {"supporting_customer_refs": [2]},
    {"supporting_customer_refs": []},
    {"supporting_customer_refs": [1, 1]},
    {"clock_role": "start_time"},
    {"lookup_scope": "any"},
    {"known_cap_score": True},
    {"known_cap_score": float("inf")},
    {"known_cap_score": -1},
    {"max_duration_hours": True},
    {"max_duration_hours": float("inf")},
    {"max_duration_hours": -1},
    {"accepted_stake_range": {"min": 3, "max": 2}},
    {"accepted_stake_range": {"min": "2", "max": 3}},
    {"opening_window": {"start_at": "2026-11-26T09:00:00", "end_at": "2026-11-26T18:00:00+08:00"}},
    {"opening_window": {"start_at": "2026-11-26T18:00:01+08:00", "end_at": "2026-11-26T18:00:00+08:00"}},
    {"opening_window": {"start_at": "2026-02-03T10:00:00+08:00", "end_at": "2026-02-03T17:00:00+08:00"}},
    {"opening_window": None, "has_fixed_opening_authority": True},
    {"lookup_scope": "specific_table", "has_fixed_opening_authority": True},
])
def test_invalid_contract_is_not_authority_or_cached(bad):
    """非法引用、未授权窗口、错类型和无穷数不能成为查局或写入证明。"""
    store = InMemoryAgentStore()
    _activate(store)
    message = _incoming("帮我找红中2-3块的，我九点半前能到")
    _persist(store, message)
    client = ReviewClient([_decision(**bad), _decision()])
    trace = InMemoryTraceRecorder()
    assert _check(store, client, message, trace) is None
    assert _check(store, client, message, trace) == _decision()
    assert len(client.calls) == 2
    assert any(event.step == "table_availability_authorization_invalid" for event in trace.get_trace(message.message_id))


def test_supporting_refs_cannot_cite_delivered_shop_or_recalled_user_rows():
    """已送达店家话只帮助理解目标，不能替本人提供开场权限或条件。"""
    store = InMemoryAgentStore()
    _activate(store, started=SENT - timedelta(minutes=2))
    recalled = replace(_incoming("晚上随便开", sent=SENT - timedelta(minutes=2), message_id="recalled"),
                       metadata={"context_invalidated": True})
    store.append_user_turn(recalled, "recalled")
    with using_clock(lambda: SENT - timedelta(minutes=1)):
        store.append_assistant_turn("dm", "晚上7点可以开", "shop", metadata={"delivery_status": "delivered"})
    message = _incoming()
    _persist(store, message)
    client = ReviewClient([_decision(2, supporting_customer_refs=[1]), _decision(2)])
    assert _check(store, client, message) is None
    assert _check(store, client, message) == _decision(2)
    rows = json.loads(client.calls[0][0][1]["content"])["conversation"]
    assert [row["speaker"] for row in rows] == ["shop", "customer"]


@pytest.mark.parametrize("explicit_date", ["2026-12-02", "12月2日", "下周三"])
def test_explicit_future_dates_and_independent_cap_are_preserved(explicit_date):
    """明确的远期计划可合法落在当前日之外，明确封顶和档位区间各自保留。"""
    store = InMemoryAgentStore()
    _activate(store, started=SENT - timedelta(minutes=1))
    store.append_user_turn(_incoming(f"{explicit_date}下午两点到五点，你帮我选个开场，底2到3块，封顶16",
                                     sent=SENT - timedelta(minutes=1), message_id="prior"), "prior")
    message = _incoming(f"{explicit_date}两点前我能到")
    _persist(store, message)
    utc_window = {"start_at": "2026-12-02T06:00:00Z", "end_at": "2026-12-02T09:00:00Z"}
    local_window = {"start_at": "2026-12-02T14:00:00+08:00", "end_at": "2026-12-02T17:00:00+08:00"}
    decision = _decision(2, has_fixed_opening_authority=True, opening_window=utc_window,
                         known_cap_score=16, supporting_customer_refs=[1, 2])
    assert _check(store, ReviewClient([decision]), message) == {**decision, "opening_window": local_window}


def test_failed_transport_or_duplicate_fields_never_grant_or_cache_authority():
    """失败不冒充已核验；同一条真实来话以后可以安全重试独立判断。"""
    store = InMemoryAgentStore()
    _activate(store)
    message = _incoming()
    _persist(store, message)
    duplicate = json.dumps(_decision())[:-1] + ', "clock_role": "opening"}'
    client = ReviewClient([TimeoutError("transport timeout"), duplicate, _decision()])
    trace = InMemoryTraceRecorder()
    assert _check(store, client, message, trace) is None
    assert _check(store, client, message, trace) is None
    assert _check(store, client, message, trace) == _decision()
    assert [event.step for event in trace.get_trace("current")] == [
        "table_availability_authorization_prompt", "table_availability_authorization_unavailable",
        "table_availability_authorization_prompt", "table_availability_authorization_invalid",
        "table_availability_authorization_prompt", "table_availability_authorization_result",
    ]


def test_current_prompt_cannot_be_cited_after_new_receipt_or_be_fabricated():
    """模型返回前新消息抢先落盘时，旧语义结果不授权也不进入缓存。"""
    store = InMemoryAgentStore()
    _activate(store)
    message = _incoming()
    client = ReviewClient([_decision()])
    assert _check(store, client, message) is None and not client.calls
    _persist(store, message)
    assert _check(store, client, replace(message, text="我晚上十点前到")) is None and not client.calls

    class UpdateDuringReview(ReviewClient):
        """模拟后台模型等待期间，同一顾客补发真实消息。"""

        def complete(self, messages, **options):
            _persist(store, _incoming("我再看看", sent=SENT + timedelta(seconds=1), message_id="new"))
            return super().complete(messages, **options)

    client = UpdateDuringReview([_decision()])
    trace = InMemoryTraceRecorder()
    assert _check(store, client, message, trace) is None and len(client.calls) == 1
    assert not store.idempotency_ledger
    assert _check(store, client, message, trace) is None and len(client.calls) == 1
    assert trace.get_trace("current")[-1].step == "table_availability_authorization_superseded"
