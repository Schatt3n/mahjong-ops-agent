"""验证独立开场核对只能引用同一任务本人来话，且不会重复消耗或放大权限。"""

from dataclasses import replace
from datetime import datetime, timedelta
import json

import pytest

from mahjong_agent_runtime import (
    AgentAction, InMemoryAgentStore, InMemoryTraceRecorder, SQLiteAgentStore, ToolCall, ToolGateway, UserMessage,
)
from mahjong_agent_runtime.customer_visible_review import build_reply_self_review_payload
from mahjong_agent_runtime.domains.context_builders.tool_results import compact_tool_payload
from mahjong_agent_runtime.domains.delivered_game_offer import game_offer_for_review
from mahjong_agent_runtime.domains.table_availability_authorization import (
    check_table_availability, qualified_table_clock, requires_table_clock_review,
)
from mahjong_agent_runtime.group_chat.parsing import parse_explicit_need
from mahjong_agent_runtime.models import using_clock
from mahjong_agent_runtime.runtime import message_idempotency_key
from mahjong_agent_runtime.services import ToolExecutionService
from mahjong_agent_runtime.tool_consistency import apply_approximate_opening_search_evidence, validate_table_availability


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


@pytest.mark.parametrize("text,clock,minute,qualifier", [
    ("我七点四十一能到", "七点四十一", 41, None),
    ("七点四十一分以后我才到", "七点四十一分", 41, "以后"),
    ("七点 四十分 左右能来", "七点 四十分", 40, "左右"),
    ("七点零五能到", "七点零五", 5, None),
    ("七点〇五分能到", "七点〇五分", 5, None),
    ("七点五分前能来", "七点五分", 5, "前"),
    ("七点十能到", "七点十", 10, None),
    ("七点五十九能到", "七点五十九", 59, None),
    ("七点三元的还有位吗", "七点", 0, None),
    ("七点三十块封顶的那桌", "七点", 0, None),
    ("七点三十分钟之后回我", "七点", 0, None),
    ("七点\n四十一块的都行", "七点", 0, None),
])
def test_clock_observation_reads_han_minutes_without_absorbing_adjacent_amounts(text, clock, minute, qualifier):
    """完整的汉字分钟与紧邻的金额、时长和另一条气泡分别识别。"""
    at = SENT.replace(hour=18, minute=30)
    observed = qualified_table_clock(_incoming(text, sent=at))
    assert observed["clock_count"] == 1
    assert observed["clock_text"] == clock and observed["qualifier"] == qualifier
    assert observed["local_candidate_at"] == f"2026-11-26T19:{minute:02d}:00+08:00"


@pytest.mark.parametrize("text", [
    "七点十几分能到", "七点四十几能到", "七点四十多分左右能来", "七点六十一分到", "七点三四分能到",
])
def test_unknown_han_minute_does_not_become_a_precise_clock(text):
    """未确定或非法的分钟保留在原话，不能截成整点或某个精确分钟。"""
    assert qualified_table_clock(_incoming(text, sent=SENT.replace(hour=18))) is None


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
            "personal": {"status": "completed", "matched_count": 0, "call_id": "red-center",
                         "submitted_filters_without_results": {"game_type": "red_center_mahjong", "stake": "1",
                                                               "start_time_range": window, "must_end_by": departure}},
            "specific_table": {"status": "completed", "matched_count": 1, "call_id": "board"},
        }
        continuity = executed_requirement_contract([personal, named])
        assert continuity["source_call_id"] == "red-center"
        assert continuity["preserve_requirement"]["game_type"] == "red_center_mahjong"
        assert set(store.games) == {original.game_id} and store.require_game(original.game_id).to_dict() == before


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
@pytest.mark.parametrize("base_missed_other_target", [False, True], ids=["initial-review", "scope-recheck"])
@pytest.mark.parametrize("qualifier,start,end,expected_match", [
    ("以后", "2026-11-26T19:00:00", "2026-11-27T06:00:00", "late"),
    ("以前", "2026-11-26T18:35:00", "2026-11-26T19:00:00", "early"),
])
def test_mixed_lookup_keeps_qualified_opening_bound_when_verifier_returns_only_daypart(
    store_kind, base_missed_other_target, qualifier, start, end, expected_match, tmp_path,
):
    """原桌能独立查到，另查必须保留明确的开场边界，无窗口的模型判定也不能放宽。"""
    at = SENT.replace(hour=18, minute=35)
    store = InMemoryAgentStore() if store_kind == "memory" else SQLiteAgentStore(tmp_path / "mixed-daypart.sqlite3")
    message = _incoming(f"现在凑人的那桌是川麻？另外晚上七点{qualifier}开的1元红中有没有？", sent=at)
    dayparts = {"date": "2026-11-26", "parts": ["evening"], "source_ref": 1}
    decision = _decision(lookup_scope="specific_table", also_requests_personal_options=True,
                         clock_role="opening", opening_window=None, opening_dayparts=dayparts,
                         accepted_stake_range={"min": 1, "max": 1})
    independent = ([{key: value for key, value in (decision | {"opening_dayparts": None}).items()
                     if key != "also_requests_personal_options"}, decision] if base_missed_other_target else [decision])
    client, trace = ReviewClient(independent), InMemoryTraceRecorder()
    service = ToolExecutionService(store=store, tool_gateway=ToolGateway(store), trace_recorder=trace,
                                   table_availability_client=client)
    verified_window = {"start_at": f"{start}+08:00", "end_at": f"{end}+08:00"}
    widened_window = {"start_at": at.isoformat(), "end_at": "2026-11-26T23:59:59.999999+08:00"}
    with using_clock(lambda: at):
        _activate(store, started=at)
        _persist(store, message)
        games = {}
        for host, game_type, opening in (("board", "sichuan_mahjong", {"start_time_kind": "asap_when_full"}),
                                         ("early", "red_center_mahjong", {"planned_start_at": "2026-11-26T18:50:00+08:00"}),
                                         ("late", "red_center_mahjong", {"planned_start_at": "2026-11-27T02:00:00+08:00"}),
                                         ("next_morning", "red_center_mahjong", {"planned_start_at": "2026-11-27T06:01:00+08:00"})):
            games[host], _ = store.create_game(conversation_id=host, organizer_id=host, organizer_name="桌主",
                requirement={"game_type": game_type, "stake": "1", **opening},
                known_players=[{"customer_id": host}], trace_id=host)
        initial_states = {game.game_id: game.to_dict() for game in games.values()}

        def query(scope, terms, step):
            action = AgentAction(goal="分别核对原桌和另查", objective_status="needs_tool", reasoning_summary="分开查询",
                tool_calls=[ToolCall("search_current_games", {"requirement": terms, "lookup_scope": scope,
                                                             "include_nearby": False})])
            return service.execute_tool_calls(action, message=message, trace_id=message.message_id,
                previous_step_tool_results=[], step_index=step, run_id="mixed-daypart-run",
                run_version=store.conversation_version(message.conversation_id)).tool_results[0]

        personal = {"game_type": "red_center_mahjong", "stake": "1"}
        rejected = query("personal", {**personal, "start_time_range": widened_window}, 1)
        assert not rejected.called and not rejected.allowed
        assert rejected.result["validation_error"]["code"] == "verified_table_availability_conflict"
        assert rejected.result["table_availability_evidence"]["opening_window"] == verified_window
        assert all(boundary in rejected.error for boundary in verified_window.values())
        named = query("specific_table", {"game_type": "sichuan_mahjong"}, 2)
        filtered = query("personal", {**personal, "start_time_range": verified_window}, 3)
        wrong_stake = query("personal", {**personal, "stake": "2", "start_time_range": verified_window}, 4)
        for actual, expected in ((named, "board"), (filtered, expected_match)):
            assert actual.called and actual.allowed and actual.error is None
            assert [row["game"]["game_id"] for row in actual.result["matches"]] == [games[expected].game_id]
        assert "start_time_range" not in named.result["requirement"]
        assert filtered.result["requirement"]["start_time_range"] == verified_window
        assert not wrong_stake.called and "档位区间" in wrong_stake.error
        assert len(client.calls) == (2 if base_missed_other_target else 1)
        if base_missed_other_target:
            primary, recheck = client.calls
            assert primary[0][1] == recheck[0][1]
            assert "两项并列的查局目的" in recheck[0][0]["content"]
            assert 0 < recheck[1]["timeout_seconds"] <= primary[1]["timeout_seconds"]
            assert any(event.content.get("review_purpose") == "independent_personal_read"
                       and event.step == "table_availability_authorization_reused"
                       for event in trace.get_trace(message.message_id))
        assert {game_id: store.require_game(game_id).to_dict() for game_id in store.games} == initial_states
        assert not store.room_reservations


@pytest.mark.parametrize("recheck", ["original_only", "personal_only", "no_verified_opening", "invalid", "timeout"])
def test_personal_read_recheck_keeps_scope_and_timing_guards_and_is_not_repeated(recheck):
    """不支持两项、缺少开场证据或核验失败时，个人查询不能绕行。"""
    at = SENT.replace(hour=18, minute=35)
    store = InMemoryAgentStore()
    message = _incoming("刚才那桌是川麻吗？我大约八点到，想看下赶不赶得上。", sent=at)
    original = _decision(lookup_scope="specific_table", clock_role="arrival_estimate", opening_window=None,
                         accepted_stake_range=None)
    recheck_result = {
        "original_only": original,
        "personal_only": original | {"lookup_scope": "personal"},
        "no_verified_opening": original | {"also_requests_personal_options": True, "clock_role": "other"},
        "invalid": original | {"also_requests_personal_options": True, "supporting_customer_refs": [99]},
        "timeout": TimeoutError("recheck timed out"),
    }[recheck]
    client, trace = ReviewClient([original, recheck_result]), InMemoryTraceRecorder()
    service = ToolExecutionService(store=store, tool_gateway=ToolGateway(store), trace_recorder=trace,
                                   table_availability_client=client)
    _activate(store, started=at)
    _persist(store, message)
    window = {"start_at": "2026-11-26T20:00:00+08:00", "end_at": "2026-11-27T06:00:00+08:00"}
    turns = [ToolCall("search_current_games", {"lookup_scope": "personal", "requirement": {"start_time_range": window}},
                      call_id="personal", depends_on=[]),
             ToolCall("search_current_games", {"lookup_scope": "specific_table", "requirement": {"game_type": "sichuan_mahjong"}},
                      call_id="named", depends_on=[])]
    for step in (1, 2):
        action = AgentAction(goal="查局", objective_status="needs_tool", reasoning_summary="核对范围", tool_calls=turns)
        personal, named = service.execute_tool_calls(action, message=message, trace_id=message.message_id,
            previous_step_tool_results=[], step_index=step, run_id="personal-recheck-denied",
            run_version=store.conversation_version(message.conversation_id)).tool_results
        assert not personal.called and not personal.allowed
        assert ("开场时间没有核实依据" if recheck == "no_verified_opening" else "查局用途") in personal.error
        assert named.called and named.allowed
    assert len(client.calls) == 2
    assert not store.games and not store.invite_drafts and not store.room_reservations


def test_v5_0103_unverified_personal_opening_rejection_explains_how_to_query_without_it(tmp_path):
    """已查的原桌与未查的个人库存分开，只接受模型显式修正后的查询。"""
    at = datetime.fromisoformat("2026-11-25T19:02:00+08:00")
    prior_at = at - timedelta(minutes=2)
    named_at = "2026-11-25T19:30:00+08:00"
    prior = _incoming("老板，七点半371那桌没搭上，现在有没有晚一点的无烟桌，底注0.5或1都行，我打三四个小时，能搭一桌吗",
                      sent=prior_at, message_id="prior")
    message = _incoming("老板晚上好，那我再问一下七点半那桌有没有搭上；没位的话你帮我组一桌也行，0.5或1底都行，无烟，打三四个小时，你看能组得起来吗？",
                        sent=at)
    verified = _decision(current=3, lookup_scope="specific_table", clock_role="opening", opening_window=None,
                         accepted_stake_range={"min": 0.5, "max": 1})
    independent = verified | {"also_requests_personal_options": True, "clock_role": "other", "supporting_customer_refs": [1, 3]}
    reviewer, trace = ReviewClient([verified, independent]), InMemoryTraceRecorder()
    store = SQLiteAgentStore(tmp_path / "v5-0103-personal.sqlite3")
    service = ToolExecutionService(store, ToolGateway(store), trace, table_availability_client=reviewer)
    terms = {"smoke_preference": "no_smoking", "stake_range": {"min": 0.5, "max": 1}}

    def query(scope, requirement, *, step):
        action = AgentAction("分别查原桌和个人选项", "needs_tool", "区分查询对象", tool_calls=[
            ToolCall("search_current_games", {"lookup_scope": scope, "include_nearby": False, "requirement": requirement},
                     call_id=f"lookup-{step}", depends_on=[]),
        ])
        return service.execute_tool_calls(action, message=message, trace_id=message.message_id,
            previous_step_tool_results=[], step_index=step, run_id="v5-0103-personal",
            run_version=store.conversation_version(message.conversation_id)).tool_results[0]

    def executed_scopes():
        return [event.content["call"]["arguments"]["lookup_scope"]
                for event in trace.get_trace(message.message_id) if event.step == "tool_called"]

    with using_clock(lambda: prior_at):
        other, _ = store.create_game(conversation_id="other-host", organizer_id="other-host", organizer_name="其他牌友",
            requirement={**terms, "game_type": "red_center_mahjong", "planned_start_at": "2026-11-25T20:15:00+08:00",
                         "start_time_kind": "scheduled"}, known_players=[], trace_id="seed-alternative")
        _activate(store, started=prior_at)
        _persist(store, prior)
        store.append_assistant_turn("dm", "今晚这会儿无烟、0.5到1底的现成桌暂时没查到能搭的诶，要不我帮你组一桌？", "shop",
                                    metadata={"delivery_status": "delivered"})

    with using_clock(lambda: at):
        _persist(store, message)
        named = query("specific_table", {"game_type": "hangzhou_mahjong", "planned_start_at": named_at, "start_time": "19:30",
                                        "start_time_kind": "scheduled"}, step=1)
        assert named.called and named.allowed and named.result["matches"] == []
        personal_window = {"start_at": "2026-11-25T19:00:00+08:00", "end_at": "2026-11-26T06:00:00+08:00"}
        rejected = query("personal", {**terms, "start_time_range": personal_window}, step=2)
        assert not rejected.called and not rejected.allowed and "matches" not in rejected.result
        assert rejected.result["validation_error"]["code"] == "verified_table_availability_conflict"
        proof = rejected.result["table_availability_evidence"]
        assert proof["opening_window"] is None and "arrival_lookup_floor_at" not in proof
        assert all(field in rejected.error for field in ("lookup_scope=personal", "start_time_range", "planned_start_at", "start_time"))
        assert "保留其他已核实条件" in rejected.error
        assert "原桌回执只能回答原桌" in rejected.result["instruction"]
        assert "另查没有实际回执前不要说没有合适的桌" in rejected.result["instruction"]
        feedback = compact_tool_payload(rejected.result)
        assert feedback["validation_error"]["message"] == rejected.error
        assert feedback["instruction"] == rejected.result["instruction"]
        assert executed_scopes() == ["specific_table"]

        corrected = query("personal", terms, step=3)
        assert corrected.called and corrected.allowed and corrected.error is None
        requirement = corrected.result["requirement"]
        assert {field: requirement[field] for field in terms} == terms
        assert not {"start_time_range", "planned_start_at", "start_time"}.intersection(requirement)
        assert [match["game"]["game_id"] for match in corrected.result["matches"]] == [other.game_id]

    assert len(reviewer.calls) == 2 and executed_scopes() == ["specific_table", "personal"]
    assert set(store.games) == {other.game_id} and not store.invite_drafts and not store.room_reservations
    store._connection.close()


@pytest.mark.parametrize("recheck", ["both", "personal_only", "invalid", "timeout"])
def test_rolling_0022_missing_named_table_scope_requires_confirmed_paired_read(tmp_path, recheck):
    """原桌散场问句与本人另查分别核实；复核不替单独越界调用或写入授权。"""
    at = datetime.fromisoformat("2026-11-26T19:35:00+08:00")
    original_opening = "2026-11-26T19:30:00+08:00"
    arrival = "2026-11-26T20:05:00+08:00"
    window = {"start_at": arrival, "end_at": "2026-11-26T23:59:59.999999+08:00"}
    message = _incoming("老板 看板上19:30那桌川麻快散了吧 有没有0.5或1元档的杭麻桌 我到店得20:05左右", sent=at)
    primary = _multi_decision("other", "arrival_estimate") | {"accepted_stake_range": {"min": 0.5, "max": 1}}
    # 故意改变别桌档位，证明原桌的复核不会扩散到本人的查询或既有写入边界。
    both = _multi_decision("opening", "arrival_estimate", lookup_scope="specific_table",
        also_requests_personal_options=True) | {"accepted_stake_range": {"min": 2, "max": 3}}
    second = {"both": both, "personal_only": primary, "invalid": both | {"supporting_customer_refs": [42]},
              "timeout": TimeoutError("scope recheck unavailable")}[recheck]
    reviewer, trace = ReviewClient([primary, second]), InMemoryTraceRecorder()
    store = SQLiteAgentStore(tmp_path / "mixed-0022.sqlite3")
    service = ToolExecutionService(store, ToolGateway(store), trace, table_availability_client=reviewer)

    def submit(*calls, step):
        action = AgentAction("分别核对川麻原桌和本人另一桌", "needs_tool", "分开查询用途", tool_calls=list(calls))
        return service.execute_tool_calls(action, message=message, trace_id=message.message_id,
            previous_step_tool_results=[], step_index=step, run_id="rolling-0022-named-recheck",
            run_version=store.conversation_version(message.conversation_id)).tool_results

    with using_clock(lambda: at - timedelta(minutes=15)):
        original, _ = store.create_game(conversation_id="original-owner", organizer_id="original-owner",
            organizer_name="牌友", requirement={"game_type": "sichuan_mahjong", "stake": 1,
            "planned_start_at": original_opening, "start_time_kind": "scheduled"}, known_players=[], trace_id="seed-original")
        other, _ = store.create_game(conversation_id="hangzhou-owner", organizer_id="hangzhou-owner",
            organizer_name="另一牌友", requirement={"game_type": "hangzhou_mahjong", "stake": 1,
            "planned_start_at": "2026-11-26T20:30:00+08:00", "start_time_kind": "scheduled"},
            known_players=[], trace_id="seed-other")

    with using_clock(lambda: at):
        _activate(store, started=at)
        _persist(store, message)
        specific = ToolCall("search_current_games", {"lookup_scope": "specific_table",
            "requirement": {"game_type": "sichuan_mahjong", "start_time": "19:30", "start_time_kind": "scheduled"}},
            call_id="lookup_original_table", depends_on=[])
        personal = ToolCall("search_current_games", {"lookup_scope": "personal", "include_nearby": False,
            "requirement": {"game_type": "hangzhou_mahjong", "stake_range": {"min": 0.5, "max": 1}, "start_time_range": window}},
            call_id="lookup_hangzhou_personal", depends_on=[])
        isolated, = submit(specific, step=1)
        assert not isolated.called and "查局用途与核实的原话不一致" in isolated.error and len(reviewer.calls) == 1

        overnight = replace(personal, arguments=personal.arguments | {"requirement": personal.arguments["requirement"] | {
            "start_time_range": {"start_at": arrival, "end_at": "2026-11-27T06:00:00+08:00"},
        }})
        original_pair, wrong_scope = submit(specific, overnight, step=2)
        assert not wrong_scope.called and window["end_at"] in wrong_scope.error
        corrected, = submit(personal, step=3)
        named, own = submit(specific, personal, step=4)
        for actual in (corrected, own):
            assert actual.called and actual.allowed
            assert [row["game"]["game_id"] for row in actual.result["matches"]] == [other.game_id]
            assert actual.result["requirement"]["start_time_range"] == window
        for actual in (original_pair, named):
            if recheck == "both":
                assert actual.called and actual.allowed
                match, = actual.result["matches"]
                assert match["game"]["game_id"] == original.game_id and match["game"]["status"] == "forming"
                assert "stake_range" not in actual.result["requirement"] and "start_time_range" not in actual.result["requirement"]
            else:
                assert not actual.called and "查局用途与核实的原话不一致" in actual.error

        if recheck == "both":
            write = ToolCall("create_game", {"organizer_id": "guest", "organizer_name": "顾客", "known_players": [],
                "requirement": {"game_type": "hangzhou_mahjong", "stake_range": {"min": 0.5, "max": 1}, "planned_start_at": arrival,
                                "start_time_kind": "scheduled"}}, call_id="write", depends_on=["lookup_original_table"])
            _, _, denied = submit(specific, personal, write, step=5)
            assert not denied.called and "还没选定" in denied.error
            assert denied.result["table_availability_evidence"]["lookup_scope"] == "personal"
            assert "also_requests_personal_options" not in denied.result["table_availability_evidence"]
    first_call, recheck_call = reviewer.calls
    assert first_call[0][1] == recheck_call[0][1]
    assert "快散了吧" in recheck_call[0][0]["content"]
    assert 0 < recheck_call[1]["timeout_seconds"] <= first_call[1]["timeout_seconds"]
    events = trace.get_trace(message.message_id)
    assert any(event.content.get("review_purpose") == "independent_specific_table_read"
               and event.step == "table_availability_authorization_reused" for event in events)
    assert set(store.games) == {original.game_id, other.game_id} and not store.invite_drafts and not store.room_reservations
    store._connection.close()


def test_personal_read_recheck_is_not_available_to_a_write_in_the_same_plan():
    """原桌核验继续约束同计划的写工具，另一条查询的复核不能改变它。"""
    at = SENT.replace(hour=18, minute=35)
    store = InMemoryAgentStore()
    message = _incoming("八点开那桌是川麻吗？另外九点开的一块红中有吗？", sent=at)
    original = _decision(lookup_scope="specific_table", clock_role="other", opening_window=None,
                         accepted_stake_range=None,
                         clock_roles=[{"clock_ref": 1, "role": "opening"}, {"clock_ref": 2, "role": "opening"}])
    opening = "2026-11-26T21:00:00+08:00"
    range_at_opening = {"start_at": opening, "end_at": opening}
    confirmed = original | {"also_requests_personal_options": True, "opening_window": range_at_opening}
    client, trace = ReviewClient([original, confirmed]), InMemoryTraceRecorder()
    creation = ReviewClient([])
    service = ToolExecutionService(store=store, tool_gateway=ToolGateway(store), trace_recorder=trace,
                                   table_availability_client=client, creation_authorization_client=creation)
    _activate(store, started=at)
    _persist(store, message)
    with using_clock(lambda: at):
        action = AgentAction(goal="查局", objective_status="needs_tool", reasoning_summary="核对范围", tool_calls=[
            ToolCall("search_current_games", {"lookup_scope": "personal", "requirement": {"start_time_range": range_at_opening}},
                     call_id="personal", depends_on=[]),
            ToolCall("create_game", {"organizer_id": "guest", "organizer_name": "顾客", "known_players": [],
                     "requirement": {"stake": "1", "start_time_kind": "scheduled", "planned_start_at": opening}},
                     call_id="write", depends_on=["personal"]),
        ])
        personal, write = service.execute_tool_calls(action, message=message, trace_id=message.message_id,
            previous_step_tool_results=[], step_index=1, run_id="personal-recheck-write-isolation",
            run_version=store.conversation_version(message.conversation_id)).tool_results
    assert personal.called and personal.allowed
    assert not write.called and not write.allowed and "未核实为本人组织新局" in write.error
    assert not write.result["table_availability_evidence"].get("also_requests_personal_options")
    assert len(client.calls) == 2 and not creation.calls
    assert not store.games and not store.invite_drafts and not store.room_reservations


def test_personal_read_recheck_is_invalidated_by_new_input_before_execution():
    """核验期间到达更新来话，旧版本的另查不进入网关。"""
    store = InMemoryAgentStore()
    _activate(store)
    message = _incoming("午后那桌是川麻吗？两点以后有红中吗？")
    _persist(store, message)
    original = _decision(lookup_scope="specific_table", clock_role="opening", opening_window=None, accepted_stake_range=None)

    class NewInputDuringRecheck(ReviewClient):
        """第二次核验等待期间提交更新的通道来话。"""

        def complete(self, messages, **options):
            if self.calls:
                _persist(store, _incoming("我先不看了。", sent=SENT + timedelta(seconds=1), message_id="new"))
            return super().complete(messages, **options)

    client = NewInputDuringRecheck([original, original | {"also_requests_personal_options": True}])
    trace = InMemoryTraceRecorder()
    service = ToolExecutionService(store=store, tool_gateway=ToolGateway(store), trace_recorder=trace,
                                   table_availability_client=client)
    action = AgentAction(goal="查其他局", objective_status="needs_tool", reasoning_summary="核对范围", tool_calls=[
        ToolCall("search_current_games", {"lookup_scope": "personal", "requirement": {}}),
    ])
    execution = service.execute_tool_calls(action, message=message, trace_id=message.message_id,
        previous_step_tool_results=[], step_index=1, run_id="personal-recheck-stale",
        run_version=store.conversation_version(message.conversation_id))
    assert len(client.calls) == 2 and execution.stop_loop
    assert all(not result.called for result in execution.tool_results)
    events = trace.get_trace(message.message_id)
    assert all(event.step != "tool_called" for event in events)
    assert any(event.step == "table_availability_authorization_superseded"
               and event.content.get("review_purpose") == "independent_personal_read" for event in events)


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


def test_rolling_0102_single_numbered_arrival_estimate_keeps_later_tables_visible(tmp_path):
    """单钟点按编号核实到店，不把估计锁成开局；真实库仍可提供晚桌与待确认近邻。"""
    at = datetime.fromisoformat("2026-11-26T19:05:00+08:00")
    arrival = "2026-11-26T19:30:00+08:00"
    window = {"start_at": arrival, "end_at": "2026-11-26T23:59:59.999999+08:00"}
    message = _incoming("老板，刚才说的组一桌红中一块底，到七点半还行吗，我这边能到", sent=at)
    store = SQLiteAgentStore(tmp_path / "rolling-0102-read.sqlite3")
    trace = InMemoryTraceRecorder()
    decision = _decision(clock_role="other", opening_window=None, opening_dayparts=None, accepted_stake_range=None,
                         agreed_duration_hours=None, max_duration_hours=None, opening_customer_ref=None,
                         clock_roles=[{"clock_ref": 1, "role": "arrival_estimate"}])
    review = ReviewClient([decision])
    service = ToolExecutionService(store, ToolGateway(store), trace, table_availability_client=review)
    terms = {"game_type": "red_center_mahjong", "stake": 1}

    def submit(tool, step):
        action = AgentAction("按本人可到的时间查红中", "needs_tool", "核对开场角色", tool_calls=[tool])
        result, = service.execute_tool_calls(action, message=message, trace_id=message.message_id,
            previous_step_tool_results=[], step_index=step, run_id="rolling-0102-read",
            run_version=store.conversation_version(message.conversation_id)).tool_results
        return result

    with using_clock(lambda: at):
        _activate(store, started=at)
        _persist(store, message)
        games = []
        for host, start in (("nearby", "2026-11-26T19:25:00+08:00"), ("later", "2026-11-26T20:00:00+08:00")):
            game, _ = store.create_game(conversation_id=host, organizer_id=host, organizer_name="牌友",
                requirement=terms | {"start_time_kind": "scheduled", "planned_start_at": start},
                known_players=[], trace_id="seed")
            games.append(game)
        early, later = games
        point = terms | {"planned_start_at": arrival, "start_time": "19:30", "start_time_kind": "scheduled"}
        rejected = submit(ToolCall("search_current_games", {"lookup_scope": "personal", "requirement": point}), 1)
        assert not rejected.called and "不要另填固定开场" in rejected.error and arrival in rejected.error
        evidence = rejected.result["table_availability_evidence"]
        assert evidence["clock_role"] == "arrival_estimate" and evidence["arrival_lookup_floor_at"] == arrival
        assert evidence["opening_window"] is None and evidence["has_fixed_opening_authority"] is False

        corrected = submit(ToolCall("search_current_games", {"lookup_scope": "personal",
            "requirement": terms | {"start_time_range": window}}), 2)
        assert corrected.called and corrected.allowed and not corrected.error
        assert [row["game"]["game_id"] for row in corrected.result["matches"]] == [later.game_id]
        nearby, = corrected.result["nearby_options"]
        assert nearby["game"]["game_id"] == early.game_id and nearby["requires_timing_confirmation"] is True

        rejected_create = submit(ToolCall("create_game", {"organizer_id": "guest", "organizer_name": "顾客",
            "known_players": [], "requirement": point}), 3)
        assert not rejected_create.called and "还没选定" in rejected_create.error

    assert len(review.calls) == 1 and set(store.games) == {early.game_id, later.game_id}
    assert not store.room_reservations and not store.invite_drafts
    assert [event.content["call"]["name"] for event in trace.get_trace("current") if event.step == "tool_called"] == [
        "search_current_games",
    ]
    store._connection.close()


@pytest.mark.parametrize("clock_ref", [True, 2])
def test_single_arrival_role_does_not_accept_a_nonmatching_observation_reference(clock_ref):
    """中性的单钟点标记不能被未绑定的编号提升为本人具体到店过滤。"""
    stamp = SENT.replace(hour=19, minute=5)
    message = _incoming("我七点半左右到，有桌吗？", sent=stamp)
    store = InMemoryAgentStore()
    _activate(store, started=stamp)
    _persist(store, message)
    checked = _check(store, ReviewClient([_decision(clock_role="other", opening_window=None,
        clock_roles=[{"clock_ref": clock_ref, "role": "arrival_estimate"}])]), message)
    assert checked["clock_role"] == "other" and "arrival_lookup_floor_at" not in checked


def test_rolling_0038_numbered_opening_keeps_evening_search_after_eight(tmp_path):
    """同一来话的一点底是档位，编号核实的八点以后约束跨午夜的真实查桌。"""
    at = datetime.fromisoformat("2026-11-25T19:00:00+08:00")
    message = _incoming("老板 有杭州麻将的桌吗 一点底 八点以后有的话我可以打", sent=at)
    store = SQLiteAgentStore(tmp_path / "numbered-opening.sqlite3")
    stakes = {"min": 1, "max": 1}
    decision = _decision(clock_role="other", opening_window=None, accepted_stake_range=stakes,
        clock_roles=[{"clock_ref": 1, "role": "opening"}],
        opening_dayparts={"date": "2026-11-25", "parts": ["evening"], "source_ref": 1})
    review = ReviewClient([decision])
    creation = ReviewClient([])
    service = ToolExecutionService(store, ToolGateway(store), InMemoryTraceRecorder(),
                                   table_availability_client=review, creation_authorization_client=creation)
    terms = {"game_type": "hangzhou_mahjong", "stake": 1}
    window = {"start_at": "2026-11-25T20:00:00+08:00", "end_at": "2026-11-26T06:00:00+08:00"}
    with using_clock(lambda: at):
        _activate(store, started=at)
        _persist(store, message)
        for start in ("19:45", "20:15"):
            store.create_game(conversation_id=start, organizer_id=start, organizer_name="牌友",
                requirement=terms | {"start_time_kind": "scheduled", "planned_start_at": f"2026-11-25T{start}:00+08:00"},
                known_players=[], trace_id=start)
        too_early = {**window, "start_at": at.isoformat()}
        rejected = _execute_scope_search(service, message, terms | {"start_time_range": too_early}, step=1)
        assert not rejected.called and window["start_at"] in rejected.error
        evidence = rejected.result["table_availability_evidence"]
        assert evidence["clock_role"] == "opening" and evidence["opening_window"] == window
        assert evidence["has_fixed_opening_authority"] is False
        found = _execute_scope_search(service, message, terms | {"start_time_range": window}, step=2)
        assert found.called and found.allowed and found.result["requirement"]["start_time_range"] == window
        assert [row["game"]["planned_start_at"] for row in found.result["matches"]] == ["2026-11-25T20:15:00+08:00"]
        assert not found.result.get("nearby_options")
        fixed = ToolCall("create_game", {"organizer_id": "guest", "organizer_name": "顾客", "known_players": [],
            "requirement": terms | {"planned_start_at": window["start_at"], "start_time_kind": "scheduled"}})
        action = AgentAction("核实开场", "needs_tool", "检查无固定权限", tool_calls=[fixed])
        created, = service.execute_tool_calls(action, message=message, trace_id=message.message_id,
            previous_step_tool_results=[], step_index=3, run_id="numbered-opening",
            run_version=store.conversation_version(message.conversation_id)).tool_results
        assert not created.called and "还没选定" in created.error
    assert len(review.calls) == 1 and len(store.games) == 2 and not creation.calls and not store.room_reservations
    store._connection.close()


@pytest.mark.parametrize(("text", "role", "ref", "expected_role", "expected_end"), [
    ("晚上十点之前有杭麻吗", "other", 1, "opening", "2026-11-26T22:00:00+08:00"),
    ("晚上十点之前有杭麻吗", "other", 2, "other", "2026-11-27T06:00:00+08:00"),
    ("晚上十点之前有杭麻吗", "other", True, "other", "2026-11-27T06:00:00+08:00"),
    ("晚上十点之前有杭麻吗", "wait_until", 1, "wait_until", "2026-11-27T06:00:00+08:00"),
    ("晚上十点前后有杭麻吗", "other", 1, "other", "2026-11-27T06:00:00+08:00"),
    ("晚上十点有杭麻吗", "other", 1, "other", "2026-11-27T06:00:00+08:00"),
])
def test_numbered_opening_requires_a_valid_reference_and_single_sided_qualifier(
    text, role, ref, expected_role, expected_end,
):
    """编号不替换已有非中性角色，也不能把近似或裸钟点当作单侧范围边界。"""
    at = SENT.replace(hour=19)
    message = _incoming(text, sent=at)
    store = InMemoryAgentStore()
    _activate(store, started=at)
    _persist(store, message)
    review = ReviewClient([_decision(clock_role=role, opening_window=None,
        clock_roles=[{"clock_ref": ref, "role": "opening"}],
        opening_dayparts={"date": "2026-11-26", "parts": ["evening"], "source_ref": 1})])
    checked = _check(store, review, message)
    assert checked["clock_role"] == expected_role and checked["opening_window"]["end_at"] == expected_end
    assert checked["has_fixed_opening_authority"] is False


def test_numbered_opening_does_not_invent_a_window_without_independent_scope():
    """只有单侧词和编号时不替顾客补出宽度或选定开场。"""
    at = SENT.replace(hour=19)
    message = _incoming("晚点有杭麻吗，八点以后", sent=at)
    store = InMemoryAgentStore()
    _activate(store, started=at)
    _persist(store, message)
    decision = _decision(clock_role="other", opening_window=None, opening_dayparts=None,
        clock_roles=[{"clock_ref": 1, "role": "opening"}])
    checked = _check(store, ReviewClient([decision]), message)
    assert checked["clock_role"] == "other" and checked["opening_window"] is None
    assert checked["has_fixed_opening_authority"] is False


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


@pytest.mark.parametrize("hard_end", [True, False])
def test_v4_0038_asap_recruitment_expiry_does_not_bind_the_end_of_play(tmp_path, hard_end):
    """保留本人截止的只读查询。人齐开没有对应散场机制时不能写已托付的新局。"""
    at = datetime.fromisoformat("2026-11-26T09:37:00+08:00")
    cutoff = "2026-11-26T16:00:00+08:00"
    store = SQLiteAgentStore(tmp_path / "rolling-0038.sqlite3")
    prior = _incoming("老板 上午有杭州麻将一点底的桌吗 现在就能到 还有位吗", sent=at - timedelta(minutes=2),
                      message_id="first-lookup")
    text = "老板 那就帮我组一桌杭州麻将一点底 我现在能到"
    message = _incoming(text + (" 下午四点前都有空" if hard_end else " 人齐开"), sent=at)
    reviewer = ReviewClient([_decision(3, clock_role="departure_deadline", opening_window=None,
        accepted_stake_range={"min": 1, "max": 1}, supporting_customer_refs=[1, 3])])
    creation = ReviewClient([{"scope": "create", "customer_ref": 3, "shop_offer_ref": None}])
    trace = InMemoryTraceRecorder()
    service = ToolExecutionService(store, ToolGateway(store), trace, table_availability_client=reviewer,
                                   creation_authorization_client=creation)
    terms = {"game_type": "hangzhou_mahjong", "stake": 1}

    def run_tool(name, arguments, step):
        action = AgentAction(goal="按本人委托处理杭麻组局", objective_status="needs_tool", reasoning_summary="核实时间",
                             tool_calls=[ToolCall(name, arguments)])
        return service.execute_tool_calls(action, message=message, trace_id=message.message_id,
            previous_step_tool_results=[], step_index=step, run_id="rolling-0038",
            run_version=store.conversation_version("dm")).tool_results[0]

    def create(step, requirement):
        return run_tool("create_game", {"organizer_id": "guest", "organizer_name": "顾客", "known_players": [],
                                        "requirement": terms | requirement}, step)

    with using_clock(lambda: prior.sent_at):
        _activate(store, started=prior.sent_at)
        _persist(store, prior)
        store.append_assistant_turn("dm", "上午现在没有能搭的杭麻一块底桌诶，要不我帮你组一桌？", "offer",
                                   metadata={"delivery_status": "delivered"})
    with using_clock(lambda: at):
        _persist(store, message)
        original = create(1, {"start_time_kind": "asap_when_full"})
        if not hard_end:
            assert original.called and original.allowed
            game, = store.games.values()
            assert game.expires_at.isoformat() == "2026-11-26T13:37:00+08:00" and game.planned_end_at is None
            assert not reviewer.calls and len(creation.calls) == 1
        else:
            assert not original.called and not original.allowed
            assert "人齐开无法落实" in original.error and cutoff in original.error
            assert original.result["table_availability_evidence"]["must_end_by"] == cutoff
            capped = create(2, {"start_time_kind": "asap_when_full", "max_duration_hours": 2})
            assert not capped.called and not capped.allowed and "人齐开无法落实" in capped.error
            guessed = create(3, {"start_time_kind": "scheduled", "planned_start_at": "2026-11-26T12:00:00+08:00"})
            assert not guessed.called and not guessed.allowed and "还没选定" in guessed.error
            assert not store.games and not store.invite_drafts and not creation.calls
            assert not any(event.step == "tool_called" for event in trace.get_trace(message.message_id))

            games = {}
            for minute in (0, 15):
                host = f"existing-{minute}"
                game, _ = store.create_game(conversation_id=host, organizer_id=host, organizer_name=host,
                    known_players=[{"customer_id": host}], trace_id=host,
                    requirement=terms | {"planned_start_at": f"2026-11-26T12:{minute:02d}:00+08:00", "duration_hours": 4})
                games[minute] = game.game_id
            query = run_tool("search_current_games", {"lookup_scope": "personal", "requirement": terms | {"must_end_by": cutoff}}, 4)
            assert query.called and query.allowed and query.result["requirement"]["must_end_by"] == cutoff
            assert [item["game"]["game_id"] for item in query.result["matches"]] == [games[0]]
            assert set(store.games) == set(games.values()) and len(reviewer.calls) == 1
        assert not store.room_reservations
    store._connection.close()


def test_v5_0146_repeated_arrival_time_prompts_only_the_opening_decision(tmp_path):
    """同任务的到店钟点不能偷换成开场。解开该歧义后才可实际建局。"""
    at = datetime.fromisoformat("2026-11-25T19:02:00+08:00")
    start = "2026-11-25T19:30:00+08:00"
    soft_end = "2026-11-25T23:00:00+08:00"
    prior = _incoming("七点半我能到 红中2块3块都行 这桌有位置吗", sent=at - timedelta(minutes=2), message_id="prior")
    message = _incoming("那帮我组一桌吧 红中2块3块都行 七点半可以 我打到十一点左右", sent=at, message_id="ambiguous")
    agreed = _incoming("就定晚上七点半开吧，我想打到十一点左右。", sent=at + timedelta(minutes=2), message_id="confirmed")
    stakes = {"min": 2, "max": 3}
    initial = _multi_decision("arrival_estimate", "play_until") | {
        "current_ref": 3, "supporting_customer_refs": [1, 3], "accepted_stake_range": stakes,
    }
    confirmed = _multi_decision("opening", "play_until") | {
        "current_ref": 5, "supporting_customer_refs": [3, 5], "accepted_stake_range": stakes,
        "has_fixed_opening_authority": True, "opening_customer_ref": 5,
        "opening_window": {"start_at": start, "end_at": start},
    }
    review = ReviewClient([initial, confirmed])
    creation = ReviewClient([{"scope": "create", "customer_ref": 3, "shop_offer_ref": None}])
    store = SQLiteAgentStore(tmp_path / "rolling-v5-0146.sqlite3")
    trace = InMemoryTraceRecorder()
    service = ToolExecutionService(store, ToolGateway(store), trace, table_availability_client=review,
                                   creation_authorization_client=creation)
    terms = {"game_type": "red_center_mahjong", "stake_range": stakes}

    def run_tool(source, call, step):
        with using_clock(lambda: source.sent_at):
            action = AgentAction(goal="处理已收到的组局委托", objective_status="needs_tool", reasoning_summary="核实开场",
                                 tool_calls=[call])
            return service.execute_tool_calls(action, message=source, trace_id=source.message_id,
                previous_step_tool_results=[], step_index=step, run_id=source.message_id,
                run_version=store.conversation_version("dm")).tool_results[0]

    with using_clock(lambda: prior.sent_at):
        _activate(store, started=prior.sent_at)
        _persist(store, prior)
        store.append_assistant_turn("dm", "七点半这个点，红中2块3块的现成桌现在没有能搭的诶，要不我帮你组一桌？", "offer",
                                   metadata={"delivery_status": "delivered"})
    with using_clock(lambda: at):
        _persist(store, message)
        other, _ = store.create_game(conversation_id="unrelated", organizer_id="host", organizer_name="桌主",
            known_players=[{"customer_id": "host"}], trace_id="unrelated", requirement={
                "game_type": "hangzhou_mahjong", "stake": 2, "planned_start_at": start,
            })

    fixed = {"start_time_kind": "scheduled", "planned_start_at": start}
    read = run_tool(message, ToolCall("search_current_games", {"lookup_scope": "personal", "requirement": terms | {
        "start_time_range": {"start_at": start, "end_at": "2026-11-25T23:59:59+08:00"},
    }}), 1)
    assert read.called and read.allowed and not read.result["matches"]
    assert read.result["customer_reply_contract"]["evidence_scope"]["unfiltered_available_pool_empty"] is False

    write = ToolCall("create_game", {"organizer_id": "guest", "organizer_name": "顾客", "known_players": [],
                                      "requirement": terms | fixed})
    refused = run_tool(message, write, 2)
    assert not refused.called and not refused.allowed and not creation.calls
    assert all(clause in refused.error for clause in (
        "本次未建局", "顾客已经明确委托组局时", f"{start} 只核实为本人可到店", "只问这个时间是否也是本桌开场",
        "仅询局时按已核实的查局结果回复",
        "waiting_user", "不要重新征询组局委托", f"{soft_end} 的柔性偏好", "不当成硬截止或固定时长",
    ))
    evidence = refused.result["table_availability_evidence"]
    assert not evidence["has_fixed_opening_authority"] and evidence["play_until_at"] == soft_end
    with using_clock(lambda: at):
        other_clock = ToolCall("create_game", {"requirement": terms | fixed | {"planned_start_at": "2026-11-25T20:00:00+08:00"}})
        feedback = validate_table_availability(other_clock, evidence)
        assert "还没选定" in feedback and "只核实为本人可到店" not in feedback
        assert "单独请顾客确认具体开场" in feedback and "实际查到的现成桌或房态" in feedback
        assert "不能报成已登记席位、已有缺口或保证开打" in feedback
    assert set(store.games) == {other.game_id}

    with using_clock(lambda: agreed.sent_at):
        store.append_assistant_turn("dm", "现在没有合适现成桌。七点半你能到，那这桌定七点半开可以吗？", "ask-opening",
                                   metadata={"delivery_status": "delivered"})
        _persist(store, agreed)
    created = run_tool(agreed, write, 1)
    assert created.called and created.allowed and created.result["game"]["planned_start_at"] == start
    assert all(key not in created.result["game"]["requirement"] for key in ("must_end_by", "duration_hours", "max_duration_hours"))
    assert len(review.calls) == 2 and len(creation.calls) == 1
    store._connection.close()


def test_rolling_0146_verified_clocks_replace_only_unsupported_earlier_read_duration(tmp_path):
    """回放两次独立核验和真实查写，修订旧查局时长并保留玩法、离店和建局授权守卫。"""
    at = datetime.fromisoformat("2026-11-25T19:05:00+08:00")
    open_at, depart_at = "2026-11-25T20:00:00+08:00", "2026-11-25T23:00:00+08:00"
    store = SQLiteAgentStore(tmp_path / "rolling-0146.sqlite3")
    _activate(store, started=at - timedelta(minutes=5))
    history = [
        ("晚上有合适的红中桌吗 我最多能打到十一点", "今晚到十一点前现在没有能搭的红中桌诶，要不我帮你组一桌？"),
        ("行 红中两块底 帮我组一桌 我最多打到十一点",
         "红中两块底这桌我给你组，不过你23点要走，人齐开没法保证那个点前能散。你看定个几点开场？比如20点或20点半，我好按这个时间凑人。"),
    ]
    for index, (customer_text, shop_text) in enumerate(history):
        prior_at = at - timedelta(minutes=4 - index * 2)
        prior = _incoming(customer_text, sent=prior_at, message_id=f"prior-{index}")
        with using_clock(lambda: prior_at):
            _persist(store, prior)
            store.append_assistant_turn("dm", shop_text, "shop", metadata={"delivery_status": "delivered"})
    message = _incoming("八点开行 两小时五十分钟到十一点 能凑到人吗", sent=at)
    decision = _multi_decision("opening", "departure_deadline") | {
        "accepted_stake_range": {"min": 2, "max": 2}, "opening_window": {"start_at": open_at, "end_at": open_at},
        "has_fixed_opening_authority": True, "opening_customer_ref": 5, "current_ref": 5,
        "supporting_customer_refs": [1, 3, 5],
    }
    review = ReviewClient([TimeoutError("first independent check unavailable"), decision])
    authorization = ReviewClient([{"scope": "create", "customer_ref": 3, "shop_offer_ref": None}])
    trace = InMemoryTraceRecorder()
    service = ToolExecutionService(store, ToolGateway(store), trace, table_availability_client=review,
                                   creation_authorization_client=authorization)
    terms = {"game_type": "red_center_mahjong", "stake": 2, "start_time_kind": "scheduled",
             "start_time": "20:00", "planned_start_at": open_at}
    results = []
    with using_clock(lambda: at):
        _persist(store, message)

        def submit(call):
            action = AgentAction(goal="按已核实的开场组一桌", objective_status="needs_tool", reasoning_summary="核对条款",
                                 tool_calls=[call])
            result, = service.execute_tool_calls(action, message=message, trace_id=message.message_id,
                previous_step_tool_results=results, step_index=len(results) + 1, run_id="rolling-0146",
                run_version=store.conversation_version(message.conversation_id)).tool_results
            results.append(result)
            return result

        def create(patch, call_id):
            return submit(ToolCall("create_game", {"organizer_id": "guest", "organizer_name": "顾客", "known_players": [],
                                                   "requirement": terms | patch}, call_id=call_id))

        stale_read = submit(ToolCall("search_current_games", {"lookup_scope": "personal",
            "requirement": terms | {"duration_hours": 2.8333}}, call_id="c1"))
        assert stale_read.called and stale_read.allowed and stale_read.result["matches"] == []

        wrong_exact_duration = create({"duration_hours": 2.8333}, "bad-duration")
        assert wrong_exact_duration.result["validation_error"]["code"] == "verified_table_availability_conflict"
        assert "max_duration_hours=3.0" in wrong_exact_duration.error
        rejected_archived_proposal = create({"max_duration_hours": 3, "must_end_by": depart_at}, "bad-read-field")
        assert not rejected_archived_proposal.called
        changed_game = create({"max_duration_hours": 3, "game_type": "hangzhou_mahjong"}, "bad-game")
        assert changed_game.result["validation_error"]["code"] == "tool_requirement_consistency_rejected"
        assert "game_type" in changed_game.error and "duration_hours" not in changed_game.error
        assert not store.games and not authorization.calls

        corrected = create({"max_duration_hours": 3}, "corrected")
        assert corrected.called and corrected.allowed and not corrected.error
        game, = store.games.values()
        assert game.planned_start_at.isoformat() == open_at and game.planned_end_at.isoformat() == depart_at
        assert game.requirement["max_duration_hours"] == 3 and "duration_hours" not in game.requirement

    assert len(review.calls) == 2 and len(authorization.calls) == 1
    assert stale_read.result["requirement"]["duration_hours"] == 2.8333
    event, = [item.content for item in trace.get_trace(message.message_id) if item.step == "tool_read_duration_superseded"]
    assert event == {
        "field": "duration_hours", "previous_duration_hours": 2.8333,
        "source_tool": "search_current_games", "source_call_id": "c1", "source_evidence_index": 0,
        "opening_customer_ref": 5, "supporting_customer_refs": [1, 3, 5], "planned_start_at": open_at,
        "must_end_by": depart_at, "max_duration_hours": 3, "source_message_id": message.message_id,
        "run_version": store.conversation_version(message.conversation_id), "step_index": 5, "create_call_id": "corrected",
    }
    store._connection.close()


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


@pytest.mark.parametrize("complete", [True, False])
def test_fixed_opening_rechecks_an_unsupported_customer_citation_before_allowing_a_read(monkeypatch, complete):
    """真响应的坏引用用原证据及剩余预算复判一次，来源不全则仍然无授权。"""
    from mahjong_agent_runtime.domains import table_availability_authorization as module

    store = InMemoryAgentStore()
    _activate(store, started=MULTI_SENT - timedelta(minutes=5))
    store.append_user_turn(_incoming("我七点半有空", sent=MULTI_SENT - timedelta(minutes=4), message_id="prior"), "prior")
    message = _incoming("八点开行 两小时五十分钟到十一点 能凑到人吗", sent=MULTI_SENT)
    _persist(store, message)
    point = "2026-11-26T20:00:00+08:00"
    corrected = _multi_decision("opening", "departure_deadline", current_ref=2, supporting_customer_refs=[1, 2],
        opening_customer_ref=2, has_fixed_opening_authority=True) | {"opening_window": {"start_at": point, "end_at": point}}
    old = corrected | {"opening_customer_ref": 1, "max_duration_hours": 2.83}
    review = ReviewClient([old, corrected])
    trace = InMemoryTraceRecorder()
    times = iter((10.0, 13.0))
    monkeypatch.setattr(module, "monotonic", lambda: next(times))
    monkeypatch.setattr(module, "model_response_source", lambda *args, **kwargs: {"http_response_complete": complete})
    result = _check(store, review, message, trace)

    events = trace.get_trace(message.message_id)
    assert events[1].step == "table_availability_authorization_invalid"
    assert "not supported by its clock references" in events[1].content["error"]
    assert len(review.calls) == (2 if complete else 1)
    if complete:
        assert result["opening_customer_ref"] == 2 and result["must_end_by"] == "2026-11-26T23:00:00+08:00"
        assert result.get("max_duration_hours") is None
        assert review.calls[1][0][:2] == review.calls[0][0]
        assert review.calls[1][1]["timeout_seconds"] == 9.0
        assert events[-1].content["attempt"] == 2
        assert _check(store, review, message, trace, step=2) == result
    else:
        assert result is None and len(events) == 2


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
    assert _check(store, client, message) == {**without_dayparts, "arrival_by_at": "2026-11-26T09:30:00+08:00"}


def test_explicit_future_opening_daypart_uses_actual_date_and_overnight_boundary():
    """有本人明确指定的开场日期才采用未来时段，延续到该日所属晚间的次日清晨。"""
    store = InMemoryAgentStore()
    prior_at = SENT - timedelta(minutes=2)
    _activate(store, started=prior_at)
    _persist(store, _incoming("下周三下午或者晚上开都行", sent=prior_at, message_id="prior"))
    message = _incoming("晚上九点半前我能到")
    _persist(store, message)
    dayparts = {"date": "2026-12-02", "parts": ["afternoon", "evening"], "source_ref": 1}
    decision = _decision(2, opening_window=None, opening_dayparts=dayparts, supporting_customer_refs=[1, 2])
    expected = {"start_at": "2026-12-02T12:00:00+08:00", "end_at": "2026-12-03T06:00:00+08:00"}
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


def _execute_scope_search(service, message, requirement, *, step, nearby=False):
    """把同一原来话及修正参数送进真实查局工具入口。"""
    action = AgentAction(goal="查现成桌", objective_status="needs_tool", reasoning_summary="按来话查现成桌",
        tool_calls=[ToolCall("search_current_games", {"lookup_scope": "personal", "include_nearby": nearby,
                                                     "requirement": requirement})])
    return service.execute_tool_calls(action, message=message, trace_id=message.message_id, step_index=step,
        previous_step_tool_results=[], run_id="duration-and-arrival-run",
        run_version=service.store.conversation_version(message.conversation_id)).tool_results[0]


@pytest.mark.parametrize(("duration_phrase", "hard_cap"), [
    ("打四个小时左右", False),
    ("大概能打四个小时", False),
    ("最多打四个小时左右", False),
    ("最多打四个小时", True),
    ("想打四个小时左右，实际不超过4h", True),
])
def test_rolling_personal_duration_preference_cannot_become_an_invented_booking_cap(duration_phrase, hard_cap, tmp_path):
    """保留原询问的烟况、价位与到店下界，只有明确上限才能排除稍长的桌。"""
    at = datetime.fromisoformat("2026-11-25T19:00:00+08:00")
    text = f"你好，请问现在还有局能搭一桌吗？我想打无烟的，2 到 3 块底，{duration_phrase}；能打的时间是 19 点 09 分以后。"
    message = _incoming(text, sent=at)
    store = SQLiteAgentStore(tmp_path / "duration-authority.sqlite3")
    stakes = {"min": 2, "max": 3}
    review = ReviewClient([_decision(clock_role="arrival_not_before", opening_window=None, accepted_stake_range=stakes,
                                    max_duration_hours=4, agreed_duration_hours=None)])
    service = ToolExecutionService(store, ToolGateway(store), InMemoryTraceRecorder(), table_availability_client=review)
    games = {}
    with using_clock(lambda: at):
        _activate(store, started=at)
        _persist(store, message)
        for hours in (4, 4.25):
            host = f"host-{hours}"
            game, _ = store.create_game(conversation_id=host, organizer_id=host, organizer_name="桌主", trace_id=host,
                known_players=[{"customer_id": host}], requirement={"stake": 2, "smoke_preference": "no_smoking",
                    "planned_start_at": "2026-11-25T19:30:00+08:00", "duration_hours": hours})
            games[hours] = game.game_id
        requirement = {"stake_range": stakes, "smoke_preference": "no_smoking", "start_time_range": {
            "start_at": "2026-11-25T19:09:00+08:00", "end_at": "2026-11-25T23:59:59+08:00",
        }}
        uncapped = _execute_scope_search(service, message, requirement, step=1)
        capped = _execute_scope_search(service, message, {**requirement, "max_duration_hours": 4}, step=2)
    if hard_cap:
        assert not uncapped.called and "明确的最多可打时长" in uncapped.error
        actual = capped
        assert {row["game"]["game_id"] for row in actual.result["matches"]} == {games[4]}
    else:
        assert not capped.called and "只是偏好" in capped.error
        assert capped.result["table_availability_evidence"]["max_duration_hours"] is None
        actual = uncapped
        assert {row["game"]["game_id"] for row in actual.result["matches"]} == set(games.values())
    assert actual.allowed and actual.called and actual.error is None
    assert len(review.calls) == 1 and len(store.games) == 2 and not store.room_reservations
    store._connection.close()


def test_rolling_latest_arrival_does_not_exclude_the_live_earlier_table(tmp_path):
    """同一本人先问一底杭麻，再说八点前到时，留在已选价位且不虚构八点后开场。"""
    at = datetime.fromisoformat("2026-11-25T19:00:00+08:00")
    message = _incoming("老板 我八点前到 一底杭州麻将有位吗 有烟没烟都行", sent=at)
    store = SQLiteAgentStore(tmp_path / "arrival-by-authority.sqlite3")
    review = ReviewClient([_decision(3, clock_role="arrival_by", opening_window=None, opening_dayparts=None,
                                    accepted_stake_range={"min": 1, "max": 1}, supporting_customer_refs=[1, 3])])
    service = ToolExecutionService(store, ToolGateway(store), InMemoryTraceRecorder(), table_availability_client=review)
    before = at - timedelta(minutes=2)
    _activate(store, started=before)
    _persist(store, _incoming("老板 杭州麻将还有位吗 一底的", sent=before, message_id="prior"))
    with using_clock(lambda: before + timedelta(minutes=1)):
        store.append_assistant_turn("dm", "一元底的杭州麻将现在没有现成能搭的诶，要不我帮你组一桌？", "shop",
                                    metadata={"delivery_status": "delivered"})
    with using_clock(lambda: at):
        _persist(store, message)
        games = {}
        for starts in ("19:45", "20:15"):
            game, _ = store.create_game(conversation_id=starts, organizer_id=starts, organizer_name="桌主", trace_id=starts,
                known_players=[{"customer_id": starts}], requirement={"game_type": "hangzhou_mahjong", "stake": 1,
                    "planned_start_at": f"2026-11-25T{starts}:00+08:00"})
            games[starts] = game.game_id
        requirement = {"game_type": "hangzhou_mahjong", "stake": 1, "smoke_preference": "any",
                       "requesting_party": {"seat_count": 1}}
        miss = _execute_scope_search(service, message, {**requirement, "start_time_range": {
            "start_at": "2026-11-25T20:00:00+08:00", "end_at": "2026-11-26T06:00:00+08:00",
        }}, step=1, nearby=True)
        assert not miss.called and not miss.allowed and "移除本次未核实的开场过滤" in miss.error
        assert miss.result["table_availability_evidence"]["arrival_by_at"] == "2026-11-25T20:00:00+08:00"
        stored = _execute_scope_search(service, message, requirement, step=2)
    assert stored.called and stored.allowed and not stored.error
    assert {row["game"]["game_id"] for row in stored.result["matches"]} == set(games.values())
    assert len(review.calls) == 1 and len(store.games) == 2 and not store.room_reservations
    store._connection.close()


@pytest.mark.parametrize(("daypart", "deadline", "prior", "expected"), [
    ("今天傍晚", "九点半", None, "21:30"),
    ("今天傍晚", "晚上九点半", None, "21:30"),
    ("今天傍晚", "早上九点半", None, "09:30"),
    ("今天傍晚", "九点半 上午", None, "09:30"),
    ("今天傍晚", "九点半 AM", None, "09:30"),
    ("今天傍晚", "09:30", None, "09:30"),
    ("今天傍晚", "09点30分", None, "09:30"),
    ("今天傍晚", "一点", None, "13:00"),
    ("今天傍晚", "九点半", "我早上九点半必须走", "09:30"),
    ("晚上好，今天", "九点半", None, "09:30"),
])
def test_rolling_evening_scope_preserves_unique_hard_departure_and_real_conflicts(daypart, deadline, prior, expected, tmp_path):
    """同句已有傍晚查局依据才消歧口语钟点，早晨、数字时点和已引用截止保持原意。"""
    at = datetime.fromisoformat("2026-11-26T09:05:00+08:00")
    text = f"{daypart}有合适的红中桌吗 我最多打到{deadline}"
    message = _incoming(text, sent=at)
    store = SQLiteAgentStore(tmp_path / "evening-departure-authority.sqlite3")
    current_ref = 2 if prior else 1
    review = ReviewClient([_decision(current_ref, clock_role="departure_deadline", opening_window=None,
        opening_dayparts={"date": "2026-11-26", "parts": ["evening"], "source_ref": current_ref},
        accepted_stake_range=None)])
    service = ToolExecutionService(store, ToolGateway(store), InMemoryTraceRecorder(), table_availability_client=review)
    before = at - timedelta(minutes=2)
    _activate(store, started=before)
    if prior:
        _persist(store, _incoming(prior, sent=before, message_id="prior"))
    with using_clock(lambda: at):
        _persist(store, message)
        games = {}
        for starts, duration in (("18:30", 3), ("20:00", 2)):
            game, _ = store.create_game(conversation_id=starts, organizer_id=starts, organizer_name="桌主", trace_id=starts,
                known_players=[{"customer_id": starts}], requirement={"game_type": "red_center_mahjong", "stake": 1,
                    "planned_start_at": f"2026-11-26T{starts}:00+08:00", "duration_hours": duration})
            games[starts] = game.game_id
        opening = {"start_at": "2026-11-26T18:00:00+08:00", "end_at": "2026-11-27T06:00:00+08:00"}
        requirement = {"game_type": "red_center_mahjong", "start_time_range": opening}
        unchecked = _execute_scope_search(service, message, requirement, step=1)
        assert not unchecked.allowed and not unchecked.called
        status = unchecked.result["table_availability_evidence"]
        floor = f"2026-11-26T{expected}:00+08:00"
        assert status["must_end_by"] == status["play_until_at"] == floor and status["opening_window"] == opening
        with_deadline = _execute_scope_search(service, message, {**requirement, "must_end_by": floor}, step=2)
    if expected == "21:30":
        assert "must_end_by=2026-11-26T21:30:00+08:00" in unchecked.error
        assert with_deadline.called and with_deadline.allowed
        assert [row["game"]["game_id"] for row in with_deadline.result["matches"]] == [games["18:30"]]
    else:
        assert "澄清" in unchecked.error and "澄清" in with_deadline.error and not with_deadline.called
    assert len(review.calls) == 1 and len(store.games) == 2 and not store.room_reservations
    store._connection.close()


@pytest.mark.parametrize(("ending", "model_collapses_window", "selected_start", "authorized", "explicit_point"), [
    ("七点半到九点能开就好", False, "19:30", False, False),
    ("七点半到九点能开就好", True, "19:30", False, False),
    ("七点半至九点这段时间都能打", False, "19:30", False, False),
    ("七点半到九点能开就好，老板不要帮我定具体开场", False, "19:30", False, False),
    ("七点半到九点都行，你帮我选个开场", False, "20:00", True, False),
    ("七点半到九点都行，还是七点半开吧", True, "19:30", True, True),
])
def test_opening_interval_preserves_read_and_requires_separate_authority_for_fixed_create(
    ending, model_collapses_window, selected_start, authorized, explicit_point, tmp_path,
):
    """引用区间端点不能写死开场，本人明确选时或委托选时仍可实际建局。"""
    sent = datetime.fromisoformat("2026-11-25T19:02:00+08:00")
    message = _incoming(f"对 帮我组一桌 四川麻将 无烟的 0.5到1块底都行 {ending}", sent=sent)
    store = SQLiteAgentStore(tmp_path / "fixed-opening-range.sqlite3")
    bounds = {"start_at": "2026-11-25T19:30:00+08:00", "end_at": "2026-11-25T21:00:00+08:00"}
    point = f"2026-11-25T{selected_start}:00+08:00"
    fixed_window = {"start_at": point, "end_at": point}
    model_window = fixed_window if model_collapses_window else bounds
    stakes = {"min": 0.5, "max": 1}
    clock_roles = ["opening", "opening"] + (["opening"] if explicit_point else [])
    verification = _multi_decision(*clock_roles, opening_customer_ref=1, has_fixed_opening_authority=True)
    review = ReviewClient([verification | {"opening_window": model_window, "accepted_stake_range": stakes}])
    creation = ReviewClient([{"scope": "create", "customer_ref": 1, "shop_offer_ref": None}])
    trace = InMemoryTraceRecorder()
    service = ToolExecutionService(store, ToolGateway(store), trace, table_availability_client=review,
                                   creation_authorization_client=creation)
    terms = {"game_type": "sichuan_mahjong", "smoke_preference": "no_smoking", "stake_range": stakes}

    with using_clock(lambda: sent):
        _activate(store, started=sent)
        _persist(store, message)
        reading_window = fixed_window if explicit_point else bounds
        lookup = _execute_scope_search(service, message, terms | {"start_time_range": reading_window}, step=1)
        assert lookup.called and lookup.allowed and lookup.result["requirement"]["start_time_range"] == reading_window
        call = ToolCall("create_game", {"organizer_id": "guest", "organizer_name": "顾客", "known_players": [],
            "requirement": terms | {"planned_start_at": point, "start_time_kind": "scheduled"}})
        created = service.execute_tool_calls(
            AgentAction(goal="组织同一桌", objective_status="needs_tool", reasoning_summary="核对开场并组局", tool_calls=[call]),
            message=message, trace_id=message.message_id, previous_step_tool_results=[lookup], step_index=2,
            run_id="verify-range-fixed-opening", run_version=store.conversation_version("dm"),
        ).tool_results[0]

    if authorized:
        assert created.called and created.allowed and len(creation.calls) == 1
        game, = store.games.values()
        assert game.planned_start_at.isoformat() == point
    else:
        assert not created.called and not created.allowed and not creation.calls and not store.games
        evidence = created.result["table_availability_evidence"]
        assert evidence["has_fixed_opening_authority"] is False and evidence["opening_customer_ref"] is None
        assert evidence["opening_window"] == bounds and "还没选定" in created.error
    assert len(review.calls) == 1
    store._connection.close()


@pytest.mark.parametrize(("text", "hour", "needs_review"), [
    ("帮我组一桌杭麻，今晚都行", 11, True),
    ("帮我组个下午都能开的川麻", 11, True),
    ("老板 那就人齐了开吧 午后都行 一点底杭麻", 9, True),
    ("明晚有桌吗", 22, True),
    ("老板晚上好，杭麻打无烟的", 11, False),
    ("今晚可不可以通知我一下", 11, False),
    ("午后可以 通知我一下", 9, False),
    ("午后都行动了", 9, False),
    ("晚上好，今天都行", 11, False),
    ("昨天晚上都行", 11, False),
    ("次日晚上都行", 11, False),
    ("今晚都行", 23, False),
    ("川麻人齐开就好", 11, False),
])
def test_named_opening_review_only_activates_for_explicit_future_scope_without_a_clock(text, hour, needs_review):
    """只有明确未来开场时段才补走时间核对，普通问候和当前人齐开的默认路径照常。"""
    message = _incoming(text, sent=SENT.replace(hour=hour))
    assert qualified_table_clock(message) is None
    assert requires_table_clock_review(message) is needs_review


@pytest.mark.parametrize(("wording", "selected_time", "delegated", "model_invents_point"), [
    ("今晚都行", None, False, False),
    ("今晚都行", None, False, True),
    ("今晚都行，你帮我选个开场", "20:00", True, False),
    ("今晚八点开吧", "20:00", False, False),
])
def test_rolling_0437_keeps_future_named_opening_and_uses_real_write_permission(
    wording, selected_time, delegated, model_invents_point, tmp_path,
):
    """白天来话说今晚，允许查当晚的桌，不能安排在晚上前已过期的人齐开。"""
    sent = datetime.fromisoformat("2026-11-26T11:07:00+08:00")
    message = _incoming(f"老板好 那就帮我组一桌吧 杭麻 底注1到3 无烟 {wording}", sent=sent)
    store = SQLiteAgentStore(tmp_path / "named-evening.sqlite3")
    evening = {"start_at": "2026-11-26T18:00:00+08:00", "end_at": "2026-11-27T06:00:00+08:00"}
    point = "2026-11-26T20:00:00+08:00"
    point_window = {"start_at": point, "end_at": point}
    explicit = selected_time is not None and not delegated
    stakes = {"min": 1, "max": 3}
    decision = _decision(clock_role="opening" if explicit else "other",
        opening_window=point_window if explicit or model_invents_point else None,
        opening_dayparts=None, accepted_stake_range=stakes,
        has_fixed_opening_authority=selected_time is not None or model_invents_point,
        opening_customer_ref=1 if selected_time is not None or model_invents_point else None,
    )
    review = ReviewClient([decision])
    creation = ReviewClient([{"scope": "create", "customer_ref": 1, "shop_offer_ref": None}])
    trace = InMemoryTraceRecorder()
    service = ToolExecutionService(store, ToolGateway(store), trace, table_availability_client=review,
                                   creation_authorization_client=creation)
    terms = {"game_type": "hangzhou_mahjong", "smoke_preference": "no_smoking", "stake_range": stakes}
    with using_clock(lambda: sent):
        _activate(store, started=sent)
        _persist(store, message)

        def create(step, schedule):
            call = ToolCall("create_game", {"organizer_id": "guest", "organizer_name": "顾客", "known_players": [],
                                            "requirement": {**terms, **schedule}})
            return service.execute_tool_calls(
                AgentAction(goal="组桌", objective_status="needs_tool", reasoning_summary="核对开场", tool_calls=[call]),
                message=message, trace_id=message.message_id, previous_step_tool_results=[], step_index=step,
                run_id="named-opening-private", run_version=store.conversation_version("dm"),
            ).tool_results[0]

        if not explicit:
            premature = create(1, {"start_time_kind": "asap_when_full"})
            assert not premature.called and not premature.allowed and not creation.calls and not store.games
            timing = premature.result["table_availability_evidence"]
            assert timing["opening_window"] == evening
            assert timing["has_fixed_opening_authority"] is delegated
            assert "最早开场" in premature.error
        read_window = point_window if explicit else evening
        lookup = _execute_scope_search(service, message, terms | {"start_time_range": read_window}, step=2)
        assert lookup.called and lookup.allowed and lookup.result["requirement"]["start_time_range"] == read_window
        fixed = create(3, {"start_time_kind": "scheduled", "planned_start_at": point})

    if selected_time is not None:
        assert fixed.called and fixed.allowed and len(creation.calls) == 1
        game, = store.games.values()
        assert game.planned_start_at.isoformat() == point
        assert game.expires_at > sent.replace(hour=18)
    else:
        assert not fixed.called and not fixed.allowed and not creation.calls and not store.games
    assert len(review.calls) == 1 and not store.room_reservations
    observed = json.loads(review.calls[0][0][1]["content"])["current_clock_observation"]
    assert observed["clock_count"] == (1 if explicit else 0)
    if not explicit:
        assert observed["named_future_opening"] == {"date": "2026-11-26", "parts": ["evening"]}
    store._connection.close()


@pytest.mark.parametrize("review_outcome", ["ordinary_asap", "unavailable", "wrong_scope"])
def test_named_future_opening_unverified_cannot_use_quiet_asap_default(review_outcome):
    """无时间偏好的人齐开照常，明确未来时段缺失语义核对时不写成现在。"""
    sent = datetime.fromisoformat("2026-11-26T11:07:00+08:00")
    store = InMemoryAgentStore()
    ordinary = review_outcome == "ordinary_asap"
    text = ("老板晚上好，帮我组一桌杭麻，1到3块无烟，人齐开"
            if ordinary else "老板好，帮我组一桌杭麻，1到3块无烟，今晚都行")
    message = _incoming(text, sent=sent)
    response = TimeoutError("review unavailable") if review_outcome == "unavailable" else _decision(
        clock_role="other", lookup_scope="other", opening_window=None,
    )
    review = ReviewClient([] if ordinary else [response])
    service = ToolExecutionService(store, ToolGateway(store), InMemoryTraceRecorder(), table_availability_client=review)
    call = ToolCall("create_game", {"organizer_id": "guest", "organizer_name": "顾客", "known_players": [],
        "requirement": {"game_type": "hangzhou_mahjong", "stake_range": {"min": 1, "max": 3},
                        "smoke_preference": "no_smoking", "start_time_kind": "asap_when_full"}})
    with using_clock(lambda: sent):
        _activate(store, started=sent)
        _persist(store, message)
        result = service.execute_tool_calls(
            AgentAction(goal="组桌", objective_status="needs_tool", reasoning_summary="按原话核对", tool_calls=[call]),
            message=message, trace_id=message.message_id, previous_step_tool_results=[], step_index=1,
            run_id="named-opening-without-verification", run_version=store.conversation_version("dm"),
        ).tool_results[0]

    if ordinary:
        assert result.called and result.allowed and not review.calls
        game, = store.games.values()
        assert game.expires_at.isoformat() == "2026-11-26T15:07:00+08:00"
    else:
        assert not result.called and not result.allowed and not store.games and len(review.calls) == 1


def test_named_future_opening_keeps_late_same_origin_date_anchor():
    """晚上过了才给的「明晚」仍用明确次日晚间，不套当天已过的开场。"""
    sent = SENT.replace(hour=23)
    message = _incoming("明晚都行", sent=sent)
    store = InMemoryAgentStore()
    _activate(store, started=sent)
    _persist(store, message)
    review = ReviewClient([_decision(clock_role="other", opening_window=None, opening_dayparts=None)])
    checked = _check(store, review, message)

    assert checked["opening_window"] == {"start_at": "2026-11-27T18:00:00+08:00",
                                         "end_at": "2026-11-28T06:00:00+08:00"}
    assert checked["has_fixed_opening_authority"] is False


@pytest.mark.parametrize(("text", "observed"), [
    ("今天杭州麻将一点底的还有位吗", []),
    ("今天有1点档的杭麻吗", []),
    ("今天有一点元底的杭麻吗", []),
    ("红中一点底，下午一点开", ["一点"]),
    ("红中一点底，下午一点到", ["一点"]),
    ("红中一点底，下午两点半到，四点开", ["两点半", "四点"]),
    ("下午一点底注一块", ["一点"]),
    ("川麻一点的小局", ["一点"]),
])
def test_stake_units_preserve_only_actual_or_still_ambiguous_clock_candidates(text, observed):
    """带底注单位的点数不占用参考编号，普通钟点与含糊表达仍给模型核对。"""
    parsed = qualified_table_clock(_incoming(text))
    actual = parsed.get("candidates", [parsed]) if parsed else []
    assert [item["clock_text"] for item in actual] == observed
    if len(actual) > 1:
        assert [item["clock_ref"] for item in actual] == list(range(1, len(actual) + 1))
    group_projection = parse_explicit_need(text, anchor=SENT)
    if observed:
        assert group_projection["clock_observation"]["clock_text"] == observed[0]
        assert group_projection["clock_observation"]["clock_count"] == len(observed)
    else:
        assert "clock_observation" not in group_projection
        assert "planned_start_at" not in group_projection


def test_rolling_0038_stake_only_message_does_not_lock_broad_search_to_one_oclock(tmp_path):
    """同一来话可查下午的不同开场，不把一点底写成13:00的查询条件。"""
    sent = datetime.fromisoformat("2026-11-26T09:37:00+08:00")
    message = _incoming("老板 那今天杭州麻将一点底的还有位吗 组一桌也行", sent=sent)
    store = SQLiteAgentStore(tmp_path / "one-point-stake-lookup.sqlite3")
    review = ReviewClient([])
    trace = InMemoryTraceRecorder()
    service = ToolExecutionService(store, ToolGateway(store), trace, table_availability_client=review)
    span = {"start_at": "2026-11-26T12:00:00+08:00", "end_at": "2026-11-26T18:00:00+08:00"}
    with using_clock(lambda: sent):
        _activate(store, started=sent)
        _persist(store, message)
        games = []
        for opening in ("14:00", "16:00"):
            game, _ = store.create_game(conversation_id=opening, organizer_id=opening, organizer_name="桌主",
                trace_id=opening, known_players=[{"customer_id": opening}],
                requirement={"game_type": "hangzhou_mahjong", "stake": 1,
                             "planned_start_at": f"2026-11-26T{opening}:00+08:00"})
            games.append(game.game_id)
        lookup = _execute_scope_search(
            service, message, {"game_type": "hangzhou_mahjong", "stake": 1, "start_time_range": span}, step=1,
        )
    assert lookup.called and lookup.allowed and lookup.result["requirement"]["start_time_range"] == span
    assert {row["game"]["game_id"] for row in lookup.result["matches"]} == set(games)
    assert not review.calls and not store.room_reservations and len(store.games) == 2
    store._connection.close()


@pytest.mark.parametrize("write_authorized", [True, False])
def test_same_game_opening_confirmation_carries_prior_flexible_departure_into_reply_evidence(tmp_path, write_authorized):
    """跨轮保留顾客柔性离场目标，仍经真实建局守卫，不写硬限时或默认安排。"""
    sent = SENT.replace(hour=19)
    earlier = sent - timedelta(minutes=2)
    prior = _incoming("老板好 组一桌还行吗 能打到十点半 红中2到3块 无烟 我七点半左右能到",
                      sent=earlier, message_id="prior-end")
    current = _incoming("老板好 那就按七点半开吧 红中2到3块 无烟 帮我组一桌", sent=sent)
    store = SQLiteAgentStore(tmp_path / "prior-play-until.sqlite3")
    _activate(store, started=earlier)
    with using_clock(lambda: earlier):
        _persist(store, prior)
        store.append_assistant_turn("dm", "你这桌打算几点开？我按这个时间帮你组一桌。",
                                    "shop", metadata={"delivery_status": "delivered"})
    with using_clock(lambda: sent):
        _persist(store, current)
        opening = "2026-11-26T19:30:00+08:00"
        desired_end = "2026-11-26T22:30:00+08:00"
        reference = {"customer_ref": 1, "clock_ref": 1}
        decision = _decision(3, clock_role="opening", opening_customer_ref=3,
                             opening_window={"start_at": opening, "end_at": opening},
                             supporting_customer_refs=[1, 3], has_fixed_opening_authority=True,
                             prior_play_until=reference)
        review = ReviewClient([decision])
        creation = ReviewClient([{"scope": "create" if write_authorized else "inquire",
                                  "customer_ref": 3, "shop_offer_ref": None}])
        trace = InMemoryTraceRecorder()
        service = ToolExecutionService(store, ToolGateway(store), trace, table_availability_client=review,
                                       creation_authorization_client=creation)
        requirement = {"game_type": "red_center_mahjong", "stake_range": {"min": 2, "max": 3},
                       "smoke_preference": "no_smoking", "planned_start_at": opening, "start_time_kind": "scheduled"}

        def execute(step, **updates):
            call = ToolCall("create_game", {"organizer_id": "guest", "organizer_name": "顾客", "known_players": [],
                                            "requirement": {**requirement, **updates}})
            action = AgentAction(goal="按确认开场组织同一桌", objective_status="needs_tool", reasoning_summary="已选开场",
                                 tool_calls=[call])
            return service.execute_tool_calls(action, message=current, trace_id=current.message_id, step_index=step,
                previous_step_tool_results=[], run_id="carry-prior-duration",
                run_version=store.conversation_version("dm")).tool_results[0]

        for index, field, value in ((1, "max_duration_hours", 3), (2, "duration_hours", 3)):
            invalid = execute(index, **{field: value})
            assert not invalid.called and not invalid.allowed
            assert invalid.result["table_availability_evidence"]["play_until_at"] == desired_end
        assert not store.games and not creation.calls
        created = execute(3)

    assert len(creation.calls) == 1
    if not write_authorized:
        assert not created.allowed and not store.games
        assert "table_availability_evidence" not in created.result
        store._connection.close()
        return
    assert created.called and created.allowed and len(store.games) == 1
    game = next(iter(store.games.values()))
    assert game.planned_start_at.isoformat() == opening
    assert all(key not in game.requirement for key in ("must_end_by", "max_duration_hours", "duration_hours"))
    proof = {"play_until_at": desired_end, "prior_play_until": {**reference, "source_message_id": "prior-end"}}
    compact = compact_tool_payload(created.result)
    assert compact["table_availability_evidence"] == proof
    action = AgentAction(goal="告诉顾客这桌正在组", objective_status="waiting_user", reasoning_summary="询问离场条件",
                         reply_to_user="七点半这桌给你记上了。十点半是得走吗？")
    presented = build_reply_self_review_payload(
        message=current, action=action, review_items=[{"recipient_id": "guest", "text": action.reply_to_user}],
        context_payload={"turn_tool_evidence": [{**created.to_dict(), "result": compact}]}, review_scope="reply_to_user",
    )
    assert presented["turn_tool_evidence"][0]["result"]["table_availability_evidence"] == proof
    assert len(review.calls) == 1
    history = json.loads(review.calls[0][0][1]["content"])["conversation"]
    assert history[0]["clock_candidates"][0]["local_candidate_at"] == desired_end
    store._connection.close()


@pytest.mark.parametrize(("text", "source_change", "citation", "current_hard", "agreed_duration"), [
    ("那就按七点半开吧，最晚十点走", None, {"customer_ref": 1, "clock_ref": 1}, True, None),
    ("那就按七点半开吧，就打两个钟头", None, {"customer_ref": 1, "clock_ref": 1}, False, 2),
    ("今晚就算了，之前时间不用管", None, None, False, None),
    ("另一桌改成今晚七点半开", "new_task", {"customer_ref": 1, "clock_ref": 1}, False, None),
    ("那就按七点半开吧", "prior_day", {"customer_ref": 1, "clock_ref": 1}, False, None),
    ("那就按七点半开吧", None, {"customer_ref": 2, "clock_ref": 1}, False, None),
    ("那就按七点半开吧", None, {"customer_ref": 1, "clock_ref": 3}, False, None),
])
def test_prior_flexible_departure_yields_to_current_scope_or_replacement(text, source_change, citation,
                                                                 current_hard, agreed_duration):
    """日期和原话引用收窄继承，明确新条件及撤回继续按本轮原话核对。"""
    sent = SENT.replace(hour=19)
    earlier = sent - (timedelta(days=1, minutes=2) if source_change == "prior_day" else timedelta(minutes=2))
    store = InMemoryAgentStore()
    _activate(store, started=earlier)
    _persist(store, _incoming("能打到十点半 红中无烟", sent=earlier, message_id="prior-end"))
    if source_change == "new_task":
        _activate(store, started=sent, reset=True)
    message = _incoming(text, sent=sent)
    _persist(store, message)
    if citation is None:
        assert _check(store, ReviewClient([]), message) is None
        return
    current_ref = 1 if source_change == "new_task" else 2
    opening = "2026-11-26T19:30:00+08:00"
    specific = _decision(current_ref, clock_role="opening",
        opening_window={"start_at": opening, "end_at": opening}, has_fixed_opening_authority=True,
        opening_customer_ref=current_ref, supporting_customer_refs=list(range(1, current_ref + 1)),
        prior_play_until=citation, agreed_duration_hours=agreed_duration)
    if current_hard:
        specific["clock_role"] = "other"
        specific["clock_roles"] = [{"clock_ref": 1, "role": "opening"}, {"clock_ref": 2, "role": "departure_deadline"}]
    checked = _check(store, ReviewClient([specific]), message)
    assert checked is not None and "prior_play_until" not in checked
    if current_hard:
        assert checked["must_end_by"] == checked["play_until_at"] == "2026-11-26T22:00:00+08:00"
    else:
        assert "must_end_by" not in checked and "play_until_at" not in checked
    assert checked["agreed_duration_hours"] == agreed_duration


@pytest.mark.parametrize(("prior_text", "clock_ref", "expected"), [
    ("我八点到店", 1, None),
    ("我八点到店，能打到十点半", 1, None),
    ("我八点到店，能打到十点半", 2, "22:30"),
    ("最晚能打到十点半", 1, None),
    ("必须打到十点半", 1, None),
    ("只能打到十点半", 1, None),
    ("能打到十点半就得走", 1, None),
    ("我不能打到十点半", 1, None),
    ("最晚八点能到店，想玩到晚上十点半", 2, "22:30"),
    ("最晚八点能到店 想玩到晚上十点半", 2, "22:30"),
    ("我可以打到大概十点半左右", 1, "22:30"),
])
def test_prior_flexible_clock_must_belong_to_its_own_play_until_clause(prior_text, clock_ref, expected):
    """模型误引其它时间或硬边界时丢弃柔性引用，别句的限定词不能互相污染。"""
    sent = SENT.replace(hour=19)
    earlier = sent - timedelta(minutes=2)
    store = InMemoryAgentStore()
    _activate(store, started=earlier)
    _persist(store, _incoming(prior_text, sent=earlier, message_id="previous-source"))
    message = _incoming("那就按七点半开吧", sent=sent)
    _persist(store, message)
    opening = "2026-11-26T19:30:00+08:00"
    decision = _decision(2, clock_role="opening", opening_customer_ref=2,
        opening_window={"start_at": opening, "end_at": opening}, has_fixed_opening_authority=True,
        supporting_customer_refs=[1, 2], prior_play_until={"customer_ref": 1, "clock_ref": clock_ref})
    checked = _check(store, ReviewClient([decision]), message)

    assert checked["has_fixed_opening_authority"] is True and "must_end_by" not in checked
    if expected is None:
        assert "prior_play_until" not in checked and "play_until_at" not in checked
    else:
        assert checked["play_until_at"] == f"2026-11-26T{expected}:00+08:00"
        assert checked["prior_play_until"]["source_message_id"] == "previous-source"


def test_evening_arrival_can_disambiguate_same_date_departure_without_selecting_opening(tmp_path):
    """同句已核实晚间到店，查局保留离店限时和原档位，不授予约开场。"""
    at = datetime.fromisoformat("2026-11-26T09:09:30+08:00")
    message = _incoming("老板上午好，今天傍晚六点半以后我能到店，最多打到十点，"
                        "你帮我看看有没有无烟的川麻桌能搭一桌吗，底注2、3都行", sent=at)
    store = SQLiteAgentStore(tmp_path / "arrival-with-departure.sqlite3")
    stakes = {"min": 2, "max": 3}
    verification = _multi_decision("arrival_not_before", "departure_deadline", opening_dayparts=None,
                                   max_duration_hours=None, agreed_duration_hours=None)
    review = ReviewClient([verification | {"accepted_stake_range": stakes}])
    service = ToolExecutionService(store, ToolGateway(store), InMemoryTraceRecorder(), table_availability_client=review)
    _activate(store, started=at)
    with using_clock(lambda: at):
        _persist(store, message)
        games = {}
        for start, duration in (("18:00", 3), ("19:00", 3), ("20:00", 3)):
            game, _ = store.create_game(conversation_id=start, organizer_id=start, organizer_name="桌主",
                trace_id=start, known_players=[{"customer_id": start}],
                requirement={"game_type": "sichuan_mahjong", "stake": 2, "smoke_preference": "no_smoking",
                             "planned_start_at": f"2026-11-26T{start}:00+08:00", "duration_hours": duration})
            games[start] = game.game_id
        floor, deadline = "2026-11-26T18:30:00+08:00", "2026-11-26T22:00:00+08:00"
        requirement = {"game_type": "sichuan_mahjong", "stake_range": stakes, "smoke_preference": "no_smoking",
                       "start_time_range": {"start_at": floor, "end_at": deadline}}
        rejected = _execute_scope_search(service, message, requirement, step=1)
        assert not rejected.called and not rejected.allowed
        assert f"must_end_by={deadline}" in rejected.error
        evidence = rejected.result["table_availability_evidence"]
        assert evidence["arrival_lookup_floor_at"] == floor
        assert evidence["must_end_by"] == evidence["play_until_at"] == deadline
        assert evidence["opening_window"] is None and evidence["has_fixed_opening_authority"] is False
        assert evidence["agreed_duration_hours"] is None and evidence["max_duration_hours"] is None
        lookup = _execute_scope_search(service, message, {**requirement, "must_end_by": deadline}, step=2)
    assert lookup.called and lookup.allowed and not lookup.error
    assert [row["game"]["game_id"] for row in lookup.result["matches"]] == [games["19:00"]]
    assert len(review.calls) == 1 and len(store.games) == 3 and not store.room_reservations
    store._connection.close()


@pytest.mark.parametrize(("text", "arrival_role", "deadline_role", "cutoff"), [
    ("今天傍晚六点半以后我能到，最多打到上午十点", "arrival_not_before", "departure_deadline", "26T10:00"),
    ("今天傍晚六点半以后我能到，最多打到十点 上午", "arrival_not_before", "departure_deadline", "26T10:00"),
    ("今天傍晚六点半以后我能到，最多打到十点 AM", "arrival_not_before", "departure_deadline", "26T10:00"),
    ("今天傍晚六点半以后我能到，最多打到09点30分", "arrival_not_before", "departure_deadline", "26T09:30"),
    ("傍晚六点半以后我能到，最多打到十点", "arrival_not_before", "departure_deadline", "26T10:00"),
    ("傍晚六点半以后我能到，最多打到十点，你今天开门吗", "arrival_not_before", "departure_deadline", "26T10:00"),
    ("2026-11-26傍晚六点半以后我能到，最多打到十点", "arrival_not_before", "departure_deadline", "26T22:00"),
    ("今天傍晚六点半以后我能到，最多打到次日十点", "arrival_not_before", "departure_deadline", "26T10:00"),
    ("今天傍晚六点半以后我能到，最多打到明天十点", "arrival_not_before", "departure_deadline", "27T10:00"),
    ("今天傍晚六点半以后我能到，想打到十点左右", "arrival_not_before", "play_until", None),
    ("今天傍晚六点半之前我能到，最多打到十点", "arrival_by", "departure_deadline", "26T10:00"),
    ("老板晚上好，今天我六点半以后能到，最多打到十点", "arrival_not_before", "departure_deadline", "26T10:00"),
    ("今天傍晚六点半以后我能到，最多打到一点", "arrival_not_before", "departure_deadline", "26T13:00"),
    ("我今晚七点左右到，必须在十点半前走，找个川麻桌", "arrival_estimate", "departure_deadline", "26T22:30"),
])
def test_arrival_scoped_departure_preserves_qualified_clocks_and_real_ambiguity(text, arrival_role, deadline_role, cutoff):
    """保留具体早晨、跨日日期、柔性时长和缺少限定的钟点作为独立边界。"""
    at = datetime.fromisoformat("2026-11-26T09:09:30+08:00")
    message = _incoming(text, sent=at)
    store = InMemoryAgentStore()
    _activate(store, started=at)
    _persist(store, message)
    decision = _check(store, ReviewClient([_multi_decision(arrival_role, deadline_role, opening_dayparts=None)]), message)

    expected = f"2026-11-{cutoff}:00+08:00" if cutoff is not None else None
    assert decision.get("must_end_by") == expected
    assert decision["opening_window"] is None and decision["has_fixed_opening_authority"] is False


@pytest.mark.parametrize(("text", "role", "floor", "game_type", "stakes", "explicit_evening"), [
    ("老板好，晚上六点半后有空，想打四川麻将，2到3元底，无烟的，能搭一桌吗？",
     "arrival_not_before", "18:30", "sichuan_mahjong", {"min": 2, "max": 3}, True),
    ("老板好 看板上红中那桌烟况待定 我今晚七点半能到 想打杭麻 有没有0.5档或1档的杭麻桌",
     "arrival_estimate", "19:30", "hangzhou_mahjong", {"min": 0.5, "max": 1}, True),
    ("老板晚上好，杭麻0.5档或1档有吗，我七点半能到",
     "arrival_estimate", "19:30", "hangzhou_mahjong", {"min": 0.5, "max": 1}, False),
])
def test_personal_evening_request_can_read_matching_stock_after_midnight(text, role, floor, game_type, stakes,
                                                                  explicit_evening, tmp_path):
    """复现两条滚动原话和问候对照，只有独立核实为晚间查局才能查次日清晨。"""
    at = SENT.replace(minute=9, second=30)
    store = SQLiteAgentStore(tmp_path / "overnight-personal.sqlite3")
    message = _incoming(text, sent=at)
    dayparts = {"date": at.date().isoformat(), "parts": ["evening"], "source_ref": 1} if explicit_evening else None
    review = ReviewClient([_decision(clock_role=role, opening_window=None, opening_dayparts=dayparts,
                                    accepted_stake_range=stakes)])
    service = ToolExecutionService(store=store, tool_gateway=ToolGateway(store), trace_recorder=InMemoryTraceRecorder(),
                                   table_availability_client=review)
    with using_clock(lambda: at):
        _activate(store, started=at)
        _persist(store, message)
        games = {}
        for host, starts in (("night", "2026-11-26T20:00:00+08:00"), ("after-midnight", "2026-11-27T00:30:00+08:00")):
            game, _ = store.create_game(conversation_id=host, organizer_id=host, organizer_name="桌主",
                known_players=[{"customer_id": host}], trace_id=host, requirement={
                    "game_type": game_type, "stake": stakes["max"], "planned_start_at": starts,
                })
            games[host] = game.game_id

        def query(end, step):
            tool = ToolCall("search_current_games", {"lookup_scope": "personal", "include_nearby": False, "requirement": {
                "game_type": game_type, "stake_range": stakes, "start_time_range": {
                    "start_at": f"2026-11-26T{floor}:00+08:00", "end_at": end,
                },
            }})
            action = AgentAction(goal="找合适的现成桌", objective_status="needs_tool", reasoning_summary="查现成桌",
                                 tool_calls=[tool])
            return service.execute_tool_calls(action, message=message, trace_id=message.message_id, step_index=step,
                previous_step_tool_results=[], run_id="personal-evening", run_version=store.conversation_version("dm")).tool_results[0]

        overnight = query("2026-11-27T06:00:00+08:00", 1)
        if explicit_evening:
            assert overnight.called and overnight.allowed
            assert {option["game"]["game_id"] for option in overnight.result["matches"]} == set(games.values())
        else:
            assert not overnight.called and not overnight.allowed and "2026-11-26T23:59:59.999999+08:00" in overnight.error
            same_day = query("2026-11-26T23:59:59+08:00", 2)
            assert same_day.called and same_day.allowed
            assert {option["game"]["game_id"] for option in same_day.result["matches"]} == {games["night"]}

    assert len(review.calls) == 1 and len(store.games) == 2 and not store.room_reservations
    store._connection.close()


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_han_minutes_keep_the_exact_arrival_lookup_boundary(backend, tmp_path):
    """实际工具保留本人来话的分钟，过滤更早的桌并拒绝可能提前开场的建局。"""
    at = SENT.replace(hour=19, minute=30)
    store = SQLiteAgentStore(tmp_path / "han-minutes.sqlite3") if backend == "sqlite" else InMemoryAgentStore()
    message = _incoming("老板现在有一块底的无烟桌吗？我七点四十一能到，没现成的就帮我凑一桌", sent=at)
    review = ReviewClient([_decision(clock_role="arrival_estimate", opening_window=None,
                                     accepted_stake_range={"min": 1, "max": 1})])
    authorization = ReviewClient([{"scope": "create", "customer_ref": 1, "shop_offer_ref": None}])
    service = ToolExecutionService(store=store, tool_gateway=ToolGateway(store), trace_recorder=InMemoryTraceRecorder(),
                                   table_availability_client=review, creation_authorization_client=authorization)
    games = {}
    with using_clock(lambda: at):
        _activate(store, started=at)
        _persist(store, message)
        for opening in ("19:40", "19:45"):
            host = f"host-{opening}"
            games[opening], _ = store.create_game(conversation_id=host, organizer_id=host, organizer_name="桌主",
                known_players=[{"customer_id": host}], trace_id=host, requirement={
                    "smoke_preference": "no_smoking", "stake": "1", "planned_start_at": f"2026-11-26T{opening}:00+08:00",
                })

        def submit(tool, step):
            action = AgentAction(goal="帮顾客找一桌能赶上的无烟桌", objective_status="needs_tool", reasoning_summary="核对条件",
                                 tool_calls=[tool])
            return service.execute_tool_calls(action, message=message, trace_id=message.message_id,
                previous_step_tool_results=[], step_index=step, run_id="han-minutes",
                run_version=store.conversation_version(message.conversation_id)).tool_results[0]

        def query(start):
            return ToolCall("search_current_games", {"requirement": {"smoke_preference": "no_smoking", "stake": "1", "start_time_range": {
                "start_at": f"2026-11-26T{start}:00+08:00", "end_at": "2026-11-26T23:59:00+08:00",
            }}, "lookup_scope": "personal", "include_nearby": False})

        wrong = submit(query("19:00"), 1)
        assert not wrong.allowed and not wrong.called and "19:41:00+08:00" in wrong.error
        result = submit(query("19:41"), 2)
        too_early = submit(ToolCall("create_game", {"organizer_id": "guest", "organizer_name": "顾客", "known_players": [],
            "requirement": {"start_time_kind": "asap_when_full", "smoke_preference": "no_smoking", "stake": "1",
                            "extra_constraints": ["19:41 到"]}}), 3)
        assert not too_early.called and not too_early.allowed and "人齐开" in too_early.error
        assert too_early.result["table_availability_evidence"]["arrival_lookup_floor_at"] == "2026-11-26T19:41:00+08:00"

    assert result.allowed and result.called and result.error is None
    assert {row["game"]["game_id"] for row in result.result["matches"]} == {games["19:45"].game_id}
    assert set(store.games) == {game.game_id for game in games.values()} and not store.room_reservations
    assert not authorization.calls and len(review.calls) == 1
    if backend == "sqlite":
        store._connection.close()


@pytest.mark.parametrize("qualifier", ["前后", "左右", ""])
def test_personal_approximate_opening_reads_actual_band_without_selecting_a_start(tmp_path, qualifier):
    """V4 原话能查九点前后的真实桌，保留三小时硬上限和真正固定钟点的对照。"""
    at = datetime.fromisoformat("2026-11-25T19:00:00+08:00")
    center = "2026-11-25T21:00:00+08:00"
    actual_range = {"start_at": "2026-11-25T20:30:00+08:00", "end_at": "2026-11-25T21:30:00+08:00"}
    store = SQLiteAgentStore(tmp_path / "opening-approximation.sqlite3")
    message = _incoming(f"老板好 想问今晚9点{qualifier}有没有无烟的杭州麻将 底注2到3 我最多打三个钟 能搭一桌吗", sent=at)
    client = ReviewClient([_decision(clock_role="opening", clock_roles=None, max_duration_hours=3,
        agreed_duration_hours=None, opening_window={"start_at": center, "end_at": center},
        opening_dayparts=None, opening_customer_ref=None)])
    trace = InMemoryTraceRecorder()
    service = ToolExecutionService(store=store, tool_gateway=ToolGateway(store), trace_recorder=trace,
                                   table_availability_client=client)
    condition = {"game_type": "hangzhou_mahjong", "smoke_preference": "no_smoking"}

    def submit(name, arguments, step):
        action = AgentAction(goal="给顾客查今晚现成桌", objective_status="needs_tool", reasoning_summary="核实可打时段",
                             tool_calls=[ToolCall(name, arguments)])
        return service.execute_tool_calls(action, message=message, trace_id=message.message_id,
            previous_step_tool_results=[], step_index=step, run_id="around-opening",
            run_version=store.conversation_version(message.conversation_id)).tool_results[0]

    def read(window, step, *, cap=3, nearby=False):
        requirement = {**condition, "stake_range": {"min": 2, "max": 3}, "start_time_range": window}
        if cap is not None:
            requirement["max_duration_hours"] = cap
        return submit("search_current_games", {"lookup_scope": "personal", "requirement": requirement,
                                              "include_nearby": nearby}, step)

    with using_clock(lambda: at):
        _activate(store, started=at)
        _persist(store, message)
        games = {}
        for opening, hours in (("20:15", 2), ("20:35", 3), ("20:55", 2), ("21:00", 3), ("21:08", 3),
                               ("21:10", 4), ("21:25", 2), ("22:00", 2)):
            host = f"around-{opening}"
            game, _ = store.create_game(conversation_id=host, organizer_id=host, organizer_name=host,
                known_players=[{"customer_id": host}], requirement={**condition, "stake": "2", "duration_hours": hours,
                    "planned_start_at": f"2026-11-25T{opening}:00+08:00"}, trace_id=host)
            games[opening] = game.game_id

        assert "最多可打时长" in read(actual_range, 1, cap=None).error
        result = read(actual_range, 2)
        assert result.called is bool(qualifier)
        if qualifier:
            assert result.result["requirement"]["start_time_range"] == actual_range
            expected = {games[at] for at in ("20:35", "20:55", "21:00", "21:08", "21:25")}
            matches = result.result["matches"]
            assert {row["game"]["game_id"] for row in matches} == expected
            flagged = expected - {games["21:00"]}
            semantics = result.result["customer_reply_contract"]["search_result_semantics"]
            assert semantics["actionable_match_count"] == 1
            assert semantics["timing_confirmation_required_count"] == len(flagged)
            assert semantics["returned_game_count"] == len(expected)
            compact = compact_tool_payload(result.result)
            assert {row["game"]["game_id"] for row in compact["matches"] if row.get("requires_timing_confirmation")} == flagged
            review_action = AgentAction(goal="介绍实际开场", objective_status="waiting_user", reasoning_summary="等顾客确认")
            presented = build_reply_self_review_payload(message=message, action=review_action, review_items=[],
                context_payload={"turn_tool_evidence": [result.to_dict()]}, review_scope="reply_to_user")
            timing = presented["semantic_fidelity_contract"]["game_availability_evidence"]["latest_search"]
            assert {row["game_id"] for row in timing["options_requiring_timing_confirmation"]} == flagged
            point = read({"start_at": center, "end_at": center}, 3, nearby=True)
            assert {row["game"]["game_id"] for row in point.result["nearby_options"]} == {games["20:55"], games["21:08"]}
            assert "包含该中心" in read({"start_at": center, "end_at": center}, 4).error
            assert "包含该中心" in read({"start_at": "2026-11-25T20:00:00+08:00",
                                         "end_at": "2026-11-25T20:45:00+08:00"}, 5).error
            assert "范围保持" in read({"start_at": "2026-11-25T20:30:00+08:00",
                                      "end_at": "2026-11-26T07:00:00+08:00"}, 6).error
            for step, opening in enumerate(({"planned_start_at": center}, {"start_time_kind": "asap_when_full"}), 7):
                create = submit("create_game", {"organizer_id": "guest", "organizer_name": "顾客", "known_players": [],
                    "requirement": {**condition, "stake_range": {"min": 2, "max": 3}, "max_duration_hours": 3,
                                    **opening}}, step)
                assert not create.allowed and not create.called and "近似开场" in create.error
                assert create.result["table_availability_evidence"]["opening_window"] is None
        else:
            assert "按核实的时段" in result.error
            exact = read({"start_at": center, "end_at": center}, 3)
            assert [row["game"]["game_id"] for row in exact.result["matches"]] == [games["21:00"]]
            assert not exact.result["matches"][0].get("requires_timing_confirmation")

    assert set(store.games) == set(games.values()) and not store.room_reservations and len(client.calls) == 1
    store._connection.close()


@pytest.mark.parametrize("hours,option_kind", [(2, "matches"), (None, "alternatives")])
def test_only_off_center_inventory_is_a_timing_decision_in_review_and_offer(hours, option_kind):
    """真正查到的单桌仍可介绍，时长待核实的备选也保留时间待确认标记。"""
    at = datetime.fromisoformat("2026-11-25T19:00:00+08:00")
    store = InMemoryAgentStore()
    message = _incoming("九点前后最多打三个钟，有局吗？", sent=at)
    action = AgentAction(goal="介绍可选牌局", objective_status="waiting_user", reasoning_summary="问顾客是否赶得上")
    opening = "2026-11-25T21:25:00+08:00"
    requirements = {"stake": "2", "game_type": "hangzhou_mahjong"}
    timing = {"nearby_opening_search": {"center_at": "2026-11-25T21:00:00+08:00",
        "period_start_at": "2026-11-25T18:00:00+08:00", "period_end_at": "2026-11-26T05:59:59.999999+08:00"}}
    with using_clock(lambda: at):
        game, _ = store.create_game(conversation_id="host", organizer_id="host", organizer_name="桌主",
            known_players=[{"customer_id": "host"}], trace_id="table", requirement={**requirements,
                "planned_start_at": opening, **({"duration_hours": hours} if hours is not None else {})})
        call = ToolCall("search_current_games", {"lookup_scope": "personal", "include_nearby": False,
            "requirement": {**requirements, "max_duration_hours": 3, "start_time_range": {
                "start_at": "2026-11-25T20:30:00+08:00", "end_at": "2026-11-25T21:30:00+08:00",
            }}})
        raw = ToolGateway(store).execute(call, trace_id="read", conversation_id="dm", sender_id="guest",
                                         sender_name="顾客", step_index=1, source_message_id=message.message_id)
        ordinary = raw.result["customer_reply_contract"]["search_result_semantics"]
        assert "timing_confirmation_required_count" not in ordinary
        if hours:
            assert ordinary["status"] == "actionable_matches" and ordinary["actionable_match_count"] == 1
            assert game_offer_for_review(store, message, action, [raw.to_dict()])["game_id"] == game.game_id

        verified = apply_approximate_opening_search_evidence(raw, timing)
        decision = verified.result["customer_reply_contract"]["search_result_semantics"]
        assert decision["status"] == "decision_required_alternatives" and decision["actionable_match_count"] == 0
        assert decision["timing_confirmation_required_count"] == decision["returned_game_count"] == 1
        compact = compact_tool_payload(verified.result)
        candidate, = compact[option_kind]
        assert candidate["game"]["game_id"] == game.game_id and candidate["requires_timing_confirmation"] is True
        if hours is None:
            assert candidate["match_kind"] == "duration_confirmation_required"
        assert game_offer_for_review(store, message, action, [{**verified.to_dict(), "result": compact}]) is None
        context = {"turn_tool_evidence": [{**verified.to_dict(), "result": compact}]}
        items = build_reply_self_review_payload(message=message, action=action, review_items=[], context_payload=context,
                                               review_scope="reply_to_user")
        latest = items["semantic_fidelity_contract"]["game_availability_evidence"]["latest_search"]
        assert latest["options_requiring_timing_confirmation"] == [{"game_id": game.game_id, "planned_start_at": opening}]
        assert "submitted_filters_without_results" not in latest


@pytest.mark.parametrize("evening_evidence", ["uncited", "cited_tonight", "greeting_only"])
def test_v5_0405_keeps_approximate_opening_center_and_hard_arrival_floor(tmp_path, evening_evidence):
    """实际 V5 两个钟点分开核验，只有确有同目标晚间原话并被引用时才允许跨日。"""
    at = datetime.fromisoformat("2026-11-25T19:00:00+08:00")
    arrival = "2026-11-25T19:41:00+08:00"
    center = "2026-11-25T19:50:00+08:00"
    night = "2026-11-26T06:00:00+08:00"
    store = SQLiteAgentStore(tmp_path / "rolling-0405-two-clocks.sqlite3")
    prior = _incoming("老板晚上好 想问下今晚还有局能搭一桌吗 无烟的 档位2到3 我大概八点前能到",
                      sent=at, message_id="0405-lookup")
    if evening_evidence == "greeting_only":
        prior = replace(prior, text=prior.text.replace("今晚", ""))
    message = _incoming("老板我这边现在最早也要七点四十一才能到 要是组局时间能往七点五十左右挪一下的话 "
                        "无烟2到3档我就能打 大概能打三个小时 你看能行吗", sent=at)
    cited = evening_evidence != "uncited"
    raw = _decision(3, clock_role="other", clock_roles=[{"clock_ref": 1, "role": "arrival_not_before"},
                                                       {"clock_ref": 2, "role": "opening"}],
                    opening_window=None, agreed_duration_hours=None, max_duration_hours=3,
                    opening_customer_ref=None, supporting_customer_refs=[1, 3] if cited else [3],
                    opening_dayparts={"date": "2026-11-25", "parts": ["evening"], "source_ref": 1} if cited else None)
    review = ReviewClient([raw])
    trace = InMemoryTraceRecorder()
    service = ToolExecutionService(store, ToolGateway(store), trace, table_availability_client=review)
    terms = {"game_type": "hangzhou_mahjong", "stake_range": {"min": 2, "max": 3}, "smoke_preference": "no_smoking"}

    def run_tool(name, arguments, step):
        action = AgentAction(goal="查实际能赶上的桌", objective_status="needs_tool", reasoning_summary="核实开场和到店",
                             tool_calls=[ToolCall(name, arguments)])
        return service.execute_tool_calls(action, message=message, trace_id=message.message_id,
            previous_step_tool_results=[], step_index=step, run_id="rolling-0405-two-clocks",
            run_version=store.conversation_version("dm")).tool_results[0]

    def read(step, start, end, *, nearby=False, extra=None):
        return run_tool("search_current_games", {"lookup_scope": "personal", "include_nearby": nearby,
            "requirement": terms | {"start_time_range": {"start_at": start, "end_at": end}} | (extra or {})}, step)

    with using_clock(lambda: at):
        _activate(store, started=at)
        _persist(store, prior)
        store.append_assistant_turn("dm", "今晚这会儿没有能搭的无烟2-3档桌诶，要不我帮你组一桌？", "offer",
                                   metadata={"delivery_status": "delivered"})
        _persist(store, message)
        too_early = "2026-11-25T19:40:00+08:00"
        later = "2026-11-25T19:57:00+08:00"
        overnight = "2026-11-26T00:10:00+08:00"
        games = {}
        for index, opening in enumerate((too_early, center, later, overnight)):
            host = f"0405-table-{index}"
            game, _ = store.create_game(conversation_id=host, organizer_id=host, organizer_name=host,
                known_players=[{"customer_id": host}], trace_id=host,
                requirement=terms | {"planned_start_at": opening, "duration_hours": 3})
            games[opening] = game.game_id

        end = "2026-11-25T20:05:00+08:00"
        approximation = read(1, arrival, end, extra={"duration_hours": 3})
        assert not approximation.called and "大致打牌时长" in approximation.error
        evidence = approximation.result["table_availability_evidence"]
        assert evidence["nearby_opening_search"]["center_at"] == center
        assert evidence["arrival_lookup_floor_at"] == arrival and evidence["opening_window"] is None
        assert evidence["agreed_duration_hours"] is None and evidence["max_duration_hours"] is None
        assert not evidence["has_fixed_opening_authority"]
        before = read(2, too_early, end)
        assert not before.called and arrival in before.error
        result = read(3, arrival, end)
        assert result.called and result.allowed
        assert result.result["requirement"]["start_time_range"] == {"start_at": arrival, "end_at": end}
        assert {row["game"]["game_id"] for row in result.result["matches"]} == {games[center], games[later]}
        neighbor, = read(4, center, center, nearby=True).result["nearby_options"]
        assert neighbor["game"]["game_id"] == games[later] and neighbor["requires_timing_confirmation"]

        next_morning = read(5, arrival, night)
        if evening_evidence == "cited_tonight":
            assert next_morning.called and next_morning.allowed
            assert {row["game"]["game_id"] for row in next_morning.result["matches"]} == {
                games[center], games[later], games[overnight],
            }
            proof, = [event.content["input"] for event in trace.get_trace(message.message_id)
                      if event.step == "table_availability_authorization_prompt"]
            cited_source, = [row for row in proof["conversation"] if row["ref"] == evidence["opening_dayparts"]["source_ref"]]
            assert cited_source["speaker"] == "customer" and cited_source["text"] == prior.text
        else:
            assert not next_morning.called and "23:59:59.999999" in next_morning.error
        for step, opening in enumerate(({"planned_start_at": center}, {"start_time_kind": "asap_when_full"}), 6):
            create = run_tool("create_game", {"organizer_id": "guest", "organizer_name": "顾客", "known_players": [],
                "requirement": terms | opening}, step)
            assert not create.called and not create.allowed and "近似开场" in create.error
        assert set(store.games) == set(games.values()) and not store.room_reservations and len(review.calls) == 1
    store._connection.close()


def test_approximate_han_minutes_allow_a_read_without_authorizing_creation():
    """近似到店允许查桌。分钟没有核实依据时，不能借无精确词面跳过建桌核对。"""
    at = SENT.replace(hour=18, minute=55)
    store = InMemoryAgentStore()
    message = _incoming("一块底的还有位吗？我七点十几分能到，没现成的帮我凑一桌", sent=at)
    review = ReviewClient([])
    authorization = ReviewClient([{"scope": "create", "customer_ref": 1, "shop_offer_ref": None}])
    service = ToolExecutionService(store=store, tool_gateway=ToolGateway(store), trace_recorder=InMemoryTraceRecorder(),
                                   table_availability_client=review, creation_authorization_client=authorization)

    def submit(tool, step):
        action = AgentAction(goal="核对能赶上的桌", objective_status="needs_tool", reasoning_summary="核对时间",
                             tool_calls=[tool])
        return service.execute_tool_calls(action, message=message, trace_id=message.message_id,
            previous_step_tool_results=[], step_index=step, run_id="approximate-minutes",
            run_version=store.conversation_version(message.conversation_id)).tool_results[0]

    with using_clock(lambda: at):
        _activate(store, started=at)
        _persist(store, message)
        search = submit(ToolCall("search_current_games", {"requirement": {"stake": "1"}, "lookup_scope": "personal"}), 1)
        create = submit(ToolCall("create_game", {"organizer_id": "guest", "organizer_name": "顾客", "known_players": [],
            "requirement": {"start_time_kind": "asap_when_full", "stake": "1"}}), 2)

    assert search.allowed and search.called
    assert not create.allowed and not create.called and "钟点用途尚未核实" in create.error
    assert create.result["table_availability_evidence"] is None
    assert not review.calls and not authorization.calls and not store.games


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
