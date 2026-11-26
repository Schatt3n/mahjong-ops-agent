"""时间一致性保留来源日期与时区，钟点不能凭壁钟或词法候选冒充绝对瞬时。"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime

import pytest

from mahjong_agent_runtime import (
    AgentAction, InMemoryAgentStore, InMemoryTraceRecorder, SQLiteAgentStore,
    ToolCall, ToolGateway, ToolResult, UserMessage,
)
from mahjong_agent_runtime.services import ToolExecutionService
from mahjong_agent_runtime.models import using_clock
from mahjong_agent_runtime.tool_consistency import (
    executed_requirement_contract, validate_explicit_task_fact_consistency, validate_table_availability,
    validate_tool_call_consistency,
)


ANCHOR = "2026-10-30T18:30:00+08:00"
UTC_EQUIVALENT = "2026-10-30T10:30:00Z"


def test_new_table_preserves_duration_cap_from_executed_lookup():
    """最多打三小时要沿用为新局上限，不能漏掉或换成三小时固定约定。"""
    lookup = ToolResult("search_current_games", True, True, result={"requirement": {"max_duration_hours": 3}})
    fixed = ToolCall("create_game", {"requirement": {"duration_hours": 3}})
    capped = ToolCall("create_game", {"requirement": {"max_duration_hours": 3}})
    assert "max_duration_hours" in validate_tool_call_consistency(fixed, [lookup])
    assert validate_tool_call_consistency(capped, [lookup]) is None
    assert executed_requirement_contract([lookup])["preserve_requirement"] == {"max_duration_hours": 3}


def test_flexible_game_cannot_start_before_future_lookup_window():
    """即使局有效期足够覆盖未来窗口，无开场下界的人齐开也可能在顾客到达前启动。"""
    window = {"start_at": "2026-11-26T09:30:00+08:00", "end_at": "2026-11-26T18:00:00+08:00"}
    lookup = ToolResult("search_current_games", True, True, result={"requirement": {"start_time_range": window}})
    call = ToolCall("create_game", {"requirement": {"start_time_kind": "asap_when_full"}})
    with using_clock(lambda: datetime.fromisoformat("2026-11-26T09:07:00+08:00")):
        error = validate_tool_call_consistency(call, [lookup])
        assert "no enforceable earliest start" in error and window["start_at"] in error
    with using_clock(lambda: datetime.fromisoformat(window["start_at"])):
        assert validate_tool_call_consistency(call, [lookup]) is None


def test_verified_arrival_window_and_stake_do_not_authorize_fixed_opening():
    """核实同一顾客的时段后校验工具参数，授权找桌与承诺固定时间分开。"""
    start = "2026-11-26T09:30:00+08:00"
    window = {"start_at": start, "end_at": "2026-11-26T18:00:00+08:00"}
    stakes = {"min": 2, "max": 3}
    evidence = {"lookup_scope": "personal", "opening_window": window, "accepted_stake_range": stakes,
                "known_cap_score": None, "has_fixed_opening_authority": False}
    def lookup(requirement):
        return validate_table_availability(ToolCall("search_current_games", {"requirement": requirement}), evidence)

    assert "封顶" in lookup({"stake": 2, "cap_score": 3, "start_time_range": window})
    assert "封顶" in lookup({"stake_range": stakes, "cap_score": 3, "start_time_range": window})
    assert "时段" in lookup({"stake_range": stakes, "planned_start_at": start, "start_time_range": window})
    assert "时段" in lookup({"stake_range": stakes, "planned_start_at": start})
    assert lookup({"stake_range": stakes, "start_time_range": window}) is None
    confirmed = evidence | {"has_fixed_opening_authority": True, "opening_window": {"start_at": start, "end_at": start}}
    assert validate_table_availability(ToolCall("search_current_games", {"requirement": {
        "stake_range": stakes, "planned_start_at": start, "start_time_kind": "scheduled",
    }}), confirmed) is None
    create = ToolCall("create_game", {"requirement": {"stake_range": stakes, "planned_start_at": start,
                                                     "start_time_kind": "scheduled"}})
    with using_clock(lambda: datetime.fromisoformat("2026-11-26T09:07:00+08:00")):
        assert "还没选定" in validate_table_availability(create, evidence)
        assert validate_table_availability(create, evidence | {"has_fixed_opening_authority": True}) is None
        assert validate_table_availability(create, evidence | {"lookup_scope": "specific_table"}) is None


def test_tool_service_enforces_verified_terms_before_store_read_and_write(monkeypatch):
    """工具正常入口把独立核实的档位用于第一次实际查局，错误提案及未选开场的写均无业务动作。"""
    import mahjong_agent_runtime.services.tool_service as tool_service

    at = datetime.fromisoformat("2026-11-26T09:07:00+08:00")
    window = {"start_at": "2026-11-26T09:30:00+08:00", "end_at": "2026-11-26T18:00:00+08:00"}
    stakes = {"min": 2, "max": 3}
    evidence = {"lookup_scope": "personal", "opening_window": window, "accepted_stake_range": stakes,
                "known_cap_score": None, "has_fixed_opening_authority": False}
    seen = []
    def independent_check(_store, _client, message, **_):
        seen.append(message.message_id)
        return evidence

    monkeypatch.setattr(tool_service, "check_table_availability", independent_check)
    with using_clock(lambda: at):
        store = InMemoryAgentStore()
        message = UserMessage("arrival-chat", "guest", "顾客", "帮我凑一桌，我九点半前能到，2块3块都行",
                              message_id="arrival-source", sent_at=at)
        store.append_user_turn(message, "arrival-turn")
        service = ToolExecutionService(store=store, tool_gateway=ToolGateway(store),
            trace_recorder=InMemoryTraceRecorder(), table_availability_client=object())

        def submit(call, step):
            action = AgentAction(goal="找能赶上的桌", objective_status="needs_tool", reasoning_summary="先核实", tool_calls=[call])
            return service.execute_tool_calls(action, message=message, trace_id="arrival-turn",
                previous_step_tool_results=[], step_index=step, run_id="arrival-run",
                run_version=store.conversation_version(message.conversation_id)).tool_results[0]

        wrong = submit(ToolCall("search_current_games", {"requirement": {"stake": 2, "cap_score": 3}}), 1)
        assert not wrong.called and not wrong.allowed and wrong.result["table_availability_evidence"] == evidence
        corrected = submit(ToolCall("search_current_games", {"requirement": {"stake_range": stakes, "start_time_range": window}}), 2)
        assert corrected.called and corrected.allowed and corrected.result["requirement"]["start_time_range"] == window
        unsafe = submit(ToolCall("create_game", {"organizer_id": "guest", "organizer_name": "顾客", "known_players": [], "requirement": {
            "stake_range": stakes, "planned_start_at": window["start_at"], "start_time_kind": "scheduled",
        }}), 3)
        assert not unsafe.called and unsafe.result["validation_error"]["code"] == "verified_table_availability_conflict"
        assert seen == [message.message_id] * 3
        assert not store.games


def explicit_error(reference, proposed):
    call = ToolCall(name="search_current_games", arguments={"requirement": proposed})
    original = deepcopy(call.arguments)
    error, _ = validate_explicit_task_fact_consistency(call, {"explicit_task_facts": {"facts": reference}})
    assert call.arguments == original
    return error


@pytest.mark.parametrize("reference,proposed", [
    ({"start_time": "18:30", "planned_start_at": ANCHOR}, {"start_time": ANCHOR}),
    ({"start_time": "18:30", "planned_start_at": ANCHOR}, {"start_time": UTC_EQUIVALENT}),
    ({"start_time": ANCHOR, "planned_start_at": ANCHOR}, {"start_time": "18:30", "planned_start_at": UTC_EQUIVALENT}),
    ({"start_time": ANCHOR}, {"start_time": UTC_EQUIVALENT}),
    ({"planned_start_at": ANCHOR}, {"planned_start_at": UTC_EQUIVALENT}),
    ({"start_time": "2026-10-30T00:30:00+08:00"}, {"start_time": "2026-10-29T16:30:00Z"}),
    ({"start_time": "00:30", "planned_start_at": "2026-10-30T00:30:00+08:00"}, {"start_time": "2026-10-29T16:30:00Z"}),
    ({"start_time": "18.30"}, {"start_time": "18:30"}),
])
def test_time_guard_accepts_only_proven_representation_equivalence(reference, proposed):
    assert explicit_error(reference, proposed) is None


@pytest.mark.parametrize("reference,proposed", [
    ({"start_time": "18:30"}, {"start_time": ANCHOR}),
    ({"start_time": "18:30", "planned_start_at": "2026-10-30T18:30:00"}, {"start_time": ANCHOR}),
    ({"start_time": "18:30", "planned_start_at": ANCHOR}, {"start_time": "2026-10-31T18:30:00+08:00"}),
    ({"start_time": "18:30", "planned_start_at": ANCHOR}, {"start_time": "2026-10-30T18:30:00+09:00"}),
    ({"start_time": "18:30", "planned_start_at": ANCHOR}, {"start_time": "2026-10-30T10:31:00Z"}),
    ({"start_time": "18:30", "planned_start_at": ANCHOR}, {"start_time": "18:30"}),
    ({"start_time": ANCHOR}, {"start_time": "18:30"}),
    ({"start_time": ANCHOR, "planned_start_at": ANCHOR}, {"start_time": "18:30"}),
    ({"start_time": "18:30", "planned_start_at": ANCHOR}, {"start_time": ANCHOR, "planned_start_at": "2026-10-31T18:30:00+08:00"}),
    ({"start_time": "18:30", "planned_start_at": ANCHOR}, {"start_time": ANCHOR, "planned_start_at": "2026-10-30T18:30:00"}),
    ({"start_time": "18:30", "planned_start_at": "2026-10-30T17:30:00+08:00"}, {"start_time": "2026-10-30T17:30:00+08:00"}),
    ({"start_time": ANCHOR}, {"start_time": "2026-10-31T18:30:00+08:00"}),
    ({"planned_start_at": ANCHOR}, {"planned_start_at": "2026-10-30T18:30:00+09:00"}),
    ({"planned_start_at": ANCHOR}, {"planned_start_at": "2026-10-30T18:30:00"}),
])
def test_time_guard_rejects_missing_anchor_day_offset_and_internal_conflicts(reference, proposed):
    assert explicit_error(reference, proposed)


@pytest.mark.parametrize("proposed,allowed", [
    ({"start_time": "18:30", "planned_start_at": UTC_EQUIVALENT}, True),
    ({"start_time": UTC_EQUIVALENT}, True),
    ({"start_time": "18:30"}, False),
    ({"start_time": "18:30", "planned_start_at": "2026-10-31T18:30:00+08:00"}, False),
    ({"start_time": "18:30", "planned_start_at": "2026-10-30T18:30:00+09:00"}, False),
    ({"start_time": "18:30", "planned_start_at": "2026-10-30T18:30:00"}, False),
])
def test_read_to_create_retains_absolute_date_and_compares_utc_instant(proposed, allowed):
    read = ToolResult(name="search_current_games", called=True, allowed=True,
                      result={"requirement": {"start_time": "18:30", "planned_start_at": ANCHOR}})
    call = ToolCall(name="create_game", arguments={"requirement": proposed})
    before = deepcopy(call.arguments)
    assert (validate_tool_call_consistency(call, [read]) is None) is allowed
    assert call.arguments == before


def test_failed_read_cannot_establish_an_absolute_time_anchor():
    failed_read = ToolResult(name="search_current_games", called=True, allowed=True, error="query failed",
                            result={"requirement": {"start_time": "18:30", "planned_start_at": ANCHOR}})
    call = ToolCall(name="create_game", arguments={"requirement": {"start_time": "19:30"}})
    assert validate_tool_call_consistency(call, [failed_read]) is None


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_flexible_creation_must_not_expire_before_the_successfully_searched_times(backend, tmp_path):
    """同一轮改做人齐开也不能在前序查局的可用时段前到期；当天固定可及时建局。"""
    store = InMemoryAgentStore() if backend == "memory" else SQLiteAgentStore(tmp_path / "query-window.sqlite3")
    stamp = datetime.fromisoformat("2026-11-26T11:07:00+08:00")
    evening = {"start_at": "2026-11-26T19:15:00+08:00", "end_at": "2026-11-26T23:59:00+08:00"}
    try:
        gateway = ToolGateway(store)
        message = UserMessage("time-window", "customer", "顾客", "帮我凑一桌吧 我七点一刻左右能到",
                              message_id="source-window", sent_at=stamp)
        service = ToolExecutionService(store=store, tool_gateway=gateway, trace_recorder=InMemoryTraceRecorder())
        with using_clock(lambda: stamp):
            lookup = gateway.execute(ToolCall("search_current_games", {"requirement": {"start_time_range": evening}}),
                trace_id="window", step_index=1, conversation_id=message.conversation_id,
                sender_id=message.sender_id, sender_name=message.sender_name)
            assert lookup.called and lookup.allowed and executed_requirement_contract([lookup])["searched_opening_window"] == evening
            create = ToolCall("create_game", {"organizer_id": message.sender_id, "organizer_name": message.sender_name,
                                              "requirement": {"start_time_kind": "asap_when_full"}})
            result, = service.execute_tool_calls(AgentAction("组桌", "needs_tool", "查局后执行", tool_calls=[create]),
                message=message, trace_id="window", previous_step_tool_results=[lookup], step_index=2,
                run_id="window-run", run_version=store.conversation_version(message.conversation_id), context_payload={}).tool_results
            assert not result.called and not result.allowed and not result.state_transitions
            assert result.result["validation_error"]["code"] == "tool_requirement_consistency_rejected"
            assert "2026-11-26T15:07:00+08:00" in result.error and evening["start_at"] in result.error
            assert not store.games and not store.invite_drafts
            corrected = ToolCall("create_game", {**create.arguments, "requirement": {
                "start_time_kind": "scheduled", "planned_start_at": "2026-11-26T19:30:00+08:00"}})
            assert validate_tool_call_consistency(corrected, [lookup]) is None
            narrowed = {"start_at": "2026-11-26T11:07:00+08:00", "end_at": "2026-11-26T14:00:00+08:00"}
            valid_lookup = ToolResult("search_current_games", True, True, result={"requirement": {"start_time_range": narrowed}})
            assert validate_tool_call_consistency(create, [lookup, valid_lookup]) is None
    finally:
        if backend == "sqlite":
            store._connection.close()


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("anchor_source", ["current", "condition", "missing", "other_sender", "candidate_only"])
def test_service_only_uses_source_bound_current_anchor(backend, anchor_source, tmp_path):
    store = InMemoryAgentStore() if backend == "memory" else SQLiteAgentStore(tmp_path / "time-anchor.sqlite3")
    try:
        trace = InMemoryTraceRecorder()
        message = UserMessage(conversation_id="time-source", sender_id="customer", sender_name="客户",
                              text="2026年10月30日18:30，东八区", message_id="time-source-message")
        store.append_user_turn(message, "time-source")
        store.append_user_turn(UserMessage(conversation_id=message.conversation_id, sender_id="other", sender_name="其他客户",
                                         text="2026年10月30日18:30，东八区", message_id="other-message"), "other-trace")
        assertions = [{"field": "start_time", "value": "18:30", "role": "current", "source_message_id": message.message_id}]
        if anchor_source != "candidate_only":
            assertions.append({"field": "planned_start_at", "value": ANCHOR,
                "role": "condition" if anchor_source == "condition" else "current",
                "source_message_id": "absent" if anchor_source == "missing" else "other-message" if anchor_source == "other_sender" else message.message_id})
        call = ToolCall(name="search_current_games", arguments={"requirement": {"start_time": UTC_EQUIVALENT}})
        service = ToolExecutionService(store=store, tool_gateway=ToolGateway(store=store), trace_recorder=trace)
        processed = service.execute_tool_calls(
            AgentAction(goal="按客户时间查局", objective_status="needs_tool", reasoning_summary="测试来源锚", tool_calls=[call],
                        objective_state={"fact_assertions": assertions}),
            message=message, trace_id="time-source-test", previous_step_tool_results=[], step_index=1,
            run_id="time-source-run", run_version=store.conversation_version(message.conversation_id),
            context_payload={"explicit_task_facts": {"candidates": {"planned_start_at": ANCHOR}}},
        )
        result = processed.tool_results[0]
        assert result.called is (anchor_source == "current")
        assert result.allowed is (anchor_source == "current")
        assert (result.error is None) is (anchor_source == "current")
        assert store.games == {} and not store.invite_drafts
    finally:
        if backend == "sqlite":
            store._connection.close()
