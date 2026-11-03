"""房态缺省窗口由可信时钟计算；双 Store 查询不替客户预约，也不覆盖半边显式时间。"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta

import pytest

from mahjong_agent_runtime import InMemoryAgentStore, SQLiteAgentStore, ToolCall, ToolGateway
from mahjong_agent_runtime.domains.context_builders.tool_results import tool_result_for_context
from mahjong_agent_runtime.domains.tools import search_tools


STAMP = datetime.fromisoformat("2031-11-02T22:45:30.123456+08:00")


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    """每例独立运行双实现，不接触已有运行库或线上数据。"""
    result = InMemoryAgentStore() if request.param == "memory" else SQLiteAgentStore(tmp_path / "room-window.sqlite3")
    yield result
    if isinstance(result, SQLiteAgentStore):
        result._connection.close()


def query(gateway, arguments, *, message_id="room-window-query"):
    """经过真实网关验证、权限与幂等边界，不能直接调用 handler 冒充合法工具。"""
    return gateway.execute(ToolCall("check_room_availability", deepcopy(arguments)),
        trace_id="room-window-trace", conversation_id="room-window-conversation", sender_id="room-window-customer",
        sender_name="测试客户", step_index=1, source_message_id=message_id)


def business_snapshot(store):
    """只比较业务状态；网关持久化查询审计和幂等结果不属于预约或招募副作用。"""
    return deepcopy({
        "games": [item.to_dict() for item in store.games.values()],
        "reservations": [item.to_dict() for item in store.room_reservations.values()],
        "invites": [item.to_dict() for item in store.invite_drafts.values()],
        "outbound": [item.to_dict() for item in store.outbound_message_drafts.values()],
        "scheduled": [item.to_dict() for item in store.scheduled_tasks.values()],
        "transitions": [item.to_dict() for item in store.transitions],
    })


@pytest.mark.parametrize("inventory", ["available", "occupied", "unconfigured"])
def test_default_window_uses_one_trusted_clock_read_and_preserves_inventory_truth(store, monkeypatch, inventory):
    """同一可信 now 跨午夜加四小时，库存未知、无房、有房不能混为一谈。"""
    calls = []

    def clock():
        calls.append(True)
        return STAMP + timedelta(minutes=len(calls) - 1)

    monkeypatch.setattr(search_tools, "now", clock, raising=False)
    store.configure_rooms([] if inventory == "unconfigured" else ["room-private-id"])
    if inventory == "occupied":
        store.reserve_room(conversation_id="prior-booking", game_id=None, room_id="room-private-id",
            start_at=STAMP - timedelta(minutes=1), end_at=STAMP + timedelta(hours=5), trace_id="prior-booking")
    before = business_snapshot(store)
    result = query(ToolGateway(store), {})
    assert result.called and result.allowed and result.error is None
    assert calls == [True]
    assert result.result["start_at"] == STAMP.isoformat()
    assert result.result["end_at"] == "2031-11-03T02:45:30.123456+08:00"
    assert result.result["configured"] is (inventory != "unconfigured")
    assert result.result["room_count"] == int(inventory != "unconfigured")
    assert result.result["available_count"] == int(inventory == "available")
    policy = result.result["next_step_policy"]
    assert policy["query_window_source"] == "backend_now_plus_4h"
    assert policy["query_window_default_hours"] == 4
    assert policy["may_state_room_available"] is (inventory == "available")
    assert result.state_transitions == []
    assert business_snapshot(store) == before
    projected = tool_result_for_context(result)["result"]
    assert projected["start_at"] == result.result["start_at"]
    assert projected["end_at"] == result.result["end_at"]
    assert projected["next_step_policy"]["query_window_source"] == "backend_now_plus_4h"


def test_default_window_keeps_existing_half_open_reservation_boundaries(store, monkeypatch):
    """恰在起点结束或终点开始的预约不冲突；窗口内任何重叠仍占用房间。"""
    monkeypatch.setattr(search_tools, "now", lambda: STAMP, raising=False)
    end = STAMP + timedelta(hours=4)
    intervals = {
        "ends_at_start": (STAMP - timedelta(hours=1), STAMP),
        "starts_at_end": (end, end + timedelta(hours=1)),
        "overlaps_start": (STAMP - timedelta(hours=1), STAMP + timedelta(microseconds=1)),
        "overlaps_end": (end - timedelta(microseconds=1), end + timedelta(hours=1)),
        "covers_window": (STAMP - timedelta(hours=1), end + timedelta(hours=1)),
        "inside_window": (STAMP + timedelta(hours=1), STAMP + timedelta(hours=2)),
    }
    store.configure_rooms(list(intervals))
    for room_id, (start_at, end_at) in intervals.items():
        store.reserve_room(conversation_id="prior-booking", game_id=None, room_id=room_id,
            start_at=start_at, end_at=end_at, trace_id="prior-booking")
    before = business_snapshot(store)
    result = query(ToolGateway(store), {})
    assert result.called and result.allowed and not result.error
    assert set(result.result["available_room_ids"]) == {"ends_at_start", "starts_at_end"}
    assert set(result.result["occupied_room_ids"]) == set(intervals) - {"ends_at_start", "starts_at_end"}
    assert result.result["available_count"] == 2
    assert business_snapshot(store) == before


def test_complete_explicit_interval_wins_without_reading_default_clock(store, monkeypatch):
    """完整显式时段交原 Store 查询，不用当前时间覆盖或截断客户已给区间。"""
    def forbidden_clock():
        raise AssertionError("explicit interval must not read the default clock")

    monkeypatch.setattr(search_tools, "now", forbidden_clock, raising=False)
    store.configure_rooms(["explicit-room"])
    arguments = {"start_at": "2031-11-04T18:00:00+08:00", "end_at": "2031-11-05T01:30:00+08:00"}
    before = business_snapshot(store)
    result = query(ToolGateway(store), arguments)
    assert result.called and result.allowed and not result.error
    assert result.result["start_at"] == arguments["start_at"]
    assert result.result["end_at"] == arguments["end_at"]
    assert result.result["next_step_policy"]["query_window_source"] == "explicit_interval"
    assert "query_window_default_hours" not in result.result["next_step_policy"]
    assert business_snapshot(store) == before


@pytest.mark.parametrize("arguments", [
    {"start_at": STAMP.isoformat()}, {"end_at": STAMP.isoformat()},
    {"start_at": None, "end_at": None}, {"start_at": "", "end_at": ""},
    {"start_at": STAMP.isoformat(), "end_at": ""},
    {"start_at": "", "end_at": (STAMP + timedelta(hours=1)).isoformat()},
    {"start_at": True, "end_at": False}, {"now": STAMP.isoformat()},
])
def test_partial_empty_or_forged_clock_arguments_stay_schema_rejections(store, monkeypatch, arguments):
    """只有真正省略两端才默认；半边、空值和模型伪造时钟不能触发查询。"""
    calls = []
    monkeypatch.setattr(search_tools, "now", lambda: calls.append(True) or STAMP, raising=False)
    before = business_snapshot(store)
    result = query(ToolGateway(store), arguments)
    assert not result.called and not result.allowed and result.error
    assert result.result == {} and result.state_transitions == []
    assert calls == []
    assert business_snapshot(store) == before


@pytest.mark.parametrize("arguments", [
    {"start_at": "not-a-date", "end_at": "not-a-date"},
    {"start_at": STAMP.isoformat(), "end_at": STAMP.isoformat()},
    {"start_at": STAMP.isoformat(), "end_at": (STAMP - timedelta(hours=1)).isoformat()},
])
def test_invalid_explicit_interval_is_not_replaced_by_default(store, monkeypatch, arguments):
    """完整但非法的时段沿用领域拒绝，不退回四小时窗口伪装查询成功。"""
    calls = []
    monkeypatch.setattr(search_tools, "now", lambda: calls.append(True) or STAMP, raising=False)
    before = business_snapshot(store)
    result = query(ToolGateway(store), arguments)
    assert not result.called and not result.allowed and result.error
    assert calls == [] and business_snapshot(store) == before


def test_same_message_replay_retains_executed_window_but_new_query_uses_new_now(store, monkeypatch):
    """幂等重放保留已执行区间；不同消息的合法读取才重新取当前时钟。"""
    calls = []

    def clock():
        value = STAMP + timedelta(hours=len(calls))
        calls.append(value)
        return value

    monkeypatch.setattr(search_tools, "now", clock, raising=False)
    store.configure_rooms(["room-a"])
    gateway = ToolGateway(store)
    first = query(gateway, {})
    replay = query(gateway, {})
    latest = query(gateway, {}, message_id="new-room-query")
    assert first.called and first.allowed and not first.error
    assert replay.deduplicated and replay.result == first.result
    assert latest.called and latest.allowed and not latest.error and not latest.deduplicated
    assert calls == [STAMP, STAMP + timedelta(hours=1)]
    assert latest.result["start_at"] == (STAMP + timedelta(hours=1)).isoformat()
    assert latest.result["end_at"] == (STAMP + timedelta(hours=5)).isoformat()
    assert not store.room_reservations and not store.games and not store.scheduled_tasks


def test_default_read_still_obeys_tool_permissions_before_reading_clock(store, monkeypatch):
    """可省时间不扩大工具权限；部署拒绝时连默认时钟与库存都不读取。"""
    calls = []
    monkeypatch.setattr(search_tools, "now", lambda: calls.append(True) or STAMP, raising=False)
    gateway = ToolGateway(store, allowed_execution_modes=set())
    result = query(gateway, {})
    assert not result.called and not result.allowed and result.error
    assert calls == [] and not store.room_reservations


def test_sqlite_restart_preserves_default_query_window_and_read_only_result(tmp_path, monkeypatch):
    """重启后重放同一查询读取持久结果，不以新时钟冒充旧窗口，也不新增预约。"""
    monkeypatch.setattr(search_tools, "now", lambda: STAMP, raising=False)
    path = tmp_path / "room-restart.sqlite3"
    first_store = SQLiteAgentStore(path)
    first_store.configure_rooms(["room-a"])
    first = query(ToolGateway(first_store), {})
    assert first.called and first.allowed and not first.error
    first_store._connection.close()

    def forbidden_clock():
        raise AssertionError("persisted query replay must retain the original clock window")

    monkeypatch.setattr(search_tools, "now", forbidden_clock)
    reopened = SQLiteAgentStore(path)
    try:
        replay = query(ToolGateway(reopened), {})
        assert replay.deduplicated and replay.result == first.result
        assert not reopened.room_reservations and not reopened.games and not reopened.scheduled_tasks
    finally:
        reopened._connection.close()
