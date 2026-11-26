"""邻近开场只作为带差值的只读备选，区间硬条件仍独立生效。"""

from copy import deepcopy
from datetime import datetime

import pytest

from mahjong_agent_runtime import CustomerProfile, InMemoryAgentStore, SQLiteAgentStore, ToolCall, ToolGateway
from mahjong_agent_runtime.models import using_clock


def _at(clock: str) -> datetime:
    """使用固定的带时区时间，避免实时时钟使邻近场次变成已开场。"""
    return datetime.fromisoformat(f"2026-11-25T{clock}:00+08:00")


@pytest.fixture(params=["memory", "sqlite_reopened"])
def nearby_inventory(request, tmp_path):
    """准备一场正选、区间两边的备选、越界和容易误荐的负例。"""
    db_path = tmp_path / "nearby.sqlite3"
    store = InMemoryAgentStore() if request.param == "memory" else SQLiteAgentStore(db_path)
    before = _at("18:46")
    inventory = {
        "before": ("18:55", {}),
        "within": ("20:37", {}),
        "after": ("21:04", {}),
        "beyond_early": ("18:54", {}),
        "beyond_late": ("21:05", {}),
        "one_seat": ("21:01", {"host_seats": 3}),
        "different_stake": ("21:02", {"stake": "2"}),
        "different_smoke": ("18:58", {"smoke_preference": "smoke_ok"}),
        "different_duration": ("18:57", {"duration_hours": 5}),
        "different_game": ("21:03", {"game_type": "sichuan_mahjong"}),
    }
    ids = {}
    with using_clock(lambda: before):
        store.upsert_customer(CustomerProfile(customer_id="requester", display_name="客户",
            preferred_games=["hangzhou_mahjong"], preferred_stakes=["1", "2"]))
        for label, (clock, adjustments) in inventory.items():
            terms = {"game_type": "hangzhou_mahjong", "stake": "1", "smoke_preference": "no_smoking",
                     "duration_hours": 3, "planned_start_at": _at(clock).isoformat()}
            overrides = dict(adjustments)
            host_seats = overrides.pop("host_seats", 1)
            host = f"host-{label}"
            game, _ = store.create_game(
                conversation_id=f"nearby-{label}", organizer_id=host, organizer_name="牌友",
                requirement=terms | overrides,
                known_players=[{"customer_id": host, "display_name": "牌友", "seat_count": host_seats}],
                trace_id=f"setup-nearby-{label}",
            )
            ids[label] = game.game_id
        if request.param == "sqlite_reopened":
            store._connection.close()
            store = SQLiteAgentStore(db_path)
        try:
            yield store, ids
        finally:
            if isinstance(store, SQLiteAgentStore):
                store._connection.close()


def _query(store, *, start="19:00", end="21:00", limit=8, include_nearby=None):
    """经过正常工具网关查询，同时核对没有动过牌局或产生外发。"""
    requirement = {"game_type": "hangzhou_mahjong", "stake": "1", "smoke_preference": "no_smoking",
                   "duration_hours": 3, "requesting_party": {"contact_id": "requester", "seat_count": 2}}
    if start is not None:
        requirement["start_time_range"] = {"start_at": _at(start).isoformat(), "end_at": _at(end).isoformat()}
    arguments = {"requirement": requirement, "limit": limit}
    if include_nearby is not None:
        arguments["include_nearby"] = include_nearby
    submitted = deepcopy(arguments)
    before = {key: game.to_dict() for key, game in store.games.items()}
    result = ToolGateway(store).execute(
        ToolCall(name="search_current_games", arguments=arguments), trace_id="nearby-search",
        conversation_id="request-chat", sender_id="requester", sender_name="客户", step_index=1,
    )
    assert result.called and result.allowed and not result.error, result
    assert arguments == submitted
    assert {key: game.to_dict() for key, game in store.games.items()} == before
    assert not store.invite_drafts and not store.outbound_message_drafts and not store.room_reservations
    return result.result


def _nearby(result):
    """按牌局索引备选，在断言中比较每个差值与相对于区间的方向。"""
    assert all(row["match_kind"] == "nearby_start_option" and row["requires_timing_confirmation"] is True
               for row in result["nearby_options"])
    return {row["game"]["game_id"]: (row["relative_to_window"], row["offset_minutes"])
            for row in result["nearby_options"]}


def test_range_keeps_exact_match_and_labels_only_compatible_nearby_options(nearby_inventory):
    """两个更近但冲突的条件和一席牌桌都不能抢占可用的邻近位置。"""
    store, ids = nearby_inventory
    payload = _query(store, limit=1)
    assert [(row["game"]["game_id"], row["match_kind"]) for row in payload["matches"]] == [(ids["within"], "exact")]
    assert payload["alternatives"] == []
    assert _nearby(payload) == {ids["before"]: ("before", 5), ids["after"]: ("after", 4)}

    disabled = _query(store, limit=1, include_nearby=False)
    assert "nearby_options" not in disabled
    assert disabled["matches"] == payload["matches"]
    assert len(_query(store, start="18:52", end="21:05", limit=1)["matches"]) == 1
    assert "nearby_options" not in _query(store, start=None)


def test_nearby_window_includes_ten_minutes_and_excludes_past_starts(nearby_inventory):
    """前后边界都按距离计算，已开场的精确条件牌桌不能再当作备选。"""
    store, ids = nearby_inventory
    limits = _query(store, start="19:05", end="20:54")
    assert _nearby(limits) == {ids["before"]: ("before", 10), ids["after"]: ("after", 10)}
    assert {row["game"]["game_id"] for row in limits["matches"]} == {ids["within"]}
    # 将两个边界各向内移一分钟，原备选都差十一分钟，不得被较近的冲突牌桌替换。
    assert _nearby(_query(store, start="19:06", end="20:53")) == {}

    with using_clock(lambda: _at("18:57")):
        upcoming = _query(store)
    assert _nearby(upcoming) == {ids["after"]: ("after", 4)}


def test_nearby_without_an_exact_match_still_reports_no_exact_match(nearby_inventory):
    """附近可选桌不升级为命中，回复合同也保持没有区间内可执行匹配。"""
    store, ids = nearby_inventory
    payload = _query(store, start="20:42", end="20:57")
    assert not payload["matches"]
    assert _nearby(payload) == {ids["within"]: ("before", 5), ids["after"]: ("after", 7)}
    semantics = payload["customer_reply_contract"]["search_result_semantics"]
    assert semantics["status"] == "no_actionable_match" and semantics["actionable_match_count"] == 0


def test_closest_option_can_be_beyond_first_five_scored_candidates(nearby_inventory):
    """同等条件的局超过五桌，仍须比较临近窗口内的全部可入局开场。"""
    store, _ = nearby_inventory
    for index in range(6):
        host = f"crowded-{index}"
        game, _ = store.create_game(
            conversation_id=host, organizer_id=host, organizer_name="牌友",
            requirement={"game_type": "hangzhou_mahjong", "stake": "1", "smoke_preference": "no_smoking",
                         "duration_hours": 3, "planned_start_at": _at("18:59" if index == 5 else "18:51").isoformat()},
            known_players=[{"customer_id": host, "display_name": "牌友", "seat_count": 1}], trace_id=host,
        )
    options = _nearby(_query(store, limit=1))
    assert options[game.game_id] == ("before", 1)
