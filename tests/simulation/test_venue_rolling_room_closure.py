"""通过正式沙盒房态验证限时检修、四小时实际占房和同房复订。"""

from datetime import datetime, timedelta

import pytest

from mahjong_agent_runtime import SQLiteAgentStore
from mahjong_agent_runtime.models import GameStatus, using_clock

from .venue_rolling_world import PhysicalActorEvent, VenueRollingWorld


_ROOM = "东一包间"
_ALTERNATE = "东二包间"
_ATTENDEES = ("closure-a", "closure-b", "closure-c", "closure-d")
_REOPEN = datetime.fromisoformat("2026-11-25T18:00:00+08:00")
_CLOSURE_START = _REOPEN - timedelta(hours=1)
_FINISH = _REOPEN + timedelta(hours=4)
_REBOOK_END = _FINISH + timedelta(hours=1)


def _confirmed_game(store):
    """先用正式组局和四名具名成员的报名事务建立真实库存。"""
    organizer, *others = _ATTENDEES
    game, _ = store.create_game(
        conversation_id="closure-organizer-private", organizer_id=organizer, organizer_name="组织者",
        known_players=[], trace_id="closure-setup-game",
        requirement={"start_time_kind": "scheduled", "planned_start_at": _REOPEN.isoformat(),
                     "smoke_preference": "no_smoking", "duration_kind": "fixed", "duration_hours": 4},
    )
    for actor_id in others:
        game, _ = store.join_game(game_id=game.game_id, customer_id=actor_id, display_name=actor_id,
                                  trace_id="closure-setup-confirmation")
    return game


def test_limited_maintenance_hold_then_occupied_room_and_rebooking_do_not_overlap(tmp_path):
    """检修仅按时段阻挡房间，实际局到点须有离场证据才释放并允许同房复订。"""
    store = SQLiteAgentStore(tmp_path / "agent.sqlite3")
    with using_clock(lambda: _CLOSURE_START):
        store.configure_rooms([_ROOM, _ALTERNATE])
        maintenance, _ = store.reserve_room(
            conversation_id="shop-inventory", game_id=None, room_id=_ROOM, start_at=_CLOSURE_START,
            end_at=_REOPEN, trace_id="temporary-maintenance-hold",
        )
        game = _confirmed_game(store)
        inventory = store.search_room_availability(start_at=_CLOSURE_START, end_at=_REOPEN)
        assert inventory["room_count"] == 2
        assert inventory["available_room_ids"] == [_ALTERNATE]
        assert inventory["occupied_room_ids"] == [_ROOM]
        assert maintenance.status == "held" and maintenance.game_id is None
        assert game.status == GameStatus.READY and not game.is_playing
        # 只要跨过检修的最后一秒，预约也不能借用其后的空闲时段入场。
        with pytest.raises(ValueError, match="room is unavailable"):
            store.reserve_room(conversation_id=game.conversation_id, game_id=game.game_id, room_id=_ROOM,
                               start_at=_REOPEN - timedelta(seconds=1), end_at=_FINISH,
                               trace_id="rejected-maintenance-overlap")
        assert len(store.room_reservations) == 1
        assert _ROOM in store.search_room_availability(start_at=_REOPEN, end_at=_FINISH)["available_room_ids"]

    with using_clock(lambda: _REOPEN):
        booking, _ = store.reserve_room(
            conversation_id=game.conversation_id, game_id=game.game_id, room_id=_ROOM,
            start_at=_REOPEN, end_at=_FINISH, trace_id="book-room-at-maintenance-end",
        )
    world = VenueRollingWorld(store, tmp_path)
    arrivals = [PhysicalActorEvent(f"arrival:{actor}", game.game_id, actor, "arrive", _REOPEN, "controlled")
                for actor in _ATTENDEES]
    assert world.schedule(game.game_id, start_at=_REOPEN, arrivals=arrivals)["status"] == "scheduled"
    assert world.tick(_REOPEN)["actually_started"] == 1
    game = store.require_game(game.game_id)
    assert game.play_reservation_id == booking.reservation_id
    assert store.room_reservations[booking.reservation_id].status == "occupied"
    assert world.tick(_FINISH - timedelta(minutes=1))["occupied_one_minute_before_end"] == 1

    with using_clock(lambda: _FINISH):
        # 计划结束并不等于实际散场。同房的紧接预约此刻依然不能落库。
        inventory = store.search_room_availability(start_at=_FINISH, end_at=_REBOOK_END)
        assert inventory["room_count"] == 2 and inventory["available_room_ids"] == [_ALTERNATE]
        assert inventory["occupied_room_ids"] == [_ROOM]
        assert store.require_game(game.game_id).is_playing
        with pytest.raises(ValueError, match="room is unavailable"):
            store.reserve_room(conversation_id="next-private", game_id=None, room_id=_ROOM,
                               start_at=_FINISH, end_at=_REBOOK_END, trace_id="rejected-before-actual-release")
        assert len(store.room_reservations) == 2

        departures = [PhysicalActorEvent(f"departure:{actor}", game.game_id, actor, "leave", _FINISH, "controlled")
                      for actor in _ATTENDEES]
        report = world.tick(_FINISH, departures=departures)
        assert report["status"] == "passed"
        assert report["four_hour_finishes"] == report["released_rooms"] == report["same_room_rebookable"] == 1
        reopened = SQLiteAgentStore(store.path)
        assert _ROOM in reopened.search_room_availability(start_at=_FINISH, end_at=_REBOOK_END)["available_room_ids"]
        rebooked, _ = reopened.reserve_room(conversation_id="next-private", game_id=None, room_id=_ROOM,
                                           start_at=_FINISH, end_at=_REBOOK_END, trace_id="book-after-actual-release")

    persisted = SQLiteAgentStore(store.path)
    game = persisted.require_game(game.game_id)
    assert game.status == GameStatus.FINISHED and game.actual_ended_at - game.actual_started_at == timedelta(hours=4)
    holds = sorted(persisted.room_reservations.values(), key=lambda hold: hold.start_at)
    assert [(hold.room_id, hold.reservation_id, hold.status) for hold in holds] == [
        (_ROOM, maintenance.reservation_id, "held"),
        (_ROOM, booking.reservation_id, "released"),
        (_ROOM, rebooked.reservation_id, "held"),
    ]
    assert [hold.start_at for hold in holds] == [_CLOSURE_START, _REOPEN, _FINISH]
    assert [hold.end_at for hold in holds] == [_REOPEN, _FINISH, _REBOOK_END]
    room_traces = {transition.trace_id for transition in persisted.transitions if transition.entity_type == "room_reservation"}
    assert not room_traces & {"rejected-maintenance-overlap", "rejected-before-actual-release"}
