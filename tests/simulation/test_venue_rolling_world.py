"""用正式沙盒库存核对四小时物理局和跨库恢复，不请求模型。"""

from dataclasses import replace
from datetime import datetime, timedelta

import pytest

from mahjong_agent_runtime import SQLiteAgentStore
from mahjong_agent_runtime.models import GameStatus, using_clock

from .venue_rolling_world import PhysicalActorEvent, VenueRollingWorld
from .venue_scenario import VENUE


_ATTENDEES = ("friend-a", "friend-b", "friend-c", "friend-d")
_PLANNED = datetime.fromisoformat("2026-11-25T19:00:00+08:00")
_START = _PLANNED + timedelta(minutes=3)
_END = _START + timedelta(hours=4)


def _fixture(tmp_path, *, attendees=_ATTENDEES, duration_kind="fixed", smoke="no_smoking"):
    """构造组件外已通过业务接口建局和报名的隔离库存。"""
    store = SQLiteAgentStore(tmp_path / "agent.sqlite3")
    store.configure_rooms([room["id"] for room in VENUE["rooms"]])
    with using_clock(lambda: _PLANNED - timedelta(hours=1)):
        store.reserve_room(conversation_id="maintenance", game_id=None, room_id="东一包间",
                           start_at=_PLANNED - timedelta(minutes=30), end_at=_PLANNED + timedelta(hours=1),
                           trace_id="known-maintenance")
        game, _ = store.create_game(conversation_id="organizer-private", organizer_id=attendees[0],
            organizer_name="发起人", known_players=[], trace_id="fixture-creation",
            requirement={"start_time_kind": "scheduled", "planned_start_at": _PLANNED.isoformat(),
                         "smoke_preference": smoke, "duration_kind": duration_kind, "duration_hours": 4})
        for actor_id in attendees[1:]:
            game, _ = store.join_game(game_id=game.game_id, customer_id=actor_id, display_name=actor_id,
                                      trace_id="fixture-commitment")
    return store, game


def _decisions(game_id, action, at):
    """明确标注是组件外的离线角色决定，复用稳定的来源编号。"""
    return [PhysicalActorEvent(f"{action}:{actor}", game_id, actor, action, at, "controlled") for actor in _ATTENDEES]


def _schedule(world, game):
    """在预定时间之后开打，所有确认到场事件都发生在实际开局之前。"""
    return world.schedule(game.game_id, start_at=_START,
                          arrivals=_decisions(game.game_id, "arrive", _PLANNED - timedelta(minutes=5)))


def test_world_runs_actual_four_hours_and_same_room_is_rebookable_after_release(tmp_path):
    """三次物理 tick 分别核实真实开局、四小时前占用、散场与原房间可用。"""
    store, game = _fixture(tmp_path)
    world = VenueRollingWorld(store, tmp_path)
    assert world.report()["status"] == "not_observed"
    scheduled = _schedule(world, game)
    assert scheduled["status"] == "scheduled"
    assert world.tick(_START - timedelta(seconds=1))["actually_started"] == 0
    started = world.tick(_START)
    played = store.require_game(game.game_id)
    reservation = store.room_reservations[played.play_reservation_id]
    assert started["status"] == "pending" and started["actually_started"] == 1
    assert played.actual_started_at == _START and played.expected_play_end_at == _END
    assert played.actual_attendee_ids == _ATTENDEES
    assert reservation.room_id == "东二包间" and reservation.status == "occupied"
    assert started["evidence"][0]["game_id"] == game.game_id
    assert len(started["evidence"][0]["decision"]["arrivals"]) == 4
    assert started["evidence"][0]["decision"]["arrivals"]["friend-a"]["event_id"] == "arrive:friend-a"

    departures = _decisions(game.game_id, "leave", _END)
    before = world.tick(_END - timedelta(minutes=1), departures=departures)
    assert before["occupied_one_minute_before_end"] == 1 and before["four_hour_finishes"] == 0
    finished = world.tick(_END, departures=departures)
    assert finished["status"] == "passed"
    assert finished["four_hour_finishes"] == finished["released_rooms"] == finished["same_room_rebookable"] == 1
    reopened = SQLiteAgentStore(store.path)
    persisted = reopened.require_game(game.game_id)
    assert persisted.status == GameStatus.FINISHED and persisted.actual_ended_at - persisted.actual_started_at == timedelta(hours=4)
    assert reopened.room_reservations[reservation.reservation_id].status == "released"
    assert len(reopened.room_reservations) == 2  # 设备检修与本局实际占房，可订用正式查询核对。
    finish = finished["evidence"][-1]
    assert finish["details"]["reservation_id"] == reservation.reservation_id
    assert {row["event_id"] for row in finish["decision"]["departures"].values()} == {event.event_id for event in departures}
    world_reopened = VenueRollingWorld(reopened, tmp_path)
    assert _schedule(world_reopened, game) == scheduled
    assert world_reopened.tick(_END) == finished


@pytest.mark.parametrize("condition", ["unfilled", "duration_default", "unmatched", "early", "late", "closing"])
def test_world_does_not_fill_missing_roster_or_start_unconfirmed_games(tmp_path, condition):
    """少人、时长未确认、签到与同桌不符或跨关门时间都没有物理候选。"""
    attendees = _ATTENDEES[:3] if condition == "unfilled" else _ATTENDEES
    store, game = _fixture(tmp_path, attendees=attendees, duration_kind="unknown" if condition == "duration_default" else "fixed")
    world = VenueRollingWorld(store, tmp_path)
    arrivals = _decisions(game.game_id, "arrive", _PLANNED)
    start = _START
    if condition == "unmatched":
        arrivals[-1] = replace(arrivals[-1], actor_id="spectator")
    elif condition == "early":
        arrivals[-1] = replace(arrivals[-1], at=_PLANNED - timedelta(hours=2))
    elif condition == "late":
        arrivals[-1] = replace(arrivals[-1], at=_START + timedelta(minutes=1))
    elif condition == "closing":
        start = _START.replace(hour=20)
    assert world.schedule(game.game_id, start_at=start, arrivals=arrivals)["status"] == "not_observed"
    assert world.tick(_END)["candidate_games"] == 0
    assert not store.require_game(game.game_id).is_playing
    assert len(store.room_reservations) == 1


def test_world_does_not_open_in_a_room_that_conflicts_with_confirmed_smoke_policy(tmp_path):
    """沿用正式烟况别名，只剩吸烟包间时保留阻塞，原局和预约不变。"""
    store, game = _fixture(tmp_path, smoke=["无烟"])
    store.configure_rooms(["西侧包间"])
    world = VenueRollingWorld(store, tmp_path)
    _schedule(world, game)
    report = world.tick(_START)
    assert report["status"] == "blocked" and report["candidate_games"] == 1 and report["actually_started"] == 0
    assert report["evidence"][0]["result"]["reason"] == "no_compatible_available_room"
    assert not store.require_game(game.game_id).is_playing and len(store.room_reservations) == 1


def test_world_keeps_room_occupied_if_a_player_has_not_left_at_four_hours(tmp_path):
    """时间到达时，别桌来客或提前的动作都不能取代本局最后一人的离场。"""
    store, game = _fixture(tmp_path)
    world = VenueRollingWorld(store, tmp_path)
    _schedule(world, game)
    departures = _decisions(game.game_id, "leave", _END)
    false_sources = [replace(departures[-1], game_id="other-game"),
                     replace(departures[-1], at=_END - timedelta(minutes=1)),
                     replace(departures[-1], actor_id="spectator")]
    report = world.tick(_END, departures=[*departures[:3], *false_sources])
    assert report["status"] == "blocked" and report["missing_departures_at_four_hours"] == 1
    assert report["actually_started"] == report["occupied_one_minute_before_end"] == 1
    assert report["released_rooms"] == report["same_room_rebookable"] == 0
    result = report["evidence"][-1]["result"]
    assert result["missing_actor_ids"] == ["friend-d"] and result["room_still_occupied"]
    reopened = SQLiteAgentStore(store.path)
    persisted = reopened.require_game(game.game_id)
    assert persisted.is_playing and reopened.room_reservations[persisted.play_reservation_id].status == "occupied"


@pytest.mark.parametrize("interrupted_step", ["start", "finish"])
def test_world_reuses_transitions_and_role_evidence_after_business_commit(tmp_path, monkeypatch, interrupted_step):
    """正式事务提交后丢失办结记录，重开能复用原参数和原回执，没有二次占用或释放。"""
    store, game = _fixture(tmp_path)
    world = VenueRollingWorld(store, tmp_path)
    _schedule(world, game)
    journal_finish = world.journal.finish

    def stop_before_marking(event, result):
        """模拟业务和事件账本两个持久库之间的精确中断。"""
        if event.kind == interrupted_step:
            raise RuntimeError("interrupt after durable business write")
        return journal_finish(event, result)

    monkeypatch.setattr(world.journal, "finish", stop_before_marking)
    departures = _decisions(game.game_id, "leave", _END)
    through = _START if interrupted_step == "start" else _END
    with pytest.raises(RuntimeError, match="durable business write"):
        world.tick(through, departures=departures)
    reopened = SQLiteAgentStore(store.path)
    new_world = VenueRollingWorld(reopened, tmp_path)
    # 散场计划已经保存，恢复者无需再从角色生成离场决定。
    report = new_world.tick(_END, departures=departures if interrupted_step == "start" else ())
    assert report["status"] == "passed"
    assert new_world.tick(_END) == report
    transitions = [row for row in reopened.transitions if row.trace_id.startswith("venue-physical:")]
    assert sum(row.to_status == "occupied" for row in transitions) == 1
    assert sum(row.to_status == "released" for row in transitions) == 1
    assert len([row for row in reopened.room_reservations.values() if row.game_id == game.game_id]) == 1
