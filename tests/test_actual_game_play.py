from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import pytest

from mahjong_agent_runtime import InMemoryAgentStore, SQLiteAgentStore
from mahjong_agent_runtime.models import DEFAULT_TZ, GameStatus, now, using_clock


ATTENDEES = ("a", "b", "c", "d")
START = datetime(2031, 4, 10, 18, tzinfo=DEFAULT_TZ)


@pytest.fixture(params=["memory", "sqlite"])
def venue(request, tmp_path):
    path = tmp_path / "venue.sqlite3"
    store = InMemoryAgentStore() if request.param == "memory" else SQLiteAgentStore(path)
    store.configure_rooms(["east", "west"])
    return store, path


def create_ready_game(store, *, start_at=START, guests=ATTENDEES):
    with using_clock(lambda: start_at - timedelta(hours=1)):
        game, _ = store.create_game(
            conversation_id=f"chat:{guests[0]}", organizer_id=guests[0], organizer_name=guests[0],
            requirement={"start_time_kind": "scheduled", "planned_start_at": start_at.isoformat(), "duration_hours": 4},
            known_players=[], trace_id="make-game",
        )
        for attendee in guests[1:]:
            game, _ = store.join_game(game_id=game.game_id, customer_id=attendee, display_name=attendee, trace_id="confirm")
    return game


def test_actual_game_lifecycle_keeps_room_until_trusted_finish(venue):
    store, path = venue
    game = create_ready_game(store)
    assert game.status == GameStatus.READY
    end = START + timedelta(hours=4)
    with using_clock(lambda: START - timedelta(minutes=10)):
        previous, _ = store.reserve_room(conversation_id=game.conversation_id, game_id=game.game_id,
            room_id="east", start_at=START, end_at=end, trace_id="pre-booked")

    with using_clock(lambda: START):
        game, transitions = store.start_game_play(game.game_id, attendee_ids=ATTENDEES, room_id="east", at=now(), trace_id="start")
        assert game.is_playing and game.actual_started_at == START and game.expected_play_end_at == end
        assert game.actual_attendee_ids == ATTENDEES
        assert game.play_reservation_id == previous.reservation_id
        assert [(item.from_status, item.to_status) for item in transitions] == [("held", "occupied"), ("ready", "playing")]
        assert store.start_game_play(game.game_id, attendee_ids=ATTENDEES[::-1], room_id="east", at=now(), trace_id="retry")[1] == []
        assert store.search_room_availability(start_at=now(), end_at=end)["occupied_room_ids"] == ["east"]
        with pytest.raises(ValueError, match="before its agreed"):
            store.finish_game_play(game.game_id, at=end - timedelta(seconds=1), trace_id="early")
        with pytest.raises(ValueError, match="trusted shop"):
            store.update_game_status(game_id=game.game_id, status="finished", reason="generic", trace_id="premature-update")
        with pytest.raises(ValueError, match="trusted shop"):
            store.record_candidate_reply(game_id=game.game_id, customer_id="b", display_name="B", status="declined", trace_id="late-claim")

    if isinstance(store, SQLiteAgentStore):
        store = SQLiteAgentStore(path)
    with using_clock(lambda: end):
        # 常规查询会检查计划到期，仍必须等待真实散场事件。
        assert store.active_games()[0].is_playing
        assert store.search_room_availability(start_at=end, end_at=end + timedelta(hours=1))["occupied_room_ids"] == ["east"]
        game, transitions = store.finish_game_play(game.game_id, at=now(), trace_id="finish")
        assert game.status == GameStatus.FINISHED and not game.is_playing
        assert game.actual_ended_at - game.actual_started_at == timedelta(hours=4)
        assert game.actual_attendee_ids == ATTENDEES
        assert all(item.occurred_at == end for item in transitions)
        assert [(item.entity_type, item.to_status) for item in transitions] == [
            ("game_play", "finished"), ("game", "finished"), ("room_reservation", "released"),
        ]
        assert store.finish_game_play(game.game_id, at=now(), trace_id="finish-retry")[1] == []
        assert store.search_room_availability(start_at=end, end_at=end + timedelta(hours=1))["available_count"] == 2
    if isinstance(store, SQLiteAgentStore):
        store = SQLiteAgentStore(path)
    assert store.require_game(game.game_id).actual_ended_at == end
    assert store.room_reservations[previous.reservation_id].status == "released"


def test_conflicting_booking_and_unidentified_arrival_leave_no_partial_start(venue):
    store, _ = venue
    game = create_ready_game(store)
    with using_clock(lambda: START):
        store.reserve_room(conversation_id="another-booking", game_id=None, room_id="east", start_at=now(),
                           end_at=now() + timedelta(hours=4), trace_id="blocked")
        with pytest.raises(ValueError, match="distinct identified"):
            store.start_game_play(game.game_id, attendee_ids=("a", "b", "c", "other"), room_id="west", at=now(), trace_id="wrong-attendee")
        with pytest.raises(ValueError, match="unavailable"):
            store.start_game_play(game.game_id, attendee_ids=ATTENDEES, room_id="east", at=now(), trace_id="room-conflict")
        assert store.require_game(game.game_id).actual_started_at is None
        assert len(store.room_reservations) == 1
        started, _ = store.start_game_play(game.game_id, attendee_ids=ATTENDEES, room_id="west", at=now(), trace_id="start")
        assert store.room_reservations[started.play_reservation_id].room_id == "west"


def test_concurrent_starts_cannot_use_same_room(venue):
    store, path = venue
    second_attendees = ("e", "f", "g", "h")
    games = [create_ready_game(store), create_ready_game(store, guests=second_attendees)]
    other_worker = SQLiteAgentStore(path) if isinstance(store, SQLiteAgentStore) else store

    def start(target, game, attendees):
        try:
            return target.start_game_play(game.game_id, attendee_ids=attendees, room_id="east", at=START, trace_id=game.game_id)[0]
        except ValueError as exc:
            assert "room is unavailable" in str(exc)
            return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = [executor.submit(start, store, games[0], ATTENDEES),
                    executor.submit(start, other_worker, games[1], second_attendees)]

    assert sum(outcome.result() is not None for outcome in outcomes) == 1
    assert sum(game.is_playing for game in store.games.values()) == 1
    assert len(store.room_reservations) == 1
