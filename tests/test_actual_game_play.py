from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import pytest

from mahjong_agent_runtime import InMemoryAgentStore, SQLiteAgentStore
from mahjong_agent_runtime.domains.game_domain import duration_hours_from_requirement
from mahjong_agent_runtime.domains.model_context import game_requirement_for_conversation
from mahjong_agent_runtime.group_chat.projections import public_group_game_summary
from mahjong_agent_runtime.models import DEFAULT_TZ, GameStatus, now, using_clock


ATTENDEES = ("a", "b", "c", "d")
START = datetime(2031, 4, 10, 18, tzinfo=DEFAULT_TZ)


@pytest.fixture(params=["memory", "sqlite"])
def venue(request, tmp_path):
    path = tmp_path / "venue.sqlite3"
    store = InMemoryAgentStore() if request.param == "memory" else SQLiteAgentStore(path)
    store.configure_rooms(["east", "west"])
    return store, path


def create_ready_game(store, *, start_at=START, guests=ATTENDEES, duration_terms=None):
    with using_clock(lambda: start_at - timedelta(hours=1)):
        game, _ = store.create_game(
            conversation_id=f"chat:{guests[0]}", organizer_id=guests[0], organizer_name=guests[0],
            requirement={"start_time_kind": "scheduled", "planned_start_at": start_at.isoformat(),
                         **({"duration_hours": 4} if duration_terms is None else duration_terms)},
            known_players=[], trace_id="make-game",
        )
        for attendee in guests[1:]:
            game, _ = store.join_game(game_id=game.game_id, customer_id=attendee, display_name=attendee, trace_id="confirm")
    return game


@pytest.mark.parametrize("duration_terms, provisioning_hours", [({}, 4), ({"max_duration_hours": 3}, 3)])
def test_unknown_duration_provisioning_does_not_exceed_agreed_maximum(venue, duration_terms, provisioning_hours):
    """上限约束内部排期、实际用房和安全散场，具体约定仍然未知。"""
    store, path = venue
    game = create_ready_game(store, duration_terms=duration_terms)
    end = START + timedelta(hours=provisioning_hours)
    if isinstance(store, SQLiteAgentStore):
        store = SQLiteAgentStore(path)
    saved = store.require_game(game.game_id)
    assert saved.planned_end_at == saved.expires_at == end
    assert "duration_hours" not in saved.requirement
    assert saved.requirement.get("max_duration_hours") == duration_terms.get("max_duration_hours")
    assert "planned_end_at" not in game_requirement_for_conversation(saved.requirement)
    assert public_group_game_summary(saved)["planned_end_at"] is None
    started, _ = store.start_game_play(game.game_id, attendee_ids=ATTENDEES, room_id="east", at=START, trace_id="start")
    assert started.expected_play_end_at == end
    assert store.room_reservations[started.play_reservation_id].end_at == end
    if isinstance(store, SQLiteAgentStore):
        store = SQLiteAgentStore(path)
    with using_clock(lambda: end):
        assert store.search_room_availability(start_at=end, end_at=end + timedelta(minutes=30))["occupied_room_ids"] == ["east"]
        finished, _ = store.finish_game_play(game.game_id, at=end, trace_id="finish")
        assert finished.actual_ended_at == end
        assert store.search_room_availability(start_at=end, end_at=end + timedelta(minutes=30))["available_count"] == 2


@pytest.mark.parametrize("terms, expected", [
    ({"max_duration_hours": 6}, 4),
    ({"max_duration_hours": "nan"}, 4),
    ({"duration_kind": "overnight"}, 8),
    ({"duration_hours": 2, "max_duration_hours": 3}, 2),
    ({"duration_hours": 4, "max_duration_hours": 3}, 4),
])
def test_provisioning_clamps_only_the_fallback_duration(terms, expected):
    """保留显式时长的既有计算，上限只约束非约定的预估值。"""
    assert duration_hours_from_requirement(terms) == expected


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


def test_owner_assigns_one_public_room_and_actual_start_reuses_it(venue):
    """预分房只写实有包间；开打仍须可信到场，且继续持有同一间到散场。"""
    store, path = venue
    game = create_ready_game(store)
    game_before = game.to_dict()
    early = START - timedelta(minutes=5)
    assigned, changes = store.assign_game_room(
        game.game_id, room_id="east", public_room_label="东一包间", at=early, trace_id="operator-assign",
    )
    assert (assigned.game_id, assigned.conversation_id, assigned.status) == (game.game_id, game.conversation_id, "confirmed")
    assert (assigned.start_at, assigned.end_at) == (START, game.planned_end_at)
    assert assigned.public_room_label == "东一包间"
    assert store.require_game(game.game_id).to_dict() == game_before
    assert [(change.from_status, change.to_status, change.reason, change.trace_id) for change in changes] == [
        (None, "confirmed", "owner_assigned_game_room", "operator-assign"),
    ]

    if isinstance(store, SQLiteAgentStore):
        store = SQLiteAgentStore(path)
    assert store.room_reservations[assigned.reservation_id].to_dict() == assigned.to_dict()
    again, changes = store.assign_game_room(
        game.game_id, room_id="east", public_room_label=" 东一包间 ", at=START, trace_id="operator-retry",
    )
    assert not changes and again.to_dict() == assigned.to_dict()
    assert [change.trace_id for change in store.transitions if change.reason == "owner_assigned_game_room"] == [
        "operator-assign",
    ]
    with pytest.raises(ValueError, match="different room assignment"):
        store.assign_game_room(game.game_id, room_id="east", public_room_label="东二包间", at=early, trace_id="relabel")
    with pytest.raises(ValueError, match="conflicting room reservation"):
        store.assign_game_room(game.game_id, room_id="west", public_room_label="西侧包间", at=early, trace_id="move")
    with pytest.raises(ValueError, match="current assigned room"):
        store.start_game_play(game.game_id, attendee_ids=ATTENDEES, room_id="west", at=START, trace_id="wrong-room")
    assert store.require_game(game.game_id).to_dict() == game_before
    assert len(store.room_reservations) == 1

    started, _ = store.start_game_play(game.game_id, attendee_ids=ATTENDEES, room_id="east", at=START, trace_id="play")
    assert started.play_reservation_id == assigned.reservation_id
    occupied = store.room_reservations[assigned.reservation_id]
    assert (occupied.status, occupied.public_room_label) == ("occupied", "东一包间")
    with pytest.raises(ValueError, match="has not started"):
        store.assign_game_room(game.game_id, room_id="east", public_room_label="东一包间", at=START, trace_id="late-assignment")
    store.finish_game_play(game.game_id, at=game.planned_end_at, trace_id="finished")
    assert store.room_reservations[assigned.reservation_id].status == "released"


def test_owner_room_assignment_checks_agreed_window_and_real_inventory(venue):
    """已满与有空席有别；未知时长、临时标签和他人订房都不能生成分配。"""
    store, _ = venue
    at = START - timedelta(minutes=5)
    ready = create_ready_game(store)
    unknown_terms = create_ready_game(store, guests=("u", "v", "w", "x"), duration_terms={})
    with using_clock(lambda: at):
        forming, _ = store.create_game(
            conversation_id="one", organizer_id="one", organizer_name="单人", requirement={
                "planned_start_at": START.isoformat(), "start_time_kind": "scheduled", "duration_hours": 4,
            }, known_players=[], trace_id="forming",
        )
        blocked, _ = store.reserve_room(
            conversation_id="other-booking", game_id=None, start_at=START, end_at=START + timedelta(hours=1),
            room_id="east", trace_id="conflict",
        )
    for target, params, expected in [
        (forming, {}, "full ready game"),
        (unknown_terms, {}, "agreed remaining game window"),
        (ready, {"room_id": "missing"}, "not configured"),
        (ready, {"room_id": "east"}, "unavailable"),
        (ready, {"public_room_label": "  "}, "valid public room label"),
        (ready, {"public_room_label": "西侧包间\u2028第二句话"}, "valid public room label"),
        (ready, {"at": at.replace(tzinfo=None)}, "requires a timezone"),
        (ready, {"at": ready.planned_end_at}, "has not started"),
    ]:
        arguments = {"room_id": "west", "public_room_label": "西侧包间", "at": at, "trace_id": "rejected", **params}
        with pytest.raises(ValueError, match=expected):
            store.assign_game_room(target.game_id, **arguments)
    assert set(store.room_reservations) == {blocked.reservation_id}
    assert not any(change.trace_id == "rejected" for change in store.transitions)

    late = START + timedelta(minutes=5)
    assigned, _ = store.assign_game_room(
        ready.game_id, room_id="west", public_room_label="西侧包间", at=late, trace_id="late-but-unstarted",
    )
    assert (assigned.start_at, assigned.end_at) == (late, ready.planned_end_at)
    with using_clock(lambda: late):
        store.update_game_status(game_id=ready.game_id, status="cancelled", reason="cancelled-by-host", trace_id="cancel")
    assert store.room_reservations[assigned.reservation_id].status == "released"
    assert store.room_reservations[blocked.reservation_id].status == "held"


def test_owner_can_confirm_an_exact_preexisting_hold_without_duplicate(venue):
    """已有同局、同间、同窗预约时只补确认和公开房名。"""
    store, _ = venue
    game = create_ready_game(store)
    at = START - timedelta(minutes=5)
    with using_clock(lambda: at):
        held, _ = store.reserve_room(conversation_id=game.conversation_id, game_id=game.game_id, room_id="east",
                                     start_at=START, end_at=game.planned_end_at, trace_id="existing-hold")
    assigned, changes = store.assign_game_room(
        game.game_id, room_id="east", public_room_label="东一包间", at=at, trace_id="owner-confirm",
    )
    assert assigned.reservation_id == held.reservation_id
    assert set(store.room_reservations) == {held.reservation_id}
    assert (changes[0].from_status, changes[0].to_status) == ("held", "confirmed")
    assert assigned.public_room_label == "东一包间"


@pytest.mark.parametrize("same_game", [False, True])
def test_concurrent_owner_room_assignments_cannot_duplicate_room_or_game(venue, same_game):
    """同局不同包间和同包间不同局两种竞态都只能留下一个成功结果。"""
    store, path = venue
    first = create_ready_game(store)
    second = first if same_game else create_ready_game(store, guests=("e", "f", "g", "h"))
    worker = SQLiteAgentStore(path) if isinstance(store, SQLiteAgentStore) else store
    at = START - timedelta(minutes=5)

    def assign(target, game, room):
        try:
            return target.assign_game_room(
                game.game_id, room_id=room, public_room_label=f"包间 {room}", at=at, trace_id=f"{game.game_id}-{room}",
            )[0]
        except ValueError as exc:
            assert str(exc) == ("game already has a conflicting room reservation" if same_game
                                else "room is unavailable for the game window")
            return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        attempts = [executor.submit(assign, store, first, "east"),
                    executor.submit(assign, worker, second, "west" if same_game else "east")]
    assert sum(attempt.result() is not None for attempt in attempts) == 1
    assert len(store.room_reservations) == 1
    assert len([change for change in store.transitions if change.reason == "owner_assigned_game_room"]) == 1
    assert all(game.actual_started_at is None for game in store.games.values())
