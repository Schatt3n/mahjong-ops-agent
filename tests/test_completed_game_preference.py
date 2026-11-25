"""用正式签到和散场记录验证客户自己的跨任务历史偏好。"""

from dataclasses import replace
from datetime import datetime, timedelta
import json

import pytest

from mahjong_agent_runtime import AgentContextBuilder, InMemoryAgentStore, SQLiteAgentStore, ToolGateway, UserMessage
from mahjong_agent_runtime.models import DEFAULT_TZ, CustomerProfile, GameStatus, using_clock


START = datetime(2031, 4, 10, 18, tzinfo=DEFAULT_TZ)
CUSTOMER = "habitual-person"
OTHER = "same-nickname-person"
ATTENDEES = ("host", CUSTOMER, "east-friend", "west-friend")
DEFAULT_TERMS = {"game_type": "hangzhou_mahjong", "stake": "0.5", "smoke_preference": "no_smoking"}


@pytest.fixture(params=["memory", "sqlite"])
def venue(request, tmp_path):
    store = InMemoryAgentStore() if request.param == "memory" else SQLiteAgentStore(tmp_path / "venue.sqlite3")
    store.configure_rooms(["east"])
    store.upsert_customer(CustomerProfile(CUSTOMER, "小周"))
    store.upsert_customer(CustomerProfile(OTHER, "小周"))
    return store


def _play(store, day, *, terms=None, attendees=ATTENDEES, outcome="played", party_with_known_guest=False):
    """通过真实 Store 开局、散场，或构造同等报名但没有完赛记录的对照。"""
    start = START + timedelta(days=day)
    setup = [{"customer_id": attendees[0], "seat_count": 2, "known_member_ids": list(attendees[:2])}
             ] if party_with_known_guest else []
    individual_guests = attendees[2:] if party_with_known_guest else attendees[1:]
    with using_clock(lambda: start - timedelta(hours=1)):
        game, _ = store.create_game(
            conversation_id=f"historical-chat:{day}", organizer_id=attendees[0], organizer_name=attendees[0],
            requirement={
                **(DEFAULT_TERMS if terms is None else terms),
                "start_time_kind": "scheduled", "planned_start_at": start.isoformat(), "duration_hours": 2,
            }, known_players=setup, trace_id=f"create:{day}",
        )
        for guest in individual_guests:
            game, _ = store.join_game(game_id=game.game_id, customer_id=guest, display_name=guest, trace_id=f"join:{day}")
    assert game.status == GameStatus.READY
    if outcome == "ready":
        return game
    if outcome == "paper_finished":
        closed, _ = store.update_game_status(game_id=game.game_id, status="finished", reason="manual record",
                                              trace_id=f"closed-without-attendance:{day}")
        assert closed.actual_started_at is None
        return closed
    game, _ = store.start_game_play(game.game_id, attendee_ids=attendees, room_id="east", at=start,
                                   trace_id=f"checked-in:{day}")
    if outcome == "playing":
        return game
    finished, _ = store.finish_game_play(game.game_id, at=start + timedelta(hours=2), trace_id=f"finished:{day}")
    assert finished.actual_ended_at == start + timedelta(hours=2)
    return finished


def _build_new_task(store, customer_id=CUSTOMER):
    """重启业务任务后读取生产模型上下文，保留接入前的稳定身份。"""
    conversation_id = f"current-chat:{customer_id}"
    stamp = START + timedelta(days=9)
    previous, _ = store.activate_task_context(
        conversation_id=conversation_id, customer_id=customer_id, trace_id="previous",
        activity_at=stamp - timedelta(days=1), started_at=stamp - timedelta(days=1),
        reason="fixture_old_goal", force_new=True, archive_previous=True,
    )
    selected, _ = store.activate_task_context(
        conversation_id=conversation_id, customer_id=customer_id, trace_id="current",
        activity_at=stamp, started_at=stamp, reason="fixture_new_goal", force_new=True, archive_previous=True,
    )
    assert previous.task_context_id != selected.task_context_id
    return AgentContextBuilder(store=store, tool_gateway=ToolGateway(store)).build(UserMessage(
        conversation_id=conversation_id, sender_id=customer_id, sender_name="小周", message_id="fresh-turn",
        sent_at=stamp, text="帮我组个19:00的局呗",
    ), trace_id="current")


def test_three_true_plays_survive_reopen_and_new_task_without_mutating_explicit_profile(venue):
    """个别同桌者无独立报名行也按实际到店名单计算，昵称相同不能共用证据。"""
    for day in range(2):
        _play(venue, day)
    assert venue.completed_game_preference(CUSTOMER) is None
    third = _play(venue, 2, party_with_known_guest=True)
    assert CUSTOMER in third.actual_attendee_ids and CUSTOMER not in {player.customer_id for player in third.participants}
    assert venue.finish_game_play(third.game_id, at=third.actual_ended_at, trace_id="same-finish")[1] == []
    if isinstance(venue, SQLiteAgentStore):
        venue = SQLiteAgentStore(venue.path)

    expected = {**DEFAULT_TERMS, "completed_games": 3}
    built = _build_new_task(venue)
    assert venue.completed_game_preference(CUSTOMER) == expected
    assert built.payload["sender_profile"]["attendance_preference"] == expected
    assert json.loads(built.messages[-1]["content"])["sender_profile"]["attendance_preference"] == expected
    assert set(expected) == {"game_type", "stake", "smoke_preference", "completed_games"}
    assert "attendance_preference" not in _build_new_task(venue, OTHER).payload["sender_profile"]
    assert venue.customers[CUSTOMER] == CustomerProfile(CUSTOMER, "小周")


def test_reservations_unfinished_games_and_other_customers_do_not_teach(venue):
    """已报名未签到、库中标已结束、尚未散场及其他人的已散场都不能补足样本。"""
    _play(venue, 0)
    _play(venue, 1)
    _play(venue, 2, outcome="ready")
    _play(venue, 3, outcome="paper_finished")
    _play(venue, 4, attendees=("another-host", OTHER, "unique-friend1", "unique-friend2"))
    _play(venue, 5, outcome="playing")

    assert venue.completed_game_preference(CUSTOMER) is None
    assert venue.completed_game_preference(OTHER) is None


@pytest.mark.parametrize("other_terms", [
    {"game_type": "sichuan_mahjong"},
    {"stake": "1"},
    {"smoke_preference": "smoking"},
    {"smoke_preference": None},
    {"cap_score": 16},
    {"variant": "different_rules"},
])
def test_mixed_or_incomplete_actual_history_cannot_become_single_default(venue, other_terms):
    """已观察到不同玩法/档位/烟况或不完整记录时，不能把其余牌局冒充一致。"""
    for day in range(3):
        _play(venue, day)
    assert venue.completed_game_preference(CUSTOMER) is not None
    _play(venue, 3, terms={**DEFAULT_TERMS, **other_terms})
    assert venue.completed_game_preference(CUSTOMER) is None


@pytest.mark.parametrize("profile_preferences", [
    {"preferred_games": ["sichuan_mahjong"], "smoke_preference": "any"},
    {"preferred_games": ["hangzhou_mahjong", "sichuan_mahjong"]},
    {"preferred_stakes": ["0.5", "1"]},
])
def test_explicit_profile_is_authoritative_when_it_conflicts_with_actual_history(venue, profile_preferences):
    """声明的偏好保留原值，历史观察不作为冲突任务的默认条件。"""
    for day in range(3):
        _play(venue, day)
    explicit = replace(venue.customers[CUSTOMER], **profile_preferences)
    venue.upsert_customer(explicit)
    built = _build_new_task(venue).payload["sender_profile"]

    assert "attendance_preference" not in built
    assert all(built[field] == value for field, value in profile_preferences.items())
    assert venue.customers[CUSTOMER] == explicit
