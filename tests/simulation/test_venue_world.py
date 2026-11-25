"""验证试用演练的独立日历、临时事件和顾客知识边界。"""

from datetime import datetime, timedelta

import pytest

from mahjong_agent_runtime.models import DEFAULT_TZ

from .venue_actors import ActorContext, generate_personas
from .venue_world import VenueWorld


def _at(day: int, hour: int, minute: int = 0) -> datetime:
    """以明确的营业日和门店时区复现同一批事件。"""
    return datetime(2026, 11, day, hour, minute, tzinfo=DEFAULT_TZ)


def test_daily_schedule_is_repeatable_and_workday_day_shift_does_not_play_at_noon():
    """固定种子仅重现来源事实，换种子或开店日仍保留不同班次与地域。"""
    people = generate_personas(42, count=100)
    world = VenueWorld(17, _at(25, 12))
    schedule = [world.availability(person) for person in people]
    assert schedule == [VenueWorld(17, _at(25, 12)).availability(person) for person in people]
    assert schedule != [VenueWorld(21, _at(25, 12)).availability(person) for person in people]
    assert {"day_shift", "evening_shift", "leave", "flexible", "rest_day"} <= {plan.work_status for plan in schedule}
    assert any(plan.available_for(_at(25, 12), 1) for plan in schedule)
    assert all(not plan.available_for(_at(25, 12), 1) for plan in schedule if plan.work_status == "day_shift")
    saturday = [world.availability(person, at=_at(28, 12)) for person in people]
    assert all(plan.free_from.date() == _at(28, 12).date() for plan in saturday)
    assert sum(plan.work_status == "rest_day" for plan in saturday) > sum(
        plan.work_status == "rest_day" for plan in schedule
    )


def test_open_world_roster_preserves_identities_and_supports_three_distinct_games():
    """新场景的玩法配比不改变旧场景使用的名册与指定种子。"""
    people = generate_personas(8, count=15)
    world = VenueWorld(72, _at(25, 19))
    roster = world.personas(people)
    assert roster == world.personas(tuple(reversed(people)))
    assert roster == VenueWorld(72, _at(28, 19)).personas(people)
    assert {person.sender_id for person in roster} == {person.sender_id for person in people}
    assert {"杭州麻将", "四川麻将", "红中麻将"} <= {person.preferred_game for person in roster}
    assert {person.sender_id: person.temperament for person in roster} == {
        person.sender_id: person.temperament for person in people
    }


def test_traffic_commute_and_rain_derive_different_possible_arrivals():
    """迟到跨度取自可复现事实，临时无法外出时不能给演员可达时间。"""
    world = VenueWorld(31, _at(25, 19))
    person = next(person for person in generate_personas(8, count=25)
                  if world.availability(person).available_for(_at(25, 19), 2))
    start = _at(25, 19)
    traffic = world.role_event(person, "traffic", _at(25, 18, 48), scheduled_start=start)["details"]
    assert traffic["can_arrive"] and 8 <= traffic["delay_minutes"] <= 18
    assert datetime.fromisoformat(traffic["earliest_arrival"]) == start + timedelta(minutes=traffic["delay_minutes"])
    commute = world.role_event(person, "long_commute", _at(25, 18, 57), scheduled_start=start)["details"]
    assert commute["can_arrive"] and commute["location"] == "across_town"
    assert commute["delay_minutes"] == commute["travel_minutes"] - 3
    rain = world.role_event(person, "rain", _at(25, 18, 52), scheduled_start=start)["details"]
    assert rain["event_condition"] == "heavy_rain_travel_suspended" and not rain["can_arrive"]
    assert "earliest_arrival" not in rain and "delay_minutes" not in rain
    with pytest.raises(ValueError, match="原定开局时间"):
        world.role_event(person, "traffic", _at(25, 18, 48))


def test_repeat_visit_only_includes_own_morning_history_and_visible_invite():
    """顾客可见自己的上午经历，仍由实际送达的邀请触发外地顾客。"""
    world = VenueWorld(53, _at(25, 13))
    people = generate_personas(8, count=40)
    visitor = next(person for person in people if world.availability(person).available_for(_at(25, 10, 30), 1)
                   and world.availability(person).available_for(_at(25, 14), 1))
    other = next(person for person in people if person != visitor)
    seen = {"message_id": "invite-m1", "kind": "private", "sender_id": "simulation-agent",
            "recipient_id": other.sender_id, "text": "下午能来吗？"}
    invite = world.role_event(other, "out_of_town", _at(25, 13), caused_by=seen["message_id"])
    repeat = world.role_event(visitor, "repeat_visit", _at(25, 13))
    assert repeat["details"]["can_arrive"]
    previous_end = datetime.fromisoformat(repeat["details"]["previous_session_end"])
    assert previous_end.date() == _at(25, 13).date() and previous_end < _at(25, 12)

    other_context = ActorContext(other, _at(25, 13).isoformat(), history=(seen,), events=(repeat, invite)).to_dict()
    assert other_context["delivery"]["message_id"] == "invite-m1"
    assert [event["kind"] for event in other_context["events"]] == ["out_of_town"]
    assert other_context["events"][0]["details"]["location"] == "outside_city"
    assert not other_context["events"][0]["details"]["can_arrive"]

    visitor_context = ActorContext(visitor, _at(25, 13).isoformat(), history=(seen,), events=(repeat, invite)).to_dict()
    assert not visitor_context["history"] and visitor_context["delivery"] is None
    assert [event["kind"] for event in visitor_context["events"]] == ["repeat_visit"]
    assert visitor_context["events"][0]["details"]["previous_session_end"] == previous_end.isoformat()
