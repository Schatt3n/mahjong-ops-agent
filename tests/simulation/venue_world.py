"""生成试用演练中只有顾客本人知道的日程、出行和临时变动。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta
from hashlib import sha256
import random
from typing import Any, Literal, Sequence

from .venue_actors import Persona


WorldEventKind = Literal["session_open", "rain", "traffic", "long_commute", "out_of_town", "repeat_visit"]

_GAMES = ("杭州麻将", "四川麻将", "红中麻将")
_WORK_STATUSES = ("day_shift", "evening_shift", "flexible", "leave", "rest_day")
_FREE_BOUND_HOURS = {
    "day_shift": ((17, 19), (22, 23)),
    "evening_shift": ((9, 11), (14, 16)),
    "flexible": ((9, 12), (21, 23)),
    "leave": ((9, 11), (21, 23)),
    "rest_day": ((9, 11), (21, 23)),
}


@dataclass(frozen=True, slots=True)
class DayAvailability:
    """顾客当天私有的工况和空闲跨度，用真实钟点约束可执行的邀约。"""

    work_status: str
    location: str
    travel_minutes: int
    free_from: datetime
    free_until: datetime

    def available_for(self, start: datetime, duration_hours: float = 0) -> bool:
        """本地顾客才可能参加，整段活动必须落在其个人空档内。"""
        if duration_hours < 0:
            raise ValueError("活动时长不能为负")
        return (self.location != "outside_city" and self.free_from <= start < self.free_until
                and start + timedelta(hours=duration_hours) <= self.free_until)


@dataclass(frozen=True, slots=True)
class VenueWorld:
    """固定种子复现个人事实，同一人的某天状态不受调度并发和他人决策影响。"""

    seed: int
    base_at: datetime

    def __post_init__(self) -> None:
        if self.base_at.utcoffset() is None:
            raise ValueError("门店演练时刻需要明确时区")

    def _rng(self, actor: str, day: date | None, stream: str) -> random.Random:
        """独立派生不同用途的随机流，调用顺序不会改变已有场景。"""
        key = f"{self.seed}|{actor}|{day}|{stream}"
        return random.Random(int.from_bytes(sha256(key.encode()).digest()[:8], "big"))

    def personas(self, people: Sequence[Persona]) -> tuple[Persona, ...]:
        """只调整新演练的玩法分布，保留原有角色身份、行为习惯和历史用例种子。"""
        roster = sorted(people, key=lambda person: person.sender_id)
        rng = self._rng("venue", None, "roster")
        rng.shuffle(roster)
        return tuple(replace(person, preferred_game=_GAMES[index] if index < len(_GAMES)
                             else rng.choices(_GAMES, weights=(55, 27, 18), k=1)[0])
                     for index, person in enumerate(roster))

    def availability(self, persona: Persona, *, at: datetime | None = None) -> DayAvailability:
        """工作日白天空闲来自当日班次、弹性安排、请假或休息，不把在岗时段当空闲。"""
        day = self._local_time(at).date()
        rng = self._rng(persona.sender_id, day, "calendar")
        weights = (52, 11, 13, 7, 17) if day.weekday() < 5 else (12, 9, 11, 0, 68)
        work = rng.choices(_WORK_STATUSES, weights=weights, k=1)[0]
        start_bounds, end_bounds = _FREE_BOUND_HOURS[work]
        free_from = datetime.combine(day, time(rng.randint(*start_bounds), rng.choice((0, 30))), self.base_at.tzinfo)
        free_until = datetime.combine(day, time(rng.randint(*end_bounds), rng.choice((0, 30))), self.base_at.tzinfo)
        location = rng.choices(("near_store", "across_town", "outside_city"), weights=(65, 30, 5), k=1)[0]
        travel = rng.randint(8, 22) if location == "near_store" else rng.randint(26, 42) if location == "across_town" else 0
        return DayAvailability(work, location, travel, free_from, free_until)

    def historical_private_disturbance(self, persona: Persona, *, at: datetime) -> WorldEventKind | None:
        """独立抽取续演顾客的出行变动，调用顺序不干扰当天日程或其他事件。"""
        day = self._local_time(at).date()
        return self._rng(persona.sender_id, day, "historical-private-disturbance").choice((None, "out_of_town"))

    def apply_work_overtime(self, facts: dict[str, Any], *, busy_until: datetime) -> dict[str, Any]:
        """加班推迟最早出发，合并已有交通限制，保留本人原先的最晚散场条件。"""
        end = self._local_time(busy_until)
        revised = {**facts, "work_busy_until": end.isoformat()}
        if revised.get("event_condition") == "ordinary_day":
            revised["event_condition"] = "unexpected_work_overtime"
        if earliest := revised.get("earliest_arrival"):
            new_arrival = max(datetime.fromisoformat(earliest), end + timedelta(minutes=revised["travel_minutes"]))
            revised["can_arrive"] = new_arrival < datetime.fromisoformat(revised["free_until"])
            if revised["can_arrive"]:
                revised["earliest_arrival"] = new_arrival.isoformat()
                if planned := revised.get("scheduled_start"):
                    revised["delay_minutes"] = max(0, int((new_arrival - datetime.fromisoformat(planned)).total_seconds() // 60))
            else:
                revised.pop("earliest_arrival")
                revised.pop("delay_minutes", None)
        return revised

    def _local_time(self, at: datetime | None) -> datetime:
        """店家和顾客的日历统一以开店时区为准，拒绝无法判断工作日的无时区时间。"""
        at = self.base_at if at is None else at
        if at.utcoffset() is None:
            raise ValueError("顾客事件时刻需要明确时区")
        return at.astimezone(self.base_at.tzinfo)

    def role_event(self, persona: Persona, kind: WorldEventKind, at: datetime, *,
                   scheduled_start: datetime | None = None, caused_by: str | None = None) -> dict[str, Any]:
        """投影只给这位演员的私人事实，话术和行动交给角色模型决定。"""
        at = self._local_time(at)
        start = self._local_time(scheduled_start) if scheduled_start is not None else None
        if start is not None and start.date() != at.date():
            raise ValueError("当前演练事件和约定开局必须在同一营业日")
        day = self.availability(persona, at=at)
        rng = self._rng(persona.sender_id, at.date(), kind)
        location, travel = day.location, day.travel_minutes
        blocked = location == "outside_city"
        condition = "ordinary_day"
        earliest = None
        delay = None
        previous_end = None
        if kind in {"rain", "traffic", "long_commute"} and blocked:
            raise ValueError("本地行程事件要求顾客当天在本地")
        if kind == "out_of_town":
            location, travel, blocked, condition = "outside_city", 0, True, "cannot_travel_today"
        elif kind == "rain":
            blocked, condition = True, "heavy_rain_travel_suspended"
        elif kind == "traffic":
            if start is None:
                raise ValueError("堵车延误需要原定开局时间")
            delay_in_traffic = rng.randint(8, 18)
            earliest = max(start, day.free_from, at) + timedelta(minutes=delay_in_traffic)
            condition = "en_route_with_traffic_delay"
        elif kind == "long_commute":
            location, travel, condition = "across_town", rng.randint(27, 39), "still_far_from_shop"
        elif kind == "repeat_visit":
            condition = "played_in_morning_free_again"
            morning_end = datetime.combine(at.date(), time(12), self.base_at.tzinfo)
            previous_end = morning_end - timedelta(minutes=rng.randint(1, 20))
            if not day.available_for(previous_end - timedelta(hours=1), 1) or at < morning_end:
                raise ValueError("回访顾客当天必须实际具备上午活动安排和之后的回访时间")
        elif kind != "session_open":
            raise ValueError("未知的顾客私人事件")
        if not blocked and earliest is None:
            earliest = max(at + timedelta(minutes=travel), day.free_from)
        can_arrive = earliest is not None and earliest < day.free_until
        if start is not None and can_arrive:
            delay = max(0, int((earliest - start).total_seconds() // 60))
        facts = {
            "work_status": day.work_status, "location": location, "free_from": day.free_from.isoformat(),
            "free_until": day.free_until.isoformat(), "travel_minutes": travel if location != "outside_city" else None,
            "scheduled_start": start.isoformat() if start is not None else None,
            "earliest_arrival": earliest.isoformat() if can_arrive else None, "delay_minutes": delay,
            "previous_session_end": previous_end.isoformat() if previous_end is not None else None,
            "event_condition": condition, "can_arrive": can_arrive,
        }
        return {
            "event_id": f"world:{persona.sender_id}:{kind}:{at.isoformat()}",
            "kind": kind, "at": at.isoformat(), "visible_to": (persona.sender_id,), "caused_by": caused_by,
            "details": {key: value for key, value in facts.items() if value is not None},
        }
