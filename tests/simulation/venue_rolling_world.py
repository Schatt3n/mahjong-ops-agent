"""推进滚动沙盒中已确认的物理牌局，保留同一局的到场、占用和散场证据。"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

from mahjong_agent_runtime import SQLiteAgentStore
from mahjong_agent_runtime.domains.game_domain import START_KIND_SCHEDULED, duration_hours_from_requirement
from mahjong_agent_runtime.domains.model_context import has_agreed_game_duration
from mahjong_agent_runtime.domains.value_utils import smoke_matches
from mahjong_agent_runtime.models import DEFAULT_TZ, Game, GameStatus, using_clock

from .venue_hours import VenueHours
from .venue_role_journal import RoleEvent, VenueRoleJournal
from .venue_scenario import PLAY_DURATION, PROBE_DURATION, VENUE


ARRIVE_BEFORE_START = timedelta(minutes=1)


@dataclass(frozen=True, slots=True)
class PhysicalActorEvent:
    """已验证的角色动作引用，由调用方从真实决定或受控离线决定提供。"""

    event_id: str
    game_id: str
    actor_id: str
    action: Literal["arrive", "leave"]
    at: datetime
    source: Literal["model", "controlled"]

    def evidence(self) -> dict[str, str]:
        """保存角色决定来源、同一局和准确的营业时间，供恢复后追溯。"""
        return {"event_id": self.event_id, "game_id": self.game_id, "actor_id": self.actor_id,
                "action": self.action, "at": _venue_time(self.at).isoformat(), "source": self.source}


class VenueRollingWorld:
    """只调度沙盒实际牌局，未收到真实动作时保留未开局或未散场证据。"""

    def __init__(self, store: SQLiteAgentStore, run_dir: Path) -> None:
        """物理事件独立归档，业务操作始终使用同一运行目录的沙盒库存。"""
        self.run_dir = Path(run_dir).resolve()
        if store.path.resolve() != self.run_dir / "agent.sqlite3":
            raise ValueError("physical play requires the current run's sandbox business store")
        self.store = store
        self.journal = VenueRoleJournal(self.run_dir / "physical_events.sqlite3")
        self.hours = VenueHours.from_config(VENUE)

    def schedule(self, game_id: str, *, start_at: datetime, arrivals: Iterable[PhysicalActorEvent]) -> dict:
        """签到来源必须与已落库四席一致，实际四小时须在商量好的连续营业窗口内。"""
        start = _venue_time(start_at)
        end = start + PLAY_DURATION
        opens, closes = self.hours.service_window(start)
        midnight = start.replace(hour=0, minute=0, second=0, microsecond=0)
        # 跨夜窗口保留本营业段在午夜前的签到。全天营业只额外接受预设的一分钟提前签到。
        arrival_floor = min(midnight, start - ARRIVE_BEFORE_START if closes is None else opens)
        arrivals = _actor_evidence(arrivals, game_id=game_id, action="arrive", at=start, not_before=arrival_floor)
        details = {"actual_start_at": start.isoformat(), "actual_end_at": end.isoformat(), "arrivals": arrivals}
        with self.journal.run_owner():
            existing = next((event for event in self.journal.events()
                             if event.actor_id == game_id and event.kind == "start"), None)
            if existing is None:
                game = self.store.games.get(game_id)
                if game is None or not _four_hour_candidate(game):
                    return {"status": "not_observed", "reason": "no_full_scheduled_four_hour_game"}
                if set(arrivals) != _committed_ids(game) or any(
                    datetime.fromisoformat(row["at"]) < game.created_at for row in arrivals.values()
                ):
                    return {"status": "not_observed", "reason": "missing_or_unmatched_arrivals"}
                if not self.accepts_start(game, start):
                    return {"status": "not_observed", "reason": "outside_game_or_venue_hours"}
            event = self.journal.enqueue(actor_id=game_id, kind="start", origin_id=game_id,
                                         due_at=start, details=details)
        return {"status": "scheduled", "event_id": event.event_id}

    def accepts_start(self, game: Game, start: datetime) -> bool:
        """核对实际开场仍承接原约定与到期条件，延迟后的结束时刻也要能落在该窗口。"""
        if game.planned_start_at is None:
            return False
        planned = _venue_time(game.planned_start_at)
        return (planned <= start and (game.expires_at is None or start < game.expires_at)
                and self.hours.covers(planned, start + PLAY_DURATION))

    def tick(self, through: datetime, *, departures: Iterable[PhysicalActorEvent] = ()) -> dict:
        """按虚拟时刻处理到期事件；本轮已真实决定的离场必须随散场时刻传入。"""
        until = _venue_time(through)
        departure_events = tuple(departures)
        with self.journal.run_owner():
            while any(event.status == "pending" and event.due_at <= until for event in self.journal.events()):
                event, = self.journal.claim_wave(1)
                at = _venue_time(event.due_at)
                with using_clock(lambda: at):
                    if event.decision is None:
                        event = self.journal.save_decision(event, self._plan(event, at, departure_events))
                    self.journal.finish(event, self._execute(event, at))
            return self.report()

    def _plan(self, event: RoleEvent, at: datetime, departures: tuple[PhysicalActorEvent, ...]) -> dict:
        """先保存房间选择和真实动作来源，使业务提交后再次执行能使用原参数。"""
        if event.kind == "start":
            game = self.store.require_game(event.actor_id)
            return {"room_id": self._room(game, at), "arrivals": event.details["arrivals"]}
        if event.kind == "finish":
            game = self.store.require_game(event.actor_id)
            evidence = _actor_evidence(departures, game_id=event.actor_id, action="leave", at=at)
            return {"departures": {actor_id: row for actor_id, row in evidence.items() if actor_id in game.actual_attendee_ids}}
        return {"room_id": event.details["room_id"]}

    def _room(self, game: Game, at: datetime) -> str | None:
        """优先使用本局合法预约，其次选择可用且满足容量和烟况的房间。"""
        end = at + PLAY_DURATION
        smoke = game.requirement.get("smoke_preference")
        rooms = [room["id"] for room in VENUE["rooms"] if room["seats"] >= game.seats_total
                 and smoke_matches(smoke, "smoking" if room["smoking"] else "no_smoking")]
        booked = {item.room_id for item in self.store.room_reservations.values() if item.game_id == game.game_id
                  and item.status in {"held", "confirmed"} and item.start_at < end and item.end_at > at}
        available = self.store.search_room_availability(start_at=at, end_at=end)["available_room_ids"]
        return next((room for room in rooms if room in booked),
                    next((room for room in rooms if room in available), None))

    def _execute(self, event: RoleEvent, at: datetime) -> dict:
        """开局、临界占房和散场分别使用各自到期的虚拟时刻执行。"""
        if event.kind == "start":
            return self._start(event, at)
        if event.kind == "finish":
            return self._finish(event, at)
        room_id = event.details["room_id"]
        availability = self.store.search_room_availability(start_at=at, end_at=at + timedelta(minutes=2))
        receipt = self.store.room_reservations.get(event.details["reservation_id"])
        occupied = (receipt is not None and receipt.status == "occupied" and receipt.game_id == event.actor_id
                    and receipt.room_id == room_id and receipt.start_at <= at < receipt.end_at
                    and room_id in availability["occupied_room_ids"])
        return {"status": "occupied" if occupied else "not_occupied", "room_id": room_id,
                "reservation_id": event.details["reservation_id"], "availability": availability}

    def _start(self, event: RoleEvent, at: datetime) -> dict:
        """用正式事务提交物理开局和占房，安排临界时刻的后续观察。"""
        room_id = event.decision["room_id"]
        if room_id is None:
            return {"status": "blocked", "reason": "no_compatible_available_room"}
        current = self.store.require_game(event.actor_id)
        if current.actual_started_at is None and not _four_hour_candidate(current):
            return {"status": "blocked", "reason": "committed_game_changed"}
        if current.actual_started_at is None and not self.accepts_start(current, at):
            return {"status": "blocked", "reason": "outside_game_or_venue_hours"}
        trace_id = self._trace(event)
        try:
            game, _ = self.store.start_game_play(event.actor_id, room_id=room_id, at=at, trace_id=trace_id,
                                                 attendee_ids=sorted(event.decision["arrivals"]))
        except ValueError as exc:
            return {"status": "blocked", "reason": "start_rejected", "error": str(exc)}
        transitions = self._transitions(trace_id)
        end = datetime.fromisoformat(event.details["actual_end_at"])
        if (game.actual_started_at != at or game.expected_play_end_at != end or not any(
                row["entity_id"] == game.play_reservation_id and row["to_status"] == "occupied" for row in transitions)):
            return {"status": "blocked", "reason": "missing_start_transition"}
        details = {"room_id": room_id, "reservation_id": game.play_reservation_id}
        for kind, due in (("before_end", end - timedelta(minutes=1)), ("finish", end)):
            self.journal.enqueue(actor_id=game.game_id, kind=kind, origin_id=game.game_id, due_at=due, details=details)
        return {"status": "started", **details, "actual_started_at": at.isoformat(),
                "expected_play_end_at": end.isoformat(), "transitions": transitions}

    def _finish(self, event: RoleEvent, at: datetime) -> dict:
        """四小时且全员离场才提交散场；重开库存核实释放及同房可用性。"""
        game = self.store.require_game(event.actor_id)
        room_id = event.details["room_id"]
        reservation_id = event.details["reservation_id"]
        missing = set(game.actual_attendee_ids) - event.decision["departures"].keys()
        if missing:
            occupied = self.store.room_reservations[reservation_id].status == "occupied"
            return {"status": "still_playing", "reason": "missing_departures_at_four_hours",
                    "missing_actor_ids": sorted(missing), "room_still_occupied": occupied}
        trace_id = self._trace(event)
        try:
            self.store.finish_game_play(event.actor_id, at=at, trace_id=trace_id)
        except ValueError as exc:
            return {"status": "blocked", "reason": "finish_rejected", "error": str(exc)}
        reopened = SQLiteAgentStore(self.store.path)
        persisted = reopened.require_game(event.actor_id)
        released = reopened.room_reservations[reservation_id]
        transitions = self._transitions(trace_id, store=reopened)
        exact = (persisted.status == GameStatus.FINISHED and persisted.actual_ended_at == at
                 and persisted.actual_ended_at - persisted.actual_started_at == PLAY_DURATION)
        room_released = (released.room_id == room_id and released.status == "released" and any(
            row["entity_id"] == reservation_id and row["to_status"] == "released" for row in transitions))
        next_opens, next_closes = self.hours.service_window(at)
        available_at = max(at, next_opens)
        available_until = available_at + PROBE_DURATION
        if next_closes is not None:
            available_until = min(available_until, next_closes)
        availability = reopened.search_room_availability(start_at=available_at, end_at=available_until)
        return {"status": "finished" if exact and room_released else "blocked", "actual_four_hour_finish": exact,
                "actual_ended_at": persisted.actual_ended_at.isoformat() if persisted.actual_ended_at else None,
                "room_released": room_released,
                "same_room_rebookable": room_id in availability["available_room_ids"], "availability": availability,
                "room_id": room_id, "reservation_id": reservation_id, "transitions": transitions}

    @staticmethod
    def _trace(event: RoleEvent) -> str:
        """事务、跨库恢复和事件账本引用同一个稳定追踪键。"""
        return f"venue-physical:{event.event_id}"

    def _transitions(self, trace_id: str, *, store: SQLiteAgentStore | None = None) -> list[dict]:
        """幂等调用可能空返回，因此从持久业务历史核实原始事务。"""
        return [row.to_dict() for row in (store or self.store).transitions if row.trace_id == trace_id]

    def report(self) -> dict:
        """区分尚未观察到候选、实际开局、按时散场与仍未离场。"""
        events = self.journal.events()
        results: dict = {}
        evidence = []
        for event in events:
            results.setdefault(event.kind, {})[event.actor_id] = event.result or {}
            row = event.to_dict()
            row["game_id"] = row.pop("actor_id")
            row.pop("claim_id")
            evidence.append(row)
        starts = results.get("start", {})
        before = results.get("before_end", {})
        finishes = results.get("finish", {})
        completed = [game_id for game_id, result in finishes.items() if result.get("status") == "finished"
                     and result["same_room_rebookable"] and before.get(game_id, {}).get("status") == "occupied"]
        blocked = [result for rows in results.values() for result in rows.values()
                   if result.get("status") in {"blocked", "still_playing", "not_occupied"}]
        unavailable = sum(row.get("status") == "finished" and not row["same_room_rebookable"] for row in finishes.values())
        if not starts:
            status = "not_observed"
        elif blocked or unavailable:
            status = "blocked"
        else:
            status = "passed" if len(completed) == len(starts) else "pending"
        return {"status": status, "candidate_games": len(starts),
                "actually_started": sum(row.get("status") == "started" for row in starts.values()),
                "occupied_one_minute_before_end": sum(row.get("status") == "occupied" for row in before.values()),
                "four_hour_finishes": sum(row.get("actual_four_hour_finish", False) for row in finishes.values()),
                "released_rooms": sum(row.get("room_released", False) for row in finishes.values()),
                "same_room_rebookable": sum(row.get("same_room_rebookable", False) for row in finishes.values()),
                "post_release_room_unavailable": unavailable,
                "missing_departures_at_four_hours": sum(row.get("status") == "still_playing" for row in finishes.values()),
                "evidence": evidence}


def _venue_time(at: datetime) -> datetime:
    """房态数据库按门店时区保存，候选和调度输入必须具有真实时区。"""
    if not isinstance(at, datetime) or at.utcoffset() is None:
        raise ValueError("physical venue events require timezone-aware timestamps")
    return at.astimezone(DEFAULT_TZ)


def _committed_ids(game: Game) -> set[str]:
    """只使用真实牌局席位承诺，匿名随行人数无法映射到真实签到。"""
    return {actor_id for claim in game.seat_claims() for actor_id in (claim["contact_id"], *claim["known_member_ids"]) if actor_id}


def _four_hour_candidate(game: Game) -> bool:
    """固定四小时和具名四席都必须来自已落库需求，默认过期时间不能替代。"""
    return (game.status == GameStatus.READY and game.actual_started_at is None
            and game.seats_total == game.seat_summary()["claimed_seats"] == len(_committed_ids(game)) == 4
            and game.planned_start_at is not None and game.planned_end_at == game.planned_start_at + PLAY_DURATION
            and game.requirement.get("start_time_kind") == START_KIND_SCHEDULED
            and game.requirement.get("duration_kind") in (None, "fixed")
            and has_agreed_game_duration(game.requirement) and duration_hours_from_requirement(game.requirement) == 4)


def _actor_evidence(events: Iterable[PhysicalActorEvent], *, game_id: str, action: str, at: datetime,
                    not_before: datetime | None = None) -> dict:
    """只接受同局有效时段的决定，离场要发生在实际散场时点。"""
    rows = {}
    ids = set()
    first = not_before or at
    for event in events:
        if event.game_id != game_id or event.action != action:
            continue
        stamp = _venue_time(event.at)
        if not first <= stamp <= at:
            continue
        if not event.event_id or not event.actor_id or event.source not in {"model", "controlled"}:
            raise ValueError("physical actor evidence requires a real decision reference")
        row = event.evidence()
        if rows.get(event.actor_id) == row:
            continue
        if event.event_id in ids or event.actor_id in rows:
            raise ValueError("conflicting physical actor evidence")
        ids.add(event.event_id)
        rows[event.actor_id] = row
    return rows
