"""核对店家为已满定时局分配包间时的权威窗口与物理房态。"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import replace
from datetime import datetime
from unicodedata import category

from ..models import DEFAULT_TZ, Game, GameStatus, RoomReservation, StateTransition, new_id
from .model_context import has_agreed_game_duration


_BLOCKING_ROOM_STATUSES = frozenset({"held", "confirmed", "occupied"})


def room_reservation_blocks_window(
    reservation: RoomReservation, *, start: datetime, end: datetime, at: datetime,
) -> bool:
    """使用半开时间窗；占用超过预计散场仍未释放时继续视为占房。"""
    return (
        reservation.status in _BLOCKING_ROOM_STATUSES
        and reservation.start_at < end
        and (reservation.end_at > start or reservation.status == "occupied" and reservation.end_at <= at)
    )


def prepare_game_room_assignment(
    game: Game, *, room_id: str, public_room_label: str, at: datetime, trace_id: str,
    room_ids: Sequence[str], reservations: Iterable[RoomReservation],
) -> tuple[RoomReservation, list[StateTransition]]:
    """在 Store 锁内为同局预留真实空房，生成可审计的新预约或返回幂等结果。"""
    if not isinstance(at, datetime) or at.utcoffset() is None:
        raise ValueError("room assignment time requires a timezone")
    stamp = at.astimezone(DEFAULT_TZ)
    if (game.status != GameStatus.READY or game.seats_total != 4
            or game.seat_summary()["claimed_seats"] != 4 or game.actual_started_at is not None
            or game.actual_ended_at is not None or game.expires_at is not None and game.expires_at <= stamp):
        raise ValueError("only a full ready game that has not started can be assigned a room")
    end = game.planned_end_at
    if (game.planned_start_at is None or end is None or not has_agreed_game_duration(game.requirement)
            or end <= max(stamp, game.planned_start_at)):
        raise ValueError("room assignment requires an agreed remaining game window")
    start = max(stamp, game.planned_start_at)
    if room_id not in room_ids:
        raise ValueError("room is not configured")
    label = public_room_label.strip() if isinstance(public_room_label, str) else ""
    if not 0 < len(label) <= 80 or any(category(char)[0] == "C" or category(char) in {"Zl", "Zp"} for char in label):
        raise ValueError("a valid public room label is required")

    live = [item for item in reservations
            if item.status in _BLOCKING_ROOM_STATUSES and (item.end_at > stamp or item.status == "occupied")]
    own = [item for item in live if item.game_id == game.game_id]
    if (len(own) > 1 or any(item.room_id != room_id or item.conversation_id != game.conversation_id
                           or item.status == "occupied" for item in own)):
        raise ValueError("game already has a conflicting room reservation")
    previous = own[0] if own else None
    if previous is not None and previous.public_room_label is not None:
        # 服务端时钟随重试前进；已经审批过的同一房名和窗口无需再次写入或缩短。
        if (previous.status == "confirmed" and previous.public_room_label == label
                and game.planned_start_at <= previous.start_at <= start and previous.end_at == end):
            return previous, []
        raise ValueError("game already has a different room assignment")
    if previous is not None and (previous.start_at, previous.end_at) != (start, end):
        raise ValueError("game already has a different room reservation window")
    if any(item is not previous and item.room_id == room_id
           and room_reservation_blocks_window(item, start=start, end=end, at=stamp) for item in live):
        raise ValueError("room is unavailable for the game window")

    if previous is None:
        assigned = RoomReservation(
            reservation_id=new_id("room_reservation"), room_id=room_id, conversation_id=game.conversation_id,
            game_id=game.game_id, start_at=start, end_at=end, status="confirmed", source_trace_id=trace_id,
            created_at=stamp, updated_at=stamp, public_room_label=label,
        )
    else:
        assigned = replace(previous, status="confirmed", public_room_label=label, updated_at=stamp,
                           source_trace_id=trace_id)
    transition = StateTransition(
        "room_reservation", assigned.reservation_id, previous.status if previous else None,
        "confirmed", "owner_assigned_game_room", trace_id, occurred_at=stamp,
    )
    return assigned, [transition]
