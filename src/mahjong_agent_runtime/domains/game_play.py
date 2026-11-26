"""校验可信门店开局与散场事件，并为 Store 生成一次原子提交的变化。"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import replace
from datetime import datetime, timedelta

from ..models import DEFAULT_TZ, Game, GameStatus, RoomReservation, StateTransition, new_id
from .game_commitments import active_game_participant_ids, game_commitment_window
from .game_domain import apply_game_recruitment_policy, duration_hours_from_requirement, normalize_datetime
from .room_assignment import room_reservation_blocks_window


def prepare_game_play_start(
    game: Game,
    *,
    attendee_ids: Sequence[str],
    room_id: str,
    at: datetime,
    trace_id: str,
    games: Iterable[Game],
    room_ids: Sequence[str],
    reservations: Iterable[RoomReservation],
) -> tuple[Game, RoomReservation, list[StateTransition]] | None:
    """按正式承诺及房态验证到场，返回未提交的开局变化，完全相同的重放返回空。"""
    # 与库存查询使用同一时区，SQLite 的时间范围索引可直接比较同一格式的值。
    stamp = normalize_datetime(at).astimezone(DEFAULT_TZ)
    attendees = tuple(sorted(attendee_ids))
    reservations_by_id = {reservation.reservation_id: reservation for reservation in reservations}
    if game.actual_started_at is not None:
        existing = reservations_by_id.get(game.play_reservation_id or "")
        if game.actual_started_at == stamp and game.actual_attendee_ids == attendees and existing and existing.room_id == room_id:
            return None
        raise ValueError("game already has a different actual start record")
    if game.status != GameStatus.READY or game.remaining_seats() != 0:
        raise ValueError("only a full ready game can start")
    if stamp < (game.planned_start_at or game.created_at) or (game.expires_at and stamp >= game.expires_at):
        raise ValueError("game cannot start outside its start window")
    committed_ids = {
        customer_id
        for claim in game.seat_claims()
        for customer_id in (claim["contact_id"], *claim["known_member_ids"])
        if customer_id
    }
    if len(attendees) != game.seats_total or len(set(attendees)) != game.seats_total or set(attendees) != committed_ids:
        raise ValueError("actual attendees must be distinct identified occupants of all committed seats")

    end = stamp + timedelta(hours=duration_hours_from_requirement(game.requirement))
    for other in games:
        if other.game_id == game.game_id or other.status != GameStatus.READY:
            continue
        if not committed_ids.intersection(other.actual_attendee_ids or active_game_participant_ids(other)):
            continue
        other_start, other_end = game_commitment_window(other)
        if other.is_playing or (stamp < other_end and other_start < end):
            raise ValueError(f"attendees already committed to overlapping game: {other.game_id}")

    if room_id not in room_ids:
        raise ValueError(f"room is not configured: {room_id}")
    # 已经获准对参加者说明的房间必须沿用；换房不能借开打事件静默发生。
    assigned = [
        reservation for reservation in reservations_by_id.values()
        if reservation.game_id == game.game_id and reservation.public_room_label is not None
        and reservation.status == "confirmed"
    ]
    if (len(assigned) > 1 or any(reservation.room_id != room_id or reservation.conversation_id != game.conversation_id
                                 or not reservation.start_at <= stamp < reservation.end_at for reservation in assigned)):
        raise ValueError("actual game start must use its current assigned room")
    overlaps = [
        reservation for reservation in reservations_by_id.values()
        if reservation.room_id == room_id and room_reservation_blocks_window(reservation, start=stamp, end=end, at=stamp)
    ]
    if len(overlaps) > 1 or any(reservation.game_id != game.game_id or reservation.status == "occupied" for reservation in overlaps):
        raise ValueError(f"room is unavailable: {room_id}")
    previous = overlaps[0] if overlaps else None
    if previous is None:
        reservation = RoomReservation(
            reservation_id=new_id("room_reservation"), room_id=room_id, conversation_id=game.conversation_id,
            game_id=game.game_id, start_at=stamp, end_at=end, status="occupied", source_trace_id=trace_id,
            created_at=stamp, updated_at=stamp,
        )
    else:
        reservation = replace(previous, start_at=stamp, end_at=end, status="occupied", updated_at=stamp)
    updated = replace(
        game, actual_started_at=stamp, expected_play_end_at=end, actual_attendee_ids=attendees,
        play_reservation_id=reservation.reservation_id, updated_at=stamp,
    )
    transitions = [
        StateTransition("room_reservation", reservation.reservation_id, previous.status if previous else None,
                        "occupied", "game_play_started", trace_id, occurred_at=stamp),
        StateTransition("game_play", game.game_id, "ready", "playing", "game_play_started", trace_id, occurred_at=stamp),
    ]
    return updated, reservation, transitions


def prepare_game_play_finish(game: Game, *, at: datetime, trace_id: str) -> tuple[Game, list[StateTransition]] | None:
    """仅在实际开局且约定时长已到时办结，保存散场发生的确切时间。"""
    stamp = normalize_datetime(at).astimezone(DEFAULT_TZ)
    if game.actual_ended_at is not None:
        if game.actual_ended_at == stamp:
            return None
        raise ValueError("game already has a different actual finish record")
    if not game.is_playing or game.status != GameStatus.READY or game.expected_play_end_at is None:
        raise ValueError("game has no actual start to finish")
    if stamp < game.expected_play_end_at:
        raise ValueError("game cannot finish before its agreed play duration elapsed")
    updated = replace(game, status=GameStatus.FINISHED, actual_ended_at=stamp, closed_reason="game_play_completed", updated_at=stamp)
    apply_game_recruitment_policy(updated, at=stamp)
    return updated, [
        StateTransition("game_play", game.game_id, "playing", "finished", updated.closed_reason, trace_id, occurred_at=stamp),
        StateTransition("game", game.game_id, game.status.value, updated.status.value, updated.closed_reason, trace_id, occurred_at=stamp),
    ]
