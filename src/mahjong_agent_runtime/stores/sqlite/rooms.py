"""实现 SQLite 存储层的rooms数据访问与事务操作。

"""

from __future__ import annotations

from typing import Any
from datetime import datetime
from ...models import (
    RoomReservation,
    StateTransition,
    new_id,
    now,
)
from ...domains import parse_datetime_value
from ...domains.room_assignment import prepare_game_room_assignment
from .serialization import (
    _dumps,
    _loads,
    _now_iso,
    _room_reservation_from_payload,
)

class SQLiteRoomsStoreMixin:
    """从兼容性存储中提取的后端特定操作。"""

    __slots__ = ()

    def configure_rooms(self, room_ids: list[str]) -> None:
        normalized = list(dict.fromkeys(str(item).strip() for item in room_ids if str(item).strip()))
        with self._lock, self._connection:
            self._connection.execute("DELETE FROM runtime_rooms")
            self._connection.executemany(
                "INSERT INTO runtime_rooms(room_id, updated_at) VALUES (?, ?)",
                [(room_id, _now_iso()) for room_id in normalized],
            )

    def search_room_availability(self, *, start_at: Any, end_at: Any) -> dict[str, Any]:
        """查询房间可用性；只返回满足当前作用域和约束的结果。"""
        start = parse_datetime_value(start_at)
        end = parse_datetime_value(end_at)
        if start is None or end is None or end <= start:
            raise ValueError("start_at and end_at must be valid datetimes with end_at after start_at")
        with self._lock:
            room_ids = [
                str(row["room_id"])
                for row in self._connection.execute("SELECT room_id FROM runtime_rooms ORDER BY room_id").fetchall()
            ]
            occupied = {
                str(row["room_id"])
                for row in self._connection.execute(
                    """
                    SELECT DISTINCT room_id
                    FROM runtime_room_reservations
                    WHERE status IN ('held', 'confirmed', 'occupied') AND start_at < ?
                        AND (end_at > ? OR (status = 'occupied' AND end_at <= ?))
                    """,
                    (end.isoformat(), start.isoformat(), now().isoformat()),
                ).fetchall()
            }
            available = [room_id for room_id in room_ids if room_id not in occupied]
            return {
                "configured": bool(room_ids),
                "start_at": start.isoformat(),
                "end_at": end.isoformat(),
                "room_count": len(room_ids),
                "available_room_ids": available,
                "occupied_room_ids": sorted(occupied),
                "available_count": len(available),
            }

    def reserve_room(
        self,
        *,
        conversation_id: str,
        game_id: str | None,
        start_at: Any,
        end_at: Any,
        room_id: str | None,
        trace_id: str,
    ) -> tuple[RoomReservation, StateTransition]:
        start = parse_datetime_value(start_at)
        end = parse_datetime_value(end_at)
        if start is None or end is None or end <= start:
            raise ValueError("start_at and end_at must be valid datetimes with end_at after start_at")
        with self._write_transaction():
            availability = self.search_room_availability(start_at=start, end_at=end)
            if not availability["configured"]:
                raise ValueError("room inventory is not configured")
            chosen = str(room_id or "").strip()
            available = list(availability["available_room_ids"])
            if chosen and chosen not in available:
                raise ValueError(f"room is unavailable: {chosen}")
            if not chosen:
                if not available:
                    raise ValueError("no room is available for the requested interval")
                chosen = available[0]
            reservation = RoomReservation(
                reservation_id=new_id("room_reservation"),
                room_id=chosen,
                conversation_id=conversation_id,
                game_id=game_id,
                start_at=start,
                end_at=end,
                source_trace_id=trace_id,
            )
            self._save_room_reservation(reservation)
            transition = StateTransition(
                "room_reservation",
                reservation.reservation_id,
                None,
                reservation.status,
                "reserve_room",
                trace_id,
            )
            self._append_transition(transition)
            return reservation, transition

    def assign_game_room(
        self, game_id: str, *, room_id: str, public_room_label: str, at: datetime, trace_id: str,
    ) -> tuple[RoomReservation, list[StateTransition]]:
        """同一写事务核对实时房态和本局条件，防止两个店员重复分房或占用同一间。"""
        with self._write_transaction():
            reservation, transitions = prepare_game_room_assignment(
                self.require_game(game_id), room_id=room_id, public_room_label=public_room_label, at=at,
                trace_id=trace_id, room_ids=self.room_ids, reservations=self.room_reservations.values(),
            )
            if transitions:
                self._save_room_reservation(reservation)
                self._append_transition(transitions[0])
            return reservation, transitions

    def _save_room_reservation(self, reservation: RoomReservation) -> None:
        """在调用方写事务中同步房态索引与预约事实，保留原预约标识。"""
        self._connection.execute(
            """
            INSERT INTO runtime_room_reservations(
                reservation_id, room_id, conversation_id, game_id, start_at, end_at, status, payload, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(reservation_id) DO UPDATE SET
                start_at=excluded.start_at, end_at=excluded.end_at, status=excluded.status,
                payload=excluded.payload, updated_at=excluded.updated_at
            """,
            (
                reservation.reservation_id, reservation.room_id, reservation.conversation_id,
                reservation.game_id or "", reservation.start_at.isoformat(), reservation.end_at.isoformat(),
                reservation.status, _dumps(reservation.to_dict()), reservation.updated_at.isoformat(),
            ),
        )

    def _release_room_reservations_for_game(
        self,
        game_id: str,
        *,
        trace_id: str,
        reason: str,
        at: datetime | None = None,
    ) -> list[StateTransition]:
        """在调用方事务内释放局所占用的房间，实际散场沿用其可信事件时间。"""
        rows = self._connection.execute(
            """
            SELECT payload
            FROM runtime_room_reservations
            WHERE game_id = ? AND status IN ('held', 'confirmed', 'occupied')
            """,
            (game_id,),
        ).fetchall()
        transitions: list[StateTransition] = []
        stamp = at or now()
        for row in rows:
            reservation = _room_reservation_from_payload(_loads(row["payload"]))
            old = reservation.status
            reservation.status = "released"
            reservation.updated_at = stamp
            self._save_room_reservation(reservation)
            transition = StateTransition(
                "room_reservation",
                reservation.reservation_id,
                old,
                reservation.status,
                reason,
                trace_id,
                occurred_at=stamp,
            )
            transitions.append(transition)
            self._append_transition(transition)
        return transitions
