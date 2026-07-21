"""SQLite rooms store operations."""

from __future__ import annotations

from typing import Any
from datetime import datetime
from ...models import (
    DEFAULT_TZ,
    RoomReservation,
    StateTransition,
    new_id,
)
from ...store import parse_datetime_value
from .serialization import (
    _dumps,
    _loads,
    _now_iso,
    _room_reservation_from_payload,
)

class SQLiteRoomsStoreMixin:
    """Backend-specific operations extracted from the compatibility store."""

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
                    WHERE status IN ('held', 'confirmed') AND start_at < ? AND end_at > ?
                    """,
                    (end.isoformat(), start.isoformat()),
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
            self._connection.execute(
                """
                INSERT INTO runtime_room_reservations(
                    reservation_id, room_id, conversation_id, game_id, start_at, end_at, status, payload, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    reservation.reservation_id,
                    reservation.room_id,
                    reservation.conversation_id,
                    reservation.game_id or "",
                    reservation.start_at.isoformat(),
                    reservation.end_at.isoformat(),
                    reservation.status,
                    _dumps(reservation.to_dict()),
                    reservation.updated_at.isoformat(),
                ),
            )
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

    def _release_room_reservations_for_game(
        self,
        game_id: str,
        *,
        trace_id: str,
        reason: str,
    ) -> list[StateTransition]:
        rows = self._connection.execute(
            """
            SELECT payload
            FROM runtime_room_reservations
            WHERE game_id = ? AND status IN ('held', 'confirmed')
            """,
            (game_id,),
        ).fetchall()
        transitions: list[StateTransition] = []
        for row in rows:
            reservation = _room_reservation_from_payload(_loads(row["payload"]))
            old = reservation.status
            reservation.status = "released"
            reservation.updated_at = datetime.now(DEFAULT_TZ)
            self._connection.execute(
                """
                UPDATE runtime_room_reservations
                SET status = ?, payload = ?, updated_at = ?
                WHERE reservation_id = ?
                """,
                (
                    reservation.status,
                    _dumps(reservation.to_dict()),
                    reservation.updated_at.isoformat(),
                    reservation.reservation_id,
                ),
            )
            transition = StateTransition(
                "room_reservation",
                reservation.reservation_id,
                old,
                reservation.status,
                reason,
                trace_id,
            )
            transitions.append(transition)
            self._append_transition(transition)
        return transitions
