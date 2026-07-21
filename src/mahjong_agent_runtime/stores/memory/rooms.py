"""InMemory rooms store operations."""

from __future__ import annotations

from typing import Any
from ...models import (
    RoomReservation,
    StateTransition,
    new_id,
    now,
)
from ...domains import parse_datetime_value

class InMemoryRoomsStoreMixin:
    """Backend-specific operations extracted from the compatibility store."""

    __slots__ = ()

    def configure_rooms(self, room_ids: list[str]) -> None:
        with self._lock:
            self.room_ids = list(dict.fromkeys(str(item).strip() for item in room_ids if str(item).strip()))

    def search_room_availability(self, *, start_at: Any, end_at: Any) -> dict[str, Any]:
        start = parse_datetime_value(start_at)
        end = parse_datetime_value(end_at)
        if start is None or end is None or end <= start:
            raise ValueError("start_at and end_at must be valid datetimes with end_at after start_at")
        with self._lock:
            occupied = {
                item.room_id
                for item in self.room_reservations.values()
                if item.status in {"held", "confirmed"} and item.start_at < end and item.end_at > start
            }
            available = [room_id for room_id in self.room_ids if room_id not in occupied]
            return {
                "configured": bool(self.room_ids),
                "start_at": start.isoformat(),
                "end_at": end.isoformat(),
                "room_count": len(self.room_ids),
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
        availability = self.search_room_availability(start_at=start_at, end_at=end_at)
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
            start_at=parse_datetime_value(start_at) or now(),
            end_at=parse_datetime_value(end_at) or now(),
            source_trace_id=trace_id,
        )
        transition = StateTransition(
            "room_reservation",
            reservation.reservation_id,
            None,
            reservation.status,
            "reserve_room",
            trace_id,
        )
        with self._lock:
            # Recheck under the mutation lock to avoid two local callers taking
            # the same room after a shared availability snapshot.
            latest = self.search_room_availability(start_at=start_at, end_at=end_at)
            if chosen not in latest["available_room_ids"]:
                raise ValueError(f"room is unavailable: {chosen}")
            self.room_reservations[reservation.reservation_id] = reservation
            self.transitions.append(transition)
        return reservation, transition

    def _release_room_reservations_for_game_locked(
        self,
        game_id: str,
        *,
        trace_id: str,
        reason: str,
    ) -> list[StateTransition]:
        transitions: list[StateTransition] = []
        for reservation in self.room_reservations.values():
            if reservation.game_id != game_id or reservation.status not in {"held", "confirmed"}:
                continue
            old = reservation.status
            reservation.status = "released"
            reservation.updated_at = now()
            transitions.append(
                StateTransition(
                    "room_reservation",
                    reservation.reservation_id,
                    old,
                    reservation.status,
                    reason,
                    trace_id,
                )
            )
        return transitions
