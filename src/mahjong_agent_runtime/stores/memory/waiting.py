"""In-memory waiting-demand operations."""

from __future__ import annotations

from datetime import datetime

from ...models import WaitingDemand, WaitingDemandStatus, now


class InMemoryWaitingDemandStoreMixin:
    """Keep waiting-list transitions atomic under the aggregate store lock."""

    __slots__ = ()

    def waiting_demand(self, demand_id: str) -> WaitingDemand | None:
        with self._lock:
            return self.waiting_demands.get(str(demand_id))

    def insert_waiting_demand(self, demand: WaitingDemand) -> str:
        with self._lock:
            if demand.demand_id in self.waiting_demands:
                raise ValueError(f"waiting demand already exists: {demand.demand_id}")
            self.waiting_demands[demand.demand_id] = demand
            return demand.demand_id

    def list_active_demands(self, *, at: datetime | None = None) -> list[WaitingDemand]:
        stamp = at or now()
        with self._lock:
            return sorted(
                (
                    item
                    for item in self.waiting_demands.values()
                    if item.status == WaitingDemandStatus.ACTIVE and item.expires_at > stamp
                ),
                key=lambda item: (item.created_at, item.demand_id),
            )

    def update_demand_status(
        self,
        demand_id: str,
        status: WaitingDemandStatus | str,
        matched_game_id: str | None = None,
    ) -> WaitingDemand:
        with self._lock:
            demand = self.waiting_demands.get(str(demand_id))
            if demand is None:
                raise ValueError(f"waiting demand not found: {demand_id}")
            demand.status = status if isinstance(status, WaitingDemandStatus) else WaitingDemandStatus(str(status))
            if matched_game_id is not None:
                demand.matched_game_id = str(matched_game_id)
            return demand

    def claim_waiting_demand_match(
        self,
        demand_id: str,
        game_id: str,
        *,
        at: datetime | None = None,
    ) -> WaitingDemand | None:
        stamp = at or now()
        with self._lock:
            demand = self.waiting_demands.get(str(demand_id))
            if (
                demand is None
                or demand.status != WaitingDemandStatus.ACTIVE
                or demand.expires_at <= stamp
            ):
                return None
            demand.status = WaitingDemandStatus.MATCHED
            demand.matched_game_id = str(game_id)
            return demand

    def release_waiting_demand_match(self, demand_id: str, game_id: str) -> WaitingDemand | None:
        """Return a failed dispatch claim to the active queue."""

        with self._lock:
            demand = self.waiting_demands.get(str(demand_id))
            if (
                demand is None
                or demand.status != WaitingDemandStatus.MATCHED
                or demand.matched_game_id != str(game_id)
            ):
                return None
            demand.status = WaitingDemandStatus.ACTIVE
            demand.matched_game_id = None
            return demand

    def cancel_waiting_demands(
        self,
        *,
        conversation_id: str,
        sender_id: str,
        demand_id: str | None = None,
    ) -> list[WaitingDemand]:
        with self._lock:
            cancelled: list[WaitingDemand] = []
            for demand in self.waiting_demands.values():
                if demand.conversation_id != conversation_id or demand.sender_id != sender_id:
                    continue
                if demand_id and demand.demand_id != demand_id:
                    continue
                if demand.status not in {WaitingDemandStatus.ACTIVE, WaitingDemandStatus.MATCHED}:
                    continue
                demand.status = WaitingDemandStatus.CANCELLED
                cancelled.append(demand)
            return cancelled

    def expire_stale_demands(
        self,
        *,
        at: datetime | None = None,
        trace_id: str | None = None,
    ) -> list[WaitingDemand]:
        del trace_id
        stamp = at or now()
        with self._lock:
            expired: list[WaitingDemand] = []
            for demand in self.waiting_demands.values():
                if (
                    demand.status in {WaitingDemandStatus.ACTIVE, WaitingDemandStatus.MATCHED}
                    and demand.expires_at <= stamp
                ):
                    demand.status = WaitingDemandStatus.EXPIRED
                    expired.append(demand)
            return expired


__all__ = ["InMemoryWaitingDemandStoreMixin"]
