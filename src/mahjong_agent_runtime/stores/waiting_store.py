"""Passive waiting-demand persistence contract."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from ..models import WaitingDemand, WaitingDemandStatus


class WaitingDemandStore(Protocol):
    """Persistence operations for reverse-triggered demand matching."""

    @property
    def waiting_demands(self) -> dict[str, WaitingDemand]: ...

    def waiting_demand(self, demand_id: str) -> WaitingDemand | None: ...

    def insert_waiting_demand(self, demand: WaitingDemand) -> str: ...

    def list_active_demands(self, *, at: datetime | None = None) -> list[WaitingDemand]: ...

    def update_demand_status(
        self,
        demand_id: str,
        status: WaitingDemandStatus | str,
        matched_game_id: str | None = None,
    ) -> WaitingDemand: ...

    def claim_waiting_demand_match(
        self,
        demand_id: str,
        game_id: str,
        *,
        at: datetime | None = None,
    ) -> WaitingDemand | None: ...

    def release_waiting_demand_match(self, demand_id: str, game_id: str) -> WaitingDemand | None: ...

    def cancel_waiting_demands(
        self,
        *,
        conversation_id: str,
        sender_id: str,
        demand_id: str | None = None,
    ) -> list[WaitingDemand]: ...

    def expire_stale_demands(
        self,
        *,
        at: datetime | None = None,
        trace_id: str | None = None,
    ) -> list[WaitingDemand]: ...


__all__ = ["WaitingDemandStore"]
