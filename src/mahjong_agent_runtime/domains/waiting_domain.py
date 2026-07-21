"""Domain constants for durable waiting-list maintenance."""

from __future__ import annotations

from datetime import datetime, timedelta

from ..models import DEFAULT_TZ


WAITING_DEMAND_EXPIRY_TASK_TYPE = "expire_waiting_demands"
WAITING_DEMAND_EXPIRY_INTERVAL_SECONDS = 60


def next_waiting_expiry_due(at: datetime) -> datetime:
    """Return the next minute boundary so restarts converge on one task ID."""

    stamp = at if at.tzinfo is not None else at.replace(tzinfo=DEFAULT_TZ)
    return stamp.replace(second=0, microsecond=0) + timedelta(minutes=1)


def waiting_expiry_task_id(due_at: datetime) -> str:
    stamp = due_at if due_at.tzinfo is not None else due_at.replace(tzinfo=DEFAULT_TZ)
    return f"waiting-demand-expiry:{stamp.strftime('%Y%m%d%H%M')}"


__all__ = [
    "WAITING_DEMAND_EXPIRY_INTERVAL_SECONDS",
    "WAITING_DEMAND_EXPIRY_TASK_TYPE",
    "next_waiting_expiry_due",
    "waiting_expiry_task_id",
]
