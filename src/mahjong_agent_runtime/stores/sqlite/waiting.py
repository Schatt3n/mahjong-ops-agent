"""SQLite waiting-demand operations."""

from __future__ import annotations

from datetime import datetime
import sqlite3

from ...models import WaitingDemand, WaitingDemandStatus, now
from .serialization import _dumps, _loads


class SQLiteWaitingDemandStoreMixin:
    """Persist and atomically claim passive matching requests."""

    __slots__ = ()

    @property
    def waiting_demands(self) -> dict[str, WaitingDemand]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM waiting_demands ORDER BY created_at, id"
            ).fetchall()
            return {
                str(row["id"]): _waiting_demand_from_row(row)
                for row in rows
            }

    def waiting_demand(self, demand_id: str) -> WaitingDemand | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM waiting_demands WHERE id = ?",
                (str(demand_id),),
            ).fetchone()
            return _waiting_demand_from_row(row) if row is not None else None

    def insert_waiting_demand(self, demand: WaitingDemand) -> str:
        with self._write_transaction():
            try:
                self._connection.execute(
                    """
                    INSERT INTO waiting_demands(
                        id, conversation_id, sender_id, sender_name, demand,
                        status, created_at, expires_at, matched_game_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        demand.demand_id,
                        demand.conversation_id,
                        demand.sender_id,
                        demand.sender_name,
                        _dumps(demand.demand),
                        demand.status.value,
                        demand.created_at.isoformat(),
                        demand.expires_at.isoformat(),
                        demand.matched_game_id,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"waiting demand already exists: {demand.demand_id}") from exc
        return demand.demand_id

    def list_active_demands(self, *, at: datetime | None = None) -> list[WaitingDemand]:
        stamp = at or now()
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM waiting_demands
                WHERE status = ? AND expires_at > ?
                ORDER BY created_at, id
                """,
                (WaitingDemandStatus.ACTIVE.value, stamp.isoformat()),
            ).fetchall()
            return [_waiting_demand_from_row(row) for row in rows]

    def update_demand_status(
        self,
        demand_id: str,
        status: WaitingDemandStatus | str,
        matched_game_id: str | None = None,
    ) -> WaitingDemand:
        target = status if isinstance(status, WaitingDemandStatus) else WaitingDemandStatus(str(status))
        with self._write_transaction():
            row = self._connection.execute(
                "SELECT * FROM waiting_demands WHERE id = ?",
                (str(demand_id),),
            ).fetchone()
            if row is None:
                raise ValueError(f"waiting demand not found: {demand_id}")
            final_game_id = matched_game_id if matched_game_id is not None else row["matched_game_id"]
            self._connection.execute(
                "UPDATE waiting_demands SET status = ?, matched_game_id = ? WHERE id = ?",
                (target.value, final_game_id, str(demand_id)),
            )
            updated = self._connection.execute(
                "SELECT * FROM waiting_demands WHERE id = ?",
                (str(demand_id),),
            ).fetchone()
            return _waiting_demand_from_row(updated)

    def claim_waiting_demand_match(
        self,
        demand_id: str,
        game_id: str,
        *,
        at: datetime | None = None,
    ) -> WaitingDemand | None:
        stamp = at or now()
        with self._write_transaction():
            cursor = self._connection.execute(
                """
                UPDATE waiting_demands
                SET status = ?, matched_game_id = ?
                WHERE id = ? AND status = ? AND expires_at > ?
                """,
                (
                    WaitingDemandStatus.MATCHED.value,
                    str(game_id),
                    str(demand_id),
                    WaitingDemandStatus.ACTIVE.value,
                    stamp.isoformat(),
                ),
            )
            if cursor.rowcount != 1:
                return None
            row = self._connection.execute(
                "SELECT * FROM waiting_demands WHERE id = ?",
                (str(demand_id),),
            ).fetchone()
            return _waiting_demand_from_row(row)

    def release_waiting_demand_match(self, demand_id: str, game_id: str) -> WaitingDemand | None:
        """Release only the exact claim that failed to dispatch."""

        with self._write_transaction():
            cursor = self._connection.execute(
                """
                UPDATE waiting_demands
                SET status = ?, matched_game_id = NULL
                WHERE id = ? AND status = ? AND matched_game_id = ?
                """,
                (
                    WaitingDemandStatus.ACTIVE.value,
                    str(demand_id),
                    WaitingDemandStatus.MATCHED.value,
                    str(game_id),
                ),
            )
            if cursor.rowcount != 1:
                return None
            row = self._connection.execute(
                "SELECT * FROM waiting_demands WHERE id = ?",
                (str(demand_id),),
            ).fetchone()
            return _waiting_demand_from_row(row)

    def cancel_waiting_demands(
        self,
        *,
        conversation_id: str,
        sender_id: str,
        demand_id: str | None = None,
    ) -> list[WaitingDemand]:
        with self._write_transaction():
            clauses = [
                "conversation_id = ?",
                "sender_id = ?",
                "status IN (?, ?)",
            ]
            parameters: list[str] = [
                str(conversation_id),
                str(sender_id),
                WaitingDemandStatus.ACTIVE.value,
                WaitingDemandStatus.MATCHED.value,
            ]
            if demand_id:
                clauses.append("id = ?")
                parameters.append(str(demand_id))
            where = " AND ".join(clauses)
            rows = self._connection.execute(
                f"SELECT * FROM waiting_demands WHERE {where}",
                tuple(parameters),
            ).fetchall()
            ids = [str(row["id"]) for row in rows]
            if ids:
                placeholders = ",".join("?" for _ in ids)
                self._connection.execute(
                    f"UPDATE waiting_demands SET status = ? WHERE id IN ({placeholders})",
                    (WaitingDemandStatus.CANCELLED.value, *ids),
                )
            cancelled: list[WaitingDemand] = []
            for row in rows:
                item = _waiting_demand_from_row(row)
                item.status = WaitingDemandStatus.CANCELLED
                cancelled.append(item)
            return cancelled

    def expire_stale_demands(
        self,
        *,
        at: datetime | None = None,
        trace_id: str | None = None,
    ) -> list[WaitingDemand]:
        del trace_id
        stamp = at or now()
        with self._write_transaction():
            rows = self._connection.execute(
                """
                SELECT * FROM waiting_demands
                WHERE status IN (?, ?) AND expires_at <= ?
                ORDER BY created_at, id
                """,
                (
                    WaitingDemandStatus.ACTIVE.value,
                    WaitingDemandStatus.MATCHED.value,
                    stamp.isoformat(),
                ),
            ).fetchall()
            ids = [str(row["id"]) for row in rows]
            if ids:
                placeholders = ",".join("?" for _ in ids)
                self._connection.execute(
                    f"UPDATE waiting_demands SET status = ? WHERE id IN ({placeholders})",
                    (WaitingDemandStatus.EXPIRED.value, *ids),
                )
            expired: list[WaitingDemand] = []
            for row in rows:
                item = _waiting_demand_from_row(row)
                item.status = WaitingDemandStatus.EXPIRED
                expired.append(item)
            return expired


def _waiting_demand_from_row(row: sqlite3.Row) -> WaitingDemand:
    return WaitingDemand(
        demand_id=str(row["id"]),
        conversation_id=str(row["conversation_id"]),
        sender_id=str(row["sender_id"]),
        sender_name=str(row["sender_name"] or ""),
        demand=dict(_loads(row["demand"])),
        status=WaitingDemandStatus(str(row["status"])),
        created_at=datetime.fromisoformat(str(row["created_at"])),
        expires_at=datetime.fromisoformat(str(row["expires_at"])),
        matched_game_id=str(row["matched_game_id"] or "") or None,
    )


__all__ = ["SQLiteWaitingDemandStoreMixin"]
