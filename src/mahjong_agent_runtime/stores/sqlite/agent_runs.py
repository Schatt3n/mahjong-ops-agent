"""SQLite checkpoints and lease claims for resumable Agent runs."""

from __future__ import annotations

from datetime import datetime

from ...agent_state import AgentRunState, AgentRunStatus
from ...models import StateTransition, now
from .serialization import _dumps, _loads


class SQLiteAgentRunStoreMixin:
    """Persist one compact checkpoint after every completed loop step."""

    __slots__ = ()

    def create_agent_run(self, state: AgentRunState) -> AgentRunState:
        state.updated_at = now()
        with self._write_transaction():
            self._connection.execute(
                """
                INSERT INTO runtime_agent_runs(
                    run_id, conversation_id, run_version, status,
                    lease_owner, lease_until, payload, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                _run_row_values(state),
            )
        return AgentRunState.from_dict(state.to_dict())

    def agent_run(self, run_id: str) -> AgentRunState | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM runtime_agent_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        return AgentRunState.from_dict(_loads(row["payload"])) if row else None

    def save_agent_run(
        self,
        state: AgentRunState,
        *,
        expected_lease_owner: str | None = None,
    ) -> bool:
        state.updated_at = now()
        with self._write_transaction():
            sql = """
                UPDATE runtime_agent_runs
                SET conversation_id = ?, run_version = ?, status = ?,
                    lease_owner = ?, lease_until = ?, payload = ?, updated_at = ?
                WHERE run_id = ?
            """
            params: tuple = (
                state.conversation_id,
                state.run_version,
                state.status.value,
                state.lease_owner,
                state.lease_until.isoformat() if state.lease_until else None,
                _dumps(state.to_dict()),
                state.updated_at.isoformat(),
                state.run_id,
            )
            if expected_lease_owner is not None:
                sql += " AND lease_owner = ?"
                params = (*params, expected_lease_owner)
            cursor = self._connection.execute(sql, params)
            return int(cursor.rowcount) == 1

    def recoverable_agent_runs(
        self,
        *,
        at: datetime,
        limit: int = 100,
    ) -> list[AgentRunState]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT payload FROM runtime_agent_runs
                WHERE status = ?
                   OR (status = ? AND lease_until IS NOT NULL AND lease_until <= ?)
                ORDER BY updated_at, run_id
                LIMIT ?
                """,
                (
                    AgentRunStatus.RECOVERABLE.value,
                    AgentRunStatus.RUNNING.value,
                    at.isoformat(),
                    max(1, int(limit)),
                ),
            ).fetchall()
        return [AgentRunState.from_dict(_loads(row["payload"])) for row in rows]

    def claim_agent_run(
        self,
        run_id: str,
        *,
        lease_owner: str,
        lease_until: datetime,
        at: datetime,
    ) -> AgentRunState | None:
        with self._write_transaction():
            row = self._connection.execute(
                "SELECT payload FROM runtime_agent_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if row is None:
                return None
            state = AgentRunState.from_dict(_loads(row["payload"]))
            recoverable = state.status == AgentRunStatus.RECOVERABLE
            expired = (
                state.status == AgentRunStatus.RUNNING
                and state.lease_until is not None
                and state.lease_until <= at
            )
            if not recoverable and not expired:
                return None
            state.status = AgentRunStatus.RUNNING
            state.lease_owner = str(lease_owner)
            state.lease_until = lease_until
            state.attempts += 1
            state.updated_at = at
            self._save_agent_run_in_transaction(state)
            return state

    def supersede_agent_runs(
        self,
        conversation_id: str,
        *,
        current_version: int,
        trace_id: str,
    ) -> list[StateTransition]:
        transitions: list[StateTransition] = []
        with self._write_transaction():
            rows = self._connection.execute(
                """
                SELECT payload FROM runtime_agent_runs
                WHERE conversation_id = ?
                  AND run_version < ?
                  AND status IN (?, ?)
                """,
                (
                    conversation_id,
                    int(current_version),
                    AgentRunStatus.RUNNING.value,
                    AgentRunStatus.RECOVERABLE.value,
                ),
            ).fetchall()
            for row in rows:
                state = AgentRunState.from_dict(_loads(row["payload"]))
                old = state.status.value
                state.status = AgentRunStatus.SUPERSEDED
                state.runtime_status = AgentRunStatus.SUPERSEDED.value
                state.last_error = "newer conversation message superseded this execution"
                state.lease_owner = ""
                state.lease_until = None
                state.completed_at = now()
                state.updated_at = state.completed_at
                self._save_agent_run_in_transaction(state)
                transition = StateTransition(
                    entity_type="agent_run",
                    entity_id=state.run_id,
                    from_status=old,
                    to_status=state.status.value,
                    reason="newer_conversation_version",
                    trace_id=trace_id,
                )
                self._append_transition(transition)
                transitions.append(transition)
        return transitions

    def _save_agent_run_in_transaction(self, state: AgentRunState) -> None:
        self._connection.execute(
            """
            UPDATE runtime_agent_runs
            SET conversation_id = ?, run_version = ?, status = ?,
                lease_owner = ?, lease_until = ?, payload = ?, updated_at = ?
            WHERE run_id = ?
            """,
            (
                state.conversation_id,
                state.run_version,
                state.status.value,
                state.lease_owner,
                state.lease_until.isoformat() if state.lease_until else None,
                _dumps(state.to_dict()),
                state.updated_at.isoformat(),
                state.run_id,
            ),
        )


def _run_row_values(state: AgentRunState) -> tuple:
    return (
        state.run_id,
        state.conversation_id,
        state.run_version,
        state.status.value,
        state.lease_owner,
        state.lease_until.isoformat() if state.lease_until else None,
        _dumps(state.to_dict()),
        state.updated_at.isoformat(),
    )


__all__ = ["SQLiteAgentRunStoreMixin"]
