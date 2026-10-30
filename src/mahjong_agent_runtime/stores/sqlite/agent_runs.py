"""实现 SQLite 存储层的智能体runs数据访问与事务操作。

"""

from __future__ import annotations

from datetime import datetime

from ...agent_state import AgentRunState, AgentRunStatus
from ...models import StateTransition, now
from .serialization import _dumps, _loads


class SQLiteAgentRunStoreMixin:
    """在每完成一个循环步骤后持久化一个紧凑检查点。"""

    __slots__ = ()

    def has_unfinished_user_goal(self, conversation_id: str, sender_id: str) -> bool:
        """只查询当前会话版本，历史未结运行或无缓存 receipt 不会续接已经替换的目标。"""
        with self._lock:
            current_version = self.conversation_version(conversation_id)
            rows = self._connection.execute(
                "SELECT payload FROM runtime_agent_runs WHERE conversation_id = ? AND run_version = ? AND status IN (?, ?)",
                (conversation_id, current_version, AgentRunStatus.RUNNING.value, AgentRunStatus.RECOVERABLE.value),
            ).fetchall()
            if any((_loads(row["payload"]).get("message") or {}).get("sender_id") == sender_id for row in rows):
                return True
            turns = self._connection.execute(
                "SELECT payload FROM runtime_conversation_turns WHERE conversation_id = ? AND role = ?",
                (conversation_id, "user"),
            ).fetchall()
            for row in turns:
                turn = _loads(row["payload"])
                key = (turn.get("metadata") or {}).get("input_receipt_key")
                if (turn.get("sender_id") == sender_id and key
                        and (turn.get("metadata") or {}).get("conversation_version") == current_version
                        and self.idempotent_message_result(str(key)) is None):
                    return True
        return False

    def create_agent_run(self, state: AgentRunState) -> AgentRunState:
        """创建智能体run，并返回持久化后的权威结果。"""
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
        """保存智能体run，确保后续读取能够得到一致快照。"""
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
        """原子领取智能体run，防止并发消费者重复处理。"""
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
