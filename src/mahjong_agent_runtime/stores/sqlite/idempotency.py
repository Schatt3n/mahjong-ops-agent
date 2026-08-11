"""实现 SQLite 存储层的幂等数据访问与事务操作。

"""

from __future__ import annotations

from datetime import timedelta

from ...models import AgentRuntimeResult, ToolResult, now
from ..idempotency_common import IDEMPOTENCY_CLAIM_LEASE_SECONDS, tool_result_is_in_progress
from .serialization import (
    _datetime_from_payload,
    _dumps,
    _loads,
    _now_iso,
    _runtime_result_from_payload,
    _tool_result_from_payload,
)


class SQLiteIdempotencyStoreMixin:
    """使用 SQLite 唯一约束和 CAS 更新实现进程安全的认领。"""

    __slots__ = ()

    def idempotent_result(self, key: str | None) -> ToolResult | None:
        if not key:
            return None
        with self._lock:
            row = self._connection.execute(
                "SELECT payload, created_at FROM runtime_idempotency_ledger WHERE idempotency_key = ?",
                (key,),
            ).fetchone()
            if row is None:
                return None
            result = _tool_result_from_payload(_loads(row["payload"]))
            if tool_result_is_in_progress(result):
                claimed_at = _datetime_from_payload(row["created_at"])
                if claimed_at <= now() - timedelta(seconds=IDEMPOTENCY_CLAIM_LEASE_SECONDS):
                    return None
            return result

    def claim_idempotent_result(
        self,
        key: str | None,
        claimed_result: ToolResult,
    ) -> tuple[bool, ToolResult | None]:
        """原子领取idempotent结果，防止并发消费者重复处理。"""
        if not key:
            return True, None
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                INSERT INTO runtime_idempotency_ledger(idempotency_key, payload, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(idempotency_key) DO NOTHING
                """,
                (key, _dumps(claimed_result.to_dict()), _now_iso()),
            )
            if cursor.rowcount == 1:
                return True, None
            row = self._connection.execute(
                "SELECT payload, created_at FROM runtime_idempotency_ledger WHERE idempotency_key = ?",
                (key,),
            ).fetchone()
            if row is None:
                return False, None
            existing = _tool_result_from_payload(_loads(row["payload"]))
            claimed_at = _datetime_from_payload(row["created_at"])
            if tool_result_is_in_progress(existing) and claimed_at <= now() - timedelta(
                seconds=IDEMPOTENCY_CLAIM_LEASE_SECONDS
            ):
                cursor = self._connection.execute(
                    """
                    UPDATE runtime_idempotency_ledger
                    SET payload = ?, created_at = ?
                    WHERE idempotency_key = ? AND created_at = ?
                    """,
                    (_dumps(claimed_result.to_dict()), _now_iso(), key, row["created_at"]),
                )
                if cursor.rowcount == 1:
                    return True, None
            return False, existing

    def remember_result(self, key: str | None, result: ToolResult) -> None:
        if not key:
            return
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO runtime_idempotency_ledger(idempotency_key, payload, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(idempotency_key) DO UPDATE SET
                    payload=excluded.payload,
                    created_at=excluded.created_at
                """,
                (key, _dumps(result.to_dict()), _now_iso()),
            )

    def idempotent_message_result(self, message_id: str | None) -> AgentRuntimeResult | None:
        if not message_id:
            return None
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM runtime_message_results WHERE message_id = ?",
                (message_id,),
            ).fetchone()
            if row is None:
                return None
            return _runtime_result_from_payload(_loads(row["payload"]))

    def remember_message_result(self, message_id: str | None, result: AgentRuntimeResult) -> None:
        if not message_id:
            return
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO runtime_message_results(message_id, conversation_id, trace_id, payload, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(message_id) DO NOTHING
                """,
                (message_id, result.conversation_id, result.trace_id, _dumps(result.to_dict()), _now_iso()),
            )
