"""SQLite administration store operations."""

from __future__ import annotations

from typing import Any
from ...models import new_id
from .serialization import (
    _dumps,
    _now_iso,
)

class SQLiteAdministrationStoreMixin:
    """Backend-specific operations extracted from the compatibility store."""

    __slots__ = ()

    def clear_runtime_state(
        self,
        *,
        include_customers: bool = False,
        include_badcases: bool = False,
    ) -> dict[str, int]:
        tables = [
            ("game_participants", "runtime_game_participants"),
            ("games", "runtime_games"),
            ("invite_drafts", "runtime_invite_drafts"),
            ("outbound_message_drafts", "runtime_outbound_message_drafts"),
            ("room_reservations", "runtime_room_reservations"),
            ("state_transitions", "runtime_state_transitions"),
            ("conversation_turns", "runtime_conversation_turns"),
            ("conversation_checkpoints", "runtime_conversation_checkpoints"),
            ("task_context_checkpoints", "runtime_task_context_checkpoints"),
            ("task_contexts", "runtime_task_contexts"),
            ("conversation_versions", "runtime_conversation_versions"),
            ("idempotency_ledger", "runtime_idempotency_ledger"),
            ("message_results", "runtime_message_results"),
            ("message_references", "runtime_message_references"),
            ("task_memories", "runtime_task_memories"),
            ("pending_memory_candidates", "runtime_pending_memory_candidates"),
            ("pending_input_batches", "runtime_pending_input_batches"),
            ("scheduled_tasks", "runtime_scheduled_agent_tasks"),
            ("agent_runs", "runtime_agent_runs"),
            ("waiting_demands", "waiting_demands"),
        ]
        if include_customers:
            tables.append(("customers", "runtime_customers"))
            tables.append(("customer_relationships", "runtime_customer_relationships"))
        else:
            tables.append(("customers", ""))
            tables.append(("customer_relationships", ""))
        if include_badcases:
            tables.append(("badcases", "runtime_badcases"))
        else:
            tables.append(("badcases", ""))
        with self._lock, self._connection:
            deleted: dict[str, int] = {}
            for key, table in tables:
                if not table:
                    deleted[key] = 0
                    continue
                row = self._connection.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()
                deleted[key] = int(row["count"] if row else 0)
            for _, table in tables:
                if table:
                    self._connection.execute(f"DELETE FROM {table}")
            return deleted

    def record_badcase(self, payload: dict[str, Any], *, trace_id: str, conversation_id: str) -> dict[str, Any]:
        with self._lock, self._connection:
            from ...models import new_id

            record = {"badcase_id": new_id("badcase"), "trace_id": trace_id, "conversation_id": conversation_id, **dict(payload)}
            self._connection.execute(
                """
                INSERT INTO runtime_badcases(badcase_id, trace_id, conversation_id, payload, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (record["badcase_id"], trace_id, conversation_id, _dumps(record), _now_iso()),
            )
            return record
