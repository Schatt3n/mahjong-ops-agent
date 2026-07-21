"""SQLite input aggregation store operations."""

from __future__ import annotations

from typing import Any
from datetime import datetime
from datetime import timedelta
from ...models import (
    PendingInputBatch,
    PendingInputBatchStatus,
    StateTransition,
    new_id,
    now,
)
from ...domains import (
    PENDING_INPUT_PROCESSING_LEASE_SECONDS,
    pending_input_batch_key,
)
from .serialization import (
    _dumps,
    _loads,
    _pending_input_batch_from_payload,
)

class SQLiteInputAggregationStoreMixin:
    """Backend-specific operations extracted from the compatibility store."""

    __slots__ = ()

    def upsert_pending_input_fragment(
        self,
        message,
        *,
        trace_id: str,
        quiet_deadline: datetime,
    ) -> tuple[PendingInputBatch, StateTransition | None, bool]:
        """Persist one fragment and reset the quiet deadline in one transaction."""

        key = pending_input_batch_key(message.conversation_id, message.sender_id)
        fragment = message.to_dict()
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT payload FROM runtime_pending_input_batches WHERE batch_key = ?",
                (key,),
            ).fetchone()
            existing = _pending_input_batch_from_payload(_loads(row["payload"])) if row else None
            message_id = str(fragment.get("message_id") or "")
            if existing is not None and message_id and any(
                str(item.get("message_id") or "") == message_id for item in existing.fragments
            ):
                return existing, None, False
            if existing is None or existing.status in {
                PendingInputBatchStatus.COMPLETED,
                PendingInputBatchStatus.IGNORED,
                PendingInputBatchStatus.FAILED,
            }:
                batch = PendingInputBatch(
                    batch_id=new_id("input_batch"),
                    conversation_id=message.conversation_id,
                    sender_id=message.sender_id,
                    sender_name=message.sender_name,
                    fragments=[fragment],
                    quiet_deadline=quiet_deadline,
                    source_channel=str(message.metadata.get("channel") or ""),
                )
                previous_status = "absent"
            else:
                batch = existing
                previous_status = batch.status.value
                batch.fragments.append(fragment)
                batch.sender_name = message.sender_name or batch.sender_name
                batch.version += 1
                batch.status = PendingInputBatchStatus.PENDING
                batch.quiet_deadline = quiet_deadline
                batch.updated_at = now()
                batch.decision = {}
                if message.metadata.get("channel"):
                    batch.source_channel = str(message.metadata["channel"])
            self._save_pending_input_batch(batch)
            transition = StateTransition(
                "pending_input_batch",
                batch.batch_id,
                previous_status,
                batch.status.value,
                "input_fragment_buffered",
                trace_id,
            )
            self._append_transition(transition)
            return batch, transition, True

    def pending_input_batch(self, conversation_id: str, sender_id: str) -> PendingInputBatch | None:
        key = pending_input_batch_key(conversation_id, sender_id)
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM runtime_pending_input_batches WHERE batch_key = ?",
                (key,),
            ).fetchone()
            return _pending_input_batch_from_payload(_loads(row["payload"])) if row else None

    def due_pending_input_batches(self, *, at: datetime, limit: int = 100) -> list[PendingInputBatch]:
        with self._lock:
            stale_before = at - timedelta(seconds=PENDING_INPUT_PROCESSING_LEASE_SECONDS)
            rows = self._connection.execute(
                """
                SELECT payload FROM runtime_pending_input_batches
                WHERE (status = ? AND quiet_deadline <= ?)
                   OR (status = ? AND updated_at <= ?)
                ORDER BY quiet_deadline ASC
                LIMIT ?
                """,
                (
                    PendingInputBatchStatus.PENDING.value,
                    at.isoformat(),
                    PendingInputBatchStatus.PROCESSING.value,
                    stale_before.isoformat(),
                    max(1, int(limit)),
                ),
            ).fetchall()
            return [_pending_input_batch_from_payload(_loads(row["payload"])) for row in rows]

    def claim_pending_input_batch(
        self,
        *,
        batch_id: str,
        expected_version: int,
        trace_id: str,
    ) -> tuple[PendingInputBatch | None, StateTransition | None]:
        """Compare-and-set claim used by immediate and delayed dispatchers."""

        with self._lock, self._connection:
            row = self._connection.execute(
                """
                SELECT payload FROM runtime_pending_input_batches
                WHERE batch_id = ? AND version = ?
                """,
                (batch_id, int(expected_version)),
            ).fetchone()
            if row is None:
                return None, None
            batch = _pending_input_batch_from_payload(_loads(row["payload"]))
            stale_before = now() - timedelta(seconds=PENDING_INPUT_PROCESSING_LEASE_SECONDS)
            if batch.status != PendingInputBatchStatus.PENDING and not (
                batch.status == PendingInputBatchStatus.PROCESSING and batch.updated_at <= stale_before
            ):
                return None, None
            old = batch.status.value
            batch.status = PendingInputBatchStatus.PROCESSING
            batch.updated_at = now()
            cursor = self._connection.execute(
                """
                UPDATE runtime_pending_input_batches
                SET status = ?, payload = ?, updated_at = ?
                WHERE batch_id = ? AND version = ? AND status = ? AND updated_at = ?
                """,
                (
                    batch.status.value,
                    _dumps(batch.to_dict()),
                    batch.updated_at.isoformat(),
                    batch_id,
                    int(expected_version),
                    old,
                    _pending_input_batch_from_payload(_loads(row["payload"])).updated_at.isoformat(),
                ),
            )
            if cursor.rowcount != 1:
                return None, None
            transition = StateTransition(
                "pending_input_batch",
                batch.batch_id,
                old,
                batch.status.value,
                "input_batch_claimed",
                trace_id,
            )
            self._append_transition(transition)
            return batch, transition

    def record_pending_input_decision(
        self,
        *,
        batch_id: str,
        expected_version: int,
        decision: dict[str, Any],
    ) -> PendingInputBatch | None:
        """Persist the model boundary decision while keeping the batch pending."""

        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT payload FROM runtime_pending_input_batches WHERE batch_id = ? AND version = ?",
                (batch_id, int(expected_version)),
            ).fetchone()
            if row is None:
                return None
            batch = _pending_input_batch_from_payload(_loads(row["payload"]))
            batch.decision = dict(decision)
            batch.updated_at = now()
            self._save_pending_input_batch(batch)
            return batch

    def finish_pending_input_batch(
        self,
        *,
        batch_id: str,
        expected_version: int,
        status: PendingInputBatchStatus,
        trace_id: str,
        decision: dict[str, Any] | None = None,
    ) -> tuple[PendingInputBatch | None, StateTransition | None]:
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT payload FROM runtime_pending_input_batches WHERE batch_id = ? AND version = ?",
                (batch_id, int(expected_version)),
            ).fetchone()
            if row is None:
                return None, None
            batch = _pending_input_batch_from_payload(_loads(row["payload"]))
            old = batch.status.value
            batch.status = status
            batch.decision = dict(decision or {})
            batch.updated_at = now()
            self._save_pending_input_batch(batch)
            transition = StateTransition(
                "pending_input_batch",
                batch.batch_id,
                old,
                status.value,
                "input_batch_finished",
                trace_id,
            )
            self._append_transition(transition)
            return batch, transition
