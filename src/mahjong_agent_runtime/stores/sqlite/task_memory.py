"""SQLite task memory store operations."""

from __future__ import annotations

from typing import Any
from ...models import (
    PendingMemoryCandidate,
    StateTransition,
    TaskMemory,
    new_id,
)
from ...store import is_avoid_playing_memory
from .serialization import (
    _loads,
    _pending_memory_candidate_from_payload,
    _task_memory_from_payload,
)

class SQLiteTaskMemoryStoreMixin:
    """Backend-specific operations extracted from the compatibility store."""

    __slots__ = ()

    def record_task_memory(
        self,
        *,
        conversation_id: str,
        customer_id: str,
        memory_type: str,
        field: str,
        value: Any,
        target_customer_id: str | None = None,
        evidence: str = "",
        confidence: float = 0.0,
        risk_level: str = "medium",
        scope: str = "current_task",
        metadata: dict[str, Any] | None = None,
        trace_id: str,
    ) -> tuple[TaskMemory, StateTransition]:
        with self._write_transaction():
            from ...models import new_id

            task_context = self.current_task_context(conversation_id, customer_id)
            memory_metadata = dict(metadata or {})
            if task_context is not None:
                memory_metadata.setdefault("task_context_id", task_context.task_context_id)
            memory = TaskMemory(
                memory_id=new_id("task_memory"),
                conversation_id=conversation_id,
                customer_id=customer_id,
                memory_type=memory_type,
                field=field,
                value=value,
                target_customer_id=target_customer_id,
                evidence=evidence,
                confidence=float(confidence or 0.0),
                risk_level=risk_level or "medium",
                scope=scope or "current_task",
                source_trace_id=trace_id,
                metadata=memory_metadata,
            )
            transition = StateTransition("task_memory", memory.memory_id, None, memory.status, "record_user_memory", trace_id)
            self._save_task_memory(memory)
            self._append_transition(transition)
            return memory, transition

    def record_pending_memory_candidate(
        self,
        *,
        conversation_id: str,
        customer_id: str,
        memory_type: str,
        field: str,
        value: Any,
        operation: str = "set",
        target_customer_id: str | None = None,
        evidence: str = "",
        confidence: float = 0.0,
        risk_level: str = "medium",
        scope: str = "long_term",
        metadata: dict[str, Any] | None = None,
        trace_id: str,
    ) -> tuple[PendingMemoryCandidate, StateTransition]:
        with self._write_transaction():
            from ...models import new_id

            task_context = self.current_task_context(conversation_id, customer_id)
            candidate_metadata = dict(metadata or {})
            if task_context is not None:
                candidate_metadata.setdefault("task_context_id", task_context.task_context_id)
            candidate = PendingMemoryCandidate(
                candidate_id=new_id("memory_candidate"),
                conversation_id=conversation_id,
                customer_id=customer_id,
                memory_type=memory_type,
                field=field,
                value=value,
                operation=operation or "set",
                target_customer_id=target_customer_id,
                evidence=evidence,
                confidence=float(confidence or 0.0),
                risk_level=risk_level or "medium",
                scope=scope or "long_term",
                source_trace_id=trace_id,
                metadata=candidate_metadata,
            )
            transition = StateTransition(
                "pending_memory_candidate",
                candidate.candidate_id,
                None,
                candidate.status,
                "record_user_memory",
                trace_id,
            )
            self._save_pending_memory_candidate(candidate)
            self._append_transition(transition)
            return candidate, transition

    def task_memory_context(self, conversation_id: str, customer_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            task_context = self.current_task_context(conversation_id, customer_id or "") if customer_id else None
            params: list[Any] = [conversation_id]
            condition = "conversation_id = ? AND status = 'active'"
            if customer_id:
                condition += " AND (customer_id = ? OR target_customer_id = ?)"
                params.extend([customer_id, customer_id])
            rows = self._connection.execute(
                f"""
                SELECT payload
                FROM runtime_task_memories
                WHERE {condition}
                ORDER BY updated_at DESC
                """,
                tuple(params),
            ).fetchall()
            memories = [_task_memory_from_payload(_loads(row["payload"])) for row in rows]
            return [
                item.to_dict()
                for item in memories
                if task_context is None
                or item.metadata.get("task_context_id") == task_context.task_context_id
                or (
                    not item.metadata.get("task_context_id")
                    and item.updated_at >= task_context.started_at
                )
            ]

    def pending_memory_candidates_for_context(
        self,
        conversation_id: str,
        customer_id: str | None = None,
        *,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        with self._lock:
            task_context = self.current_task_context(conversation_id, customer_id or "") if customer_id else None
            params: list[Any] = [conversation_id]
            condition = "conversation_id = ? AND status = 'pending_review'"
            if customer_id:
                condition += " AND (customer_id = ? OR target_customer_id = ?)"
                params.extend([customer_id, customer_id])
            rows = self._connection.execute(
                f"""
                SELECT payload
                FROM runtime_pending_memory_candidates
                WHERE {condition}
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (*params, int(limit)),
            ).fetchall()
            candidates = [_pending_memory_candidate_from_payload(_loads(row["payload"])) for row in rows]
            return [
                item.to_dict()
                for item in candidates
                if task_context is None
                or item.metadata.get("task_context_id") == task_context.task_context_id
                or (
                    not item.metadata.get("task_context_id")
                    and item.updated_at >= task_context.started_at
                )
            ]

    def task_memory_excluded_customer_ids(
        self,
        conversation_id: str | None,
        anchor_ids: list[str] | set[str] | None,
    ) -> list[str]:
        if not conversation_id:
            return []
        anchors = {str(item) for item in anchor_ids or [] if str(item or "").strip()}
        if not anchors:
            return []
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT payload
                FROM runtime_task_memories
                WHERE conversation_id = ? AND status = 'active'
                """,
                (conversation_id,),
            ).fetchall()
            excluded: list[str] = []
            for row in rows:
                memory = _task_memory_from_payload(_loads(row["payload"]))
                if memory.customer_id not in anchors or not is_avoid_playing_memory(memory):
                    continue
                task_context = self.current_task_context(conversation_id, memory.customer_id)
                memory_context_id = str(memory.metadata.get("task_context_id") or "")
                if task_context is not None and not (
                    memory_context_id == task_context.task_context_id
                    or (not memory_context_id and memory.updated_at >= task_context.started_at)
                ):
                    continue
                target_id = str(memory.target_customer_id or "")
                if target_id and target_id not in excluded:
                    excluded.append(target_id)
            return excluded
