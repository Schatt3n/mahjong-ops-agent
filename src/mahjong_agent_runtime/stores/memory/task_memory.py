"""InMemory task memory store operations."""

from __future__ import annotations

from typing import Any
from ...models import (
    PendingMemoryCandidate,
    StateTransition,
    TaskMemory,
    new_id,
)
from ...domains import is_avoid_playing_memory

class InMemoryTaskMemoryStoreMixin:
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
        with self._lock:
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
            self.task_memories[memory.memory_id] = memory
            transition = StateTransition(
                entity_type="task_memory",
                entity_id=memory.memory_id,
                from_status=None,
                to_status=memory.status,
                reason="record_user_memory",
                trace_id=trace_id,
            )
            self.transitions.append(transition)
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
        with self._lock:
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
            self.pending_memory_candidates[candidate.candidate_id] = candidate
            transition = StateTransition(
                entity_type="pending_memory_candidate",
                entity_id=candidate.candidate_id,
                from_status=None,
                to_status=candidate.status,
                reason="record_user_memory",
                trace_id=trace_id,
            )
            self.transitions.append(transition)
            return candidate, transition

    def task_memory_context(self, conversation_id: str, customer_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            task_context = self.current_task_context(conversation_id, customer_id or "") if customer_id else None
            memories = [
                item.to_dict()
                for item in self.task_memories.values()
                if item.status == "active"
                and item.conversation_id == conversation_id
                and (not customer_id or item.customer_id == customer_id or item.target_customer_id == customer_id)
                and (
                    task_context is None
                    or item.metadata.get("task_context_id") == task_context.task_context_id
                    or (
                        not item.metadata.get("task_context_id")
                        and item.updated_at >= task_context.started_at
                    )
                )
            ]
            memories.sort(key=lambda item: str(item.get("updated_at") or ""), reverse=True)
            return memories

    def pending_memory_candidates_for_context(
        self,
        conversation_id: str,
        customer_id: str | None = None,
        *,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        with self._lock:
            task_context = self.current_task_context(conversation_id, customer_id or "") if customer_id else None
            candidates = [
                item.to_dict()
                for item in self.pending_memory_candidates.values()
                if item.status == "pending_review"
                and item.conversation_id == conversation_id
                and (not customer_id or item.customer_id == customer_id or item.target_customer_id == customer_id)
                and (
                    task_context is None
                    or item.metadata.get("task_context_id") == task_context.task_context_id
                    or (
                        not item.metadata.get("task_context_id")
                        and item.updated_at >= task_context.started_at
                    )
                )
            ]
            candidates.sort(key=lambda item: str(item.get("updated_at") or ""), reverse=True)
            return candidates[: int(limit)]

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
            excluded: list[str] = []
            for item in self.task_memories.values():
                if item.status != "active" or item.conversation_id != conversation_id:
                    continue
                if item.customer_id not in anchors:
                    continue
                task_context = self.current_task_context(conversation_id, item.customer_id)
                memory_context_id = str(item.metadata.get("task_context_id") or "")
                if task_context is not None and not (
                    memory_context_id == task_context.task_context_id
                    or (not memory_context_id and item.updated_at >= task_context.started_at)
                ):
                    continue
                if not is_avoid_playing_memory(item):
                    continue
                target_id = str(item.target_customer_id or "")
                if target_id and target_id not in excluded:
                    excluded.append(target_id)
            return excluded
