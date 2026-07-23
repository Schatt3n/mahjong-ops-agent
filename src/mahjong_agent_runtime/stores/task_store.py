"""Scheduled work, task memory, and input aggregation contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol

from ..models import (
    PendingInputBatch,
    PendingMemoryCandidate,
    ScheduledAgentTask,
    StateTransition,
    TaskMemory,
)


class TaskStore(Protocol):
    """Persistence operations for durable and short-lived Agent work."""

    @property
    def task_memories(self) -> dict[str, TaskMemory]: ...

    @property
    def pending_memory_candidates(self) -> dict[str, PendingMemoryCandidate]: ...

    @property
    def pending_input_batches(self) -> dict[str, PendingInputBatch]: ...

    @property
    def scheduled_tasks(self) -> dict[str, ScheduledAgentTask]: ...

    @property
    def badcases(self) -> list[dict[str, Any]]: ...

    def record_task_memory(self, **kwargs: Any) -> tuple[TaskMemory, StateTransition]: ...

    def record_pending_memory_candidate(
        self,
        **kwargs: Any,
    ) -> tuple[PendingMemoryCandidate, StateTransition]: ...

    def task_memory_context(
        self,
        conversation_id: str,
        customer_id: str | None = None,
    ) -> list[dict[str, Any]]: ...

    def pending_memory_candidates_for_context(
        self,
        conversation_id: str,
        customer_id: str | None = None,
    ) -> list[dict[str, Any]]: ...

    def task_memory_excluded_customer_ids(
        self,
        conversation_id: str,
        customer_id: str | None = None,
    ) -> list[str]: ...

    def scheduled_task_for_game(self, game_id: str) -> ScheduledAgentTask | None: ...

    def ensure_game_recruitment_task(
        self,
        game_id: str,
        *,
        trace_id: str,
    ) -> tuple[ScheduledAgentTask | None, StateTransition | None]: ...

    def ensure_waiting_demand_expiration_task(
        self,
        *,
        due_at: datetime,
        trace_id: str,
    ) -> tuple[ScheduledAgentTask, StateTransition | None]: ...

    def due_scheduled_tasks(self, *, at: datetime, limit: int = 100) -> list[ScheduledAgentTask]: ...

    def open_game_recruitment(
        self,
        game_id: str,
        *,
        trace_id: str,
        at: datetime | None = None,
    ) -> tuple[Any, list[StateTransition]]: ...

    def claim_scheduled_task(
        self,
        task_id: str,
        *,
        at: datetime,
        lease_seconds: int,
        trace_id: str,
    ) -> ScheduledAgentTask | None: ...

    def complete_scheduled_task(
        self,
        task_id: str,
        *,
        trace_id: str,
    ) -> ScheduledAgentTask: ...

    def fail_scheduled_task(
        self,
        task_id: str,
        *,
        error: str,
        retry_at: datetime | None,
        trace_id: str,
    ) -> ScheduledAgentTask: ...

    def upsert_pending_input_fragment(self, **kwargs: Any) -> PendingInputBatch: ...

    def pending_input_batch(
        self,
        conversation_id: str,
        sender_id: str,
    ) -> PendingInputBatch | None: ...

    def due_pending_input_batches(self, *, at: datetime, limit: int = 100) -> list[PendingInputBatch]: ...

    def claim_pending_input_batch(self, **kwargs: Any) -> PendingInputBatch | None: ...

    def record_pending_input_decision(self, **kwargs: Any) -> PendingInputBatch: ...

    def queue_pending_input_batch(self, **kwargs: Any) -> PendingInputBatch: ...

    def finish_pending_input_batch(self, **kwargs: Any) -> PendingInputBatch: ...

    def record_badcase(
        self,
        payload: dict[str, Any],
        *,
        trace_id: str,
        conversation_id: str,
    ) -> dict[str, Any]: ...

    def clear_runtime_state(
        self,
        *,
        conversation_id: str | None = None,
        include_customers: bool = False,
        include_badcases: bool = False,
    ) -> dict[str, int]: ...
