"""定义任务存储相关的存储接口，隔离领域层与具体持久化实现。

"""

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
    """定义任务存储需要遵守的抽象接口契约。

"""

    @property
    def task_memories(self) -> dict[str, TaskMemory]:
        ...

    @property
    def pending_memory_candidates(self) -> dict[str, PendingMemoryCandidate]:
        ...

    @property
    def pending_input_batches(self) -> dict[str, PendingInputBatch]:
        ...

    @property
    def scheduled_tasks(self) -> dict[str, ScheduledAgentTask]:
        ...

    @property
    def badcases(self) -> list[dict[str, Any]]:
        ...

    def record_task_memory(self, **kwargs: Any) -> tuple[TaskMemory, StateTransition]:
        """记录任务记忆，形成后续决策和审计所需的持久证据。"""
        ...

    def record_pending_memory_candidate(
        self,
        **kwargs: Any,
    ) -> tuple[PendingMemoryCandidate, StateTransition]:
        """记录pending记忆候选人，形成后续决策和审计所需的持久证据。"""
        ...

    def task_memory_context(
        self,
        conversation_id: str,
        customer_id: str | None = None,
    ) -> list[dict[str, Any]]:
        ...

    def pending_memory_candidates_for_context(
        self,
        conversation_id: str,
        customer_id: str | None = None,
    ) -> list[dict[str, Any]]:
        ...

    def task_memory_excluded_customer_ids(
        self,
        conversation_id: str,
        customer_id: str | None = None,
    ) -> list[str]:
        ...

    def scheduled_task_for_game(self, game_id: str) -> ScheduledAgentTask | None:
        ...

    def ensure_game_recruitment_task(
        self,
        game_id: str,
        *,
        trace_id: str,
    ) -> tuple[ScheduledAgentTask | None, StateTransition | None]:
        """确保牌局recruitment任务满足后续步骤依赖的不变量。"""
        ...

    def ensure_waiting_demand_expiration_task(
        self,
        *,
        due_at: datetime,
        trace_id: str,
    ) -> tuple[ScheduledAgentTask, StateTransition | None]:
        """确保候补需求expiration任务满足后续步骤依赖的不变量。"""
        ...

    def due_scheduled_tasks(self, *, at: datetime, limit: int = 100) -> list[ScheduledAgentTask]:
        ...

    def open_game_recruitment(
        self,
        game_id: str,
        *,
        trace_id: str,
        at: datetime | None = None,
    ) -> tuple[Any, list[StateTransition]]:
        ...

    def claim_scheduled_task(
        self,
        task_id: str,
        *,
        at: datetime,
        lease_seconds: int,
        trace_id: str,
    ) -> ScheduledAgentTask | None:
        """原子领取定时任务，防止并发消费者重复处理。"""
        ...

    def complete_scheduled_task(
        self,
        task_id: str,
        *,
        trace_id: str,
    ) -> ScheduledAgentTask:
        ...

    def fail_scheduled_task(
        self,
        task_id: str,
        *,
        error: str,
        retry_at: datetime | None,
        trace_id: str,
    ) -> ScheduledAgentTask:
        ...

    def upsert_pending_input_fragment(self, **kwargs: Any) -> PendingInputBatch:
        """新增或更新pending输入消息片段，使重复写入收敛到同一条记录。"""
        ...

    def pending_input_batch(
        self,
        conversation_id: str,
        sender_id: str,
    ) -> PendingInputBatch | None:
        ...

    def due_pending_input_batches(self, *, at: datetime, limit: int = 100) -> list[PendingInputBatch]:
        ...

    def claim_pending_input_batch(self, **kwargs: Any) -> PendingInputBatch | None:
        """原子领取pending输入批次，防止并发消费者重复处理。"""
        ...

    def record_pending_input_decision(self, **kwargs: Any) -> PendingInputBatch:
        """记录pending输入决策，形成后续决策和审计所需的持久证据。"""
        ...

    def queue_pending_input_batch(self, **kwargs: Any) -> PendingInputBatch:
        ...

    def finish_pending_input_batch(self, **kwargs: Any) -> PendingInputBatch:
        ...

    def record_badcase(
        self,
        payload: dict[str, Any],
        *,
        trace_id: str,
        conversation_id: str,
    ) -> dict[str, Any]:
        """记录badcase，形成后续决策和审计所需的持久证据。"""
        ...

    def clear_runtime_state(
        self,
        *,
        conversation_id: str | None = None,
        include_customers: bool = False,
        include_badcases: bool = False,
    ) -> dict[str, int]:
        """清理运行时状态，用于显式重置受控状态。"""
        ...
