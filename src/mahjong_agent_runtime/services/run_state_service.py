"""编排run状态服务相关的应用服务，不直接承载通道或数据库细节。

"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ..agent_state import (
    AgentRunState,
    AgentRunStatus,
    lease_deadline,
    redact_runtime_payload,
    snapshot_budgets,
)
from ..customer_visible_contract import CUSTOMER_EXECUTION_FAILURE_REPLY
from ..models import AgentAction, PendingInputBatchStatus, StateTransition, ToolResult, UserMessage, now
from ..progress import ProgressMonitor
from ..runtime_components import TurnBudgets
from ..stores import AgentStore


class AgentRunLeaseLostError(RuntimeError):
    """表示智能体runleaselost异常场景失败时抛出的领域异常。

"""


@dataclass(slots=True)
class AgentRunStateManager:
    """创建、续租和结束运行记录，并为中断恢复保存完整步骤状态。

    上游：``AgentRuntime`` 与 ``AgentLoop``。下游：``AgentRunStore`` Protocol；生产实现
    位于 ``stores/sqlite/agent_runs.py``，测试实现位于 ``stores/memory/agent_runs.py``。
    """

    store: AgentStore
    trace_recorder: Any
    lease_seconds: int = 120
    max_attempts: int = 3
    worker_id: str = ""

    def __post_init__(self) -> None:
        """完成数据类初始化后的依赖组装和不变量校验。"""
        self.lease_seconds = max(10, int(self.lease_seconds))
        self.max_attempts = max(1, int(self.max_attempts))
        if not self.worker_id:
            self.worker_id = f"worker_{uuid.uuid4().hex[:12]}"

    def start(
        self,
        message: UserMessage,
        *,
        trace_id: str,
        run_id: str,
        run_version: int,
    ) -> AgentRunState:
        """在模型执行开始前创建初始租约。"""

        input_wait = self._confirmed_input_wait(message, trace_id=trace_id, run_version=run_version)
        state = AgentRunState(
            run_id=run_id,
            trace_id=trace_id,
            conversation_id=message.conversation_id,
            run_version=run_version,
            message=redact_runtime_payload(message.to_dict()),
            handoff_wait_delivered=input_wait is not None,
            lease_owner=self.worker_id,
            lease_until=lease_deadline(lease_seconds=self.lease_seconds),
        )
        created = self.store.create_agent_run(state)
        self.trace_recorder.record(
            trace_id,
            "agent_run_started",
            {
                "run_id": run_id,
                "run_version": run_version,
                "lease_owner": self.worker_id,
                "next_step_index": 1,
                **({"prior_input_wait_delivery": input_wait} if input_wait else {}),
            },
        )
        return created

    def _confirmed_input_wait(self, message: UserMessage, *, trace_id: str, run_version: int) -> dict | None:
        """只继承当前已领取入口批次中原来话实际送达的等待回执。"""
        window = message.metadata.get("input_window")
        if not isinstance(window, dict) or window.get("trigger") != "input_gate_recovery":
            return None
        batch = self.store.pending_input_batch(message.conversation_id, message.sender_id)
        if (batch is None or batch.status is not PendingInputBatchStatus.PROCESSING
                or batch.batch_id != window.get("batch_id") or batch.version != window.get("batch_version")):
            return None
        recovery = batch.decision.get("gate_recovery") or {}
        version = recovery.get("conversation_version")
        source_ids = [fragment.get("message_id") for fragment in batch.fragments]
        if (recovery.get("open") is not True or recovery.get("wait_delivered") is not True
                or recovery.get("origin_trace_id") != trace_id or recovery.get("attempt") != 1
                or type(version) is not int or version + 1 != run_version
                or run_version != self.store.conversation_version(message.conversation_id)
                or not source_ids or not all(isinstance(source, str) and source for source in source_ids)
                or source_ids != window.get("source_message_ids")):
            return None
        route = recovery.get("route")
        if route == "complete_message" and len(source_ids) == 1:
            expected_message_id = source_ids[0]
        elif route == "aggregated":
            expected_message_id = f"{batch.batch_id}:v{batch.version}"
        else:
            return None
        if expected_message_id != message.message_id:
            return None
        # 这里使用 Store 记录，来话或模型输出携带的回执字段没有抑制外发的权限。
        return {"batch_id": batch.batch_id, "batch_version": batch.version, "source_message_id": source_ids[-1]}

    def checkpoint(
        self,
        state: AgentRunState,
        *,
        next_step_index: int,
        actions: list[AgentAction],
        tool_results: list[ToolResult],
        pending_tool_results: list[ToolResult],
        turn_tool_evidence: list[ToolResult],
        transitions: list[StateTransition],
        budgets: TurnBudgets,
        progress_monitor: ProgressMonitor,
        final_reply: str = "",
        runtime_status: str = "",
    ) -> AgentRunState:
        """在完整且可重放安全的 Agent 步骤后续订租约。"""

        # 审查步骤可以在写入 checkpoint 之前打开人工待办和记录等待话术回执。
        current = self.store.agent_run(state.run_id)
        if current is not None and current.lease_owner == self.worker_id:
            state.handoff_reason = current.handoff_reason
            state.handoff_wait_delivered = current.handoff_wait_delivered
        state.next_step_index = max(1, int(next_step_index))
        state.actions = [item.to_dict() for item in actions]
        state.tool_results = [item.to_dict() for item in tool_results]
        state.pending_tool_results = [item.to_dict() for item in pending_tool_results]
        state.turn_tool_evidence = [item.to_dict() for item in turn_tool_evidence]
        state.transitions = [item.to_dict() for item in transitions]
        state.budget_state = snapshot_budgets(budgets)
        state.progress_state = progress_monitor.snapshot()
        state.final_reply = str(final_reply or "")
        state.runtime_status = str(runtime_status or "")
        state.status = AgentRunStatus.RUNNING
        state.lease_owner = self.worker_id
        state.lease_until = lease_deadline(lease_seconds=self.lease_seconds)
        saved = self.store.save_agent_run(
            state,
            expected_lease_owner=self.worker_id,
        )
        if not saved:
            raise AgentRunLeaseLostError(f"agent run lease lost: {state.run_id}")
        self.trace_recorder.record(
            state.trace_id,
            "agent_run_checkpointed",
            {
                "run_id": state.run_id,
                "next_step_index": state.next_step_index,
                "action_count": len(state.actions),
                "tool_result_count": len(state.tool_results),
                "pending_tool_result_count": len(state.pending_tool_results),
                "runtime_status": state.runtime_status,
            },
        )
        return state

    def request_handoff(self, run_id: str, *, reason: str) -> bool:
        """同一运行只打开一次店家待办，持旧租约的执行者不得重新打开。"""
        state = self.store.agent_run(run_id)
        if (state is None or state.lease_owner != self.worker_id or state.status != AgentRunStatus.RUNNING
                or state.handoff_reason):
            return False
        state.handoff_reason = reason
        return self.store.save_agent_run(state, expected_lease_owner=self.worker_id)

    def mark_recoverable(self, run_id: str, *, error: BaseException) -> AgentRunState | None:
        """释放失败的工作者租约，使另一个进程可以继续。"""

        state = self.store.agent_run(run_id)
        if state is None or state.status in {
            AgentRunStatus.COMPLETED,
            AgentRunStatus.SUPERSEDED,
        }:
            return state
        state.last_error = f"{type(error).__name__}: {error}"
        state.lease_owner = ""
        state.lease_until = None
        if state.attempts >= self.max_attempts:
            state.status = AgentRunStatus.FAILED
            state.runtime_status = AgentRunStatus.FAILED.value
            state.completed_at = now()
        else:
            state.status = AgentRunStatus.RECOVERABLE
        saved = self.store.save_agent_run(
            state,
            expected_lease_owner=self.worker_id,
        )
        if not saved:
            # 更新的客户轮次或其他工作进程已赢得竞争，
            # 当前过期工作进程不得覆盖权威运行状态。
            return self.store.agent_run(run_id)
        self.trace_recorder.record(
            state.trace_id,
            "agent_run_recoverable" if state.status == AgentRunStatus.RECOVERABLE else "agent_run_failed",
            {
                "run_id": run_id,
                "status": state.status.value,
                "attempts": state.attempts,
                "error_type": type(error).__name__,
                "error": str(error),
            },
            level="ERROR",
        )
        return state

    def complete(
        self,
        run_id: str,
        *,
        final_reply: str,
        runtime_status: str,
    ) -> AgentRunState | None:
        """仅在最终消息结果持久化后标记为完成。"""

        state = self.store.agent_run(run_id)
        if state is None:
            return None
        state.status = AgentRunStatus.SUPERSEDED if runtime_status == "superseded" else AgentRunStatus.COMPLETED
        state.final_reply = str(final_reply or "")
        state.runtime_status = str(runtime_status or "completed")
        resolved_reason = (state.handoff_reason if runtime_status in {"completed", "waiting_user"}
                           and final_reply != CUSTOMER_EXECUTION_FAILURE_REPLY else "")
        if resolved_reason:
            state.handoff_reason = ""
        state.last_error = ""
        state.lease_owner = ""
        state.lease_until = None
        state.completed_at = now()
        saved = self.store.save_agent_run(
            state,
            expected_lease_owner=self.worker_id,
        )
        if not saved:
            raise AgentRunLeaseLostError(f"agent run lease lost before completion: {run_id}")
        if resolved_reason:
            self.trace_recorder.record(state.trace_id, "agent_runtime_attention_resolved", {
                "run_id": run_id, "run_version": state.run_version, "reason": resolved_reason,
                "message_id": state.message.get("message_id"), "runtime_status": state.runtime_status,
            })
        self.trace_recorder.record(
            state.trace_id,
            "agent_run_completed",
            {
                "run_id": run_id,
                "runtime_status": state.runtime_status,
                "next_step_index": state.next_step_index,
            },
        )
        return state

    def supersede_stale(
        self,
        conversation_id: str,
        *,
        current_version: int,
        trace_id: str,
    ) -> list[StateTransition]:
        """较新的用户轮次会使较早输入的未完成推理失效。"""

        transitions = self.store.supersede_agent_runs(
            conversation_id,
            current_version=current_version,
            trace_id=trace_id,
        )
        if transitions:
            self.trace_recorder.record(
                trace_id,
                "agent_runs_superseded",
                {
                    "conversation_id": conversation_id,
                    "current_version": current_version,
                    "run_ids": [item.entity_id for item in transitions],
                },
            )
        return transitions

    def recoverable(self, *, at: datetime | None = None, limit: int = 100) -> list[AgentRunState]:
        return self.store.recoverable_agent_runs(
            at=at or now(),
            limit=max(1, int(limit)),
        )

    def claim(self, run_id: str, *, at: datetime | None = None) -> AgentRunState | None:
        stamp = at or now()
        state = self.store.claim_agent_run(
            run_id,
            lease_owner=self.worker_id,
            lease_until=lease_deadline(
                lease_seconds=self.lease_seconds,
                at=stamp,
            ),
            at=stamp,
        )
        if state is not None:
            self.trace_recorder.record(
                state.trace_id,
                "agent_run_recovery_claimed",
                {
                    "run_id": state.run_id,
                    "attempts": state.attempts,
                    "next_step_index": state.next_step_index,
                    "worker_id": self.worker_id,
                },
            )
        return state


__all__ = [
    "AgentRunLeaseLostError",
    "AgentRunStateManager",
]
