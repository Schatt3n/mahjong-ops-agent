from __future__ import annotations

"""Application service for Agent-run checkpoints, leases, and recovery."""

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
from ..models import AgentAction, StateTransition, ToolResult, UserMessage, now
from ..progress import ProgressMonitor
from ..runtime_components import TurnBudgets
from ..stores import AgentStore


class AgentRunLeaseLostError(RuntimeError):
    """Raised when another worker owns the checkpoint lease."""


@dataclass(slots=True)
class AgentRunStateManager:
    """Keep persistence mechanics outside the thin AgentLoop."""

    store: AgentStore
    trace_recorder: Any
    lease_seconds: int = 120
    max_attempts: int = 3
    worker_id: str = ""

    def __post_init__(self) -> None:
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
        """Create the initial lease before model execution starts."""

        state = AgentRunState(
            run_id=run_id,
            trace_id=trace_id,
            conversation_id=message.conversation_id,
            run_version=run_version,
            message=redact_runtime_payload(message.to_dict()),
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
            },
        )
        return created

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
        """Renew the lease after a complete, replay-safe Agent step."""

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

    def mark_recoverable(self, run_id: str, *, error: BaseException) -> AgentRunState | None:
        """Release a failed worker lease so another process can continue."""

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
            # A newer customer turn or another worker won the race. Never let
            # this stale worker overwrite the authoritative run state.
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
        """Mark completion only after the final message result is durable."""

        state = self.store.agent_run(run_id)
        if state is None:
            return None
        state.status = AgentRunStatus.COMPLETED
        state.final_reply = str(final_reply or "")
        state.runtime_status = str(runtime_status or "completed")
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
        """A newer user turn invalidates unfinished reasoning from older input."""

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
