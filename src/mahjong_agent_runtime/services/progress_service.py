from __future__ import annotations

"""No-progress and cycle handling for the agent loop."""

import json
from dataclasses import dataclass
from typing import Any

from ..hooks import HookManager
from ..models import AgentAction, ToolResult, UserMessage
from ..progress import ProgressDecision, ProgressMonitor
from ..runtime_components import ProgressHandlingResult
from ..stores import AgentStore
from .action_service import ActionProcessor


@dataclass(slots=True)
class ProgressGuardService:
    """Turn a progress diagnosis into replan feedback or a safe terminal result."""

    store: AgentStore
    trace_recorder: Any
    action_processor: ActionProcessor
    hook_manager: HookManager | None = None

    def handle_self_assessment(
        self,
        action: AgentAction,
        *,
        message: UserMessage,
        trace_id: str,
        run_id: str,
        run_version: int,
        step_index: int,
    ) -> ProgressHandlingResult | None:
        """Honor an explicit model stall report before any proposed tool executes."""

        assessment = action.self_assessment
        if assessment is None:
            return None
        payload = {
            "step_index": step_index,
            "progress": assessment.progress,
            "should_escalate": assessment.should_escalate,
            "goal": action.goal,
            "objective_status": action.objective_status,
        }
        self.trace_recorder.record(trace_id, "agent_self_assessment", payload)
        self._emit("after_progress_check", trace_id=trace_id, payload={"source": "model", **payload})
        if assessment.progress != "stalled" or not assessment.should_escalate:
            return None

        self.trace_recorder.record(
            trace_id,
            "agent_self_escalated",
            {**payload, "run_id": run_id, "run_version": run_version},
            level="WARN",
        )
        return self._needs_help_result(
            message,
            trace_id=trace_id,
            run_id=run_id,
            run_version=run_version,
            reason="agent_self_reported_stalled",
        )

    def handle(
        self,
        decision: ProgressDecision,
        *,
        message: UserMessage,
        trace_id: str,
        pending_tool_results: list[ToolResult],
        run_id: str,
        run_version: int,
        progress_monitor: ProgressMonitor,
    ) -> ProgressHandlingResult:
        """Trace progress and either continue, request one replan, or abort."""

        payload = decision.to_dict()
        self.trace_recorder.record(trace_id, "agent_progress_checked", payload)
        self._emit("after_progress_check", trace_id=trace_id, payload=payload)
        if not decision.detected:
            return ProgressHandlingResult(pending_tool_results=list(pending_tool_results))

        guard_result = progress_monitor.feedback_result(decision)
        next_pending = [*pending_tool_results, guard_result]
        self.store.append_tool_turn(
            message.conversation_id,
            json.dumps([guard_result.to_dict()], ensure_ascii=False),
            trace_id,
        )
        self.trace_recorder.record(trace_id, "agent_loop_detected", payload, level="WARN")
        self.trace_recorder.record(trace_id, "tool_result", guard_result.to_dict(), level="WARN")
        if decision.should_replan:
            self.trace_recorder.record(
                trace_id,
                "agent_replan_requested",
                {**payload, "feedback": guard_result.to_dict()},
                level="WARN",
            )
            return ProgressHandlingResult(
                pending_tool_results=next_pending,
                guard_result=guard_result,
            )

        self.trace_recorder.record(
            trace_id,
            "agent_loop_aborted",
            {**payload, "run_id": run_id, "run_version": run_version},
            level="ERROR",
        )
        result = self._needs_help_result(
            message,
            trace_id=trace_id,
            run_id=run_id,
            run_version=run_version,
            reason="agent_loop_no_progress",
            pending_tool_results=next_pending,
            guard_result=guard_result,
        )
        return result

    def _needs_help_result(
        self,
        message: UserMessage,
        *,
        trace_id: str,
        run_id: str,
        run_version: int,
        reason: str,
        pending_tool_results: list[ToolResult] | None = None,
        guard_result: ToolResult | None = None,
    ) -> ProgressHandlingResult:
        """Create the shared terminal outcome for model- and backend-detected stalls."""

        final_reply = ""
        if not self.action_processor.run_is_stale(message, run_version):
            final_reply = "这个我先转人工确认一下。"
            self.action_processor.append_pending_assistant_turn(
                message.conversation_id,
                final_reply,
                trace_id,
                run_id=run_id,
                run_version=run_version,
            )
        self.trace_recorder.record(
            trace_id,
            "final_output",
            {"reply": final_reply, "reason": reason, "status": "needs_help"},
            level="WARN",
        )
        return ProgressHandlingResult(
            pending_tool_results=list(pending_tool_results or []),
            guard_result=guard_result,
            final_reply=final_reply,
            stop_loop=True,
            runtime_status="needs_help",
        )

    def _emit(self, event_name: str, *, trace_id: str, payload: dict[str, Any]) -> None:
        if self.hook_manager is not None:
            self.hook_manager.emit(event_name, trace_id=trace_id, payload=payload)


__all__ = ["ProgressGuardService"]
