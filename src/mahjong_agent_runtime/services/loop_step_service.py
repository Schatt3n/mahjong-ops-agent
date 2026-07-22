from __future__ import annotations

"""One iteration of context -> model -> tools/reply -> progress evaluation."""

from dataclasses import dataclass

from ..models import AgentAction, StateTransition, ToolResult, UserMessage
from ..progress import ProgressMonitor, build_progress_hint
from ..runtime_components import TurnBudgets
from .action_service import ActionProcessor
from .context_service import ContextLifecycleManager
from .contracts import LoopStepOutcome
from .progress_service import ProgressGuardService


@dataclass(slots=True)
class AgentLoopStepService:
    """Execute one auditable iteration without owning cross-step state."""

    context_lifecycle: ContextLifecycleManager
    action_processor: ActionProcessor
    progress_guard: ProgressGuardService

    def execute(
        self,
        message: UserMessage,
        *,
        trace_id: str,
        run_id: str,
        run_version: int,
        step_index: int,
        budgets: TurnBudgets,
        pending_tool_results: list[ToolResult],
        progress_monitor: ProgressMonitor,
        action_history: list[AgentAction],
    ) -> LoopStepOutcome:
        """Build context, obtain one action, execute it, and evaluate progress."""

        private_loop = str((message.metadata or {}).get("source") or "") != "group"
        progress_hint = (
            build_progress_hint(action_history)
            if private_loop and step_index >= 2 and action_history
            else None
        )
        built = self.context_lifecycle.build_and_trace_context(
            message,
            trace_id=trace_id,
            pending_tool_results=pending_tool_results,
            run_id=run_id,
            run_version=run_version,
            step_index=step_index,
            progress_hint=progress_hint,
        )
        built, summary_transition = self.context_lifecycle.summarize_and_rebuild_context_if_needed(
            message,
            built=built,
            trace_id=trace_id,
            pending_tool_results=pending_tool_results,
            run_id=run_id,
            run_version=run_version,
            step_index=step_index,
            budget=budgets.agent,
            progress_hint=progress_hint,
        )
        model_step = self.action_processor.call_agent_action(
            message,
            trace_id=trace_id,
            built_messages=built.messages,
            step_index=step_index,
            budget=budgets.agent,
            run_id=run_id,
            run_version=run_version,
        )
        if model_step.stop_loop:
            return LoopStepOutcome(
                summary_transition=summary_transition,
                final_reply=model_step.final_reply or "",
                stop_loop=True,
            )
        action = model_step.action
        if action is None:
            return LoopStepOutcome(
                summary_transition=summary_transition,
                final_reply="这个我先转人工确认一下。",
                stop_loop=True,
            )
        if model_step.errors:
            return self._handle_contract_error(
                action=action,
                raw_response=model_step.raw_response,
                errors=model_step.errors,
                message=message,
                trace_id=trace_id,
                run_id=run_id,
                run_version=run_version,
                step_index=step_index,
                summary_transition=summary_transition,
                progress_monitor=progress_monitor,
            )

        self.action_processor.trace_action_plan(
            action,
            trace_id=trace_id,
            step_index=step_index,
            previous_tool_result_count=len(pending_tool_results),
        )
        if private_loop:
            self_assessment = self.progress_guard.handle_self_assessment(
                action,
                message=message,
                trace_id=trace_id,
                run_id=run_id,
                run_version=run_version,
                step_index=step_index,
            )
            if self_assessment is not None and self_assessment.stop_loop:
                return LoopStepOutcome(
                    action=action,
                    summary_transition=summary_transition,
                    final_reply=self_assessment.final_reply,
                    stop_loop=True,
                    runtime_status=self_assessment.runtime_status,
                )
        processed = self.action_processor.process_action(
            action,
            message=message,
            trace_id=trace_id,
            context_payload=built.payload,
            previous_pending_tool_results=pending_tool_results,
            step_index=step_index,
            budgets=budgets,
            run_id=run_id,
            run_version=run_version,
        )
        if processed.stop_loop:
            return LoopStepOutcome(
                action=processed.action,
                tool_results=processed.tool_results,
                pending_tool_results=processed.pending_tool_results,
                summary_transition=summary_transition,
                final_reply=processed.final_reply or "",
                stop_loop=True,
                runtime_status=processed.action.objective_status,
            )
        if not processed.continue_loop:
            return LoopStepOutcome(
                action=processed.action,
                tool_results=processed.tool_results,
                pending_tool_results=processed.pending_tool_results,
                summary_transition=summary_transition,
            )
        decision = progress_monitor.observe_action(
            processed.action,
            processed.tool_results,
            step_index=step_index,
        )
        progress = self.progress_guard.handle(
            decision,
            message=message,
            trace_id=trace_id,
            pending_tool_results=processed.pending_tool_results,
            run_id=run_id,
            run_version=run_version,
            progress_monitor=progress_monitor,
        )
        tool_results = list(processed.tool_results)
        if progress.guard_result is not None:
            tool_results.append(progress.guard_result)
        return LoopStepOutcome(
            action=processed.action,
            tool_results=tool_results,
            pending_tool_results=progress.pending_tool_results,
            summary_transition=summary_transition,
            final_reply=progress.final_reply,
            stop_loop=progress.stop_loop,
            runtime_status=progress.runtime_status,
        )

    def _handle_contract_error(
        self,
        *,
        action: AgentAction,
        raw_response: str,
        errors: list[str],
        message: UserMessage,
        trace_id: str,
        run_id: str,
        run_version: int,
        step_index: int,
        summary_transition: StateTransition | None,
        progress_monitor: ProgressMonitor,
    ) -> LoopStepOutcome:
        pending = self.action_processor.record_action_contract_feedback(
            message,
            trace_id=trace_id,
            raw_response=raw_response,
            errors=errors,
            step_index=step_index,
        )
        decision = progress_monitor.observe_runtime_feedback(
            "action_contract_error",
            {"errors": errors},
            step_index=step_index,
        )
        progress = self.progress_guard.handle(
            decision,
            message=message,
            trace_id=trace_id,
            pending_tool_results=pending,
            run_id=run_id,
            run_version=run_version,
            progress_monitor=progress_monitor,
        )
        results = [progress.guard_result] if progress.guard_result is not None else []
        return LoopStepOutcome(
            action=action,
            tool_results=results,
            pending_tool_results=progress.pending_tool_results,
            summary_transition=summary_transition,
            final_reply=progress.final_reply,
            stop_loop=progress.stop_loop,
            runtime_status=progress.runtime_status,
        )


__all__ = ["AgentLoopStepService"]
