from __future__ import annotations

"""Thin goal-driven orchestration loop."""

from dataclasses import dataclass
from typing import Any

from ..agent_state import (
    AgentRunState,
    restore_budgets,
    restore_tool_result,
    restore_transition,
)
from ..budget import TokenBudget
from ..hooks import HookManager
from ..models import AgentAction, AgentRuntimeResult, ToolResult, UserMessage
from ..progress import ProgressMonitor
from ..runtime_components import TurnBudgets
from ..stores import AgentStore
from ..task_context import TaskContextManager
from .action_service import ActionProcessor
from .context_service import ContextLifecycleManager
from .loop_step_service import AgentLoopStepService
from .loop_support import fresh_turn_budgets, handle_max_steps, prepare_turn
from .progress_service import ProgressGuardService
from .run_state_service import AgentRunStateManager


@dataclass(slots=True)
class AgentLoop:
    """Run prepare -> step until terminal -> aggregate auditable results."""

    store: AgentStore
    trace_recorder: Any
    context_lifecycle: ContextLifecycleManager
    action_processor: ActionProcessor
    task_context_manager: TaskContextManager
    token_budget: TokenBudget
    review_token_budget: TokenBudget
    text_generation_token_budget: TokenBudget
    max_steps: int = 8
    repeated_observation_limit: int = 2
    consecutive_no_progress_limit: int = 2
    max_progress_replans: int = 1
    max_cycle_period: int = 3
    hook_manager: HookManager | None = None
    run_state_manager: AgentRunStateManager | None = None

    def run(
        self,
        message: UserMessage,
        *,
        trace_id: str,
        run_id: str,
        run_version: int,
        run_state: AgentRunState | None = None,
    ) -> AgentRuntimeResult:
        """Execute bounded model/tool iterations for one ordered conversation turn."""

        budgets = self.fresh_turn_budgets()
        progress_monitor = self._progress_monitor()
        if run_state is None or not run_state.turn_prepared:
            transitions = prepare_turn(
                store=self.store,
                trace_recorder=self.trace_recorder,
                hook_manager=self.hook_manager,
                task_context_manager=self.task_context_manager,
                message=message,
                trace_id=trace_id,
            )
            actions: list[AgentAction] = []
            tool_results: list[ToolResult] = []
            pending_tool_results: list[ToolResult] = []
            turn_tool_evidence: list[ToolResult] = []
            final_reply = ""
            runtime_status = "completed"
            first_step_index = 1
            if self.run_state_manager is not None and run_state is not None:
                run_state.turn_prepared = True
                self.run_state_manager.checkpoint(
                    run_state,
                    next_step_index=1,
                    actions=actions,
                    tool_results=tool_results,
                    pending_tool_results=pending_tool_results,
                    turn_tool_evidence=turn_tool_evidence,
                    transitions=transitions,
                    budgets=budgets,
                    progress_monitor=progress_monitor,
                )
        else:
            transitions = [
                restore_transition(item)
                for item in run_state.transitions
            ]
            actions = [
                AgentAction.from_payload(item)
                for item in run_state.actions
            ]
            tool_results = [
                restore_tool_result(item)
                for item in run_state.tool_results
            ]
            pending_tool_results = [
                restore_tool_result(item)
                for item in run_state.pending_tool_results
            ]
            turn_tool_evidence = [
                restore_tool_result(item)
                for item in run_state.turn_tool_evidence
            ]
            restore_budgets(budgets, run_state.budget_state)
            progress_monitor.restore(run_state.progress_state)
            final_reply = run_state.final_reply
            runtime_status = run_state.runtime_status or "completed"
            first_step_index = max(1, run_state.next_step_index)
            self.trace_recorder.record(
                trace_id,
                "agent_run_resumed",
                {
                    "run_id": run_id,
                    "run_version": run_version,
                    "next_step_index": first_step_index,
                    "action_count": len(actions),
                    "tool_result_count": len(tool_results),
                },
            )
        step_service = self._step_service()

        if final_reply and run_state is not None:
            return self._result(
                message=message,
                trace_id=trace_id,
                actions=actions,
                tool_results=tool_results,
                transitions=transitions,
                final_reply=final_reply,
                runtime_status=runtime_status,
            )

        for step_index in range(first_step_index, self.max_steps + 1):
            outcome = step_service.execute(
                message,
                trace_id=trace_id,
                run_id=run_id,
                run_version=run_version,
                step_index=step_index,
                budgets=budgets,
                pending_tool_results=pending_tool_results,
                turn_tool_evidence=turn_tool_evidence,
                progress_monitor=progress_monitor,
                action_history=actions,
            )
            if outcome.summary_transition is not None:
                transitions.append(outcome.summary_transition)
            if outcome.action is not None:
                actions.append(outcome.action)
            tool_results.extend(outcome.tool_results)
            pending_tool_results = outcome.pending_tool_results
            # ``evidence_results`` may include internal contract feedback that
            # the next model step needs but that is not an executed business
            # tool result exposed by AgentRuntimeResult.
            step_evidence = outcome.evidence_results or outcome.tool_results
            turn_tool_evidence.extend(step_evidence)
            if outcome.stop_loop:
                final_reply = outcome.final_reply or ""
                runtime_status = outcome.runtime_status or (
                    outcome.action.objective_status if outcome.action is not None else "needs_human"
                )
            if self.run_state_manager is not None and run_state is not None:
                self.run_state_manager.checkpoint(
                    run_state,
                    next_step_index=step_index + 1,
                    actions=actions,
                    tool_results=tool_results,
                    pending_tool_results=pending_tool_results,
                    turn_tool_evidence=turn_tool_evidence,
                    transitions=transitions,
                    budgets=budgets,
                    progress_monitor=progress_monitor,
                    final_reply=final_reply if outcome.stop_loop else "",
                    runtime_status=runtime_status if outcome.stop_loop else "",
                )
            if outcome.stop_loop:
                break
        else:
            final_reply = handle_max_steps(
                action_processor=self.action_processor,
                trace_recorder=self.trace_recorder,
                message=message,
                trace_id=trace_id,
                run_id=run_id,
                run_version=run_version,
            )
            runtime_status = "needs_help"
            if self.run_state_manager is not None and run_state is not None:
                self.run_state_manager.checkpoint(
                    run_state,
                    next_step_index=self.max_steps + 1,
                    actions=actions,
                    tool_results=tool_results,
                    pending_tool_results=pending_tool_results,
                    turn_tool_evidence=turn_tool_evidence,
                    transitions=transitions,
                    budgets=budgets,
                    progress_monitor=progress_monitor,
                    final_reply=final_reply,
                    runtime_status=runtime_status,
                )

        return self._result(
            message=message,
            trace_id=trace_id,
            actions=actions,
            tool_results=tool_results,
            transitions=transitions,
            final_reply=final_reply,
            runtime_status=runtime_status,
        )

    @staticmethod
    def _result(
        *,
        message: UserMessage,
        trace_id: str,
        actions: list[AgentAction],
        tool_results: list[ToolResult],
        transitions: list,
        final_reply: str,
        runtime_status: str,
    ) -> AgentRuntimeResult:
        """Aggregate restored and current steps into the public runtime result."""

        transitions.extend(
            transition
            for result in tool_results
            if not result.deduplicated
            for transition in result.state_transitions
        )
        return AgentRuntimeResult(
            trace_id=trace_id,
            conversation_id=message.conversation_id,
            final_reply=final_reply,
            status=runtime_status,
            actions=actions,
            tool_results=tool_results,
            state_transitions=transitions,
        )

    def fresh_turn_budgets(self) -> TurnBudgets:
        """Create isolated call counters while retaining configured token limits."""

        return fresh_turn_budgets(
            self.token_budget,
            self.review_token_budget,
            self.text_generation_token_budget,
        )

    def _step_service(self) -> AgentLoopStepService:
        progress_guard = ProgressGuardService(
            store=self.store,
            trace_recorder=self.trace_recorder,
            action_processor=self.action_processor,
            hook_manager=self.hook_manager,
        )
        return AgentLoopStepService(
            context_lifecycle=self.context_lifecycle,
            action_processor=self.action_processor,
            progress_guard=progress_guard,
        )

    def _progress_monitor(self) -> ProgressMonitor:
        return ProgressMonitor(
            repeated_observation_limit=self.repeated_observation_limit,
            consecutive_no_progress_limit=self.consecutive_no_progress_limit,
            max_replan_attempts=self.max_progress_replans,
            max_cycle_period=self.max_cycle_period,
        )

__all__ = ["AgentLoop"]
