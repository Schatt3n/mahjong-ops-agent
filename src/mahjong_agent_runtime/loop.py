from __future__ import annotations

"""The thin goal-driven agent loop."""

import json
from dataclasses import dataclass
from typing import Any

from .budget import TokenBudget
from .hooks import HookManager
from .lifecycle import ContextLifecycleManager
from .models import AgentRuntimeResult, StateTransition, ToolResult, UserMessage
from .processing import ActionProcessor
from .progress import ProgressDecision, ProgressMonitor
from .runtime_components import ProgressHandlingResult, TurnBudgets
from .store import InMemoryAgentStore
from .task_context import TaskContextManager


@dataclass(slots=True)
class AgentLoop:
    """Run buildContext -> callLLM -> executeTools/appendResults until terminal."""

    store: InMemoryAgentStore
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

    def run(self, message: UserMessage, *, trace_id: str, run_id: str, run_version: int) -> AgentRuntimeResult:
        budgets = self.fresh_turn_budgets()
        pre_model_transitions: list[StateTransition] = []
        task_context = self.task_context_manager.prepare(message, trace_id=trace_id)
        pre_model_transitions.extend(task_context.transitions)
        self.trace_recorder.record(trace_id, "task_context_prepared", task_context.to_dict())
        self._emit("after_task_context_prepared", trace_id=trace_id, payload=task_context.to_dict())
        self.store.append_user_turn(message, trace_id)
        self.trace_recorder.record(trace_id, "user_input", {"message": message.to_dict()})
        self._emit("after_user_turn_appended", trace_id=trace_id, payload={"message": message.to_dict()})

        actions = []
        tool_results: list[ToolResult] = []
        pending_tool_results: list[ToolResult] = []
        final_reply = ""
        progress_monitor = ProgressMonitor(
            repeated_observation_limit=self.repeated_observation_limit,
            consecutive_no_progress_limit=self.consecutive_no_progress_limit,
            max_replan_attempts=self.max_progress_replans,
            max_cycle_period=self.max_cycle_period,
        )

        for step_index in range(1, self.max_steps + 1):
            built = self.context_lifecycle.build_and_trace_context(
                message,
                trace_id=trace_id,
                pending_tool_results=pending_tool_results,
                run_id=run_id,
                run_version=run_version,
                step_index=step_index,
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
            )
            if summary_transition is not None:
                pre_model_transitions.append(summary_transition)

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
                final_reply = model_step.final_reply or ""
                break

            action = model_step.action
            if action is None:
                final_reply = "这个我先转人工确认一下。"
                break
            actions.append(action)

            if model_step.errors:
                pending_tool_results = self.action_processor.record_action_contract_feedback(
                    message,
                    trace_id=trace_id,
                    raw_response=model_step.raw_response,
                    errors=model_step.errors,
                    step_index=step_index,
                )
                decision = progress_monitor.observe_runtime_feedback(
                    "action_contract_error",
                    {"errors": model_step.errors},
                    step_index=step_index,
                )
                progress_handling = self._handle_progress_decision(
                    decision,
                    message=message,
                    trace_id=trace_id,
                    pending_tool_results=pending_tool_results,
                    run_id=run_id,
                    run_version=run_version,
                    progress_monitor=progress_monitor,
                )
                pending_tool_results = progress_handling.pending_tool_results
                if progress_handling.guard_result is not None:
                    tool_results.append(progress_handling.guard_result)
                if progress_handling.stop_loop:
                    final_reply = progress_handling.final_reply or ""
                    break
                continue

            self.action_processor.trace_action_plan(
                action,
                trace_id=trace_id,
                step_index=step_index,
                previous_tool_result_count=len(pending_tool_results),
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
            actions[-1] = processed.action
            tool_results.extend(processed.tool_results)
            pending_tool_results = processed.pending_tool_results
            if processed.stop_loop:
                final_reply = processed.final_reply or ""
                break
            if processed.continue_loop:
                decision = progress_monitor.observe_action(
                    processed.action,
                    processed.tool_results,
                    step_index=step_index,
                )
                progress_handling = self._handle_progress_decision(
                    decision,
                    message=message,
                    trace_id=trace_id,
                    pending_tool_results=pending_tool_results,
                    run_id=run_id,
                    run_version=run_version,
                    progress_monitor=progress_monitor,
                )
                pending_tool_results = progress_handling.pending_tool_results
                if progress_handling.guard_result is not None:
                    tool_results.append(progress_handling.guard_result)
                if progress_handling.stop_loop:
                    final_reply = progress_handling.final_reply or ""
                    break
                continue
        else:
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
                {"reply": final_reply, "reason": "max_steps_exceeded"},
                level="WARN",
            )

        transitions = pre_model_transitions + [
            transition
            for result in tool_results
            if not result.deduplicated
            for transition in result.state_transitions
        ]
        return AgentRuntimeResult(
            trace_id=trace_id,
            conversation_id=message.conversation_id,
            final_reply=final_reply,
            actions=actions,
            tool_results=tool_results,
            state_transitions=transitions,
        )

    def fresh_turn_budgets(self) -> TurnBudgets:
        return TurnBudgets(
            agent=TokenBudget(
                max_tokens_per_call=self.token_budget.max_tokens_per_call,
                max_calls_per_turn=self.token_budget.max_calls_per_turn,
            ),
            review=TokenBudget(
                max_tokens_per_call=self.review_token_budget.max_tokens_per_call,
                max_calls_per_turn=self.review_token_budget.max_calls_per_turn,
            ),
            text_generation=TokenBudget(
                max_tokens_per_call=self.text_generation_token_budget.max_tokens_per_call,
                max_calls_per_turn=self.text_generation_token_budget.max_calls_per_turn,
            ),
        )

    def _handle_progress_decision(
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

        decision_payload = decision.to_dict()
        self.trace_recorder.record(trace_id, "agent_progress_checked", decision_payload)
        self._emit("after_progress_check", trace_id=trace_id, payload=decision_payload)
        if not decision.detected:
            return ProgressHandlingResult(pending_tool_results=list(pending_tool_results))

        guard_result = progress_monitor.feedback_result(decision)
        next_pending_results = [*pending_tool_results, guard_result]
        self.store.append_tool_turn(
            message.conversation_id,
            json.dumps([guard_result.to_dict()], ensure_ascii=False),
            trace_id,
        )
        self.trace_recorder.record(trace_id, "agent_loop_detected", decision_payload, level="WARN")
        self.trace_recorder.record(trace_id, "tool_result", guard_result.to_dict(), level="WARN")

        if decision.should_replan:
            self.trace_recorder.record(
                trace_id,
                "agent_replan_requested",
                {**decision_payload, "feedback": guard_result.to_dict()},
                level="WARN",
            )
            return ProgressHandlingResult(
                pending_tool_results=next_pending_results,
                guard_result=guard_result,
            )

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
            "agent_loop_aborted",
            {**decision_payload, "run_id": run_id, "run_version": run_version},
            level="ERROR",
        )
        self.trace_recorder.record(
            trace_id,
            "final_output",
            {"reply": final_reply, "reason": "agent_loop_no_progress"},
            level="WARN",
        )
        return ProgressHandlingResult(
            pending_tool_results=next_pending_results,
            guard_result=guard_result,
            final_reply=final_reply,
            stop_loop=True,
        )

    def _emit(self, event_name: str, *, trace_id: str, payload: dict[str, Any]) -> None:
        if self.hook_manager is not None:
            self.hook_manager.emit(event_name, trace_id=trace_id, payload=payload)
