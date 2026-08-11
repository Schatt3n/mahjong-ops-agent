"""为旧版 ``AgentRuntime`` 调用提供兼容委托，不在此承载新的业务逻辑。"""

from __future__ import annotations

import threading
from typing import Any

from .budget import TokenBudget
from .context import BuiltContext
from .hooks import HookManager
from .models import AgentAction, AgentRuntimeResult, StateTransition, ToolResult, UserMessage
from .runtime_components import ActionProcessingResult, ModelActionStep, TurnBudgets
from .services import ActionProcessor, AgentLoop, ContextLifecycleManager, ToolExecutionService
from .stores import AgentStore
from .visibility import CustomerVisibleProcessor


class RuntimeCompatibilityMixin:
    """将历史 AgentRuntime 辅助函数委托给其所属服务。"""

    __slots__ = ()

    store: AgentStore
    hook_manager: HookManager
    agent_loop: AgentLoop
    context_lifecycle: ContextLifecycleManager
    action_processor: ActionProcessor
    tool_execution_service: ToolExecutionService
    _conversation_locks: dict[str, threading.RLock]
    _conversation_locks_guard: threading.RLock

    def _handle_once(
        self, message: UserMessage, *, trace_id: str, run_id: str, run_version: int
    ) -> AgentRuntimeResult:
        return self.agent_loop.run(message, trace_id=trace_id, run_id=run_id, run_version=run_version)

    def _fresh_turn_budgets(self) -> TurnBudgets:
        return self.agent_loop.fresh_turn_budgets()

    def _build_and_trace_context(
        self,
        message: UserMessage,
        *,
        trace_id: str,
        pending_tool_results: list[ToolResult],
        run_id: str,
        run_version: int,
        step_index: int,
    ) -> BuiltContext:
        return self.context_lifecycle.build_and_trace_context(
            message,
            trace_id=trace_id,
            pending_tool_results=pending_tool_results,
            run_id=run_id,
            run_version=run_version,
            step_index=step_index,
        )

    def _summarize_and_rebuild_context_if_needed(
        self,
        message: UserMessage,
        *,
        built: BuiltContext,
        trace_id: str,
        pending_tool_results: list[ToolResult],
        run_id: str,
        run_version: int,
        step_index: int,
        budget: TokenBudget,
    ) -> tuple[BuiltContext, StateTransition | None]:
        return self.context_lifecycle.summarize_and_rebuild_context_if_needed(
            message,
            built=built,
            trace_id=trace_id,
            pending_tool_results=pending_tool_results,
            run_id=run_id,
            run_version=run_version,
            step_index=step_index,
            budget=budget,
        )

    def _call_agent_action(
        self,
        message: UserMessage,
        *,
        trace_id: str,
        built_messages: list[dict[str, str]],
        step_index: int,
        budget: TokenBudget,
        run_id: str,
        run_version: int,
    ) -> ModelActionStep:
        return self.action_processor.call_agent_action(
            message,
            trace_id=trace_id,
            built_messages=built_messages,
            step_index=step_index,
            budget=budget,
            run_id=run_id,
            run_version=run_version,
        )

    def _record_action_contract_feedback(
        self,
        message: UserMessage,
        *,
        trace_id: str,
        raw_response: str,
        errors: list[str],
        step_index: int,
    ) -> list[ToolResult]:
        return self.action_processor.record_action_contract_feedback(
            message,
            trace_id=trace_id,
            raw_response=raw_response,
            errors=errors,
            step_index=step_index,
        )

    def _trace_action_plan(
        self,
        action: AgentAction,
        *,
        trace_id: str,
        step_index: int,
        previous_tool_result_count: int,
    ) -> None:
        self.action_processor.trace_action_plan(
            action,
            trace_id=trace_id,
            step_index=step_index,
            previous_tool_result_count=previous_tool_result_count,
        )

    def _process_tool_action(
        self,
        action: AgentAction,
        *,
        message: UserMessage,
        trace_id: str,
        context_payload: dict[str, Any],
        previous_pending_tool_results: list[ToolResult],
        step_index: int,
        budgets: TurnBudgets,
        run_id: str,
        run_version: int,
    ) -> ActionProcessingResult:
        return self.action_processor.process_tool_action(
            action,
            message=message,
            trace_id=trace_id,
            context_payload=context_payload,
            previous_pending_tool_results=previous_pending_tool_results,
            step_index=step_index,
            budgets=budgets,
            run_id=run_id,
            run_version=run_version,
        )

    def _execute_tool_calls(
        self,
        action: AgentAction,
        *,
        message: UserMessage,
        trace_id: str,
        previous_step_tool_results: list[ToolResult],
        step_index: int,
        run_id: str,
        run_version: int,
    ) -> ActionProcessingResult:
        return self.tool_execution_service.execute_tool_calls(
            action,
            message=message,
            trace_id=trace_id,
            previous_step_tool_results=previous_step_tool_results,
            step_index=step_index,
            run_id=run_id,
            run_version=run_version,
        )

    def _process_reply_action(
        self,
        action: AgentAction,
        *,
        message: UserMessage,
        trace_id: str,
        context_payload: dict[str, Any],
        budgets: TurnBudgets,
        run_id: str,
        run_version: int,
    ) -> ActionProcessingResult:
        return self.action_processor.process_reply_action(
            action,
            message=message,
            trace_id=trace_id,
            context_payload=context_payload,
            budgets=budgets,
            run_id=run_id,
            run_version=run_version,
        )

    def _apply_customer_visible_rewrites(
        self, action: AgentAction, result: ToolResult, *, trace_id: str
    ) -> AgentAction:
        return self.action_processor.apply_customer_visible_rewrites(
            action, result, trace_id=trace_id
        )

    @staticmethod
    def _customer_visible_rewrites(result: ToolResult) -> dict[str, str]:
        return ActionProcessor.customer_visible_rewrites(result)

    def _run_is_stale(self, conversation_id: str, run_version: int) -> bool:
        return self.store.conversation_version(conversation_id) != int(run_version)

    def _stale_write_tool_result(
        self,
        *,
        call_name: str,
        conversation_id: str,
        run_id: str,
        run_version: int,
    ) -> ToolResult | None:
        message = UserMessage(
            conversation_id=conversation_id,
            sender_id="",
            sender_name="",
            text="",
        )
        return self.tool_execution_service.stale_write_tool_result(
            call_name=call_name,
            message=message,
            run_id=run_id,
            run_version=run_version,
        )

    def _append_pending_assistant_turn(
        self,
        conversation_id: str,
        text: str,
        trace_id: str,
        *,
        run_id: str,
        run_version: int,
    ) -> None:
        self.action_processor.append_pending_assistant_turn(
            conversation_id,
            text,
            trace_id,
            run_id=run_id,
            run_version=run_version,
        )

    def _conversation_lock(self, conversation_id: str) -> threading.RLock:
        key = conversation_id or "default"
        with self._conversation_locks_guard:
            lock = self._conversation_locks.get(key)
            if lock is None:
                lock = threading.RLock()
                self._conversation_locks[key] = lock
            return lock

    def _customer_visible_processor(self) -> CustomerVisibleProcessor:
        return self.action_processor.customer_visible_processor()

    def _run_customer_visible_text_generation(
        self,
        *,
        message: UserMessage,
        trace_id: str,
        action: AgentAction,
        items: list[dict[str, Any]],
        context_payload: dict[str, Any],
        turn_budget: TokenBudget,
        generation_scope: str,
    ) -> ToolResult | None:
        return self._customer_visible_processor().run_text_generation(
            message=message,
            trace_id=trace_id,
            action=action,
            items=items,
            context_payload=context_payload,
            turn_budget=turn_budget,
            generation_scope=generation_scope,
        )

    def _run_customer_visible_content_review(
        self,
        *,
        message: UserMessage,
        trace_id: str,
        action: AgentAction,
        review_items: list[dict[str, Any]],
        context_payload: dict[str, Any],
        turn_budget: TokenBudget,
        review_scope: str,
    ) -> ToolResult | None:
        return self._customer_visible_processor().run_content_review(
            message=message,
            trace_id=trace_id,
            action=action,
            review_items=review_items,
            context_payload=context_payload,
            turn_budget=turn_budget,
            review_scope=review_scope,
        )

    def _emit(self, event_name: str, trace_id: str, payload: dict[str, Any]) -> None:
        self.hook_manager.emit(event_name, trace_id=trace_id, payload=payload)


__all__ = ["RuntimeCompatibilityMixin"]
