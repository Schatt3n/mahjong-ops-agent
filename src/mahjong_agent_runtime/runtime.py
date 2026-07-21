from __future__ import annotations

"""Goal-driven Agent runtime entrypoint.

AgentRuntime is intentionally thin: it owns message entry, conversation locks,
idempotency, trace identity, run versioning, and result persistence. The actual
agent loop, action processing, tool execution, and context lifecycle are delegated
to focused components.
"""

import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .budget import TokenBudget
from .context import AgentContextBuilder
from .coordination import CoordinationManager, default_coordination_manager
from .copywriting import DEFAULT_CUSTOMER_VISIBLE_TEXT_PROMPT_PATH
from .hooks import HookManager
from .lifecycle import ContextLifecycleManager
from .llm import AgentLLMClient
from .loop import AgentLoop
from .models import AgentAction, AgentRuntimeResult, StateTransition, ToolResult, UserMessage
from .processing import ActionProcessor, ToolExecutionService
from .runtime_components import ActionProcessingResult, ModelActionStep, TurnBudgets
from .store import InMemoryAgentStore
from .stores import AgentStore
from .summary import ContextSummaryManager
from .task_context import TaskContextManager
from .tools import ToolGateway
from .tracing import InMemoryTraceRecorder
from .visibility import (
    DEFAULT_REPLY_SELF_REVIEW_PROMPT_PATH,
    CustomerVisibleProcessor,
    build_reply_self_review_payload,
    normalize_item_reviews,
)


@dataclass(slots=True)
class AgentRuntime:
    """Production boundary for one goal-driven Mahjong operation agent."""

    llm_client: AgentLLMClient
    store: AgentStore = field(default_factory=InMemoryAgentStore)
    tool_gateway: ToolGateway | None = None
    trace_recorder: Any = field(default_factory=InMemoryTraceRecorder)
    token_budget: TokenBudget = field(default_factory=TokenBudget)
    review_token_budget: TokenBudget = field(default_factory=TokenBudget)
    customer_visible_text_generation_token_budget: TokenBudget = field(default_factory=TokenBudget)
    max_steps: int = 8
    repeated_observation_limit: int = 2
    consecutive_no_progress_limit: int = 2
    max_progress_replans: int = 1
    max_cycle_period: int = 3
    max_parallel_read_tools: int = 4
    llm_timeout_seconds: float = 45.0
    context_summary_preemptive_ratio: float = 0.85
    task_context_idle_seconds: int = 4 * 60 * 60
    customer_visible_text_generation_enabled: bool = False
    customer_visible_text_generation_client: AgentLLMClient | None = None
    customer_visible_text_generation_prompt_path: Path = DEFAULT_CUSTOMER_VISIBLE_TEXT_PROMPT_PATH
    reply_self_review_enabled: bool = False
    reply_self_review_client: AgentLLMClient | None = None
    reply_self_review_prompt_path: Path = DEFAULT_REPLY_SELF_REVIEW_PROMPT_PATH
    context_summary_manager: ContextSummaryManager | None = None
    hook_manager: HookManager = field(default_factory=HookManager)
    coordination_manager: CoordinationManager | None = None
    context_builder: AgentContextBuilder = field(init=False)
    context_lifecycle: ContextLifecycleManager = field(init=False)
    task_context_manager: TaskContextManager = field(init=False)
    tool_execution_service: ToolExecutionService = field(init=False)
    action_processor: ActionProcessor = field(init=False)
    agent_loop: AgentLoop = field(init=False)
    _conversation_locks: dict[str, threading.RLock] = field(default_factory=dict, init=False, repr=False)
    _conversation_locks_guard: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.tool_gateway is None:
            self.tool_gateway = ToolGateway(self.store)
        if self.coordination_manager is None:
            self.coordination_manager = default_coordination_manager(self.store)
        if self.tool_gateway.trace_recorder is None:
            self.tool_gateway.trace_recorder = self.trace_recorder
        self.context_builder = AgentContextBuilder(self.store, self.tool_gateway)
        self.task_context_manager = TaskContextManager(
            self.store,
            idle_reset_seconds=self.task_context_idle_seconds,
        )
        self.context_lifecycle = ContextLifecycleManager(
            context_builder=self.context_builder,
            trace_recorder=self.trace_recorder,
            context_summary_manager=self.context_summary_manager,
            context_summary_preemptive_ratio=self.context_summary_preemptive_ratio,
            hook_manager=self.hook_manager,
        )
        self.tool_execution_service = ToolExecutionService(
            store=self.store,
            tool_gateway=self.tool_gateway,
            trace_recorder=self.trace_recorder,
            hook_manager=self.hook_manager,
            max_parallel_read_tools=self.max_parallel_read_tools,
        )
        self.action_processor = ActionProcessor(
            llm_client=self.llm_client,
            store=self.store,
            trace_recorder=self.trace_recorder,
            tool_execution_service=self.tool_execution_service,
            llm_timeout_seconds=self.llm_timeout_seconds,
            customer_visible_text_generation_enabled=self.customer_visible_text_generation_enabled,
            customer_visible_text_generation_client=self.customer_visible_text_generation_client,
            customer_visible_text_generation_prompt_path=self.customer_visible_text_generation_prompt_path,
            reply_self_review_enabled=self.reply_self_review_enabled,
            reply_self_review_client=self.reply_self_review_client,
            reply_self_review_prompt_path=self.reply_self_review_prompt_path,
            hook_manager=self.hook_manager,
        )
        self.agent_loop = AgentLoop(
            store=self.store,
            trace_recorder=self.trace_recorder,
            context_lifecycle=self.context_lifecycle,
            action_processor=self.action_processor,
            task_context_manager=self.task_context_manager,
            token_budget=self.token_budget,
            review_token_budget=self.review_token_budget,
            text_generation_token_budget=self.customer_visible_text_generation_token_budget,
            max_steps=self.max_steps,
            repeated_observation_limit=self.repeated_observation_limit,
            consecutive_no_progress_limit=self.consecutive_no_progress_limit,
            max_progress_replans=self.max_progress_replans,
            max_cycle_period=self.max_cycle_period,
            hook_manager=self.hook_manager,
        )

    def handle_user_message(self, message: UserMessage, *, trace_id: str | None = None) -> AgentRuntimeResult:
        """Handle one user message with per-conversation ordering and idempotency."""

        assert self.coordination_manager is not None
        with self.coordination_manager.lock(f"conversation:{message.conversation_id or 'default'}"):
            actual_trace_id = trace_id or f"trace_{uuid.uuid4().hex[:12]}"
            self._emit("message_received", actual_trace_id, {"message": message.to_dict()})
            message_key = message_idempotency_key(message)
            cached = self.store.idempotent_message_result(message_key)
            if cached is not None:
                self.trace_recorder.record(actual_trace_id, "user_input", {"message": message.to_dict()})
                self.trace_recorder.record(
                    actual_trace_id,
                    "message_deduplicated",
                    {
                        "message_id": message.message_id,
                        "message_idempotency_key": message_key,
                        "original_trace_id": cached.trace_id,
                    },
                )
                self.trace_recorder.record(
                    actual_trace_id,
                    "final_output",
                    {"reply": cached.final_reply, "reason": "message_deduplicated"},
                )
                self._emit("message_deduplicated", actual_trace_id, {"message_key": message_key})
                return cached

            run_id = f"run_{uuid.uuid4().hex[:12]}"
            run_version, version_transition = self.store.advance_conversation_version(
                message.conversation_id,
                trace_id=actual_trace_id,
                reason="user_message_received",
            )
            superseded_counts, superseded_transitions = self.store.supersede_pending_outputs(
                message.conversation_id,
                sender_id=message.sender_id,
                trace_id=actual_trace_id,
                reason="new_user_message_superseded_previous_pending_output",
            )
            self.trace_recorder.record(
                actual_trace_id,
                "conversation_version_advanced",
                {
                    "conversation_id": message.conversation_id,
                    "run_id": run_id,
                    "run_version": run_version,
                    "transition": version_transition.to_dict(),
                },
            )
            self.trace_recorder.record(
                actual_trace_id,
                "pending_outputs_superseded",
                {
                    "conversation_id": message.conversation_id,
                    "run_id": run_id,
                    "run_version": run_version,
                    "counts": superseded_counts,
                    "transitions": [item.to_dict() for item in superseded_transitions],
                },
            )
            self._emit(
                "before_agent_loop",
                actual_trace_id,
                {"run_id": run_id, "run_version": run_version, "message": message.to_dict()},
            )
            result = self.agent_loop.run(
                message,
                trace_id=actual_trace_id,
                run_id=run_id,
                run_version=run_version,
            )
            self._emit(
                "after_agent_loop",
                actual_trace_id,
                {"run_id": run_id, "run_version": run_version, "result": result.to_dict()},
            )
            result.state_transitions = [version_transition, *superseded_transitions, *result.state_transitions]
            try:
                summary_result = self.context_lifecycle.maybe_summarize_after_turn(
                    conversation_id=message.conversation_id,
                    trace_id=actual_trace_id,
                )
                if summary_result is not None and summary_result.transition is not None:
                    result.state_transitions.append(summary_result.transition)
            except Exception as exc:
                self.trace_recorder.record(
                    actual_trace_id,
                    "context_summary_error",
                    {"error_type": type(exc).__name__, "error": str(exc)},
                    level="ERROR",
                )
            self.store.remember_message_result(message_key, result)
            self._emit(
                "after_turn_finished",
                actual_trace_id,
                {"run_id": run_id, "run_version": run_version, "result": result.to_dict()},
            )
            return result

    def handle_system_event(self, message: UserMessage, *, trace_id: str | None = None) -> AgentRuntimeResult:
        """Re-enter the same goal-driven loop for durable background work.

        Unlike a new customer message, a system event does not advance the
        conversation version and does not supersede pending customer output.
        The event is still ordered by the conversation lock, traced and
        idempotent. ``ActionProcessor`` keeps its terminal reply internal while
        customer-facing invite drafts still pass normal generation and review.
        """

        assert self.coordination_manager is not None
        metadata = dict(message.metadata or {})
        metadata.update({"internal_event": True, "delivery_mode": "internal_only"})
        message.metadata = metadata
        with self.coordination_manager.lock(f"conversation:{message.conversation_id or 'default'}"):
            actual_trace_id = trace_id or f"trace_system_{uuid.uuid4().hex[:12]}"
            message_key = message_idempotency_key(message)
            cached = self.store.idempotent_message_result(message_key)
            if cached is not None:
                self.trace_recorder.record(
                    actual_trace_id,
                    "system_event_deduplicated",
                    {"message_id": message.message_id, "original_trace_id": cached.trace_id},
                )
                return cached

            run_id = f"run_system_{uuid.uuid4().hex[:12]}"
            run_version = self.store.conversation_version(message.conversation_id)
            self.trace_recorder.record(
                actual_trace_id,
                "system_event_received",
                {
                    "message": message.to_dict(),
                    "run_id": run_id,
                    "run_version": run_version,
                },
            )
            result = self.agent_loop.run(
                message,
                trace_id=actual_trace_id,
                run_id=run_id,
                run_version=run_version,
            )
            try:
                summary_result = self.context_lifecycle.maybe_summarize_after_turn(
                    conversation_id=message.conversation_id,
                    trace_id=actual_trace_id,
                )
                if summary_result is not None and summary_result.transition is not None:
                    result.state_transitions.append(summary_result.transition)
            except Exception as exc:
                self.trace_recorder.record(
                    actual_trace_id,
                    "context_summary_error",
                    {"error_type": type(exc).__name__, "error": str(exc), "trigger": "system_event"},
                    level="ERROR",
                )
            self.store.remember_message_result(message_key, result)
            self.trace_recorder.record(
                actual_trace_id,
                "system_event_completed",
                {"result": result.to_dict(), "delivery_mode": "internal_only"},
            )
            return result

    def _handle_once(self, message: UserMessage, *, trace_id: str, run_id: str, run_version: int) -> AgentRuntimeResult:
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
    ) -> Any:
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
        built: Any,
        trace_id: str,
        pending_tool_results: list[ToolResult],
        run_id: str,
        run_version: int,
        step_index: int,
        budget: TokenBudget,
    ) -> tuple[Any, StateTransition | None]:
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

    def _apply_customer_visible_rewrites(self, action: AgentAction, result: ToolResult, *, trace_id: str) -> AgentAction:
        return self.action_processor.apply_customer_visible_rewrites(action, result, trace_id=trace_id)

    @staticmethod
    def _customer_visible_rewrites(result: ToolResult) -> dict[str, str]:
        return ActionProcessor.customer_visible_rewrites(result)

    def _run_is_stale(self, conversation_id: str, run_version: int) -> bool:
        return self.action_processor.run_is_stale(conversation_id, run_version)

    def _stale_write_tool_result(
        self,
        *,
        call_name: str,
        conversation_id: str,
        run_id: str,
        run_version: int,
    ) -> ToolResult | None:
        return self.tool_execution_service.stale_write_tool_result(
            call_name=call_name,
            conversation_id=conversation_id,
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


def message_idempotency_key(message: UserMessage) -> str:
    """Build the backend idempotency key for one inbound user message."""

    return f"conversation:{message.conversation_id}:sender:{message.sender_id}:message:{message.message_id}"
