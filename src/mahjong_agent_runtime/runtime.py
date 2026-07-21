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
from .llm import AgentLLMClient
from .models import AgentRuntimeResult, UserMessage
from .runtime_compat import RuntimeCompatibilityMixin
from .services import ActionProcessor, AgentLoop, ContextLifecycleManager, ToolExecutionService
from .stores import AgentStore
from .stores.memory import InMemoryAgentStore
from .summary import ContextSummaryManager
from .task_context import TaskContextManager
from .domains.tools import ToolGateway
from .tracing import InMemoryTraceRecorder
from .visibility import DEFAULT_REPLY_SELF_REVIEW_PROMPT_PATH, build_reply_self_review_payload, normalize_item_reviews


@dataclass(slots=True)
class AgentRuntime(RuntimeCompatibilityMixin):
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

def message_idempotency_key(message: UserMessage) -> str:
    """Build the backend idempotency key for one inbound user message."""

    return f"conversation:{message.conversation_id}:sender:{message.sender_id}:message:{message.message_id}"
