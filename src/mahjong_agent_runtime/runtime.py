"""组装并驱动麻将运营智能体运行时，负责消息入口、会话协调、幂等恢复和主循环调度。

"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .agent_state import AgentRunState, restore_user_message
from .budget import TokenBudget
from .context import AgentContextBuilder
from .coordination import CoordinationManager, default_coordination_manager
from .copywriting import DEFAULT_CUSTOMER_VISIBLE_TEXT_PROMPT_PATH
from .hooks import HookManager
from .group_chat import GroupBoardTrigger
from .llm import AgentLLMClient
from .matching import MatchTrigger, OutboundDispatcher
from .models import AgentRuntimeResult, SystemTriggerMessage, UserMessage
from .runtime_compat import RuntimeCompatibilityMixin
from .services import (
    ActionProcessor,
    AgentLoop,
    AgentRunStateManager,
    ContextLifecycleManager,
    ToolExecutionService,
)
from .stores import AgentStore
from .stores.memory import InMemoryAgentStore
from .summary import ContextSummaryManager
from .task_context import TaskContextManager
from .domains.tools import ToolGateway
from .tracing import InMemoryTraceRecorder
from .visibility import DEFAULT_REPLY_SELF_REVIEW_PROMPT_PATH, build_reply_self_review_payload, normalize_item_reviews


@dataclass(slots=True)
class AgentRuntime(RuntimeCompatibilityMixin):
    """麻将运营智能体的统一运行入口，负责可靠接收消息并把业务决策交给主循环。

"""

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
    agent_run_lease_seconds: int = 120
    agent_run_max_attempts: int = 3
    context_summary_preemptive_ratio: float = 0.85
    task_context_idle_seconds: int = 4 * 60 * 60
    customer_visible_text_generation_enabled: bool = False
    customer_visible_text_generation_client: AgentLLMClient | None = None
    customer_visible_text_generation_prompt_path: Path = DEFAULT_CUSTOMER_VISIBLE_TEXT_PROMPT_PATH
    reply_self_review_enabled: bool = False
    reply_self_review_client: AgentLLMClient | None = None
    reply_self_review_prompt_path: Path = DEFAULT_REPLY_SELF_REVIEW_PROMPT_PATH
    context_summary_manager: ContextSummaryManager | None = None
    recovered_result_handler: Callable[[UserMessage, AgentRuntimeResult, AgentRunState], None] | None = None
    hook_manager: HookManager = field(default_factory=HookManager)
    coordination_manager: CoordinationManager | None = None
    context_builder: AgentContextBuilder = field(init=False)
    context_lifecycle: ContextLifecycleManager = field(init=False)
    task_context_manager: TaskContextManager = field(init=False)
    tool_execution_service: ToolExecutionService = field(init=False)
    action_processor: ActionProcessor = field(init=False)
    run_state_manager: AgentRunStateManager = field(init=False)
    agent_loop: AgentLoop = field(init=False)
    outbound_dispatcher: OutboundDispatcher = field(init=False)
    match_trigger: MatchTrigger = field(init=False)
    group_board_trigger: GroupBoardTrigger = field(init=False)
    _conversation_locks: dict[str, threading.RLock] = field(default_factory=dict, init=False, repr=False)
    _conversation_locks_guard: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        """组装一次运行所需的可靠性组件，并把业务能力挂到统一主循环上。

        ``AgentRuntime`` 是组合根而不是业务决策者：模型决策、上下文构建、
        工具执行、运行检查点、主动通知分别由独立组件承担，这里只负责连线，
        从而避免把麻将领域判断重新堆回运行时入口。
        """
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
        self.run_state_manager = AgentRunStateManager(
            store=self.store,
            trace_recorder=self.trace_recorder,
            lease_seconds=max(
                int(self.agent_run_lease_seconds),
                int(self.llm_timeout_seconds * 2) + 30,
            ),
            max_attempts=self.agent_run_max_attempts,
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
            run_state_manager=self.run_state_manager,
        )
        self.outbound_dispatcher = OutboundDispatcher(
            store=self.store,
            system_trigger_handler=self.handle_system_trigger,
            trace_recorder=self.trace_recorder,
        )
        self.match_trigger = MatchTrigger(
            store=self.store,
            dispatcher=self.outbound_dispatcher,
            trace_recorder=self.trace_recorder,
        )
        self.group_board_trigger = GroupBoardTrigger(
            store=self.store,
            trace_recorder=self.trace_recorder,
        )
        self.hook_manager.register("after_tool_execute", self.match_trigger)
        self.hook_manager.register("after_tool_execute", self.group_board_trigger)

    def handle_user_message(self, message: UserMessage, *, trace_id: str | None = None) -> AgentRuntimeResult:
        """处理一条用户消息：执行消息幂等、会话串行化、输入持久化并驱动主循环。

"""

        assert self.coordination_manager is not None
        # 外部通道不能通过伪造 metadata 获得历史任务恢复权限；只有可信系统入口
        # 才能写入这个私有字段，防止用户把不属于自己的任务上下文带入本轮。
        public_metadata = dict(message.metadata or {})
        public_metadata.pop("_trusted_source_task_context_id", None)
        message.metadata = public_metadata
        # 同一会话串行执行，避免多条消息交错修改状态；不同会话仍可并行处理。
        with self.coordination_manager.lock(f"conversation:{message.conversation_id or 'default'}"):
            actual_trace_id = trace_id or f"trace_{uuid.uuid4().hex[:12]}"
            self._emit("message_received", actual_trace_id, {"message": message.to_dict()})
            message_key = message_idempotency_key(message)
            # 通道重投或网络重试时直接返回第一次结果，不重复调用模型和有副作用工具。
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
            # 新用户输入代表会话事实可能发生变化，因此推进版本，并淘汰旧版本尚未
            # 发出的草稿或运行；这样晚到的旧结果不会覆盖用户刚补充的新条件。
            run_version, version_transition = self.store.advance_conversation_version(
                message.conversation_id,
                trace_id=actual_trace_id,
                reason="user_message_received",
            )
            run_superseded_transitions = self.run_state_manager.supersede_stale(
                message.conversation_id,
                current_version=run_version,
                trace_id=actual_trace_id,
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
            result = self._execute_agent_loop(
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
            result.state_transitions = [
                version_transition,
                *run_superseded_transitions,
                *superseded_transitions,
                *result.state_transitions,
            ]
            # 摘要属于回合后的维护工作。失败只记录审计，不影响已经生成的业务回复。
            try:
                summary_result = self.context_lifecycle.maybe_summarize_after_turn(
                    conversation_id=message.conversation_id,
                    trace_id=actual_trace_id,
                    task_context_id=self.context_lifecycle.summary_task_context_id(message),
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
            self.run_state_manager.complete(
                run_id,
                final_reply=result.final_reply,
                runtime_status=result.status,
            )
            self._emit(
                "after_turn_finished",
                actual_trace_id,
                {"run_id": run_id, "run_version": run_version, "result": result.to_dict()},
            )
            return result

    def handle_system_event(self, message: UserMessage, *, trace_id: str | None = None) -> AgentRuntimeResult:
        """让定时任务等可信内部事件复用主循环，但不制造一轮新的客户对话。

        内部事件不会推进会话版本，也不会废弃待发送的客户文本；它仍然接受会话锁、
        幂等和审计约束。主循环产生的终止回复仅供内部记录，但事件触发的邀约草稿等
        客户可见内容仍必须经过正常的话术生成与审查。
        """

        assert self.coordination_manager is not None
        metadata = dict(message.metadata or {})
        metadata.update({"internal_event": True, "delivery_mode": "internal_only"})
        source_task_context_id = str(metadata.get("task_context_id") or "")
        if source_task_context_id:
            metadata["_trusted_source_task_context_id"] = source_task_context_id
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
            result = self._execute_agent_loop(
                message,
                trace_id=actual_trace_id,
                run_id=run_id,
                run_version=run_version,
            )
            try:
                summary_result = self.context_lifecycle.maybe_summarize_after_turn(
                    conversation_id=message.conversation_id,
                    trace_id=actual_trace_id,
                    task_context_id=self.context_lifecycle.summary_task_context_id(message),
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
            self.run_state_manager.complete(
                run_id,
                final_reply=result.final_reply,
                runtime_status=result.status,
            )
            self.trace_recorder.record(
                actual_trace_id,
                "system_event_completed",
                {"result": result.to_dict(), "delivery_mode": "internal_only"},
            )
            return result

    def handle_system_trigger(
        self,
        trigger: SystemTriggerMessage,
        *,
        trace_id: str | None = None,
    ) -> AgentRuntimeResult:
        """处理会主动联系客户的系统触发，并沿用正常的生成、审查与发送链路。

        它与内部事件的区别是最终文本要给客户看；与新用户消息的区别是它不会推进
        会话版本，也不能覆盖更新的客户输入。跨会话匹配唤醒目标客户时，仍先获取
        目标会话锁，保证主动通知和该客户正在发送的消息顺序一致。
        """

        assert self.coordination_manager is not None
        message = trigger.to_user_message()
        source_task_context_id = str(trigger.payload.get("task_context_id") or "")
        if source_task_context_id:
            message.metadata = {
                **dict(message.metadata),
                "_trusted_source_task_context_id": source_task_context_id,
            }
        with self.coordination_manager.lock(f"conversation:{trigger.conversation_id or 'default'}"):
            actual_trace_id = trace_id or f"trace_trigger_{uuid.uuid4().hex[:12]}"
            message_key = message_idempotency_key(message)
            cached = self.store.idempotent_message_result(message_key)
            if cached is not None:
                self.trace_recorder.record(
                    actual_trace_id,
                    "system_trigger_deduplicated",
                    {
                        "trigger_id": trigger.trigger_id,
                        "message_id": message.message_id,
                        "original_trace_id": cached.trace_id,
                    },
                )
                return cached

            run_id = f"run_trigger_{uuid.uuid4().hex[:12]}"
            run_version = self.store.conversation_version(trigger.conversation_id)
            self.trace_recorder.record(
                actual_trace_id,
                "system_trigger_received",
                {
                    "trigger": trigger.to_dict(),
                    "run_id": run_id,
                    "run_version": run_version,
                },
            )
            self._emit(
                "before_agent_loop",
                actual_trace_id,
                {"run_id": run_id, "run_version": run_version, "message": message.to_dict()},
            )
            result = self._execute_agent_loop(
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
            try:
                summary_result = self.context_lifecycle.maybe_summarize_after_turn(
                    conversation_id=trigger.conversation_id,
                    trace_id=actual_trace_id,
                    task_context_id=self.context_lifecycle.summary_task_context_id(message),
                )
                if summary_result is not None and summary_result.transition is not None:
                    result.state_transitions.append(summary_result.transition)
            except Exception as exc:
                self.trace_recorder.record(
                    actual_trace_id,
                    "context_summary_error",
                    {"error_type": type(exc).__name__, "error": str(exc), "trigger": "system_trigger"},
                    level="ERROR",
                )
            self.store.remember_message_result(message_key, result)
            self.run_state_manager.complete(
                run_id,
                final_reply=result.final_reply,
                runtime_status=result.status,
            )
            self.trace_recorder.record(
                actual_trace_id,
                "system_trigger_completed",
                {"trigger_id": trigger.trigger_id, "result": result.to_dict()},
            )
            self._emit(
                "after_turn_finished",
                actual_trace_id,
                {"run_id": run_id, "run_version": run_version, "result": result.to_dict()},
            )
            return result

    def resume_recoverable_runs(
        self,
        limit: int = 20,
        *,
        at: datetime | None = None,
    ) -> list[AgentRuntimeResult]:
        """领取租约过期或中断的检查点，从最后一个已完成步骤继续执行。

        恢复前会再次核对会话版本；如果客户已经发送了更新消息，旧运行会被标记为
        过期而不是重放，避免宕机恢复把旧条件重新写回当前局。
        """

        assert self.coordination_manager is not None
        recovered: list[AgentRuntimeResult] = []
        candidates = self.run_state_manager.recoverable(at=at, limit=limit)
        for candidate in candidates:
            with self.coordination_manager.lock(
                f"conversation:{candidate.conversation_id or 'default'}"
            ):
                claimed = self.run_state_manager.claim(candidate.run_id, at=at)
                if claimed is None:
                    continue
                current_version = self.store.conversation_version(claimed.conversation_id)
                if current_version != claimed.run_version:
                    self.run_state_manager.supersede_stale(
                        claimed.conversation_id,
                        current_version=current_version,
                        trace_id=claimed.trace_id,
                    )
                    continue
                message = restore_user_message(claimed.message)
                message_key = message_idempotency_key(message)
                cached = self.store.idempotent_message_result(message_key)
                try:
                    result = cached or self._execute_agent_loop(
                        message,
                        trace_id=claimed.trace_id,
                        run_id=claimed.run_id,
                        run_version=claimed.run_version,
                        run_state=claimed,
                    )
                    self._finalize_recovered_result(
                        run_state=claimed,
                        message=message,
                        message_key=message_key,
                        result=result,
                        already_durable=cached is not None,
                    )
                    recovered.append(result)
                except Exception as exc:
                    # 无论失败发生在主循环、持久化还是恢复后的通道投递，都保留同一个
                    # 终止检查点供有界重试，避免重复执行已经成功的前序工具。
                    self.run_state_manager.mark_recoverable(claimed.run_id, error=exc)
                    continue
        return recovered

    def _finalize_recovered_result(
        self,
        *,
        run_state: AgentRunState,
        message: UserMessage,
        message_key: str,
        result: AgentRuntimeResult,
        already_durable: bool,
    ) -> None:
        """持久化恢复结果并补齐投递、钩子和运行终态，不重复已落库的工作。"""

        if not already_durable:
            try:
                summary_result = self.context_lifecycle.maybe_summarize_after_turn(
                    conversation_id=message.conversation_id,
                    trace_id=run_state.trace_id,
                    task_context_id=self.context_lifecycle.summary_task_context_id(message),
                )
                if summary_result is not None and summary_result.transition is not None:
                    result.state_transitions.append(summary_result.transition)
            except Exception as exc:
                self.trace_recorder.record(
                    run_state.trace_id,
                    "context_summary_error",
                    {
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "trigger": "agent_run_recovery",
                    },
                    level="ERROR",
                )
            self.store.remember_message_result(message_key, result)

        self._emit(
            "after_agent_loop",
            run_state.trace_id,
            {
                "run_id": run_state.run_id,
                "run_version": run_state.run_version,
                "recovered": True,
                "result": result.to_dict(),
            },
        )
        if self.recovered_result_handler is not None:
            self.recovered_result_handler(message, result, run_state)
        self.run_state_manager.complete(
            run_state.run_id,
            final_reply=result.final_reply,
            runtime_status=result.status,
        )
        self.trace_recorder.record(
            run_state.trace_id,
            "agent_run_recovery_completed",
            {
                "run_id": run_state.run_id,
                "run_version": run_state.run_version,
                "result": result.to_dict(),
            },
        )
        self._emit(
            "after_turn_finished",
            run_state.trace_id,
            {
                "run_id": run_state.run_id,
                "run_version": run_state.run_version,
                "recovered": True,
                "result": result.to_dict(),
            },
        )

    def _execute_agent_loop(
        self,
        message: UserMessage,
        *,
        trace_id: str,
        run_id: str,
        run_version: int,
        run_state: AgentRunState | None = None,
    ) -> AgentRuntimeResult:
        """调用主循环；运行时只负责可靠性外壳，不在这里编写业务语义分支。

"""

        state = run_state or self.run_state_manager.start(
            message,
            trace_id=trace_id,
            run_id=run_id,
            run_version=run_version,
        )
        try:
            return self.agent_loop.run(
                message,
                trace_id=trace_id,
                run_id=run_id,
                run_version=run_version,
                run_state=state,
            )
        except Exception as exc:
            self.run_state_manager.mark_recoverable(run_id, error=exc)
            raise


def message_idempotency_key(message: UserMessage) -> str:
    """从通道、会话和消息标识生成稳定幂等键，防止同一输入被重复处理。

"""

    return f"conversation:{message.conversation_id}:sender:{message.sender_id}:message:{message.message_id}"
