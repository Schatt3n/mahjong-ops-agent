from __future__ import annotations

"""Goal-driven Agent 主运行时。

设计理念：
- 主 loop 只负责编排，不承载具体业务语义规则。
- 模型负责理解用户目标、规划下一步、提出工具调用或回复。
- 后端负责合同校验、预算控制、工具权限、幂等、顺序、状态落库和审计。
- 凡是客户可见文本，不管来自最终回复还是工具参数，都先经过话术生成和安全审查。
"""

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .action_contract import parse_action
from .budget import TokenBudget
from .context import AgentContextBuilder, estimate_tokens
from .copywriting import (
    DEFAULT_CUSTOMER_VISIBLE_TEXT_PROMPT_PATH,
    action_with_customer_visible_rewrites,
)
from .llm import AgentLLMClient
from .models import AgentAction, AgentRuntimeResult, StateTransition, ToolResult, UserMessage
from .store import InMemoryAgentStore
from .summary import ContextSummaryManager
from .tool_consistency import latest_read_requirement, validate_tool_call_consistency
from .tools import ToolGateway
from .tracing import InMemoryTraceRecorder
from .visibility import (
    CUSTOMER_VISIBLE_CONTENT_REVIEW_TOOL_NAME,
    CUSTOMER_VISIBLE_TEXT_GENERATION_NAME,
    DEFAULT_REPLY_SELF_REVIEW_PROMPT_PATH,
    REPLY_SELF_REVIEW_TOOL_NAME,
    CustomerVisibleProcessor,
    build_reply_self_review_payload,
    customer_visible_content_review_approved,
    customer_visible_items_for_action,
    normalize_item_reviews,
)


@dataclass(slots=True)
class TurnBudgets:
    """一次用户消息处理过程中的三类 LLM 预算。

    agent 用于主模型循环；review 用于客户可见文本审查；text_generation 用于话术生成。
    三者分开统计，避免话术/审查把主 agent 的预算吃光，也方便后续单独调参。
    """

    agent: TokenBudget
    review: TokenBudget
    text_generation: TokenBudget


@dataclass(slots=True)
class ModelActionStep:
    """主模型单步输出的标准包装。

    一轮 LLM 调用可能得到合法 AgentAction，也可能因为超时、预算、合同错误而直接结束。
    这个对象把这些结果收敛成主 loop 能理解的统一结构。
    """

    action: AgentAction | None
    raw_response: str = ""
    errors: list[str] = field(default_factory=list)
    final_reply: str | None = None
    stop_loop: bool = False


@dataclass(slots=True)
class ActionProcessingResult:
    """处理一个 AgentAction 后交还给主 loop 的结果。

    tool_results 是本轮真实或虚拟工具结果；pending_tool_results 会回喂给下一轮模型；
    final_reply 表示本次用户消息已经可以结束；continue_loop 表示工具结果还需要模型继续判断。
    """

    action: AgentAction
    tool_results: list[ToolResult] = field(default_factory=list)
    pending_tool_results: list[ToolResult] = field(default_factory=list)
    final_reply: str | None = None
    stop_loop: bool = False
    continue_loop: bool = False


@dataclass(slots=True)
class AgentRuntime:
    """Agent 运行时的总控编排器。

    它不是业务规则引擎，而是受控执行环境：接收用户消息，构建上下文，调用模型，
    校验模型输出，执行工具，把工具结果回喂模型，并保证状态写入、客户可见文本和日志审计安全可控。
    """

    llm_client: AgentLLMClient
    store: InMemoryAgentStore = field(default_factory=InMemoryAgentStore)
    tool_gateway: ToolGateway | None = None
    trace_recorder: Any = field(default_factory=InMemoryTraceRecorder)
    token_budget: TokenBudget = field(default_factory=TokenBudget)
    review_token_budget: TokenBudget = field(default_factory=TokenBudget)
    customer_visible_text_generation_token_budget: TokenBudget = field(
        default_factory=TokenBudget
    )
    max_steps: int = 8
    llm_timeout_seconds: float = 45.0
    context_summary_preemptive_ratio: float = 0.85
    customer_visible_text_generation_enabled: bool = False
    customer_visible_text_generation_client: AgentLLMClient | None = None
    customer_visible_text_generation_prompt_path: Path = (
        DEFAULT_CUSTOMER_VISIBLE_TEXT_PROMPT_PATH
    )
    reply_self_review_enabled: bool = False
    reply_self_review_client: AgentLLMClient | None = None
    reply_self_review_prompt_path: Path = DEFAULT_REPLY_SELF_REVIEW_PROMPT_PATH
    context_summary_manager: ContextSummaryManager | None = None
    context_builder: AgentContextBuilder = field(init=False)
    _conversation_locks: dict[str, threading.RLock] = field(
        default_factory=dict, init=False, repr=False
    )
    _conversation_locks_guard: threading.RLock = field(
        default_factory=threading.RLock, init=False, repr=False
    )

    def __post_init__(self) -> None:
        """初始化默认工具网关、trace 注入和上下文构建器。"""

        if self.tool_gateway is None:
            self.tool_gateway = ToolGateway(self.store)
        if self.tool_gateway.trace_recorder is None:
            self.tool_gateway.trace_recorder = self.trace_recorder
        self.context_builder = AgentContextBuilder(self.store, self.tool_gateway)

    def handle_user_message(
        self, message: UserMessage, *, trace_id: str | None = None
    ) -> AgentRuntimeResult:
        """处理一条用户消息的外层入口。

        这里负责会话级并发锁、消息幂等、run/version 推进、旧的待发送回复失效、摘要 checkpoint 触发。
        真正的 agent loop 放在 _handle_once，避免外层入口和模型循环混在一起。
        """

        with self._conversation_lock(message.conversation_id):
            actual_trace_id = trace_id or f"trace_{uuid.uuid4().hex[:12]}"
            message_key = message_idempotency_key(message)
            cached = self.store.idempotent_message_result(message_key)
            if cached is not None:
                self.trace_recorder.record(
                    actual_trace_id, "user_input", {"message": message.to_dict()}
                )
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
                return cached
            run_id = f"run_{uuid.uuid4().hex[:12]}"
            run_version, version_transition = self.store.advance_conversation_version(
                message.conversation_id,
                trace_id=actual_trace_id,
                reason="user_message_received",
            )
            superseded_counts, superseded_transitions = (
                self.store.supersede_pending_outputs(
                    message.conversation_id,
                    sender_id=message.sender_id,
                    trace_id=actual_trace_id,
                    reason="new_user_message_superseded_previous_pending_output",
                )
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
            result = self._handle_once(
                message,
                trace_id=actual_trace_id,
                run_id=run_id,
                run_version=run_version,
            )
            result.state_transitions = [
                version_transition,
                *superseded_transitions,
                *result.state_transitions,
            ]
            if self.context_summary_manager is not None:
                try:
                    summary_result = (
                        self.context_summary_manager.maybe_summarize_after_turn(
                            conversation_id=message.conversation_id,
                            trace_id=actual_trace_id,
                        )
                    )
                    if summary_result.transition is not None:
                        result.state_transitions.append(summary_result.transition)
                except Exception as exc:
                    self.trace_recorder.record(
                        actual_trace_id,
                        "context_summary_error",
                        {"error_type": type(exc).__name__, "error": str(exc)},
                        level="ERROR",
                    )
            self.store.remember_message_result(message_key, result)
            return result

    def _handle_once(
        self, message: UserMessage, *, trace_id: str, run_id: str, run_version: int
    ) -> AgentRuntimeResult:
        """执行一次目标驱动 agent loop。

        loop 的核心节奏是：构建上下文 -> 调主模型 -> 校验合同 -> 分流到工具或回复 ->
        把工具结果回喂模型。它只决定流程走向，不在这里写具体业务 if-else。
        """

        budgets = self._fresh_turn_budgets()
        self.store.append_user_turn(message, trace_id)
        self.trace_recorder.record(
            trace_id, "user_input", {"message": message.to_dict()}
        )
        actions: list[AgentAction] = []
        tool_results: list[ToolResult] = []
        pending_tool_results: list[ToolResult] = []
        pre_model_transitions: list[StateTransition] = []
        final_reply = ""

        for step_index in range(1, self.max_steps + 1):
            built = self._build_and_trace_context(
                message,
                trace_id=trace_id,
                pending_tool_results=pending_tool_results,
                run_id=run_id,
                run_version=run_version,
                step_index=step_index,
            )
            built, summary_transition = self._summarize_and_rebuild_context_if_needed(
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
            model_step = self._call_agent_action(
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
                pending_tool_results = self._record_action_contract_feedback(
                    message,
                    trace_id=trace_id,
                    raw_response=model_step.raw_response,
                    errors=model_step.errors,
                    step_index=step_index,
                )
                continue

            self._trace_action_plan(
                action,
                trace_id=trace_id,
                step_index=step_index,
                previous_tool_result_count=len(pending_tool_results),
            )
            processed = (
                self._process_tool_action(
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
                if action.tool_calls
                else self._process_reply_action(
                    action,
                    message=message,
                    trace_id=trace_id,
                    context_payload=built.payload,
                    budgets=budgets,
                    run_id=run_id,
                    run_version=run_version,
                )
            )
            actions[-1] = processed.action
            tool_results.extend(processed.tool_results)
            pending_tool_results = processed.pending_tool_results
            if processed.stop_loop:
                final_reply = processed.final_reply or ""
                break
            if processed.continue_loop:
                continue
        else:
            final_reply = "这个我先转人工确认一下。"
            self._append_pending_assistant_turn(
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

    def _fresh_turn_budgets(self) -> TurnBudgets:
        """为当前用户消息复制一份独立预算计数器。

        TokenBudget 内部有 calls_this_turn 状态；每条用户消息必须从 0 开始统计，
        不能复用 runtime 级默认对象，否则不同消息会互相污染预算。
        """

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
                max_tokens_per_call=self.customer_visible_text_generation_token_budget.max_tokens_per_call,
                max_calls_per_turn=self.customer_visible_text_generation_token_budget.max_calls_per_turn,
            ),
        )

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
        """构建本轮喂给主模型的上下文并完整写 trace。

        previous_tool_results 是上一轮工具结果，模型需要看到它才能决定下一步。
        这里同时记录 context_packed、context_built、llm_prompt，保证每次决策可回溯。
        """

        built = self.context_builder.build(
            message,
            trace_id=trace_id,
            previous_tool_results=pending_tool_results,
            run_id=run_id,
            run_version=run_version,
        )
        self.trace_recorder.record(trace_id, "context_packed", built.audit)
        self.trace_recorder.record(trace_id, "context_built", built.payload)
        self.trace_recorder.record(
            trace_id,
            "llm_prompt",
            {"messages": built.messages, "step_index": step_index},
        )
        return built

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
        """在主模型调用前根据上下文预算主动摘要并重建上下文。

        摘要不能只在一轮结束后做；如果当前 prompt 已经接近单次调用上限，
        应先压缩旧对话为 checkpoint，再重新 build，最后再进入预算 reserve。
        """

        estimated = sum(estimate_tokens(item.get("content", "")) for item in built.messages)
        threshold = max(1, int(budget.max_tokens_per_call * self.context_summary_preemptive_ratio))
        self.trace_recorder.record(
            trace_id,
            "context_budget_precheck",
            {
                "estimated_tokens": estimated,
                "max_tokens_per_call": budget.max_tokens_per_call,
                "trigger_threshold_tokens": threshold,
                "context_summary_enabled": self.context_summary_manager is not None,
                "step_index": step_index,
            },
        )
        if self.context_summary_manager is None or estimated < threshold:
            return built, None
        checkpoint = built.payload.get("conversation_checkpoint") if isinstance(built.payload, dict) else None
        if isinstance(checkpoint, dict) and checkpoint.get("source_trace_id") == trace_id:
            self.trace_recorder.record(
                trace_id,
                "context_summary_budget_already_applied",
                {
                    "estimated_tokens": estimated,
                    "max_tokens_per_call": budget.max_tokens_per_call,
                    "trigger_threshold_tokens": threshold,
                    "step_index": step_index,
                },
            )
            return built, None
        try:
            summary_result = self.context_summary_manager.summarize_for_context_budget(
                conversation_id=message.conversation_id,
                trace_id=trace_id,
                estimated_context_tokens=estimated,
                max_context_tokens=budget.max_tokens_per_call,
                trigger_threshold_tokens=threshold,
            )
        except Exception as exc:
            self.trace_recorder.record(
                trace_id,
                "context_summary_error",
                {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "trigger": "context_budget",
                },
                level="ERROR",
            )
            return built, None
        if summary_result.transition is None:
            self.trace_recorder.record(
                trace_id,
                "context_summary_budget_not_applied",
                summary_result.to_dict(),
                level="WARN",
            )
            return built, None
        rebuilt = self._build_and_trace_context(
            message,
            trace_id=trace_id,
            pending_tool_results=pending_tool_results,
            run_id=run_id,
            run_version=run_version,
            step_index=step_index,
        )
        rebuilt_estimated = sum(estimate_tokens(item.get("content", "")) for item in rebuilt.messages)
        self.trace_recorder.record(
            trace_id,
            "context_rebuilt_after_summary",
            {
                "previous_estimated_tokens": estimated,
                "rebuilt_estimated_tokens": rebuilt_estimated,
                "checkpoint": summary_result.checkpoint.to_dict() if summary_result.checkpoint else None,
                "transition": summary_result.transition.to_dict(),
                "step_index": step_index,
            },
        )
        return rebuilt, summary_result.transition

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
        """调用主模型并解析成 AgentAction。

        这里只做预算、超时、原始响应记录和合同解析；不执行工具、不落库业务状态。
        如果模型失败或预算不足，返回一个可被主 loop 终止的安全结果。
        """

        budget_decision = budget.reserve(built_messages)
        self.trace_recorder.record(
            trace_id, "budget_checked", budget_decision.to_dict()
        )
        if not budget_decision.allowed:
            final_reply = "这个我先转人工确认一下。"
            self.trace_recorder.record(
                trace_id,
                "final_output",
                {"reply": final_reply, "reason": budget_decision.reason},
                level="WARN",
            )
            self._append_pending_assistant_turn(
                message.conversation_id,
                final_reply,
                trace_id,
                run_id=run_id,
                run_version=run_version,
            )
            return ModelActionStep(action=None, final_reply=final_reply, stop_loop=True)

        started = time.perf_counter()
        try:
            raw_response = self.llm_client.complete(
                built_messages,
                trace_id=trace_id,
                timeout_seconds=self.llm_timeout_seconds,
            )
        except Exception as exc:
            final_reply = "这个我先转人工确认一下。"
            self.trace_recorder.record(
                trace_id,
                "llm_error",
                {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "elapsed_ms": int((time.perf_counter() - started) * 1000),
                },
                level="ERROR",
            )
            self.trace_recorder.record(
                trace_id,
                "final_output",
                {"reply": final_reply, "reason": "llm_error"},
                level="WARN",
            )
            self._append_pending_assistant_turn(
                message.conversation_id,
                final_reply,
                trace_id,
                run_id=run_id,
                run_version=run_version,
            )
            return ModelActionStep(action=None, final_reply=final_reply, stop_loop=True)

        self.trace_recorder.record(
            trace_id,
            "llm_response",
            {
                "content": raw_response,
                "elapsed_ms": int((time.perf_counter() - started) * 1000),
                "step_index": step_index,
            },
        )
        action, errors = parse_action(raw_response)
        return ModelActionStep(action=action, raw_response=raw_response, errors=errors)

    def _record_action_contract_feedback(
        self,
        message: UserMessage,
        *,
        trace_id: str,
        raw_response: str,
        errors: list[str],
        step_index: int,
    ) -> list[ToolResult]:
        """把模型合同错误包装成工具结果回喂模型。

        设计上不直接把合同错误变成人工兜底；如果还有 loop 步数，就给模型一次修正机会。
        这能让主模型学会遵守 AgentAction 契约，同时不执行任何不可信工具。
        """

        self.trace_recorder.record(
            trace_id,
            "action_contract_error",
            {"errors": errors, "step_index": step_index},
            level="WARN",
        )
        feedback = ToolResult(
            name="agent_action_contract",
            called=False,
            allowed=False,
            result={
                "errors": list(errors),
                "raw_response": raw_response,
                "instruction": "Fix the AgentAction JSON contract. If waiting for user, use objective_status=waiting_user with non-empty reply_to_user. If tools are needed, use objective_status=needs_tool with at least one tool_call.",
            },
            error="AgentAction contract invalid: " + "; ".join(errors),
        )
        self.trace_recorder.record(
            trace_id, "contract_error_feedback", feedback.to_dict(), level="WARN"
        )
        self.store.append_tool_turn(
            message.conversation_id,
            json.dumps([feedback.to_dict()], ensure_ascii=False),
            trace_id,
        )
        return [feedback]

    def _trace_action_plan(
        self,
        action: AgentAction,
        *,
        trace_id: str,
        step_index: int,
        previous_tool_result_count: int,
    ) -> None:
        """记录模型提出的目标、计划、状态和工具名。

        这一步是可观测的关键：当回复或工具调用不符合预期时，可以从 trace 看到模型当时的目标理解、
        计划修订原因和它准备调用哪些工具。
        """

        self.trace_recorder.record(trace_id, "action_proposed", action.to_dict())
        self.trace_recorder.record(
            trace_id,
            "objective_plan_proposed",
            {
                "step_index": step_index,
                "goal": action.goal,
                "objective_status": action.objective_status,
                "objective_state": dict(action.objective_state),
                "objective_plan": [dict(item) for item in action.objective_plan],
                "plan_revision_reason": action.plan_revision_reason,
                "previous_tool_result_count": previous_tool_result_count,
                "tool_call_names": [call.name for call in action.tool_calls],
            },
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
        """处理包含 tool_calls 的 AgentAction。

        工具参数里可能包含候选人邀约等客户可见文本，所以先做话术生成和安全审查；
        审查通过后才进入真正的工具执行。审查失败会把结果回喂模型，让模型重写而不是硬编码兜底。
        """

        processor = self._customer_visible_processor()
        collected_results: list[ToolResult] = []
        review_items = customer_visible_items_for_action(action)
        text_generation_result = processor.run_text_generation(
            message=message,
            trace_id=trace_id,
            action=action,
            items=review_items,
            context_payload=context_payload,
            turn_budget=budgets.text_generation,
            generation_scope="tool_calls",
        )
        if text_generation_result is not None:
            collected_results.append(text_generation_result)
            action = self._apply_customer_visible_rewrites(
                action, text_generation_result, trace_id=trace_id
            )
            review_items = customer_visible_items_for_action(action)

        review_result = processor.run_content_review(
            message=message,
            trace_id=trace_id,
            action=action,
            review_items=review_items,
            context_payload=context_payload,
            turn_budget=budgets.review,
            review_scope="tool_calls",
        )
        if review_result is not None:
            collected_results.append(review_result)
            self.trace_recorder.record(trace_id, "tool_result", review_result.to_dict())
            self.store.append_tool_turn(
                message.conversation_id,
                json.dumps([review_result.to_dict()], ensure_ascii=False),
                trace_id,
            )
            if not customer_visible_content_review_approved(review_result):
                return ActionProcessingResult(
                    action=action,
                    tool_results=collected_results,
                    pending_tool_results=[review_result],
                    continue_loop=True,
                )

        execution = self._execute_tool_calls(
            action,
            message=message,
            trace_id=trace_id,
            previous_step_tool_results=list(previous_pending_tool_results),
            step_index=step_index,
            run_id=run_id,
            run_version=run_version,
        )
        collected_results.extend(execution.tool_results)
        if execution.stop_loop:
            return ActionProcessingResult(
                action=action,
                tool_results=collected_results,
                pending_tool_results=execution.pending_tool_results,
                final_reply=execution.final_reply,
                stop_loop=True,
            )
        return ActionProcessingResult(
            action=action,
            tool_results=collected_results,
            pending_tool_results=execution.pending_tool_results,
            continue_loop=True,
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
        """顺序执行模型提出的工具调用。

        后端在这里做跨工具参数一致性、过期 run 拦截、工具网关执行、状态变更日志和工具结果落入短期记忆。
        这里是模型意图变成系统动作的边界，因此必须比普通函数调用更保守。
        """

        tool_results: list[ToolResult] = []
        pending_tool_results: list[ToolResult] = []
        blocked_by_consistency = False
        blocked_by_stale_run = False
        for call_index, call in enumerate(action.tool_calls, start=1):
            consistency_error = validate_tool_call_consistency(
                call, previous_step_tool_results + pending_tool_results
            )
            if consistency_error:
                reference_requirement = latest_read_requirement(
                    previous_step_tool_results + pending_tool_results,
                    tool_name="search_current_games",
                )
                result = ToolResult(
                    name=call.name,
                    called=False,
                    allowed=False,
                    result={
                        "instruction": (
                            "Fix the tool arguments and call the tool again. Preserve explicit requirement fields "
                            "from previous read-only tool results unless the user has clearly changed them."
                        ),
                        "call": call.to_dict(),
                        "reference_tool_name": "search_current_games",
                        "reference_requirement": reference_requirement or {},
                    },
                    error=consistency_error,
                )
                tool_results.append(result)
                pending_tool_results.append(result)
                self.trace_recorder.record(
                    trace_id,
                    "tool_argument_consistency_error",
                    {
                        "call": call.to_dict(),
                        "error": consistency_error,
                        "step_index": step_index,
                    },
                    level="WARN",
                )
                self.trace_recorder.record(
                    trace_id, "tool_result", result.to_dict(), level="WARN"
                )
                blocked_by_consistency = True
                break

            stale_result = self._stale_write_tool_result(
                call_name=call.name,
                conversation_id=message.conversation_id,
                run_id=run_id,
                run_version=run_version,
            )
            if stale_result is not None:
                tool_results.append(stale_result)
                pending_tool_results.append(stale_result)
                self.trace_recorder.record(
                    trace_id,
                    "conversation_run_stale",
                    stale_result.to_dict(),
                    level="WARN",
                )
                self.trace_recorder.record(
                    trace_id, "tool_result", stale_result.to_dict(), level="WARN"
                )
                blocked_by_stale_run = True
                break

            self.trace_recorder.record(
                trace_id,
                "tool_called",
                {"call": call.to_dict(), "step_index": step_index},
            )
            result = self.tool_gateway.execute(
                call,
                trace_id=trace_id,
                conversation_id=message.conversation_id,
                sender_id=message.sender_id,
                sender_name=message.sender_name,
                step_index=step_index * 100 + call_index,
                source_message_id=message.message_id,
            )
            tool_results.append(result)
            pending_tool_results.append(result)
            self.trace_recorder.record(trace_id, "tool_result", result.to_dict())
            for transition in result.state_transitions:
                step = (
                    "state_transition_replayed"
                    if result.deduplicated
                    else "state_transition"
                )
                self.trace_recorder.record(trace_id, step, transition.to_dict())

        self.store.append_tool_turn(
            message.conversation_id,
            json.dumps(
                [item.to_dict() for item in pending_tool_results], ensure_ascii=False
            ),
            trace_id,
        )
        if blocked_by_stale_run:
            final_reply = ""
            self.trace_recorder.record(
                trace_id,
                "final_output",
                {
                    "reply": final_reply,
                    "reason": "conversation_run_stale",
                    "run_id": run_id,
                    "run_version": run_version,
                    "current_version": self.store.conversation_version(
                        message.conversation_id
                    ),
                },
                level="WARN",
            )
            return ActionProcessingResult(
                action=action,
                tool_results=tool_results,
                pending_tool_results=pending_tool_results,
                final_reply=final_reply,
                stop_loop=True,
            )
        if blocked_by_consistency:
            self.trace_recorder.record(
                trace_id,
                "tool_argument_consistency_feedback",
                {"results": [item.to_dict() for item in pending_tool_results]},
                level="WARN",
            )
        return ActionProcessingResult(
            action=action,
            tool_results=tool_results,
            pending_tool_results=pending_tool_results,
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
        """处理不需要继续调用工具的最终回复。

        最终回复也属于客户可见文本，所以同样经过话术生成和内容审查。
        审查通过后只写入 pending assistant turn，由外部通道决定是否真正发送。
        """

        processor = self._customer_visible_processor()
        collected_results: list[ToolResult] = []
        proposed_reply = action.reply_to_user.strip()
        if action.needs_human and not proposed_reply:
            proposed_reply = "这个我先转人工确认一下。"

        review_item = {
            "item_id": "reply_to_user",
            "source": "reply_to_user",
            "recipient_id": message.sender_id,
            "recipient_name": message.sender_name,
            "text": proposed_reply,
        }
        text_generation_result = processor.run_text_generation(
            message=message,
            trace_id=trace_id,
            action=action,
            items=[review_item],
            context_payload=context_payload,
            turn_budget=budgets.text_generation,
            generation_scope="reply_to_user",
        )
        if text_generation_result is not None:
            collected_results.append(text_generation_result)
            rewrites = self._customer_visible_rewrites(text_generation_result)
            if rewrites.get("reply_to_user"):
                proposed_reply = rewrites["reply_to_user"].strip()
                action = action_with_customer_visible_rewrites(action, rewrites)
                self.trace_recorder.record(
                    trace_id,
                    "action_after_customer_visible_text_generation",
                    action.to_dict(),
                )
                review_item = {**review_item, "text": proposed_reply}

        review_result = processor.run_content_review(
            message=message,
            trace_id=trace_id,
            action=action,
            review_items=[review_item],
            context_payload=context_payload,
            turn_budget=budgets.review,
            review_scope="reply_to_user",
        )
        if review_result is not None:
            collected_results.append(review_result)
            self.trace_recorder.record(trace_id, "tool_result", review_result.to_dict())
            self.store.append_tool_turn(
                message.conversation_id,
                json.dumps([review_result.to_dict()], ensure_ascii=False),
                trace_id,
            )
            if not customer_visible_content_review_approved(review_result):
                return ActionProcessingResult(
                    action=action,
                    tool_results=collected_results,
                    pending_tool_results=[review_result],
                    continue_loop=True,
                )

        if self._run_is_stale(message.conversation_id, run_version):
            final_reply = ""
            self.trace_recorder.record(
                trace_id,
                "conversation_run_stale",
                {
                    "run_id": run_id,
                    "run_version": run_version,
                    "current_version": self.store.conversation_version(
                        message.conversation_id
                    ),
                    "blocked": "final_reply",
                },
                level="WARN",
            )
            self.trace_recorder.record(
                trace_id,
                "final_output",
                {
                    "reply": final_reply,
                    "reason": "conversation_run_stale",
                    "run_id": run_id,
                    "run_version": run_version,
                    "current_version": self.store.conversation_version(
                        message.conversation_id
                    ),
                },
                level="WARN",
            )
            return ActionProcessingResult(
                action=action,
                tool_results=collected_results,
                final_reply=final_reply,
                stop_loop=True,
            )

        self._append_pending_assistant_turn(
            message.conversation_id,
            proposed_reply,
            trace_id,
            run_id=run_id,
            run_version=run_version,
        )
        self.trace_recorder.record(
            trace_id,
            "final_output",
            {"reply": proposed_reply, "objective_status": action.objective_status},
        )
        return ActionProcessingResult(
            action=action,
            tool_results=collected_results,
            final_reply=proposed_reply,
            stop_loop=True,
        )

    def _apply_customer_visible_rewrites(
        self, action: AgentAction, result: ToolResult, *, trace_id: str
    ) -> AgentAction:
        """把话术生成器返回的改写结果应用回 AgentAction。

        工具调用参数中的 message_text 可能被改写；主 loop 后续必须使用改写后的 action 做审查和执行。
        """

        rewrites = self._customer_visible_rewrites(result)
        if not rewrites:
            return action
        rewritten = action_with_customer_visible_rewrites(action, rewrites)
        self.trace_recorder.record(
            trace_id,
            "action_after_customer_visible_text_generation",
            rewritten.to_dict(),
        )
        return rewritten

    @staticmethod
    def _customer_visible_rewrites(result: ToolResult) -> dict[str, str]:
        """从话术生成工具结果中提取 item_id -> final_text 映射。"""

        return {
            str(item.get("item_id") or ""): str(item.get("final_text") or "")
            for item in result.result.get("item_rewrites", [])
            if isinstance(item, dict)
        }

    def _run_is_stale(self, conversation_id: str, run_version: int) -> bool:
        """判断当前 run 是否已经被同会话的新消息超越。"""

        return self.store.conversation_version(conversation_id) != int(run_version)

    def _stale_write_tool_result(
        self,
        *,
        call_name: str,
        conversation_id: str,
        run_id: str,
        run_version: int,
    ) -> ToolResult | None:
        """为过期 run 的写工具生成一个拒绝执行结果。

        如果用户在旧流程还没结束时又补充了新消息，旧流程不能再创建局、生成草稿或写状态；
        否则会出现“用户刚改条件，旧条件还在落库”的并发错乱。
        """

        definition = (
            self.tool_gateway.tools.get(call_name) if self.tool_gateway else None
        )
        if definition is None or definition.execution_mode not in {
            "state_write",
            "draft_write",
        }:
            return None
        current_version = self.store.conversation_version(conversation_id)
        if current_version == int(run_version):
            return None
        return ToolResult(
            name=call_name,
            called=False,
            allowed=False,
            result={
                "run_id": run_id,
                "run_version": run_version,
                "current_version": current_version,
                "instruction": (
                    "This run is stale because a newer user message advanced the conversation version. "
                    "Do not write state or create drafts from the old version; rebuild context from the latest user input."
                ),
            },
            error="stale run: conversation version changed before a state-writing tool could execute",
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
        """记录一条待外发的 assistant 回复。

        当前系统默认外发前仍可审批/关闭通道，因此这里标记为 pending_operator_send，
        真正发送由 Web/WeChaty 通道层决定。
        """

        self.store.append_assistant_turn(
            conversation_id,
            text,
            trace_id,
            metadata={
                "delivery_status": "pending_operator_send",
                "run_id": run_id,
                "conversation_version": run_version,
            },
        )

    def _conversation_lock(self, conversation_id: str) -> threading.RLock:
        """返回会话级锁，保证同一个 conversation 内的消息顺序处理。"""

        key = conversation_id or "default"
        with self._conversation_locks_guard:
            lock = self._conversation_locks.get(key)
            if lock is None:
                lock = threading.RLock()
                self._conversation_locks[key] = lock
            return lock

    def _customer_visible_processor(self) -> CustomerVisibleProcessor:
        """创建客户可见文本处理器。

        这是主 loop 与“话术生成/安全审查”子链路的边界。runtime 只提供依赖和开关，
        具体生成、审查、合同解析都在 visibility.py 中完成。
        """

        return CustomerVisibleProcessor(
            llm_client=self.llm_client,
            trace_recorder=self.trace_recorder,
            timeout_seconds=self.llm_timeout_seconds,
            text_generation_enabled=self.customer_visible_text_generation_enabled,
            text_generation_client=self.customer_visible_text_generation_client,
            text_generation_prompt_path=self.customer_visible_text_generation_prompt_path,
            review_enabled=self.reply_self_review_enabled,
            review_client=self.reply_self_review_client,
            review_prompt_path=self.reply_self_review_prompt_path,
        )

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
        """兼容入口：运行客户可见话术生成。

        一些脚本仍直接调用 runtime 的旧私有方法，所以保留薄封装；实际逻辑已迁移到 CustomerVisibleProcessor。
        """

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
        """兼容入口：运行客户可见内容审查。

        保留这个方法是为了不破坏现有脚本调用；主逻辑已从 runtime 拆到 visibility.py。
        """

        return self._customer_visible_processor().run_content_review(
            message=message,
            trace_id=trace_id,
            action=action,
            review_items=review_items,
            context_payload=context_payload,
            turn_budget=turn_budget,
            review_scope=review_scope,
        )


def message_idempotency_key(message: UserMessage) -> str:
    """生成用户消息幂等键。

    幂等维度包含 conversation、sender 和 source message id，防止上游重复投递导致同一消息被处理两次。
    """

    return f"conversation:{message.conversation_id}:sender:{message.sender_id}:message:{message.message_id}"
