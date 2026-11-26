"""管理上下文构建、预算压缩、审计记录和会话检查点生命周期。

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..budget import TokenBudget
from ..domains.context_builders import AgentContextBuilder, BuiltContext
from ..domains.context_builders.builder import main_agent_system_message_with_packed_tools, main_agent_user_message
from ..domains.context_builders.tool_results import (
    compact_tool_feedback_for_transport,
    reference_duplicate_latest_tool_results,
    reference_shared_query_contracts,
    reference_shared_tool_schemas,
)
from ..token_estimation import estimate_tokens
from ..hooks import HookManager
from ..models import AgentAction, StateTransition, ToolResult, UserMessage
from ..summary import ContextSummaryManager, ContextSummaryResult


_TERMINAL_TOOL_ERRORS = {f"{status} must not include tool_calls"
                        for status in ("waiting_user", "completed", "needs_human", "unknown")}
_TOOL_PHASE_RETRY = (
    "【修正刚被拒绝的动作】上一份动作同时提交了工具调用和终态，所有调用均未执行。"
    "如果同一目标仍有已授权且必要的动作，现在只输出执行工具的一步："
    '"objective_status":"needs_tool","reply_to_user":"","stop_reason":{"can_stop":false,"depends_on_tool_results":true}。'
    "重提完整动作，只提交当下可执行的调用。先看真实工具结果，再完成或向顾客提问。"
    "不要把上一步未执行的计划当成已完成。"
)


@dataclass(slots=True)
class ContextLifecycleManager:
    """管理模型调用前的上下文生命周期，以及整轮结束后的摘要维护。

    上游：单步执行器在每次主模型调用前使用构建/预算压缩方法；``AgentRuntime`` 在
    整轮结束后调用摘要维护方法。下游：``AgentContextBuilder`` 与可选的
    ``ContextSummaryManager``。摘要是可重建的派生投影，不是业务事实源。
    """

    context_builder: AgentContextBuilder
    trace_recorder: Any
    context_summary_manager: ContextSummaryManager | None = None
    context_summary_preemptive_ratio: float = 0.85
    hook_manager: HookManager | None = None

    def build_and_trace_context(
        self,
        message: UserMessage,
        *,
        trace_id: str,
        pending_tool_results: list[ToolResult],
        turn_tool_evidence: list[ToolResult],
        run_id: str,
        run_version: int,
        step_index: int,
        progress_hint: str | None = None,
        previous_action: AgentAction | None = None,
    ) -> BuiltContext:
        """委托 ContextBuilder 构建只读投影，并记录本次模型实际看到的内容。

        上游：``AgentLoopStepService.execute``。边界：本方法不写麻将业务状态；
        ``after_context_built`` 是供旁路订阅者使用的间接回调点。
        """

        built = self.context_builder.build(
            message,
            trace_id=trace_id,
            previous_tool_results=pending_tool_results,
            turn_tool_evidence=turn_tool_evidence,
            run_id=run_id,
            run_version=run_version,
            previous_action=previous_action,
        )
        if any(item.name == "agent_action_contract" and _TERMINAL_TOOL_ERRORS.intersection(item.result.get("errors") or ())
               for item in pending_tool_results):
            # 只提升本步正式合同错误的修复提示。原始提案仍是未执行的审计证据，绝不在这里改写工具。
            built.messages.insert(1, {"role": "system", "content": _TOOL_PHASE_RETRY})
        if progress_hint:
            # 提示仍是可信的系统指令，但步骤变化不能改写可跨请求复用的首条系统消息。
            built.messages.insert(len(built.messages) - 1, {"role": "system", "content": progress_hint})
            self.trace_recorder.record(
                trace_id,
                "agent_progress_hint_injected",
                {
                    "step_index": step_index,
                    "line_count": len(progress_hint.splitlines()),
                    "hint": progress_hint,
                },
            )
        self.trace_recorder.record(trace_id, "context_packed", built.audit)
        self.trace_recorder.record(trace_id, "context_built", built.payload)
        self.trace_recorder.record(
            trace_id,
            "llm_prompt",
            {"messages": built.messages, "step_index": step_index},
        )
        self._emit(
            "after_context_built",
            trace_id=trace_id,
            payload={"step_index": step_index, "audit": built.audit, "payload": built.payload},
        )
        return built

    def summarize_and_rebuild_context_if_needed(
        self,
        message: UserMessage,
        *,
        built: BuiltContext,
        trace_id: str,
        pending_tool_results: list[ToolResult],
        turn_tool_evidence: list[ToolResult],
        run_id: str,
        run_version: int,
        step_index: int,
        budget: TokenBudget,
        progress_hint: str | None = None,
        previous_action: AgentAction | None = None,
    ) -> tuple[BuiltContext, StateTransition | None]:
        """预算接近阈值时尝试生成并持久化摘要 checkpoint。

        摘要可能因合同、置信度或事实校验失败而不落库；只有返回 transition 后才会从
        Store 重建。此时重建不可省略，因为当前 ``BuiltContext`` 仍是旧快照。
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
        built, estimated = self._deduplicate_transport_context(
            built,
            trace_id=trace_id,
            estimated=estimated,
            step_index=step_index,
            max_tokens_per_call=budget.max_tokens_per_call,
            summary_threshold_tokens=threshold if self.context_summary_manager is not None else None,
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
        # 预防性摘要只压缩对话与检查点。当前调用仍在硬预算内，且把这部分全去掉也达不到目标时，
        # 继续用原始来话和已压缩工具证据；真正超过硬预算时仍尝试摘要回收空间。
        history = estimate_tokens({"recent_conversation": built.payload.get("recent_conversation") or [],
                                   "conversation_checkpoint": checkpoint})
        if estimated <= budget.max_tokens_per_call and estimated - history >= threshold:
            self.trace_recorder.record(trace_id, "context_summary_budget_not_reclaimable", {
                "step_index": step_index, "estimated_tokens": estimated,
                "trigger_threshold_tokens": threshold, "estimated_history_tokens": history,
            })
            return built, None
        summary_result = self._summarize(
            message,
            built=built,
            trace_id=trace_id,
            estimated=estimated,
            threshold=threshold,
            budget=budget,
            run_version=run_version,
        )
        if summary_result is None or summary_result.transition is None:
            if summary_result is not None:
                self.trace_recorder.record(
                    trace_id,
                    "context_summary_budget_not_applied",
                    summary_result.to_dict(),
                    level="WARN",
                )
            return built, None

        rebuilt = self.build_and_trace_context(
            message,
            trace_id=trace_id,
            pending_tool_results=pending_tool_results,
            turn_tool_evidence=turn_tool_evidence,
            run_id=run_id,
            run_version=run_version,
            step_index=step_index,
            progress_hint=progress_hint,
            # 摘要压缩重建不能抹掉同一运行的计划；仍作为低信任提案传递。
            previous_action=previous_action,
        )
        rebuilt_estimated = sum(estimate_tokens(item.get("content", "")) for item in rebuilt.messages)
        rebuilt, rebuilt_estimated = self._deduplicate_transport_context(
            rebuilt,
            trace_id=trace_id,
            estimated=rebuilt_estimated,
            step_index=step_index,
            max_tokens_per_call=budget.max_tokens_per_call,
            summary_threshold_tokens=threshold,
        )
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
        self._emit(
            "after_context_rebuilt",
            trace_id=trace_id,
            payload={
                "step_index": step_index,
                "previous_estimated_tokens": estimated,
                "rebuilt_estimated_tokens": rebuilt_estimated,
            },
        )
        return rebuilt, summary_result.transition

    def _deduplicate_transport_context(
        self,
        built: BuiltContext,
        *,
        trace_id: str,
        estimated: int,
        step_index: int,
        max_tokens_per_call: int,
        summary_threshold_tokens: int | None = None,
    ) -> tuple[BuiltContext, int]:
        """在所有预算下压缩发送消息中的精确副本，保留完整审计载荷。

        完整结果保留在 ``turn_tool_evidence`` 中，静态查询合同保留在 planning_contract；
        只替换完全相等的重复项。触及摘要阈值也先压缩后端 JSON，避免静态合同造成多次摘要网络调用。
        业务状态、原始对话及来源始终留在审计和对应投影中。
        """

        payload = built.payload
        latest = payload.get("previous_tool_results")
        evidence = payload.get("turn_tool_evidence")
        transport_source = payload
        compacted_count = 0
        if isinstance(latest, list) and isinstance(evidence, list):
            referenced, compacted_count = reference_duplicate_latest_tool_results(latest, evidence)
            transport_source = {**payload, "previous_tool_results": referenced}
        transport_payload, contract_count = reference_shared_query_contracts(transport_source)
        prior_user = built.messages[-1]
        other_tokens = estimated - estimate_tokens(prior_user["content"])
        result_counts = {
            "latest_tool_result_reference_count": compacted_count,
            "shared_query_contract_reference_count": contract_count,
        }
        if compacted_count or contract_count:
            transport_payload["context_budget"] = {
                **(transport_payload.get("context_budget") or {}),
                **result_counts,
                "latest_tool_results_reference_source": "turn_tool_evidence",
            }
            user_message = main_agent_user_message(transport_payload)
        else:
            user_message = prior_user
        system_message = built.messages[0]
        compacted_estimated = other_tokens + estimate_tokens(user_message["content"])
        value_count = 0
        schema_count = 0
        packing_limit = min(max_tokens_per_call, summary_threshold_tokens - 1) if summary_threshold_tokens else max_tokens_per_call
        if compacted_estimated > packing_limit:
            dense_user = main_agent_user_message(transport_payload, dense=True)
            dense_estimated = other_tokens + estimate_tokens(dense_user["content"])
            if dense_estimated > packing_limit or compacted_estimated > max_tokens_per_call:
                value_count = compact_tool_feedback_for_transport(transport_payload)
                schema_count = reference_shared_tool_schemas(transport_payload)
                if value_count:
                    transport_payload["context_budget"] = {
                        **(transport_payload.get("context_budget") or {}),
                        "tool_evidence_value_ref_rule": (
                            "见 tool_evidence_value_ref 时，先在 turn_tool_evidence[index] 按 path 读取完整字段值；"
                            "该引用只替代相同字段值，各条工具调用、执行状态、客人原话与实际局况仍各自核对。"
                        ),
                    }
                if schema_count:
                    transport_payload["context_budget"] = {
                        **(transport_payload.get("context_budget") or {}),
                        "shared_tool_schema_ref_rule": (
                            "系统消息的 available_tools 中的 $ref 指向动态上下文中的 shared_tool_schemas，先按路径展开。"
                        ),
                    }
                referenced_user = main_agent_user_message(transport_payload, dense=True)
                referenced_system = system_message
                referenced_estimated = other_tokens + estimate_tokens(referenced_user["content"])
                if schema_count:
                    referenced_system = main_agent_system_message_with_packed_tools(system_message, transport_payload)
                    referenced_estimated += estimate_tokens(referenced_system["content"]) - estimate_tokens(system_message["content"])
                if referenced_estimated < dense_estimated:
                    dense_user, dense_estimated = referenced_user, referenced_estimated
                else:
                    referenced_system = system_message
                    value_count = schema_count = 0
            else:
                referenced_system = system_message
            if dense_estimated < compacted_estimated:
                user_message, compacted_estimated = dense_user, dense_estimated
                system_message = referenced_system
            else:
                value_count = schema_count = 0
        if user_message == prior_user and system_message == built.messages[0]:
            return built, estimated
        # 另建发送消息和审计，避免内存 trace 的原始 llm_prompt 一起被修改。
        built.messages = [system_message, *built.messages[1:-1], user_message]
        built.audit = {
            **built.audit,
            **result_counts,
            "tool_evidence_value_reference_count": value_count,
            "shared_tool_schema_reference_count": schema_count,
        }
        self.trace_recorder.record(
            trace_id,
            "context_tool_feedback_deduplicated",
            {
                "step_index": step_index,
                "compacted_result_count": compacted_count,
                "shared_query_contract_reference_count": contract_count,
                "tool_evidence_value_reference_count": value_count,
                "shared_tool_schema_reference_count": schema_count,
                "previous_estimated_tokens": estimated,
                "compacted_estimated_tokens": compacted_estimated,
                "saved_estimated_tokens": max(0, estimated - compacted_estimated),
                "reference_source": "turn_tool_evidence",
            },
        )
        self.trace_recorder.record(
            trace_id,
            "llm_prompt_after_context_compaction",
            {"messages": built.messages, "step_index": step_index},
        )
        return built, compacted_estimated

    def maybe_summarize_after_turn(
        self,
        *,
        conversation_id: str,
        trace_id: str,
        task_context_id: str | None = None,
        expected_run_version: int | None = None,
    ) -> ContextSummaryResult | None:
        """在一轮结束后评估摘要阈值，必要时异步语义上生成新的会话检查点。

"""

        if self.context_summary_manager is None:
            return None
        return self.context_summary_manager.maybe_summarize_after_turn(
            conversation_id=conversation_id,
            trace_id=trace_id,
            task_context_id=task_context_id,
            expected_run_version=expected_run_version,
        )

    def _summarize(
        self,
        message: UserMessage,
        *,
        built: BuiltContext,
        trace_id: str,
        estimated: int,
        threshold: int,
        budget: TokenBudget,
        run_version: int,
    ) -> ContextSummaryResult | None:
        assert self.context_summary_manager is not None
        task_context_id = self.summary_task_context_id(message, built=built)
        try:
            return self.context_summary_manager.summarize_for_context_budget(
                conversation_id=message.conversation_id,
                trace_id=trace_id,
                estimated_context_tokens=estimated,
                max_context_tokens=budget.max_tokens_per_call,
                trigger_threshold_tokens=threshold,
                task_context_id=task_context_id,
                expected_run_version=run_version,
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
            return None

    def summary_task_context_id(
        self,
        message: UserMessage,
        *,
        built: BuiltContext | None = None,
    ) -> str | None:
        """返回由后端解析出的本轮总结范围。

        受信任的定时/系统事件可以显式恢复较早的任务。普通消息使用 ``TaskContextManager`` 选定并已暴露在已构建上下文中的任务；用户文本绝不选择此 id。
        """

        metadata = message.metadata if isinstance(message.metadata, dict) else {}
        trusted_source = str(metadata.get("_trusted_source_task_context_id") or "")
        if trusted_source:
            task_context = (
                self.context_summary_manager.store.get_task_context(trusted_source)
                if self.context_summary_manager
                else None
            )
            if task_context is not None and task_context.conversation_id == message.conversation_id:
                return trusted_source
        if built is not None:
            window = built.payload.get("task_context_window") if isinstance(built.payload, dict) else None
            if isinstance(window, dict):
                resolved = str(window.get("task_context_id") or "")
                if resolved:
                    return resolved
        if self.context_summary_manager is None:
            return None
        task_context = self.context_summary_manager.store.current_task_context(
            message.conversation_id,
            message.sender_id,
        )
        return task_context.task_context_id if task_context is not None else None

    def _emit(self, event_name: str, *, trace_id: str, payload: dict[str, Any]) -> None:
        if self.hook_manager is not None:
            self.hook_manager.emit(event_name, trace_id=trace_id, payload=payload)


__all__ = ["ContextLifecycleManager"]
