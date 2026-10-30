"""编排智能体主循环中的单个步骤，串联上下文、模型、工具和进展检测。

"""

from __future__ import annotations

from dataclasses import dataclass

from ..customer_visible_contract import CUSTOMER_EXECUTION_FAILURE_REPLY

from ..models import AgentAction, StateTransition, ToolResult, UserMessage
from ..progress import ProgressMonitor, build_progress_hint
from ..runtime_components import TurnBudgets
from .action_service import ActionProcessor
from .context_service import ContextLifecycleManager
from .contracts import LoopStepOutcome
from .progress_service import ProgressGuardService
from .objective_continuation import blocking_continuation


@dataclass(slots=True)
class AgentLoopStepService:
    """编排主循环中的一个模型步骤。

    上游：``AgentLoop.run`` 每迭代一次调用一次。下游顺序：构建上下文 -> 预算前摘要
    -> 主模型 -> 动作合同/continuation 校验 -> 动作处理 -> 进展检测。私聊额外注入
    ``progress_hint``，处理模型自评和 ``blocking_continuation``；群聊跳过这三项，但仍
    执行后端 ``ProgressMonitor`` 与 ``ProgressGuardService``。
    """

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
        turn_tool_evidence: list[ToolResult],
        progress_monitor: ProgressMonitor,
        action_history: list[AgentAction],
    ) -> LoopStepOutcome:
        """执行一个步骤，并返回“继续循环或停止”的结构化结果。"""

        if self.action_processor.run_is_stale(message, run_version):
            return self._superseded(message, trace_id=trace_id, run_id=run_id, run_version=run_version)
        private_loop = str((message.metadata or {}).get("source") or "") != "group"
        progress_hint = (
            build_progress_hint(action_history)
            if private_loop and step_index >= 2 and action_history
            else None
        )
        # 历史属于当前 run（恢复时由其检查点还原）；新输入的运行没有前态，不能借用别的目标。
        previous_action = action_history[-1] if action_history else None
        built = self.context_lifecycle.build_and_trace_context(
            message,
            trace_id=trace_id,
            pending_tool_results=pending_tool_results,
            turn_tool_evidence=turn_tool_evidence,
            run_id=run_id,
            run_version=run_version,
            step_index=step_index,
            progress_hint=progress_hint,
            previous_action=previous_action,
        )
        built, summary_transition = self.context_lifecycle.summarize_and_rebuild_context_if_needed(
            message,
            built=built,
            trace_id=trace_id,
            pending_tool_results=pending_tool_results,
            turn_tool_evidence=turn_tool_evidence,
            run_id=run_id,
            run_version=run_version,
            step_index=step_index,
            budget=budgets.agent,
            progress_hint=progress_hint,
            previous_action=previous_action,
        )
        if self.action_processor.run_is_stale(message, run_version):
            return self._superseded(message, trace_id=trace_id, run_id=run_id, run_version=run_version)
        model_step = self.action_processor.call_agent_action(
            message,
            trace_id=trace_id,
            built_messages=built.messages,
            step_index=step_index,
            budget=budgets.agent,
            run_id=run_id,
            run_version=run_version,
            turn_tool_evidence=turn_tool_evidence,
            previous_step_tool_results=pending_tool_results,
        )
        # 在途网络请求可以自然返回，但旧计划不能越过这一安全边界执行。
        if self.action_processor.run_is_stale(message, run_version):
            return self._superseded(message, trace_id=trace_id, run_id=run_id, run_version=run_version, action=model_step.action)
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
                final_reply=CUSTOMER_EXECUTION_FAILURE_REPLY,
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
            previous_tool_result_count=len(turn_tool_evidence),
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
        continuation = blocking_continuation(action, turn_tool_evidence) if private_loop else None
        if continuation is not None:
            return self._handle_continuation_rejection(
                action=action,
                continuation=continuation,
                message=message,
                trace_id=trace_id,
                run_id=run_id,
                run_version=run_version,
                step_index=step_index,
                summary_transition=summary_transition,
                progress_monitor=progress_monitor,
            )
        processed = self.action_processor.process_action(
            action,
            message=message,
            trace_id=trace_id,
            context_payload=built.payload,
            previous_pending_tool_results=turn_tool_evidence,
            step_index=step_index,
            budgets=budgets,
            run_id=run_id,
            run_version=run_version,
        )
        if self.action_processor.run_is_stale(message, run_version):
            return self._superseded(message, trace_id=trace_id, run_id=run_id, run_version=run_version,
                                    action=processed.action, tool_results=processed.tool_results)
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

    def _superseded(
        self, message: UserMessage, *, trace_id: str, run_id: str, run_version: int,
        action: AgentAction | None = None, tool_results: list[ToolResult] | None = None,
    ) -> LoopStepOutcome:
        """保留真实执行证据，拒绝旧提案；最新输入由同一 Runtime 的后续轮次重规划。"""
        results = list(tool_results or [])
        if tool_results is None and action is not None:
            results = [ToolResult(name=call.name, call_id=call.call_id, called=False, allowed=False,
                                  error="stale run: newer user input superseded this unexecuted action") for call in action.tool_calls]
        payload = {"run_id": run_id, "run_version": run_version,
                   "current_version": self.action_processor.store.conversation_version(message.conversation_id)}
        self.action_processor.trace_recorder.record(trace_id, "conversation_run_stale", payload, level="WARN")
        self.action_processor.trace_recorder.record(trace_id, "final_output", {**payload, "reply": "", "reason": "conversation_run_stale"})
        return LoopStepOutcome(action=action, tool_results=results, pending_tool_results=results,
                               final_reply="", stop_loop=True, runtime_status="superseded")

    def _handle_continuation_rejection(
        self,
        *,
        action: AgentAction,
        continuation: dict,
        message: UserMessage,
        trace_id: str,
        run_id: str,
        run_version: int,
        step_index: int,
        summary_transition: StateTransition | None,
        progress_monitor: ProgressMonitor,
    ) -> LoopStepOutcome:
        feedback = self.action_processor.record_objective_continuation_feedback(
            message,
            trace_id=trace_id,
            action=action,
            continuation=continuation,
            step_index=step_index,
        )
        decision = progress_monitor.observe_runtime_feedback(
            "objective_continuation_rejected",
            {"continuation": continuation, "rejected_status": action.objective_status},
            step_index=step_index,
        )
        progress = self.progress_guard.handle(
            decision,
            message=message,
            trace_id=trace_id,
            pending_tool_results=[feedback],
            run_id=run_id,
            run_version=run_version,
            progress_monitor=progress_monitor,
        )
        tool_results = [feedback]
        if progress.guard_result is not None:
            tool_results.append(progress.guard_result)
        return LoopStepOutcome(
            action=action,
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
        public_results: list[ToolResult] = []
        evidence_results = list(pending)
        if progress.guard_result is not None:
            public_results.append(progress.guard_result)
            evidence_results.append(progress.guard_result)
        return LoopStepOutcome(
            # actions 仅保留通过动作合同的主提案；拒绝原文和错误仍在 trace、反馈及其检查点中。
            # 因此下一步的 previous_decision 不会把非法 JSON 的局部解释当成已接受计划。
            action=None,
            tool_results=public_results,
            evidence_results=evidence_results,
            pending_tool_results=progress.pending_tool_results,
            summary_transition=summary_transition,
            final_reply=progress.final_reply,
            stop_loop=progress.stop_loop,
            runtime_status=progress.runtime_status,
        )


__all__ = ["AgentLoopStepService"]
