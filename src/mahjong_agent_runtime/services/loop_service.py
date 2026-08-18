"""实现有界的目标驱动智能体主循环，在模型决策、工具执行和停止条件之间迭代。

"""

from __future__ import annotations

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
    """执行有界的“构建上下文、调用模型、执行工具、回喂结果”循环，直到目标进入终态。

    导航：由 ``AgentRuntime._execute_agent_loop`` 调用；每次迭代委托
    ``AgentLoopStepService.execute``。这里只控制预算、检查点、停止条件和恢复位置，
    不解释用户意图，也不实现麻将业务规则。
    """

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
        """执行一次有界循环，直到等待用户、完成、转人工或安全停止。

        首次运行由 ``prepare_turn`` 落输入并建立任务上下文；恢复运行从
        ``AgentRunState`` 的下一步续跑，不重复已完成的工具。每个完整步骤都会写
        checkpoint，因此进程中断不等于整轮从头开始。
        """

        # 每一轮输入使用独立预算和进展监控器，避免上一个用户的消耗或循环状态污染本轮。
        budgets = self.fresh_turn_budgets()
        progress_monitor = self._progress_monitor()
        if run_state is None or not run_state.turn_prepared:
            # 首次执行负责持久化输入、建立任务上下文；恢复运行则直接使用检查点，
            # 不能再次写入用户消息或重放已经完成的工具。
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
            # 检查点保存的动作、观察、预算和进展状态共同决定从哪里续跑。
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

        # 主循环只承载稳定的机制步骤：构建上下文 -> 模型决策 -> 执行工具或回复 ->
        # 回喂工具结果。麻将语义、通道策略和副作用实现均由边界组件承担。
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
            # evidence_results 还可能包含契约校验反馈：模型下一步需要看到它，
            # 但它并不代表真实执行过的业务工具，因此不放进对外 ToolResult 列表。
            step_evidence = outcome.evidence_results or outcome.tool_results
            turn_tool_evidence.extend(step_evidence)
            if outcome.stop_loop:
                final_reply = outcome.final_reply or ""
                runtime_status = outcome.runtime_status or (
                    outcome.action.objective_status if outcome.action is not None else "needs_human"
                )
            if self.run_state_manager is not None and run_state is not None:
                # 每完成一步就落检查点；进程中断后从下一步继续，而不是整轮重放。
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
            # 达到步数上限说明目标未在有界循环内收敛，安全停止并交给人工/后续任务，
            # 不能让模型无限调用工具消耗预算。
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
        """把恢复步骤和本次新步骤合并为统一、可审计的运行结果。"""

        # 幂等命中的工具没有再次执行副作用，其历史状态迁移也不能重复计入本轮。
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
        """按配置上限创建本轮独立的主模型、话术生成和审查调用计数器。"""

        return fresh_turn_budgets(
            self.token_budget,
            self.review_token_budget,
            self.text_generation_token_budget,
        )

    def _step_service(self) -> AgentLoopStepService:
        """组装单步执行器，并为本轮接入无进展检测和安全重规划能力。"""
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
        """创建进展监控器，用动作与观察指纹识别重复、停滞和短周期循环。"""
        return ProgressMonitor(
            repeated_observation_limit=self.repeated_observation_limit,
            consecutive_no_progress_limit=self.consecutive_no_progress_limit,
            max_replan_attempts=self.max_progress_replans,
            max_cycle_period=self.max_cycle_period,
        )

__all__ = ["AgentLoop"]
