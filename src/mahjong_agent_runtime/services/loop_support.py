"""编排循环support相关的应用服务，不直接承载通道或数据库细节。

"""

from __future__ import annotations

from typing import Any

from ..customer_visible_contract import CUSTOMER_EXECUTION_FAILURE_REPLY

from ..budget import TokenBudget
from ..hooks import HookManager
from ..models import StateTransition, UserMessage
from ..runtime_components import TurnBudgets
from ..stores import AgentStore
from ..task_context import TaskContextManager
from .action_service import ActionProcessor


def fresh_turn_budgets(
    agent: TokenBudget,
    review: TokenBudget,
    text_generation: TokenBudget,
) -> TurnBudgets:
    """在保留配置限制的同时创建隔离计数器。"""

    return TurnBudgets(
        agent=_fresh_budget(agent),
        review=_fresh_budget(review),
        text_generation=_fresh_budget(text_generation),
    )


def prepare_turn(
    *,
    store: AgentStore,
    trace_recorder: Any,
    hook_manager: HookManager | None,
    task_context_manager: TaskContextManager,
    message: UserMessage,
    trace_id: str,
) -> list[StateTransition]:
    """打开任务上下文并持久化追加当前用户轮次。"""

    task_context = task_context_manager.prepare(message, trace_id=trace_id)
    trace_recorder.record(trace_id, "task_context_prepared", task_context.to_dict())
    _emit(hook_manager, "after_task_context_prepared", trace_id, task_context.to_dict())
    store.append_user_turn(message, trace_id)
    trace_recorder.record(trace_id, "user_input", {"message": message.to_dict()})
    _emit(hook_manager, "after_user_turn_appended", trace_id, {"message": message.to_dict()})
    return list(task_context.transitions)


def handle_max_steps(
    *,
    action_processor: ActionProcessor,
    trace_recorder: Any,
    message: UserMessage,
    trace_id: str,
    run_id: str,
    run_version: int,
) -> str:
    """处理maxsteps，协调本层职责范围内的完整流程。

"""

    reply = CUSTOMER_EXECUTION_FAILURE_REPLY
    action_processor.append_pending_assistant_turn(
        message.conversation_id,
        reply,
        trace_id,
        run_id=run_id,
        run_version=run_version,
    )
    trace_recorder.record(
        trace_id,
        "final_output",
        {"reply": reply, "reason": "max_steps_exceeded"},
        level="WARN",
    )
    return reply


def _fresh_budget(budget: TokenBudget) -> TokenBudget:
    return TokenBudget(
        max_tokens_per_call=budget.max_tokens_per_call,
        max_calls_per_turn=budget.max_calls_per_turn,
    )


def _emit(
    hook_manager: HookManager | None,
    event_name: str,
    trace_id: str,
    payload: dict[str, Any],
) -> None:
    if hook_manager is not None:
        hook_manager.emit(event_name, trace_id=trace_id, payload=payload)


__all__ = ["fresh_turn_budgets", "handle_max_steps", "prepare_turn"]
