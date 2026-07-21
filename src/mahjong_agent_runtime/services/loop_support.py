from __future__ import annotations

"""Small lifecycle helpers kept outside the orchestration loop."""

from typing import Any

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
    """Create isolated counters while retaining configured limits."""

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
    """Open task context and durably append the current user turn."""

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
    """Persist and trace the bounded-loop fallback."""

    reply = "这个我先转人工确认一下。"
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
