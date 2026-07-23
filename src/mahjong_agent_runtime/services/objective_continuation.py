"""Generic terminal-action validation for tool-owned continuation contracts."""

from __future__ import annotations

from typing import Any

from ..models import AgentAction, ToolResult


TERMINAL_STATUSES = {"waiting_user", "completed", "needs_human", "unknown"}


def blocking_continuation(
    action: AgentAction,
    previous_tool_results: list[ToolResult],
) -> dict[str, Any] | None:
    """Return the latest unresolved continuation when a terminal action is premature."""

    if action.tool_calls or action.objective_status not in TERMINAL_STATUSES:
        return None
    for result in reversed(previous_tool_results):
        continuation = result.result.get("continuation") if isinstance(result.result, dict) else None
        if not isinstance(continuation, dict):
            continue
        if bool(continuation.get("can_stop")):
            return None
        allowed = {
            str(item)
            for item in continuation.get("allowed_terminal_statuses") or []
            if str(item).strip()
        }
        if action.objective_status in allowed:
            return None
        return dict(continuation)
    return None


def continuation_feedback(action: AgentAction, continuation: dict[str, Any]) -> ToolResult:
    """Build a virtual tool result that asks the model to resume its unfinished plan."""

    return ToolResult(
        name="objective_continuation_contract",
        called=False,
        allowed=False,
        result={
            "rejected_objective_status": action.objective_status,
            "continuation": dict(continuation),
            "instruction": (
                "The proposed terminal action was rejected because the latest successful tool left an unresolved "
                "continuation contract. Replan from its authoritative_facts and pending_capabilities. Choose an "
                "available tool that advances the obligation; do not repeat completed reads or merely acknowledge."
            ),
        },
        error="terminal action rejected: unresolved tool continuation",
    )


__all__ = ["blocking_continuation", "continuation_feedback"]
