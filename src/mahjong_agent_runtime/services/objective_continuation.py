"""编排objectivecontinuation相关的应用服务，不直接承载通道或数据库细节。

"""

from __future__ import annotations

from typing import Any

from ..models import AgentAction, ToolResult


TERMINAL_STATUSES = {"waiting_user", "completed", "needs_human", "unknown"}


def blocking_continuation(
    action: AgentAction,
    previous_tool_results: list[ToolResult],
) -> dict[str, Any] | None:
    """在终止操作过早时返回最新的未解决延续。"""

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
    """构造一个虚拟工具结果，要求模型继续其未完成的计划。"""

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
