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
    """逐个核对具名责任；无关成功不能消除失败责任，旧无名合同仍遵守最近结果。"""

    if action.tool_calls or action.objective_status not in TERMINAL_STATUSES:
        return None
    seen_keys = set()
    legacy_resolved = False
    for result in reversed(previous_tool_results):
        continuation = result.result.get("continuation") if isinstance(result.result, dict) else None
        if not isinstance(continuation, dict):
            continue
        if result.called and result.allowed and not result.error:
            seen_keys.update(str(key) for key in continuation.get("resolved_obligation_keys") or [] if str(key))
        key = str(continuation.get("obligation_key") or "")
        if key:
            if key in seen_keys:
                continue
            seen_keys.add(key)
        elif legacy_resolved:
            continue
        else:
            legacy_resolved = True
        if bool(continuation.get("can_stop")):
            continue
        allowed = {
            str(item)
            for item in continuation.get("allowed_terminal_statuses") or []
            if str(item).strip()
        }
        if action.objective_status in allowed:
            continue
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
                "The proposed terminal action was rejected because a tool left an unresolved "
                "continuation contract. Replan from its authoritative_facts and pending_capabilities. Choose an "
                "available tool that advances the obligation; do not repeat completed reads or merely acknowledge."
            ),
        },
        error="terminal action rejected: unresolved tool continuation",
    )


__all__ = ["blocking_continuation", "continuation_feedback"]
