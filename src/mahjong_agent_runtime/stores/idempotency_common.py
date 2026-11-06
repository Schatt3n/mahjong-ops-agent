"""定义幂等common相关的存储接口，隔离领域层与具体持久化实现。

"""

from __future__ import annotations

from ..models import ToolResult


IDEMPOTENCY_CLAIM_LEASE_SECONDS = 120


def tool_result_is_retryable_no_effect(result: ToolResult) -> bool:
    """只有执行器确认无副作用的失败可重领；仍须由工具定义显式允许该机制。"""
    return bool(not result.called and not result.allowed and result.error
                and isinstance(result.result, dict) and result.result.get("retryable_no_effect") is True)


def tool_result_is_in_progress(result: ToolResult) -> bool:
    """判断账本条目是否代表进行中的执行声明。"""

    return bool(
        not result.called
        and result.allowed
        and isinstance(result.result, dict)
        and result.result.get("idempotency_status") == "claimed"
    )

