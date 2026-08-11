"""定义幂等common相关的存储接口，隔离领域层与具体持久化实现。

"""

from __future__ import annotations

from ..models import ToolResult


IDEMPOTENCY_CLAIM_LEASE_SECONDS = 120


def tool_result_is_in_progress(result: ToolResult) -> bool:
    """判断账本条目是否代表进行中的执行声明。"""

    return bool(
        not result.called
        and result.allowed
        and isinstance(result.result, dict)
        and result.result.get("idempotency_status") == "claimed"
    )

