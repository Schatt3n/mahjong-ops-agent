"""定义幂等存储相关的存储接口，隔离领域层与具体持久化实现。

"""

from __future__ import annotations

from typing import Protocol

from ..models import AgentRuntimeResult, ToolResult


class IdempotencyStore(Protocol):
    """定义幂等存储需要遵守的抽象接口契约。

"""

    def idempotent_result(self, key: str | None) -> ToolResult | None:
        ...

    def claim_idempotent_result(
        self,
        key: str | None,
        claimed_result: ToolResult,
    ) -> tuple[bool, ToolResult | None]:
        """原子领取idempotent结果，防止并发消费者重复处理。"""
        ...

    def remember_result(self, key: str | None, result: ToolResult) -> None:
        ...

    def idempotent_message_result(self, message_id: str | None) -> AgentRuntimeResult | None:
        ...

    def remember_message_result(
        self,
        message_id: str | None,
        result: AgentRuntimeResult,
    ) -> None:
        ...

