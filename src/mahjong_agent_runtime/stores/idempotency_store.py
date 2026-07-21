"""Tool- and message-level idempotency persistence contracts."""

from __future__ import annotations

from typing import Protocol

from ..models import AgentRuntimeResult, ToolResult


class IdempotencyStore(Protocol):
    """Persistence operations that make retries safe."""

    def idempotent_result(self, key: str | None) -> ToolResult | None: ...

    def claim_idempotent_result(
        self,
        key: str | None,
        claimed_result: ToolResult,
    ) -> tuple[bool, ToolResult | None]: ...

    def remember_result(self, key: str | None, result: ToolResult) -> None: ...

    def idempotent_message_result(self, message_id: str | None) -> AgentRuntimeResult | None: ...

    def remember_message_result(
        self,
        message_id: str | None,
        result: AgentRuntimeResult,
    ) -> None: ...

