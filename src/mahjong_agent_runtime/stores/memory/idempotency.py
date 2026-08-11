"""实现内存存储层的幂等数据访问能力，主要用于本地运行和测试。

"""

from __future__ import annotations

from datetime import timedelta

from ...models import AgentRuntimeResult, ToolResult, now
from ..idempotency_common import IDEMPOTENCY_CLAIM_LEASE_SECONDS, tool_result_is_in_progress


class InMemoryIdempotencyStoreMixin:
    """在进程内字典上实现安全的工具和消息重试。"""

    __slots__ = ()

    def idempotent_result(self, key: str | None) -> ToolResult | None:
        with self._lock:
            normalized_key = key or ""
            existing = self.idempotency_ledger.get(normalized_key)
            claimed_at = self.idempotency_claimed_at.get(normalized_key)
            if existing is not None and tool_result_is_in_progress(existing) and claimed_at is not None:
                if claimed_at <= now() - timedelta(seconds=IDEMPOTENCY_CLAIM_LEASE_SECONDS):
                    self.idempotency_ledger.pop(normalized_key, None)
                    self.idempotency_claimed_at.pop(normalized_key, None)
                    return None
            return existing

    def claim_idempotent_result(
        self,
        key: str | None,
        claimed_result: ToolResult,
    ) -> tuple[bool, ToolResult | None]:
        """原子领取idempotent结果，防止并发消费者重复处理。"""
        if not key:
            return True, None
        with self._lock:
            existing = self.idempotency_ledger.get(key)
            if existing is not None:
                claimed_at = self.idempotency_claimed_at.get(key)
                claim_expired = (
                    tool_result_is_in_progress(existing)
                    and claimed_at is not None
                    and claimed_at <= now() - timedelta(seconds=IDEMPOTENCY_CLAIM_LEASE_SECONDS)
                )
                if not claim_expired:
                    return False, existing
            self.idempotency_ledger[key] = claimed_result
            self.idempotency_claimed_at[key] = now()
            return True, None

    def remember_result(self, key: str | None, result: ToolResult) -> None:
        if not key:
            return
        with self._lock:
            self.idempotency_ledger[key] = result
            self.idempotency_claimed_at.pop(key, None)

    def idempotent_message_result(self, message_id: str | None) -> AgentRuntimeResult | None:
        with self._lock:
            return self.message_results.get(message_id or "")

    def remember_message_result(self, message_id: str | None, result: AgentRuntimeResult) -> None:
        if not message_id:
            return
        with self._lock:
            self.message_results.setdefault(message_id, result)

