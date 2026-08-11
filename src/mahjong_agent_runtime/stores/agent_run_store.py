"""定义智能体run存储相关的存储接口，隔离领域层与具体持久化实现。

"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from ..agent_state import AgentRunState
from ..models import StateTransition


class AgentRunStore(Protocol):
    """定义智能体run存储需要遵守的抽象接口契约。

"""

    def create_agent_run(self, state: AgentRunState) -> AgentRunState:
        """创建智能体run，并返回持久化后的权威结果。"""
        ...

    def agent_run(self, run_id: str) -> AgentRunState | None:
        ...

    def save_agent_run(
        self,
        state: AgentRunState,
        *,
        expected_lease_owner: str | None = None,
    ) -> bool:
        """保存智能体run，确保后续读取能够得到一致快照。"""
        ...

    def recoverable_agent_runs(
        self,
        *,
        at: datetime,
        limit: int = 100,
    ) -> list[AgentRunState]:
        ...

    def claim_agent_run(
        self,
        run_id: str,
        *,
        lease_owner: str,
        lease_until: datetime,
        at: datetime,
    ) -> AgentRunState | None:
        """原子领取智能体run，防止并发消费者重复处理。"""
        ...

    def supersede_agent_runs(
        self,
        conversation_id: str,
        *,
        current_version: int,
        trace_id: str,
    ) -> list[StateTransition]:
        ...


__all__ = ["AgentRunStore"]
