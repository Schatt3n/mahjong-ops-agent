"""定义候补存储相关的存储接口，隔离领域层与具体持久化实现。

"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from ..models import WaitingDemand, WaitingDemandStatus


class WaitingDemandStore(Protocol):
    """定义候补需求存储需要遵守的抽象接口契约。

"""

    @property
    def waiting_demands(self) -> dict[str, WaitingDemand]:
        ...

    def waiting_demand(self, demand_id: str) -> WaitingDemand | None:
        ...

    def insert_waiting_demand(self, demand: WaitingDemand) -> str:
        """新增候补需求，由存储层保证主键和事务约束。"""
        ...

    def list_active_demands(self, *, at: datetime | None = None) -> list[WaitingDemand]:
        """列出activedemands；结果保持稳定的领域顺序。"""
        ...

    def update_demand_status(
        self,
        demand_id: str,
        status: WaitingDemandStatus | str,
        matched_game_id: str | None = None,
    ) -> WaitingDemand:
        """更新需求状态，同时维护相关状态不变量。"""
        ...

    def claim_waiting_demand_match(
        self,
        demand_id: str,
        game_id: str,
        *,
        at: datetime | None = None,
    ) -> WaitingDemand | None:
        """原子领取候补需求match，防止并发消费者重复处理。"""
        ...

    def release_waiting_demand_match(self, demand_id: str, game_id: str) -> WaitingDemand | None:
        """释放候补需求match，使其他任务能够继续竞争该资源。"""
        ...

    def cancel_waiting_demands(
        self,
        *,
        conversation_id: str,
        sender_id: str,
        demand_id: str | None = None,
    ) -> list[WaitingDemand]:
        """取消候补demands，同步终止后续不应再执行的动作。"""
        ...

    def expire_stale_demands(
        self,
        *,
        at: datetime | None = None,
        trace_id: str | None = None,
    ) -> list[WaitingDemand]:
        """将过期的staledemands标记为终态并保留审计记录。"""
        ...


__all__ = ["WaitingDemandStore"]
