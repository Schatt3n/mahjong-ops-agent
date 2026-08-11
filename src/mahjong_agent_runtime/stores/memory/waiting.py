"""实现内存存储层的候补数据访问能力，主要用于本地运行和测试。

"""

from __future__ import annotations

from datetime import datetime

from ...models import WaitingDemand, WaitingDemandStatus, now


class InMemoryWaitingDemandStoreMixin:
    """在聚合存储锁下保持等待列表转换的原子性。"""

    __slots__ = ()

    def waiting_demand(self, demand_id: str) -> WaitingDemand | None:
        with self._lock:
            return self.waiting_demands.get(str(demand_id))

    def insert_waiting_demand(self, demand: WaitingDemand) -> str:
        """新增候补需求，由存储层保证主键和事务约束。"""
        with self._lock:
            if demand.demand_id in self.waiting_demands:
                raise ValueError(f"waiting demand already exists: {demand.demand_id}")
            self.waiting_demands[demand.demand_id] = demand
            return demand.demand_id

    def list_active_demands(self, *, at: datetime | None = None) -> list[WaitingDemand]:
        """列出activedemands；结果保持稳定的领域顺序。"""
        stamp = at or now()
        with self._lock:
            return sorted(
                (
                    item
                    for item in self.waiting_demands.values()
                    if item.status == WaitingDemandStatus.ACTIVE and item.expires_at > stamp
                ),
                key=lambda item: (item.created_at, item.demand_id),
            )

    def update_demand_status(
        self,
        demand_id: str,
        status: WaitingDemandStatus | str,
        matched_game_id: str | None = None,
    ) -> WaitingDemand:
        """更新需求状态，同时维护相关状态不变量。"""
        with self._lock:
            demand = self.waiting_demands.get(str(demand_id))
            if demand is None:
                raise ValueError(f"waiting demand not found: {demand_id}")
            demand.status = status if isinstance(status, WaitingDemandStatus) else WaitingDemandStatus(str(status))
            if matched_game_id is not None:
                demand.matched_game_id = str(matched_game_id)
            return demand

    def claim_waiting_demand_match(
        self,
        demand_id: str,
        game_id: str,
        *,
        at: datetime | None = None,
    ) -> WaitingDemand | None:
        """原子领取候补需求match，防止并发消费者重复处理。"""
        stamp = at or now()
        with self._lock:
            demand = self.waiting_demands.get(str(demand_id))
            if (
                demand is None
                or demand.status != WaitingDemandStatus.ACTIVE
                or demand.expires_at <= stamp
            ):
                return None
            demand.status = WaitingDemandStatus.MATCHED
            demand.matched_game_id = str(game_id)
            return demand

    def release_waiting_demand_match(self, demand_id: str, game_id: str) -> WaitingDemand | None:
        """释放候补需求match，使其他任务能够继续竞争该资源。

"""

        with self._lock:
            demand = self.waiting_demands.get(str(demand_id))
            if (
                demand is None
                or demand.status != WaitingDemandStatus.MATCHED
                or demand.matched_game_id != str(game_id)
            ):
                return None
            demand.status = WaitingDemandStatus.ACTIVE
            demand.matched_game_id = None
            return demand

    def cancel_waiting_demands(
        self,
        *,
        conversation_id: str,
        sender_id: str,
        demand_id: str | None = None,
    ) -> list[WaitingDemand]:
        """取消候补demands，同步终止后续不应再执行的动作。"""
        with self._lock:
            cancelled: list[WaitingDemand] = []
            for demand in self.waiting_demands.values():
                if demand.conversation_id != conversation_id or demand.sender_id != sender_id:
                    continue
                if demand_id and demand.demand_id != demand_id:
                    continue
                if demand.status not in {WaitingDemandStatus.ACTIVE, WaitingDemandStatus.MATCHED}:
                    continue
                demand.status = WaitingDemandStatus.CANCELLED
                cancelled.append(demand)
            return cancelled

    def expire_stale_demands(
        self,
        *,
        at: datetime | None = None,
        trace_id: str | None = None,
    ) -> list[WaitingDemand]:
        """将过期的staledemands标记为终态并保留审计记录。"""
        del trace_id
        stamp = at or now()
        with self._lock:
            expired: list[WaitingDemand] = []
            for demand in self.waiting_demands.values():
                if (
                    demand.status in {WaitingDemandStatus.ACTIVE, WaitingDemandStatus.MATCHED}
                    and demand.expires_at <= stamp
                ):
                    demand.status = WaitingDemandStatus.EXPIRED
                    expired.append(demand)
            return expired


__all__ = ["InMemoryWaitingDemandStoreMixin"]
