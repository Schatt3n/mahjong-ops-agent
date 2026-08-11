"""实现内存存储层的客户数据访问能力，主要用于本地运行和测试。

"""

from __future__ import annotations

from typing import Any
from ...models import (
    CustomerProfile,
    CustomerRelationship,
    Game,
    GameStatus,
)
from ...domains import (
    customer_option_load,
    normalize_requirement,
    relationship_context_for_sender,
    relationship_pair_key,
    score_customer,
    score_customer_relationships,
    task_memory_anchor_ids,
)

class InMemoryCustomerStoreMixin:
    """从兼容性存储中提取的后端特定操作。"""

    __slots__ = ()

    def upsert_customer(self, profile: CustomerProfile) -> None:
        """新增或更新客户，使重复写入收敛到同一条记录。"""
        with self._lock:
            self.customers[profile.customer_id] = profile

    def upsert_customer_relationship(self, relationship: CustomerRelationship) -> None:
        """新增或更新客户关系，使重复写入收敛到同一条记录。"""
        with self._lock:
            self.customer_relationships[relationship_pair_key(relationship.customer_a_id, relationship.customer_b_id)] = relationship

    def relationship_between(self, customer_id: str, other_customer_id: str) -> CustomerRelationship | None:
        with self._lock:
            return self.customer_relationships.get(relationship_pair_key(customer_id, other_customer_id))

    def relationship_context_for_sender(self, sender_id: str, games: list[Game]) -> list[dict[str, Any]]:
        with self._lock:
            return relationship_context_for_sender(
                sender_id=sender_id,
                games=games,
                customers=self.customers,
                relationship_lookup=self.relationship_between,
            )

    def search_customers(
        self,
        requirement: dict[str, Any],
        *,
        exclude_customer_ids: list[str] | None = None,
        limit: int = 8,
        sender_id: str | None = None,
        conversation_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """查询customers；只返回满足当前作用域和约束的结果。"""
        requirement = normalize_requirement(requirement)
        excluded = set(exclude_customer_ids or [])
        anchor_ids = task_memory_anchor_ids(requirement, sender_id=sender_id, excluded_customer_ids=excluded)
        # 已在局内的参与方是关系锚点，不能再成为本局候选人；
        # 无论模型是否重复传入排除列表都必须成立。
        excluded.update(anchor_ids)
        excluded.update(self.task_memory_excluded_customer_ids(conversation_id, anchor_ids))
        with self._lock:
            self._expire_stale_games_locked(trace_id="system_lifecycle")
            active_games = [
                game
                for game in self.games.values()
                if game.status in {GameStatus.FORMING, GameStatus.INVITING, GameStatus.READY}
            ]
            scored: list[dict[str, Any]] = []
            for customer in self.customers.values():
                if customer.no_contact or customer.customer_id in excluded:
                    continue
                committed, provisional_count = customer_option_load(
                    customer.customer_id,
                    requirement,
                    active_games,
                )
                if committed:
                    continue
                score, reasons = score_customer(requirement, customer)
                if provisional_count:
                    score -= min(15, provisional_count * 3)
                    reasons.append(f"provisional_in_{provisional_count}_overlapping_options")
                relationship_score, relationship_reasons, blocked = score_customer_relationships(
                    customer.customer_id,
                    anchor_ids,
                    self.relationship_between,
                )
                if blocked:
                    continue
                score += relationship_score
                reasons.extend(relationship_reasons)
                if score <= 0:
                    continue
                scored.append({"customer": customer.to_model_context(), "score": score, "reasons": reasons})
            scored.sort(key=lambda item: item["score"], reverse=True)
            return scored[: int(limit)]
