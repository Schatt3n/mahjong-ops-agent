"""定义客户存储相关的存储接口，隔离领域层与具体持久化实现。

"""

from __future__ import annotations

from typing import Any, Protocol

from ..models import CustomerProfile, CustomerRelationship, Game


class CustomerStore(Protocol):
    """定义客户存储需要遵守的抽象接口契约。

"""

    @property
    def customers(self) -> dict[str, CustomerProfile]:
        ...

    @property
    def customer_relationships(self) -> dict[str, CustomerRelationship]:
        ...

    def upsert_customer(self, profile: CustomerProfile) -> None:
        """新增或更新客户，使重复写入收敛到同一条记录。"""
        ...

    def upsert_customer_relationship(self, relationship: CustomerRelationship) -> None:
        """新增或更新客户关系，使重复写入收敛到同一条记录。"""
        ...

    def relationship_between(
        self,
        customer_id: str,
        other_customer_id: str,
    ) -> CustomerRelationship | None:
        ...

    def relationship_context_for_sender(
        self,
        sender_id: str,
        games: list[Game],
    ) -> list[dict[str, Any]]:
        ...

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
        ...

