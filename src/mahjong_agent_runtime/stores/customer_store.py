"""Customer and relationship persistence contracts."""

from __future__ import annotations

from typing import Any, Protocol

from ..models import CustomerProfile, CustomerRelationship, Game


class CustomerStore(Protocol):
    """Persistence operations owned by the customer domain."""

    @property
    def customers(self) -> dict[str, CustomerProfile]: ...

    @property
    def customer_relationships(self) -> dict[str, CustomerRelationship]: ...

    def upsert_customer(self, profile: CustomerProfile) -> None: ...

    def upsert_customer_relationship(self, relationship: CustomerRelationship) -> None: ...

    def relationship_between(
        self,
        customer_id: str,
        other_customer_id: str,
    ) -> CustomerRelationship | None: ...

    def relationship_context_for_sender(
        self,
        sender_id: str,
        games: list[Game],
    ) -> list[dict[str, Any]]: ...

    def search_customers(
        self,
        requirement: dict[str, Any],
        *,
        exclude_customer_ids: list[str] | None = None,
        limit: int = 8,
        sender_id: str | None = None,
        conversation_id: str | None = None,
    ) -> list[dict[str, Any]]: ...

