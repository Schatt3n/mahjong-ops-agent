"""SQLite customer store operations."""

from __future__ import annotations

from typing import Any
from ...models import (
    CustomerProfile,
    CustomerRelationship,
    Game,
    GameStatus,
)
from ...store import (
    customer_option_load,
    normalize_requirement,
    relationship_context_for_sender,
    relationship_pair_key,
    score_customer,
    score_customer_relationships,
    task_memory_anchor_ids,
)
from .serialization import (
    _dumps,
    _loads,
    _now_iso,
    _relationship_from_payload,
)

class SQLiteCustomerStoreMixin:
    """Backend-specific operations extracted from the compatibility store."""

    __slots__ = ()

    def upsert_customer(self, profile: CustomerProfile) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO runtime_customers(customer_id, payload, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(customer_id) DO UPDATE SET
                    payload=excluded.payload,
                    updated_at=excluded.updated_at
                """,
                (profile.customer_id, _dumps(profile.to_dict()), _now_iso()),
            )

    def upsert_customer_relationship(self, relationship: CustomerRelationship) -> None:
        pair_key = relationship_pair_key(relationship.customer_a_id, relationship.customer_b_id)
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO runtime_customer_relationships(pair_key, customer_a_id, customer_b_id, payload, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(pair_key) DO UPDATE SET
                    customer_a_id=excluded.customer_a_id,
                    customer_b_id=excluded.customer_b_id,
                    payload=excluded.payload,
                    updated_at=excluded.updated_at
                """,
                (
                    pair_key,
                    relationship.customer_a_id,
                    relationship.customer_b_id,
                    _dumps(relationship.to_dict()),
                    relationship.updated_at.isoformat(),
                ),
            )

    def relationship_between(self, customer_id: str, other_customer_id: str) -> CustomerRelationship | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM runtime_customer_relationships WHERE pair_key = ?",
                (relationship_pair_key(customer_id, other_customer_id),),
            ).fetchone()
            if row is None:
                return None
            return _relationship_from_payload(_loads(row["payload"]))

    def relationship_context_for_sender(self, sender_id: str, games: list[Game]) -> list[dict[str, Any]]:
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
        requirement = normalize_requirement(requirement)
        excluded = set(exclude_customer_ids or [])
        anchor_ids = task_memory_anchor_ids(requirement, sender_id=sender_id, excluded_customer_ids=excluded)
        excluded.update(self.task_memory_excluded_customer_ids(conversation_id, anchor_ids))
        self._expire_stale_games(trace_id="system_lifecycle")
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
