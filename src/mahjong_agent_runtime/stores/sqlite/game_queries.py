"""SQLite game queries store operations."""

from __future__ import annotations

from typing import Any
from ...models import (
    Game,
    GameStatus,
)
from ...domains import (
    game_contains_customer,
    game_for_model_context,
    join_projection,
    normalize_requirement,
    requested_seat_count_from_search_requirement,
    score_requirement,
    task_memory_anchor_ids,
)

class SQLiteGameQueriesStoreMixin:
    """Backend-specific operations extracted from the compatibility store."""

    __slots__ = ()

    def search_current_games(
        self,
        requirement: dict[str, Any],
        limit: int = 8,
        *,
        sender_id: str | None = None,
        conversation_id: str | None = None,
    ) -> list[dict[str, Any]]:
        requirement = normalize_requirement(requirement)
        scored: list[dict[str, Any]] = []
        requested_seats = requested_seat_count_from_search_requirement(requirement, default=1)
        anchor_ids = task_memory_anchor_ids(requirement, sender_id=sender_id)
        task_excluded = set(self.task_memory_excluded_customer_ids(conversation_id, anchor_ids))
        for game in self.active_games():
            if game.remaining_seats() <= 0:
                continue
            if task_excluded and any(game_contains_customer(game, customer_id) for customer_id in task_excluded):
                continue
            score, reasons = score_requirement(requirement, game.requirement)
            if requirement and score <= 0:
                continue
            scored.append(
                {
                    "game": game_for_model_context(game, self.customers),
                    "score": score,
                    "reasons": reasons or ["active_open_game"],
                    "join_projection": join_projection(game, sender_id=sender_id, requested_seats=requested_seats),
                }
            )
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[: int(limit)]

    def active_game_for_customer(self, customer_id: str) -> Game | None:
        self._expire_stale_games(trace_id="system_lifecycle")
        for game in self.games.values():
            if game.status.value not in {GameStatus.FORMING.value, GameStatus.INVITING.value, GameStatus.READY.value}:
                continue
            if any(item.customer_id == customer_id and item.status in {"joined", "confirmed"} for item in game.participants):
                return game
        return None
