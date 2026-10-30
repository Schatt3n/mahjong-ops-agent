"""实现 SQLite 存储层的牌局queries数据访问与事务操作。

"""

from __future__ import annotations

from typing import Any
from ...models import (
    Game,
    GameStatus,
)
from ...domains import (
    classify_current_game_match,
    game_contains_customer,
    game_for_model_context,
    join_projection,
    requested_seat_count_from_search_requirement,
    score_requirement,
    task_memory_anchor_ids,
)
from ...domains.customer_domain import scored_game_requirement_fields
from ...domains.game_domain import normalize_current_game_query

class SQLiteGameQueriesStoreMixin:
    """从兼容存储中提取的后端特定操作。"""

    __slots__ = ()

    def search_current_games(
        self,
        requirement: dict[str, Any],
        limit: int = 8,
        *,
        sender_id: str | None = None,
        conversation_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """先按请求方座位与当前作用域排除不可入局对象，再评分并截取结果。"""
        requirement = normalize_current_game_query(requirement)
        has_scored_filters = bool(scored_game_requirement_fields(requirement))
        scored: list[dict[str, Any]] = []
        requested_seats = requested_seat_count_from_search_requirement(requirement, default=1)
        anchor_ids = task_memory_anchor_ids(requirement, sender_id=sender_id)
        task_excluded = set(self.task_memory_excluded_customer_ids(conversation_id, anchor_ids))
        customer = self.customers.get(sender_id or "")
        for game in self.active_games():
            if game.remaining_seats() <= 0:
                continue
            # 容量是可执行匹配的硬边界；评分与limit之前过滤，避免不可入座的局挤掉可用结果。
            projection = join_projection(game, sender_id=sender_id, requested_seats=requested_seats)
            if projection["would_overfill_game"]:
                continue
            if task_excluded and any(game_contains_customer(game, customer_id) for customer_id in task_excluded):
                continue
            score, reasons = score_requirement(requirement, game.requirement)
            match_kind, decision_required_fields = classify_current_game_match(
                requirement,
                game.requirement,
                customer,
            )
            if match_kind == "incompatible":
                continue
            if has_scored_filters and score <= 0:
                continue
            scored.append(
                {
                    "game": game_for_model_context(game, self.customers),
                    "score": score,
                    "reasons": reasons or ["active_open_game"],
                    "match_kind": match_kind,
                    "decision_required_fields": decision_required_fields,
                    "join_projection": projection,
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
