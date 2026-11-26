"""实现群聊场景中的projections能力。

"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from ..domains import START_KIND_ASAP_WHEN_FULL
from ..domains.game_domain import OPENING_NOT_BEFORE, flexible_opening_not_before
from ..domains.game_participants import planned_start_has_passed
from ..domains.model_context import has_agreed_game_duration
from ..models import Game, GameStatus


_PUBLIC_REQUIREMENT_KEYS = (
    "requested_game",
    "game_type",
    "game_variant",
    "stake",
    "stake_range",
    "stake_label",
    "base_stake",
    "cap_score",
    "smoke_preference",
    "start_time_kind",
    "start_time",
    OPENING_NOT_BEFORE,
    "duration_kind",
    "duration_hours",
)


def public_group_game_summary(game: Game) -> dict[str, Any]:
    """返回可能跨越公共房间或客户边界的事实。

    持久化的需求还包含内部参与方/联系人结构。通过允许列表构建此投影可防止这些结构进入模型提示，并使隐私独立于模型遵从性。座位数从聚合派生，而非信任复制到原始请求中的过期字段。
    """

    requirement = dict(game.requirement or {})
    seat_summary = game.seat_summary()
    remaining = int(seat_summary["remaining_seats"])
    claimed = int(seat_summary["claimed_seats"])
    public_requirement = {
        key: requirement.get(key)
        for key in _PUBLIC_REQUIREMENT_KEYS
        if requirement.get(key) is not None
    }
    public_requirement.update(
        {
            "known_player_count": claimed,
            "needed_seats": remaining,
        }
    )
    return {
        "game_id": game.game_id,
        "status": game.status.value,
        "public_requirement": public_requirement,
        "seats_total": int(game.seats_total),
        "claimed_seats": claimed,
        "remaining_seats": remaining,
        "seat_code": f"{claimed}7{remaining}",
        "planned_start_at": game.planned_start_at.isoformat() if game.planned_start_at else None,
        "planned_end_at": (game.planned_end_at.isoformat()
                           if game.planned_end_at and has_agreed_game_duration(requirement) else None),
    }


def public_game_start_display(game: Game) -> str:
    """展示带日期的权威时间，避免跨日同钟点的不同牌局在看板中无法区分。"""

    if str(game.requirement.get("start_time_kind") or "") == START_KIND_ASAP_WHEN_FULL:
        if floor := flexible_opening_not_before(game.requirement):
            return f"{floor:%m月%d日 %H:%M}后，开场待定"
        return "人齐开"
    if game.planned_start_at is not None:
        return game.planned_start_at.strftime("%Y-%m-%d %H:%M")
    return str(game.requirement.get("start_time") or "时间待定")


def board_registration_facts(game: Game, *, at: datetime) -> dict[str, bool]:
    """投影所有读者共用的认领条件；个人晚到同意只由占席事务核对。"""
    elapsed = planned_start_has_passed(game, at=at)
    expired = game.expires_at is not None and game.expires_at <= at
    self_recruitment = game.requirement.get("recruitment_mode") == "board_only"
    return {
        "planned_start_passed": elapsed,
        "registration_expired": expired,
        "self_recruitment": self_recruitment,
        "direct_claimable": (
            not elapsed and not expired and not self_recruitment
            and game.status in {GameStatus.FORMING, GameStatus.INVITING}
            and game.remaining_seats() > 0
        ),
    }


__all__ = ["board_registration_facts", "public_game_start_display", "public_group_game_summary"]
