"""实现群聊场景中的projections能力。

"""

from __future__ import annotations

from typing import Any

from ..domains import START_KIND_ASAP_WHEN_FULL
from ..models import Game


_PUBLIC_REQUIREMENT_KEYS = (
    "requested_game",
    "game_type",
    "game_variant",
    "stake",
    "stake_label",
    "base_stake",
    "cap_score",
    "smoke_preference",
    "start_time_kind",
    "start_time",
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
        "planned_end_at": game.planned_end_at.isoformat() if game.planned_end_at else None,
    }


def public_game_start_display(game: Game) -> str:
    """渲染公共开始时间，不暴露内部枚举。"""

    if str(game.requirement.get("start_time_kind") or "") == START_KIND_ASAP_WHEN_FULL:
        return "人齐开"
    if game.planned_start_at is not None:
        return game.planned_start_at.strftime("%H:%M")
    return str(game.requirement.get("start_time") or "时间待定")


__all__ = ["public_game_start_display", "public_group_game_summary"]
