"""用已散场且实际到店的牌局推导客户自己的参考偏好。"""

from __future__ import annotations

from collections.abc import Iterable
from math import isfinite
from typing import TypedDict

from ..models import CustomerProfile, Game, GameStatus
from .stake_values import format_number, parse_stake_value
from .value_utils import first_present_value, normalize_smoke_preference


MIN_COMPLETED_GAMES = 3


class AttendancePreference(TypedDict):
    """只暴露一致的对局条件与样本量，供新任务补全缺失条件。"""

    game_type: str
    stake: str
    smoke_preference: str
    completed_games: int


def completed_play_attendees(game: Game) -> tuple[str, ...]:
    """只认真实开打并散场的到店名单，报名、手工结案和未到场不构成偏好证据。"""
    if (game.status != GameStatus.FINISHED or game.closed_reason != "game_play_completed"
            or game.actual_started_at is None or game.actual_ended_at is None
            or game.actual_ended_at <= game.actual_started_at):
        return ()
    return tuple(game.actual_attendee_ids)


def completed_game_preference(games: Iterable[Game], customer_id: str) -> AttendancePreference | None:
    """只从可信开局和散场留下的同一客户记录推导一致组合。"""
    seen: set[str] = set()
    common = None
    for game in games:
        if game.game_id in seen or customer_id not in completed_play_attendees(game):
            continue
        observed = _known_conditions(game.requirement)
        if observed is None or (common is not None and observed != common):
            return None
        common = observed
        seen.add(game.game_id)
    if common is None or len(seen) < MIN_COMPLETED_GAMES:
        return None
    game_type, stake, smoke = common
    return AttendancePreference(game_type=game_type, stake=stake, smoke_preference=smoke, completed_games=len(seen))


def _known_conditions(requirement: dict) -> tuple[str, str, str] | None:
    """要求牌局记录已确定玩法、固定基础档和烟况，避免猜测区间或额外玩法。"""
    game_type = first_present_value(requirement, "game_type", "requested_game")
    stake = _single_stake(first_present_value(requirement, "stake", "base_stake", "base_score"))
    smoke = normalize_smoke_preference(requirement.get("smoke_preference"))
    if (not isinstance(game_type, str) or game_type.strip() in {"", "any", "unknown"}
            or stake is None or first_present_value(
                requirement, "stake_range", "cap_score", "cap_stake", "cap", "cap_limit", "game_variant", "variant", "ruleset",
            ) is not None
            or smoke not in {"no_smoking", "smoking"}):
        return None
    return game_type.strip(), stake, smoke


def _single_stake(value: object) -> str | None:
    """只接纳可确定的有限正数底注，旧画像里的其他表示不转成默认值。"""
    try:
        stake = parse_stake_value(value)
    except ValueError:
        return None
    if stake is None or not isfinite(stake[0]) or stake[0] <= 0 or stake[1] is not None:
        return None
    return format_number(stake[0])


def attendance_preference_agrees(profile: CustomerProfile, preference: AttendancePreference) -> bool:
    """已有正式画像表达的条件优先，历史组合与之冲突时整体不作默认值。"""
    if profile.preferred_games and set(profile.preferred_games) != {preference["game_type"]}:
        return False
    if profile.preferred_stakes and {_single_stake(value) for value in profile.preferred_stakes} != {preference["stake"]}:
        return False
    return (not profile.smoke_preference
            or normalize_smoke_preference(profile.smoke_preference) == preference["smoke_preference"])
