"""封装牌局domain相关的领域规则和不变量。

"""

from __future__ import annotations

from typing import Any
from datetime import datetime
import re
from datetime import timedelta
from ..models import (
    DEFAULT_TZ,
    Game,
    GameStatus,
    RecruitmentStatus,
    StateTransition,
    now,
)
from .stake_values import (
    format_number,
    parse_number,
    parse_stake_value,
    parse_stake_range,
)
from .value_utils import (
    first_present_value,
    is_blank_value,
)

DEFAULT_ASAP_GAME_TTL_HOURS = 4

DEFAULT_UNKNOWN_DURATION_HOURS = 4

DEFAULT_OVERNIGHT_DURATION_HOURS = 8

START_KIND_SCHEDULED = "scheduled"

START_KIND_ASAP_WHEN_FULL = "asap_" "when_full"

DURATION_KIND_OVERNIGHT = "overnight"

DEFAULT_RECRUITMENT_LEAD_HOURS = 2

GAME_RECRUITMENT_TASK_TYPE = "activate_game_recruitment"

ALLOWED_GAME_TRANSITIONS = {
    GameStatus.FORMING.value: {
        GameStatus.INVITING.value,
        GameStatus.READY.value,
        GameStatus.CANCELLED.value,
    },
    GameStatus.INVITING.value: {
        GameStatus.READY.value,
        GameStatus.CANCELLED.value,
        GameStatus.FINISHED.value,
    },
    GameStatus.READY.value: {GameStatus.FINISHED.value, GameStatus.CANCELLED.value},
    GameStatus.CANCELLED.value: set(),
    GameStatus.FINISHED.value: set(),
}

PROTECTED_REQUIREMENT_PATCH_FIELDS = {
    "current_player_count",
    "current_player_ids",
    "existing_player_ids",
    "known_player_count",
    "known_player_ids",
    "needed_seats",
    "organizer_id",
    "participant_ids",
    "party_size",
    "player_count",
    "remaining_seats",
    "requester_id",
    "requesting_party",
    "parties",
    "seat_claims",
    "seats_total",
    "planned_end_at",
    "lifecycle_expires_at",
    "lifecycle_ttl_hours",
    "latest_start_at",
    "recruitment_opens_at",
    "recruitment_status",
    "recruitment_lead_hours",
}

def normalize_requirement(requirement: dict[str, Any] | None) -> dict[str, Any]:
    """规范化模型提交的业务条件；区间独立保存，不推断自然语言或选择固定底注。"""

    normalized = dict(requirement or {})
    stake_value = first_present_value(
        normalized,
        "stake",
        "base_stake",
        "base_score",
        "preferred_stake",
        "level",
    )
    cap_value = first_present_value(normalized, "cap_score", "cap_stake", "cap", "cap_limit")
    bounds = parse_stake_range(normalized["stake_range"]) if "stake_range" in normalized else parse_stake_range(stake_value)
    if "stake_range" in normalized and bounds is None:
        raise ValueError("stake_range must be an object with min and max")
    if bounds is not None:
        if not is_blank_value(stake_value) and parse_stake_range(stake_value) != bounds:
            raise ValueError("stake_range cannot be combined with a different fixed stake")
        normalized["stake_range"] = {"min": bounds[0], "max": bounds[1]}
        # 区间是尚未收窄的有效条件，绝不以最小值冒充已经选定的底注。
        for key in ("stake", "base_stake", "base_score", "preferred_stake", "level"):
            normalized.pop(key, None)
        normalized["stake_label"] = f"{format_number(bounds[0])}～{format_number(bounds[1])}元"
    parsed = parse_stake_value(stake_value) if bounds is None else None
    cap_number = parse_number(cap_value)
    if parsed is not None or cap_number is not None:
        base_number = parsed[0] if parsed else parse_number(stake_value)
        parsed_cap = parsed[1] if parsed else None
        final_cap = parsed_cap if parsed_cap is not None else cap_number
        if base_number is not None:
            normalized["base_stake"] = base_number
            normalized["stake"] = format_number(base_number)
        if final_cap is not None:
            normalized["cap_score"] = final_cap
            if bounds is not None:
                normalized["stake_label"] += f"，{format_number(final_cap)}封顶"
        if base_number is not None and final_cap is not None:
            normalized.setdefault("stake_label", f"{format_number(base_number)}-{format_number(final_cap)}")
            normalized.setdefault("level", f"{format_number(base_number)}-{format_number(final_cap)}")
        elif base_number is not None:
            normalized.setdefault("stake_label", format_number(base_number))

    known_player_count = _whole_number(normalized.get("known_player_count"))
    needed_seats = _whole_number(normalized.get("needed_seats"))
    if (
        known_player_count is not None
        and needed_seats is not None
        and 1 <= known_player_count <= 3
        and 1 <= needed_seats <= 3
        and known_player_count + needed_seats == 4
    ):
        normalized["known_player_count"] = known_player_count
        normalized["needed_seats"] = needed_seats
        normalized["seat_format"] = f"{known_player_count}7{needed_seats}"
    return normalized


def merge_requirement_patch(requirement: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """替换档位表达时清除原表达的派生字段，避免固定值与区间在同一局中并存。"""
    base = dict(requirement)
    if any(key in patch for key in ("stake_range", "stake", "base_stake", "base_score", "preferred_stake", "level")):
        if "stake_range" in base or "stake_range" in patch:
            base.pop("user_visible_summary", None)
        for key in ("stake_range", "stake", "base_stake", "base_score", "preferred_stake", "level", "stake_label"):
            base.pop(key, None)
    return normalize_requirement({**base, **patch})


def _whole_number(value: Any) -> int | None:
    """解析非负整数域计数，但不接受布尔值。"""

    if isinstance(value, bool):
        return None
    parsed = parse_number(value)
    if parsed is None or parsed < 0 or not float(parsed).is_integer():
        return None
    return int(parsed)

def apply_game_lifecycle(game: Game) -> None:
    """应用牌局生命周期规则，并保持领域状态合法。"""
    lifecycle = derive_game_lifecycle(game.requirement, created_at=game.created_at)
    game.planned_start_at = lifecycle["planned_start_at"]
    game.planned_end_at = lifecycle["planned_end_at"]
    game.expires_at = lifecycle["expires_at"]
    game.requirement = {
        **dict(game.requirement),
        **lifecycle["requirement_patch"],
    }
    apply_game_recruitment_policy(game)

def game_recruitment_task_id(game_id: str) -> str:
    """使用确定性任务ID，从而调度更新不会创建重复项。"""

    return f"scheduled_recruitment_{game_id}"

def recruitment_open_time(game: Game) -> datetime | None:
    if game.planned_start_at is None:
        return None
    return normalize_datetime(game.planned_start_at) - timedelta(hours=DEFAULT_RECRUITMENT_LEAD_HOURS)

def apply_game_recruitment_policy(game: Game, *, at: datetime | None = None) -> None:
    """应用牌局recruitment策略规则，并保持领域状态合法。

    """

    stamp = normalize_datetime(at or now())
    opens_at = recruitment_open_time(game)
    game.recruitment_opens_at = opens_at
    if game.status == GameStatus.CANCELLED:
        status = RecruitmentStatus.CANCELLED
    elif game.status == GameStatus.FINISHED or game.status == GameStatus.READY:
        status = RecruitmentStatus.COMPLETED
    elif opens_at is not None and stamp < opens_at:
        status = RecruitmentStatus.SCHEDULED
    elif game.status == GameStatus.INVITING:
        status = RecruitmentStatus.ACTIVE
    else:
        status = RecruitmentStatus.OPEN
    game.recruitment_status = status
    patch = {
        "recruitment_status": status.value,
        "recruitment_lead_hours": DEFAULT_RECRUITMENT_LEAD_HOURS,
    }
    if opens_at is not None:
        patch["recruitment_opens_at"] = opens_at.isoformat()
    else:
        game.requirement.pop("recruitment_opens_at", None)
    game.requirement = {**dict(game.requirement), **patch}

def game_schedule_sort_key(game: Game) -> tuple[datetime, datetime, str]:
    """将固定时间牌局按时间顺序排前，然后是灵活牌局。"""

    distant_future = datetime.max.replace(tzinfo=DEFAULT_TZ)
    return (
        normalize_datetime(game.planned_start_at) if game.planned_start_at else distant_future,
        normalize_datetime(game.created_at),
        game.game_id,
    )

def expire_game_if_stale(game: Game, *, at: datetime, trace_id: str) -> StateTransition | None:
    """将过期的牌局ifstale标记为终态并保留审计记录。"""
    if game.status.value not in {GameStatus.FORMING.value, GameStatus.INVITING.value, GameStatus.READY.value}:
        return None
    if game.expires_at is None:
        apply_game_lifecycle(game)
    if game.expires_at is None or at < game.expires_at:
        return None
    old = game.status.value
    if game.status == GameStatus.READY or game.remaining_seats() <= 0:
        game.status = GameStatus.FINISHED
        game.closed_reason = "game_time_elapsed"
    else:
        game.status = GameStatus.CANCELLED
        game.closed_reason = "expired_without_full_table"
    apply_game_recruitment_policy(game, at=at)
    game.updated_at = at
    return StateTransition("game", game.game_id, old, game.status.value, game.closed_reason, trace_id)

def derive_game_lifecycle(requirement: dict[str, Any], *, created_at: datetime) -> dict[str, Any]:
    """根据权威事实推导牌局生命周期，不引入未经证实的信息。"""
    requirement = dict(requirement or {})
    created_at = normalize_datetime(created_at)
    start_kind = str(requirement.get("start_time_kind") or "").strip()
    planned_start_at = first_datetime_value(
        requirement,
        "planned_start_at",
        "start_at",
        "start_datetime",
        "start_time_at",
        "start_time",
    )
    if planned_start_at is None and start_kind == START_KIND_SCHEDULED:
        planned_start_at = parse_start_time_on_created_date(requirement.get("start_time"), created_at=created_at)
    duration_hours = duration_hours_from_requirement(requirement)
    planned_end_at: datetime | None = None
    if planned_start_at is not None:
        planned_end_at = planned_start_at + timedelta(hours=duration_hours)

    if start_kind == START_KIND_ASAP_WHEN_FULL:
        expires_at = created_at + timedelta(hours=DEFAULT_ASAP_GAME_TTL_HOURS)
    elif planned_end_at is not None:
        expires_at = planned_end_at
    elif planned_start_at is not None:
        expires_at = planned_start_at + timedelta(hours=DEFAULT_UNKNOWN_DURATION_HOURS)
    else:
        expires_at = created_at + timedelta(hours=DEFAULT_ASAP_GAME_TTL_HOURS)

    patch: dict[str, Any] = {
        "lifecycle_expires_at": expires_at.isoformat(),
        "lifecycle_ttl_hours": round((expires_at - created_at).total_seconds() / 3600, 3),
    }
    if planned_start_at is not None:
        patch["planned_start_at"] = planned_start_at.isoformat()
    if planned_end_at is not None:
        patch["planned_end_at"] = planned_end_at.isoformat()
    if start_kind == START_KIND_ASAP_WHEN_FULL:
        patch["latest_start_at"] = expires_at.isoformat()
    return {
        "planned_start_at": planned_start_at,
        "planned_end_at": planned_end_at,
        "expires_at": expires_at,
        "requirement_patch": patch,
    }

def duration_hours_from_requirement(requirement: dict[str, Any]) -> float:
    raw = first_present_value(requirement, "duration_hours", "duration")
    parsed = parse_number(raw)
    if parsed is not None and parsed > 0:
        return min(24.0, parsed)
    duration_kind = str(requirement.get("duration_kind") or "").strip()
    if duration_kind == DURATION_KIND_OVERNIGHT:
        return DEFAULT_OVERNIGHT_DURATION_HOURS
    return DEFAULT_UNKNOWN_DURATION_HOURS

def first_datetime_value(payload: dict[str, Any], *keys: str) -> datetime | None:
    for key in keys:
        value = payload.get(key)
        if is_blank_value(value):
            continue
        parsed = parse_datetime_value(value)
        if parsed is not None:
            return parsed
    return None

def parse_datetime_value(value: Any) -> datetime | None:
    """解析datetimevalue，并把非结构化输入转换为稳定合同。"""
    if is_blank_value(value):
        return None
    if isinstance(value, datetime):
        return normalize_datetime(value)
    try:
        return normalize_datetime(datetime.fromisoformat(str(value)))
    except ValueError:
        return None

def normalize_datetime(value: datetime) -> datetime:
    """规范化datetime，消除等价输入之间的表示差异。"""
    if value.tzinfo is None:
        return value.replace(tzinfo=DEFAULT_TZ)
    return value

def parse_start_time_on_created_date(value: Any, *, created_at: datetime) -> datetime | None:
    """解析starttimeoncreateddate，并把非结构化输入转换为稳定合同。"""
    if is_blank_value(value):
        return None
    text = str(value).strip()
    match = re.fullmatch(r"(?P<hour>\d{1,2})(?::|：)?(?P<minute>\d{2})?", text)
    if match is None:
        match = re.fullmatch(r"(?P<hour>\d{1,2})\s*(?:点|时)(?P<minute>\d{1,2})?", text)
    if match is None:
        return None
    hour = int(match.group("hour"))
    minute = int(match.group("minute") or 0)
    if hour > 23 or minute > 59:
        return None
    return created_at.replace(hour=hour, minute=minute, second=0, microsecond=0)
