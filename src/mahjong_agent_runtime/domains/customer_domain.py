"""封装客户domain相关的领域规则和不变量。

"""

from __future__ import annotations

from datetime import datetime
from typing import Any
import math
from ..models import DEFAULT_TZ, CustomerProfile, Game
from .game_domain import (
    START_KIND_ASAP_WHEN_FULL,
    START_KIND_SCHEDULED,
    first_datetime_value,
    normalize_requirement,
    parse_start_time_on_created_date,
)
from .stake_values import stake_options, stake_requirements_overlap, unresolved_stake_preferences
from .value_utils import (
    first_present_value,
    is_blank_value,
    list_values_for_keys,
    normalize_smoke_preference,
    smoke_matches,
    smoke_value_set,
    value_matches,
)

HOST_CONFIRMATION_REQUIRED = "host_confirmation_required"


def classify_current_game_match(
    query: dict[str, Any],
    target: dict[str, Any],
    customer: CustomerProfile | None = None,
) -> tuple[str, list[str]]:
    """对当前牌局进行分类，不将显式冲突转化为匹配。

    明确的精确时长/时长上限属于硬约束；未知时长只作为待核实项。其他玩法、
    档位或烟况冲突仍须由可靠画像支持才能成为备选，不能由时长未知绕过。
    """

    query = normalize_requirement(query)
    target = normalize_requirement(target)
    duration_state = current_game_duration_match(query, target)
    if duration_state == "incompatible":
        return "incompatible", ["duration_hours"]
    decision_fields = current_game_decision_required_fields(query, target)
    if target.get("recruitment_mode") == "board_only":
        return _classify_board_only_game(query, target, customer, decision_fields)
    if not decision_fields:
        return "exact", []
    other_fields = [name for name in decision_fields if name != "duration_hours"]
    if duration_state == "unknown" and not other_fields:
        return "duration_confirmation_required", decision_fields
    if customer is not None and _profile_supports_alternative(customer, target, other_fields):
        return "profile_supported_alternative", decision_fields
    return "incompatible", decision_fields


def _classify_board_only_game(
    query: dict[str, Any], target: dict[str, Any], customer: CustomerProfile | None, decision_fields: list[str],
) -> tuple[str, list[str]]:
    """区分已知冲突和待核实条件。可见的群友自招局始终由桌主决定能否加入。"""
    unresolved = ["host_confirmation"]
    if "game_type" in decision_fields:
        return "incompatible", ["game_type"]
    target_stakes = stake_options(target)
    if not target_stakes:
        unresolved.append("stake")
    elif "stake" in decision_fields:
        if stake_options(query) and not stake_requirements_overlap(query, target, allow_unknown_cap=True):
            return "incompatible", ["stake"]
        unresolved.append("stake")
    if "smoke_preference" in decision_fields:
        target_smoke = smoke_value_set(first_present_value(target, "smoke_preference", "smoke"))
        if target_smoke and "any" not in target_smoke:
            return "incompatible", ["smoke_preference"]
        unresolved.append("smoke_preference")
    if "duration_hours" in decision_fields:
        unresolved.append("duration_hours")
    # 偏好由桌主自行判断。只有已明确的硬限制与已知相反性别才能排除，原因不向另一方披露。
    strict_gender = target.get("required_gender")
    if strict_gender in ("男", "女") and customer and customer.gender in ("男", "女") and customer.gender != strict_gender:
        return "incompatible", ["host_confirmation"]
    for field, aliases in (
        ("game_variant", ("game_variant", "variant")), ("ruleset", ("ruleset",)), ("date", ("date",)),
        ("cap_score", CURRENT_GAME_CAP_FIELDS),
    ):
        wanted = first_present_value(query, *aliases)
        if is_blank_value(wanted):
            continue
        known = first_present_value(target, *aliases)
        if is_blank_value(known):
            unresolved.append(field)
        elif not value_matches(wanted, known):
            return "incompatible", [field]

    opening_kinds = (START_KIND_SCHEDULED, START_KIND_ASAP_WHEN_FULL)
    time_keys = ("planned_start_at", "start_at", "start_datetime", "start_time_at", "start_time")
    wanted_start, known_start = first_datetime_value(query, *time_keys), first_datetime_value(target, *time_keys)
    wanted_kind = START_KIND_SCHEDULED if wanted_start is not None else query.get("start_time_kind")
    known_kind = START_KIND_SCHEDULED if known_start is not None else target.get("start_time_kind")
    if wanted_kind in opening_kinds and known_kind != wanted_kind:
        if known_kind in opening_kinds:
            return "incompatible", ["start_time_kind"]
        unresolved.append("start_time_kind")
    if wanted_start is not None:
        if known_start is not None and known_start != wanted_start:
            return "incompatible", ["planned_start_at"]
        if known_start is None:
            unresolved.append("planned_start_at")
    # 单独比较明确钟点，固定日期只用于规范化，不把看板的19:00猜成某天的安排。
    anchor = datetime(2000, 1, 1, tzinfo=DEFAULT_TZ)
    wanted_clock = wanted_start or parse_start_time_on_created_date(query.get("start_time"), created_at=anchor)
    known_clock = known_start or parse_start_time_on_created_date(target.get("start_time"), created_at=anchor)
    if wanted_clock is not None:
        if known_clock is None:
            unresolved.append("start_time")
        elif wanted_clock.astimezone(DEFAULT_TZ).time() != known_clock.astimezone(DEFAULT_TZ).time():
            return "incompatible", ["start_time"]
    return HOST_CONFIRMATION_REQUIRED, unresolved


def board_only_search_game_projection(game: Game) -> dict[str, Any]:
    """提供核实选项所需的公开桌况和内部局引用。过滤参与者、私密偏好及自由文本。"""
    requirement = game.requirement
    keys = ("game_type", "game_variant", "ruleset", "smoke_preference", "date", "start_time_kind", "start_time",
            "duration_kind", "duration_hours", "max_duration_hours")
    public = {key: requirement[key] for key in keys if requirement.get(key) is not None}
    if stake_options(requirement):
        for key in ("stake", "stake_range", "base_stake", "cap_score", "stakes"):
            if requirement.get(key) is not None:
                public[key] = requirement[key]
    if game.planned_start_at is not None:
        public["planned_start_at"] = game.planned_start_at.isoformat()
    seats = game.seat_summary()
    return {"game_id": game.game_id, "status": game.status.value, "requirement": public,
            "seat_summary": {key: seats[key] for key in ("seats_total", "claimed_seats", "remaining_seats")}}


def current_game_decision_required_fields(
    query: dict[str, Any],
    target: dict[str, Any],
) -> list[str]:
    """返回与当前牌局不同的显式客户约束。"""

    query = normalize_requirement(query)
    target = normalize_requirement(target)
    fields: list[str] = []

    query_game = first_present_value(query, "game_type", "preferred_game", "preferred_games", "game_types")
    if not is_blank_value(query_game):
        target_game = first_present_value(target, "game_type", "preferred_game", "preferred_games", "game_types")
        if not value_matches(query_game, target_game):
            fields.append("game_type")

    query_stakes = list_values_for_keys(query, "stake_range", "stake", "preferred_stake", "preferred_stakes", "stakes", "accepted_stakes")
    target_stakes = list_values_for_keys(target, "stake_range", "stake", "preferred_stake", "preferred_stakes", "stakes", "accepted_stakes")
    if query_stakes and not _stake_options_overlap(query_stakes, target_stakes, query=query, target=target):
        fields.append("stake")

    query_smoke = first_present_value(query, "smoke_preference", "smoke")
    if not is_blank_value(query_smoke):
        target_smoke = first_present_value(target, "smoke_preference", "smoke")
        if not _strict_game_smoke_matches(query_smoke, target_smoke):
            fields.append("smoke_preference")
    if current_game_duration_match(query, target) in {"unknown", "incompatible"}:
        fields.append("duration_hours")
    return fields


def current_game_duration_match(query: dict[str, Any], target: dict[str, Any]) -> str:
    """比较显式小时协议，不从开局/结束时间或生命周期默认值推断约定时长。

    未声明数值约束时保留原时长类别排序；声明后已知冲突不允许画像覆盖，
    目标只有系统派生结束时间时仍为未知，必须单独核实。
    """
    bounds = {}
    for key in ("duration_hours", "max_duration_hours"):
        if key not in query:
            continue
        value = query[key]
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value) or value <= 0):
            raise ValueError(f"{key} must be a finite positive number")
        bounds[key] = value
    if not bounds:
        return "unconstrained"
    if ("duration_hours" in bounds and "max_duration_hours" in bounds
            and bounds["duration_hours"] > bounds["max_duration_hours"]):
        # 自相矛盾的结构化交集为空，未知目标也不能成为看似可协商的符合项。
        return "incompatible"
    value = first_present_value(target, "duration_hours", "duration")
    try:
        duration = None if isinstance(value, bool) else float(value)
    except (TypeError, ValueError, OverflowError):
        duration = None
    if duration is None or not math.isfinite(duration) or duration <= 0:
        # 明确约定的上限可以证明另一个更宽上限，但不能证明精确时长。
        # 这也保证按同一局的权威max约束寻找候选人不会误报与自己冲突。
        target_max = target.get("max_duration_hours")
        try:
            maximum = None if isinstance(target_max, bool) else float(target_max)
        except (TypeError, ValueError, OverflowError):
            maximum = None
        if ("duration_hours" in bounds and maximum is not None and math.isfinite(maximum)
                and maximum > 0 and bounds["duration_hours"] > maximum):
            return "incompatible"
        if ("duration_hours" not in bounds and maximum is not None
                and math.isfinite(maximum) and maximum > 0 and maximum <= bounds["max_duration_hours"]):
            return "matched"
        return "unknown"
    if ("duration_hours" in bounds and duration != bounds["duration_hours"]
            or "max_duration_hours" in bounds and duration > bounds["max_duration_hours"]):
        return "incompatible"
    return "matched"


def _stake_options_overlap(
    query_values: list[Any],
    target_values: list[Any],
    *,
    query: dict[str, Any] | None = None,
    target: dict[str, Any] | None = None,
) -> bool:
    """固定值、多选与区间共用比较器，不能把区间缩成下界。"""
    return stake_requirements_overlap(
        {**(query or {}), "stakes": query_values},
        {**(target or {}), "stakes": target_values},
    )


def _strict_game_smoke_matches(query_value: Any, target_value: Any) -> bool:
    query_values = {
        normalize_smoke_preference(item)
        for item in list_values_for_keys({"value": query_value}, "value")
    }
    target_values = {
        normalize_smoke_preference(item)
        for item in list_values_for_keys({"value": target_value}, "value")
    }
    if not query_values or "any" in query_values:
        return True
    if not target_values or "any" in target_values:
        return False
    return bool(query_values & target_values)


def _profile_supports_alternative(
    customer: CustomerProfile,
    target: dict[str, Any],
    decision_fields: list[str],
) -> bool:
    """只有画像确实接受备选局条件，才把冲突局作为待决策选项返回。"""
    for field in decision_fields:
        if field == "game_type":
            target_game = first_present_value(target, "game_type", "preferred_game", "preferred_games", "game_types")
            if not customer.preferred_games or not value_matches(customer.preferred_games, target_game):
                return False
        elif field == "stake":
            target_stakes = list_values_for_keys(target, "stake_range", "stake", "preferred_stake", "preferred_stakes", "stakes")
            if not customer.preferred_stakes or not _stake_options_overlap(customer.preferred_stakes, target_stakes, target=target):
                return False
        elif field == "smoke_preference":
            target_smoke = first_present_value(target, "smoke_preference", "smoke")
            if is_blank_value(customer.smoke_preference):
                return False
            if (
                normalize_smoke_preference(customer.smoke_preference) != "any"
                and not _strict_game_smoke_matches(customer.smoke_preference, target_smoke)
            ):
                return False
    return True

# 只枚举检索真正执行的字段协议，不把任意附加键当成已经应用的过滤条件。
CURRENT_GAME_SCORE_FIELDS = (
    ("game_type", 30, ("game_type", "preferred_game", "preferred_games", "game_types")),
    ("stake", 25, ("stake_range", "stake", "preferred_stake", "preferred_stakes", "stakes", "accepted_stakes")),
    ("smoke_preference", 15, ("smoke_preference", "smoke")),
    ("start_time_kind", 10, ("start_time_kind", "start_time")),
    ("duration_kind", 10, ("duration_kind", "duration")),
)
CURRENT_GAME_CAP_FIELDS = ("cap_score", "cap_stake", "cap", "cap_limit")


def scored_game_requirement_fields(requirement: dict[str, Any]) -> dict[str, Any]:
    """投影实际排序或硬约束字段；附加参数不能冒充已执行过滤。"""
    query = normalize_requirement(requirement)
    fields = {key: first_present_value(query, *aliases) for key, _, aliases in CURRENT_GAME_SCORE_FIELDS}
    fields['cap_score'] = first_present_value(query, *CURRENT_GAME_CAP_FIELDS)
    if "stake_range" in query:
        fields["stake_range"] = fields.pop("stake")
    fields.update({key: query[key] for key in ("duration_hours", "max_duration_hours") if key in query})
    return {key: value for key, value in fields.items() if not is_blank_value(value)}


def score_requirement(query: dict[str, Any], target: dict[str, Any]) -> tuple[int, list[str]]:
    """按已声明字段评分；与检索作用域共用字段表，避免说明与执行漂移。"""
    query = normalize_requirement(query)
    target = normalize_requirement(target)
    score = 0
    reasons: list[str] = []
    for key, weight, aliases in CURRENT_GAME_SCORE_FIELDS:
        query_value = first_present_value(query, *aliases)
        if is_blank_value(query_value):
            continue
        target_value = first_present_value(target, *aliases)
        if key == "stake":
            matches = stake_requirements_overlap(query, target)
        else:
            matches = smoke_matches(query_value, target_value) if key == "smoke_preference" else value_matches(query_value, target_value)
        if matches:
            score += weight
            reasons.append(f"{key}_matched")
        elif key in {"game_type", "stake", "smoke_preference"}:
            score -= weight
            reasons.append(f"{key}_mismatched")
    cap_query = first_present_value(query, *CURRENT_GAME_CAP_FIELDS)
    if not is_blank_value(cap_query):
        cap_target = first_present_value(target, "cap_score", "cap_stake", "cap", "cap_limit")
        if value_matches(cap_query, cap_target):
            score += 8
            reasons.append("cap_score_matched")
        elif not is_blank_value(cap_target):
            return -999, [*reasons, "cap_score_mismatched"]
        else:
            score -= 8
            reasons.append("cap_score_unknown")
    duration_state = current_game_duration_match(query, target)
    if duration_state == "incompatible":
        return -999, [*reasons, "duration_hours_outside_constraint"]
    if duration_state == "matched":
        score += 10
        reasons.append("duration_hours_matched")
    elif duration_state == "unknown":
        # 允许单独列出待核实时长的公开选项，分类层不会把它作为严格匹配。
        score += 1
        reasons.append("duration_hours_requires_confirmation")
    return score, reasons

def score_customer(requirement: dict[str, Any], customer: CustomerProfile) -> tuple[int, list[str]]:
    """排除已知的区间冲突；未解释画像保留给模型，并且不计为已匹配。"""
    requirement = normalize_requirement(requirement)
    options = stake_options(requirement)
    range_constrained = "stake_range" in requirement or any(lower != upper for lower, upper, _ in options)
    unresolved = unresolved_stake_preferences(customer.preferred_stakes)
    matched = stake_requirements_overlap(
        requirement, {"preferred_stakes": customer.preferred_stakes}, allow_unknown_cap=True,
    )
    if range_constrained and not matched and not unresolved:
        return -999, ["stake_range_not_matched"]
    score = 0
    reasons: list[str] = []
    if range_constrained and not matched and unresolved:
        reasons.append("stake_preference_requires_model_interpretation")
    game_query = first_present_value(requirement, "game_type", "preferred_game", "preferred_games", "game_types")
    smoke_query = first_present_value(requirement, "smoke_preference", "smoke")
    if value_matches(game_query, customer.preferred_games):
        score += 30
        reasons.append("game_type_matched")
    stake_score, stake_reasons = score_stake_preference(requirement, customer.preferred_stakes)
    score += stake_score
    reasons.extend(stake_reasons)
    if smoke_matches(smoke_query, customer.smoke_preference):
        score += 10
        reasons.append("smoke_matched")
    gender = requirement.get("preferred_gender") or requirement.get("gender")
    if value_matches(gender, customer.gender):
        score += 10
        reasons.append("gender_matched")
    score += int(max(0.0, min(1.0, customer.response_score)) * 10)
    score -= int(max(0.0, customer.fatigue_score) * 10)
    return score, reasons

def score_stake_preference(requirement: dict[str, Any], preferred_stakes: list[str]) -> tuple[int, list[str]]:
    """按相交的可接受底注计分，保留固定底注相同但封顶不同的旧有降分语义。"""
    if not stake_options(requirement):
        return 0, []
    target = {"preferred_stakes": preferred_stakes}
    if stake_requirements_overlap(requirement, target, allow_unknown_cap=True):
        return 25, ["stake_matched"]
    if stake_requirements_overlap(requirement, target, ignore_cap=True):
        return 12, ["stake_base_matched", "cap_score_mismatched"]
    return 0, []
