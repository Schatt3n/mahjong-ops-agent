"""封装客户domain相关的领域规则和不变量。

"""

from __future__ import annotations

from typing import Any
from ..models import CustomerProfile
from .game_domain import normalize_requirement
from .stake_values import stake_options, stake_requirements_overlap, unresolved_stake_preferences
from .value_utils import (
    first_present_value,
    is_blank_value,
    list_values_for_keys,
    normalize_smoke_preference,
    smoke_matches,
    value_matches,
)


def classify_current_game_match(
    query: dict[str, Any],
    target: dict[str, Any],
    customer: CustomerProfile | None = None,
) -> tuple[str, list[str]]:
    """对当前牌局进行分类，不将显式冲突转化为匹配。

    精确匹配仍可能在可协商字段（如邻近时间或时长）上存在差异。仅当请求者的稳定档案明确支持返回值时，牌局类型、赌注或吸烟规则上的冲突才会作为备选项暴露。否则，该牌局与此搜索不兼容。
    """

    query = normalize_requirement(query)
    target = normalize_requirement(target)
    decision_fields = current_game_decision_required_fields(query, target)
    if not decision_fields:
        return "exact", []
    if customer is not None and _profile_supports_alternative(customer, target, decision_fields):
        return "profile_supported_alternative", decision_fields
    return "incompatible", decision_fields


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
    return fields


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
    """返回排序器实际识别的条件；附加参数不能让无条件浏览变成假空匹配。"""
    query = normalize_requirement(requirement)
    fields = {key: first_present_value(query, *aliases) for key, _, aliases in CURRENT_GAME_SCORE_FIELDS}
    fields['cap_score'] = first_present_value(query, *CURRENT_GAME_CAP_FIELDS)
    if "stake_range" in query:
        fields["stake_range"] = fields.pop("stake")
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
