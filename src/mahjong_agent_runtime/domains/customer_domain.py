"""Domain rules for customer domain."""

from __future__ import annotations

from typing import Any
from ..models import CustomerProfile
from .game_domain import normalize_requirement
from .stake_values import parse_number
from .value_utils import (
    first_present_value,
    is_blank_value,
    list_values_for_keys,
    smoke_matches,
    value_matches,
)

def score_requirement(query: dict[str, Any], target: dict[str, Any]) -> tuple[int, list[str]]:
    query = normalize_requirement(query)
    target = normalize_requirement(target)
    score = 0
    reasons: list[str] = []
    for key, weight, aliases in [
        ("game_type", 30, ("game_type", "preferred_game", "preferred_games", "game_types")),
        ("stake", 25, ("stake", "preferred_stake", "preferred_stakes", "stakes")),
        ("smoke_preference", 15, ("smoke_preference", "smoke")),
        ("start_time_kind", 10, ("start_time_kind", "start_time")),
        ("duration_kind", 10, ("duration_kind", "duration")),
    ]:
        query_value = first_present_value(query, *aliases)
        if is_blank_value(query_value):
            continue
        target_value = first_present_value(target, *aliases)
        if value_matches(query_value, target_value):
            score += weight
            reasons.append(f"{key}_matched")
        elif key in {"game_type", "stake", "smoke_preference"}:
            score -= weight
            reasons.append(f"{key}_mismatched")
    cap_query = first_present_value(query, "cap_score", "cap_stake", "cap", "cap_limit")
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
    requirement = normalize_requirement(requirement)
    score = 0
    reasons: list[str] = []
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
    query_values = list_values_for_keys(requirement, "stake", "base_stake", "preferred_stake", "preferred_stakes", "stakes")
    if not query_values:
        return 0, []
    cap_query = first_present_value(requirement, "cap_score", "cap_stake", "cap", "cap_limit")
    exact_base_match = False
    base_only_match = False
    for query_value in query_values:
        normalized_query = normalize_requirement({"stake": query_value, "cap_score": cap_query})
        query_base = parse_number(first_present_value(normalized_query, "base_stake", "stake"))
        query_cap = parse_number(first_present_value(normalized_query, "cap_score", "cap_stake", "cap", "cap_limit"))
        for preferred in preferred_stakes:
            normalized_preference = normalize_requirement({"stake": preferred})
            preferred_base = parse_number(first_present_value(normalized_preference, "base_stake", "stake"))
            preferred_cap = parse_number(first_present_value(normalized_preference, "cap_score", "cap_stake", "cap", "cap_limit"))
            if query_base is None or preferred_base is None or query_base != preferred_base:
                continue
            if query_cap is None or preferred_cap is None or query_cap == preferred_cap:
                exact_base_match = True
                break
            base_only_match = True
        if exact_base_match:
            break
    if exact_base_match:
        return 25, ["stake_matched"]
    if base_only_match:
        return 12, ["stake_base_matched", "cap_score_mismatched"]
    return 0, []
