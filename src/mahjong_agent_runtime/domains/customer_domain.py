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
    normalize_smoke_preference,
    smoke_matches,
    value_matches,
)


def classify_current_game_match(
    query: dict[str, Any],
    target: dict[str, Any],
    customer: CustomerProfile | None = None,
) -> tuple[str, list[str]]:
    """Classify a current game without turning explicit conflicts into matches.

    Exact matches may still differ on negotiable fields such as nearby time or
    duration. A conflict on game type, stake, or smoking rule is only exposed
    as an alternative when the requester's stable profile explicitly supports
    the returned value. Otherwise the game is incompatible with this search.
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
    """Return explicit customer constraints that differ from a current game."""

    query = normalize_requirement(query)
    target = normalize_requirement(target)
    fields: list[str] = []

    query_game = first_present_value(query, "game_type", "preferred_game", "preferred_games", "game_types")
    if not is_blank_value(query_game):
        target_game = first_present_value(target, "game_type", "preferred_game", "preferred_games", "game_types")
        if not value_matches(query_game, target_game):
            fields.append("game_type")

    query_stakes = list_values_for_keys(query, "stake", "preferred_stake", "preferred_stakes", "stakes")
    target_stakes = list_values_for_keys(target, "stake", "preferred_stake", "preferred_stakes", "stakes")
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
    if not query_values or not target_values:
        return False
    query_cap = first_present_value(query or {}, "cap_score", "cap_stake", "cap", "cap_limit")
    target_cap = first_present_value(target or {}, "cap_score", "cap_stake", "cap", "cap_limit")
    for query_value in query_values:
        normalized_query = normalize_requirement({"stake": query_value, "cap_score": query_cap})
        query_base = parse_number(first_present_value(normalized_query, "base_stake", "stake"))
        normalized_query_cap = parse_number(
            first_present_value(normalized_query, "cap_score", "cap_stake", "cap", "cap_limit")
        )
        for target_value in target_values:
            normalized_target = normalize_requirement({"stake": target_value, "cap_score": target_cap})
            target_base = parse_number(first_present_value(normalized_target, "base_stake", "stake"))
            normalized_target_cap = parse_number(
                first_present_value(normalized_target, "cap_score", "cap_stake", "cap", "cap_limit")
            )
            if query_base is None or target_base is None or query_base != target_base:
                continue
            if normalized_query_cap is None or normalized_query_cap == normalized_target_cap:
                return True
    return False


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
    for field in decision_fields:
        if field == "game_type":
            target_game = first_present_value(target, "game_type", "preferred_game", "preferred_games", "game_types")
            if not customer.preferred_games or not value_matches(customer.preferred_games, target_game):
                return False
        elif field == "stake":
            target_stakes = list_values_for_keys(target, "stake", "preferred_stake", "preferred_stakes", "stakes")
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
        matches = smoke_matches(query_value, target_value) if key == "smoke_preference" else value_matches(query_value, target_value)
        if matches:
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
