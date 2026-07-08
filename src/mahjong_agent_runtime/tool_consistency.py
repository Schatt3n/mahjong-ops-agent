from __future__ import annotations

from typing import Any

from .models import ToolResult
from .store import normalize_requirement


CONSISTENT_REQUIREMENT_FIELDS = (
    "game_type",
    "stake",
    "smoke_preference",
    "start_time_kind",
    "start_time",
    "duration_kind",
    "duration_hours",
    "known_player_count",
    "needed_seats",
)


def validate_tool_call_consistency(call: Any, previous_tool_results: list[ToolResult]) -> str | None:
    if getattr(call, "name", "") != "create_game":
        return None
    current_requirement = call.arguments.get("requirement") if isinstance(call.arguments, dict) else None
    if not isinstance(current_requirement, dict):
        return None
    reference_requirement = latest_read_requirement(previous_tool_results, tool_name="search_current_games")
    if not reference_requirement:
        return None
    current_requirement = normalize_requirement(current_requirement)
    reference_requirement = normalize_requirement(reference_requirement)
    mismatches: list[str] = []
    for field in CONSISTENT_REQUIREMENT_FIELDS:
        expected = normalized_requirement_value(reference_requirement.get(field))
        if expected in {None, ""}:
            continue
        actual = normalized_requirement_value(current_requirement.get(field))
        if actual != expected:
            mismatches.append(f"{field}: expected {expected!r} from previous search_current_games, got {actual!r}")
    if not mismatches:
        return None
    return (
        "tool argument consistency violation: create_game.requirement conflicts with previous "
        "search_current_games.requirement; " + "; ".join(mismatches)
    )


def latest_read_requirement(previous_tool_results: list[ToolResult], *, tool_name: str) -> dict[str, Any] | None:
    for result in reversed(previous_tool_results):
        if result.result.get("reference_tool_name") == tool_name and isinstance(result.result.get("reference_requirement"), dict):
            return result.result["reference_requirement"]
        if result.name != tool_name or not result.called or not result.allowed:
            continue
        requirement = result.result.get("requirement")
        if isinstance(requirement, dict):
            return requirement
    return None


def normalized_requirement_value(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [normalized_requirement_value(item) for item in value]
    return value
