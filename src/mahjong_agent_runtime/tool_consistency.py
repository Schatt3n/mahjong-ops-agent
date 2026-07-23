from __future__ import annotations

"""跨工具参数一致性校验。

设计理念：
- 模型可以根据工具结果动态调整计划，但不能在没有用户新输入的情况下偷偷改关键条件。
- 例如上一轮按 16:00/0.5/无烟搜索现有局，下一轮 create_game 不能变成人齐开/1块/有烟。
- 这类校验是生产边界，不是补业务 if-else；它保护的是“读到什么条件，就按什么条件继续写”。
"""

import re
from typing import Any

from .models import ToolResult
from .domains import normalize_requirement
from .domains.value_utils import normalize_smoke_preference


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

EXPLICIT_FACT_TOOL_ARGUMENTS = {
    "search_current_games": "requirement",
    "search_customers": "requirement",
    "create_game": "requirement",
    "update_game_requirement": "requirement_patch",
}

EXPLICIT_FACT_CONSISTENT_FIELDS = (
    "game_type",
    "stake",
    "smoke_preference",
    "start_time_kind",
    "start_time",
    "planned_start_at",
    "duration_hours",
    "known_player_count",
    "needed_seats",
    "seat_format",
)


def validate_tool_call_consistency(call: Any, previous_tool_results: list[ToolResult]) -> str | None:
    """校验当前写工具是否和上一轮读工具条件一致。

    当前只约束 create_game 与最近一次 search_current_games 的 requirement。
    如果发现关键槽位漂移，返回错误文本给主 loop，主 loop 会把错误作为工具结果回喂模型修正。
    """

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


def validate_explicit_task_fact_consistency(
    call: Any,
    context_payload: dict[str, Any] | None,
) -> tuple[str | None, dict[str, Any]]:
    """Reject a tool proposal that rewrites an explicit current-task fact.

    This guard is field-generic: it compares the canonical fact projection with
    whichever requirement fields the model actually proposed.  It does not
    infer user intent or invent missing fields, and therefore is not a
    sentence-specific business branch.
    """

    argument_name = EXPLICIT_FACT_TOOL_ARGUMENTS.get(str(getattr(call, "name", "")))
    if argument_name is None or not isinstance(getattr(call, "arguments", None), dict):
        return None, {}
    proposed = call.arguments.get(argument_name)
    if not isinstance(proposed, dict):
        return None, {}
    projected = dict((context_payload or {}).get("explicit_task_facts") or {})
    reference = projected.get("facts") if isinstance(projected.get("facts"), dict) else {}
    if not reference:
        return None, {}

    normalized_proposed = normalize_requirement(proposed)
    normalized_reference = normalize_requirement(reference)
    mismatches: list[str] = []
    for field in EXPLICIT_FACT_CONSISTENT_FIELDS:
        if field not in normalized_proposed or field not in normalized_reference:
            continue
        expected = normalized_requirement_field_value(field, normalized_reference.get(field))
        actual = normalized_requirement_field_value(field, normalized_proposed.get(field))
        if actual != expected:
            mismatches.append(f"{field}: expected {expected!r} from explicit task facts, got {actual!r}")
    if not mismatches:
        return None, normalized_reference
    return (
        f"tool argument consistency violation: {call.name}.{argument_name} conflicts with "
        "explicit current-task facts; " + "; ".join(mismatches),
        normalized_reference,
    )


def latest_read_requirement(previous_tool_results: list[ToolResult], *, tool_name: str) -> dict[str, Any] | None:
    """从工具结果中找到最近一次读工具使用的 requirement。

    同时兼容普通工具结果和之前一致性校验失败时返回的 reference_requirement，
    这样模型连续修正多轮时仍能看到原始参考条件。
    """

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
    """把 requirement 字段归一成可比较的值。"""

    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [normalized_requirement_value(item) for item in value]
    return value


def normalized_requirement_field_value(field: str, value: Any) -> Any:
    """Canonicalize aliases that the general requirement normalizer preserves."""

    if field == "smoke_preference":
        return normalize_smoke_preference(value)
    if field == "start_time" and isinstance(value, str):
        match = re.fullmatch(r"\s*(\d{1,2})[:：.](\d{2})\s*", value)
        if match:
            return f"{int(match.group(1)):02d}:{int(match.group(2)):02d}"
    if field in {"duration_hours", "known_player_count", "needed_seats"}:
        try:
            return float(value)
        except (TypeError, ValueError):
            pass
    return normalized_requirement_value(value)
