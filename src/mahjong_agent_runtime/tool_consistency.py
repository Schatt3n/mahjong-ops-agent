"""跨工具参数一致性校验。

设计理念：
- 模型可以根据工具结果动态调整计划，但不能在没有用户新输入的情况下偷偷改关键条件。
- 例如上一轮按 16:00/0.5/无烟搜索现有局，下一轮 create_game 不能变成人齐开/1块/有烟。
- 同一客户条件需要保留，但现成局目标桌况与另一个新局的占座快照不能机械复制。
- 这类校验是生产边界，不是补业务 if-else；人数事实仍由明确断言和 Store 参与者投影保护。
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from .models import ToolResult
from .domains import normalize_requirement
from .domains.game_participants import normalize_game_participants, seat_count_from_payload
from .domains.tools.shared import known_players_with_requesting_party
from .domains.value_utils import normalize_smoke_preference


# 这些条件在查现成局与新建局之间含义相同；人数是各自 Game 的快照，不能跨局复制。
CONSISTENT_REQUIREMENT_FIELDS = (
    "game_type",
    "stake",
    "stake_range",
    "cap_score",
    "smoke_preference",
    "start_time_kind",
    "start_time",
    "planned_start_at",
    "duration_kind",
    "duration_hours",
)

GAME_SEAT_SNAPSHOT_FIELDS = frozenset({"known_player_count", "needed_seats", "seat_format"})
READ_REQUIREMENT_TOOLS = frozenset({"search_current_games", "search_customers"})

EXPLICIT_FACT_TOOL_ARGUMENTS = {
    "search_current_games": "requirement",
    "search_customers": "requirement",
    "create_game": "requirement",
    "update_game_requirement": "requirement_patch",
}

EXPLICIT_FACT_CONSISTENT_FIELDS = (
    "game_type",
    "stake",
    "stake_range",
    "cap_score",
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

    当前只约束 create_game 与最近一次 search_current_games 中作用域相同的客户条件。
    查询的目标桌况不代表请求方人数；新局人数交给显式事实校验和 Store 参与者投影。
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
        if expected is None or expected == "":
            continue
        actual = normalized_requirement_value(current_requirement.get(field))
        equal = (_time_requirement_values_match(field, current_requirement, reference_requirement)
                 if field in {"start_time", "planned_start_at"} else actual == expected)
        if not equal:
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
    """按工具作用域校验主模型绑定的当前客户事实，并返回可审计的冲突原因。

    查询现成局的人数描述目标桌况；客户己方人数只与明确的请求方占座字段比较。
    创建局时同时核对声明人数和 Store 将使用的参与者投影，防止另一组参数覆盖事实。
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
    # 更新工具接收局部补丁；仅改时间等字段时，原有区间仍由 Store 保留。
    replaces_stake = argument_name != "requirement_patch" or any(
        field in proposed for field in ("stake_range", "stake", "base_stake", "base_score", "preferred_stake", "level")
    )
    for field in EXPLICIT_FACT_CONSISTENT_FIELDS:
        if call.name in READ_REQUIREMENT_TOOLS and field in GAME_SEAT_SNAPSHOT_FIELDS:
            continue
        if field == "stake_range" and replaces_stake and field in normalized_reference and field not in normalized_proposed:
            mismatches.append("stake_range: explicit accepted range must not be dropped or replaced by a fixed stake")
            continue
        if field == "stake" and field in normalized_reference and "stake_range" in normalized_proposed:
            mismatches.append("stake: explicit fixed stake must not be widened into a range")
            continue
        if field == "planned_start_at" and field in normalized_reference and "start_time" in normalized_proposed:
            # ISO start_time 自带日期；钟点不能以缺省 planned_start_at 的方式丢掉已确认日期。
            if not _time_requirement_values_match(field, normalized_proposed, normalized_reference):
                mismatches.append(f"{field}: expected {normalized_reference[field]!r} from explicit task facts, got {normalized_proposed.get(field)!r}")
            continue
        if field not in normalized_proposed or field not in normalized_reference:
            continue
        expected = normalized_requirement_field_value(field, normalized_reference.get(field))
        actual = normalized_requirement_field_value(field, normalized_proposed.get(field))
        equal = (_time_requirement_values_match(field, normalized_proposed, normalized_reference)
                 if field in {"start_time", "planned_start_at"} else actual == expected)
        if not equal:
            mismatches.append(f"{field}: expected {expected!r} from explicit task facts, got {actual!r}")
    mismatches.extend(_explicit_party_count_mismatches(call, normalized_proposed, normalized_reference))
    if not mismatches:
        return None, normalized_reference
    return (
        f"tool argument consistency violation: {call.name}.{argument_name} conflicts with "
        "explicit current-task facts; " + "; ".join(mismatches),
        normalized_reference,
    )


def _explicit_party_size(payload: dict[str, Any]) -> Any:
    """读取合同中明确填写的己方座位数；字段缺省时保留未知，不推断为一个人。"""

    party = payload.get("requesting_party")
    sources = [party, payload] if isinstance(party, dict) else [payload]
    for source in sources:
        for field in ("seat_count", "party_size"):
            if source.get(field) is not None and source[field] != "":
                return normalized_requirement_field_value("known_player_count", source[field])
    return None


def _explicit_party_count_mismatches(
    call: Any,
    proposed: dict[str, Any],
    reference: dict[str, Any],
) -> list[str]:
    """按查询请求方或创建参与者的真实作用域核对人数，不把目标桌况当己方人数。"""

    expected_party = _explicit_party_size(reference)
    expected_count = normalized_requirement_field_value("known_player_count", reference.get("known_player_count"))
    mismatches: list[str] = []
    if call.name == "search_current_games":
        actual = _explicit_party_size(proposed)
        expected = expected_party if expected_party is not None else expected_count
        if actual is not None and expected is not None and actual != expected:
            mismatches.append(f"requesting_party.seat_count: expected {expected!r} from explicit task facts, got {actual!r}")
        return mismatches
    if call.name != "create_game" or (expected_count is None and expected_party is None):
        return mismatches
    organizer_id = call.arguments.get("organizer_id")
    if not isinstance(organizer_id, str) or not organizer_id.strip():
        # 必填项和类型由 Gateway schema 报错；这里不以投影补齐身份或绕过授权。
        return mismatches
    raw_players = call.arguments.get("known_players")
    players = known_players_with_requesting_party(
        known_players=raw_players if isinstance(raw_players, list) else [],
        requesting_party=call.arguments.get("requesting_party"),
    )
    participants = normalize_game_participants(
        organizer_id=organizer_id,
        organizer_name=str(call.arguments.get("organizer_name") or organizer_id),
        known_players=players,
        default_requester_seat_count=seat_count_from_payload(proposed, default=1),
    )
    # 与 Store 使用同一个参与者投影；分开列出的同行者也属于新局实际已占座位。
    occupied = sum(item.seat_count for item in participants if item.status in {"joined", "confirmed"})
    if expected_count is not None and occupied != expected_count:
        mismatches.append(f"initial occupied seats: expected {expected_count!r} from explicit task facts, got {occupied!r}")
    requester = next((item for item in participants if item.customer_id == organizer_id), None)
    if expected_party is not None and requester is not None and requester.seat_count != expected_party:
        mismatches.append(
            f"requesting_party.seat_count: expected {expected_party!r} from explicit task facts, got {requester.seat_count!r}"
        )
    return mismatches


def latest_read_requirement(previous_tool_results: list[ToolResult], *, tool_name: str) -> dict[str, Any] | None:
    """从工具结果中找到最近一次读工具使用的 requirement。

    同时兼容普通工具结果和之前一致性校验失败时返回的 reference_requirement，
    这样模型连续修正多轮时仍能看到原始参考条件。
    普通失败查询不建立日期锚；错误反馈只能延续其携带的既有权威参考。
    """

    for result in reversed(previous_tool_results):
        if result.result.get("reference_tool_name") == tool_name and isinstance(result.result.get("reference_requirement"), dict):
            return result.result["reference_requirement"]
        if result.name != tool_name or not result.called or not result.allowed or result.error:
            continue
        requirement = result.result.get("requirement")
        if isinstance(requirement, dict):
            return requirement
    return None


def _aware_datetime(value: Any) -> datetime | None:
    """只解析带明确偏移的 ISO 瞬时，不借当前时钟或默认时区补日期。"""
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None and parsed.utcoffset() is not None else None


def _clock_parts(value: Any) -> tuple[int, int] | None:
    """读取稳定钟点协议；钟点不包含日期，也不能充当绝对时间证据。"""
    match = re.fullmatch(r"\s*(\d{1,2})[:：.](\d{2})\s*", value) if isinstance(value, str) else None
    if match and int(match[1]) < 24 and int(match[2]) < 60:
        return int(match[1]), int(match[2])
    return None


def _time_requirement_values_match(field: str, proposed: dict[str, Any], reference: dict[str, Any]) -> bool:
    """比较日期与钟点的类型语义，不把同一 HH:MM 的不同天或不同瞬时视为相等。

    reference 来自来源已绑定的当前断言或真实成功查询。clock/ISO 混用只使用其中
    已确认的 aware planned_start_at 锚；词法 candidates 和当前壁钟都不参与。
    绝对时间换成钟点时，proposed 还须保留同一绝对锚，防止读到日期却写丢日期。
    """
    expected, actual = reference.get(field), proposed.get(field)
    expected_at, actual_at = _aware_datetime(expected), _aware_datetime(actual)
    if field == "planned_start_at":
        actual_at = actual_at or (_aware_datetime(proposed.get("start_time")) if actual is None else None)
        if expected_at is not None and actual_at is not None:
            return expected_at.astimezone(timezone.utc) == actual_at.astimezone(timezone.utc)
        return normalized_requirement_value(actual) == normalized_requirement_value(expected)
    anchor = _aware_datetime(reference.get("planned_start_at"))
    proposed_anchor = _aware_datetime(proposed.get("planned_start_at"))
    if expected_at is not None and reference.get("planned_start_at") not in (None, "") and anchor is None:
        return False
    if actual_at is not None and proposed.get("planned_start_at") not in (None, "") and proposed_anchor is None:
        return False
    if expected_at is not None and anchor is not None and expected_at != anchor:
        return False
    if actual_at is not None and proposed_anchor is not None and actual_at != proposed_anchor:
        return False
    if expected_at is not None and actual_at is not None:
        return expected_at.astimezone(timezone.utc) == actual_at.astimezone(timezone.utc)
    if expected_at is not None or actual_at is not None:
        if anchor is None:
            return False
        if expected_at is not None:
            return (expected_at == anchor and proposed_anchor == anchor
                    and _clock_parts(actual) == (anchor.hour, anchor.minute))
        return _clock_parts(expected) == (anchor.hour, anchor.minute) and actual_at == anchor
    return normalized_requirement_field_value(field, actual) == normalized_requirement_field_value(field, expected)


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
    """对通用需求规范化器保留的别名执行规范化处理。"""

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
