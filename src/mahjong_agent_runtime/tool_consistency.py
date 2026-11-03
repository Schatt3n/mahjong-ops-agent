"""跨工具参数一致性校验。

设计理念：
- 模型可以根据工具结果动态调整计划，但不能在没有用户新输入的情况下偷偷改关键条件。
- 例如上一轮按 16:00/0.5/无烟搜索现有局，下一轮 create_game 不能变成人齐开/1块/有烟。
- 同一客户条件需要保留，但现成局目标桌况与另一个新局的占座快照不能机械复制。
- 这类校验是生产边界，不是补业务 if-else；人数事实仍由明确断言和 Store 参与者投影保护。
"""

from __future__ import annotations

import re
import json
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from .models import ToolResult
from .domains import normalize_requirement
from .domains.game_participants import normalize_game_participants, seat_count_from_payload
from .domains.tools.shared import known_players_with_requesting_party
from .domains.value_utils import normalize_smoke_preference
from .domains.stake_values import stake_options


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
    # 输入规范化失败也属于拒绝结果，不能在进入 Gateway 前把主循环异常终止。
    # 这里不放宽领域不变量；原始读参考仍由调用方完整回灌供模型重新核实。
    try:
        current_requirement = normalize_requirement(current_requirement)
    except ValueError as exc:
        return f"tool argument consistency violation: invalid proposed create_game.requirement; {exc}"
    try:
        reference_requirement = normalize_requirement(reference_requirement)
    except ValueError as exc:
        return f"tool argument consistency violation: invalid previous search_current_games requirement reference; {exc}"
    mismatches: list[str] = []
    for field, value in _requirement_fields_to_preserve(reference_requirement).items():
        if field in {"stake", "stake_range"} and _stake_representations_equivalent(current_requirement, reference_requirement):
            continue
        expected = normalized_requirement_value(value)
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
    binding_error, reference = _fact_reference_for_call(call, projected)
    if binding_error:
        return "tool argument consistency violation: " + binding_error, reference
    if not reference:
        return None, {}

    # 当前断言经过身份/来源筛选仍可能互相冲突，例如同时绑定固定档位与接受区间。
    # 保留完整原始参考并拒绝本步，让模型按原话重核角色/来源；不能猜一项优先后执行。
    try:
        normalized_reference = normalize_requirement(reference)
    except ValueError as exc:
        return ("tool argument consistency violation: invalid explicit current-task fact reference; " + str(exc),
                deepcopy(reference))
    try:
        normalized_proposed = normalize_requirement(proposed)
    except ValueError as exc:
        return (f"tool argument consistency violation: invalid proposed {call.name}.{argument_name}; {exc}",
                normalized_reference)
    mismatches: list[str] = []
    # 更新工具接收局部补丁；仅改时间等字段时，原有区间仍由 Store 保留。
    replaces_stake = argument_name != "requirement_patch" or any(
        field in proposed for field in ("stake_range", "stake", "base_stake", "base_score", "preferred_stake", "level")
    )
    for field in EXPLICIT_FACT_CONSISTENT_FIELDS:
        if call.name in READ_REQUIREMENT_TOOLS and field in GAME_SEAT_SNAPSHOT_FIELDS:
            continue
        if field in {"stake", "stake_range"} and _stake_representations_equivalent(normalized_proposed, normalized_reference):
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


def _stake_representations_equivalent(proposed: dict[str, Any], reference: dict[str, Any]) -> bool:
    """固定底注与等值闭区间可互换，但金额集合及封顶必须完全相同。

    复用领域数值协议，不把包含或相交当等价，也不改写模型参数或权威事实。
    未知档位不能凭两个空集合相等获得通过；其他条件仍由各自字段校验。
    """
    # 规范化固定值可能同时保留stake与base_stake，表示重复不能影响金额集合。
    proposed_options = set(stake_options(proposed))
    reference_options = set(stake_options(reference))
    return bool(proposed_options) and proposed_options == reference_options


def _fact_reference_for_call(call: Any, projected: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    """按声明的调用/牌局范围选择来源已核验事实，不按拟执行参数反推或挑选真值。

    全局事实约束每个调用，局事实只约束同game_id，调用事实只约束当前call_id。
    同范围冲突必须反馈模型重新绑定；不能用后写值覆盖、把候选集合当任选值，
    或让更窄范围静默覆盖全局条件。这里不解析任何开放式自然语言。
    """
    reference = deepcopy(projected.get("facts")) if isinstance(projected.get("facts"), dict) else {}
    call_ids = projected.get("binding_call_ids", [])
    for assertion in projected.get("bound_assertions", []):
        scope = assertion.get("scope", {})
        if (not isinstance(scope, dict) or set(scope) - {"call_id", "game_id"}
                or any(not isinstance(v, str) or not v.strip() for v in scope.values())):
            return "invalid fact scope: use a non-empty scope.call_id or scope.game_id", reference
        if "call_id" in scope and scope["call_id"] not in call_ids:
            return "invalid fact scope: scope.call_id must reference a tool call in this action", reference
        if not _fact_scope_applies_to_call(scope, call):
            continue
        name = str(assertion.get("field"))
        value = assertion.get("value")
        # 精确保留JSON类型；冲突不因同源、先后顺序或与调用参数恰好相同而消失。
        if name in reference and json.dumps(reference[name], sort_keys=True, ensure_ascii=False) != json.dumps(value, sort_keys=True, ensure_ascii=False):
            return (f"conflicting current facts for {name} in the same tool scope; "
                    "bind different plans with scope.call_id/scope.game_id instead of global duplicate fields", reference)
        reference[name] = deepcopy(value)
    return None, reference


def _fact_scope_applies_to_call(scope: dict[str, str], call: Any) -> bool:
    """事实校验和来源反馈共用范围选择，不能把另一方案来源错绑到当前调用。"""
    return (("call_id" not in scope or scope["call_id"] == call.call_id)
            and ("game_id" not in scope or scope["game_id"] == call.arguments.get("game_id")))


def explicit_fact_sources_for_call(call: Any, context_payload: dict[str, Any] | None) -> dict[str, Any]:
    """保留当前调用事实的已核验来源；同字段多来源全部反馈，不按后写覆盖。"""
    projected = (context_payload or {}).get("explicit_task_facts", {})
    entries: dict[str, list[dict[str, Any]]] = {}
    for assertion in projected.get("bound_assertions", []):
        scope = assertion.get("scope", {})
        if not isinstance(scope, dict) or not _fact_scope_applies_to_call(scope, call):
            continue
        source = {"source_message_id": assertion["source_message_id"], "role": "current"}
        if scope:
            source["scope"] = deepcopy(scope)
        values = entries.setdefault(str(assertion.get("field")), [])
        if source not in values:
            values.append(source)
    return {field: values[0] if len(values) == 1 else {"assertions": values}
            for field, values in entries.items()}


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


def _latest_read_requirement_reference(previous_tool_results: list[ToolResult], *, tool_name: str) -> dict[str, Any] | None:
    """统一选择守卫与上下文使用的真实读参数或运行时拒绝反馈，不读取模型计划。

    反馈携带的reference_requirement沿用既有参考，失败重查不能清除它；新的成功读
    可以替换旧参考。反馈自己的call_id属于被拒绝的写提案，不能冒充原查询身份。
    """
    for index in range(len(previous_tool_results) - 1, -1, -1):
        result = previous_tool_results[index]
        if result.result.get("reference_tool_name") == tool_name and isinstance(result.result.get("reference_requirement"), dict):
            return {"requirement": result.result["reference_requirement"], "source_tool": tool_name,
                    "source_call_id": None, "source_evidence_index": index, "source_kind": "runtime_reference_feedback"}
        if result.name != tool_name or not result.called or not result.allowed or result.error:
            continue
        requirement = result.result.get("requirement")
        if isinstance(requirement, dict):
            return {"requirement": requirement, "source_tool": tool_name,
                    "source_call_id": result.call_id if isinstance(result.call_id, str) and result.call_id else None,
                    "source_evidence_index": index, "source_kind": "successful_read"}
    return None


def _requirement_fields_to_preserve(normalized_requirement: dict[str, Any]) -> dict[str, Any]:
    """从已规范化参数选择守卫实际保护的同作用域字段，深复制以隔离投影修改。"""
    return {field: deepcopy(normalized_requirement[field]) for field in CONSISTENT_REQUIREMENT_FIELDS
            if normalized_requirement_value(normalized_requirement.get(field)) not in (None, "")}


def executed_requirement_contract(previous_tool_results: list[ToolResult]) -> dict[str, Any] | None:
    """显式投影本轮已执行读参数，防止模型在各工具阶段无依据重解释同一条件。

    此投影与确定性守卫共用参考选择及保护字段，只证明执行过哪些参数，不证明
    原语义必然正确或产生写权限。新消息的运行不传旧证据；有据修订需先重新成功查。
    """
    reference = _latest_read_requirement_reference(previous_tool_results, tool_name="search_current_games")
    if reference is None or not reference["requirement"]:
        return None
    preserve = _requirement_fields_to_preserve(normalize_requirement(reference["requirement"]))
    if not preserve:
        return None
    return {
        "authority": "executed_parameters_not_user_facts",
        "scope": "current_run_only",
        "source_tool": reference["source_tool"],
        "source_call_id": reference["source_call_id"],
        "source_evidence_index": reference["source_evidence_index"],
        "source_kind": reference["source_kind"],
        "target_tool": "create_game",
        "target_argument_path": "arguments.requirement",
        "preserve_requirement": preserve,
        "revision_policy": (
            "默认复用preserve_requirement中的实际已查条件；模型计划、known_facts或fact_assertions的重写不能自行替换。"
            "若用户新输入或来源重核支持修订，在plan_revision_reason说明依据，先按修订条件重新成功查询，再据新结果写入。"
            "失败查询不替换有效参考；新运行以新用户输入为准，不从历史运行继承此投影。"
        ),
        "boundary": (
            "这里只记录真实执行参数，不把模型先前理解升级为不可修订的用户事实，也不授予建局或外发权限。"
            "目标桌况人数、参与者身份及查询容量不在本投影中，不能机械复制成新局占座。"
        ),
    }


def latest_read_requirement(previous_tool_results: list[ToolResult], *, tool_name: str) -> dict[str, Any] | None:
    """从工具结果中找到最近一次读工具使用的 requirement。

    同时兼容普通工具结果和之前一致性校验失败时返回的 reference_requirement，
    这样模型连续修正多轮时仍能看到原始参考条件。
    普通失败查询不建立日期锚；错误反馈只能延续其携带的既有权威参考。
    """

    reference = _latest_read_requirement_reference(previous_tool_results, tool_name=tool_name)
    return reference["requirement"] if reference is not None else None


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
