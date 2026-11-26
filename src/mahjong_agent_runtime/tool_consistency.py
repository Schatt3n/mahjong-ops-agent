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
from dataclasses import replace
from datetime import datetime, time, timezone
from typing import Any

from .models import ToolCall, ToolResult, now
from .domains import normalize_requirement
from .domains.game_domain import (
    START_KIND_ASAP_WHEN_FULL, current_game_query_end_deadline, current_game_query_start_range, derive_game_lifecycle,
)
from .domains.game_participants import normalize_game_participants, seat_count_from_payload
from .domains.tools.shared import current_game_search_reply_contract, known_players_with_requesting_party
from .domains.value_utils import normalize_smoke_preference
from .domains.stake_values import parse_number, stake_options


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
    "max_duration_hours",
)

GAME_SEAT_SNAPSHOT_FIELDS = frozenset({"known_player_count", "needed_seats", "seat_format"})
READ_REQUIREMENT_TOOLS = frozenset({"search_current_games", "search_customers"})
_REQUESTING_PARTY_SEAT_FACT_FIELDS = frozenset({"seat_count", "party_size", "requesting_party.seat_count"})
EXPLICIT_PARTY_FACT_FIELDS = _REQUESTING_PARTY_SEAT_FACT_FIELDS | {"known_player_count"}

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
    "max_duration_hours",
    "known_player_count",
    "needed_seats",
    "seat_format",
)


def validate_table_availability(call: Any, evidence: dict[str, Any] | None) -> str | None:
    """核对独立确认的到店、档位和时长，区分查现成桌与承诺新局开场。"""
    if call.name not in {"search_current_games", "create_game"} or not isinstance(call.arguments, dict):
        return None
    requirement = call.arguments.get("requirement")
    proposed_end = isinstance(requirement, dict) and "must_end_by" in requirement
    if proposed_end and call.name == "create_game":
        return "must_end_by 只用于核对现成桌结束时间，不能写入新局条件。"
    if evidence is None:
        if proposed_end:
            return "缺少本次顾客硬离店时间的核实依据，不能按 must_end_by 过滤。"
        return "这条来话中的钟点用途尚未核实，先查现成桌或澄清开场，暂不能新建。" if call.name == "create_game" else None
    scope = evidence.get("lookup_scope")
    if call.name == "create_game" and evidence.get("historical_table_opening_not_confirmed"):
        return (
            "本次未建局。顾客在问以前提过的桌，提到的钟点没有确认一个新开场。"
            "先据实回答原桌是否已安排。如果顾客要继续组局，只确认这次的日期和开场时间，不要重提同一建局。"
        )
    if call.name == "search_current_games":
        requested_scope = call.arguments.get("lookup_scope")
        if evidence.get("also_requests_personal_options") is True:
            if requested_scope not in ("specific_table", "personal"):
                return "同一来话既问原桌也要另找。分别调用并填写 lookup_scope：原桌用 specific_table，另找用 personal。"
            scope = requested_scope
        elif scope in ("specific_table", "personal") and requested_scope is not None and requested_scope != scope:
            return f"查局用途与核实的原话不一致。本次应使用 lookup_scope={scope}，保留原来的问题范围。"
    if scope != "personal":
        if proposed_end:
            return "点名查原桌或盘点应保留该桌的真实条件，请移除个人条件 must_end_by 后查询。"
        if (call.name == "search_current_games" and scope == "specific_table"
                and isinstance(requirement, dict) and "start_time_range" in requirement):
            return "原桌查找不按开场区间滤掉未定开场的桌。已有明确开场才用该桌的固定钟点；否则按原桌其他特征查询。"
        if (call.name == "search_current_games" and scope == "specific_table"
                and _uses_personal_clock_to_locate_table(requirement, evidence)):
            return "定位顾客问的原桌时，不能把本人到店、离店或等候的钟点当作该桌开场。请按原桌特征查余位。"
        if call.name == "create_game" and evidence.get("named_future_opening"):
            return "请先核实顾客提到的未来开场时段，再创建新局。不能直接按现在人齐开登记。"
        if call.name == "create_game" and evidence.get("clock_roles"):
            return "本次来话未核实为本人组织新局及开场的条件，不能用到店或离店的时间直接登记。"
        return None
    try:
        proposed = normalize_requirement(requirement)
        requested_end = current_game_query_end_deadline(proposed)
    except (TypeError, ValueError, OverflowError):
        return "拟提交的查局或建局条件无效。"
    stakes = evidence.get("accepted_stake_range")
    if stakes and not _stake_representations_equivalent(
        proposed, {"stake_range": stakes, "cap_score": evidence.get("known_cap_score")},
    ):
        return "请使用核实过的可接受档位区间。封顶是另一件事，不能由区间端点推断。"
    duration_cap = evidence.get("max_duration_hours")
    proposed_cap = parse_number(proposed.get("max_duration_hours"))
    supported_window = evidence.get("opening_window")
    limit = current_game_query_start_range({"start_time_range": supported_window}) if supported_window else None
    arrival_floor = _aware_datetime(evidence.get("arrival_lookup_floor_at"))
    play_until = _aware_datetime(evidence.get("play_until_at"))
    hard_departure = _aware_datetime(evidence.get("must_end_by"))
    exact_duration = proposed.get("duration_hours", proposed.get("duration"))
    agreed_duration = evidence.get("agreed_duration_hours")
    fixed_opening = limit[0] if limit and limit[0] == limit[1] and evidence.get("has_fixed_opening_authority") else None
    departure_cap = ((hard_departure - fixed_opening).total_seconds() / 3600
                     if hard_departure and fixed_opening and fixed_opening < hard_departure else None)
    paired_creation = call.name == "create_game" and departure_cap is not None
    if duration_cap is not None and not paired_creation and proposed_cap != duration_cap:
        return "请在 max_duration_hours 保留顾客明确的最多可打时长。上限不等于同桌约定的固定时长。"
    proposed_hours = parse_number(exact_duration)
    unsupported_duration = exact_duration is not None and (proposed_hours is None or proposed_hours != agreed_duration)
    if evidence.get("approximate_play_duration_only") and (unsupported_duration or proposed_cap is not None and not paired_creation):
        return "顾客说的大致打牌时长只是偏好，请保留其他条件重试，省略 duration_hours 和 max_duration_hours。"
    if arrival_floor is not None and limit and limit[0].astimezone(arrival_floor.tzinfo).date() != arrival_floor.date():
        arrival_floor = None  # 本轮的到店钟点不能剪掉另一个明确日期的开场时段。
    earliest = max(limit[0], arrival_floor) if limit and arrival_floor else limit[0] if limit else arrival_floor
    if call.name == "search_current_games":
        if hard_departure and earliest and earliest >= hard_departure:
            return "已核实的最早开场不早于离店时间，先澄清可参加的时段。"
        if requested_end != hard_departure:
            if hard_departure:
                return (f"现成局请带上已核实的 must_end_by={hard_departure.isoformat()}，按每桌实际约定时段匹配。"
                        "离店钟点不可改写为统一的小时上限。")
            return "本次未核实为硬离店，不能把希望或大约打到的时点写入 must_end_by。"
        if play_until and (unsupported_duration or (proposed_cap is not None and duration_cap is None)):
            return "可打到的钟点不是固定时长或小时上限。请保留已核实的时间条件，省略未明确约定的小时数。"
        if "nearby_opening_search" in evidence:
            return _validate_approximate_opening_search(call, proposed, evidence)
        if earliest is None:
            arrival_by = _aware_datetime(evidence.get("arrival_by_at"))
            if arrival_by and _uses_arrival_by_as_opening_filter(proposed, arrival_by):
                return (f"已核实顾客最迟 {arrival_by.isoformat()} 到店。请保留其他条件，移除本次未核实的开场过滤后重试。"
                        "找到较早的桌可再确认顾客能否赶上。")
            if evidence.get("clock_role") == "arrival_estimate" and _uses_point_opening_filter(proposed):
                return ("当前仅核实为顾客的到店估计，尚不能确定查局的固定开场。请保留其他筛选，移除单点开场后查现成桌。"
                        "到店和实际能否赶上仍按查到的桌逐个核对。")
            return None
        if (fixed_opening == earliest
                and "start_time_range" not in proposed and _aware_datetime(proposed.get("planned_start_at")) == earliest):
            return None
        try:
            query_window = current_game_query_start_range(proposed)
        except ValueError:
            query_window = None
        if (query_window is None or query_window[0] != earliest or proposed.get("start_time_kind")
                or (limit and query_window[1] != limit[1])
                or (not limit and (query_window[1] <= earliest
                                   or query_window[1].astimezone(earliest.tzinfo).date() != earliest.date()))):
            end = limit[1] if limit else datetime.combine(earliest.date(), time.max, earliest.tzinfo)
            nearby = "，include_nearby=false" if arrival_floor and evidence.get("clock_role") == "arrival_not_before" else ""
            return (
                "这次还没有查到库存结果。请保留其他查局条件，按核实的时段用 start_time_range 重试。"
                f"start_at={earliest.isoformat()}，end_at={end.isoformat()}{nearby}。"
                "不要另填固定开场或人齐开的开场方式。"
            )
        if arrival_floor and evidence.get("clock_role") == "arrival_not_before" and call.arguments.get("include_nearby") is not False:
            return "顾客给了最早到店的硬界限，请用 include_nearby=false 避免将无法赶上的早场作为备选。"
        return None

    # 能赶上某桌只说明可以考虑参加，不能把到店时刻写成店家和顾客已经约好的开场。
    if "nearby_opening_search" in evidence:
        return "顾客给了要查的近似开场，尚未确认如何开新局。先反馈实查选项或确认开场安排。"
    stamp = now()
    try:
        lifecycle = derive_game_lifecycle(proposed, created_at=stamp)
    except (TypeError, ValueError, OverflowError):
        return "拟建局的开场时间无效。"
    planned, ends_at = lifecycle["planned_start_at"], lifecycle["planned_end_at"]
    if planned is not None:
        if not evidence.get("has_fixed_opening_authority"):
            if planned == arrival_floor:
                arrival = "最早可到店" if evidence.get("clock_role") == "arrival_not_before" else "可到店"
                soft_end = (f"保留顾客想打到 {play_until.isoformat()} 的柔性偏好，不当成硬截止或固定时长。"
                            if play_until and hard_departure is None else "")
                return (
                    f"本次未建局。{planned.isoformat()} 只核实为本人{arrival}的时间，还没选定固定开场。"
                    "先反馈现成桌的实查结果。顾客已经明确委托组局时，只问这个时间是否也是本桌开场，"
                    "用 waiting_user 等本人答复，不要重新征询组局委托。仅询局时按已核实的查局结果回复。"
                    f"不要再次尝试同一建局。{soft_end}"
                )
            return (
                "本次未建局。顾客还没选定或委托你定一个固定开场。已有明确组局委托时保留委托，"
                "先反馈实际查到的现成桌或房态，再单独请顾客确认具体开场。"
                "可行的候选开场仍是方案，不能报成已登记席位、已有缺口或保证开打。不要再次尝试同一建局。"
            )
        if (earliest and planned < earliest) or (limit and planned > limit[1]):
            return "拟建局的固定开场超出了核实过的可开打时段。"
    elif earliest and stamp < earliest:
        return "这个人齐开没有能落实的最早开场，可能在顾客可开打时段之前成桌。"
    if hard_departure and planned is None:
        # 人齐开的有效期约束招募，不能充当这一桌的散场时间。
        return (
            f"本次没有建局。顾客最晚只能打到 {hard_departure.isoformat()}，人齐开无法落实这个散场界限。"
            "保留组局委托。可以按已核实的 must_end_by 查能按时结束的现成桌，或先问顾客能接受的具体开场。"
        )
    if play_until:
        if evidence.get("prior_play_until") and (requested_end is not None or proposed_cap is not None):
            return "前文可打到的钟点仍是柔性条件。它不能充当硬离店或相减算出的小时上限。"
        if paired_creation and planned == fixed_opening:
            # 两端是已确认开场与本人离店。相减只限定局能安排多久，不能冒充同桌已约定的固定时长。
            available_hours = min(departure_cap, duration_cap) if duration_cap is not None else departure_cap
            if agreed_duration is not None and agreed_duration > available_hours:
                return "已约定的时长超过顾客可参加的时间，先核实这一处冲突。"
            if agreed_duration is not None and proposed_hours != agreed_duration:
                return f"开场已核实。请保留该开场并用已约定的 duration_hours={agreed_duration} 重试建局。"
            if (agreed_duration is None and (exact_duration is not None or proposed_cap != available_hours)
                    or duration_cap is not None and proposed_cap != available_hours):
                return (
                    f"开场已核实为 {fixed_opening.isoformat()}，请保留该开场，使用 "
                    f"max_duration_hours={available_hours} 作为可安排上限后重试建局。"
                    "未约定实际时长时省略 duration_hours，已确认的开场和离店时间无需再问。"
                )
        if unsupported_duration:
            return "不能把到店和离店时间相减当作顾客已约定的固定时长。"
    if hard_departure and ends_at and ends_at > hard_departure:
        return f"拟建局的安排超出顾客离店时间 {hard_departure.isoformat()}，应先核实实际开场或时长。"
    return None


def _approximate_opening_bounds(evidence: dict) -> tuple[datetime, datetime, datetime]:
    timing = evidence["nearby_opening_search"]
    center, start, end = (datetime.fromisoformat(timing[key]) for key in ("center_at", "period_start_at", "period_end_at"))
    if evidence.get("clock_role") == "arrival_not_before" and (floor := _aware_datetime(evidence.get("arrival_lookup_floor_at"))):
        start = max(start, floor)
    return center, start, end


def _validate_approximate_opening_search(call: ToolCall, proposed: dict, evidence: dict) -> str | None:
    """查附近可以提实际筛选范围。该范围不替顾客选定开场或承诺接受。"""
    try:
        window = current_game_query_start_range(proposed)
    except ValueError:
        return "拟查的开场范围无效。"
    has_fixed_filter = any(proposed.get(key) for key in ("start_time_kind", "start_time", "planned_start_at"))
    if evidence["nearby_opening_search"] is None:
        return "这个近似钟点尚不能核实。保留其他条件，移除猜测的开场过滤。" if window or has_fixed_filter else None
    center, start, end = _approximate_opening_bounds(evidence)
    if center < start:
        return "顾客给出的近似开场早于本人能到的时间。先确认能赶上的开场范围。"
    if (window is None or has_fixed_filter or not start <= window[0] <= center <= window[1] <= end
            or window[0] == window[1] and call.arguments.get("include_nearby") is False):
        return (f"顾客要查 {center.isoformat()} 前后的现成桌。请用包含该中心的 start_time_range，"
                f"范围保持在 {start.isoformat()} 至 {end.isoformat()} 内，不填固定开场。"
                "用中心单点检索时保留临近备选。")
    return None


def apply_approximate_opening_search_evidence(result: ToolResult, evidence: dict | None) -> ToolResult:
    """保留实际 SQL 区间与其他匹配条件，单列每张非中心场次还需确认时间。"""
    if (result.name != "search_current_games" or not result.called or not result.allowed or result.error
            or not evidence or not evidence.get("nearby_opening_search")
            or result.result.get("lookup_scope") == "specific_table"):
        return result
    center, start, end = _approximate_opening_bounds(evidence)
    projected = dict(result.result)
    for key in ("matches", "alternatives", "nearby_options"):
        if key not in projected:
            continue
        options = []
        for option in projected[key]:
            opening = _aware_datetime(option["game"].get("planned_start_at"))
            if opening is None or not start <= opening <= end:
                continue
            options.append({**option, "requires_timing_confirmation": True} if opening != center else option)
        projected[key] = options
    if isinstance(contract := projected.get("customer_reply_contract"), dict):
        projected["customer_reply_contract"] = current_game_search_reply_contract(
            projected["requirement"], projected["matches"], projected.get("alternatives"),
            unfiltered_available_pool_empty=contract["evidence_scope"]["unfiltered_available_pool_empty"],
        )
    return replace(result, result=projected)


def superseded_read_duration_evidence(
    call: ToolCall, previous_tool_results: list[ToolResult], verified: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """记录同次来话内已核实的开场与硬离店如何替换旧个人查局的未约定时长。"""
    if (call.name != "create_game" or not verified or verified.get("lookup_scope") != "personal"
            or verified.get("has_fixed_opening_authority") is not True
            or verified.get("agreed_duration_hours") is not None or verified.get("max_duration_hours") is not None):
        return None
    current_ref = verified.get("current_ref")
    roles = {item["role"] for item in verified.get("clock_roles") or ()}
    if (type(current_ref) is not int or verified.get("opening_customer_ref") != current_ref
            or current_ref not in (verified.get("supporting_customer_refs") or ())
            or not {"opening", "departure_deadline"} <= roles):
        return None
    window = verified.get("opening_window") or {}
    opening, departure = _aware_datetime(window.get("start_at")), _aware_datetime(verified.get("must_end_by"))
    if (opening is None or departure is None or opening >= departure
            or opening != _aware_datetime(window.get("end_at")) or departure != _aware_datetime(verified.get("play_until_at"))):
        return None
    proposed = call.arguments.get("requirement")
    if (not isinstance(proposed, dict) or _aware_datetime(proposed.get("planned_start_at")) != opening
            or validate_table_availability(call, verified) is not None):
        return None
    reference = _latest_read_requirement_reference(previous_tool_results, tool_name="search_current_games")
    receipt = _latest_read_requirement_reference(
        previous_tool_results, tool_name="search_current_games", successful_only=True,
    )
    if reference is None or receipt is None or reference["requirement"] != receipt["requirement"]:
        return None
    read = previous_tool_results[receipt["source_evidence_index"]]
    old = receipt["requirement"]
    hours = parse_number(old.get("duration_hours"))
    # 若这次真实查局已核实同一硬离店，现成局守卫必要求 must_end_by；保留已查过的库存范围。
    if (read.result.get("lookup_scope") != "personal" or read.deduplicated or not read.idempotency_key
            or "must_end_by" in old or _aware_datetime(old.get("planned_start_at")) != opening
            or hours is None or not 0 < hours < (departure - opening).total_seconds() / 3600):
        return None
    return {
        "field": "duration_hours",
        "previous_duration_hours": old["duration_hours"],
        "source_tool": read.name,
        "source_call_id": read.call_id,
        "source_evidence_index": receipt["source_evidence_index"],
        "opening_customer_ref": current_ref,
        "supporting_customer_refs": verified["supporting_customer_refs"],
        "planned_start_at": opening.isoformat(),
        "must_end_by": departure.isoformat(),
        "max_duration_hours": proposed["max_duration_hours"],
    }


def validate_tool_call_consistency(
    call: Any, previous_tool_results: list[ToolResult], *, read_duration_superseded: bool = False,
) -> str | None:
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
        if field == "duration_hours" and read_duration_superseded:
            continue
        if field in {"stake", "stake_range"} and _stake_representations_equivalent(current_requirement, reference_requirement):
            continue
        expected = normalized_requirement_value(value)
        actual = normalized_requirement_value(current_requirement.get(field))
        equal = (_time_requirement_values_match(field, current_requirement, reference_requirement)
                 if field in {"start_time", "planned_start_at"} else actual == expected)
        if not equal:
            mismatches.append(f"{field}: expected {expected!r} from previous search_current_games, got {actual!r}")
    opening_window = current_game_query_start_range(reference_requirement)
    if opening_window is not None:
        # 检索窗口不替客户选时刻。人齐开没有下限，可能在已查的未来窗口之前成桌，不能把有效期当成开场下限。
        stamp = now()
        try:
            lifecycle = derive_game_lifecycle(current_requirement, created_at=stamp)
        except (ValueError, OverflowError) as exc:
            return f"tool argument consistency violation: invalid proposed create_game lifecycle; {exc}"
        flexible = current_requirement.get("start_time_kind") == START_KIND_ASAP_WHEN_FULL or lifecycle["planned_start_at"] is None
        if flexible and lifecycle["expires_at"] <= opening_window[0]:
            mismatches.append(f"flexible game expires at {lifecycle['expires_at'].isoformat()} before it can reach "
                              f"the searched opening window starting at {opening_window[0].isoformat()}")
        elif flexible and stamp < opening_window[0]:
            mismatches.append(f"flexible game can start before the searched opening window starting at "
                              f"{opening_window[0].isoformat()}; it has no enforceable earliest start")
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
    if call.name == "create_game":
        party_error = _bind_group_creation_party(call, context_payload or {}, reference)
        if party_error:
            return "tool argument consistency violation: " + party_error, reference
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
        if (field == "cap_score" and call.name == "create_game" and field not in normalized_proposed
                and parse_number(normalized_reference.get(field)) is not None):
            mismatches.append("cap_score: explicit current-task cap must be retained when creating a game")
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


def validate_model_create_game_stake(call: ToolCall) -> str | None:
    """主模型发起招募需要有效档位；受信群自招入口仍可登记档位待确认的局。"""

    if call.name != "create_game" or not isinstance(call.arguments, dict):
        return None
    requirement = call.arguments.get("requirement")
    if not isinstance(requirement, dict):
        return None  # 参数缺失和类型错误仍由工具 schema 处理。
    try:
        normalized = normalize_requirement(requirement)
    except ValueError as exc:
        return f"tool argument consistency violation: invalid proposed create_game.requirement; {exc}"
    if stake_options(normalized):
        return None
    return (
        "tool argument consistency violation: create_game requires a confirmed stake, set of stakes, or stake range. "
        "If the stake is unknown, retain the delegation and ask for it without creating a game."
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
    source_mismatches = []
    for assertion in projected.get("bound_assertions", []):
        scope = assertion.get("scope", {})
        if (not isinstance(scope, dict) or set(scope) - {"call_id", "game_id"}
                or any(not isinstance(v, str) or not v.strip() for v in scope.values())):
            return "invalid fact scope: use a non-empty scope.call_id or scope.game_id", reference
        if "call_id" in scope and scope["call_id"] not in call_ids:
            return "invalid fact scope: scope.call_id must reference a tool call in this action", reference
        if not _fact_scope_applies_to_call(scope, call):
            continue
        name = _canonical_explicit_fact_field(assertion.get("field"))
        value = assertion.get("value")
        if name == "cap_score" and len(cap_candidates := assertion.get("source_cap_candidates", [])) == 1:
            source_cap = cap_candidates[0]
            if parse_number(value) != source_cap:
                source_mismatches.append(f"cap_score: source {assertion['source_message_id']!r} states {source_cap:g}, got {value!r}")
                # 只在拒绝反馈中附上来源数值。模型必须重提有据的动作，不能静默替换执行参数。
                value = source_cap
        # 精确保留JSON类型；冲突不因同源、先后顺序或与调用参数恰好相同而消失。
        if name in reference and json.dumps(reference[name], sort_keys=True, ensure_ascii=False) != json.dumps(value, sort_keys=True, ensure_ascii=False):
            return (f"conflicting current facts for {name} in the same tool scope; "
                    "bind different plans with scope.call_id/scope.game_id instead of global duplicate fields", reference)
        reference[name] = deepcopy(value)
    return "; ".join(source_mismatches) or None, reference


def _canonical_explicit_fact_field(field: Any) -> str:
    """己方座位断言的等价写法共用一个键，仍与桌况总人数分开核验。"""
    name = str(field)
    return "seat_count" if name in _REQUESTING_PARTY_SEAT_FACT_FIELDS else name


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
        values = entries.setdefault(_canonical_explicit_fact_field(assertion.get("field")), [])
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


def _bind_group_creation_party(call: Any, context: dict[str, Any], reference: dict[str, Any]) -> str | None:
    """群委托写入核对本人已确认人数。普通私聊续接可用同一任务的本人断言补齐。"""
    topic = context.get("group_topic_context") or {}
    window = context.get("task_context_window") or {}
    delegated = topic.get("intent") in {"delegate", "create"}
    if not delegated and window.get("reset_reason") != "new_group_topic_task":
        return None
    missing_party = (
        "create_game requires confirmed party size from the current customer goal. "
        "A sender identity or an empty lookup does not establish their attendance or party size. "
        "If the customer has stated it, bind seat_count as a current fact with the matching source_message_id. "
        "Otherwise keep the delegate goal and ask for the missing party size without writing a game or a seat."
    )
    known_count = normalized_requirement_field_value("known_player_count", reference.get("known_player_count"))
    if not delegated:
        return None if _explicit_party_size(reference) is not None or known_count is not None else missing_party

    # 群入口为本人消息提取的己方人数独立于查询参数和主模型计划。使用同目标最新有值的来源，
    # 因为后续只改时间等条件不撤销原先已知人数。主模型仍负责判断原话的语义角色。
    own_sources = topic.get("own_condition_sources") or []
    for source in reversed(own_sources):
        party_size = (source.get("constraints") or {}).get("party_size")
        if type(party_size) is not int or not 1 <= party_size <= 4:
            continue
        if known_count is not None and known_count != party_size:
            return f"known_player_count: expected {party_size!r} from the current group goal, got {known_count!r}"
        reference["known_player_count"] = party_size
        return None

    # 群路由可能漏掉明确人数。本人原话的词法候选须与主模型绑定的当前角色、来源和本次调用一致。
    # 这是来源交叉校验，查询结果、工具默认值和单独的模型断言都不能补出人数。
    source_ids = {source.get("source_message_id") for source in own_sources}
    supported = set()
    for assertion in (context.get("explicit_task_facts") or {}).get("bound_assertions", []):
        if (assertion.get("field") not in EXPLICIT_PARTY_FACT_FIELDS or assertion.get("source_message_id") not in source_ids
                or not _fact_scope_applies_to_call(assertion.get("scope", {}), call)):
            continue
        candidates = assertion.get("source_own_party_candidates", [])
        count = normalized_requirement_field_value("known_player_count", assertion.get("value"))
        if count in candidates:
            supported.add(count)
    if len(supported) == 1 and (known_count is None or known_count in supported):
        reference["known_player_count"] = int(supported.pop())
        return None
    return missing_party


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


def _latest_read_requirement_reference(
    previous_tool_results: list[ToolResult], *, tool_name: str, successful_only: bool = False,
) -> dict[str, Any] | None:
    """统一选择守卫与上下文使用的真实读参数或运行时拒绝反馈，不读取模型计划。

    反馈携带的reference_requirement沿用既有参考，失败重查不能清除它；新的成功读
    可以替换旧参考。反馈自己的call_id属于被拒绝的写提案，不能冒充原查询身份。
    核实旧参考的确切出处时用 successful_only 只选择成功读回执。
    """
    for index in range(len(previous_tool_results) - 1, -1, -1):
        result = previous_tool_results[index]
        if (not successful_only and result.result.get("reference_tool_name") == tool_name
                and isinstance(result.result.get("reference_requirement"), dict)):
            return {"requirement": result.result["reference_requirement"], "source_tool": tool_name,
                    "source_call_id": None, "source_evidence_index": index, "source_kind": "runtime_reference_feedback"}
        if result.name != tool_name or not result.called or not result.allowed or result.error:
            continue
        # 原桌定位只核对该桌状态，不能用它的档位和开场覆盖同轮另找或新组的个人条件。
        if tool_name == "search_current_games" and result.result.get("lookup_scope") == "specific_table":
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
    normalized = normalize_requirement(reference["requirement"])
    preserve = _requirement_fields_to_preserve(normalized)
    opening_window = current_game_query_start_range(normalized)
    if not preserve and opening_window is None:
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
        **({"searched_opening_window": {"start_at": opening_window[0].isoformat(), "end_at": opening_window[1].isoformat()}}
           if opening_window else {}),
        "revision_policy": (
            "默认复用preserve_requirement中的实际已查条件；模型计划、known_facts或fact_assertions的重写不能自行替换。"
            "若用户新输入或来源重核支持修订，在plan_revision_reason说明依据，先按修订条件重新成功查询。"
            "写入前置条件已独立核实后，才按新结果填写写入参数。"
            "失败查询不替换有效参考；新运行以新用户输入为准，不从历史运行继承此投影。"
        ),
        "boundary": (
            "这里只记录真实执行参数，不把模型先前理解升级为不可修订的用户事实，也不授予建局或外发权限。"
            "开场窗口不替客户约定固定钟点。人齐开不能保障晚于未来下界，确认能落实的开场后再写。"
            "目标桌况人数、参与者身份及查询容量不在本投影中，不能机械复制成新局占座。"
            "target_tool只说明建局获准后这些参数放在哪里。空结果不会补足己方人数或档位，其中有缺项时，保留原委托并追问，暂不调用create_game。"
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


def _uses_personal_clock_to_locate_table(requirement: Any, evidence: dict[str, Any]) -> bool:
    """核对原桌查找的开场过滤是否使用了独立判明的个人钟点。"""
    clocks = {parsed for value in evidence.get("personal_clocks_not_opening", ()) if (parsed := _aware_datetime(value))}
    if not clocks or not isinstance(requirement, dict):
        return False
    window = requirement.get("start_time_range")
    if isinstance(window, dict):
        return any(_aware_datetime(window.get(edge)) in clocks for edge in ("start_at", "end_at"))
    instant = _aware_datetime(requirement.get("planned_start_at")) or _aware_datetime(requirement.get("start_time"))
    if instant:
        return instant in clocks
    point = _clock_parts(requirement.get("start_time"))
    return point is not None and any(point == (item.hour, item.minute) for item in clocks)


def _uses_arrival_by_as_opening_filter(requirement: dict[str, Any], arrival_by: datetime) -> bool:
    """识别查局把最迟到店错作同日开场下界、上界或固定时刻。"""
    window = requirement.get("start_time_range")
    if isinstance(window, dict):
        if _aware_datetime(window.get("end_at")) == arrival_by:
            return True
        instant = _aware_datetime(window.get("start_at"))
    else:
        instant = _aware_datetime(requirement.get("planned_start_at")) or _aware_datetime(requirement.get("start_time"))
    if instant is not None:
        return instant.astimezone(arrival_by.tzinfo).date() == arrival_by.date() and instant >= arrival_by
    point = _clock_parts(requirement.get("start_time"))
    return point is not None and point >= (arrival_by.hour, arrival_by.minute)


def _uses_point_opening_filter(requirement: dict[str, Any]) -> bool:
    """识别固定开场过滤及只含一个瞬时的开场区间。"""
    if any(normalized_requirement_value(requirement.get(key)) not in (None, "")
           for key in ("start_time", "planned_start_at", "start_at", "start_datetime", "start_time_at")):
        return True
    window = requirement.get("start_time_range")
    start = _aware_datetime(window.get("start_at")) if isinstance(window, dict) else None
    return start is not None and start == _aware_datetime(window.get("end_at"))


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
