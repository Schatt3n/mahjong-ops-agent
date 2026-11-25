"""实现工具域中的searchtools能力，供可信工具网关调用。

"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from ...models import ToolCall, ToolResult, now
from ...stores import AgentStore
from ..customer_domain import HOST_CONFIRMATION_REQUIRED
from ..game_domain import (
    current_game_query_start_range, normalize_current_game_query, normalize_requirement, parse_datetime_value,
)
from ..relationship_domain import task_memory_anchor_ids
from .continuation import bind_candidate_search_requirement, customer_search_continuation
from .shared import current_game_search_reply_contract


_NEARBY_START_WINDOW = timedelta(minutes=10)


def canonical_search_current_games_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """查局通配值与省略玩法生成相同查询身份，不改写模型原始参数。"""

    requirement = normalize_current_game_query(dict(arguments.get("requirement") or {}))
    return {
        "requirement": requirement,
        "limit": int(arguments.get("limit") or 8),
        **({"include_nearby": False} if "start_time_range" in requirement and arguments.get("include_nearby") is False else {}),
    }


def canonical_search_customers_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """生成规范的searchcustomersarguments表示，供比较和持久化使用。

"""

    return {
        "game_id": str(arguments.get("game_id") or ""),
        "requirement": normalize_requirement(dict(arguments.get("requirement") or {})),
        "exclude_customer_ids": sorted(
            {str(item) for item in arguments.get("exclude_customer_ids") or []}
        ),
        "limit": int(arguments.get("limit") or 8),
    }

def search_current_games(store: AgentStore, call: ToolCall, trace_id: str, conversation_id: str, sender_id: str, sender_name: str) -> ToolResult:
    """查询实际条件；仅用同请求者作用域的未过滤空池证明子范围也无可用局。"""
    requirement = normalize_current_game_query(dict(call.arguments.get("requirement") or {}))
    ranked_games = store.search_current_games(
        requirement,
        limit=int(call.arguments.get("limit") or 8),
        sender_id=sender_id,
        conversation_id=conversation_id,
    )
    window = current_game_query_start_range(requirement)
    nearby = (_nearby_opening_options(store, requirement, window, sender_id=sender_id, conversation_id=conversation_id)
              if window is not None and call.arguments.get("include_nearby") is not False else None)
    matches = [item for item in ranked_games if item.get("match_kind") == "exact"]
    alternatives = [
        item for item in ranked_games
        if item.get("match_kind") in {"profile_supported_alternative", "duration_confirmation_required", HOST_CONFIRMATION_REQUIRED}
    ]
    # 排序器会按总分排除候选，新增条件可能提高分数，故过滤查询为空不证明子范围为空。
    # 只额外读取同一请求者/会话的未过滤可用局池是否存在，不把局池内容带入工具结果。
    unfiltered_available_pool_empty = False
    if not ranked_games:
        unfiltered_available_pool_empty = not requirement or not store.search_current_games(
            {}, limit=1, sender_id=sender_id, conversation_id=conversation_id,
        )
    return ToolResult(
        name=call.name,
        called=True,
        allowed=True,
        result={
            "requirement": requirement,
            "matches": matches,
            "alternatives": alternatives,
            **({"nearby_options": nearby} if nearby is not None else {}),
            "customer_reply_contract": current_game_search_reply_contract(
                requirement, matches, alternatives,
                unfiltered_available_pool_empty=unfiltered_available_pool_empty,
            ),
        },
    )


def _nearby_opening_options(
    store: AgentStore, requirement: dict[str, Any], window: tuple[datetime, datetime], *,
    sender_id: str, conversation_id: str,
) -> list[dict[str, Any]]:
    """从同一只读权限范围查询临近边界的备选，保持真实期待范围内的匹配原样。"""
    start, end = window
    tick = timedelta(microseconds=1)
    upcoming = now()
    options: list[dict[str, Any]] = []
    for side, lower, upper, boundary in (
        ("before", start - _NEARBY_START_WINDOW, start - tick, start),
        ("after", end + tick, end + _NEARBY_START_WINDOW, end),
    ):
        lower = max(lower, upcoming)
        if lower > upper:
            continue
        query = requirement | {"start_time_range": {"start_at": lower.isoformat(), "end_at": upper.isoformat()}}
        # 普通查局按适配分数排序，截断后再比较分钟数会漏掉更近的桌。
        matches = store.search_current_games(query, limit=None, sender_id=sender_id, conversation_id=conversation_id)
        ranked: list[tuple[timedelta, dict[str, Any]]] = []
        for candidate in matches:
            if candidate.get("match_kind") != "exact":
                continue
            game = candidate["game"]
            opening = parse_datetime_value(game.get("planned_start_at") or (game.get("requirement") or {}).get("planned_start_at"))
            if opening is not None:
                ranked.append((abs(opening - boundary), candidate))
        if ranked:
            gap, candidate = min(ranked, key=lambda item: item[0])
            options.append({
                **candidate, "match_kind": "nearby_start_option", "relative_to_window": side,
                "offset_minutes": round(gap.total_seconds() / 60, 2),
            })
    return options


def check_room_availability(store: AgentStore, call: ToolCall, trace_id: str, conversation_id: str, sender_id: str, sender_name: str) -> ToolResult:
    """只读查询明确起止或明确时长，只有客户没有给时间约束才用四小时默认窗口。

    相对时长从明确的起点或可信当前时钟计算。默认窗口是运营查询政策，不代表客户预约；
    Store 继续负责日期与预约重叠判断，查询不会创建预约或组局。
    """
    arguments = call.arguments
    has_start, has_end = "start_at" in arguments, "end_at" in arguments
    duration = arguments.get("duration_minutes")
    window_policy: dict[str, Any] = {"query_window_source": "explicit_interval"}
    if duration is not None:
        if has_end:
            raise ValueError("duration_minutes cannot be combined with end_at")
        start_at = parse_datetime_value(arguments["start_at"]) if has_start else now()
        if start_at is None:
            raise ValueError("start_at must be a valid datetime when supplied with duration_minutes")
        try:
            end_at = start_at + timedelta(minutes=duration)
        except OverflowError as exc:
            raise ValueError("duration_minutes exceeds the supported datetime range") from exc
        window_policy = {"query_window_source": "explicit_duration", "query_window_duration_minutes": duration}
    elif has_start and has_end:
        start_at, end_at = arguments["start_at"], arguments["end_at"]
    elif has_start or has_end:
        raise ValueError("supply both start_at and end_at, or duration_minutes with an optional start_at")
    else:
        start_at = now()
        end_at = start_at + timedelta(hours=4)
        window_policy = {"query_window_source": "backend_now_plus_4h", "query_window_default_hours": 4}
    availability = store.search_room_availability(
        start_at=start_at,
        end_at=end_at,
    )
    return ToolResult(
        name=call.name,
        called=True,
        allowed=True,
        result={
            **availability,
            "instruction": (
                "Only state that a room is available when configured=true and available_count>0. "
                "This read does not reserve or promise a room. When configured=false, availability is unknown: "
                "a forming game may still be created with room confirmation pending, but room availability must not be promised. "
                "The returned start_at/end_at are the actual query window. A backend default is not a customer-confirmed schedule."
            ),
            "next_step_policy": {
                # 原有上下文允许列表保留 next_step_policy，默认来源随真实结果回灌而不冒充用户事实。
                **window_policy,
                "query_completed": True,
                "repeat_same_query": False,
                "may_create_forming_game_with_room_pending": not availability["configured"],
                "may_state_room_available": bool(
                    availability["configured"] and availability["available_count"] > 0
                ),
                "must_report_unavailable": bool(
                    availability["configured"] and availability["available_count"] <= 0
                ),
                "instruction": (
                    "Do not repeat check_room_availability with the same interval. Mark the room-check plan step done. "
                    "If inventory is unconfigured, continue the requested business flow with room confirmation "
                    "pending and do not claim that a room exists. If configured and available_count is zero, do not "
                    "create or promise a fixed-time game for this interval; offer another time."
                ),
            },
        },
    )

def search_customers(store: AgentStore, call: ToolCall, trace_id: str, conversation_id: str, sender_id: str, sender_name: str) -> ToolResult:
    """查询customers；只返回满足当前作用域和约束的结果。"""
    requirement, bound_game, binding_error = bind_candidate_search_requirement(
        store,
        conversation_id=conversation_id,
        sender_id=sender_id,
        requirement=dict(call.arguments.get("requirement") or {}),
        game_id=call.arguments.get("game_id"),
    )
    if binding_error:
        return ToolResult(
            name=call.name,
            called=False,
            allowed=False,
            error=binding_error,
            result={
                "requirement": requirement,
                "requested_game_id": call.arguments.get("game_id"),
                "instruction": "Retry search_customers with the authoritative game_id returned by create_game.",
            },
        )
    exclude_customer_ids = sorted(
        set(str(item) for item in call.arguments.get("exclude_customer_ids") or [])
        | set(task_memory_anchor_ids(requirement, sender_id=sender_id))
    )
    candidates = store.search_customers(
        requirement,
        exclude_customer_ids=exclude_customer_ids,
        limit=int(call.arguments.get("limit") or 8),
        sender_id=sender_id,
        conversation_id=conversation_id,
    )
    continuation = customer_search_continuation(
        store,
        conversation_id=conversation_id,
        sender_id=sender_id,
        requirement=requirement,
        candidates=candidates,
        game=bound_game,
    )
    result = {
        "requirement": requirement,
        "bound_game_id": bound_game.game_id if bound_game is not None else None,
        "requirement_source": "active_game_aggregate" if bound_game is not None else "model_proposal",
        "exclude_customer_ids": exclude_customer_ids,
        "candidates": candidates,
    }
    if continuation is not None:
        result["continuation"] = continuation
    return ToolResult(
        name=call.name,
        called=True,
        allowed=True,
        result=result,
    )
