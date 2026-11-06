"""实现工具域中的searchtools能力，供可信工具网关调用。

"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from ...models import ToolCall, ToolResult, now
from ...stores import AgentStore
from ..game_domain import normalize_current_game_query, normalize_requirement
from ..relationship_domain import task_memory_anchor_ids
from .continuation import bind_candidate_search_requirement, customer_search_continuation
from .shared import current_game_search_reply_contract


def canonical_search_current_games_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """查局通配值与省略玩法生成相同查询身份，不改写模型原始参数。"""

    return {
        "requirement": normalize_current_game_query(dict(arguments.get("requirement") or {})),
        "limit": int(arguments.get("limit") or 8),
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
    matches = [item for item in ranked_games if item.get("match_kind") == "exact"]
    alternatives = [
        item for item in ranked_games
        if item.get("match_kind") in {"profile_supported_alternative", "duration_confirmation_required"}
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
            "customer_reply_contract": current_game_search_reply_contract(
                requirement, matches, alternatives,
                unfiltered_available_pool_empty=unfiltered_available_pool_empty,
            ),
        },
    )

def check_room_availability(store: AgentStore, call: ToolCall, trace_id: str, conversation_id: str, sender_id: str, sender_name: str) -> ToolResult:
    """只读查询完整区间；均未指定时用可信后端时钟起四小时，不补猜半边显式时间。

    默认窗口是运营查询政策，不是用户已确认的到店/离店时间。只取一次 now 保持两端
    同源，Store 继续负责日期合法性和预约重叠判断，查询不会新建预约或组局。
    """
    arguments = call.arguments
    has_start, has_end = "start_at" in arguments, "end_at" in arguments
    if has_start != has_end:
        raise ValueError("start_at and end_at must be supplied together; omit both for the backend four-hour query window")
    window_policy: dict[str, Any] = {"query_window_source": "explicit_interval"}
    if has_start:
        start_at, end_at = arguments["start_at"], arguments["end_at"]
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
