"""将已校验的写工具调用适配为 ``AgentStore`` 操作和 ``ToolResult``。

导航：这些函数由 ``registry.default_tool_definitions`` 通过 ``partial(store)`` 动态绑定，
不是模型直接调用的业务实现。生产 ``store`` 是 ``SQLiteAgentStore``；权威事务、容量
校验和状态机位于其对应 Mixin 中，本模块只负责参数适配与结果投影。
"""

from __future__ import annotations

from ...models import ToolCall, ToolResult
from ...stores import AgentStore
from ..game_domain import normalize_requirement
from ..model_context import game_for_model_context
from .continuation import create_game_continuation
from .shared import (
    CANDIDATE_REPLY_NEXT_STEP_POLICIES,
    cross_game_commitment_summary,
    known_players_with_requesting_party,
)

def reserve_room(store: AgentStore, call: ToolCall, trace_id: str, conversation_id: str, sender_id: str, sender_name: str) -> ToolResult:
    """适配订房工具；原子库存校验与占用由 Store 的 ``reserve_room`` 实现。"""
    reservation, transition = store.reserve_room(
        conversation_id=conversation_id,
        game_id=str(call.arguments.get("game_id") or "") or None,
        start_at=call.arguments.get("start_at"),
        end_at=call.arguments.get("end_at"),
        room_id=str(call.arguments.get("room_id") or "") or None,
        trace_id=trace_id,
    )
    return ToolResult(
        name=call.name,
        called=True,
        allowed=True,
        result={"reservation": reservation.to_dict()},
        state_transitions=[transition],
    )

def create_game(store: AgentStore, call: ToolCall, trace_id: str, conversation_id: str, sender_id: str, sender_name: str) -> ToolResult:
    """创建牌局，并返回持久化后的权威结果。"""
    known_players = known_players_with_requesting_party(
        known_players=list(call.arguments.get("known_players") or []),
        requesting_party=call.arguments.get("requesting_party"),
    )
    game, transition = store.create_game(
        conversation_id=conversation_id,
        organizer_id=str(call.arguments["organizer_id"]),
        organizer_name=str(call.arguments["organizer_name"]),
        requirement=normalize_requirement(dict(call.arguments.get("requirement") or {})),
        known_players=known_players,
        trace_id=trace_id,
    )
    scheduled_task = store.scheduled_task_for_game(game.game_id)
    return ToolResult(
        name=call.name,
        called=True,
        allowed=True,
        result={
            "game": game_for_model_context(game, store.customers),
            "continuation": create_game_continuation(game),
            "recruitment_policy": {
                "status": game.recruitment_status.value,
                "opens_at": game.recruitment_opens_at.isoformat() if game.recruitment_opens_at else None,
                "scheduled_task": scheduled_task.to_dict() if scheduled_task else None,
                "instruction": (
                    "When status=scheduled, keep the game visible but do not search private candidates or create "
                    "invite drafts. A durable system task will re-enter the main Agent when the window opens."
                ),
            },
        },
        state_transitions=[transition],
    )

def join_game(store: AgentStore, call: ToolCall, trace_id: str, conversation_id: str, sender_id: str, sender_name: str) -> ToolResult:
    """适配入局工具；容量、冲突和参与者状态机由 Store 的 ``join_game`` 实现。"""
    game, transitions = store.join_game(
        game_id=str(call.arguments["game_id"]),
        customer_id=str(call.arguments["customer_id"]),
        display_name=str(call.arguments["display_name"]),
        seat_count=int(call.arguments.get("seat_count") or 1),
        trace_id=trace_id,
    )
    return ToolResult(
        name=call.name,
        called=True,
        allowed=True,
        result={
            "game": game_for_model_context(game, store.customers),
            "recorded_status": "confirmed",
            "next_step_policy": CANDIDATE_REPLY_NEXT_STEP_POLICIES["confirmed"],
            "cross_game_commitment": cross_game_commitment_summary(transitions),
        },
        state_transitions=transitions,
    )

def update_game_requirement(store: AgentStore, call: ToolCall, trace_id: str, conversation_id: str, sender_id: str, sender_name: str) -> ToolResult:
    """更新牌局组局条件，同时维护相关状态不变量。"""
    game, transition = store.update_game_requirement(
        game_id=str(call.arguments.get("game_id") or ""),
        requirement_patch=normalize_requirement(dict(call.arguments.get("requirement_patch") or {})),
        reason=str(call.arguments.get("reason") or ""),
        trace_id=trace_id,
    )
    return ToolResult(
        name=call.name,
        called=True,
        allowed=True,
        result={"game": game_for_model_context(game, store.customers)},
        state_transitions=[transition],
    )

def record_candidate_reply(store: AgentStore, call: ToolCall, trace_id: str, conversation_id: str, sender_id: str, sender_name: str) -> ToolResult:
    """记录候选人reply，形成后续决策和审计所需的持久证据。"""
    status = str(call.arguments["status"])
    game, transitions = store.record_candidate_reply(
        game_id=str(call.arguments["game_id"]),
        customer_id=str(call.arguments["customer_id"]),
        display_name=str(call.arguments["display_name"]),
        status=status,
        seat_count=int(call.arguments.get("seat_count") or 1),
        trace_id=trace_id,
    )
    return ToolResult(
        name=call.name,
        called=True,
        allowed=True,
        result={
            "game": game_for_model_context(game, store.customers),
            "recorded_status": status,
            "next_step_policy": CANDIDATE_REPLY_NEXT_STEP_POLICIES.get(status, {}),
            "cross_game_commitment": cross_game_commitment_summary(transitions),
        },
        state_transitions=transitions,
    )

def update_game_status(store: AgentStore, call: ToolCall, trace_id: str, conversation_id: str, sender_id: str, sender_name: str) -> ToolResult:
    """更新牌局状态，同时维护相关状态不变量。"""
    game, transition = store.update_game_status(
        game_id=str(call.arguments["game_id"]),
        status=str(call.arguments["status"]),
        reason=str(call.arguments["reason"]),
        trace_id=trace_id,
    )
    return ToolResult(
        name=call.name,
        called=True,
        allowed=True,
        result={"game": game_for_model_context(game, store.customers)},
        state_transitions=[transition],
    )
