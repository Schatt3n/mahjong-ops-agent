"""Read-only room, game, and customer search tool handlers."""

from __future__ import annotations

from typing import Any

from ...models import ToolCall, ToolResult
from ...stores import AgentStore
from ..game_domain import normalize_requirement
from ..relationship_domain import task_memory_anchor_ids
from .continuation import bind_candidate_search_requirement, customer_search_continuation
from .shared import current_game_search_reply_contract


def canonical_search_current_games_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """Collapse equivalent model arguments into one search identity."""

    return {
        "requirement": normalize_requirement(dict(arguments.get("requirement") or {})),
        "limit": int(arguments.get("limit") or 8),
    }


def canonical_search_customers_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """Collapse equivalent candidate-search arguments into one search identity."""

    return {
        "game_id": str(arguments.get("game_id") or ""),
        "requirement": normalize_requirement(dict(arguments.get("requirement") or {})),
        "exclude_customer_ids": sorted(
            {str(item) for item in arguments.get("exclude_customer_ids") or []}
        ),
        "limit": int(arguments.get("limit") or 8),
    }

def search_current_games(store: AgentStore, call: ToolCall, trace_id: str, conversation_id: str, sender_id: str, sender_name: str) -> ToolResult:
    requirement = normalize_requirement(dict(call.arguments.get("requirement") or {}))
    ranked_games = store.search_current_games(
        requirement,
        limit=int(call.arguments.get("limit") or 8),
        sender_id=sender_id,
        conversation_id=conversation_id,
    )
    matches = [item for item in ranked_games if item.get("match_kind") == "exact"]
    alternatives = [
        item for item in ranked_games if item.get("match_kind") == "profile_supported_alternative"
    ]
    return ToolResult(
        name=call.name,
        called=True,
        allowed=True,
        result={
            "requirement": requirement,
            "matches": matches,
            "alternatives": alternatives,
            "customer_reply_contract": current_game_search_reply_contract(requirement, matches, alternatives),
        },
    )

def check_room_availability(store: AgentStore, call: ToolCall, trace_id: str, conversation_id: str, sender_id: str, sender_name: str) -> ToolResult:
    availability = store.search_room_availability(
        start_at=call.arguments.get("start_at"),
        end_at=call.arguments.get("end_at"),
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
                "a forming game may still be created with room confirmation pending, but room availability must not be promised."
            ),
            "next_step_policy": {
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
