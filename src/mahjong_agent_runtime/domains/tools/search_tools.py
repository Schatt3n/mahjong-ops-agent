"""Read-only room, game, and customer search tool handlers."""

from __future__ import annotations

from ...models import ToolCall, ToolResult
from ...stores import AgentStore
from ..game_domain import normalize_requirement
from .shared import current_game_search_reply_contract

def search_current_games(store: AgentStore, call: ToolCall, trace_id: str, conversation_id: str, sender_id: str, sender_name: str) -> ToolResult:
    requirement = normalize_requirement(dict(call.arguments.get("requirement") or {}))
    matches = store.search_current_games(
        requirement,
        limit=int(call.arguments.get("limit") or 8),
        sender_id=sender_id,
        conversation_id=conversation_id,
    )
    return ToolResult(
        name=call.name,
        called=True,
        allowed=True,
        result={
            "requirement": requirement,
            "matches": matches,
            "customer_reply_contract": current_game_search_reply_contract(requirement, matches),
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
    requirement = normalize_requirement(dict(call.arguments.get("requirement") or {}))
    exclude_customer_ids = [str(item) for item in call.arguments.get("exclude_customer_ids") or []]
    candidates = store.search_customers(
        requirement,
        exclude_customer_ids=exclude_customer_ids,
        limit=int(call.arguments.get("limit") or 8),
        sender_id=sender_id,
        conversation_id=conversation_id,
    )
    return ToolResult(
        name=call.name,
        called=True,
        allowed=True,
        result={"requirement": requirement, "exclude_customer_ids": exclude_customer_ids, "candidates": candidates},
    )
