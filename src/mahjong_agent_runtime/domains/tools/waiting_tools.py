"""Tools for registering and cancelling passive matching requests."""

from __future__ import annotations

from datetime import datetime, time
from typing import Any

from ...models import (
    DEFAULT_TZ,
    StateTransition,
    ToolCall,
    ToolResult,
    WaitingDemand,
    WaitingDemandStatus,
    new_id,
    now,
)
from ...stores import AgentStore
from ..game_domain import normalize_requirement


def register_waiting_demand(
    store: AgentStore,
    call: ToolCall,
    trace_id: str,
    conversation_id: str,
    sender_id: str,
    sender_name: str,
) -> ToolResult:
    """Persist the authenticated customer's bounded waiting request."""

    stamp = now()
    expires_at = _expiry_from_arguments(call.arguments, stamp=stamp)
    normalized = normalize_requirement({"stake": call.arguments.get("stake")})
    demand = WaitingDemand(
        demand_id=new_id("waiting"),
        conversation_id=conversation_id,
        sender_id=sender_id,
        sender_name=sender_name,
        demand={
            "stake": str(normalized.get("stake") or call.arguments.get("stake") or ""),
            "smoke_preference": _normalize_smoke(call.arguments.get("smoke_preference")),
            "time_preference": str(call.arguments.get("time_preference") or ""),
            "extra_constraints": [
                str(item).strip()
                for item in call.arguments.get("extra_constraints") or []
                if str(item).strip()
            ],
            "source_channel": _channel_from_conversation(conversation_id),
        },
        created_at=stamp,
        expires_at=expires_at,
    )
    store.insert_waiting_demand(demand)
    transition = StateTransition(
        "waiting_demand",
        demand.demand_id,
        None,
        demand.status.value,
        "register_waiting_demand",
        trace_id,
    )
    return ToolResult(
        name=call.name,
        called=True,
        allowed=True,
        result={
            "waiting_demand": demand.to_dict(),
            "instruction": (
                "需求已登记。不要声称已经找到人；有匹配局时系统会主动回到该客户会话并征求确认。"
            ),
        },
        state_transitions=[transition],
    )


def cancel_waiting_demand(
    store: AgentStore,
    call: ToolCall,
    trace_id: str,
    conversation_id: str,
    sender_id: str,
    sender_name: str,
) -> ToolResult:
    """Cancel only waiting requests owned by the authenticated customer."""

    del sender_name
    cancelled = store.cancel_waiting_demands(
        conversation_id=conversation_id,
        sender_id=sender_id,
        demand_id=str(call.arguments.get("demand_id") or "") or None,
    )
    transitions = [
        StateTransition(
            "waiting_demand",
            item.demand_id,
            WaitingDemandStatus.ACTIVE.value,
            WaitingDemandStatus.CANCELLED.value,
            str(call.arguments.get("reason") or "cancel_waiting_demand"),
            trace_id,
        )
        for item in cancelled
    ]
    return ToolResult(
        name=call.name,
        called=True,
        allowed=True,
        result={
            "cancelled_demands": [item.to_dict() for item in cancelled],
            "cancelled_count": len(cancelled),
        },
        state_transitions=transitions,
    )


def _expiry_from_arguments(arguments: dict[str, Any], *, stamp: datetime) -> datetime:
    raw = str(arguments.get("expires_at") or "").strip()
    if raw:
        parsed = datetime.fromisoformat(raw)
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=DEFAULT_TZ)
    return datetime.combine(stamp.date(), time(23, 59, 59), tzinfo=DEFAULT_TZ)


def _normalize_smoke(value: object) -> str:
    text = str(value or "").strip().lower()
    aliases = {
        "无烟": "no_smoking",
        "no_smoke": "no_smoking",
        "no_smoking": "no_smoking",
        "烟": "smoking",
        "有烟": "smoking",
        "smoking": "smoking",
        "不限": "any",
        "都可": "any",
        "any": "any",
    }
    return aliases.get(text, text)


def _channel_from_conversation(conversation_id: str) -> str:
    prefix = str(conversation_id or "").partition(":")[0].lower()
    return prefix if prefix in {"wechaty", "weixin", "wechat", "douyin", "xiaohongshu"} else "internal"


__all__ = ["cancel_waiting_demand", "register_waiting_demand"]
