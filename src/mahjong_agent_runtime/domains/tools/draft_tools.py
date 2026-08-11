"""实现工具域中的草稿tools能力，供可信工具网关调用。

"""

from __future__ import annotations

from ...models import ToolCall, ToolResult
from ...stores import AgentStore
from ..model_context import invite_draft_for_model_context, outbound_message_draft_for_model_context
from .continuation import invite_draft_continuation

def create_invite_drafts(store: AgentStore, call: ToolCall, trace_id: str, conversation_id: str, sender_id: str, sender_name: str) -> ToolResult:
    """创建邀约drafts，并返回持久化后的权威结果。"""
    drafts, transitions = store.create_invite_drafts(
        game_id=str(call.arguments.get("game_id") or ""),
        invitations=list(call.arguments.get("invitations") or []),
        trace_id=trace_id,
    )
    return ToolResult(
        name=call.name,
        called=True,
        allowed=True,
        result={
            "drafts": [invite_draft_for_model_context(item, store.customers) for item in drafts],
            "continuation": invite_draft_continuation(str(call.arguments.get("game_id") or ""), len(drafts)),
        },
        state_transitions=transitions,
    )

def create_outbound_message_drafts(store: AgentStore, call: ToolCall, trace_id: str, conversation_id: str, sender_id: str, sender_name: str) -> ToolResult:
    """创建外发消息drafts，并返回持久化后的权威结果。"""
    drafts, transitions = store.create_outbound_message_drafts(
        conversation_id=conversation_id,
        drafts=list(call.arguments.get("drafts") or []),
        trace_id=trace_id,
    )
    return ToolResult(
        name=call.name,
        called=True,
        allowed=True,
        result={"drafts": [outbound_message_draft_for_model_context(item, store.customers) for item in drafts]},
        state_transitions=transitions,
    )
