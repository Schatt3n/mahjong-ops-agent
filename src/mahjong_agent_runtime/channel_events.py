"""处理有明确通道协议的非文本事件；不根据自然语言关键词猜撤回。"""

from __future__ import annotations

from typing import Any
from .models import AgentAction, AgentRuntimeResult, UserMessage


def handle_channel_event(runtime: Any, message: UserMessage, *, trace_id: str) -> AgentRuntimeResult | None:
    """验证撤回目标属于同一客户，静默处理并保留审计；返回 None 表示普通消息。"""
    metadata = message.metadata
    raw = metadata.get("raw_wechat_payload") or {}
    event_type = metadata.get("event_type") or raw.get("message_type") or metadata.get("content_type")
    if event_type not in {"message_recalled", "recall", "revoke_event"}:
        return None
    target = str(metadata.get("target_message_id") or raw.get("recalled_message_id") or "")
    changed = runtime.store.invalidate_message_context(message.conversation_id, message.sender_id, target, trace_id=trace_id) if target else False
    transitions = []
    if changed:
        _, transition = runtime.store.advance_conversation_version(message.conversation_id, trace_id=trace_id, reason="message_recalled")
        transitions.append(transition)
    message.metadata = {**metadata, "event_type": "message_recalled", "target_message_id": target, "context_invalidated": True}
    runtime.trace_recorder.record(trace_id, "user_input", {"message": message.to_dict()})
    runtime.store.append_user_turn(message, trace_id)
    action = AgentAction(goal="记录通道撤回事件", objective_status="completed", reasoning_summary="事件无需客户回复；不撤销已执行的业务动作。",
                         response_mode="silent", silence_reason="channel_recall_event", stop_reason={"can_stop": True, "why": "事件已记录", "pending_work": [], "depends_on_tool_results": False})
    runtime.trace_recorder.record(trace_id, "channel_event_processed", {"event_type": "message_recalled", "target_message_id": target, "context_invalidated": changed})
    runtime.trace_recorder.record(trace_id, "final_output", {"reply": "", "response_mode": "silent", "silence_reason": action.silence_reason})
    return AgentRuntimeResult(trace_id=trace_id, conversation_id=message.conversation_id, final_reply="", actions=[action], state_transitions=transitions)
