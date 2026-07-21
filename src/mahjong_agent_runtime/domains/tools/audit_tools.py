"""Audit, checkpoint, and memory tool handlers."""

from __future__ import annotations

from ...models import ToolCall, ToolResult
from ...stores import AgentStore

def record_badcase(store: AgentStore, call: ToolCall, trace_id: str, conversation_id: str, sender_id: str, sender_name: str) -> ToolResult:
    record = store.record_badcase(dict(call.arguments), trace_id=trace_id, conversation_id=conversation_id)
    return ToolResult(name=call.name, called=True, allowed=True, result={"recorded": True, "badcase": record})

def update_context_checkpoint(store: AgentStore, call: ToolCall, trace_id: str, conversation_id: str, sender_id: str, sender_name: str) -> ToolResult:
    checkpoint, transition = store.upsert_conversation_checkpoint(
        conversation_id=conversation_id,
        summary=str(call.arguments["summary"]),
        facts=dict(call.arguments.get("facts") or {}),
        open_questions=[str(item) for item in call.arguments.get("open_questions") or []],
        trace_id=trace_id,
    )
    return ToolResult(
        name=call.name,
        called=True,
        allowed=True,
        result={"checkpoint": checkpoint.to_dict()},
        state_transitions=[transition],
    )

def record_user_memory(store: AgentStore, call: ToolCall, trace_id: str, conversation_id: str, sender_id: str, sender_name: str) -> ToolResult:
    task_memories = []
    pending_candidates = []
    transitions = []
    for raw in call.arguments.get("task_memories") or []:
        if not isinstance(raw, dict):
            continue
        metadata = dict(raw.get("metadata") or {}) if isinstance(raw.get("metadata"), dict) else {}
        if raw.get("target_customer_name") and "target_customer_name" not in metadata:
            metadata["target_customer_name"] = str(raw.get("target_customer_name") or "")
        memory, transition = store.record_task_memory(
            conversation_id=conversation_id,
            customer_id=str(raw.get("customer_id") or sender_id),
            memory_type=str(raw.get("memory_type") or ""),
            field=str(raw.get("field") or ""),
            value=raw.get("value"),
            target_customer_id=str(raw.get("target_customer_id") or "") or None,
            evidence=str(raw.get("evidence") or ""),
            confidence=float(raw.get("confidence") or 0.0),
            risk_level=str(raw.get("risk_level") or "medium"),
            scope=str(raw.get("scope") or "current_task"),
            metadata=metadata,
            trace_id=trace_id,
        )
        task_memories.append(memory.to_dict())
        transitions.append(transition)
    for raw in call.arguments.get("pending_long_term_memories") or []:
        if not isinstance(raw, dict):
            continue
        metadata = dict(raw.get("metadata") or {}) if isinstance(raw.get("metadata"), dict) else {}
        if raw.get("target_customer_name") and "target_customer_name" not in metadata:
            metadata["target_customer_name"] = str(raw.get("target_customer_name") or "")
        candidate, transition = store.record_pending_memory_candidate(
            conversation_id=conversation_id,
            customer_id=str(raw.get("customer_id") or sender_id),
            memory_type=str(raw.get("memory_type") or ""),
            field=str(raw.get("field") or ""),
            value=raw.get("value"),
            operation=str(raw.get("operation") or "set"),
            target_customer_id=str(raw.get("target_customer_id") or "") or None,
            evidence=str(raw.get("evidence") or ""),
            confidence=float(raw.get("confidence") or 0.0),
            risk_level=str(raw.get("risk_level") or "medium"),
            scope=str(raw.get("scope") or "long_term"),
            metadata=metadata,
            trace_id=trace_id,
        )
        pending_candidates.append(candidate.to_dict())
        transitions.append(transition)
    return ToolResult(
        name=call.name,
        called=True,
        allowed=True,
        result={
            "task_memories": task_memories,
            "pending_long_term_memories": pending_candidates,
            "next_step_policy": {
                "memory_write_does_not_authorize_downstream_actions": True,
                "requires_explicit_user_request_to_expand_goal": True,
                "allows_resume_when_previous_plan_was_blocked_by_this_fact": True,
                "default_next_action": "reply_with_short_confirmation",
                "instruction": (
                    "The memory is now active, but this write does not authorize new downstream work. "
                    "Only continue search, matching, or draft creation when the current user message explicitly "
                    "requests it, or when the prior plan was already blocked waiting for exactly this fact. "
                    "Otherwise stop with a short confirmation. Pending long-term candidates are not yet profiles."
                )
            },
        },
        state_transitions=transitions,
    )
