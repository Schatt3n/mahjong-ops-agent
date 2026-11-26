"""实现工具域中的audittools能力，供可信工具网关调用。

"""

from __future__ import annotations

from typing import Any

from ...models import ConversationRole, ConversationTurn, ToolCall, ToolResult
from ...stores import AgentStore

def record_badcase(store: AgentStore, call: ToolCall, trace_id: str, conversation_id: str, sender_id: str, sender_name: str) -> ToolResult:
    """记录badcase，形成后续决策和审计所需的持久证据。"""
    record = store.record_badcase(dict(call.arguments), trace_id=trace_id, conversation_id=conversation_id)
    return ToolResult(name=call.name, called=True, allowed=True, result={"recorded": True, "badcase": record})

def update_context_checkpoint(store: AgentStore, call: ToolCall, trace_id: str, conversation_id: str, sender_id: str, sender_name: str) -> ToolResult:
    """更新上下文检查点，同时维护相关状态不变量。"""
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

def _authenticated_memory_turn(turn: ConversationTurn | None, sender_id: str) -> bool:
    """只信本人已入库的用户文本，隐藏、撤回、系统事件和纯媒体说明不构成长期证据。"""
    return bool(turn and turn.role == ConversationRole.USER and turn.sender_id == sender_id
                and isinstance(turn.metadata.get("source_message_id"), str) and turn.metadata["source_message_id"]
                and not turn.metadata.get("context_invalidated")
                and not turn.metadata.get("internal_event") and turn.metadata.get("input_source") != "system_trigger"
                and not turn.metadata.get("_trusted_visual_observation_only"))


def _pending_memory_evidence_source(
    store: AgentStore, current: ConversationTurn, *, conversation_id: str, sender_id: str,
    quote: str, requested_id: str | None,
) -> str | None:
    """用原话验证源 ID，允许本轮消息和已点名的同一任务、同一发言人原话。"""
    source = current
    current_id = current.metadata["source_message_id"]
    window = current.metadata.get("input_window")
    if isinstance(window, dict):
        batch = store.pending_input_batch(conversation_id, sender_id)
        # 合并视图不是原始消息。只在后端仍能核实这批原始片段时逐条查找，避免把换行拼出的句子当成原话。
        if batch and batch.batch_id == window.get("batch_id") and batch.version == window.get("batch_version"):
            for fragment in reversed(batch.fragments):
                fragment_id = fragment.get("message_id")
                fragment_text = fragment.get("text")
                attributes = fragment.get("metadata") or {}
                if (isinstance(fragment_id, str) and fragment_id and (requested_id is None or requested_id == fragment_id)
                        and isinstance(fragment_text, str) and quote in fragment_text
                        and isinstance(attributes, dict) and not attributes.get("_trusted_visual_observation_only")):
                    return fragment_id
        if not requested_id or requested_id == current_id:
            return None
    if requested_id and requested_id != current_id:
        source = store.find_conversation_turn(conversation_id, message_id=requested_id)
        task = store.current_task_context(conversation_id, sender_id)
        if (source is None or not _authenticated_memory_turn(source, sender_id) or task is None
                or source.metadata.get("task_context_id") != task.task_context_id
                or source.occurred_at > current.occurred_at or source.metadata.get("input_window")):
            return None
    return source.metadata["source_message_id"] if quote in source.content else None


def _bind_pending_memory_evidence(
    store: AgentStore, pending: list[dict[str, Any]], *, conversation_id: str, sender_id: str, trace_id: str,
) -> tuple[list[tuple[dict[str, Any], str, str]], int | None]:
    """先核实全部候选来源，任何一项无证据就整批拒绝，便于主模型纠正后安全重试。"""
    if not pending:
        return [], None
    current = next((turn for turn in reversed(store.recent_turns(conversation_id))
                    if turn.trace_id == trace_id and _authenticated_memory_turn(turn, sender_id)), None)
    bound = []
    for index, raw in enumerate(pending):
        quote = str(raw.get("evidence") or "").strip()
        source_id = _pending_memory_evidence_source(store, current, conversation_id=conversation_id,
            sender_id=sender_id, quote=quote, requested_id=raw.get("evidence_message_id")) if current and quote else None
        if source_id is None:
            return [], index
        bound.append((raw, quote, source_id))
    return bound, None


def record_user_memory(store: AgentStore, call: ToolCall, trace_id: str, conversation_id: str, sender_id: str, sender_name: str) -> ToolResult:
    """记录用户记忆，形成后续决策和审计所需的持久证据。"""
    pending, rejected_index = _bind_pending_memory_evidence(store,
        call.arguments.get("pending_long_term_memories") or [], conversation_id=conversation_id,
        sender_id=sender_id, trace_id=trace_id)
    if rejected_index is not None:
        error = "待审长期候选需要这位顾客已发送的连续原话，不能用转述或引用他人发言。"
        return ToolResult(name=call.name, called=False, allowed=False, error=error, result={
            "validation_error": {"code": "pending_memory_evidence_not_verbatim", "message": error,
                                 "item_index": rejected_index},
            "instruction": "本次记忆写入尚未执行。从本轮同一顾客的原话连续复制能证明长期习惯的一段；若引用本任务较早的本人消息，填原消息ID。修正后重试；没有该证据就不要登记候选。",
        })
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
    for raw, evidence, source_id in pending:
        metadata = dict(raw.get("metadata") or {}) if isinstance(raw.get("metadata"), dict) else {}
        metadata.pop("task_context_id", None)
        metadata["source_message_id"] = source_id
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
            evidence=evidence,
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
