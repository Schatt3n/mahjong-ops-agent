"""在显式引用或可信调度触发时，按边界恢复历史任务证据。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...models import ConversationTurn, UserMessage
from ...stores import AgentStore
from .conversation_context import ContextPackingPolicy
from .tool_results import turn_payload_for_context


@dataclass(slots=True)
class RecoveredTaskContexts:
    """保存从引用消息或未来预约中恢复出的历史任务上下文与审计依据。"""

    items: list[dict[str, Any]]
    audit: dict[str, Any]


def recover_referenced_task_contexts(
    store: AgentStore,
    message: UserMessage,
    current_message: dict[str, Any],
    *,
    current_task_context_id: str | None,
    packing_policy: ContextPackingPolicy,
) -> RecoveredTaskContexts:
    """仅在引用或受信任调度器点名时恢复旧证据。

    会话 ID 是隐私边界，跨会话引用不会导入原始轮次或 checkpoint。恢复内容只是解释
    当前消息的有界证据；牌局等可变状态在写入前仍须通过工具查询权威值。
    """

    requested: dict[str, set[str]] = {}
    source_turns: dict[str, ConversationTurn] = {}
    denied_cross_conversation_quote = False

    quoted = message.quoted_message
    if quoted is not None:
        quoted_conversation_id = quoted.conversation_id or message.conversation_id
        if quoted_conversation_id != message.conversation_id:
            denied_cross_conversation_quote = True
        else:
            source_turn = store.find_conversation_turn(
                message.conversation_id,
                message_id=quoted.message_id or None,
                text=quoted.text or None,
                sender_id=quoted.sender_id,
            )
            if source_turn is not None:
                task_context_id = str(source_turn.metadata.get("task_context_id") or "")
                if task_context_id and task_context_id != current_task_context_id:
                    requested.setdefault(task_context_id, set()).add("quoted_message")
                    source_turns[task_context_id] = source_turn
                    quoted_payload = dict(current_message.get("quoted_message") or {})
                    quoted_payload["text"] = quoted_payload.get("text") or source_turn.content
                    current_message["quoted_message"] = quoted_payload

    scheduled_task_context_id = _trusted_scheduled_task_context_id(message)
    if scheduled_task_context_id and scheduled_task_context_id != current_task_context_id:
        task_context = store.get_task_context(scheduled_task_context_id)
        if task_context is not None and task_context.conversation_id == message.conversation_id:
            requested.setdefault(scheduled_task_context_id, set()).add("scheduled_task")

    items: list[dict[str, Any]] = []
    recovered_turn_count = 0
    for task_context_id, sources in requested.items():
        task_context = store.get_task_context(task_context_id)
        if task_context is None or task_context.conversation_id != message.conversation_id:
            continue
        checkpoint = store.get_task_context_checkpoint(task_context_id)
        turns = store.task_context_turns(
            message.conversation_id,
            task_context_id,
            packing_policy.max_turns_considered,
        )
        if checkpoint is not None:
            evidence_turns = [turn for turn in turns if turn.occurred_at > checkpoint.updated_at]
            evidence_mode = "checkpoint"
        else:
            evidence_turns = turns
            evidence_mode = "raw_task_turns"
        packed_turns, packing_audit = packing_policy.pack_turns(evidence_turns)
        recovered_turn_count += len(packed_turns)
        source_turn = source_turns.get(task_context_id)
        items.append(
            {
                "task_context_id": task_context_id,
                "conversation_id": task_context.conversation_id,
                "customer_id": task_context.customer_id,
                "status": task_context.status,
                "started_at": task_context.started_at.isoformat(),
                "sources": sorted(sources),
                "source_turn": turn_payload_for_context(source_turn) if source_turn else None,
                "evidence_mode": evidence_mode,
                "checkpoint": checkpoint.to_dict() if checkpoint else None,
                "recent_task_turns": packed_turns,
                "packing_audit": packing_audit,
                "usage_contract": (
                    "This is bounded evidence for the explicitly referenced task. "
                    "Use it to interpret the current request, but verify mutable state with tools before writing."
                ),
            }
        )

    return RecoveredTaskContexts(
        items=items,
        audit={
            "recovered_task_context_count": len(items),
            "recovered_task_turn_count": recovered_turn_count,
            "recovered_task_context_ids": [item["task_context_id"] for item in items],
            "cross_conversation_quote_recovery_denied": denied_cross_conversation_quote,
            "scheduled_task_context_id": scheduled_task_context_id or None,
        },
    )


def _trusted_scheduled_task_context_id(message: UserMessage) -> str:
    metadata = message.metadata if isinstance(message.metadata, dict) else {}
    # 该后端私有字段只能由 AgentRuntime 在接受持久化定时任务或系统触发后写入；
    # 外部消息元数据不得借此任意选择历史任务。
    return str(metadata.get("_trusted_source_task_context_id") or "")


__all__ = ["RecoveredTaskContexts", "recover_referenced_task_contexts"]
