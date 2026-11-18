"""从持久化对话中选择当前任务片段，并按 checkpoint 与预算裁剪。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ...models import (
    ConversationCheckpoint,
    ConversationTaskContext,
    ConversationTurn,
    ToolResult,
    UserMessage,
)
from ...stores import AgentStore
from ...token_estimation import estimate_tokens
from .sanitization import context_text_preview
from .tool_results import reference_shared_query_contracts, turn_payload_for_context


@dataclass(slots=True)
class ContextPackingPolicy:
    """描述上下文packing策略采用的可配置策略参数。"""
    max_turns_considered: int = 60
    max_recent_conversation_tokens: int = 4_000

    def pack_turns(
        self,
        turns: list[ConversationTurn],
        *,
        shared_query_contract: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """按实际传输大小选历史，返回完整审计投影而不改原始轮次。

        只有调用方显式提供本轮 planning 中的真实合同锚点，才按精确相等的引用副本
        计费；不同版本仍计完整大小。最终发送由 ContextLifecycleManager 使用同一
        投影函数去重，避免静态重复先挤掉候选局、失败结果或来源等动态证据。
        """

        considered = [turn for turn in turns if not turn.metadata.get("context_invalidated")][-self.max_turns_considered :]
        included_reversed: list[dict[str, Any]] = []
        estimated_tokens = 0
        omitted_for_budget = 0
        referenced_query_contract_count = 0
        for turn in reversed(considered):
            payload = turn_payload_for_context(turn)
            budget_payload = payload
            reference_count = 0
            if shared_query_contract is not None:
                projected, reference_count = reference_shared_query_contracts({
                    "planning_contract": {"game_query_strategy_contract": shared_query_contract},
                    "recent_conversation": [payload],
                })
                budget_payload = projected["recent_conversation"][0]
            turn_tokens = estimate_tokens(budget_payload)
            if included_reversed and estimated_tokens + turn_tokens > self.max_recent_conversation_tokens:
                omitted_for_budget += 1
                continue
            included_reversed.append(payload)
            estimated_tokens += turn_tokens
            referenced_query_contract_count += reference_count
        included = list(reversed(included_reversed))
        omitted_before_window = max(0, len(turns) - len(considered))
        audit = {
            "invalidated_turn_count": sum(bool(turn.metadata.get("context_invalidated")) for turn in turns),
            "total_turns_available": len(turns),
            "included_turn_count": len(included),
            "omitted_turn_count": omitted_before_window + omitted_for_budget,
            "omitted_before_window": omitted_before_window,
            "omitted_for_budget": omitted_for_budget,
            "estimated_recent_conversation_tokens": estimated_tokens,
            "history_query_contract_reference_count": referenced_query_contract_count,
        }
        return included, audit


@dataclass(slots=True)
class ConversationContextBundle:
    """封装裁剪后的近期对话、摘要检查点和上下文预算审计信息。"""

    task_context: ConversationTaskContext | None
    checkpoint: ConversationCheckpoint | None
    recent_conversation: list[dict[str, Any]]
    owner_conversation_context: dict[str, Any]
    audit: dict[str, Any]


def build_conversation_context(
    store: AgentStore,
    message: UserMessage,
    *,
    trace_id: str,
    previous_tool_results: list[ToolResult] | None,
    packing_policy: ContextPackingPolicy,
    shared_query_contract: dict[str, Any] | None = None,
) -> ConversationContextBundle:
    """构建当前任务的近期对话投影。

    顺序：确定 ``current_task_context`` -> 选择该任务轮次 -> 去重同 trace 工具轮 ->
    排除 checkpoint 已覆盖轮次 -> 按 token 预算打包。本函数只读 Store，不做意图判断。
    参数名是 ``previous_tool_results``；当前由 ``AgentContextBuilder`` 把整轮
    ``turn_tool_evidence`` 传入，避免同一工具证据重复进入消息历史。
    """

    task_context = store.current_task_context(message.conversation_id, message.sender_id)
    conversation_checkpoint = store.get_conversation_checkpoint(message.conversation_id)
    checkpoint = conversation_checkpoint
    recent_conversation_turns = store.recent_turns(
        message.conversation_id,
        packing_policy.max_turns_considered,
    )
    raw_turns = recent_conversation_turns
    omitted_before_task_context = 0
    checkpoint_excluded_by_task_context = False
    if task_context is not None:
        # 必须先按任务查询，再施加会话级数量上限；否则活跃群聊
        # 可能把未来预约挤出最近 60 条消息。
        task_turns = store.task_context_turns(
            message.conversation_id,
            task_context.task_context_id,
            packing_policy.max_turns_considered,
        )
        known_turn_keys = {_turn_identity(turn) for turn in task_turns}
        for turn in recent_conversation_turns:
            turn_task_context_id = str(turn.metadata.get("task_context_id") or "")
            belongs_to_current = turn_task_context_id == task_context.task_context_id
            legacy_inside_window = not turn_task_context_id and turn.occurred_at >= task_context.started_at
            if legacy_inside_window and _turn_identity(turn) not in known_turn_keys:
                task_turns.append(turn)
                known_turn_keys.add(_turn_identity(turn))
            elif not belongs_to_current:
                omitted_before_task_context += 1
        raw_turns = sorted(task_turns, key=lambda turn: turn.occurred_at)
        task_checkpoint = store.get_task_context_checkpoint(task_context.task_context_id)
        checkpoint = task_checkpoint or conversation_checkpoint
        checkpoint_matches = checkpoint is not None and (
            checkpoint.task_context_id == task_context.task_context_id
            or (
                checkpoint.task_context_id is None
                and checkpoint.updated_at >= task_context.started_at
            )
        )
        if checkpoint is not None and not checkpoint_matches:
            checkpoint = None
            checkpoint_excluded_by_task_context = True

    deduplicated_current_trace_tool_turns = 0
    if previous_tool_results:
        retained_turns: list[ConversationTurn] = []
        for turn in raw_turns:
            if turn.role.value == "tool" and turn.trace_id == trace_id:
                deduplicated_current_trace_tool_turns += 1
                continue
            retained_turns.append(turn)
        raw_turns = retained_turns

    checkpoint_covered_turn_count = 0
    if checkpoint is not None:
        turns_after_checkpoint = [turn for turn in raw_turns if _turn_after_checkpoint(turn, checkpoint.updated_at)]
        checkpoint_covered_turn_count = max(0, len(raw_turns) - len(turns_after_checkpoint))
        raw_turns = turns_after_checkpoint

    recent_conversation, audit = packing_policy.pack_turns(
        raw_turns, shared_query_contract=shared_query_contract,
    )
    owner_conversation_context = _owner_conversation_context(
        message, recent_conversation_turns, recent_conversation, trace_id=trace_id,
    )
    audit = {
        **audit,
        "total_turns_available": (
            audit["total_turns_available"]
            + checkpoint_covered_turn_count
            + omitted_before_task_context
        ),
        "omitted_turn_count": (
            audit["omitted_turn_count"]
            + checkpoint_covered_turn_count
            + omitted_before_task_context
        ),
        "omitted_covered_by_checkpoint": checkpoint_covered_turn_count,
        "omitted_before_task_context": omitted_before_task_context,
        "deduplicated_current_trace_tool_turn_count": deduplicated_current_trace_tool_turns,
        "conversation_checkpoint_present": checkpoint is not None,
        "checkpoint_excluded_by_task_context": checkpoint_excluded_by_task_context,
        "conversation_checkpoint_source_trace_id": checkpoint.source_trace_id if checkpoint else None,
        "checkpoint_covered_turn_count": checkpoint_covered_turn_count,
    }
    return ConversationContextBundle(
        task_context=task_context,
        checkpoint=checkpoint,
        recent_conversation=recent_conversation,
        owner_conversation_context=owner_conversation_context,
        audit=audit,
    )


def _owner_conversation_context(
    message: UserMessage,
    recent_turns: list[ConversationTurn],
    task_history: list[dict[str, Any]],
    *,
    trace_id: str,
) -> dict[str, Any]:
    """补回同一私聊的近期人工交流，保留独立的业务任务事实与工具证据边界。"""
    metadata = message.metadata
    if (metadata.get("channel") != "agent-wechat" or not message.sender_id
            or metadata.get("is_room") or metadata.get("room_id") or metadata.get("internal_event")
            or metadata.get("input_source") == "system_trigger"
            or message.conversation_id != f"wechaty:contact:{message.sender_id}"):
        return {}

    human_turns = []
    for turn in recent_turns:
        meta = turn.metadata
        if (meta.get("channel") != "agent-wechat" or meta.get("context_invalidated")
                or turn.trace_id == trace_id or turn.occurred_at > message.sent_at):
            continue
        if turn.role.value == "assistant" and meta.get("reply_author") == "owner" and meta.get("delivery_status") == "delivered":
            human_turns.append(("owner", turn))
        elif (turn.role.value == "user" and turn.sender_id == message.sender_id
              and not meta.get("internal_event") and meta.get("input_source") != "system_trigger"):
            human_turns.append(("customer", turn))
    human_turns = human_turns[-8:]
    latest_owner_index = next(
        (index for index in range(len(human_turns) - 1, -1, -1) if human_turns[index][0] == "owner"),
        None,
    )
    if latest_owner_index is None:
        return {}
    latest_owner = human_turns[latest_owner_index][1]
    preceding_customer = next(
        (turn for speaker, turn in reversed(human_turns[:latest_owner_index]) if speaker == "customer"),
        None,
    )
    included_traces = {item["trace_id"] for item in task_history}
    # 即使老板的“可以”已经入窗，前面的客户问题被裁掉时也需要补回这段交流。
    if latest_owner.trace_id in included_traces and (
        preceding_customer is None or preceding_customer.trace_id in included_traces
    ):
        return {}
    return {"turns": [
        {"speaker": speaker, "text": context_text_preview(turn.content, 500), "sent_at": turn.occurred_at.isoformat()}
        for speaker, turn in human_turns
    ]}


def _turn_after_checkpoint(turn: ConversationTurn, checkpoint_at: datetime) -> bool:
    """延迟同步的店主回复未被旧摘要读到，不能只按手机发送时间裁掉。"""
    if turn.occurred_at > checkpoint_at:
        return True
    recorded_at = turn.metadata.get("owner_reply_recorded_at")
    return turn.metadata.get("reply_author") == "owner" and isinstance(recorded_at, str) and datetime.fromisoformat(recorded_at) > checkpoint_at


def _turn_identity(turn: ConversationTurn) -> tuple[str, str, str, str]:
    return (
        turn.trace_id,
        turn.role.value,
        turn.occurred_at.isoformat(),
        turn.content,
    )


__all__ = [
    "ContextPackingPolicy",
    "ConversationContextBundle",
    "build_conversation_context",
]
