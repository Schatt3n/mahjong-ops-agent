"""Build a bounded, task-scoped conversation window."""

from __future__ import annotations

from dataclasses import dataclass
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
from .tool_results import turn_payload_for_context


@dataclass(slots=True)
class ContextPackingPolicy:
    max_turns_considered: int = 60
    max_recent_conversation_tokens: int = 4_000

    def pack_turns(self, turns: list[ConversationTurn]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Pack newest turns into the local recent-conversation token budget."""

        considered = list(turns)[-self.max_turns_considered :]
        included_reversed: list[dict[str, Any]] = []
        estimated_tokens = 0
        omitted_for_budget = 0
        for turn in reversed(considered):
            payload = turn_payload_for_context(turn)
            turn_tokens = estimate_tokens(payload)
            if included_reversed and estimated_tokens + turn_tokens > self.max_recent_conversation_tokens:
                omitted_for_budget += 1
                continue
            included_reversed.append(payload)
            estimated_tokens += turn_tokens
        included = list(reversed(included_reversed))
        omitted_before_window = max(0, len(turns) - len(considered))
        audit = {
            "total_turns_available": len(turns),
            "included_turn_count": len(included),
            "omitted_turn_count": omitted_before_window + omitted_for_budget,
            "omitted_before_window": omitted_before_window,
            "omitted_for_budget": omitted_for_budget,
            "estimated_recent_conversation_tokens": estimated_tokens,
        }
        return included, audit


@dataclass(slots=True)
class ConversationContextBundle:
    task_context: ConversationTaskContext | None
    checkpoint: ConversationCheckpoint | None
    recent_conversation: list[dict[str, Any]]
    audit: dict[str, Any]


def build_conversation_context(
    store: AgentStore,
    message: UserMessage,
    *,
    trace_id: str,
    previous_tool_results: list[ToolResult] | None,
    packing_policy: ContextPackingPolicy,
) -> ConversationContextBundle:
    """Apply task boundaries, checkpoints, de-duplication, and token packing."""

    task_context = store.current_task_context(message.conversation_id, message.sender_id)
    checkpoint = store.get_conversation_checkpoint(message.conversation_id)
    raw_turns = store.recent_turns(message.conversation_id, packing_policy.max_turns_considered)
    omitted_before_task_context = 0
    checkpoint_excluded_by_task_context = False
    if task_context is not None:
        task_turns: list[ConversationTurn] = []
        for turn in raw_turns:
            turn_task_context_id = str(turn.metadata.get("task_context_id") or "")
            belongs_to_current = turn_task_context_id == task_context.task_context_id
            legacy_inside_window = not turn_task_context_id and turn.occurred_at >= task_context.started_at
            if belongs_to_current or legacy_inside_window:
                task_turns.append(turn)
            else:
                omitted_before_task_context += 1
        raw_turns = task_turns
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
        turns_after_checkpoint = [turn for turn in raw_turns if turn.occurred_at > checkpoint.updated_at]
        checkpoint_covered_turn_count = max(0, len(raw_turns) - len(turns_after_checkpoint))
        raw_turns = turns_after_checkpoint

    recent_conversation, audit = packing_policy.pack_turns(raw_turns)
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
        audit=audit,
    )


__all__ = [
    "ContextPackingPolicy",
    "ConversationContextBundle",
    "build_conversation_context",
]
