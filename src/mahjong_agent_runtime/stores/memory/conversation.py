"""InMemory conversation store operations."""

from __future__ import annotations

from typing import Any
from datetime import datetime
from ...models import (
    ConversationCheckpoint,
    ConversationRole,
    ConversationTaskContext,
    ConversationTurn,
    InviteStatus,
    OutboundDraftStatus,
    StateTransition,
    new_id,
    now,
)

class InMemoryConversationStoreMixin:
    """Backend-specific operations extracted from the compatibility store."""

    __slots__ = ()

    def append_user_turn(self, message, trace_id: str) -> None:
        task_context = self.current_task_context(message.conversation_id, message.sender_id)
        metadata = dict(getattr(message, "metadata", {}) or {})
        if task_context is not None:
            metadata["task_context_id"] = task_context.task_context_id
        self.append_turn(
            message.conversation_id,
            ConversationTurn(
                role=ConversationRole.USER,
                content=message.text,
                trace_id=trace_id,
                sender_id=message.sender_id,
                sender_name=message.sender_name,
                metadata=metadata,
                occurred_at=message.sent_at,
            ),
        )

    def append_assistant_turn(
        self,
        conversation_id: str,
        text: str,
        trace_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        task_context = self.latest_task_context(conversation_id)
        turn_metadata = dict(metadata or {})
        if task_context is not None:
            turn_metadata.setdefault("task_context_id", task_context.task_context_id)
        self.append_turn(
            conversation_id,
            ConversationTurn(
                role=ConversationRole.ASSISTANT,
                content=text,
                trace_id=trace_id,
                metadata=turn_metadata,
            ),
        )

    def append_tool_turn(self, conversation_id: str, text: str, trace_id: str) -> None:
        task_context = self.latest_task_context(conversation_id)
        metadata = {"task_context_id": task_context.task_context_id} if task_context else {}
        self.append_turn(
            conversation_id,
            ConversationTurn(role=ConversationRole.TOOL, content=text, trace_id=trace_id, metadata=metadata),
        )

    def append_turn(self, conversation_id: str, turn: ConversationTurn) -> None:
        with self._lock:
            self.turns.setdefault(conversation_id, []).append(turn)

    def recent_turns(self, conversation_id: str, limit: int = 30) -> list[ConversationTurn]:
        with self._lock:
            return list(self.turns.get(conversation_id, []))[-int(limit):]

    def get_conversation_checkpoint(self, conversation_id: str) -> ConversationCheckpoint | None:
        with self._lock:
            return self.conversation_checkpoints.get(conversation_id)

    def current_task_context(self, conversation_id: str, customer_id: str) -> ConversationTaskContext | None:
        with self._lock:
            matches = [
                item
                for item in self.task_contexts.values()
                if item.status == "active"
                and item.conversation_id == conversation_id
                and item.customer_id == customer_id
            ]
            return max(matches, key=lambda item: item.updated_at) if matches else None

    def latest_task_context(self, conversation_id: str) -> ConversationTaskContext | None:
        with self._lock:
            matches = [
                item
                for item in self.task_contexts.values()
                if item.status == "active" and item.conversation_id == conversation_id
            ]
            return max(matches, key=lambda item: item.updated_at) if matches else None

    def activate_task_context(
        self,
        *,
        conversation_id: str,
        customer_id: str,
        trace_id: str,
        activity_at: datetime,
        started_at: datetime,
        reason: str,
        force_new: bool,
        archive_previous: bool,
    ) -> tuple[ConversationTaskContext, list[StateTransition]]:
        """Create/reuse one business episode and retire its temporary memory on reset."""

        with self._lock:
            transitions: list[StateTransition] = []
            previous = self.current_task_context(conversation_id, customer_id)
            if previous is not None and not force_new:
                previous.updated_at = activity_at
                previous.source_trace_id = trace_id
                return previous, transitions

            if previous is not None:
                previous.status = "closed"
                previous.closed_at = activity_at
                previous.updated_at = activity_at
                transitions.append(
                    StateTransition(
                        "task_context",
                        previous.task_context_id,
                        "active",
                        "closed",
                        reason,
                        trace_id,
                    )
                )

            if archive_previous:
                previous_context_id = previous.task_context_id if previous else None
                for memory in self.task_memories.values():
                    if memory.status != "active" or memory.conversation_id != conversation_id:
                        continue
                    if memory.customer_id != customer_id and memory.target_customer_id != customer_id:
                        continue
                    memory_context_id = str(memory.metadata.get("task_context_id") or "")
                    belongs_to_previous = bool(previous_context_id and memory_context_id == previous_context_id)
                    predates_new_context = memory.updated_at < started_at
                    if not belongs_to_previous and not predates_new_context:
                        continue
                    memory.status = "archived"
                    memory.updated_at = activity_at
                    transitions.append(
                        StateTransition(
                            "task_memory",
                            memory.memory_id,
                            "active",
                            "archived",
                            "task_context_reset",
                            trace_id,
                        )
                    )

            context = ConversationTaskContext(
                task_context_id=new_id("task_context"),
                conversation_id=conversation_id,
                customer_id=customer_id,
                reset_reason=reason,
                previous_task_context_id=previous.task_context_id if previous else None,
                source_trace_id=trace_id,
                started_at=started_at,
                updated_at=activity_at,
            )
            self.task_contexts[context.task_context_id] = context
            transitions.append(
                StateTransition(
                    "task_context",
                    context.task_context_id,
                    None,
                    "active",
                    reason,
                    trace_id,
                )
            )
            self.transitions.extend(transitions)
            return context, transitions

    def conversation_version(self, conversation_id: str) -> int:
        with self._lock:
            return int(self.conversation_versions.get(conversation_id or "default", 0))

    def advance_conversation_version(
        self,
        conversation_id: str,
        *,
        trace_id: str,
        reason: str,
    ) -> tuple[int, StateTransition]:
        key = conversation_id or "default"
        with self._lock:
            old = int(self.conversation_versions.get(key, 0))
            new = old + 1
            self.conversation_versions[key] = new
            transition = StateTransition(
                entity_type="conversation_version",
                entity_id=key,
                from_status=str(old),
                to_status=str(new),
                reason=reason or "user_message_received",
                trace_id=trace_id,
            )
            self.transitions.append(transition)
            return new, transition

    def supersede_pending_outputs(
        self,
        conversation_id: str,
        *,
        sender_id: str | None = None,
        trace_id: str,
        reason: str,
    ) -> tuple[dict[str, int], list[StateTransition]]:
        key = conversation_id or "default"
        with self._lock:
            transitions: list[StateTransition] = []
            counts = {
                "invite_drafts": 0,
                "outbound_message_drafts": 0,
                "assistant_replies": 0,
            }
            game_ids = {game.game_id for game in self.games.values() if game.conversation_id == key}
            sender_is_pending_candidate = bool(
                sender_id
                and any(
                    draft.game_id in game_ids
                    and draft.customer_id == sender_id
                    and draft.status == InviteStatus.PENDING_APPROVAL
                    for draft in self.invite_drafts.values()
                )
            )
            for draft in self.invite_drafts.values():
                if sender_is_pending_candidate:
                    continue
                if draft.game_id not in game_ids or draft.status != InviteStatus.PENDING_APPROVAL:
                    continue
                old = draft.status.value
                draft.status = InviteStatus.SUPERSEDED
                draft.updated_at = now()
                draft.metadata = {
                    **dict(draft.metadata),
                    "superseded_by_trace_id": trace_id,
                    "superseded_reason": reason,
                }
                counts["invite_drafts"] += 1
                transitions.append(
                    StateTransition("invite_draft", draft.draft_id, old, draft.status.value, reason, trace_id)
                )
            for draft in self.outbound_message_drafts.values():
                if sender_is_pending_candidate:
                    continue
                if draft.conversation_id != key or draft.status != OutboundDraftStatus.PENDING_APPROVAL:
                    continue
                old = draft.status.value
                draft.status = OutboundDraftStatus.SUPERSEDED
                draft.updated_at = now()
                draft.metadata = {
                    **dict(draft.metadata),
                    "superseded_by_trace_id": trace_id,
                    "superseded_reason": reason,
                }
                counts["outbound_message_drafts"] += 1
                transitions.append(
                    StateTransition("outbound_message_draft", draft.draft_id, old, draft.status.value, reason, trace_id)
                )
            for turn in self.turns.get(key, []):
                if turn.role != ConversationRole.ASSISTANT:
                    continue
                if turn.metadata.get("delivery_status") != "pending_operator_send":
                    continue
                old = str(turn.metadata.get("delivery_status") or "")
                turn.metadata = {
                    **dict(turn.metadata),
                    "delivery_status": "superseded",
                    "superseded_by_trace_id": trace_id,
                    "superseded_reason": reason,
                }
                counts["assistant_replies"] += 1
                transitions.append(StateTransition("assistant_reply", turn.trace_id, old, "superseded", reason, trace_id))
            self.transitions.extend(transitions)
            return counts, transitions

    def upsert_conversation_checkpoint(
        self,
        *,
        conversation_id: str,
        summary: str,
        facts: dict[str, Any],
        open_questions: list[str],
        trace_id: str,
    ) -> tuple[ConversationCheckpoint, StateTransition]:
        with self._lock:
            previous = self.conversation_checkpoints.get(conversation_id)
            task_context = self.latest_task_context(conversation_id)
            checkpoint = ConversationCheckpoint(
                conversation_id=conversation_id,
                summary=summary,
                facts=dict(facts),
                open_questions=list(open_questions),
                task_context_id=task_context.task_context_id if task_context else None,
                source_trace_id=trace_id,
            )
            self.conversation_checkpoints[conversation_id] = checkpoint
            transition = StateTransition(
                entity_type="conversation_checkpoint",
                entity_id=conversation_id,
                from_status="exists" if previous else None,
                to_status="updated",
                reason="update_context_checkpoint",
                trace_id=trace_id,
            )
            self.transitions.append(transition)
            return checkpoint, transition
