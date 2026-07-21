"""Conversation history, checkpoint, and reference persistence contracts."""

from __future__ import annotations

from typing import Any, Protocol

from ..models import (
    ConversationCheckpoint,
    ConversationTaskContext,
    ConversationTurn,
    MessageReference,
    StateTransition,
    UserMessage,
)


class ConversationStore(Protocol):
    """Persistence operations scoped by conversation and task context."""

    def append_user_turn(self, message: UserMessage, trace_id: str) -> None: ...

    def append_assistant_turn(
        self,
        conversation_id: str,
        text: str,
        trace_id: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None: ...

    def append_tool_turn(self, conversation_id: str, text: str, trace_id: str) -> None: ...

    def append_turn(self, conversation_id: str, turn: ConversationTurn) -> None: ...

    def recent_turns(self, conversation_id: str, limit: int = 30) -> list[ConversationTurn]: ...

    def get_conversation_checkpoint(self, conversation_id: str) -> ConversationCheckpoint | None: ...

    def upsert_conversation_checkpoint(
        self,
        conversation_id: str,
        summary: str,
        facts: dict[str, Any],
        open_questions: list[str],
        trace_id: str,
        *,
        task_context_id: str | None = None,
    ) -> tuple[ConversationCheckpoint, StateTransition]: ...

    def current_task_context(
        self,
        conversation_id: str,
        customer_id: str,
    ) -> ConversationTaskContext | None: ...

    def latest_task_context(self, conversation_id: str) -> ConversationTaskContext | None: ...

    def activate_task_context(
        self,
        *,
        conversation_id: str,
        customer_id: str,
        trace_id: str,
        reset_reason: str,
        force_new: bool = False,
    ) -> tuple[ConversationTaskContext, list[StateTransition]]: ...

    def conversation_version(self, conversation_id: str) -> int: ...

    def advance_conversation_version(
        self,
        conversation_id: str,
        *,
        trace_id: str,
        reason: str,
    ) -> tuple[int, StateTransition]: ...

    def supersede_pending_outputs(
        self,
        *,
        conversation_id: str,
        trace_id: str,
        reason: str,
        before_version: int | None = None,
    ) -> list[StateTransition]: ...

    def register_message_reference(self, reference: MessageReference) -> None: ...

    def link_message_reference(
        self,
        *,
        message_id: str,
        conversation_id: str,
        business_ref_type: str,
        business_ref_id: str,
        text: str = "",
        channel: str | None = None,
        sender_id: str | None = None,
        sender_name: str | None = None,
        recipient_id: str | None = None,
        recipient_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MessageReference: ...

    def resolve_message_reference(
        self,
        *,
        conversation_id: str,
        message_id: str,
    ) -> MessageReference | None: ...

