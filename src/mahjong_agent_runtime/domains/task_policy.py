"""Domain rules for task policy."""

from __future__ import annotations

PENDING_INPUT_PROCESSING_LEASE_SECONDS = 120

SCHEDULED_TASK_PROCESSING_LEASE_SECONDS = 120

def pending_input_batch_key(conversation_id: str, sender_id: str) -> str:
    """Stable scope key; group members never share unfinished fragments."""

    return f"{conversation_id or 'default'}\x1f{sender_id or 'unknown'}"

def message_reference_key(conversation_id: str, message_id: str) -> str:
    return f"{str(conversation_id or '')}:{str(message_id or '')}"
