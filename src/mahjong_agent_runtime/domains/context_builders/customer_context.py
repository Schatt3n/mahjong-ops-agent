"""Build customer, task-memory, and draft context for one conversation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...models import ConversationTaskContext
from ...stores import AgentStore
from ..model_context import outbound_message_draft_for_model_context


@dataclass(slots=True)
class CustomerContextBundle:
    """Customer-specific facts safe to place in the internal model context."""

    profile: dict[str, Any] | None
    task_memories: list[dict[str, Any]]
    pending_memory_candidates: list[dict[str, Any]]
    outbound_message_drafts: list[dict[str, Any]]


def build_customer_context(
    store: AgentStore,
    *,
    conversation_id: str,
    sender_id: str,
    task_context: ConversationTaskContext | None,
) -> CustomerContextBundle:
    """Collect profile and episode-scoped memory without crossing conversations."""

    profile = store.customers.get(sender_id)
    task_memories = store.task_memory_context(conversation_id, sender_id)
    pending_memory_candidates = store.pending_memory_candidates_for_context(conversation_id, sender_id)
    outbound_message_drafts = [
        outbound_message_draft_for_model_context(item, store.customers)
        for item in store.outbound_message_drafts.values()
        if item.conversation_id == conversation_id or item.recipient_id == sender_id
        if task_context is None
        or item.metadata.get("task_context_id") == task_context.task_context_id
        or (
            not item.metadata.get("task_context_id")
            and item.created_at >= task_context.started_at
        )
    ]
    return CustomerContextBundle(
        profile=profile.to_model_context() if profile else None,
        task_memories=task_memories,
        pending_memory_candidates=pending_memory_candidates,
        outbound_message_drafts=outbound_message_drafts,
    )


def compact_candidate(candidate: Any) -> dict[str, Any]:
    if not isinstance(candidate, dict):
        return {}
    customer = candidate.get("customer") if isinstance(candidate.get("customer"), dict) else {}
    return {
        "score": candidate.get("score"),
        "reasons": candidate.get("reasons"),
        "warnings": candidate.get("warnings"),
        "relationship": candidate.get("relationship"),
        "customer": {
            "customer_id": customer.get("customer_id"),
            "display_name": customer.get("display_name"),
            "gender": customer.get("gender"),
            "preferred_games": customer.get("preferred_games"),
            "preferred_stakes": customer.get("preferred_stakes"),
            "preferred_time_tags": customer.get("preferred_time_tags"),
            "smoke_preference": customer.get("smoke_preference"),
            "response_score": customer.get("response_score"),
            "fatigue_score": customer.get("fatigue_score"),
            "no_contact": customer.get("no_contact"),
            "notes": customer.get("notes"),
        },
    }


def compact_draft(draft: Any) -> dict[str, Any]:
    if not isinstance(draft, dict):
        return {}
    return {
        "draft_id": draft.get("draft_id"),
        "game_id": draft.get("game_id"),
        "customer_id": draft.get("customer_id") or draft.get("recipient_id"),
        "display_name": draft.get("display_name") or draft.get("recipient_name"),
        "message_text": draft.get("message_text"),
        "status": draft.get("status"),
        "purpose": draft.get("purpose"),
        "channel": draft.get("channel"),
    }


__all__ = [
    "CustomerContextBundle",
    "build_customer_context",
    "compact_candidate",
    "compact_draft",
]
