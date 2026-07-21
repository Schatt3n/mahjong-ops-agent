"""Resolve quoted messages and construct their interpretation contract."""

from __future__ import annotations

from typing import Any

from ...models import UserMessage
from ...stores import AgentStore
from ..model_context import customer_visible_name
from .sanitization import (
    message_reference_for_context,
    sanitize_quoted_message_metadata_for_context,
)


def resolve_quoted_message_context(
    store: AgentStore,
    message: UserMessage,
    current_message: dict[str, Any],
) -> dict[str, Any] | None:
    """Resolve a quoted platform message to an authoritative business object."""

    quoted = message.quoted_message
    if quoted is None or not quoted.message_id:
        return None
    resolver = getattr(store, "resolve_message_reference", None)
    if not callable(resolver):
        return None
    reference = resolver(
        conversation_id=quoted.conversation_id or message.conversation_id,
        message_id=quoted.message_id,
    )
    if reference is None:
        return None
    reference_payload = message_reference_for_context(reference, store.customers)
    quoted_payload = dict(current_message.get("quoted_message") or quoted.to_dict())
    quoted_payload["business_ref_type"] = quoted_payload.get("business_ref_type") or reference.business_ref_type
    quoted_payload["business_ref_id"] = quoted_payload.get("business_ref_id") or reference.business_ref_id
    quoted_payload["conversation_id"] = quoted_payload.get("conversation_id") or reference.conversation_id
    quoted_payload["text"] = quoted_payload.get("text") or reference.text
    if reference.sender_id:
        quoted_payload["sender_name"] = customer_visible_name(
            store.customers,
            reference.sender_id,
            quoted_payload.get("sender_name") or reference.sender_name,
        )
    quoted_payload["metadata"] = {
        **dict(quoted_payload.get("metadata") or {}),
        "resolved_message_reference": {
            "business_ref_type": reference.business_ref_type,
            "business_ref_id": reference.business_ref_id,
            "channel": reference.channel,
            "recipient_id": reference.recipient_id,
            "recipient_name": customer_visible_name(
                store.customers,
                reference.recipient_id or "",
                reference.recipient_name,
            ),
            "source": reference.metadata.get("source"),
        },
    }
    quoted_payload["metadata"] = sanitize_quoted_message_metadata_for_context(quoted_payload.get("metadata"))
    current_message["quoted_message"] = quoted_payload
    return reference_payload


def build_message_reference_contract(
    message: UserMessage,
    quoted_message_context: dict[str, Any] | None,
) -> tuple[dict[str, Any], str]:
    """Return the model contract and auditable resolution status for a quote."""

    quoted_message = message.quoted_message
    quoted_message_present = quoted_message is not None
    has_provided_business_ref = bool(
        quoted_message
        and (
            quoted_message.business_ref_type
            or quoted_message.business_ref_id
        )
    )
    reference_status = "absent"
    if quoted_message_present:
        reference_status = "resolved" if quoted_message_context is not None else "unresolved"
        if quoted_message_context is None and has_provided_business_ref:
            reference_status = "provided_business_ref"
    contract = {
        "primary_binding": "quoted_message" if quoted_message_present else "current_message",
        "quoted_message_present": quoted_message_present,
        "business_reference_status": reference_status,
        "business_reference_resolved": bool(
            quoted_message_context is not None or has_provided_business_ref
        ),
        "interpretation_instruction": (
            "Interpret the current reply against current_message.quoted_message before recent_conversation or active_games."
            if quoted_message_present
            else "Interpret the current reply from current_message, then use recent context only to resolve omissions."
        ),
        "state_write_instruction": (
            "The quote has no authoritative business reference. Do not infer a state-changing acceptance, rejection, "
            "arrival, cancellation, or participant update solely from this short reply plus a nearby active game. "
            "Resolve the business object with a read tool or ask the user before a write."
            if quoted_message_present
            and quoted_message_context is None
            and not has_provided_business_ref
            else "Any state write must still be supported by the current message and authoritative business state."
        ),
    }
    return contract, reference_status


__all__ = [
    "build_message_reference_contract",
    "resolve_quoted_message_context",
]
