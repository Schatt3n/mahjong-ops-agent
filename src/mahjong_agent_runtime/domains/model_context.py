"""Domain rules for model context."""

from __future__ import annotations

from typing import Any
from ..models import (
    CustomerProfile,
    Game,
    InviteDraft,
    OutboundMessageDraft,
)

def customer_visible_name(customers: dict[str, CustomerProfile], customer_id: str, fallback: str | None = None) -> str:
    profile = customers.get(str(customer_id or ""))
    if profile:
        return profile.visible_name()
    return str(fallback or customer_id or "")

def game_for_model_context(game: Game, customers: dict[str, CustomerProfile]) -> dict[str, Any]:
    payload = game.to_dict()
    payload["organizer_name"] = customer_visible_name(customers, game.organizer_id, game.organizer_name)
    _rewrite_contact_names(payload, customers)
    return payload

def invite_draft_for_model_context(draft: InviteDraft, customers: dict[str, CustomerProfile]) -> dict[str, Any]:
    payload = draft.to_dict()
    payload["display_name"] = customer_visible_name(customers, draft.customer_id, draft.display_name)
    payload["metadata"] = visible_draft_metadata(payload.get("metadata"))
    return payload

def outbound_message_draft_for_model_context(
    draft: OutboundMessageDraft,
    customers: dict[str, CustomerProfile],
) -> dict[str, Any]:
    payload = draft.to_dict()
    payload["recipient_name"] = customer_visible_name(customers, draft.recipient_id, draft.recipient_name)
    payload["metadata"] = visible_draft_metadata(payload.get("metadata"))
    return payload

def visible_draft_metadata(metadata: Any) -> dict[str, Any]:
    metadata = dict(metadata or {}) if isinstance(metadata, dict) else {}
    return {
        key: value
        for key, value in metadata.items()
        if key in {"source", "game_id", "purpose", "channel"}
    }

def _rewrite_contact_names(payload: dict[str, Any], customers: dict[str, CustomerProfile]) -> None:
    for item in payload.get("participants") or []:
        if isinstance(item, dict):
            item["display_name"] = customer_visible_name(customers, str(item.get("customer_id") or ""), item.get("display_name"))
    for item in payload.get("parties") or []:
        if isinstance(item, dict):
            item["contact_name"] = customer_visible_name(customers, str(item.get("contact_id") or ""), item.get("contact_name"))
    for item in payload.get("seat_claims") or []:
        if isinstance(item, dict):
            item["contact_name"] = customer_visible_name(customers, str(item.get("contact_id") or ""), item.get("contact_name"))
    requirement = payload.get("requirement")
    if not isinstance(requirement, dict):
        return
    requester = requirement.get("requesting_party")
    if isinstance(requester, dict):
        requester["contact_name"] = customer_visible_name(
            customers,
            str(requester.get("contact_id") or requester.get("customer_id") or ""),
            requester.get("contact_name") or requester.get("display_name"),
        )
    for item in requirement.get("seat_claims") or []:
        if isinstance(item, dict):
            item["contact_name"] = customer_visible_name(
                customers,
                str(item.get("contact_id") or item.get("customer_id") or ""),
                item.get("contact_name") or item.get("display_name"),
            )
