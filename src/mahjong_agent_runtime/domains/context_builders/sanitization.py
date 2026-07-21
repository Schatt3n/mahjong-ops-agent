"""Sanitize channel messages before they enter the model context."""

from __future__ import annotations

from typing import Any

from ...models import MessageReference
from ..model_context import customer_visible_name


SAFE_CONTEXT_MESSAGE_METADATA_KEYS = {
    "channel",
    "platform_name",
    "source",
    "message_type",
    "source_message_id",
    "is_room",
    "self_message",
    "has_text",
    "text_source",
    "modalities",
    "media_candidates",
    "raw_observation_summary",
    "media_requires_transcription",
    "media_requires_ocr",
    "transcript_confidence",
    "ocr_confidence",
    "language",
}

SAFE_CONTEXT_QUOTED_METADATA_KEYS = {
    "source",
    "raw_chatusr",
    "platform_message_id",
    "platformMessageId",
    "source_message_id",
    "sourceMessageId",
    "message_type",
    "text_source",
    "channel",
    "resolved_message_reference",
}


def message_reference_for_context(
    reference: MessageReference,
    customers: dict[str, Any],
) -> dict[str, Any]:
    """Expose only customer-visible names and safe reference metadata."""

    payload = reference.to_dict()
    payload["sender_name"] = customer_visible_name(customers, reference.sender_id or "", reference.sender_name)
    payload["recipient_name"] = customer_visible_name(
        customers,
        reference.recipient_id or "",
        reference.recipient_name,
    )
    payload["metadata"] = sanitize_quoted_message_metadata_for_context(payload.get("metadata"))
    return payload


def context_text_preview(value: Any, limit: int = 160) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def sanitize_context_media_candidates(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    sanitized: list[dict[str, Any]] = []
    for item in value[:12]:
        if not isinstance(item, dict):
            continue
        safe_item = {
            "path": context_text_preview(item.get("path"), 160),
            "kind": context_text_preview(item.get("kind"), 40),
            "value_type": context_text_preview(item.get("value_type"), 40),
        }
        text_preview = context_text_preview(item.get("text_preview"), 120)
        if text_preview:
            safe_item["text_preview"] = text_preview
        sanitized.append({key: val for key, val in safe_item.items() if val not in {"", None}})
    return sanitized


def sanitize_context_observation_summary(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    sanitized: dict[str, int] = {}
    for key in ("quote_candidate_count", "media_candidate_count"):
        try:
            sanitized[key] = max(int(value.get(key) or 0), 0)
        except (TypeError, ValueError):
            continue
    return sanitized


def sanitize_message_metadata_for_context(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}
    sanitized: dict[str, Any] = {}
    for key, value in metadata.items():
        if key not in SAFE_CONTEXT_MESSAGE_METADATA_KEYS:
            continue
        if key == "modalities":
            if isinstance(value, list):
                sanitized[key] = [
                    context_text_preview(item, 40)
                    for item in value[:12]
                    if str(item or "").strip()
                ]
            continue
        if key == "media_candidates":
            sanitized[key] = sanitize_context_media_candidates(value)
            continue
        if key == "raw_observation_summary":
            sanitized[key] = sanitize_context_observation_summary(value)
            continue
        if isinstance(value, bool) or value is None:
            sanitized[key] = value
        elif isinstance(value, (int, float)):
            sanitized[key] = value
        elif isinstance(value, str):
            sanitized[key] = context_text_preview(value, 160)
    return sanitized


def sanitize_resolved_message_reference_for_context(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    sanitized: dict[str, Any] = {}
    for key in ("business_ref_type", "business_ref_id", "channel", "recipient_id", "recipient_name", "source"):
        raw_value = value.get(key)
        if raw_value is None:
            sanitized[key] = None
        elif isinstance(raw_value, (int, float, bool)):
            sanitized[key] = raw_value
        else:
            sanitized[key] = context_text_preview(raw_value, 160)
    return sanitized


def sanitize_quoted_message_metadata_for_context(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}
    sanitized: dict[str, Any] = {}
    for key, value in metadata.items():
        if key not in SAFE_CONTEXT_QUOTED_METADATA_KEYS:
            continue
        if key == "resolved_message_reference":
            sanitized_reference = sanitize_resolved_message_reference_for_context(value)
            if sanitized_reference:
                sanitized[key] = sanitized_reference
            continue
        if isinstance(value, bool) or value is None:
            sanitized[key] = value
        elif isinstance(value, (int, float)):
            sanitized[key] = value
        elif isinstance(value, str):
            sanitized[key] = context_text_preview(value, 160)
    return sanitized


def sanitize_current_message_for_context(current_message: dict[str, Any]) -> dict[str, Any]:
    """Remove raw channel payloads while preserving model-relevant metadata."""

    sanitized = dict(current_message)
    sanitized["metadata"] = sanitize_message_metadata_for_context(sanitized.get("metadata"))
    quoted_message = sanitized.get("quoted_message")
    if isinstance(quoted_message, dict):
        quoted_payload = dict(quoted_message)
        quoted_payload["metadata"] = sanitize_quoted_message_metadata_for_context(quoted_payload.get("metadata"))
        sanitized["quoted_message"] = quoted_payload
    return sanitized


__all__ = [
    "SAFE_CONTEXT_MESSAGE_METADATA_KEYS",
    "SAFE_CONTEXT_QUOTED_METADATA_KEYS",
    "context_text_preview",
    "message_reference_for_context",
    "sanitize_context_media_candidates",
    "sanitize_context_observation_summary",
    "sanitize_current_message_for_context",
    "sanitize_message_metadata_for_context",
    "sanitize_quoted_message_metadata_for_context",
    "sanitize_resolved_message_reference_for_context",
]
