"""JSON-like schemas shared by the built-in tool registry."""

from __future__ import annotations

requirement_schema = {"type": "object", "additionalProperties": True}
non_empty_string = {"type": "string", "minLength": 1}
known_player_schema = {
    "type": "object",
    "required": ["customer_id", "display_name"],
    "additionalProperties": True,
    "properties": {
        "customer_id": non_empty_string,
        "display_name": non_empty_string,
        "status": {"type": "string", "enum": ["joined", "confirmed"]},
        "source": {"type": "string"},
        "seat_count": {"type": "integer", "minimum": 1, "maximum": 4},
        "known_member_ids": {"type": "array", "items": {"type": "string"}},
        "anonymous_seat_count": {"type": "integer", "minimum": 0, "maximum": 4},
    },
}
requesting_party_schema = {
    "type": "object",
    "required": ["contact_id", "contact_name", "seat_count"],
    "additionalProperties": True,
    "properties": {
        "contact_id": non_empty_string,
        "contact_name": non_empty_string,
        "seat_count": {"type": "integer", "minimum": 1, "maximum": 4},
        "known_member_ids": {"type": "array", "items": {"type": "string"}},
        "anonymous_seat_count": {"type": "integer", "minimum": 0, "maximum": 4},
        "source": {"type": "string"},
    },
}
invitation_schema = {
    "type": "object",
    "required": ["customer_id", "display_name", "message_text"],
    "additionalProperties": True,
    "properties": {
        "customer_id": non_empty_string,
        "display_name": non_empty_string,
        "message_text": non_empty_string,
        "metadata": {"type": "object", "additionalProperties": True},
    },
}
outbound_message_draft_schema = {
    "type": "object",
    "required": ["recipient_id", "recipient_name", "channel", "message_text", "purpose"],
    "additionalProperties": False,
    "properties": {
        "recipient_id": non_empty_string,
        "recipient_name": non_empty_string,
        "channel": non_empty_string,
        "message_text": non_empty_string,
        "purpose": non_empty_string,
        "metadata": {"type": "object", "additionalProperties": True},
    },
}
checkpoint_schema = {
    "type": "object",
    "required": ["summary"],
    "additionalProperties": False,
    "properties": {
        "summary": non_empty_string,
        "facts": {"type": "object", "additionalProperties": True},
        "open_questions": {"type": "array", "items": non_empty_string},
    },
}
badcase_schema = {
    "type": "object",
    "required": ["reason", "input", "actual", "expected"],
    "additionalProperties": True,
    "properties": {
        "reason": non_empty_string,
        "input": {"type": "object", "additionalProperties": True},
        "actual": {"type": "object", "additionalProperties": True},
        "expected": {"type": "object", "additionalProperties": True},
        "tags": {"type": "array", "items": non_empty_string},
        "metadata": {"type": "object", "additionalProperties": True},
    },
}
memory_item_schema = {
    "type": "object",
    "required": ["customer_id", "memory_type", "field", "value", "evidence", "confidence"],
    "additionalProperties": True,
    "properties": {
        "customer_id": non_empty_string,
        "memory_type": non_empty_string,
        "field": {
            **non_empty_string,
            "description": (
                "稳定的结构化字段名。时长上限使用 max_duration_hours，明确约定时长使用 duration_hours；"
                "避免同一语义在不同轮次使用不同字段名。"
            ),
        },
        "value": {},
        "target_customer_id": {"type": "string"},
        "target_customer_name": {"type": "string"},
        "operation": {"type": "string"},
        "evidence": non_empty_string,
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "risk_level": {"type": "string", "enum": ["low", "medium", "high"]},
        "scope": {"type": "string", "enum": ["current_task", "session", "today", "long_term"]},
        "metadata": {"type": "object", "additionalProperties": True},
    },
}
memory_schema = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "task_memories": {"type": "array", "items": memory_item_schema},
        "pending_long_term_memories": {"type": "array", "items": memory_item_schema},
    },
}
