from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .workflow_models import EntityType, GameWorkflowStatus


_ALLOWED_KINDS: frozenset[str] = frozenset(
    {
        "create_game",
        "close_game",
        "record_seat_acceptance",
    }
)
_ALLOWED_KEYS_BY_KIND: dict[str, frozenset[str]] = {
    "create_game": frozenset(
        {
            "kind",
            "entity_type",
            "entity_id",
            "target_status",
            "enter_negotiating_if_outbox_created",
            "reason",
            "requirement",
        }
    ),
    "close_game": frozenset({"kind", "entity_type", "entity_id", "target_status", "reason", "requirement"}),
    "record_seat_acceptance": frozenset(
        {
            "kind",
            "entity_type",
            "entity_id",
            "target_status",
            "reason",
            "requirement",
            "participant",
            "seat_delta",
        }
    ),
}
_ALLOWED_TARGETS_BY_KIND: dict[str, frozenset[GameWorkflowStatus]] = {
    "create_game": frozenset({GameWorkflowStatus.OPEN, GameWorkflowStatus.NEGOTIATING}),
    "close_game": frozenset(
        {
            GameWorkflowStatus.CANCELLED,
            GameWorkflowStatus.EXPIRED,
            GameWorkflowStatus.COMPLETED,
        }
    ),
    "record_seat_acceptance": frozenset({GameWorkflowStatus.NEGOTIATING, GameWorkflowStatus.CONFIRMED}),
}


@dataclass(frozen=True, slots=True)
class StateWriteIntent:
    kind: str
    entity_type: str
    entity_id: str
    target_status: str
    reason: str
    requirement: dict[str, Any]
    participant: dict[str, Any] | None = None
    seat_delta: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "kind": self.kind,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "target_status": self.target_status,
            "reason": self.reason,
            "requirement": dict(self.requirement),
        }
        if self.participant is not None:
            payload["participant"] = dict(self.participant)
        if self.seat_delta is not None:
            payload["seat_delta"] = dict(self.seat_delta)
        payload.update(self.metadata)
        return payload


def parse_state_write_intent(raw: Any) -> tuple[StateWriteIntent | None, list[str]]:
    errors = validate_state_write_intent_contract(raw)
    if errors:
        return None, errors
    assert isinstance(raw, dict)
    metadata = {
        key: value
        for key, value in raw.items()
        if key
        not in {
            "kind",
            "entity_type",
            "entity_id",
            "target_status",
            "reason",
            "requirement",
            "participant",
            "seat_delta",
        }
    }
    return (
        StateWriteIntent(
            kind=str(raw["kind"]).strip(),
            entity_type=str(raw["entity_type"]).strip(),
            entity_id=str(raw["entity_id"]).strip(),
            target_status=str(raw["target_status"]).strip(),
            reason=str(raw["reason"]).strip(),
            requirement=dict(raw["requirement"]),
            participant=dict(raw["participant"]) if isinstance(raw.get("participant"), dict) else None,
            seat_delta=dict(raw["seat_delta"]) if isinstance(raw.get("seat_delta"), dict) else None,
            metadata=metadata,
        ),
        [],
    )


def validate_state_write_intent_contract(raw: Any) -> list[str]:
    if not isinstance(raw, dict):
        return ["state_write_intent must be an object"]

    errors: list[str] = []
    kind = str(raw.get("kind") or "").strip()
    if kind not in _ALLOWED_KINDS:
        errors.append(f"state_write_intent.kind invalid {kind!r}")

    allowed_keys = _ALLOWED_KEYS_BY_KIND.get(kind, frozenset())
    for key in sorted(raw):
        if key not in allowed_keys:
            errors.append(f"state_write_intent.{key} is not allowed for {kind or '<unknown>'}")

    entity_type = str(raw.get("entity_type") or "").strip()
    if entity_type != EntityType.GAME.value:
        errors.append(f"state_write_intent.entity_type invalid {entity_type!r}")

    if not _is_non_empty_string(raw.get("entity_id")):
        errors.append("state_write_intent.entity_id must be a non-empty string")

    target_status_raw = str(raw.get("target_status") or "").strip()
    try:
        target_status = GameWorkflowStatus(target_status_raw)
    except ValueError:
        errors.append(f"state_write_intent.target_status invalid {target_status_raw!r}")
    else:
        if kind in _ALLOWED_TARGETS_BY_KIND and target_status not in _ALLOWED_TARGETS_BY_KIND[kind]:
            errors.append(f"state_write_intent.target_status {target_status.value!r} is not allowed for {kind}")

    if not _is_non_empty_string(raw.get("reason")):
        errors.append("state_write_intent.reason must be a non-empty string")

    if not isinstance(raw.get("requirement"), dict):
        errors.append("state_write_intent.requirement must be an object")

    if kind == "record_seat_acceptance":
        if not isinstance(raw.get("participant"), dict):
            errors.append("state_write_intent.participant must be an object for record_seat_acceptance")
        if not isinstance(raw.get("seat_delta"), dict):
            errors.append("state_write_intent.seat_delta must be an object for record_seat_acceptance")

    return errors


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())
