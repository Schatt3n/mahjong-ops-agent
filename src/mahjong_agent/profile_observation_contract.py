from __future__ import annotations

from datetime import datetime
from typing import Any

from .models import DEFAULT_TZ
from .observability import to_trace_payload


PROFILE_OBSERVATION_FIELDS: frozenset[str] = frozenset(
    {
        "preferred_level",
        "preferred_game_type",
        "preferred_variant",
        "preferred_play_option",
        "smoke_preference",
        "usual_party_size",
        "usual_start_time",
        "duration_preference",
        "response_preference",
        "contact_preference",
        "fatigue_preference",
        "note",
    }
)
PROFILE_OBSERVATION_FIELD_ALIASES: dict[str, str] = {
    "stake_preference": "preferred_level",
    "stake_preferences": "preferred_level",
    "level_preference": "preferred_level",
    "level_preferences": "preferred_level",
    "game_type_preference": "preferred_game_type",
    "game_type_preferences": "preferred_game_type",
    "variant_preference": "preferred_variant",
    "variant_preferences": "preferred_variant",
    "play_option_preference": "preferred_play_option",
    "play_option_preferences": "preferred_play_option",
}
PROFILE_OBSERVATION_SOURCES: frozenset[str] = frozenset({"current_message", "context"})
PROFILE_OBSERVATION_RISKS: frozenset[str] = frozenset({"low", "medium"})


def canonical_profile_observation_field(value: Any) -> str:
    field = str(value or "").strip()
    return PROFILE_OBSERVATION_FIELD_ALIASES.get(field, field)


def validate_profile_observation_contract(raw: Any, *, index: int) -> list[str]:
    prefix = f"profile_observations[{index}]"
    if not isinstance(raw, dict):
        return [f"{prefix} must be an object"]

    errors: list[str] = []
    field = canonical_profile_observation_field(raw.get("field"))
    if field not in PROFILE_OBSERVATION_FIELDS:
        errors.append(f"{prefix}.field invalid {field!r}")

    value = raw.get("value")
    if to_trace_payload(value) in (None, "", [], {}):
        errors.append(f"{prefix}.value must be non-empty")

    confidence = raw.get("confidence")
    try:
        numeric_confidence = float(confidence)
    except (TypeError, ValueError):
        errors.append(f"{prefix}.confidence invalid {confidence!r}")
    else:
        if numeric_confidence < 0 or numeric_confidence > 1:
            errors.append(f"{prefix}.confidence out of range {confidence!r}")
        elif numeric_confidence < 0.65:
            errors.append(f"{prefix}.confidence below writable threshold {confidence!r}")

    source = str(raw.get("source") or "").strip()
    if source not in PROFILE_OBSERVATION_SOURCES:
        errors.append(f"{prefix}.source invalid {source!r}")

    evidence = str(raw.get("evidence") or "").strip()
    if not evidence:
        errors.append(f"{prefix}.evidence must be non-empty")

    risk = str(raw.get("risk") or "").strip().lower()
    if risk not in PROFILE_OBSERVATION_RISKS:
        errors.append(f"{prefix}.risk invalid {risk!r}")

    return errors


def normalize_profile_observation_for_storage(
    raw: Any,
    *,
    now: datetime | None = None,
) -> tuple[dict[str, Any], str | None]:
    errors = validate_profile_observation_contract(raw, index=0)
    if errors:
        return {}, _storage_rejection_reason(errors[0])
    assert isinstance(raw, dict)
    confidence = _safe_confidence(raw.get("confidence"), default=0.0)
    return {
        "field": canonical_profile_observation_field(raw.get("field")),
        "value": to_trace_payload(raw.get("value")),
        "confidence": confidence,
        "source": str(raw.get("source") or "").strip(),
        "evidence": str(raw.get("evidence") or "").strip()[:240],
        "risk": str(raw.get("risk") or "").strip().lower(),
        "created_at": (now or datetime.now(DEFAULT_TZ)).isoformat(),
    }, None


def _safe_confidence(value: Any, *, default: float) -> float:
    try:
        parsed = float(default if value is None else value)
    except (TypeError, ValueError):
        parsed = default
    return max(0.0, min(1.0, parsed))


def _storage_rejection_reason(error: str) -> str:
    normalized = error.replace("profile_observations[0].", "").replace("profile_observations[0] ", "")
    if normalized == "must be an object":
        return "observation_not_object"
    if normalized.startswith("field invalid"):
        field = normalized.removeprefix("field invalid").strip().strip("'")
        return f"field_not_allowed:{field or '<empty>'}"
    if normalized.startswith("value must be non-empty"):
        return "empty_value"
    if normalized.startswith("confidence below writable threshold"):
        return "confidence_below_threshold"
    if normalized.startswith("confidence"):
        return "invalid_confidence"
    if normalized.startswith("source invalid"):
        source = normalized.removeprefix("source invalid").strip().strip("'")
        return f"source_not_allowed:{source or '<empty>'}"
    if normalized.startswith("evidence must be non-empty"):
        return "missing_evidence"
    if normalized.startswith("risk invalid"):
        return "risk_not_allowed"
    return normalized
