from __future__ import annotations

from typing import Any

from .workflow_models import SlotSource, SlotValue


def semantic_slot_value(slot: Any) -> Any:
    if isinstance(slot, SlotValue):
        return slot.value
    if isinstance(slot, dict):
        return slot.get("value")
    return slot


def semantic_slot_confidence(slot: Any) -> float:
    if isinstance(slot, SlotValue):
        return slot.confidence
    if not isinstance(slot, dict):
        return 0.0
    return _safe_float(slot.get("confidence")) or 0.0


def semantic_slot_source(slot: Any) -> str:
    if isinstance(slot, SlotValue):
        return slot.source.value
    if not isinstance(slot, dict):
        return ""
    source = slot.get("source")
    if isinstance(source, SlotSource):
        return source.value
    return str(source or "").strip().lower()


def semantic_slot_needs_confirmation(slot: Any) -> bool:
    if isinstance(slot, SlotValue):
        return bool(slot.needs_confirmation)
    if not isinstance(slot, dict):
        return True
    return bool(slot.get("needs_confirmation"))


def semantic_slot_usable(slot: Any, *, min_confidence: float) -> bool:
    if not isinstance(slot, (dict, SlotValue)):
        return False
    if semantic_slot_needs_confirmation(slot):
        return False
    if semantic_slot_confidence(slot) < min_confidence:
        return False
    return semantic_slot_source(slot) not in {"", "unknown"}


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
