"""Domain rules for value utils."""

from __future__ import annotations

from typing import Any

def list_values_for_keys(payload: dict[str, Any], *keys: str) -> list[Any]:
    values: list[Any] = []
    for key in keys:
        value = payload.get(key)
        if is_blank_value(value):
            continue
        if isinstance(value, (list, tuple, set)):
            values.extend(item for item in value if not is_blank_value(item))
        else:
            values.append(value)
    return values

def first_present_value(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = payload.get(key)
        if not is_blank_value(value):
            return value
    return None

def is_blank_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, tuple, set)):
        return len(value) == 0
    return False

def value_set(value: Any) -> set[str]:
    if is_blank_value(value):
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(item) for item in value if not is_blank_value(item)}
    return {str(value)}

def value_matches(query_value: Any, target_value: Any) -> bool:
    if is_blank_value(query_value):
        return False
    return bool(value_set(query_value) & value_set(target_value))


def normalize_smoke_preference(value: Any) -> str:
    """Map transport/model aliases to the canonical smoking preference value."""

    text = str(value or "").strip().lower()
    return {
        "无烟": "no_smoking",
        "no_smoke": "no_smoking",
        "no_smoking": "no_smoking",
        "有烟": "smoking",
        "烟": "smoking",
        "smoking": "smoking",
        "不限": "any",
        "都可": "any",
        "烟都可": "any",
        "any": "any",
        "": "any",
    }.get(text, text)


def smoke_value_set(value: Any) -> set[str]:
    """Return canonical smoking values for scalar or multi-value inputs."""

    return {
        normalize_smoke_preference(item)
        for item in value_set(value)
        if not is_blank_value(item)
    }


def smoke_matches(query_value: Any, target_value: Any) -> bool:
    query_values = smoke_value_set(query_value)
    target_values = smoke_value_set(target_value)
    if not query_values or "any" in query_values:
        return True
    if not target_values or "any" in target_values:
        return True
    return bool(query_values & target_values)
