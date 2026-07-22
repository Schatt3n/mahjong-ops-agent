"""Compatibility and refinement rules for structured session facts."""

from __future__ import annotations


_TIME_SCOPE_MARKERS = ("今天", "今晚", "上午", "中午", "下午", "晚上", "明天", "明晚", "人齐开")


def facts_conflict(key: str, left: object, right: object) -> bool:
    """Return whether two non-empty facts cannot describe the same session."""

    if left in (None, "") or right in (None, ""):
        return False
    left_text = str(left).strip().lower()
    right_text = str(right).strip().lower()
    if left_text == right_text:
        return False
    if key != "time":
        return True
    left_is_scope = _is_time_scope(left_text)
    right_is_scope = _is_time_scope(right_text)
    if left_is_scope != right_is_scope:
        return False
    if left_is_scope and right_is_scope:
        return _time_scopes_conflict(left_text, right_text)
    return True


def merge_session_facts(base: dict, incoming: dict) -> dict:
    """Merge facts while allowing a precise time to refine a broad time scope."""

    merged = dict(base)
    for key, value in incoming.items():
        if value in (None, "", []):
            continue
        current = merged.get(key)
        if current in (None, "", []):
            merged[key] = value
        elif key == "time" and _is_time_scope(str(current)) and not _is_time_scope(str(value)):
            merged[key] = value
    return merged


def _is_time_scope(value: str) -> bool:
    return any(marker in value for marker in _TIME_SCOPE_MARKERS)


def _time_scopes_conflict(left: str, right: str) -> bool:
    left_day = "tomorrow" if "明" in left else "today" if "今" in left else None
    right_day = "tomorrow" if "明" in right else "today" if "今" in right else None
    if left_day and right_day and left_day != right_day:
        return True
    periods = (("上午", "morning"), ("中午", "noon"), ("下午", "afternoon"), ("晚上", "evening"), ("今晚", "evening"), ("明晚", "evening"))
    left_period = next((value for marker, value in periods if marker in left), None)
    right_period = next((value for marker, value in periods if marker in right), None)
    return bool(left_period and right_period and left_period != right_period)


__all__ = ["facts_conflict", "merge_session_facts"]
