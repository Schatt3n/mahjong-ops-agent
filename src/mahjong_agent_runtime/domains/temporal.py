"""Resolve colloquial local times against the message timestamp.

The model still decides whether a past or genuinely ambiguous time needs a
clarifying question.  This module only provides one canonical interpretation
for deterministic facts and tool arguments, so ``6.30`` cannot mean 18:30 in
one loop step and 06:30 in the next.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from ..models import DEFAULT_TZ, now


MORNING_MARKERS = ("凌晨", "早上", "上午", "清晨")
AFTERNOON_MARKERS = ("下午", "晚上", "今晚", "傍晚")
NOON_MARKERS = ("中午",)


@dataclass(frozen=True, slots=True)
class LocalTimeResolution:
    """Canonical local time plus an audit label for how it was selected."""

    planned_at: datetime
    display: str
    inference: str


def parse_context_datetime(value: object, *, fallback: datetime | None = None) -> datetime:
    """Parse a context timestamp and always return an aware local datetime."""

    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value or ""))
        except ValueError:
            parsed = fallback or now()
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=DEFAULT_TZ)
    return parsed.astimezone(DEFAULT_TZ)


def resolve_local_time(
    text: str,
    *,
    hour: int,
    minute: int,
    anchor: datetime | None = None,
) -> LocalTimeResolution:
    """Resolve a clock expression using explicit day-parts, then recency.

    For a same-day expression without 上午/下午, both AM and PM candidates are
    considered when possible.  The nearest candidate that has not passed is
    selected; if both have passed, the closest past candidate is retained so
    the Agent can ask whether the user meant another day instead of silently
    moving the request to tomorrow.
    """

    stamp = parse_context_datetime(anchor, fallback=now())
    target = stamp + timedelta(days=1) if "明天" in text else stamp
    explicit_daypart = _explicit_daypart_hour(text, hour)
    if explicit_daypart is not None:
        planned = target.replace(hour=explicit_daypart, minute=minute, second=0, microsecond=0)
        return LocalTimeResolution(planned, _display(planned), "explicit_daypart")

    literal = target.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if "明天" in text or hour >= 12:
        return LocalTimeResolution(literal, _display(literal), "explicit_day_or_24h")

    candidates = [literal]
    if hour + 12 <= 23:
        candidates.append(literal.replace(hour=hour + 12))
    future = [candidate for candidate in candidates if candidate >= stamp]
    if future:
        planned = min(future, key=lambda candidate: candidate - stamp)
        inference = "nearest_non_past_same_day"
    else:
        planned = min(candidates, key=lambda candidate: stamp - candidate)
        inference = "closest_past_requires_clarification"
    return LocalTimeResolution(planned, _display(planned), inference)


def _explicit_daypart_hour(text: str, hour: int) -> int | None:
    if any(marker in text for marker in MORNING_MARKERS):
        return hour
    if any(marker in text for marker in AFTERNOON_MARKERS):
        return hour + 12 if hour < 12 else hour
    if any(marker in text for marker in NOON_MARKERS):
        if hour == 12:
            return 12
        return hour + 12 if hour <= 2 else hour
    return None


def _display(value: datetime) -> str:
    return f"{value.hour:02d}:{value.minute:02d}"


__all__ = ["LocalTimeResolution", "parse_context_datetime", "resolve_local_time"]
