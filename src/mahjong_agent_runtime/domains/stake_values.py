"""Parsing and formatting helpers for Mahjong stake values."""

from __future__ import annotations

import re
from typing import Any

from .value_utils import is_blank_value


def parse_stake_value(value: Any) -> tuple[float, float | None] | None:
    if is_blank_value(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    compact = (
        text.replace("，", "")
        .replace(",", "")
        .replace(" ", "")
        .replace("元", "")
        .replace("块", "")
        .replace("档", "")
    )
    compact_match = re.fullmatch(r"(?P<base>[1-9]\d?)(?P<cap>16|32|64|128)", compact)
    if compact_match and not re.fullmatch(r"\d7\d", compact):
        base = parse_number(compact_match.group("base"))
        cap = parse_number(compact_match.group("cap"))
        if base is not None:
            return base, cap
    explicit = re.search(
        r"(?P<base>\d+(?:\.\d+)?)\s*(?:元|块)?\s*(?:底|底注|底分).{0,8}?(?:封顶|封|顶|上限)\s*(?P<cap>\d+(?:\.\d+)?)",
        text,
    )
    if explicit:
        base = parse_number(explicit.group("base"))
        cap = parse_number(explicit.group("cap"))
        if base is not None:
            return base, cap
    range_match = re.fullmatch(
        r"(?P<base>\d+(?:\.\d+)?)\s*(?:-|/|－|—|到|至)\s*(?P<cap>\d+(?:\.\d+)?)",
        text,
    )
    if range_match:
        base = parse_number(range_match.group("base"))
        cap = parse_number(range_match.group("cap"))
        if base is not None:
            return base, cap
    base = parse_number(text)
    if base is not None:
        return base, None
    return None


def parse_number(value: Any) -> float | None:
    if is_blank_value(value):
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def format_number(value: float | int) -> str:
    number = float(value)
    return str(int(number)) if number.is_integer() else f"{number:g}"
