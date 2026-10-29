"""区分固定底注、可接受区间与封顶，供查局和候选人检索共用。"""

from __future__ import annotations

import math
import re
from typing import Any

from .value_utils import first_present_value, is_blank_value, list_values_for_keys


STAKE_VALUE_KEYS = ("stake", "base_stake", "preferred_stake", "preferred_stakes", "stakes", "accepted_stakes")


def parse_stake_range(value: Any) -> tuple[float, float] | None:
    """只校验模型明确提交的闭区间，不从自由文本或两个数字猜测区间语义。"""
    if isinstance(value, dict):
        if set(value) != {"min", "max"}:
            raise ValueError("stake_range requires exactly min and max")
        lower, upper = value["min"], value["max"]
        if isinstance(lower, bool) or isinstance(upper, bool):
            raise ValueError("stake_range bounds must be numbers, not booleans")
        if not isinstance(lower, (int, float)) or not isinstance(upper, (int, float)):
            raise ValueError("stake_range bounds must be numbers")
    else:
        return None
    lower, upper = float(lower), float(upper)
    if not math.isfinite(lower) or not math.isfinite(upper) or lower <= 0 or lower > upper:
        raise ValueError("stake_range requires finite bounds with 0 < min <= max")
    return lower, upper


def parse_stake_value(value: Any) -> tuple[float, float | None] | None:
    """解析固定底注及封顶；已经识别出的区间不能再被解释为封顶。"""
    if is_blank_value(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if parse_stake_range(value) is not None:
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


def stake_options(requirement: dict[str, Any]) -> list[tuple[float, float, float | None]]:
    """把固定值、多选偏好与区间统一为闭区间选项，封顶保持独立。

    未能解析的画像字符串不冒充匹配证据。显式工具参数的合法性由需求规范化
    提前校验；这里忽略未知画像，避免一条历史备注中断整个只读检索。
    """
    cap = parse_number(first_present_value(requirement, "cap_score", "cap_stake", "cap", "cap_limit"))
    if "stake_range" in requirement:
        bounds = parse_stake_range(requirement["stake_range"])
        if bounds is None:
            raise ValueError("stake_range must be an object with min and max")
        return [(bounds[0], bounds[1], cap)]
    options: list[tuple[float, float, float | None]] = []
    for value in list_values_for_keys(requirement, *STAKE_VALUE_KEYS):
        try:
            bounds = parse_stake_range(value)
            fixed = None if bounds else parse_stake_value(value)
        except ValueError:
            continue
        if bounds is not None:
            options.append((bounds[0], bounds[1], cap))
        elif fixed is not None and math.isfinite(fixed[0]) and fixed[0] > 0:
            options.append((fixed[0], fixed[0], fixed[1] if fixed[1] is not None else cap))
    return options


def stake_requirements_overlap(
    query: dict[str, Any],
    target: dict[str, Any],
    *,
    allow_unknown_cap: bool = False,
    ignore_cap: bool = False,
) -> bool:
    """按闭区间交集匹配底注，按调用方合同判断未知封顶是否可接受。"""
    for lower, upper, cap in stake_options(query):
        for target_lower, target_upper, target_cap in stake_options(target):
            if max(lower, target_lower) > min(upper, target_upper):
                continue
            if ignore_cap or cap is None or cap == target_cap or (allow_unknown_cap and target_cap is None):
                return True
    return False


def unresolved_stake_preferences(values: list[str]) -> list[str]:
    """保留不能按既有数值协议读取的画像原文，交回模型推理而非误报匹配。"""
    return [value for value in values if not stake_options({"stake": value})]


def parse_number(value: Any) -> float | None:
    """解析number，并把非结构化输入转换为稳定合同。"""
    if is_blank_value(value):
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def format_number(value: float | int) -> str:
    number = float(value)
    return str(int(number)) if number.is_integer() else f"{number:g}"
