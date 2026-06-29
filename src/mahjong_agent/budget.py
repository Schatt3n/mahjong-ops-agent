from __future__ import annotations

import json
import math
import os
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any

from .models import DEFAULT_TZ


@dataclass(slots=True)
class LLMUsage:
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated: bool = True
    cost: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["cost"] = round(self.cost, 6)
        return data


@dataclass(slots=True)
class LLMBudgetLimits:
    max_calls_per_day: int | None = 1000
    max_tokens_per_day: int | None = 200_000
    max_cost_per_day: float | None = None
    max_tokens_per_call: int | None = 8000
    input_price_per_1k: float = 0.0
    output_price_per_1k: float = 0.0

    @classmethod
    def from_env(cls) -> "LLMBudgetLimits":
        return cls(
            max_calls_per_day=_env_int("MAHJONG_LLM_MAX_CALLS_PER_DAY", 1000),
            max_tokens_per_day=_env_int("MAHJONG_LLM_MAX_TOKENS_PER_DAY", 200_000),
            max_cost_per_day=_env_float("MAHJONG_LLM_MAX_COST_PER_DAY", None),
            max_tokens_per_call=_env_int("MAHJONG_LLM_MAX_TOKENS_PER_CALL", 8000),
            input_price_per_1k=_env_float("MAHJONG_LLM_INPUT_PRICE_PER_1K", 0.0) or 0.0,
            output_price_per_1k=_env_float("MAHJONG_LLM_OUTPUT_PRICE_PER_1K", 0.0) or 0.0,
        )


@dataclass(slots=True)
class LLMBudgetDecision:
    allowed: bool
    key: str
    reservation_id: str | None
    reason: str
    estimated_usage: LLMUsage
    remaining_calls_today: int | None
    remaining_tokens_today: int | None
    remaining_cost_today: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "key": self.key,
            "reservation_id": self.reservation_id,
            "reason": self.reason,
            "estimated_usage": self.estimated_usage.to_dict(),
            "remaining_calls_today": self.remaining_calls_today,
            "remaining_tokens_today": self.remaining_tokens_today,
            "remaining_cost_today": (
                round(self.remaining_cost_today, 6)
                if self.remaining_cost_today is not None
                else None
            ),
        }


@dataclass(slots=True)
class _BudgetCounters:
    day: date
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0


class LLMBudgetManager:
    """In-process LLM budget guard.

    Production deployments should back this with Redis or a database for
    cross-process enforcement. This implementation gives the core workflow a
    concrete budget contract without adding runtime dependencies.
    """

    def __init__(self, limits: LLMBudgetLimits | None = None) -> None:
        self.limits = limits or LLMBudgetLimits()
        self._lock = threading.Lock()
        self._counters: dict[str, _BudgetCounters] = {}
        self._reservations: dict[str, tuple[str, LLMUsage]] = {}

    @classmethod
    def from_env(cls) -> "LLMBudgetManager":
        return cls(LLMBudgetLimits.from_env())

    def reserve(
        self,
        *,
        key: str,
        model: str,
        prompt: Any,
        max_completion_tokens: int,
    ) -> LLMBudgetDecision:
        safe_key = key or "default"
        prompt_tokens = estimate_tokens(prompt)
        estimated_usage = LLMUsage(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=max(0, int(max_completion_tokens)),
            total_tokens=prompt_tokens + max(0, int(max_completion_tokens)),
            estimated=True,
        )
        estimated_usage.cost = self._cost(estimated_usage)

        with self._lock:
            counters = self._today_counters(safe_key)
            reason = self._deny_reason(counters, estimated_usage)
            if reason:
                return self._decision(
                    allowed=False,
                    key=safe_key,
                    reservation_id=None,
                    reason=reason,
                    estimated_usage=estimated_usage,
                    counters=counters,
                )

            reservation_id = f"llm_budget_{uuid.uuid4().hex[:12]}"
            self._apply_usage(counters, estimated_usage, call_delta=1)
            self._reservations[reservation_id] = (safe_key, estimated_usage)
            return self._decision(
                allowed=True,
                key=safe_key,
                reservation_id=reservation_id,
                reason="budget_reserved",
                estimated_usage=estimated_usage,
                counters=counters,
            )

    def commit(self, reservation_id: str | None, actual_usage: LLMUsage | None) -> None:
        if reservation_id is None or actual_usage is None:
            return
        with self._lock:
            reservation = self._reservations.pop(reservation_id, None)
            if reservation is None:
                return
            key, estimated_usage = reservation
            counters = self._today_counters(key)
            actual_usage.cost = self._cost(actual_usage)
            self._apply_usage(
                counters,
                LLMUsage(
                    model=actual_usage.model,
                    prompt_tokens=actual_usage.prompt_tokens - estimated_usage.prompt_tokens,
                    completion_tokens=actual_usage.completion_tokens - estimated_usage.completion_tokens,
                    total_tokens=actual_usage.total_tokens - estimated_usage.total_tokens,
                    estimated=False,
                    cost=actual_usage.cost - estimated_usage.cost,
                ),
                call_delta=0,
            )

    def snapshot(self, key: str = "default") -> dict[str, Any]:
        with self._lock:
            counters = self._today_counters(key)
            return {
                "key": key,
                "day": counters.day.isoformat(),
                "calls": counters.calls,
                "prompt_tokens": counters.prompt_tokens,
                "completion_tokens": counters.completion_tokens,
                "total_tokens": counters.total_tokens,
                "cost": round(counters.cost, 6),
                "limits": asdict(self.limits),
            }

    def _today_counters(self, key: str) -> _BudgetCounters:
        today = datetime.now(DEFAULT_TZ).date()
        counters = self._counters.get(key)
        if counters is None or counters.day != today:
            counters = _BudgetCounters(day=today)
            self._counters[key] = counters
        return counters

    def _deny_reason(self, counters: _BudgetCounters, usage: LLMUsage) -> str | None:
        if self.limits.max_tokens_per_call is not None and usage.total_tokens > self.limits.max_tokens_per_call:
            return (
                f"单次 LLM 预计 token {usage.total_tokens} "
                f"超过上限 {self.limits.max_tokens_per_call}"
            )
        if self.limits.max_calls_per_day is not None and counters.calls + 1 > self.limits.max_calls_per_day:
            return f"今日 LLM 调用次数预算已用完，上限 {self.limits.max_calls_per_day}"
        if self.limits.max_tokens_per_day is not None and counters.total_tokens + usage.total_tokens > self.limits.max_tokens_per_day:
            return f"今日 LLM token 预算不足，上限 {self.limits.max_tokens_per_day}"
        if self.limits.max_cost_per_day is not None and counters.cost + usage.cost > self.limits.max_cost_per_day:
            return f"今日 LLM 成本预算不足，上限 {self.limits.max_cost_per_day}"
        return None

    def _decision(
        self,
        *,
        allowed: bool,
        key: str,
        reservation_id: str | None,
        reason: str,
        estimated_usage: LLMUsage,
        counters: _BudgetCounters,
    ) -> LLMBudgetDecision:
        return LLMBudgetDecision(
            allowed=allowed,
            key=key,
            reservation_id=reservation_id,
            reason=reason,
            estimated_usage=estimated_usage,
            remaining_calls_today=_remaining(self.limits.max_calls_per_day, counters.calls),
            remaining_tokens_today=_remaining(self.limits.max_tokens_per_day, counters.total_tokens),
            remaining_cost_today=_remaining_float(self.limits.max_cost_per_day, counters.cost),
        )

    def _apply_usage(self, counters: _BudgetCounters, usage: LLMUsage, *, call_delta: int) -> None:
        counters.calls += call_delta
        counters.prompt_tokens += usage.prompt_tokens
        counters.completion_tokens += usage.completion_tokens
        counters.total_tokens += usage.total_tokens
        counters.cost += usage.cost

    def _cost(self, usage: LLMUsage) -> float:
        return (
            usage.prompt_tokens / 1000 * self.limits.input_price_per_1k
            + usage.completion_tokens / 1000 * self.limits.output_price_per_1k
        )


def estimate_tokens(value: Any) -> int:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    cjk_or_symbol = sum(1 for char in text if ord(char) > 127)
    ascii_chars = max(0, len(text) - cjk_or_symbol)
    return max(1, cjk_or_symbol + math.ceil(ascii_chars / 4))


def usage_from_response(data: dict[str, Any], model: str) -> LLMUsage | None:
    raw = data.get("usage")
    if not isinstance(raw, dict):
        return None
    prompt_tokens = int(raw.get("prompt_tokens") or 0)
    completion_tokens = int(raw.get("completion_tokens") or 0)
    total_tokens = int(raw.get("total_tokens") or prompt_tokens + completion_tokens)
    if total_tokens <= 0:
        return None
    return LLMUsage(
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        estimated=False,
    )


def _remaining(limit: int | None, used: int) -> int | None:
    return None if limit is None else max(0, limit - used)


def _remaining_float(limit: float | None, used: float) -> float | None:
    return None if limit is None else max(0.0, limit - used)


def _env_int(name: str, default: int | None) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    if raw.strip().lower() in {"none", "off", "disabled", "unlimited"}:
        return None
    return int(raw)


def _env_float(name: str, default: float | None) -> float | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    if raw.strip().lower() in {"none", "off", "disabled", "unlimited"}:
        return None
    return float(raw)
