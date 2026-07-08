from __future__ import annotations

from dataclasses import dataclass

from .context import estimate_tokens


@dataclass(slots=True)
class BudgetDecision:
    allowed: bool
    reason: str
    estimated_tokens: int

    def to_dict(self) -> dict[str, object]:
        return {"allowed": self.allowed, "reason": self.reason, "estimated_tokens": self.estimated_tokens}


@dataclass(slots=True)
class TokenBudget:
    max_tokens_per_call: int = 24_000
    max_calls_per_turn: int = 8
    calls_this_turn: int = 0

    def reserve(self, messages: list[dict[str, str]]) -> BudgetDecision:
        self.calls_this_turn += 1
        estimated = sum(estimate_tokens(item.get("content", "")) for item in messages)
        if self.calls_this_turn > self.max_calls_per_turn:
            return BudgetDecision(False, f"turn llm call limit exceeded: {self.max_calls_per_turn}", estimated)
        if estimated > self.max_tokens_per_call:
            return BudgetDecision(False, f"single call token estimate exceeded: {estimated}>{self.max_tokens_per_call}", estimated)
        return BudgetDecision(True, "budget_reserved", estimated)
