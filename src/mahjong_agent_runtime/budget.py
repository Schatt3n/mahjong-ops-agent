from __future__ import annotations

"""LLM 调用预算控制。

设计理念：
- 预算不是业务语义规则，而是生产稳定性的边界。
- 每次调用前先估算 token，再决定是否允许继续，避免无限 loop 或超长上下文拖垮链路。
- 只做粗估和保护，不追求和模型厂商 tokenizer 完全一致。
"""

from dataclasses import dataclass

from .token_estimation import estimate_tokens


@dataclass(slots=True)
class BudgetDecision:
    """一次预算预占用的结果。"""

    allowed: bool
    reason: str
    estimated_tokens: int

    def to_dict(self) -> dict[str, object]:
        """转成可写入 trace 的普通字典。"""

        return {"allowed": self.allowed, "reason": self.reason, "estimated_tokens": self.estimated_tokens}


@dataclass(slots=True)
class TokenBudget:
    """单条用户消息内的 LLM 调用预算。

    max_tokens_per_call 控制单次上下文大小；max_calls_per_turn 控制一次用户消息最多调用几次模型。
    calls_this_turn 是运行时计数，因此每条消息需要复制一份新的 TokenBudget。
    """

    max_tokens_per_call: int = 24_000
    max_calls_per_turn: int = 8
    calls_this_turn: int = 0

    def reserve(self, messages: list[dict[str, str]]) -> BudgetDecision:
        """预占一次模型调用预算。

        调用方在真正请求 LLM 前执行此方法；如果超出调用次数或 token 上限，就拒绝继续。
        """

        self.calls_this_turn += 1
        estimated = sum(estimate_tokens(item.get("content", "")) for item in messages)
        if self.calls_this_turn > self.max_calls_per_turn:
            return BudgetDecision(False, f"turn llm call limit exceeded: {self.max_calls_per_turn}", estimated)
        if estimated > self.max_tokens_per_call:
            return BudgetDecision(False, f"single call token estimate exceeded: {estimated}>{self.max_tokens_per_call}", estimated)
        return BudgetDecision(True, "budget_reserved", estimated)
