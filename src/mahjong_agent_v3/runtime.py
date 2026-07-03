"""Compatibility runtime imports for historical ``mahjong_agent_v3`` users."""

from __future__ import annotations

from mahjong_agent_runtime.runtime import (
    AgentRuntime as AgentRuntimeV3,
    BudgetDecision as BudgetDecisionV3,
    TokenBudget as TokenBudgetV3,
    contract_error_action,
    parse_action,
    validate_action_contract,
    validate_stop_reason_contract,
)

__all__ = [
    "AgentRuntimeV3",
    "BudgetDecisionV3",
    "TokenBudgetV3",
    "contract_error_action",
    "parse_action",
    "validate_action_contract",
    "validate_stop_reason_contract",
]
