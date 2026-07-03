"""Compatibility context imports for historical ``mahjong_agent_v3`` users."""

from __future__ import annotations

from mahjong_agent_runtime.context import (
    AgentContextBuilder as AgentContextBuilderV3,
    BuiltContext as BuiltContextV3,
    ContextPackingPolicy as ContextPackingPolicyV3,
    DEFAULT_PROMPT_PATH as DEFAULT_PROMPT_PATH_V3,
    estimate_tokens,
    output_contract as output_contract_v3,
)

__all__ = [
    "AgentContextBuilderV3",
    "BuiltContextV3",
    "ContextPackingPolicyV3",
    "DEFAULT_PROMPT_PATH_V3",
    "estimate_tokens",
    "output_contract_v3",
]
