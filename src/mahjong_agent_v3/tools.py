"""Compatibility tool gateway imports for historical ``mahjong_agent_v3`` users."""

from __future__ import annotations

from mahjong_agent_runtime.tools import (
    CANDIDATE_REPLY_STATUSES,
    GAME_STATUSES,
    ToolDefinition as ToolDefinitionV3,
    ToolGateway as ToolGatewayV3,
    ToolHandler as ToolHandlerV3,
    backend_tool_idempotency_key,
    default_tool_definitions as default_tool_definitions_v3,
    idempotency_lock_for_key,
    validate_schema,
)

__all__ = [
    "CANDIDATE_REPLY_STATUSES",
    "GAME_STATUSES",
    "ToolDefinitionV3",
    "ToolGatewayV3",
    "ToolHandlerV3",
    "backend_tool_idempotency_key",
    "default_tool_definitions_v3",
    "idempotency_lock_for_key",
    "validate_schema",
]
