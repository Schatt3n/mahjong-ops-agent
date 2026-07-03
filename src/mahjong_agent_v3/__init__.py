"""Compatibility import surface for historical ``mahjong_agent_v3`` users."""

from __future__ import annotations

from mahjong_agent_runtime import (
    AgentAction as AgentActionV3,
    AgentContextBuilder as AgentContextBuilderV3,
    AgentRuntime as AgentRuntimeV3,
    AgentRuntimeResult as AgentRuntimeResultV3,
    ConversationCheckpoint as ConversationCheckpointV3,
    CustomerProfile as CustomerProfileV3,
    Game as GameV3,
    InMemoryAgentStore as InMemoryAgentStoreV3,
    InMemoryTraceRecorder as InMemoryTraceRecorderV3,
    InviteDraft as InviteDraftV3,
    JsonlTraceRecorder as JsonlTraceRecorderV3,
    OpenAICompatibleAgentClient as OpenAICompatibleAgentClientV3,
    OutboundMessageDraft as OutboundMessageDraftV3,
    SQLiteAgentStore as SQLiteAgentStoreV3,
    StaticAgentClient as StaticAgentClientV3,
    TokenBudget as TokenBudgetV3,
    ToolCall as ToolCallV3,
    ToolGateway as ToolGatewayV3,
    ToolResult as ToolResultV3,
    UserMessage as UserMessageV3,
)

__all__ = [
    "AgentActionV3",
    "AgentContextBuilderV3",
    "AgentRuntimeResultV3",
    "AgentRuntimeV3",
    "ConversationCheckpointV3",
    "CustomerProfileV3",
    "GameV3",
    "InMemoryAgentStoreV3",
    "InMemoryTraceRecorderV3",
    "InviteDraftV3",
    "JsonlTraceRecorderV3",
    "OpenAICompatibleAgentClientV3",
    "OutboundMessageDraftV3",
    "SQLiteAgentStoreV3",
    "StaticAgentClientV3",
    "TokenBudgetV3",
    "ToolCallV3",
    "ToolGatewayV3",
    "ToolResultV3",
    "UserMessageV3",
]
