"""Stable public package for the current Mahjong Agent Runtime.

The implementation originated in ``mahjong_agent_v3``. This package is the
versionless import surface used by the current main service so operators and
new code do not have to reason about historical runtime names.
"""

from __future__ import annotations

from mahjong_agent_v3 import (
    AgentActionV3,
    AgentContextBuilderV3,
    AgentRuntimeResultV3,
    AgentRuntimeV3,
    ConversationCheckpointV3,
    CustomerProfileV3,
    GameV3,
    InMemoryAgentStoreV3,
    InMemoryTraceRecorderV3,
    InviteDraftV3,
    JsonlTraceRecorderV3,
    OpenAICompatibleAgentClientV3,
    OutboundMessageDraftV3,
    SQLiteAgentStoreV3,
    StaticAgentClientV3,
    TokenBudgetV3,
    ToolCallV3,
    ToolGatewayV3,
    ToolResultV3,
    UserMessageV3,
)

AgentAction = AgentActionV3
AgentContextBuilder = AgentContextBuilderV3
AgentRuntime = AgentRuntimeV3
AgentRuntimeResult = AgentRuntimeResultV3
ConversationCheckpoint = ConversationCheckpointV3
CustomerProfile = CustomerProfileV3
Game = GameV3
InMemoryAgentStore = InMemoryAgentStoreV3
InMemoryTraceRecorder = InMemoryTraceRecorderV3
InviteDraft = InviteDraftV3
JsonlTraceRecorder = JsonlTraceRecorderV3
OpenAICompatibleAgentClient = OpenAICompatibleAgentClientV3
OutboundMessageDraft = OutboundMessageDraftV3
SQLiteAgentStore = SQLiteAgentStoreV3
StaticAgentClient = StaticAgentClientV3
TokenBudget = TokenBudgetV3
ToolCall = ToolCallV3
ToolGateway = ToolGatewayV3
ToolResult = ToolResultV3
UserMessage = UserMessageV3

__all__ = [
    "AgentAction",
    "AgentActionV3",
    "AgentContextBuilder",
    "AgentContextBuilderV3",
    "AgentRuntime",
    "AgentRuntimeResult",
    "AgentRuntimeResultV3",
    "AgentRuntimeV3",
    "ConversationCheckpoint",
    "ConversationCheckpointV3",
    "CustomerProfile",
    "CustomerProfileV3",
    "Game",
    "GameV3",
    "InMemoryAgentStore",
    "InMemoryAgentStoreV3",
    "InMemoryTraceRecorder",
    "InMemoryTraceRecorderV3",
    "InviteDraft",
    "InviteDraftV3",
    "JsonlTraceRecorder",
    "JsonlTraceRecorderV3",
    "OpenAICompatibleAgentClient",
    "OpenAICompatibleAgentClientV3",
    "OutboundMessageDraft",
    "OutboundMessageDraftV3",
    "SQLiteAgentStore",
    "SQLiteAgentStoreV3",
    "StaticAgentClient",
    "StaticAgentClientV3",
    "TokenBudget",
    "TokenBudgetV3",
    "ToolCall",
    "ToolCallV3",
    "ToolGateway",
    "ToolGatewayV3",
    "ToolResult",
    "ToolResultV3",
    "UserMessage",
    "UserMessageV3",
]
