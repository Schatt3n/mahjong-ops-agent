"""Stable public package for the current Mahjong Agent Runtime."""

from __future__ import annotations

from .context import AgentContextBuilderV3
from .llm import OpenAICompatibleAgentClientV3, StaticAgentClientV3
from .models import (
    AgentActionV3,
    AgentRuntimeResultV3,
    ConversationCheckpointV3,
    CustomerProfileV3,
    GameV3,
    InviteDraftV3,
    OutboundMessageDraftV3,
    ToolCallV3,
    ToolResultV3,
    UserMessageV3,
)
from .runtime import AgentRuntimeV3, TokenBudgetV3
from .sqlite_store import SQLiteAgentStoreV3
from .store import InMemoryAgentStoreV3
from .tools import ToolGatewayV3
from .tracing import InMemoryTraceRecorderV3, JsonlTraceRecorderV3

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
