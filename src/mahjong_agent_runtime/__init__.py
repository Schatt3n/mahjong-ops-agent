"""Stable public package for the current Mahjong Agent Runtime."""

from __future__ import annotations

from .context import AgentContextBuilder, ContextPackingPolicy
from .llm import AgentLLMConfig, OpenAICompatibleAgentClient, StaticAgentClient
from .models import (
    AgentAction,
    AgentRuntimeResult,
    ConversationCheckpoint,
    CustomerProfile,
    Game,
    InviteDraft,
    OutboundMessageDraft,
    ToolCall,
    ToolResult,
    UserMessage,
)
from .runtime import AgentRuntime, TokenBudget
from .sqlite_store import SQLiteAgentStore
from .store import InMemoryAgentStore
from .tools import ToolGateway
from .tracing import InMemoryTraceRecorder, JsonlTraceRecorder, validate_trace


__all__ = [
    "AgentAction",
    "AgentContextBuilder",
    "AgentLLMConfig",
    "AgentRuntime",
    "AgentRuntimeResult",
    "ContextPackingPolicy",
    "ConversationCheckpoint",
    "CustomerProfile",
    "Game",
    "InMemoryAgentStore",
    "InMemoryTraceRecorder",
    "InviteDraft",
    "JsonlTraceRecorder",
    "OpenAICompatibleAgentClient",
    "OutboundMessageDraft",
    "SQLiteAgentStore",
    "StaticAgentClient",
    "TokenBudget",
    "ToolCall",
    "ToolGateway",
    "ToolResult",
    "UserMessage",
    "validate_trace",
]
