"""Stable public package for the current Mahjong Agent Runtime."""

from __future__ import annotations

from .budget import TokenBudget
from .context import AgentContextBuilder, ContextPackingPolicy
from .coordination import (
    FileCoordinationManager,
    InProcessCoordinationManager,
    RedisCoordinationManager,
    default_coordination_manager,
)
from .hooks import HookEvent, HookManager
from .input_aggregation import InputBatchDispatch, PendingInputScheduler, aggregate_pending_input_batch
from .lifecycle import ContextLifecycleManager
from .llm import AgentLLMConfig, OpenAICompatibleAgentClient, StaticAgentClient
from .loop import AgentLoop
from .models import (
    AgentAction,
    AgentRuntimeResult,
    ConversationCheckpoint,
    CustomerProfile,
    CustomerRelationship,
    Game,
    InviteDraft,
    MessageReference,
    OutboundMessageDraft,
    Party,
    PendingInputBatch,
    PendingInputBatchStatus,
    PendingMemoryCandidate,
    QuotedMessageRef,
    TaskMemory,
    ToolCall,
    ToolResult,
    UserMessage,
)
from .processing import ActionProcessor, ToolExecutionService
from .progress import ProgressDecision, ProgressMonitor, detect_tail_cycle, stable_fingerprint
from .runtime import AgentRuntime
from .runtime_components import ActionProcessingResult, ModelActionStep, ProgressHandlingResult, TurnBudgets
from .sqlite_store import SQLiteAgentStore
from .store import InMemoryAgentStore
from .summary import ContextSummaryManager, ContextSummaryPolicy, ContextSummaryResult
from .tools import ToolGateway
from .tracing import InMemoryTraceRecorder, JsonlTraceRecorder, validate_trace


__all__ = [
    "AgentAction",
    "AgentContextBuilder",
    "AgentLLMConfig",
    "AgentRuntime",
    "AgentRuntimeResult",
    "AgentLoop",
    "ActionProcessingResult",
    "ActionProcessor",
    "ContextPackingPolicy",
    "FileCoordinationManager",
    "InProcessCoordinationManager",
    "RedisCoordinationManager",
    "default_coordination_manager",
    "ContextLifecycleManager",
    "ConversationCheckpoint",
    "ContextSummaryManager",
    "ContextSummaryPolicy",
    "ContextSummaryResult",
    "CustomerProfile",
    "CustomerRelationship",
    "Game",
    "HookEvent",
    "HookManager",
    "InputBatchDispatch",
    "InMemoryAgentStore",
    "InMemoryTraceRecorder",
    "InviteDraft",
    "JsonlTraceRecorder",
    "MessageReference",
    "OpenAICompatibleAgentClient",
    "OutboundMessageDraft",
    "Party",
    "PendingInputBatch",
    "PendingInputBatchStatus",
    "PendingInputScheduler",
    "PendingMemoryCandidate",
    "ProgressDecision",
    "ProgressHandlingResult",
    "ProgressMonitor",
    "QuotedMessageRef",
    "SQLiteAgentStore",
    "StaticAgentClient",
    "TaskMemory",
    "TokenBudget",
    "ToolCall",
    "ToolExecutionService",
    "ToolGateway",
    "ToolResult",
    "ModelActionStep",
    "TurnBudgets",
    "UserMessage",
    "aggregate_pending_input_batch",
    "detect_tail_cycle",
    "stable_fingerprint",
    "validate_trace",
]
