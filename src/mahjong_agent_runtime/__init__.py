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
from .llm import AgentLLMConfig, OpenAICompatibleAgentClient, StaticAgentClient
from .matching import MatchTrigger, OutboundDispatcher, handle_waiting_expiration_task
from .domains.waiting_domain import WAITING_DEMAND_EXPIRY_TASK_TYPE, next_waiting_expiry_due
from .models import (
    AgentAction,
    AgentRuntimeResult,
    AgentSelfAssessment,
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
    RecruitmentStatus,
    ScheduledAgentTask,
    ScheduledTaskStatus,
    SystemTriggerMessage,
    TaskMemory,
    ToolCall,
    ToolResult,
    UserMessage,
    WaitingDemand,
    WaitingDemandStatus,
)
from .services import ActionProcessor, AgentLoop, ContextLifecycleManager, ToolExecutionService
from .progress import ProgressDecision, ProgressMonitor, detect_tail_cycle, stable_fingerprint
from .runtime import AgentRuntime
from .scheduled_tasks import ScheduledAgentTaskScheduler
from .runtime_components import ActionProcessingResult, ModelActionStep, ProgressHandlingResult, TurnBudgets
from .stores.memory import InMemoryAgentStore
from .stores.sqlite import SQLiteAgentStore
from .stores import (
    AgentStore,
    BaseStore,
    ConversationStore,
    CustomerStore,
    GameStore,
    GroupChatStore,
    IdempotencyStore,
    TaskStore,
    WaitingDemandStore,
)
from .summary import ContextSummaryManager, ContextSummaryPolicy, ContextSummaryResult
from .summary_evaluation import (
    ContextSummaryQualityEvaluator,
    DecisionConsistencyReport,
    DecisionSnapshot,
)
from .domains.tools import ToolGateway
from .tracing import InMemoryTraceRecorder, JsonlTraceRecorder, validate_trace


__all__ = [
    "AgentAction",
    "AgentContextBuilder",
    "AgentLLMConfig",
    "AgentRuntime",
    "AgentRuntimeResult",
    "AgentSelfAssessment",
    "AgentStore",
    "AgentLoop",
    "ActionProcessingResult",
    "ActionProcessor",
    "BaseStore",
    "ContextPackingPolicy",
    "FileCoordinationManager",
    "InProcessCoordinationManager",
    "RedisCoordinationManager",
    "default_coordination_manager",
    "ContextLifecycleManager",
    "ConversationCheckpoint",
    "ConversationStore",
    "ContextSummaryManager",
    "ContextSummaryPolicy",
    "ContextSummaryQualityEvaluator",
    "ContextSummaryResult",
    "CustomerProfile",
    "CustomerRelationship",
    "CustomerStore",
    "DecisionConsistencyReport",
    "DecisionSnapshot",
    "Game",
    "GameStore",
    "GroupChatStore",
    "HookEvent",
    "HookManager",
    "InputBatchDispatch",
    "IdempotencyStore",
    "InMemoryAgentStore",
    "InMemoryTraceRecorder",
    "InviteDraft",
    "JsonlTraceRecorder",
    "MessageReference",
    "MatchTrigger",
    "OpenAICompatibleAgentClient",
    "OutboundMessageDraft",
    "OutboundDispatcher",
    "Party",
    "PendingInputBatch",
    "PendingInputBatchStatus",
    "PendingInputScheduler",
    "PendingMemoryCandidate",
    "ProgressDecision",
    "ProgressHandlingResult",
    "ProgressMonitor",
    "QuotedMessageRef",
    "RecruitmentStatus",
    "ScheduledAgentTask",
    "ScheduledAgentTaskScheduler",
    "ScheduledTaskStatus",
    "SystemTriggerMessage",
    "SQLiteAgentStore",
    "StaticAgentClient",
    "TaskMemory",
    "TaskStore",
    "TokenBudget",
    "ToolCall",
    "ToolExecutionService",
    "ToolGateway",
    "ToolResult",
    "ModelActionStep",
    "TurnBudgets",
    "UserMessage",
    "WaitingDemand",
    "WaitingDemandStatus",
    "WaitingDemandStore",
    "WAITING_DEMAND_EXPIRY_TASK_TYPE",
    "aggregate_pending_input_batch",
    "detect_tail_cycle",
    "handle_waiting_expiration_task",
    "next_waiting_expiry_due",
    "stable_fingerprint",
    "validate_trace",
]
