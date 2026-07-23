"""In-memory aggregate store used by tests and single-process deployments."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ...models import (
    AgentRuntimeResult,
    ConversationCheckpoint,
    ConversationTaskContext,
    ConversationTurn,
    CustomerProfile,
    CustomerRelationship,
    Game,
    InviteDraft,
    MessageReference,
    OutboundMessageDraft,
    PendingInputBatch,
    PendingMemoryCandidate,
    RoomReservation,
    ScheduledAgentTask,
    StateTransition,
    TaskMemory,
    ToolResult,
    WaitingDemand,
)
from .administration import InMemoryAdministrationStoreMixin
from .agent_runs import InMemoryAgentRunStoreMixin
from .channel_observations import InMemoryChannelObservationsStoreMixin
from .conversation import InMemoryConversationStoreMixin
from .customer import InMemoryCustomerStoreMixin
from .drafts import InMemoryDraftsStoreMixin
from .games import InMemoryGamesStoreMixin
from .group_chat import InMemoryGroupChatStoreMixin
from .idempotency import InMemoryIdempotencyStoreMixin
from .input_aggregation import InMemoryInputAggregationStoreMixin
from .references import InMemoryReferencesStoreMixin
from .rooms import InMemoryRoomsStoreMixin
from .scheduling import InMemorySchedulingStoreMixin
from .task_memory import InMemoryTaskMemoryStoreMixin
from .waiting import InMemoryWaitingDemandStoreMixin


@dataclass(slots=True)
class InMemoryAgentStore(
    InMemoryAgentRunStoreMixin,
    InMemoryChannelObservationsStoreMixin,
    InMemoryCustomerStoreMixin,
    InMemoryRoomsStoreMixin,
    InMemoryConversationStoreMixin,
    InMemoryTaskMemoryStoreMixin,
    InMemorySchedulingStoreMixin,
    InMemoryInputAggregationStoreMixin,
    InMemoryReferencesStoreMixin,
    InMemoryAdministrationStoreMixin,
    InMemoryGamesStoreMixin,
    InMemoryGroupChatStoreMixin,
    InMemoryDraftsStoreMixin,
    InMemoryWaitingDemandStoreMixin,
    InMemoryIdempotencyStoreMixin,
):
    """Compose the focused in-memory store mixins behind one AgentStore API."""

    customers: dict[str, CustomerProfile] = field(default_factory=dict)
    customer_relationships: dict[str, CustomerRelationship] = field(default_factory=dict)
    games: dict[str, Game] = field(default_factory=dict)
    invite_drafts: dict[str, InviteDraft] = field(default_factory=dict)
    outbound_message_drafts: dict[str, OutboundMessageDraft] = field(default_factory=dict)
    room_ids: list[str] = field(default_factory=list)
    room_reservations: dict[str, RoomReservation] = field(default_factory=dict)
    transitions: list[StateTransition] = field(default_factory=list)
    turns: dict[str, list[ConversationTurn]] = field(default_factory=dict)
    conversation_checkpoints: dict[str, ConversationCheckpoint] = field(default_factory=dict)
    task_context_checkpoints: dict[str, ConversationCheckpoint] = field(default_factory=dict)
    task_contexts: dict[str, ConversationTaskContext] = field(default_factory=dict)
    conversation_versions: dict[str, int] = field(default_factory=dict)
    idempotency_ledger: dict[str, ToolResult] = field(default_factory=dict)
    idempotency_claimed_at: dict[str, datetime] = field(default_factory=dict)
    message_results: dict[str, AgentRuntimeResult] = field(default_factory=dict)
    message_references: dict[str, MessageReference] = field(default_factory=dict)
    task_memories: dict[str, TaskMemory] = field(default_factory=dict)
    pending_memory_candidates: dict[str, PendingMemoryCandidate] = field(default_factory=dict)
    pending_input_batches: dict[str, PendingInputBatch] = field(default_factory=dict)
    scheduled_tasks: dict[str, ScheduledAgentTask] = field(default_factory=dict)
    waiting_demands: dict[str, WaitingDemand] = field(default_factory=dict)
    channel_identities: dict[str, Any] = field(default_factory=dict)
    group_room_policies: dict[str, Any] = field(default_factory=dict)
    group_board_states: dict[str, Any] = field(default_factory=dict)
    game_conversation_link_records: dict[str, Any] = field(default_factory=dict)
    board_snapshots: dict[str, Any] = field(default_factory=dict)
    game_claims: dict[str, Any] = field(default_factory=dict)
    channel_switches: dict[str, Any] = field(default_factory=dict)
    channel_observations: dict[str, dict[str, Any]] = field(default_factory=dict)
    agent_runs: dict[str, Any] = field(default_factory=dict)
    badcases: list[dict[str, Any]] = field(default_factory=list)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)


__all__ = ["InMemoryAgentStore"]
