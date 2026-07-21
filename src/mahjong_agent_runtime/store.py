"""Backward-compatible store facade.

Business rules live in ``domains`` and backend behavior lives in ``stores``.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .models import (
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
)
from .domains import (
    ALLOWED_GAME_TRANSITIONS,
    CONFIRMED_CANDIDATE_STATUSES,
    DEFAULT_ASAP_GAME_TTL_HOURS,
    DEFAULT_OVERNIGHT_DURATION_HOURS,
    DEFAULT_RECRUITMENT_LEAD_HOURS,
    DEFAULT_UNKNOWN_DURATION_HOURS,
    DURATION_KIND_OVERNIGHT,
    GAME_RECRUITMENT_TASK_TYPE,
    GameCommitmentResolution,
    PENDING_INPUT_PROCESSING_LEASE_SECONDS,
    PROTECTED_REQUIREMENT_PATCH_FIELDS,
    SCHEDULED_TASK_PROCESSING_LEASE_SECONDS,
    START_KIND_ASAP_WHEN_FULL,
    START_KIND_SCHEDULED,
    UNCONFIRMED_CANDIDATE_STATUSES,
    _release_game_participants,
    _rewrite_contact_names,
    active_game_participant_ids,
    anonymous_seat_count_from_payload,
    apply_game_lifecycle,
    apply_game_recruitment_policy,
    canonical_game_participant_status,
    customer_option_load,
    customer_visible_name,
    derive_game_lifecycle,
    duration_hours_from_requirement,
    expire_game_if_stale,
    first_datetime_value,
    first_present_value,
    format_number,
    game_commitment_window,
    game_commitment_windows_overlap,
    game_contains_customer,
    game_for_model_context,
    game_recruitment_task_id,
    game_schedule_sort_key,
    invite_draft_for_model_context,
    invite_status_from_candidate_status,
    is_avoid_playing_memory,
    is_blank_value,
    join_projection,
    known_member_ids_from_payload,
    list_values_for_keys,
    message_reference_key,
    normalize_datetime,
    normalize_game_participants,
    normalize_game_parties,
    normalize_requirement,
    normalize_requirement_with_party,
    outbound_message_draft_for_model_context,
    parse_datetime_value,
    parse_number,
    parse_stake_value,
    parse_start_time_on_created_date,
    party_id_for_contact,
    payload_has_explicit_seat_count,
    pending_input_batch_key,
    ready_commitment_conflicts,
    recruitment_open_time,
    refresh_requirement_seat_snapshot,
    relationship_anchor_ids,
    relationship_context_for_sender,
    relationship_pair_key,
    requested_seat_count_from_search_requirement,
    requirement_commitment_window,
    requirement_overlaps_game,
    resolve_full_game_commitments,
    score_customer,
    score_customer_relationships,
    score_requirement,
    score_stake_preference,
    seat_count_from_payload,
    smoke_matches,
    task_memory_anchor_ids,
    value_matches,
    value_set,
    visible_draft_metadata,
)
from .stores.idempotency_common import (
    IDEMPOTENCY_CLAIM_LEASE_SECONDS,
    tool_result_is_in_progress,
)
from .stores.memory.idempotency import InMemoryIdempotencyStoreMixin
from .stores.memory.customer import InMemoryCustomerStoreMixin
from .stores.memory.rooms import InMemoryRoomsStoreMixin
from .stores.memory.conversation import InMemoryConversationStoreMixin
from .stores.memory.task_memory import InMemoryTaskMemoryStoreMixin
from .stores.memory.scheduling import InMemorySchedulingStoreMixin
from .stores.memory.input_aggregation import InMemoryInputAggregationStoreMixin
from .stores.memory.references import InMemoryReferencesStoreMixin
from .stores.memory.administration import InMemoryAdministrationStoreMixin
from .stores.memory.games import InMemoryGamesStoreMixin
from .stores.memory.drafts import InMemoryDraftsStoreMixin

@dataclass(slots=True)
class InMemoryAgentStore(
    InMemoryCustomerStoreMixin,
    InMemoryRoomsStoreMixin,
    InMemoryConversationStoreMixin,
    InMemoryTaskMemoryStoreMixin,
    InMemorySchedulingStoreMixin,
    InMemoryInputAggregationStoreMixin,
    InMemoryReferencesStoreMixin,
    InMemoryAdministrationStoreMixin,
    InMemoryGamesStoreMixin,
    InMemoryDraftsStoreMixin,
    InMemoryIdempotencyStoreMixin,
):
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
    badcases: list[dict[str, Any]] = field(default_factory=list)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
