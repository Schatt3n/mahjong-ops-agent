"""SQLite payload serialization and aggregate hydration helpers."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Any

from ...models import (
    AgentAction,
    AgentRuntimeResult,
    ConversationCheckpoint,
    ConversationRole,
    ConversationTaskContext,
    ConversationTurn,
    CustomerProfile,
    CustomerRelationship,
    DEFAULT_TZ,
    Game,
    GameParticipant,
    GameStatus,
    InviteDraft,
    InviteStatus,
    MessageReference,
    OutboundDraftStatus,
    OutboundMessageDraft,
    Party,
    PendingInputBatch,
    PendingInputBatchStatus,
    PendingMemoryCandidate,
    RecruitmentStatus,
    RoomReservation,
    ScheduledAgentTask,
    ScheduledTaskStatus,
    StateTransition,
    TaskMemory,
    ToolCall,
    ToolResult,
)
from ...domains import normalize_game_participants, normalize_game_parties


def customer_from_payload(payload: dict[str, Any]) -> CustomerProfile:
    return CustomerProfile(
        customer_id=str(payload.get("customer_id") or ""),
        display_name=str(payload.get("display_name") or ""),
        public_name=str(payload.get("public_name") or "") or None,
        private_remark=str(payload.get("private_remark") or ""),
        gender=payload.get("gender"),
        preferred_games=[str(item) for item in payload.get("preferred_games") or []],
        preferred_stakes=[str(item) for item in payload.get("preferred_stakes") or []],
        preferred_time_tags=[str(item) for item in payload.get("preferred_time_tags") or []],
        profile_facts=[str(item) for item in payload.get("profile_facts") or []],
        smoke_preference=payload.get("smoke_preference"),
        response_score=float(payload.get("response_score") or 0.5),
        fatigue_score=float(payload.get("fatigue_score") or 0.0),
        no_contact=bool(payload.get("no_contact")),
        notes=str(payload.get("notes") or ""),
    )


def relationship_from_payload(payload: dict[str, Any]) -> CustomerRelationship:
    return CustomerRelationship(
        customer_a_id=str(payload.get("customer_a_id") or ""),
        customer_b_id=str(payload.get("customer_b_id") or ""),
        played_together_count=int(payload.get("played_together_count") or 0),
        avoid_playing=bool(payload.get("avoid_playing")),
        notes=str(payload.get("notes") or ""),
        updated_at=datetime_from_payload(payload.get("updated_at")),
    )


def task_memory_from_payload(payload: dict[str, Any]) -> TaskMemory:
    return TaskMemory(
        memory_id=str(payload.get("memory_id") or ""),
        conversation_id=str(payload.get("conversation_id") or ""),
        customer_id=str(payload.get("customer_id") or ""),
        memory_type=str(payload.get("memory_type") or ""),
        field=str(payload.get("field") or ""),
        value=payload.get("value"),
        target_customer_id=str(payload.get("target_customer_id") or "") or None,
        evidence=str(payload.get("evidence") or ""),
        confidence=float(payload.get("confidence") or 0.0),
        risk_level=str(payload.get("risk_level") or "medium"),
        scope=str(payload.get("scope") or "current_task"),
        status=str(payload.get("status") or "active"),
        source_trace_id=payload.get("source_trace_id"),
        metadata=dict(payload.get("metadata") or {}) if isinstance(payload.get("metadata"), dict) else {},
        created_at=datetime_from_payload(payload.get("created_at")),
        updated_at=datetime_from_payload(payload.get("updated_at")),
    )


def task_context_from_payload(payload: dict[str, Any]) -> ConversationTaskContext:
    return ConversationTaskContext(
        task_context_id=str(payload.get("task_context_id") or ""),
        conversation_id=str(payload.get("conversation_id") or ""),
        customer_id=str(payload.get("customer_id") or ""),
        status=str(payload.get("status") or "active"),
        reset_reason=str(payload.get("reset_reason") or "first_message"),
        previous_task_context_id=str(payload.get("previous_task_context_id") or "") or None,
        source_trace_id=payload.get("source_trace_id"),
        started_at=datetime_from_payload(payload.get("started_at")),
        updated_at=datetime_from_payload(payload.get("updated_at")),
        closed_at=datetime_from_payload(payload.get("closed_at")) if payload.get("closed_at") else None,
    )


def pending_memory_candidate_from_payload(payload: dict[str, Any]) -> PendingMemoryCandidate:
    return PendingMemoryCandidate(
        candidate_id=str(payload.get("candidate_id") or ""),
        conversation_id=str(payload.get("conversation_id") or ""),
        customer_id=str(payload.get("customer_id") or ""),
        memory_type=str(payload.get("memory_type") or ""),
        field=str(payload.get("field") or ""),
        value=payload.get("value"),
        operation=str(payload.get("operation") or "set"),
        target_customer_id=str(payload.get("target_customer_id") or "") or None,
        evidence=str(payload.get("evidence") or ""),
        confidence=float(payload.get("confidence") or 0.0),
        risk_level=str(payload.get("risk_level") or "medium"),
        scope=str(payload.get("scope") or "long_term"),
        status=str(payload.get("status") or "pending_review"),
        source_trace_id=payload.get("source_trace_id"),
        metadata=dict(payload.get("metadata") or {}) if isinstance(payload.get("metadata"), dict) else {},
        created_at=datetime_from_payload(payload.get("created_at")),
        updated_at=datetime_from_payload(payload.get("updated_at")),
    )


def pending_input_batch_from_payload(payload: dict[str, Any]) -> PendingInputBatch:
    return PendingInputBatch(
        batch_id=str(payload.get("batch_id") or ""),
        conversation_id=str(payload.get("conversation_id") or ""),
        sender_id=str(payload.get("sender_id") or ""),
        sender_name=str(payload.get("sender_name") or ""),
        fragments=[dict(item) for item in payload.get("fragments") or [] if isinstance(item, dict)],
        version=int(payload.get("version") or 1),
        status=PendingInputBatchStatus(str(payload.get("status") or PendingInputBatchStatus.PENDING.value)),
        quiet_deadline=datetime_from_payload(payload.get("quiet_deadline")),
        source_channel=str(payload.get("source_channel") or ""),
        decision=dict(payload.get("decision") or {}) if isinstance(payload.get("decision"), dict) else {},
        created_at=datetime_from_payload(payload.get("created_at")),
        updated_at=datetime_from_payload(payload.get("updated_at")),
    )


def scheduled_agent_task_from_payload(payload: dict[str, Any]) -> ScheduledAgentTask:
    return ScheduledAgentTask(
        task_id=str(payload.get("task_id") or ""),
        task_type=str(payload.get("task_type") or ""),
        aggregate_type=str(payload.get("aggregate_type") or ""),
        aggregate_id=str(payload.get("aggregate_id") or ""),
        conversation_id=str(payload.get("conversation_id") or ""),
        subject_id=str(payload.get("subject_id") or ""),
        subject_name=str(payload.get("subject_name") or ""),
        due_at=datetime_from_payload(payload.get("due_at")),
        idempotency_key=str(payload.get("idempotency_key") or ""),
        payload=dict(payload.get("payload") or {}) if isinstance(payload.get("payload"), dict) else {},
        status=ScheduledTaskStatus(str(payload.get("status") or ScheduledTaskStatus.PENDING.value)),
        attempts=int(payload.get("attempts") or 0),
        lease_until=optional_datetime_from_payload(payload.get("lease_until")),
        last_error=str(payload.get("last_error") or ""),
        created_at=datetime_from_payload(payload.get("created_at")),
        updated_at=datetime_from_payload(payload.get("updated_at")),
        completed_at=optional_datetime_from_payload(payload.get("completed_at")),
    )


def turn_from_payload(payload: dict[str, Any]) -> ConversationTurn:
    return ConversationTurn(
        role=ConversationRole(str(payload.get("role") or ConversationRole.USER.value)),
        content=str(payload.get("content") or ""),
        trace_id=str(payload.get("trace_id") or ""),
        sender_id=payload.get("sender_id"),
        sender_name=payload.get("sender_name"),
        metadata=dict(payload.get("metadata") or {}),
        occurred_at=datetime_from_payload(payload.get("occurred_at")),
    )


def checkpoint_from_payload(payload: dict[str, Any]) -> ConversationCheckpoint:
    return ConversationCheckpoint(
        conversation_id=str(payload.get("conversation_id") or ""),
        summary=str(payload.get("summary") or ""),
        facts=dict(payload.get("facts") or {}) if isinstance(payload.get("facts"), dict) else {},
        open_questions=[str(item) for item in payload.get("open_questions") or []],
        task_context_id=str(payload.get("task_context_id") or "") or None,
        source_trace_id=payload.get("source_trace_id"),
        updated_at=datetime_from_payload(payload.get("updated_at")),
    )


def game_storage_payload(game: Game) -> dict[str, Any]:
    """Return only non-participant fields for ``runtime_games.payload``."""

    payload = game.to_dict()
    for key in ("participants", "parties", "seat_claims", "seat_summary", "remaining_seats"):
        payload.pop(key, None)
    return payload


def game_participant_from_row(row: sqlite3.Row) -> GameParticipant:
    raw_known_member_ids = json.loads(str(row["known_member_ids"] or "[]"))
    known_member_ids = raw_known_member_ids if isinstance(raw_known_member_ids, list) else []
    return GameParticipant(
        customer_id=str(row["customer_id"]),
        display_name=str(row["display_name"]),
        status=str(row["status"]),
        source=str(row["source"]),
        seat_count=max(1, int(row["seat_count"])),
        party_id=str(row["party_id"] or "") or None,
        known_member_ids=[str(item) for item in known_member_ids],
        anonymous_seat_count=max(0, int(row["anonymous_seat_count"])),
        joined_at=datetime_from_payload(row["joined_at"]),
    )


def game_from_payload(payload: dict[str, Any]) -> Game:
    participants = normalize_game_participants(
        organizer_id=str(payload.get("organizer_id") or ""),
        organizer_name=str(payload.get("organizer_name") or ""),
        known_players=list(payload.get("participants") or []),
    )
    parties = [
        Party(
            party_id=str(item.get("party_id") or f"party_{item.get('contact_id') or item.get('customer_id') or ''}"),
            contact_id=str(item.get("contact_id") or item.get("customer_id") or ""),
            contact_name=str(item.get("contact_name") or item.get("display_name") or item.get("contact_id") or ""),
            seat_count=int(item.get("seat_count") or 1),
            known_member_ids=[str(member) for member in item.get("known_member_ids") or []],
            anonymous_seat_count=int(item.get("anonymous_seat_count") or 0),
            status=str(item.get("status") or "joined"),
            source=str(item.get("source") or "requester"),
        )
        for item in payload.get("parties") or []
        if isinstance(item, dict)
    ]
    return Game(
        game_id=str(payload.get("game_id") or ""),
        conversation_id=str(payload.get("conversation_id") or ""),
        organizer_id=str(payload.get("organizer_id") or ""),
        organizer_name=str(payload.get("organizer_name") or ""),
        requirement=dict(payload.get("requirement") or {}),
        status=GameStatus(str(payload.get("status") or GameStatus.FORMING.value)),
        participants=participants,
        parties=parties,
        seats_total=int(payload.get("seats_total") or 4),
        planned_start_at=optional_datetime_from_payload(payload.get("planned_start_at")),
        planned_end_at=optional_datetime_from_payload(payload.get("planned_end_at")),
        expires_at=optional_datetime_from_payload(payload.get("expires_at") or payload.get("lifecycle_expires_at")),
        recruitment_opens_at=optional_datetime_from_payload(payload.get("recruitment_opens_at")),
        recruitment_status=RecruitmentStatus(
            str(payload.get("recruitment_status") or RecruitmentStatus.OPEN.value)
        ),
        closed_reason=str(payload.get("closed_reason") or ""),
        created_at=datetime_from_payload(payload.get("created_at")),
        updated_at=datetime_from_payload(payload.get("updated_at")),
    )


def invite_from_payload(payload: dict[str, Any]) -> InviteDraft:
    return InviteDraft(
        draft_id=str(payload.get("draft_id") or ""),
        game_id=str(payload.get("game_id") or ""),
        customer_id=str(payload.get("customer_id") or ""),
        display_name=str(payload.get("display_name") or ""),
        message_text=str(payload.get("message_text") or ""),
        status=InviteStatus(str(payload.get("status") or InviteStatus.PENDING_APPROVAL.value)),
        metadata=dict(payload.get("metadata") or {}),
        created_at=datetime_from_payload(payload.get("created_at")),
        updated_at=datetime_from_payload(payload.get("updated_at")),
    )


def outbound_message_draft_from_payload(payload: dict[str, Any]) -> OutboundMessageDraft:
    return OutboundMessageDraft(
        draft_id=str(payload.get("draft_id") or ""),
        conversation_id=str(payload.get("conversation_id") or ""),
        recipient_id=str(payload.get("recipient_id") or ""),
        recipient_name=str(payload.get("recipient_name") or ""),
        channel=str(payload.get("channel") or ""),
        message_text=str(payload.get("message_text") or ""),
        purpose=str(payload.get("purpose") or ""),
        status=OutboundDraftStatus(str(payload.get("status") or OutboundDraftStatus.PENDING_APPROVAL.value)),
        metadata=dict(payload.get("metadata") or {}),
        created_at=datetime_from_payload(payload.get("created_at")),
        updated_at=datetime_from_payload(payload.get("updated_at")),
    )


def room_reservation_from_payload(payload: dict[str, Any]) -> RoomReservation:
    return RoomReservation(
        reservation_id=str(payload.get("reservation_id") or ""),
        room_id=str(payload.get("room_id") or ""),
        conversation_id=str(payload.get("conversation_id") or ""),
        game_id=str(payload.get("game_id") or "") or None,
        start_at=datetime_from_payload(payload.get("start_at")),
        end_at=datetime_from_payload(payload.get("end_at")),
        status=str(payload.get("status") or "held"),
        source_trace_id=str(payload.get("source_trace_id") or "") or None,
        created_at=datetime_from_payload(payload.get("created_at")),
        updated_at=datetime_from_payload(payload.get("updated_at")),
    )


def message_reference_from_payload(payload: dict[str, Any]) -> MessageReference:
    return MessageReference(
        message_id=str(payload.get("message_id") or ""),
        conversation_id=str(payload.get("conversation_id") or ""),
        business_ref_type=str(payload.get("business_ref_type") or ""),
        business_ref_id=str(payload.get("business_ref_id") or ""),
        text=str(payload.get("text") or ""),
        channel=payload.get("channel"),
        sender_id=payload.get("sender_id"),
        sender_name=payload.get("sender_name"),
        recipient_id=payload.get("recipient_id"),
        recipient_name=payload.get("recipient_name"),
        metadata=dict(payload.get("metadata") or {}) if isinstance(payload.get("metadata"), dict) else {},
        created_at=datetime_from_payload(payload.get("created_at")),
    )


def transition_from_payload(payload: dict[str, Any]) -> StateTransition:
    return StateTransition(
        entity_type=str(payload.get("entity_type") or ""),
        entity_id=str(payload.get("entity_id") or ""),
        from_status=payload.get("from_status"),
        to_status=str(payload.get("to_status") or ""),
        reason=str(payload.get("reason") or ""),
        trace_id=str(payload.get("trace_id") or ""),
        occurred_at=datetime_from_payload(payload.get("occurred_at")),
    )


def tool_call_from_payload(payload: dict[str, Any]) -> ToolCall:
    return ToolCall(
        name=str(payload.get("name") or ""),
        arguments=dict(payload.get("arguments") or {}),
        reason=str(payload.get("reason") or ""),
        idempotency_key=payload.get("idempotency_key"),
        call_id=payload.get("call_id"),
        depends_on=(
            [str(item) for item in payload.get("depends_on") or []]
            if isinstance(payload.get("depends_on"), list)
            else None
        ),
    )


def action_from_payload(payload: dict[str, Any]) -> AgentAction:
    return AgentAction(
        goal=str(payload.get("goal") or ""),
        objective_status=str(payload.get("objective_status") or "unknown"),
        reasoning_summary=str(payload.get("reasoning_summary") or ""),
        reply_to_user=str(payload.get("reply_to_user") or ""),
        tool_calls=[
            tool_call_from_payload(item)
            for item in payload.get("tool_calls") or []
            if isinstance(item, dict)
        ],
        needs_human=bool(payload.get("needs_human")),
        stop_reason=dict(payload.get("stop_reason") or {}) if isinstance(payload.get("stop_reason"), dict) else {},
        badcase=payload.get("badcase") if isinstance(payload.get("badcase"), dict) else None,
    )


def tool_result_from_payload(payload: dict[str, Any]) -> ToolResult:
    return ToolResult(
        name=str(payload.get("name") or ""),
        called=bool(payload.get("called")),
        allowed=bool(payload.get("allowed")),
        call_id=payload.get("call_id"),
        result=dict(payload.get("result") or {}),
        error=payload.get("error"),
        idempotency_key=payload.get("idempotency_key"),
        deduplicated=bool(payload.get("deduplicated")),
        state_transitions=[
            transition_from_payload(item)
            for item in payload.get("state_transitions") or []
            if isinstance(item, dict)
        ],
    )


def runtime_result_from_payload(payload: dict[str, Any]) -> AgentRuntimeResult:
    return AgentRuntimeResult(
        trace_id=str(payload.get("trace_id") or ""),
        conversation_id=str(payload.get("conversation_id") or ""),
        final_reply=str(payload.get("final_reply") or ""),
        actions=[action_from_payload(item) for item in payload.get("actions") or [] if isinstance(item, dict)],
        tool_results=[
            tool_result_from_payload(item)
            for item in payload.get("tool_results") or []
            if isinstance(item, dict)
        ],
        state_transitions=[
            transition_from_payload(item)
            for item in payload.get("state_transitions") or []
            if isinstance(item, dict)
        ],
    )


def datetime_from_payload(value: Any) -> datetime:
    if value:
        parsed = datetime.fromisoformat(str(value))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=DEFAULT_TZ)
    return datetime.now(DEFAULT_TZ)


def optional_datetime_from_payload(value: Any) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(str(value))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=DEFAULT_TZ)


def dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def loads(payload: str) -> dict[str, Any]:
    raw = json.loads(payload)
    return raw if isinstance(raw, dict) else {}


def now_iso() -> str:
    return datetime.now(DEFAULT_TZ).isoformat()


# Compatibility aliases keep the existing private helper names available while
# callers are migrated incrementally.
_action_from_payload = action_from_payload
_checkpoint_from_payload = checkpoint_from_payload
_customer_from_payload = customer_from_payload
_datetime_from_payload = datetime_from_payload
_dumps = dumps
_game_from_payload = game_from_payload
_game_participant_from_row = game_participant_from_row
_game_storage_payload = game_storage_payload
_invite_from_payload = invite_from_payload
_loads = loads
_message_reference_from_payload = message_reference_from_payload
_now_iso = now_iso
_optional_datetime_from_payload = optional_datetime_from_payload
_outbound_message_draft_from_payload = outbound_message_draft_from_payload
_pending_input_batch_from_payload = pending_input_batch_from_payload
_pending_memory_candidate_from_payload = pending_memory_candidate_from_payload
_relationship_from_payload = relationship_from_payload
_room_reservation_from_payload = room_reservation_from_payload
_runtime_result_from_payload = runtime_result_from_payload
_scheduled_agent_task_from_payload = scheduled_agent_task_from_payload
_task_context_from_payload = task_context_from_payload
_task_memory_from_payload = task_memory_from_payload
_tool_call_from_payload = tool_call_from_payload
_tool_result_from_payload = tool_result_from_payload
_transition_from_payload = transition_from_payload
_turn_from_payload = turn_from_payload
