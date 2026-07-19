from __future__ import annotations

import copy
import re
import threading
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from typing import Any

from .models import (
    AgentRuntimeResult,
    ConversationCheckpoint,
    ConversationRole,
    ConversationTaskContext,
    ConversationTurn,
    CustomerProfile,
    CustomerRelationship,
    DEFAULT_TZ,
    GameParticipant,
    GameStatus,
    Game,
    InviteDraft,
    InviteStatus,
    MessageReference,
    OPEN_INVITE_STATUSES,
    OutboundDraftStatus,
    OutboundMessageDraft,
    Party,
    PendingInputBatch,
    PendingInputBatchStatus,
    PendingMemoryCandidate,
    RoomReservation,
    StateTransition,
    TaskMemory,
    ToolResult,
    new_id,
    now,
)


DEFAULT_ASAP_GAME_TTL_HOURS = 4
DEFAULT_UNKNOWN_DURATION_HOURS = 4
DEFAULT_OVERNIGHT_DURATION_HOURS = 8
START_KIND_SCHEDULED = "scheduled"
START_KIND_ASAP_WHEN_FULL = "asap_" "when_full"
DURATION_KIND_OVERNIGHT = "overnight"
PENDING_INPUT_PROCESSING_LEASE_SECONDS = 120
IDEMPOTENCY_CLAIM_LEASE_SECONDS = 120

ALLOWED_GAME_TRANSITIONS = {
    GameStatus.FORMING.value: {
        GameStatus.INVITING.value,
        GameStatus.READY.value,
        GameStatus.CANCELLED.value,
    },
    GameStatus.INVITING.value: {
        GameStatus.READY.value,
        GameStatus.CANCELLED.value,
        GameStatus.FINISHED.value,
    },
    GameStatus.READY.value: {GameStatus.FINISHED.value, GameStatus.CANCELLED.value},
    GameStatus.CANCELLED.value: set(),
    GameStatus.FINISHED.value: set(),
}

CONFIRMED_CANDIDATE_STATUSES = {"accepted", "confirmed", "arrived"}
UNCONFIRMED_CANDIDATE_STATUSES = {"declined", "negotiating", "no_reply"}
PROTECTED_REQUIREMENT_PATCH_FIELDS = {
    "current_player_count",
    "current_player_ids",
    "existing_player_ids",
    "known_player_count",
    "known_player_ids",
    "needed_seats",
    "organizer_id",
    "participant_ids",
    "party_size",
    "player_count",
    "remaining_seats",
    "requester_id",
    "requesting_party",
    "parties",
    "seat_claims",
    "seats_total",
    "planned_end_at",
    "lifecycle_expires_at",
    "lifecycle_ttl_hours",
    "latest_start_at",
}


@dataclass(slots=True)
class GameCommitmentResolution:
    """Atomic result of committing shared participants to the first full game."""

    winner_game_id: str | None
    blocked_by_game_ids: list[str] = field(default_factory=list)
    released_customer_ids: list[str] = field(default_factory=list)
    affected_game_ids: list[str] = field(default_factory=list)
    changed_games: list[Game] = field(default_factory=list)
    changed_invites: list[InviteDraft] = field(default_factory=list)
    transitions: list[StateTransition] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "winner_game_id": self.winner_game_id,
            "blocked_by_game_ids": list(self.blocked_by_game_ids),
            "released_customer_ids": list(self.released_customer_ids),
            "affected_game_ids": list(self.affected_game_ids),
        }


def active_game_participant_ids(game: Game) -> set[str]:
    return {
        str(participant.customer_id)
        for participant in game.participants
        if participant.customer_id and participant.status in {"joined", "confirmed"}
    }


def game_commitment_window(game: Game) -> tuple[datetime, datetime]:
    """Return the time range in which this game competes for a participant.

    Scheduled games use their playing interval. A game whose start time is still
    flexible can start at any point before expiry, so its conservative window
    also includes the expected duration after the latest possible start.
    """

    start = game.planned_start_at or game.created_at
    if game.planned_end_at is not None:
        end = game.planned_end_at
    else:
        latest_start = game.expires_at or (game.created_at + timedelta(hours=DEFAULT_ASAP_GAME_TTL_HOURS))
        end = latest_start + timedelta(hours=duration_hours_from_requirement(game.requirement))
    return normalize_datetime(start), normalize_datetime(end)


def game_commitment_windows_overlap(first: Game, second: Game) -> bool:
    if first.game_id == second.game_id:
        return False
    first_start, first_end = game_commitment_window(first)
    second_start, second_end = game_commitment_window(second)
    return first_start < second_end and second_start < first_end


def requirement_commitment_window(
    requirement: dict[str, Any],
    *,
    reference_at: datetime | None = None,
) -> tuple[datetime, datetime]:
    """Build a conservative competition window before a Game aggregate exists."""

    created_at = normalize_datetime(reference_at or now())
    lifecycle = derive_game_lifecycle(requirement, created_at=created_at)
    start = lifecycle["planned_start_at"] or created_at
    end = lifecycle["planned_end_at"]
    if end is None:
        end = lifecycle["expires_at"] + timedelta(hours=duration_hours_from_requirement(requirement))
    return normalize_datetime(start), normalize_datetime(end)


def requirement_overlaps_game(requirement: dict[str, Any], game: Game) -> bool:
    requested_start, requested_end = requirement_commitment_window(requirement)
    game_start, game_end = game_commitment_window(game)
    return requested_start < game_end and game_start < requested_end


def ready_commitment_conflicts(
    candidate_game: Game,
    customer_ids: set[str],
    games: list[Game],
) -> list[Game]:
    """Return READY games that already own any requested participant's time."""

    if not customer_ids:
        return []
    conflicts = []
    for other in games:
        if (
            other.game_id == candidate_game.game_id
            or other.status != GameStatus.READY
            or not game_commitment_windows_overlap(candidate_game, other)
        ):
            continue
        if customer_ids.intersection(active_game_participant_ids(other)):
            conflicts.append(other)
    return sorted(conflicts, key=lambda item: (item.updated_at, item.game_id))


def customer_option_load(
    customer_id: str,
    requirement: dict[str, Any],
    games: list[Game],
) -> tuple[bool, int]:
    """Return whether time is committed and how many provisional options overlap."""

    committed = False
    provisional_count = 0
    for game in games:
        if (
            not any(
                participant.customer_id == customer_id and participant.status in {"joined", "confirmed"}
                for participant in game.participants
            )
            or not requirement_overlaps_game(requirement, game)
        ):
            continue
        if game.status == GameStatus.READY:
            committed = True
        elif game.status in {GameStatus.FORMING, GameStatus.INVITING}:
            provisional_count += 1
    return committed, provisional_count


def _release_game_participants(
    game: Game,
    customer_ids: set[str],
    *,
    committed_game_id: str,
    invite_drafts: list[InviteDraft],
    trace_id: str,
) -> tuple[list[InviteDraft], list[StateTransition]]:
    """Release shared seats from one losing option while retaining an audit trail."""

    released = [
        participant
        for participant in game.participants
        if participant.customer_id in customer_ids and participant.status in {"joined", "confirmed"}
    ]
    if not released:
        return [], []
    transitions: list[StateTransition] = []
    for participant in released:
        old_status = participant.status
        participant.status = "superseded"
        transitions.append(
            StateTransition(
                "game_participant",
                f"{game.game_id}:{participant.customer_id}",
                f"{old_status}:seats={max(1, int(participant.seat_count))}",
                "superseded",
                f"participant_committed_to_game:{committed_game_id}",
                trace_id,
            )
        )
    changed_invites: list[InviteDraft] = []
    for draft in invite_drafts:
        if (
            draft.game_id != game.game_id
            or draft.customer_id not in customer_ids
            or draft.status not in OPEN_INVITE_STATUSES
        ):
            continue
        old = draft.status.value
        draft.status = InviteStatus.SUPERSEDED
        draft.updated_at = now()
        changed_invites.append(draft)
        transitions.append(
            StateTransition(
                "invite_draft",
                draft.draft_id,
                old,
                draft.status.value,
                f"participant_committed_to_game:{committed_game_id}",
                trace_id,
            )
        )
    game.parties = normalize_game_parties(game.participants)
    game.requirement = refresh_requirement_seat_snapshot(game.requirement, game.parties, game.remaining_seats())
    game.updated_at = now()
    return changed_invites, transitions


def resolve_full_game_commitments(
    candidate_game: Game,
    *,
    games: list[Game],
    invite_drafts: list[InviteDraft],
    trace_id: str,
) -> GameCommitmentResolution:
    """Commit shared participants to whichever overlapping game becomes full first.

    A customer may provisionally appear in any number of forming games. Once one
    overlapping option becomes READY, this function releases that customer from
    every other overlapping forming option in the same write transaction. If a
    READY option already exists, the current candidate loses those shared seats.
    """

    candidate_ids = active_game_participant_ids(candidate_game)
    ready_conflicts = ready_commitment_conflicts(candidate_game, candidate_ids, games)

    resolution = GameCommitmentResolution(winner_game_id=None)
    changed_games: dict[str, Game] = {}
    changed_invites: dict[str, InviteDraft] = {}
    if ready_conflicts:
        commitment_owner: dict[str, str] = {}
        for ready_game in ready_conflicts:
            for customer_id in candidate_ids.intersection(active_game_participant_ids(ready_game)):
                commitment_owner.setdefault(customer_id, ready_game.game_id)
        for committed_game_id in sorted(set(commitment_owner.values())):
            committed_ids = {
                customer_id
                for customer_id, owner_game_id in commitment_owner.items()
                if owner_game_id == committed_game_id
            }
            invites, transitions = _release_game_participants(
                candidate_game,
                committed_ids,
                committed_game_id=committed_game_id,
                invite_drafts=invite_drafts,
                trace_id=trace_id,
            )
            changed_invites.update({item.draft_id: item for item in invites})
            resolution.transitions.extend(transitions)
        changed_games[candidate_game.game_id] = candidate_game
        resolution.blocked_by_game_ids = sorted(game.game_id for game in ready_conflicts)
        resolution.released_customer_ids = sorted(commitment_owner)
    else:
        old = candidate_game.status.value
        candidate_game.status = GameStatus.READY
        candidate_game.updated_at = now()
        changed_games[candidate_game.game_id] = candidate_game
        resolution.winner_game_id = candidate_game.game_id
        resolution.transitions.append(
            StateTransition("game", candidate_game.game_id, old, candidate_game.status.value, "seats_full", trace_id)
        )
        for other in games:
            if (
                other.game_id == candidate_game.game_id
                or other.status not in {GameStatus.FORMING, GameStatus.INVITING}
                or not game_commitment_windows_overlap(candidate_game, other)
            ):
                continue
            shared_ids = candidate_ids.intersection(active_game_participant_ids(other))
            if not shared_ids:
                continue
            invites, transitions = _release_game_participants(
                other,
                shared_ids,
                committed_game_id=candidate_game.game_id,
                invite_drafts=invite_drafts,
                trace_id=trace_id,
            )
            changed_games[other.game_id] = other
            changed_invites.update({item.draft_id: item for item in invites})
            resolution.released_customer_ids.extend(sorted(shared_ids))
            resolution.transitions.extend(transitions)

    resolution.released_customer_ids = sorted(set(resolution.released_customer_ids))
    resolution.changed_games = list(changed_games.values())
    resolution.changed_invites = list(changed_invites.values())
    resolution.affected_game_ids = sorted(changed_games)
    return resolution


def pending_input_batch_key(conversation_id: str, sender_id: str) -> str:
    """Stable scope key; group members never share unfinished fragments."""

    return f"{conversation_id or 'default'}\x1f{sender_id or 'unknown'}"


def tool_result_is_in_progress(result: ToolResult) -> bool:
    return bool(
        not result.called
        and result.allowed
        and isinstance(result.result, dict)
        and result.result.get("idempotency_status") == "claimed"
    )


@dataclass(slots=True)
class InMemoryAgentStore:
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
    badcases: list[dict[str, Any]] = field(default_factory=list)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    def upsert_customer(self, profile: CustomerProfile) -> None:
        with self._lock:
            self.customers[profile.customer_id] = profile

    def configure_rooms(self, room_ids: list[str]) -> None:
        with self._lock:
            self.room_ids = list(dict.fromkeys(str(item).strip() for item in room_ids if str(item).strip()))

    def search_room_availability(self, *, start_at: Any, end_at: Any) -> dict[str, Any]:
        start = parse_datetime_value(start_at)
        end = parse_datetime_value(end_at)
        if start is None or end is None or end <= start:
            raise ValueError("start_at and end_at must be valid datetimes with end_at after start_at")
        with self._lock:
            occupied = {
                item.room_id
                for item in self.room_reservations.values()
                if item.status in {"held", "confirmed"} and item.start_at < end and item.end_at > start
            }
            available = [room_id for room_id in self.room_ids if room_id not in occupied]
            return {
                "configured": bool(self.room_ids),
                "start_at": start.isoformat(),
                "end_at": end.isoformat(),
                "room_count": len(self.room_ids),
                "available_room_ids": available,
                "occupied_room_ids": sorted(occupied),
                "available_count": len(available),
            }

    def reserve_room(
        self,
        *,
        conversation_id: str,
        game_id: str | None,
        start_at: Any,
        end_at: Any,
        room_id: str | None,
        trace_id: str,
    ) -> tuple[RoomReservation, StateTransition]:
        availability = self.search_room_availability(start_at=start_at, end_at=end_at)
        if not availability["configured"]:
            raise ValueError("room inventory is not configured")
        chosen = str(room_id or "").strip()
        available = list(availability["available_room_ids"])
        if chosen and chosen not in available:
            raise ValueError(f"room is unavailable: {chosen}")
        if not chosen:
            if not available:
                raise ValueError("no room is available for the requested interval")
            chosen = available[0]
        reservation = RoomReservation(
            reservation_id=new_id("room_reservation"),
            room_id=chosen,
            conversation_id=conversation_id,
            game_id=game_id,
            start_at=parse_datetime_value(start_at) or now(),
            end_at=parse_datetime_value(end_at) or now(),
            source_trace_id=trace_id,
        )
        transition = StateTransition(
            "room_reservation",
            reservation.reservation_id,
            None,
            reservation.status,
            "reserve_room",
            trace_id,
        )
        with self._lock:
            # Recheck under the mutation lock to avoid two local callers taking
            # the same room after a shared availability snapshot.
            latest = self.search_room_availability(start_at=start_at, end_at=end_at)
            if chosen not in latest["available_room_ids"]:
                raise ValueError(f"room is unavailable: {chosen}")
            self.room_reservations[reservation.reservation_id] = reservation
            self.transitions.append(transition)
        return reservation, transition

    def upsert_customer_relationship(self, relationship: CustomerRelationship) -> None:
        with self._lock:
            self.customer_relationships[relationship_pair_key(relationship.customer_a_id, relationship.customer_b_id)] = relationship

    def relationship_between(self, customer_id: str, other_customer_id: str) -> CustomerRelationship | None:
        with self._lock:
            return self.customer_relationships.get(relationship_pair_key(customer_id, other_customer_id))

    def relationship_context_for_sender(self, sender_id: str, games: list[Game]) -> list[dict[str, Any]]:
        with self._lock:
            return relationship_context_for_sender(
                sender_id=sender_id,
                games=games,
                customers=self.customers,
                relationship_lookup=self.relationship_between,
            )

    def append_user_turn(self, message, trace_id: str) -> None:
        task_context = self.current_task_context(message.conversation_id, message.sender_id)
        metadata = dict(getattr(message, "metadata", {}) or {})
        if task_context is not None:
            metadata["task_context_id"] = task_context.task_context_id
        self.append_turn(
            message.conversation_id,
            ConversationTurn(
                role=ConversationRole.USER,
                content=message.text,
                trace_id=trace_id,
                sender_id=message.sender_id,
                sender_name=message.sender_name,
                metadata=metadata,
                occurred_at=message.sent_at,
            ),
        )

    def append_assistant_turn(
        self,
        conversation_id: str,
        text: str,
        trace_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        task_context = self.latest_task_context(conversation_id)
        turn_metadata = dict(metadata or {})
        if task_context is not None:
            turn_metadata.setdefault("task_context_id", task_context.task_context_id)
        self.append_turn(
            conversation_id,
            ConversationTurn(
                role=ConversationRole.ASSISTANT,
                content=text,
                trace_id=trace_id,
                metadata=turn_metadata,
            ),
        )

    def append_tool_turn(self, conversation_id: str, text: str, trace_id: str) -> None:
        task_context = self.latest_task_context(conversation_id)
        metadata = {"task_context_id": task_context.task_context_id} if task_context else {}
        self.append_turn(
            conversation_id,
            ConversationTurn(role=ConversationRole.TOOL, content=text, trace_id=trace_id, metadata=metadata),
        )

    def append_turn(self, conversation_id: str, turn: ConversationTurn) -> None:
        with self._lock:
            self.turns.setdefault(conversation_id, []).append(turn)

    def recent_turns(self, conversation_id: str, limit: int = 30) -> list[ConversationTurn]:
        with self._lock:
            return list(self.turns.get(conversation_id, []))[-int(limit):]

    def get_conversation_checkpoint(self, conversation_id: str) -> ConversationCheckpoint | None:
        with self._lock:
            return self.conversation_checkpoints.get(conversation_id)

    def current_task_context(self, conversation_id: str, customer_id: str) -> ConversationTaskContext | None:
        with self._lock:
            matches = [
                item
                for item in self.task_contexts.values()
                if item.status == "active"
                and item.conversation_id == conversation_id
                and item.customer_id == customer_id
            ]
            return max(matches, key=lambda item: item.updated_at) if matches else None

    def latest_task_context(self, conversation_id: str) -> ConversationTaskContext | None:
        with self._lock:
            matches = [
                item
                for item in self.task_contexts.values()
                if item.status == "active" and item.conversation_id == conversation_id
            ]
            return max(matches, key=lambda item: item.updated_at) if matches else None

    def activate_task_context(
        self,
        *,
        conversation_id: str,
        customer_id: str,
        trace_id: str,
        activity_at: datetime,
        started_at: datetime,
        reason: str,
        force_new: bool,
        archive_previous: bool,
    ) -> tuple[ConversationTaskContext, list[StateTransition]]:
        """Create/reuse one business episode and retire its temporary memory on reset."""

        with self._lock:
            transitions: list[StateTransition] = []
            previous = self.current_task_context(conversation_id, customer_id)
            if previous is not None and not force_new:
                previous.updated_at = activity_at
                previous.source_trace_id = trace_id
                return previous, transitions

            if previous is not None:
                previous.status = "closed"
                previous.closed_at = activity_at
                previous.updated_at = activity_at
                transitions.append(
                    StateTransition(
                        "task_context",
                        previous.task_context_id,
                        "active",
                        "closed",
                        reason,
                        trace_id,
                    )
                )

            if archive_previous:
                previous_context_id = previous.task_context_id if previous else None
                for memory in self.task_memories.values():
                    if memory.status != "active" or memory.conversation_id != conversation_id:
                        continue
                    if memory.customer_id != customer_id and memory.target_customer_id != customer_id:
                        continue
                    memory_context_id = str(memory.metadata.get("task_context_id") or "")
                    belongs_to_previous = bool(previous_context_id and memory_context_id == previous_context_id)
                    predates_new_context = memory.updated_at < started_at
                    if not belongs_to_previous and not predates_new_context:
                        continue
                    memory.status = "archived"
                    memory.updated_at = activity_at
                    transitions.append(
                        StateTransition(
                            "task_memory",
                            memory.memory_id,
                            "active",
                            "archived",
                            "task_context_reset",
                            trace_id,
                        )
                    )

            context = ConversationTaskContext(
                task_context_id=new_id("task_context"),
                conversation_id=conversation_id,
                customer_id=customer_id,
                reset_reason=reason,
                previous_task_context_id=previous.task_context_id if previous else None,
                source_trace_id=trace_id,
                started_at=started_at,
                updated_at=activity_at,
            )
            self.task_contexts[context.task_context_id] = context
            transitions.append(
                StateTransition(
                    "task_context",
                    context.task_context_id,
                    None,
                    "active",
                    reason,
                    trace_id,
                )
            )
            self.transitions.extend(transitions)
            return context, transitions

    def conversation_version(self, conversation_id: str) -> int:
        with self._lock:
            return int(self.conversation_versions.get(conversation_id or "default", 0))

    def advance_conversation_version(
        self,
        conversation_id: str,
        *,
        trace_id: str,
        reason: str,
    ) -> tuple[int, StateTransition]:
        key = conversation_id or "default"
        with self._lock:
            old = int(self.conversation_versions.get(key, 0))
            new = old + 1
            self.conversation_versions[key] = new
            transition = StateTransition(
                entity_type="conversation_version",
                entity_id=key,
                from_status=str(old),
                to_status=str(new),
                reason=reason or "user_message_received",
                trace_id=trace_id,
            )
            self.transitions.append(transition)
            return new, transition

    def supersede_pending_outputs(
        self,
        conversation_id: str,
        *,
        sender_id: str | None = None,
        trace_id: str,
        reason: str,
    ) -> tuple[dict[str, int], list[StateTransition]]:
        key = conversation_id or "default"
        with self._lock:
            transitions: list[StateTransition] = []
            counts = {
                "invite_drafts": 0,
                "outbound_message_drafts": 0,
                "assistant_replies": 0,
            }
            game_ids = {game.game_id for game in self.games.values() if game.conversation_id == key}
            sender_is_pending_candidate = bool(
                sender_id
                and any(
                    draft.game_id in game_ids
                    and draft.customer_id == sender_id
                    and draft.status == InviteStatus.PENDING_APPROVAL
                    for draft in self.invite_drafts.values()
                )
            )
            for draft in self.invite_drafts.values():
                if sender_is_pending_candidate:
                    continue
                if draft.game_id not in game_ids or draft.status != InviteStatus.PENDING_APPROVAL:
                    continue
                old = draft.status.value
                draft.status = InviteStatus.SUPERSEDED
                draft.updated_at = now()
                draft.metadata = {
                    **dict(draft.metadata),
                    "superseded_by_trace_id": trace_id,
                    "superseded_reason": reason,
                }
                counts["invite_drafts"] += 1
                transitions.append(
                    StateTransition("invite_draft", draft.draft_id, old, draft.status.value, reason, trace_id)
                )
            for draft in self.outbound_message_drafts.values():
                if sender_is_pending_candidate:
                    continue
                if draft.conversation_id != key or draft.status != OutboundDraftStatus.PENDING_APPROVAL:
                    continue
                old = draft.status.value
                draft.status = OutboundDraftStatus.SUPERSEDED
                draft.updated_at = now()
                draft.metadata = {
                    **dict(draft.metadata),
                    "superseded_by_trace_id": trace_id,
                    "superseded_reason": reason,
                }
                counts["outbound_message_drafts"] += 1
                transitions.append(
                    StateTransition("outbound_message_draft", draft.draft_id, old, draft.status.value, reason, trace_id)
                )
            for turn in self.turns.get(key, []):
                if turn.role != ConversationRole.ASSISTANT:
                    continue
                if turn.metadata.get("delivery_status") != "pending_operator_send":
                    continue
                old = str(turn.metadata.get("delivery_status") or "")
                turn.metadata = {
                    **dict(turn.metadata),
                    "delivery_status": "superseded",
                    "superseded_by_trace_id": trace_id,
                    "superseded_reason": reason,
                }
                counts["assistant_replies"] += 1
                transitions.append(StateTransition("assistant_reply", turn.trace_id, old, "superseded", reason, trace_id))
            self.transitions.extend(transitions)
            return counts, transitions

    def upsert_conversation_checkpoint(
        self,
        *,
        conversation_id: str,
        summary: str,
        facts: dict[str, Any],
        open_questions: list[str],
        trace_id: str,
    ) -> tuple[ConversationCheckpoint, StateTransition]:
        with self._lock:
            previous = self.conversation_checkpoints.get(conversation_id)
            task_context = self.latest_task_context(conversation_id)
            checkpoint = ConversationCheckpoint(
                conversation_id=conversation_id,
                summary=summary,
                facts=dict(facts),
                open_questions=list(open_questions),
                task_context_id=task_context.task_context_id if task_context else None,
                source_trace_id=trace_id,
            )
            self.conversation_checkpoints[conversation_id] = checkpoint
            transition = StateTransition(
                entity_type="conversation_checkpoint",
                entity_id=conversation_id,
                from_status="exists" if previous else None,
                to_status="updated",
                reason="update_context_checkpoint",
                trace_id=trace_id,
            )
            self.transitions.append(transition)
            return checkpoint, transition

    def record_task_memory(
        self,
        *,
        conversation_id: str,
        customer_id: str,
        memory_type: str,
        field: str,
        value: Any,
        target_customer_id: str | None = None,
        evidence: str = "",
        confidence: float = 0.0,
        risk_level: str = "medium",
        scope: str = "current_task",
        metadata: dict[str, Any] | None = None,
        trace_id: str,
    ) -> tuple[TaskMemory, StateTransition]:
        with self._lock:
            task_context = self.current_task_context(conversation_id, customer_id)
            memory_metadata = dict(metadata or {})
            if task_context is not None:
                memory_metadata.setdefault("task_context_id", task_context.task_context_id)
            memory = TaskMemory(
                memory_id=new_id("task_memory"),
                conversation_id=conversation_id,
                customer_id=customer_id,
                memory_type=memory_type,
                field=field,
                value=value,
                target_customer_id=target_customer_id,
                evidence=evidence,
                confidence=float(confidence or 0.0),
                risk_level=risk_level or "medium",
                scope=scope or "current_task",
                source_trace_id=trace_id,
                metadata=memory_metadata,
            )
            self.task_memories[memory.memory_id] = memory
            transition = StateTransition(
                entity_type="task_memory",
                entity_id=memory.memory_id,
                from_status=None,
                to_status=memory.status,
                reason="record_user_memory",
                trace_id=trace_id,
            )
            self.transitions.append(transition)
            return memory, transition

    def record_pending_memory_candidate(
        self,
        *,
        conversation_id: str,
        customer_id: str,
        memory_type: str,
        field: str,
        value: Any,
        operation: str = "set",
        target_customer_id: str | None = None,
        evidence: str = "",
        confidence: float = 0.0,
        risk_level: str = "medium",
        scope: str = "long_term",
        metadata: dict[str, Any] | None = None,
        trace_id: str,
    ) -> tuple[PendingMemoryCandidate, StateTransition]:
        with self._lock:
            task_context = self.current_task_context(conversation_id, customer_id)
            candidate_metadata = dict(metadata or {})
            if task_context is not None:
                candidate_metadata.setdefault("task_context_id", task_context.task_context_id)
            candidate = PendingMemoryCandidate(
                candidate_id=new_id("memory_candidate"),
                conversation_id=conversation_id,
                customer_id=customer_id,
                memory_type=memory_type,
                field=field,
                value=value,
                operation=operation or "set",
                target_customer_id=target_customer_id,
                evidence=evidence,
                confidence=float(confidence or 0.0),
                risk_level=risk_level or "medium",
                scope=scope or "long_term",
                source_trace_id=trace_id,
                metadata=candidate_metadata,
            )
            self.pending_memory_candidates[candidate.candidate_id] = candidate
            transition = StateTransition(
                entity_type="pending_memory_candidate",
                entity_id=candidate.candidate_id,
                from_status=None,
                to_status=candidate.status,
                reason="record_user_memory",
                trace_id=trace_id,
            )
            self.transitions.append(transition)
            return candidate, transition

    def task_memory_context(self, conversation_id: str, customer_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            task_context = self.current_task_context(conversation_id, customer_id or "") if customer_id else None
            memories = [
                item.to_dict()
                for item in self.task_memories.values()
                if item.status == "active"
                and item.conversation_id == conversation_id
                and (not customer_id or item.customer_id == customer_id or item.target_customer_id == customer_id)
                and (
                    task_context is None
                    or item.metadata.get("task_context_id") == task_context.task_context_id
                    or (
                        not item.metadata.get("task_context_id")
                        and item.updated_at >= task_context.started_at
                    )
                )
            ]
            memories.sort(key=lambda item: str(item.get("updated_at") or ""), reverse=True)
            return memories

    def pending_memory_candidates_for_context(
        self,
        conversation_id: str,
        customer_id: str | None = None,
        *,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        with self._lock:
            task_context = self.current_task_context(conversation_id, customer_id or "") if customer_id else None
            candidates = [
                item.to_dict()
                for item in self.pending_memory_candidates.values()
                if item.status == "pending_review"
                and item.conversation_id == conversation_id
                and (not customer_id or item.customer_id == customer_id or item.target_customer_id == customer_id)
                and (
                    task_context is None
                    or item.metadata.get("task_context_id") == task_context.task_context_id
                    or (
                        not item.metadata.get("task_context_id")
                        and item.updated_at >= task_context.started_at
                    )
                )
            ]
            candidates.sort(key=lambda item: str(item.get("updated_at") or ""), reverse=True)
            return candidates[: int(limit)]

    def task_memory_excluded_customer_ids(
        self,
        conversation_id: str | None,
        anchor_ids: list[str] | set[str] | None,
    ) -> list[str]:
        if not conversation_id:
            return []
        anchors = {str(item) for item in anchor_ids or [] if str(item or "").strip()}
        if not anchors:
            return []
        with self._lock:
            excluded: list[str] = []
            for item in self.task_memories.values():
                if item.status != "active" or item.conversation_id != conversation_id:
                    continue
                if item.customer_id not in anchors:
                    continue
                task_context = self.current_task_context(conversation_id, item.customer_id)
                memory_context_id = str(item.metadata.get("task_context_id") or "")
                if task_context is not None and not (
                    memory_context_id == task_context.task_context_id
                    or (not memory_context_id and item.updated_at >= task_context.started_at)
                ):
                    continue
                if not is_avoid_playing_memory(item):
                    continue
                target_id = str(item.target_customer_id or "")
                if target_id and target_id not in excluded:
                    excluded.append(target_id)
            return excluded

    def active_games(self, conversation_id: str | None = None) -> list[Game]:
        with self._lock:
            self._expire_stale_games_locked(trace_id="system_lifecycle")
            games = [
                item
                for item in self.games.values()
                if item.status.value in {GameStatus.FORMING.value, GameStatus.INVITING.value, GameStatus.READY.value}
            ]
            if conversation_id:
                return [item for item in games if item.conversation_id == conversation_id]
            return games

    def idempotent_result(self, key: str | None) -> ToolResult | None:
        with self._lock:
            normalized_key = key or ""
            existing = self.idempotency_ledger.get(normalized_key)
            claimed_at = self.idempotency_claimed_at.get(normalized_key)
            if existing is not None and tool_result_is_in_progress(existing) and claimed_at is not None:
                if claimed_at <= now() - timedelta(seconds=IDEMPOTENCY_CLAIM_LEASE_SECONDS):
                    self.idempotency_ledger.pop(normalized_key, None)
                    self.idempotency_claimed_at.pop(normalized_key, None)
                    return None
            return existing

    def claim_idempotent_result(self, key: str | None, claimed_result: ToolResult) -> tuple[bool, ToolResult | None]:
        if not key:
            return True, None
        with self._lock:
            existing = self.idempotency_ledger.get(key)
            if existing is not None:
                claimed_at = self.idempotency_claimed_at.get(key)
                if not (
                    tool_result_is_in_progress(existing)
                    and claimed_at is not None
                    and claimed_at <= now() - timedelta(seconds=IDEMPOTENCY_CLAIM_LEASE_SECONDS)
                ):
                    return False, existing
            self.idempotency_ledger[key] = claimed_result
            self.idempotency_claimed_at[key] = now()
            return True, None

    def remember_result(self, key: str | None, result: ToolResult) -> None:
        if not key:
            return
        with self._lock:
            self.idempotency_ledger[key] = result
            self.idempotency_claimed_at.pop(key, None)

    def idempotent_message_result(self, message_id: str | None) -> AgentRuntimeResult | None:
        with self._lock:
            return self.message_results.get(message_id or "")

    def remember_message_result(self, message_id: str | None, result: AgentRuntimeResult) -> None:
        if not message_id:
            return
        with self._lock:
            self.message_results.setdefault(message_id, result)

    def upsert_pending_input_fragment(
        self,
        message,
        *,
        trace_id: str,
        quiet_deadline: datetime,
    ) -> tuple[PendingInputBatch, StateTransition | None, bool]:
        """Append one raw fragment and atomically move the batch deadline.

        Repeated platform message ids are ignored. A fragment arriving while a
        delayed worker is evaluating the old version advances ``version`` and
        returns the batch to ``pending``, making the old worker stale.
        """

        key = pending_input_batch_key(message.conversation_id, message.sender_id)
        fragment = message.to_dict()
        with self._lock:
            existing = self.pending_input_batches.get(key)
            message_id = str(fragment.get("message_id") or "")
            if existing is not None and message_id and any(
                str(item.get("message_id") or "") == message_id for item in existing.fragments
            ):
                return copy.deepcopy(existing), None, False
            if existing is None or existing.status in {
                PendingInputBatchStatus.COMPLETED,
                PendingInputBatchStatus.IGNORED,
                PendingInputBatchStatus.FAILED,
            }:
                batch = PendingInputBatch(
                    batch_id=new_id("input_batch"),
                    conversation_id=message.conversation_id,
                    sender_id=message.sender_id,
                    sender_name=message.sender_name,
                    fragments=[fragment],
                    quiet_deadline=quiet_deadline,
                    source_channel=str(message.metadata.get("channel") or ""),
                )
                previous_status = "absent"
            else:
                batch = existing
                previous_status = batch.status.value
                batch.fragments.append(fragment)
                batch.sender_name = message.sender_name or batch.sender_name
                batch.version += 1
                batch.status = PendingInputBatchStatus.PENDING
                batch.quiet_deadline = quiet_deadline
                batch.updated_at = now()
                batch.decision = {}
                if message.metadata.get("channel"):
                    batch.source_channel = str(message.metadata["channel"])
            self.pending_input_batches[key] = batch
            transition = StateTransition(
                entity_type="pending_input_batch",
                entity_id=batch.batch_id,
                from_status=previous_status,
                to_status=batch.status.value,
                reason="input_fragment_buffered",
                trace_id=trace_id,
            )
            self.transitions.append(transition)
            return copy.deepcopy(batch), transition, True

    def pending_input_batch(self, conversation_id: str, sender_id: str) -> PendingInputBatch | None:
        with self._lock:
            batch = self.pending_input_batches.get(pending_input_batch_key(conversation_id, sender_id))
            return copy.deepcopy(batch) if batch is not None else None

    def due_pending_input_batches(self, *, at: datetime, limit: int = 100) -> list[PendingInputBatch]:
        with self._lock:
            stale_before = at - timedelta(seconds=PENDING_INPUT_PROCESSING_LEASE_SECONDS)
            due = [
                item
                for item in self.pending_input_batches.values()
                if (
                    item.status == PendingInputBatchStatus.PENDING and item.quiet_deadline <= at
                )
                or (
                    item.status == PendingInputBatchStatus.PROCESSING and item.updated_at <= stale_before
                )
            ]
            return copy.deepcopy(sorted(due, key=lambda item: item.quiet_deadline)[: max(1, int(limit))])

    def claim_pending_input_batch(
        self,
        *,
        batch_id: str,
        expected_version: int,
        trace_id: str,
    ) -> tuple[PendingInputBatch | None, StateTransition | None]:
        """Claim the exact batch version; stale model decisions cannot dispatch."""

        with self._lock:
            batch = next((item for item in self.pending_input_batches.values() if item.batch_id == batch_id), None)
            stale_before = now() - timedelta(seconds=PENDING_INPUT_PROCESSING_LEASE_SECONDS)
            if (
                batch is None
                or batch.version != int(expected_version)
                or (
                    batch.status != PendingInputBatchStatus.PENDING
                    and not (
                        batch.status == PendingInputBatchStatus.PROCESSING
                        and batch.updated_at <= stale_before
                    )
                )
            ):
                return None, None
            old = batch.status.value
            batch.status = PendingInputBatchStatus.PROCESSING
            batch.updated_at = now()
            transition = StateTransition(
                "pending_input_batch",
                batch.batch_id,
                old,
                batch.status.value,
                "input_batch_claimed",
                trace_id,
            )
            self.transitions.append(transition)
            return copy.deepcopy(batch), transition

    def record_pending_input_decision(
        self,
        *,
        batch_id: str,
        expected_version: int,
        decision: dict[str, Any],
    ) -> PendingInputBatch | None:
        """Attach the model boundary decision without changing batch status."""

        with self._lock:
            batch = next((item for item in self.pending_input_batches.values() if item.batch_id == batch_id), None)
            if batch is None or batch.version != int(expected_version):
                return None
            batch.decision = dict(decision)
            batch.updated_at = now()
            return copy.deepcopy(batch)

    def finish_pending_input_batch(
        self,
        *,
        batch_id: str,
        expected_version: int,
        status: PendingInputBatchStatus,
        trace_id: str,
        decision: dict[str, Any] | None = None,
    ) -> tuple[PendingInputBatch | None, StateTransition | None]:
        with self._lock:
            batch = next((item for item in self.pending_input_batches.values() if item.batch_id == batch_id), None)
            if batch is None or batch.version != int(expected_version):
                return None, None
            old = batch.status.value
            batch.status = status
            batch.decision = dict(decision or {})
            batch.updated_at = now()
            transition = StateTransition(
                "pending_input_batch",
                batch.batch_id,
                old,
                status.value,
                "input_batch_finished",
                trace_id,
            )
            self.transitions.append(transition)
            return copy.deepcopy(batch), transition

    def register_message_reference(self, reference: MessageReference) -> None:
        if not reference.message_id:
            return
        with self._lock:
            self.message_references[message_reference_key(reference.conversation_id, reference.message_id)] = reference

    def link_message_reference(
        self,
        *,
        conversation_id: str,
        message_id: str,
        source_message_id: str | None = None,
        business_ref_type: str | None = None,
        business_ref_id: str | None = None,
        channel: str | None = None,
        text: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MessageReference:
        source = self._find_message_reference_source(
            conversation_id=conversation_id,
            source_message_id=source_message_id,
            business_ref_type=business_ref_type,
            business_ref_id=business_ref_id,
        )
        if source is None:
            raise ValueError("source message reference not found")
        linked = MessageReference(
            message_id=str(message_id or ""),
            conversation_id=str(conversation_id or source.conversation_id),
            business_ref_type=source.business_ref_type,
            business_ref_id=source.business_ref_id,
            text=str(text or source.text or ""),
            channel=str(channel or source.channel or ""),
            sender_id=source.sender_id,
            sender_name=source.sender_name,
            recipient_id=source.recipient_id,
            recipient_name=source.recipient_name,
            metadata={
                **dict(source.metadata),
                **dict(metadata or {}),
                "linked_from_message_id": source.message_id,
                "linked_from_conversation_id": source.conversation_id,
            },
        )
        self.register_message_reference(linked)
        return linked

    def _find_message_reference_source(
        self,
        *,
        conversation_id: str,
        source_message_id: str | None,
        business_ref_type: str | None,
        business_ref_id: str | None,
    ) -> MessageReference | None:
        with self._lock:
            if source_message_id:
                source = self.resolve_message_reference(conversation_id=conversation_id, message_id=source_message_id)
                if source is not None:
                    return source
            if business_ref_type and business_ref_id:
                same_conversation: MessageReference | None = None
                latest: MessageReference | None = None
                for reference in self.message_references.values():
                    if (
                        reference.business_ref_type != business_ref_type
                        or reference.business_ref_id != business_ref_id
                    ):
                        continue
                    latest = reference
                    if reference.conversation_id == conversation_id:
                        same_conversation = reference
                return same_conversation or latest
            return None

    def resolve_message_reference(
        self,
        *,
        conversation_id: str,
        message_id: str,
    ) -> MessageReference | None:
        if not message_id:
            return None
        with self._lock:
            direct = self.message_references.get(message_reference_key(conversation_id, message_id))
            if direct is not None:
                return direct
            for reference in self.message_references.values():
                if reference.message_id == message_id:
                    return reference
            return None

    def clear_runtime_state(
        self,
        *,
        include_customers: bool = False,
        include_badcases: bool = False,
    ) -> dict[str, int]:
        with self._lock:
            deleted = {
                "games": len(self.games),
                "invite_drafts": len(self.invite_drafts),
                "outbound_message_drafts": len(self.outbound_message_drafts),
                "room_reservations": len(self.room_reservations),
                "state_transitions": len(self.transitions),
                "conversation_turns": sum(len(items) for items in self.turns.values()),
                "conversation_checkpoints": len(self.conversation_checkpoints),
                "task_contexts": len(self.task_contexts),
                "conversation_versions": len(self.conversation_versions),
                "idempotency_ledger": len(self.idempotency_ledger),
                "message_results": len(self.message_results),
                "message_references": len(self.message_references),
                "task_memories": len(self.task_memories),
                "pending_memory_candidates": len(self.pending_memory_candidates),
                "pending_input_batches": len(self.pending_input_batches),
                "customers": len(self.customers) if include_customers else 0,
                "customer_relationships": len(self.customer_relationships) if include_customers else 0,
                "badcases": len(self.badcases) if include_badcases else 0,
            }
            self.games.clear()
            self.invite_drafts.clear()
            self.outbound_message_drafts.clear()
            self.room_reservations.clear()
            self.transitions.clear()
            self.turns.clear()
            self.conversation_checkpoints.clear()
            self.task_contexts.clear()
            self.conversation_versions.clear()
            self.idempotency_ledger.clear()
            self.idempotency_claimed_at.clear()
            self.message_results.clear()
            self.message_references.clear()
            self.task_memories.clear()
            self.pending_memory_candidates.clear()
            self.pending_input_batches.clear()
            if include_customers:
                self.customers.clear()
                self.customer_relationships.clear()
            if include_badcases:
                self.badcases.clear()
            return deleted

    def search_current_games(
        self,
        requirement: dict[str, Any],
        limit: int = 8,
        *,
        sender_id: str | None = None,
        conversation_id: str | None = None,
    ) -> list[dict[str, Any]]:
        with self._lock:
            requirement = normalize_requirement(requirement)
            scored: list[dict[str, Any]] = []
            requested_seats = requested_seat_count_from_search_requirement(requirement, default=1)
            anchor_ids = task_memory_anchor_ids(requirement, sender_id=sender_id)
            task_excluded = set(self.task_memory_excluded_customer_ids(conversation_id, anchor_ids))
            for game in self.active_games():
                if game.remaining_seats() <= 0:
                    continue
                if task_excluded and any(game_contains_customer(game, customer_id) for customer_id in task_excluded):
                    continue
                score, reasons = score_requirement(requirement, game.requirement)
                if requirement and score <= 0:
                    continue
                scored.append(
                    {
                        "game": game_for_model_context(game, self.customers),
                        "score": score,
                        "reasons": reasons or ["active_open_game"],
                        "join_projection": join_projection(game, sender_id=sender_id, requested_seats=requested_seats),
                    }
                )
            scored.sort(key=lambda item: item["score"], reverse=True)
            return scored[: int(limit)]

    def search_customers(
        self,
        requirement: dict[str, Any],
        *,
        exclude_customer_ids: list[str] | None = None,
        limit: int = 8,
        sender_id: str | None = None,
        conversation_id: str | None = None,
    ) -> list[dict[str, Any]]:
        requirement = normalize_requirement(requirement)
        excluded = set(exclude_customer_ids or [])
        anchor_ids = task_memory_anchor_ids(requirement, sender_id=sender_id, excluded_customer_ids=excluded)
        excluded.update(self.task_memory_excluded_customer_ids(conversation_id, anchor_ids))
        with self._lock:
            self._expire_stale_games_locked(trace_id="system_lifecycle")
            active_games = [
                game
                for game in self.games.values()
                if game.status in {GameStatus.FORMING, GameStatus.INVITING, GameStatus.READY}
            ]
            scored: list[dict[str, Any]] = []
            for customer in self.customers.values():
                if customer.no_contact or customer.customer_id in excluded:
                    continue
                committed, provisional_count = customer_option_load(
                    customer.customer_id,
                    requirement,
                    active_games,
                )
                if committed:
                    continue
                score, reasons = score_customer(requirement, customer)
                if provisional_count:
                    score -= min(15, provisional_count * 3)
                    reasons.append(f"provisional_in_{provisional_count}_overlapping_options")
                relationship_score, relationship_reasons, blocked = score_customer_relationships(
                    customer.customer_id,
                    anchor_ids,
                    self.relationship_between,
                )
                if blocked:
                    continue
                score += relationship_score
                reasons.extend(relationship_reasons)
                if score <= 0:
                    continue
                scored.append({"customer": customer.to_model_context(), "score": score, "reasons": reasons})
            scored.sort(key=lambda item: item["score"], reverse=True)
            return scored[: int(limit)]

    def active_game_for_customer(self, customer_id: str) -> Game | None:
        with self._lock:
            self._expire_stale_games_locked(trace_id="system_lifecycle")
            for game in self.games.values():
                if game.status.value not in {GameStatus.FORMING.value, GameStatus.INVITING.value, GameStatus.READY.value}:
                    continue
                if any(item.customer_id == customer_id and item.status in {"joined", "confirmed"} for item in game.participants):
                    return game
        return None

    def create_game(
        self,
        *,
        conversation_id: str,
        organizer_id: str,
        organizer_name: str,
        requirement: dict[str, Any],
        known_players: list[dict[str, Any]],
        trace_id: str,
    ) -> tuple[Game, StateTransition]:
        with self._lock:
            duplicate = next(
                (
                    item
                    for item in self.games.values()
                    if item.conversation_id == conversation_id
                    and item.organizer_id == organizer_id
                    and item.status in {GameStatus.FORMING, GameStatus.INVITING, GameStatus.READY}
                ),
                None,
            )
            if duplicate is not None:
                raise ValueError(f"active game already exists: {duplicate.game_id}")
            normalized_requirement = normalize_requirement(requirement)
            default_requester_seat_count = seat_count_from_payload(normalized_requirement, default=1)
            participants = normalize_game_participants(
                organizer_id=organizer_id,
                organizer_name=organizer_name,
                known_players=known_players,
                default_requester_seat_count=default_requester_seat_count,
            )
            parties = normalize_game_parties(participants)
            claimed_seats = sum(
                max(1, int(item.seat_count))
                for item in participants
                if item.status in {"joined", "confirmed"}
            )
            if claimed_seats > 4:
                raise ValueError(f"initial participants exceed table capacity: {claimed_seats}>4")
            game = Game(
                game_id=new_id("game"),
                conversation_id=conversation_id,
                organizer_id=organizer_id,
                organizer_name=organizer_name,
                requirement=normalize_requirement_with_party(normalized_requirement, parties),
                participants=participants,
                parties=parties,
            )
            apply_game_lifecycle(game)
            conflicts = ready_commitment_conflicts(
                game,
                active_game_participant_ids(game),
                list(self.games.values()),
            )
            if conflicts:
                raise ValueError(
                    "participants already committed to overlapping ready games: "
                    + ",".join(item.game_id for item in conflicts)
                )
            self.games[game.game_id] = game
            transition = StateTransition(
                entity_type="game",
                entity_id=game.game_id,
                from_status=None,
                to_status=game.status.value,
                reason="create_game",
                trace_id=trace_id,
            )
            self.transitions.append(transition)
            return game, transition

    def _expire_stale_games_locked(self, *, trace_id: str) -> list[StateTransition]:
        stamp = now()
        transitions: list[StateTransition] = []
        for game in self.games.values():
            transition = expire_game_if_stale(game, at=stamp, trace_id=trace_id)
            if transition is not None:
                transitions.append(transition)
                transitions.extend(
                    self._release_room_reservations_for_game_locked(
                        game.game_id,
                        trace_id=trace_id,
                        reason="game_lifecycle_closed",
                    )
                )
        if transitions:
            self.transitions.extend(transitions)
        return transitions

    def update_game_requirement(
        self,
        *,
        game_id: str,
        requirement_patch: dict[str, Any],
        reason: str,
        trace_id: str,
    ) -> tuple[Game, StateTransition]:
        """Apply a user-confirmed condition revision without changing seat ownership."""

        with self._lock:
            game = self.require_game(game_id)
            if game.status not in {GameStatus.FORMING, GameStatus.INVITING}:
                raise ValueError(f"game requirement is immutable in status={game.status.value}: {game_id}")
            protected = sorted(PROTECTED_REQUIREMENT_PATCH_FIELDS.intersection(requirement_patch))
            if protected:
                raise ValueError(f"requirement patch contains protected fields: {','.join(protected)}")
            lifecycle_fields = {
                "planned_start_at",
                "planned_end_at",
                "lifecycle_expires_at",
                "lifecycle_ttl_hours",
                "latest_start_at",
            }
            base_requirement = {
                key: value for key, value in game.requirement.items() if key not in lifecycle_fields
            }
            merged = normalize_requirement({**base_requirement, **dict(requirement_patch)})
            prospective = replace(
                game,
                requirement=refresh_requirement_seat_snapshot(merged, game.parties, game.remaining_seats()),
            )
            apply_game_lifecycle(prospective)
            conflicts = ready_commitment_conflicts(
                prospective,
                active_game_participant_ids(prospective),
                list(self.games.values()),
            )
            if conflicts:
                raise ValueError(
                    "updated requirement conflicts with committed ready games: "
                    + ",".join(item.game_id for item in conflicts)
                )
            game.requirement = prospective.requirement
            game.planned_start_at = prospective.planned_start_at
            game.planned_end_at = prospective.planned_end_at
            game.expires_at = prospective.expires_at
            game.updated_at = now()
            transition = StateTransition(
                "game_requirement",
                game.game_id,
                "configured",
                "configured",
                reason or "update_game_requirement",
                trace_id,
            )
            self.transitions.append(transition)
            return game, transition

    def _release_room_reservations_for_game_locked(
        self,
        game_id: str,
        *,
        trace_id: str,
        reason: str,
    ) -> list[StateTransition]:
        transitions: list[StateTransition] = []
        for reservation in self.room_reservations.values():
            if reservation.game_id != game_id or reservation.status not in {"held", "confirmed"}:
                continue
            old = reservation.status
            reservation.status = "released"
            reservation.updated_at = now()
            transitions.append(
                StateTransition(
                    "room_reservation",
                    reservation.reservation_id,
                    old,
                    reservation.status,
                    reason,
                    trace_id,
                )
            )
        return transitions

    def create_invite_drafts(
        self,
        *,
        game_id: str,
        invitations: list[dict[str, Any]],
        trace_id: str,
    ) -> tuple[list[InviteDraft], list[StateTransition]]:
        with self._lock:
            self._expire_stale_games_locked(trace_id=trace_id)
            game = self.require_game(game_id)
            if game.status not in {GameStatus.FORMING, GameStatus.INVITING}:
                raise ValueError(f"game does not accept invitations in status={game.status.value}: {game_id}")
            requested_customer_ids = [
                str(item.get("customer_id") or "").strip()
                for item in invitations
                if isinstance(item, dict)
            ]
            if any(not customer_id for customer_id in requested_customer_ids):
                raise ValueError("every invitation requires customer_id")
            if len(requested_customer_ids) != len(set(requested_customer_ids)):
                raise ValueError("duplicate customer_id in invitation request")
            open_customer_ids = {
                draft.customer_id
                for draft in self.invite_drafts.values()
                if draft.game_id == game_id and draft.status in OPEN_INVITE_STATUSES
            }
            duplicated = sorted(open_customer_ids.intersection(requested_customer_ids))
            if duplicated:
                raise ValueError(f"customer already has an open invitation for this game: {','.join(duplicated)}")
            transitions: list[StateTransition] = []
            if game.status == GameStatus.FORMING:
                old = game.status.value
                game.status = GameStatus.INVITING
                game.updated_at = now()
                transitions.append(
                    StateTransition("game", game.game_id, old, game.status.value, "create_invite_drafts", trace_id)
                )
            drafts: list[InviteDraft] = []
            for raw in invitations:
                if not isinstance(raw, dict):
                    continue
                draft = InviteDraft(
                    draft_id=new_id("draft"),
                    game_id=game_id,
                    customer_id=str(raw.get("customer_id") or ""),
                    display_name=str(raw.get("display_name") or raw.get("customer_id") or ""),
                    message_text=str(raw.get("message_text") or ""),
                    metadata=dict(raw.get("metadata") or {}) if isinstance(raw.get("metadata"), dict) else {},
                )
                self.invite_drafts[draft.draft_id] = draft
                self.register_message_reference(
                    MessageReference(
                        message_id=draft.draft_id,
                        conversation_id=game.conversation_id,
                        business_ref_type="invite_draft",
                        business_ref_id=draft.draft_id,
                        text=draft.message_text,
                        channel=str(draft.metadata.get("channel") or "internal"),
                        recipient_id=draft.customer_id,
                        recipient_name=draft.display_name,
                        metadata={"source": "create_invite_drafts", "game_id": game_id},
                    )
                )
                drafts.append(draft)
                transitions.append(
                    StateTransition("invite_draft", draft.draft_id, None, draft.status.value, "create_invite_drafts", trace_id)
                )
            self.transitions.extend(transitions)
            return drafts, transitions

    def create_outbound_message_drafts(
        self,
        *,
        conversation_id: str,
        drafts: list[dict[str, Any]],
        trace_id: str,
    ) -> tuple[list[OutboundMessageDraft], list[StateTransition]]:
        with self._lock:
            created: list[OutboundMessageDraft] = []
            transitions: list[StateTransition] = []
            for raw in drafts:
                if not isinstance(raw, dict):
                    continue
                draft = OutboundMessageDraft(
                    draft_id=new_id("outbound"),
                    conversation_id=conversation_id,
                    recipient_id=str(raw.get("recipient_id") or ""),
                    recipient_name=str(raw.get("recipient_name") or raw.get("recipient_id") or ""),
                    channel=str(raw.get("channel") or ""),
                    message_text=str(raw.get("message_text") or ""),
                    purpose=str(raw.get("purpose") or ""),
                    metadata=dict(raw.get("metadata") or {}) if isinstance(raw.get("metadata"), dict) else {},
                )
                self.outbound_message_drafts[draft.draft_id] = draft
                self.register_message_reference(
                    MessageReference(
                        message_id=draft.draft_id,
                        conversation_id=draft.conversation_id,
                        business_ref_type="outbound_message_draft",
                        business_ref_id=draft.draft_id,
                        text=draft.message_text,
                        channel=draft.channel,
                        recipient_id=draft.recipient_id,
                        recipient_name=draft.recipient_name,
                        metadata={"source": "create_outbound_message_drafts", "purpose": draft.purpose},
                    )
                )
                created.append(draft)
                transitions.append(
                    StateTransition(
                        "outbound_message_draft",
                        draft.draft_id,
                        None,
                        draft.status.value,
                        "create_outbound_message_drafts",
                        trace_id,
                    )
                )
            self.transitions.extend(transitions)
            return created, transitions

    def update_invite_delivery_status(
        self,
        *,
        draft_id: str,
        status: InviteStatus | str,
        trace_id: str,
        reason: str,
    ) -> tuple[InviteDraft, StateTransition]:
        """Persist the result of an externally approved invitation delivery."""

        with self._lock:
            draft = self.invite_drafts.get(draft_id)
            if draft is None:
                raise ValueError(f"invite draft not found: {draft_id}")
            target = status if isinstance(status, InviteStatus) else InviteStatus(str(status))
            allowed = {
                InviteStatus.PENDING_APPROVAL: {InviteStatus.SENT, InviteStatus.SUPERSEDED},
                InviteStatus.SENT: {InviteStatus.SENT},
            }
            if target not in allowed.get(draft.status, set()):
                raise ValueError(f"invalid invite delivery transition: {draft.status.value}->{target.value}")
            old = draft.status.value
            draft.status = target
            draft.updated_at = now()
            transition = StateTransition("invite_draft", draft_id, old, target.value, reason, trace_id)
            self.transitions.append(transition)
            return draft, transition

    def record_candidate_reply(
        self,
        *,
        game_id: str,
        customer_id: str,
        display_name: str,
        status: str,
        seat_count: int = 1,
        trace_id: str,
    ) -> tuple[Game, list[StateTransition]]:
        with self._lock:
            game = self.require_game(game_id)
            transitions: list[StateTransition] = []
            normalized_status = status.strip()
            normalized_seat_count = max(1, min(4, int(seat_count or 1)))
            existing_participant = next((item for item in game.participants if item.customer_id == customer_id), None)
            if normalized_status in CONFIRMED_CANDIDATE_STATUSES:
                conflicts = ready_commitment_conflicts(
                    game,
                    {customer_id},
                    list(self.games.values()),
                )
                if conflicts:
                    raise ValueError(
                        "customer already committed to overlapping ready game: "
                        + ",".join(item.game_id for item in conflicts)
                    )
                claimed_by_others = sum(
                    max(1, int(item.seat_count))
                    for item in game.participants
                    if item.customer_id != customer_id and item.status in {"joined", "confirmed"}
                )
                available_seats = max(0, game.seats_total - claimed_by_others)
                if normalized_seat_count > available_seats:
                    raise ValueError(
                        f"seat capacity exceeded for game {game_id}: requested={normalized_seat_count}, "
                        f"available={available_seats}"
                    )
            for draft in self.invite_drafts.values():
                if draft.game_id == game_id and draft.customer_id == customer_id:
                    old = draft.status.value
                    draft.status = invite_status_from_candidate_status(normalized_status)
                    draft.updated_at = now()
                    transitions.append(StateTransition("invite_draft", draft.draft_id, old, draft.status.value, "record_candidate_reply", trace_id))
            if normalized_status in CONFIRMED_CANDIDATE_STATUSES:
                if existing_participant is None:
                    game.participants.append(
                        GameParticipant(
                            customer_id=customer_id,
                            display_name=display_name or customer_id,
                            status="confirmed",
                            source="candidate_reply",
                            seat_count=normalized_seat_count,
                        )
                    )
                    transitions.append(
                        StateTransition(
                            "game_participant",
                            f"{game.game_id}:{customer_id}",
                            None,
                            "confirmed",
                            "record_candidate_reply",
                            trace_id,
                        )
                    )
                else:
                    old_status = existing_participant.status
                    old_seat_count = max(1, int(existing_participant.seat_count))
                    existing_participant.status = "confirmed"
                    existing_participant.seat_count = normalized_seat_count
                    if old_status != existing_participant.status or old_seat_count != normalized_seat_count:
                        transitions.append(
                            StateTransition(
                                "game_participant",
                                f"{game.game_id}:{customer_id}",
                                f"{old_status}:seats={old_seat_count}",
                                f"{existing_participant.status}:seats={normalized_seat_count}",
                                "record_candidate_reply",
                                trace_id,
                            )
                        )
            elif normalized_status in UNCONFIRMED_CANDIDATE_STATUSES and existing_participant is not None:
                old_status = existing_participant.status
                old_seat_count = max(1, int(existing_participant.seat_count))
                existing_participant.status = normalized_status
                existing_participant.seat_count = normalized_seat_count
                if old_status != existing_participant.status or old_seat_count != normalized_seat_count:
                    transitions.append(
                        StateTransition(
                            "game_participant",
                            f"{game.game_id}:{customer_id}",
                            f"{old_status}:seats={old_seat_count}",
                            f"{existing_participant.status}:seats={normalized_seat_count}",
                            "record_candidate_reply",
                            trace_id,
                        )
                    )
            game.parties = normalize_game_parties(game.participants)
            game.requirement = refresh_requirement_seat_snapshot(game.requirement, game.parties, game.remaining_seats())
            if game.remaining_seats() == 0 and game.status != GameStatus.READY:
                resolution = resolve_full_game_commitments(
                    game,
                    games=list(self.games.values()),
                    invite_drafts=list(self.invite_drafts.values()),
                    trace_id=trace_id,
                )
                transitions.extend(resolution.transitions)
            elif game.remaining_seats() > 0 and game.status == GameStatus.READY:
                old = game.status.value
                game.status = GameStatus.INVITING if any(draft.game_id == game.game_id for draft in self.invite_drafts.values()) else GameStatus.FORMING
                transitions.append(StateTransition("game", game.game_id, old, game.status.value, "seats_reopened", trace_id))
            game.updated_at = now()
            self.transitions.extend(transitions)
            return game, transitions

    def update_game_status(self, *, game_id: str, status: str, reason: str, trace_id: str) -> tuple[Game, StateTransition]:
        with self._lock:
            game = self.require_game(game_id)
            target = GameStatus(status)
            old = game.status.value
            allowed = ALLOWED_GAME_TRANSITIONS.get(old, set())
            if target.value != old and target.value not in allowed:
                raise ValueError(f"illegal game status transition: {old}->{target.value}")
            game.status = target
            if target in {GameStatus.CANCELLED, GameStatus.FINISHED}:
                game.closed_reason = reason or target.value
            game.updated_at = now()
            transition = StateTransition("game", game.game_id, old, target.value, reason or "update_game_status", trace_id)
            self.transitions.append(transition)
            if target in {GameStatus.CANCELLED, GameStatus.FINISHED}:
                self.transitions.extend(
                    self._release_room_reservations_for_game_locked(
                        game.game_id,
                        trace_id=trace_id,
                        reason="game_status_closed",
                    )
                )
            return game, transition

    def record_badcase(self, payload: dict[str, Any], *, trace_id: str, conversation_id: str) -> dict[str, Any]:
        with self._lock:
            record = {"badcase_id": new_id("badcase"), "trace_id": trace_id, "conversation_id": conversation_id, **dict(payload)}
            self.badcases.append(record)
            return record

    def require_game(self, game_id: str) -> Game:
        game = self.games.get(game_id)
        if game is None:
            raise ValueError(f"game not found: {game_id}")
        return game


def score_requirement(query: dict[str, Any], target: dict[str, Any]) -> tuple[int, list[str]]:
    query = normalize_requirement(query)
    target = normalize_requirement(target)
    score = 0
    reasons: list[str] = []
    for key, weight, aliases in [
        ("game_type", 30, ("game_type", "preferred_game", "preferred_games", "game_types")),
        ("stake", 25, ("stake", "preferred_stake", "preferred_stakes", "stakes")),
        ("smoke_preference", 15, ("smoke_preference", "smoke")),
        ("start_time_kind", 10, ("start_time_kind", "start_time")),
        ("duration_kind", 10, ("duration_kind", "duration")),
    ]:
        query_value = first_present_value(query, *aliases)
        if is_blank_value(query_value):
            continue
        target_value = first_present_value(target, *aliases)
        if value_matches(query_value, target_value):
            score += weight
            reasons.append(f"{key}_matched")
        elif key in {"game_type", "stake", "smoke_preference"}:
            score -= weight
            reasons.append(f"{key}_mismatched")
    cap_query = first_present_value(query, "cap_score", "cap_stake", "cap", "cap_limit")
    if not is_blank_value(cap_query):
        cap_target = first_present_value(target, "cap_score", "cap_stake", "cap", "cap_limit")
        if value_matches(cap_query, cap_target):
            score += 8
            reasons.append("cap_score_matched")
        elif not is_blank_value(cap_target):
            return -999, [*reasons, "cap_score_mismatched"]
        else:
            score -= 8
            reasons.append("cap_score_unknown")
    return score, reasons


def score_customer(requirement: dict[str, Any], customer: CustomerProfile) -> tuple[int, list[str]]:
    requirement = normalize_requirement(requirement)
    score = 0
    reasons: list[str] = []
    game_query = first_present_value(requirement, "game_type", "preferred_game", "preferred_games", "game_types")
    smoke_query = first_present_value(requirement, "smoke_preference", "smoke")
    if value_matches(game_query, customer.preferred_games):
        score += 30
        reasons.append("game_type_matched")
    stake_score, stake_reasons = score_stake_preference(requirement, customer.preferred_stakes)
    score += stake_score
    reasons.extend(stake_reasons)
    if smoke_matches(smoke_query, customer.smoke_preference):
        score += 10
        reasons.append("smoke_matched")
    gender = requirement.get("preferred_gender") or requirement.get("gender")
    if value_matches(gender, customer.gender):
        score += 10
        reasons.append("gender_matched")
    score += int(max(0.0, min(1.0, customer.response_score)) * 10)
    score -= int(max(0.0, customer.fatigue_score) * 10)
    return score, reasons


def normalize_requirement(requirement: dict[str, Any] | None) -> dict[str, Any]:
    normalized = dict(requirement or {})
    stake_value = first_present_value(
        normalized,
        "stake",
        "base_stake",
        "base_score",
        "preferred_stake",
        "level",
    )
    cap_value = first_present_value(normalized, "cap_score", "cap_stake", "cap", "cap_limit")
    parsed = parse_stake_value(stake_value)
    cap_number = parse_number(cap_value)
    if parsed is None and cap_number is None:
        return normalized
    base_number = parsed[0] if parsed else parse_number(stake_value)
    parsed_cap = parsed[1] if parsed else None
    final_cap = parsed_cap if parsed_cap is not None else cap_number
    if base_number is not None:
        normalized["base_stake"] = base_number
        normalized["stake"] = format_number(base_number)
    if final_cap is not None:
        normalized["cap_score"] = final_cap
    if base_number is not None and final_cap is not None:
        normalized.setdefault("stake_label", f"{format_number(base_number)}-{format_number(final_cap)}")
        normalized.setdefault("level", f"{format_number(base_number)}-{format_number(final_cap)}")
    elif base_number is not None:
        normalized.setdefault("stake_label", format_number(base_number))
    return normalized


def apply_game_lifecycle(game: Game) -> None:
    lifecycle = derive_game_lifecycle(game.requirement, created_at=game.created_at)
    game.planned_start_at = lifecycle["planned_start_at"]
    game.planned_end_at = lifecycle["planned_end_at"]
    game.expires_at = lifecycle["expires_at"]
    game.requirement = {
        **dict(game.requirement),
        **lifecycle["requirement_patch"],
    }


def expire_game_if_stale(game: Game, *, at: datetime, trace_id: str) -> StateTransition | None:
    if game.status.value not in {GameStatus.FORMING.value, GameStatus.INVITING.value, GameStatus.READY.value}:
        return None
    if game.expires_at is None:
        apply_game_lifecycle(game)
    if game.expires_at is None or at < game.expires_at:
        return None
    old = game.status.value
    if game.status == GameStatus.READY or game.remaining_seats() <= 0:
        game.status = GameStatus.FINISHED
        game.closed_reason = "game_time_elapsed"
    else:
        game.status = GameStatus.CANCELLED
        game.closed_reason = "expired_without_full_table"
    game.updated_at = at
    return StateTransition("game", game.game_id, old, game.status.value, game.closed_reason, trace_id)


def derive_game_lifecycle(requirement: dict[str, Any], *, created_at: datetime) -> dict[str, Any]:
    requirement = dict(requirement or {})
    created_at = normalize_datetime(created_at)
    start_kind = str(requirement.get("start_time_kind") or "").strip()
    planned_start_at = first_datetime_value(
        requirement,
        "planned_start_at",
        "start_at",
        "start_datetime",
        "start_time_at",
    )
    if planned_start_at is None and start_kind == START_KIND_SCHEDULED:
        planned_start_at = parse_start_time_on_created_date(requirement.get("start_time"), created_at=created_at)
    duration_hours = duration_hours_from_requirement(requirement)
    planned_end_at: datetime | None = None
    if planned_start_at is not None:
        planned_end_at = planned_start_at + timedelta(hours=duration_hours)

    if start_kind == START_KIND_ASAP_WHEN_FULL:
        expires_at = created_at + timedelta(hours=DEFAULT_ASAP_GAME_TTL_HOURS)
    elif planned_end_at is not None:
        expires_at = planned_end_at
    elif planned_start_at is not None:
        expires_at = planned_start_at + timedelta(hours=DEFAULT_UNKNOWN_DURATION_HOURS)
    else:
        expires_at = created_at + timedelta(hours=DEFAULT_ASAP_GAME_TTL_HOURS)

    patch: dict[str, Any] = {
        "lifecycle_expires_at": expires_at.isoformat(),
        "lifecycle_ttl_hours": round((expires_at - created_at).total_seconds() / 3600, 3),
    }
    if planned_start_at is not None:
        patch["planned_start_at"] = planned_start_at.isoformat()
    if planned_end_at is not None:
        patch["planned_end_at"] = planned_end_at.isoformat()
    if start_kind == START_KIND_ASAP_WHEN_FULL:
        patch["latest_start_at"] = expires_at.isoformat()
    return {
        "planned_start_at": planned_start_at,
        "planned_end_at": planned_end_at,
        "expires_at": expires_at,
        "requirement_patch": patch,
    }


def duration_hours_from_requirement(requirement: dict[str, Any]) -> float:
    raw = first_present_value(requirement, "duration_hours", "duration")
    parsed = parse_number(raw)
    if parsed is not None and parsed > 0:
        return min(24.0, parsed)
    duration_kind = str(requirement.get("duration_kind") or "").strip()
    if duration_kind == DURATION_KIND_OVERNIGHT:
        return DEFAULT_OVERNIGHT_DURATION_HOURS
    return DEFAULT_UNKNOWN_DURATION_HOURS


def first_datetime_value(payload: dict[str, Any], *keys: str) -> datetime | None:
    for key in keys:
        value = payload.get(key)
        if is_blank_value(value):
            continue
        parsed = parse_datetime_value(value)
        if parsed is not None:
            return parsed
    return None


def parse_datetime_value(value: Any) -> datetime | None:
    if is_blank_value(value):
        return None
    if isinstance(value, datetime):
        return normalize_datetime(value)
    try:
        return normalize_datetime(datetime.fromisoformat(str(value)))
    except ValueError:
        return None


def normalize_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=DEFAULT_TZ)
    return value


def parse_start_time_on_created_date(value: Any, *, created_at: datetime) -> datetime | None:
    if is_blank_value(value):
        return None
    text = str(value).strip()
    match = re.fullmatch(r"(?P<hour>\d{1,2})(?::|：)?(?P<minute>\d{2})?", text)
    if match is None:
        match = re.fullmatch(r"(?P<hour>\d{1,2})\s*(?:点|时)(?P<minute>\d{1,2})?", text)
    if match is None:
        return None
    hour = int(match.group("hour"))
    minute = int(match.group("minute") or 0)
    if hour > 23 or minute > 59:
        return None
    return created_at.replace(hour=hour, minute=minute, second=0, microsecond=0)


def parse_stake_value(value: Any) -> tuple[float, float | None] | None:
    if is_blank_value(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    compact = (
        text.replace("，", "")
        .replace(",", "")
        .replace(" ", "")
        .replace("元", "")
        .replace("块", "")
        .replace("档", "")
    )
    compact_match = re.fullmatch(r"(?P<base>[1-9]\d?)(?P<cap>16|32|64|128)", compact)
    if compact_match and not re.fullmatch(r"\d7\d", compact):
        base = parse_number(compact_match.group("base"))
        cap = parse_number(compact_match.group("cap"))
        if base is not None:
            return base, cap
    explicit = re.search(
        r"(?P<base>\d+(?:\.\d+)?)\s*(?:元|块)?\s*(?:底|底注|底分).{0,8}?(?:封顶|封|顶|上限)\s*(?P<cap>\d+(?:\.\d+)?)",
        text,
    )
    if explicit:
        base = parse_number(explicit.group("base"))
        cap = parse_number(explicit.group("cap"))
        if base is not None:
            return base, cap
    range_match = re.fullmatch(
        r"(?P<base>\d+(?:\.\d+)?)\s*(?:-|/|－|—|到|至)\s*(?P<cap>\d+(?:\.\d+)?)",
        text,
    )
    if range_match:
        base = parse_number(range_match.group("base"))
        cap = parse_number(range_match.group("cap"))
        if base is not None:
            return base, cap
    base = parse_number(text)
    if base is not None:
        return base, None
    return None


def parse_number(value: Any) -> float | None:
    if is_blank_value(value):
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def format_number(value: float | int) -> str:
    number = float(value)
    return str(int(number)) if number.is_integer() else f"{number:g}"


def score_stake_preference(requirement: dict[str, Any], preferred_stakes: list[str]) -> tuple[int, list[str]]:
    query_values = list_values_for_keys(requirement, "stake", "base_stake", "preferred_stake", "preferred_stakes", "stakes")
    if not query_values:
        return 0, []
    cap_query = first_present_value(requirement, "cap_score", "cap_stake", "cap", "cap_limit")
    exact_base_match = False
    base_only_match = False
    for query_value in query_values:
        normalized_query = normalize_requirement({"stake": query_value, "cap_score": cap_query})
        query_base = parse_number(first_present_value(normalized_query, "base_stake", "stake"))
        query_cap = parse_number(first_present_value(normalized_query, "cap_score", "cap_stake", "cap", "cap_limit"))
        for preferred in preferred_stakes:
            normalized_preference = normalize_requirement({"stake": preferred})
            preferred_base = parse_number(first_present_value(normalized_preference, "base_stake", "stake"))
            preferred_cap = parse_number(first_present_value(normalized_preference, "cap_score", "cap_stake", "cap", "cap_limit"))
            if query_base is None or preferred_base is None or query_base != preferred_base:
                continue
            if query_cap is None or preferred_cap is None or query_cap == preferred_cap:
                exact_base_match = True
                break
            base_only_match = True
        if exact_base_match:
            break
    if exact_base_match:
        return 25, ["stake_matched"]
    if base_only_match:
        return 12, ["stake_base_matched", "cap_score_mismatched"]
    return 0, []


def list_values_for_keys(payload: dict[str, Any], *keys: str) -> list[Any]:
    values: list[Any] = []
    for key in keys:
        value = payload.get(key)
        if is_blank_value(value):
            continue
        if isinstance(value, (list, tuple, set)):
            values.extend(item for item in value if not is_blank_value(item))
        else:
            values.append(value)
    return values


def relationship_pair_key(customer_id: str, other_customer_id: str) -> str:
    left = str(customer_id or "").strip()
    right = str(other_customer_id or "").strip()
    return "::".join(sorted([left, right]))


def relationship_anchor_ids(requirement: dict[str, Any], excluded_customer_ids: set[str] | list[str] | None = None) -> list[str]:
    anchors: list[str] = []
    for key in (
        "existing_player_ids",
        "known_player_ids",
        "participant_ids",
        "current_player_ids",
        "avoid_conflict_with_customer_ids",
    ):
        value = requirement.get(key)
        if isinstance(value, (list, tuple, set)):
            anchors.extend(str(item) for item in value if not is_blank_value(item))
        elif not is_blank_value(value):
            anchors.append(str(value))
    for key in ("organizer_id", "requester_id", "sender_id"):
        value = requirement.get(key)
        if not is_blank_value(value):
            anchors.append(str(value))
    anchors.extend(str(item) for item in excluded_customer_ids or [] if not is_blank_value(item))
    return list(dict.fromkeys(item for item in anchors if item))


def task_memory_anchor_ids(
    requirement: dict[str, Any],
    *,
    sender_id: str | None = None,
    excluded_customer_ids: set[str] | list[str] | None = None,
) -> list[str]:
    anchors = relationship_anchor_ids(requirement, excluded_customer_ids)
    if sender_id:
        anchors.append(str(sender_id))
    requesting_party = requirement.get("requesting_party")
    if isinstance(requesting_party, dict):
        for key in ("contact_id", "customer_id"):
            value = requesting_party.get(key)
            if not is_blank_value(value):
                anchors.append(str(value))
        for member_id in requesting_party.get("known_member_ids") or []:
            if not is_blank_value(member_id):
                anchors.append(str(member_id))
    for item in requirement.get("seat_claims") or []:
        if not isinstance(item, dict):
            continue
        for key in ("contact_id", "customer_id"):
            value = item.get(key)
            if not is_blank_value(value):
                anchors.append(str(value))
        for member_id in item.get("known_member_ids") or []:
            if not is_blank_value(member_id):
                anchors.append(str(member_id))
    return list(dict.fromkeys(item for item in anchors if str(item or "").strip()))


def is_avoid_playing_memory(memory: TaskMemory) -> bool:
    memory_type = str(memory.memory_type or "").strip()
    field_name = str(memory.field or "").strip()
    if memory_type not in {"relationship", "avoid_playing", "avoid_playing_with", "preference"}:
        return False
    if field_name not in {"avoid_playing", "avoid_customer", "avoid_playing_with", "not_play_with"}:
        return False
    value = memory.value
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "avoid", "不打", "不和他打", "不和她打"}


def game_contains_customer(game: Game, customer_id: str) -> bool:
    target = str(customer_id or "").strip()
    if not target:
        return False
    if str(game.organizer_id or "").strip() == target:
        return True
    for participant in game.participants:
        if str(participant.customer_id or "").strip() == target:
            return True
        if target in {str(item) for item in participant.known_member_ids or []}:
            return True
    for party in game.parties:
        if str(party.contact_id or "").strip() == target:
            return True
        if target in {str(item) for item in party.known_member_ids or []}:
            return True
    return False


def score_customer_relationships(
    customer_id: str,
    anchor_ids: list[str],
    relationship_lookup,
) -> tuple[int, list[str], bool]:
    score = 0
    reasons: list[str] = []
    for anchor_id in anchor_ids:
        if not anchor_id or anchor_id == customer_id:
            continue
        relationship = relationship_lookup(customer_id, anchor_id)
        if relationship is None:
            continue
        if relationship.avoid_playing:
            return 0, [f"avoid_playing_with:{anchor_id}"], True
        played_count = max(0, int(relationship.played_together_count))
        if played_count > 0:
            score += min(15, 5 + played_count)
            reasons.append(f"played_together_with:{anchor_id}")
    return score, reasons, False


def relationship_context_for_sender(
    *,
    sender_id: str,
    games: list[Game],
    customers: dict[str, CustomerProfile],
    relationship_lookup,
) -> list[dict[str, Any]]:
    context: list[dict[str, Any]] = []
    seen: set[str] = set()
    for game in games:
        participants = [
            GameParticipant(customer_id=game.organizer_id, display_name=game.organizer_name, status="organizer", source="organizer"),
            *game.participants,
        ]
        for participant in participants:
            target_id = str(participant.customer_id or "")
            if not target_id or target_id == sender_id or target_id in seen:
                continue
            seen.add(target_id)
            relationship = relationship_lookup(sender_id, target_id)
            played_count = int(relationship.played_together_count) if relationship else 0
            avoid_playing = bool(relationship.avoid_playing) if relationship else False
            if avoid_playing:
                label = "avoid_playing"
            elif played_count > 0:
                label = "played_before"
            else:
                label = "no_prior_play_record"
            context.append(
                {
                    "customer_id": target_id,
                    "display_name": customer_visible_name(customers, target_id, participant.display_name),
                    "played_together_count": played_count,
                    "avoid_playing": avoid_playing,
                    "relationship_label": label,
                    # Relationship facts may affect matching, but they are not
                    # customer-visible facts. The main/review models receive an
                    # explicit policy marker instead of inferring visibility from
                    # the presence of the field.
                    "visibility": "internal_matching_only",
                    "customer_visible": False,
                    "private_relationship_notes_omitted": bool(relationship and relationship.notes),
                }
            )
    return context


def customer_visible_name(customers: dict[str, CustomerProfile], customer_id: str, fallback: str | None = None) -> str:
    profile = customers.get(str(customer_id or ""))
    if profile:
        return profile.visible_name()
    return str(fallback or customer_id or "")


def game_for_model_context(game: Game, customers: dict[str, CustomerProfile]) -> dict[str, Any]:
    payload = game.to_dict()
    payload["organizer_name"] = customer_visible_name(customers, game.organizer_id, game.organizer_name)
    _rewrite_contact_names(payload, customers)
    return payload


def invite_draft_for_model_context(draft: InviteDraft, customers: dict[str, CustomerProfile]) -> dict[str, Any]:
    payload = draft.to_dict()
    payload["display_name"] = customer_visible_name(customers, draft.customer_id, draft.display_name)
    payload["metadata"] = visible_draft_metadata(payload.get("metadata"))
    return payload


def outbound_message_draft_for_model_context(
    draft: OutboundMessageDraft,
    customers: dict[str, CustomerProfile],
) -> dict[str, Any]:
    payload = draft.to_dict()
    payload["recipient_name"] = customer_visible_name(customers, draft.recipient_id, draft.recipient_name)
    payload["metadata"] = visible_draft_metadata(payload.get("metadata"))
    return payload


def visible_draft_metadata(metadata: Any) -> dict[str, Any]:
    metadata = dict(metadata or {}) if isinstance(metadata, dict) else {}
    return {
        key: value
        for key, value in metadata.items()
        if key in {"source", "game_id", "purpose", "channel"}
    }


def _rewrite_contact_names(payload: dict[str, Any], customers: dict[str, CustomerProfile]) -> None:
    for item in payload.get("participants") or []:
        if isinstance(item, dict):
            item["display_name"] = customer_visible_name(customers, str(item.get("customer_id") or ""), item.get("display_name"))
    for item in payload.get("parties") or []:
        if isinstance(item, dict):
            item["contact_name"] = customer_visible_name(customers, str(item.get("contact_id") or ""), item.get("contact_name"))
    for item in payload.get("seat_claims") or []:
        if isinstance(item, dict):
            item["contact_name"] = customer_visible_name(customers, str(item.get("contact_id") or ""), item.get("contact_name"))
    requirement = payload.get("requirement")
    if not isinstance(requirement, dict):
        return
    requester = requirement.get("requesting_party")
    if isinstance(requester, dict):
        requester["contact_name"] = customer_visible_name(
            customers,
            str(requester.get("contact_id") or requester.get("customer_id") or ""),
            requester.get("contact_name") or requester.get("display_name"),
        )
    for item in requirement.get("seat_claims") or []:
        if isinstance(item, dict):
            item["contact_name"] = customer_visible_name(
                customers,
                str(item.get("contact_id") or item.get("customer_id") or ""),
                item.get("contact_name") or item.get("display_name"),
            )


def normalize_game_participants(
    *,
    organizer_id: str,
    organizer_name: str,
    known_players: list[dict[str, Any]],
    default_requester_seat_count: int = 1,
) -> list[GameParticipant]:
    """Compatibility: organizer_id is the requesting contact, not the store operator."""
    participants: list[GameParticipant] = []
    seen: set[str] = set()

    requester_id = str(organizer_id or "").strip()
    if requester_id:
        requester_payload = next(
            (
                item
                for item in known_players
                if isinstance(item, dict) and str(item.get("customer_id") or "").strip() == requester_id
            ),
            {},
        )
        other_known_seats = 0
        for item in known_players:
            if not isinstance(item, dict):
                continue
            customer_id = str(item.get("customer_id") or "").strip()
            if not customer_id or customer_id == requester_id:
                continue
            other_known_seats += seat_count_from_payload(item, default=1)
        requester_default_seat_count = default_requester_seat_count
        if not payload_has_explicit_seat_count(requester_payload):
            requester_default_seat_count = max(1, int(default_requester_seat_count or 1) - other_known_seats)
        requester_seat_count = seat_count_from_payload(requester_payload, default=requester_default_seat_count)
        requester_known_member_ids = known_member_ids_from_payload(requester_payload, default_id=requester_id)
        participants.append(
            GameParticipant(
                customer_id=requester_id,
                display_name=str(organizer_name or requester_id),
                status=str(requester_payload.get("status") or "joined"),
                source="requester",
                seat_count=requester_seat_count,
                party_id=party_id_for_contact(requester_id),
                known_member_ids=requester_known_member_ids,
                anonymous_seat_count=anonymous_seat_count_from_payload(
                    requester_payload,
                    seat_count=requester_seat_count,
                    known_member_ids=requester_known_member_ids,
                ),
            )
        )
        seen.add(requester_id)

    for item in known_players:
        if not isinstance(item, dict):
            continue
        customer_id = str(item.get("customer_id") or "").strip()
        if not customer_id or customer_id in seen:
            continue
        seat_count = seat_count_from_payload(item, default=1)
        known_member_ids = known_member_ids_from_payload(item, default_id=customer_id)
        participants.append(
            GameParticipant(
                customer_id=customer_id,
                display_name=str(item.get("display_name") or customer_id),
                status=str(item.get("status") or "joined"),
                source=str(item.get("source") or "participant"),
                seat_count=seat_count,
                party_id=party_id_for_contact(customer_id),
                known_member_ids=known_member_ids,
                anonymous_seat_count=anonymous_seat_count_from_payload(
                    item,
                    seat_count=seat_count,
                    known_member_ids=known_member_ids,
                ),
            )
        )
        seen.add(customer_id)
    return participants


def normalize_game_parties(participants: list[GameParticipant]) -> list[Party]:
    parties: list[Party] = []
    seen: set[str] = set()
    for participant in participants:
        party_id = participant.party_id or party_id_for_contact(participant.customer_id)
        if party_id in seen:
            continue
        seen.add(party_id)
        seat_count = max(1, min(4, int(participant.seat_count or 1)))
        known_member_ids = list(participant.known_member_ids or [participant.customer_id])
        parties.append(
            Party(
                party_id=party_id,
                contact_id=participant.customer_id,
                contact_name=participant.display_name,
                seat_count=seat_count,
                known_member_ids=known_member_ids,
                anonymous_seat_count=max(
                    0,
                    int(participant.anonymous_seat_count or max(0, seat_count - len(known_member_ids))),
                ),
                status=participant.status,
                source=participant.source,
            )
        )
    return parties


def normalize_requirement_with_party(requirement: dict[str, Any], parties: list[Party]) -> dict[str, Any]:
    normalized = dict(requirement)
    claimed_seats = sum(max(1, int(item.seat_count)) for item in parties if item.status in {"joined", "confirmed"})
    if claimed_seats and is_blank_value(normalized.get("known_player_count")):
        normalized["known_player_count"] = claimed_seats
    if parties:
        normalized["requesting_party"] = parties[0].to_dict()
        normalized["seat_claims"] = [item.to_dict() for item in parties]
    return normalize_requirement(normalized)


def refresh_requirement_seat_snapshot(requirement: dict[str, Any], parties: list[Party], remaining_seats: int) -> dict[str, Any]:
    normalized = dict(requirement)
    claimed_seats = sum(max(1, int(item.seat_count)) for item in parties if item.status in {"joined", "confirmed"})
    normalized["known_player_count"] = claimed_seats
    normalized["needed_seats"] = max(0, int(remaining_seats))
    normalized.pop("requesting_party", None)
    normalized.pop("seat_claims", None)
    return normalize_requirement_with_party(normalized, parties)


def seat_count_from_payload(payload: dict[str, Any], *, default: int = 1) -> int:
    payload = payload if isinstance(payload, dict) else {}
    for key in ("seat_count", "seats", "party_size", "player_count", "current_player_count", "known_player_count"):
        value = payload.get(key)
        if is_blank_value(value):
            continue
        try:
            return max(1, min(4, int(value)))
        except (TypeError, ValueError):
            continue
    return max(1, min(4, int(default or 1)))


def payload_has_explicit_seat_count(payload: dict[str, Any]) -> bool:
    payload = payload if isinstance(payload, dict) else {}
    for key in ("seat_count", "seats", "party_size", "player_count", "current_player_count", "known_player_count"):
        if not is_blank_value(payload.get(key)):
            return True
    return False


def requested_seat_count_from_search_requirement(requirement: dict[str, Any], *, default: int = 1) -> int:
    requirement = requirement if isinstance(requirement, dict) else {}
    requesting_party = requirement.get("requesting_party")
    if isinstance(requesting_party, dict):
        party_payload = {
            key: value
            for key, value in requesting_party.items()
            if key not in {"known_player_count", "current_player_count", "needed_seats"}
        }
        if payload_has_explicit_seat_count(party_payload):
            return seat_count_from_payload(party_payload, default=default)
    sender_party_payload = {
        key: value
        for key, value in requirement.items()
        if key not in {"known_player_count", "current_player_count", "needed_seats"}
    }
    return seat_count_from_payload(sender_party_payload, default=default)


def known_member_ids_from_payload(payload: dict[str, Any], *, default_id: str) -> list[str]:
    payload = payload if isinstance(payload, dict) else {}
    for key in ("known_member_ids", "member_customer_ids", "members", "customer_ids"):
        value = payload.get(key)
        if not isinstance(value, list):
            continue
        ids = [str(item).strip() for item in value if str(item or "").strip()]
        if ids:
            return ids
    return [default_id] if default_id else []


def anonymous_seat_count_from_payload(
    payload: dict[str, Any],
    *,
    seat_count: int,
    known_member_ids: list[str],
) -> int:
    payload = payload if isinstance(payload, dict) else {}
    for key in ("anonymous_seat_count", "unknown_member_count", "unknown_seat_count"):
        value = payload.get(key)
        if is_blank_value(value):
            continue
        try:
            return max(0, min(4, int(value)))
        except (TypeError, ValueError):
            continue
    return max(0, int(seat_count or 1) - len(known_member_ids))


def party_id_for_contact(contact_id: str) -> str:
    return f"party_{str(contact_id or '').strip()}"


def message_reference_key(conversation_id: str, message_id: str) -> str:
    return f"{str(conversation_id or '')}:{str(message_id or '')}"


def join_projection(game: Game, *, sender_id: str | None, requested_seats: int = 1) -> dict[str, Any]:
    already_joined = bool(
        sender_id
        and any(item.customer_id == sender_id and item.status in {"joined", "confirmed"} for item in game.participants)
    )
    remaining_before = game.remaining_seats()
    seats_to_add = 0 if already_joined else max(1, min(4, int(requested_seats or 1)))
    remaining_after = max(0, remaining_before - seats_to_add)
    return {
        "sender_id": sender_id,
        "sender_already_joined": already_joined,
        "requested_seats": seats_to_add,
        "remaining_seats_before_join": remaining_before,
        "remaining_seats_after_join": remaining_after,
        "would_fill_game": remaining_before > 0 and remaining_after == 0,
        "would_overfill_game": seats_to_add > remaining_before,
    }


def first_present_value(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = payload.get(key)
        if not is_blank_value(value):
            return value
    return None


def is_blank_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, tuple, set)):
        return len(value) == 0
    return False


def value_set(value: Any) -> set[str]:
    if is_blank_value(value):
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(item) for item in value if not is_blank_value(item)}
    return {str(value)}


def value_matches(query_value: Any, target_value: Any) -> bool:
    if is_blank_value(query_value):
        return False
    return bool(value_set(query_value) & value_set(target_value))


def smoke_matches(query_value: Any, target_value: Any) -> bool:
    query_values = value_set(query_value)
    target_values = value_set(target_value)
    if not query_values or "any" in query_values:
        return True
    if not target_values or "any" in target_values:
        return True
    return value_matches(query_value, target_value)


def invite_status_from_candidate_status(status: str) -> InviteStatus:
    mapping = {
        "accepted": InviteStatus.CONFIRMED,
        "confirmed": InviteStatus.CONFIRMED,
        "arrived": InviteStatus.CONFIRMED,
        "declined": InviteStatus.DECLINED,
        "negotiating": InviteStatus.NEGOTIATING,
        "no_reply": InviteStatus.NO_REPLY,
    }
    return mapping.get(status, InviteStatus.NEGOTIATING)
