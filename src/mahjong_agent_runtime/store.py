from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from typing import Any

from .models import (
    AgentRuntimeResult,
    ConversationCheckpoint,
    ConversationRole,
    ConversationTurn,
    CustomerProfile,
    CustomerRelationship,
    GameParticipant,
    GameStatus,
    Game,
    InviteDraft,
    InviteStatus,
    MessageReference,
    OutboundDraftStatus,
    OutboundMessageDraft,
    Party,
    StateTransition,
    ToolResult,
    new_id,
    now,
)


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


@dataclass(slots=True)
class InMemoryAgentStore:
    customers: dict[str, CustomerProfile] = field(default_factory=dict)
    customer_relationships: dict[str, CustomerRelationship] = field(default_factory=dict)
    games: dict[str, Game] = field(default_factory=dict)
    invite_drafts: dict[str, InviteDraft] = field(default_factory=dict)
    outbound_message_drafts: dict[str, OutboundMessageDraft] = field(default_factory=dict)
    transitions: list[StateTransition] = field(default_factory=list)
    turns: dict[str, list[ConversationTurn]] = field(default_factory=dict)
    conversation_checkpoints: dict[str, ConversationCheckpoint] = field(default_factory=dict)
    conversation_versions: dict[str, int] = field(default_factory=dict)
    idempotency_ledger: dict[str, ToolResult] = field(default_factory=dict)
    message_results: dict[str, AgentRuntimeResult] = field(default_factory=dict)
    message_references: dict[str, MessageReference] = field(default_factory=dict)
    badcases: list[dict[str, Any]] = field(default_factory=list)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    def upsert_customer(self, profile: CustomerProfile) -> None:
        with self._lock:
            self.customers[profile.customer_id] = profile

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
        self.append_turn(
            message.conversation_id,
            ConversationTurn(
                role=ConversationRole.USER,
                content=message.text,
                trace_id=trace_id,
                sender_id=message.sender_id,
                sender_name=message.sender_name,
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
        self.append_turn(
            conversation_id,
            ConversationTurn(
                role=ConversationRole.ASSISTANT,
                content=text,
                trace_id=trace_id,
                metadata=dict(metadata or {}),
            ),
        )

    def append_tool_turn(self, conversation_id: str, text: str, trace_id: str) -> None:
        self.append_turn(conversation_id, ConversationTurn(role=ConversationRole.TOOL, content=text, trace_id=trace_id))

    def append_turn(self, conversation_id: str, turn: ConversationTurn) -> None:
        with self._lock:
            self.turns.setdefault(conversation_id, []).append(turn)

    def recent_turns(self, conversation_id: str, limit: int = 30) -> list[ConversationTurn]:
        with self._lock:
            return list(self.turns.get(conversation_id, []))[-int(limit):]

    def get_conversation_checkpoint(self, conversation_id: str) -> ConversationCheckpoint | None:
        with self._lock:
            return self.conversation_checkpoints.get(conversation_id)

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
            checkpoint = ConversationCheckpoint(
                conversation_id=conversation_id,
                summary=summary,
                facts=dict(facts),
                open_questions=list(open_questions),
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

    def active_games(self, conversation_id: str | None = None) -> list[Game]:
        with self._lock:
            games = [
                item
                for item in self.games.values()
                if item.status.value in {GameStatus.FORMING.value, GameStatus.INVITING.value, GameStatus.READY.value}
            ]
            if conversation_id:
                scoped = [item for item in games if item.conversation_id == conversation_id]
                return scoped or games
            return games

    def idempotent_result(self, key: str | None) -> ToolResult | None:
        with self._lock:
            return self.idempotency_ledger.get(key or "")

    def claim_idempotent_result(self, key: str | None, claimed_result: ToolResult) -> tuple[bool, ToolResult | None]:
        if not key:
            return True, None
        with self._lock:
            existing = self.idempotency_ledger.get(key)
            if existing is not None:
                return False, existing
            self.idempotency_ledger[key] = claimed_result
            return True, None

    def remember_result(self, key: str | None, result: ToolResult) -> None:
        if not key:
            return
        with self._lock:
            self.idempotency_ledger[key] = result

    def idempotent_message_result(self, message_id: str | None) -> AgentRuntimeResult | None:
        with self._lock:
            return self.message_results.get(message_id or "")

    def remember_message_result(self, message_id: str | None, result: AgentRuntimeResult) -> None:
        if not message_id:
            return
        with self._lock:
            self.message_results.setdefault(message_id, result)

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
                "state_transitions": len(self.transitions),
                "conversation_turns": sum(len(items) for items in self.turns.values()),
                "conversation_checkpoints": len(self.conversation_checkpoints),
                "conversation_versions": len(self.conversation_versions),
                "idempotency_ledger": len(self.idempotency_ledger),
                "message_results": len(self.message_results),
                "message_references": len(self.message_references),
                "customers": len(self.customers) if include_customers else 0,
                "customer_relationships": len(self.customer_relationships) if include_customers else 0,
                "badcases": len(self.badcases) if include_badcases else 0,
            }
            self.games.clear()
            self.invite_drafts.clear()
            self.outbound_message_drafts.clear()
            self.transitions.clear()
            self.turns.clear()
            self.conversation_checkpoints.clear()
            self.conversation_versions.clear()
            self.idempotency_ledger.clear()
            self.message_results.clear()
            self.message_references.clear()
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
    ) -> list[dict[str, Any]]:
        with self._lock:
            requirement = normalize_requirement(requirement)
            scored: list[dict[str, Any]] = []
            requested_seats = requested_seat_count_from_search_requirement(requirement, default=1)
            for game in self.active_games():
                if game.remaining_seats() <= 0:
                    continue
                score, reasons = score_requirement(requirement, game.requirement)
                if requirement and score <= 0:
                    continue
                scored.append(
                    {
                        "game": game.to_dict(),
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
    ) -> list[dict[str, Any]]:
        requirement = normalize_requirement(requirement)
        excluded = set(exclude_customer_ids or [])
        anchor_ids = relationship_anchor_ids(requirement, excluded)
        with self._lock:
            scored: list[dict[str, Any]] = []
            for customer in self.customers.values():
                if customer.no_contact or customer.customer_id in excluded:
                    continue
                if self.active_game_for_customer(customer.customer_id):
                    continue
                score, reasons = score_customer(requirement, customer)
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
                scored.append({"customer": customer.to_dict(), "score": score, "reasons": reasons})
            scored.sort(key=lambda item: item["score"], reverse=True)
            return scored[: int(limit)]

    def active_game_for_customer(self, customer_id: str) -> Game | None:
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
            normalized_requirement = normalize_requirement(requirement)
            default_requester_seat_count = seat_count_from_payload(normalized_requirement, default=1)
            participants = normalize_game_participants(
                organizer_id=organizer_id,
                organizer_name=organizer_name,
                known_players=known_players,
                default_requester_seat_count=default_requester_seat_count,
            )
            parties = normalize_game_parties(participants)
            game = Game(
                game_id=new_id("game"),
                conversation_id=conversation_id,
                organizer_id=organizer_id,
                organizer_name=organizer_name,
                requirement=normalize_requirement_with_party(normalized_requirement, parties),
                participants=participants,
                parties=parties,
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

    def create_invite_drafts(
        self,
        *,
        game_id: str,
        invitations: list[dict[str, Any]],
        trace_id: str,
    ) -> tuple[list[InviteDraft], list[StateTransition]]:
        with self._lock:
            game = self.require_game(game_id)
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
            for draft in self.invite_drafts.values():
                if draft.game_id == game_id and draft.customer_id == customer_id:
                    old = draft.status.value
                    draft.status = invite_status_from_candidate_status(normalized_status)
                    draft.updated_at = now()
                    transitions.append(StateTransition("invite_draft", draft.draft_id, old, draft.status.value, "record_candidate_reply", trace_id))
            existing_participant = next((item for item in game.participants if item.customer_id == customer_id), None)
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
                old = game.status.value
                game.status = GameStatus.READY
                transitions.append(StateTransition("game", game.game_id, old, game.status.value, "seats_full", trace_id))
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
            game.updated_at = now()
            transition = StateTransition("game", game.game_id, old, target.value, reason or "update_game_status", trace_id)
            self.transitions.append(transition)
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
            profile = customers.get(target_id)
            context.append(
                {
                    "customer_id": target_id,
                    "display_name": profile.display_name if profile else participant.display_name,
                    "played_together_count": played_count,
                    "avoid_playing": avoid_playing,
                    "relationship_label": label,
                    "notes": relationship.notes if relationship else "",
                }
            )
    return context


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
                status="joined",
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
