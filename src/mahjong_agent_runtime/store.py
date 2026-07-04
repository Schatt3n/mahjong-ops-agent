from __future__ import annotations

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
    OutboundMessageDraft,
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
    idempotency_ledger: dict[str, ToolResult] = field(default_factory=dict)
    message_results: dict[str, AgentRuntimeResult] = field(default_factory=dict)
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

    def append_assistant_turn(self, conversation_id: str, text: str, trace_id: str) -> None:
        self.append_turn(conversation_id, ConversationTurn(role=ConversationRole.ASSISTANT, content=text, trace_id=trace_id))

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
                "idempotency_ledger": len(self.idempotency_ledger),
                "message_results": len(self.message_results),
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
            self.idempotency_ledger.clear()
            self.message_results.clear()
            if include_customers:
                self.customers.clear()
                self.customer_relationships.clear()
            if include_badcases:
                self.badcases.clear()
            return deleted

    def search_current_games(self, requirement: dict[str, Any], limit: int = 8) -> list[dict[str, Any]]:
        with self._lock:
            scored: list[dict[str, Any]] = []
            for game in self.active_games():
                if game.remaining_seats() <= 0:
                    continue
                score, reasons = score_requirement(requirement, game.requirement)
                if requirement and score <= 0:
                    continue
                scored.append({"game": game.to_dict(), "score": score, "reasons": reasons or ["active_open_game"]})
            scored.sort(key=lambda item: item["score"], reverse=True)
            return scored[: int(limit)]

    def search_customers(
        self,
        requirement: dict[str, Any],
        *,
        exclude_customer_ids: list[str] | None = None,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
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
            game = Game(
                game_id=new_id("game"),
                conversation_id=conversation_id,
                organizer_id=organizer_id,
                organizer_name=organizer_name,
                requirement=dict(requirement),
                participants=[
                    GameParticipant(
                        customer_id=str(item.get("customer_id") or ""),
                        display_name=str(item.get("display_name") or item.get("customer_id") or ""),
                        status=str(item.get("status") or "joined"),
                        source=str(item.get("source") or "organizer"),
                    )
                    for item in known_players
                    if isinstance(item, dict)
                ],
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
        trace_id: str,
    ) -> tuple[Game, list[StateTransition]]:
        with self._lock:
            game = self.require_game(game_id)
            transitions: list[StateTransition] = []
            normalized_status = status.strip()
            for draft in self.invite_drafts.values():
                if draft.game_id == game_id and draft.customer_id == customer_id:
                    old = draft.status.value
                    draft.status = invite_status_from_candidate_status(normalized_status)
                    draft.updated_at = now()
                    transitions.append(StateTransition("invite_draft", draft.draft_id, old, draft.status.value, "record_candidate_reply", trace_id))
            if normalized_status in {"accepted", "confirmed", "arrived"} and not any(
                item.customer_id == customer_id and item.status in {"joined", "confirmed"} for item in game.participants
            ):
                game.participants.append(
                    GameParticipant(
                        customer_id=customer_id,
                        display_name=display_name or customer_id,
                        status="confirmed",
                        source="candidate_reply",
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
            if game.remaining_seats() == 0 and game.status != GameStatus.READY:
                old = game.status.value
                game.status = GameStatus.READY
                transitions.append(StateTransition("game", game.game_id, old, game.status.value, "seats_full", trace_id))
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
    return score, reasons


def score_customer(requirement: dict[str, Any], customer: CustomerProfile) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []
    game_query = first_present_value(requirement, "game_type", "preferred_game", "preferred_games", "game_types")
    stake_query = first_present_value(requirement, "stake", "preferred_stake", "preferred_stakes", "stakes")
    smoke_query = first_present_value(requirement, "smoke_preference", "smoke")
    if value_matches(game_query, customer.preferred_games):
        score += 30
        reasons.append("game_type_matched")
    if value_matches(stake_query, customer.preferred_stakes):
        score += 25
        reasons.append("stake_matched")
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
