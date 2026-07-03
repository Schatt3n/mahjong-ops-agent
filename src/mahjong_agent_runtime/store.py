from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from .models import (
    AgentRuntimeResultV3,
    ConversationCheckpointV3,
    ConversationRoleV3,
    ConversationTurnV3,
    CustomerProfileV3,
    GameParticipantV3,
    GameStatusV3,
    GameV3,
    InviteDraftV3,
    InviteStatusV3,
    OutboundMessageDraftV3,
    StateTransitionV3,
    ToolResultV3,
    new_id,
    now_v3,
)


ALLOWED_GAME_TRANSITIONS = {
    GameStatusV3.FORMING.value: {
        GameStatusV3.INVITING.value,
        GameStatusV3.READY.value,
        GameStatusV3.CANCELLED.value,
    },
    GameStatusV3.INVITING.value: {
        GameStatusV3.READY.value,
        GameStatusV3.CANCELLED.value,
        GameStatusV3.FINISHED.value,
    },
    GameStatusV3.READY.value: {GameStatusV3.FINISHED.value, GameStatusV3.CANCELLED.value},
    GameStatusV3.CANCELLED.value: set(),
    GameStatusV3.FINISHED.value: set(),
}


@dataclass(slots=True)
class InMemoryAgentStoreV3:
    customers: dict[str, CustomerProfileV3] = field(default_factory=dict)
    games: dict[str, GameV3] = field(default_factory=dict)
    invite_drafts: dict[str, InviteDraftV3] = field(default_factory=dict)
    outbound_message_drafts: dict[str, OutboundMessageDraftV3] = field(default_factory=dict)
    transitions: list[StateTransitionV3] = field(default_factory=list)
    turns: dict[str, list[ConversationTurnV3]] = field(default_factory=dict)
    conversation_checkpoints: dict[str, ConversationCheckpointV3] = field(default_factory=dict)
    idempotency_ledger: dict[str, ToolResultV3] = field(default_factory=dict)
    message_results: dict[str, AgentRuntimeResultV3] = field(default_factory=dict)
    badcases: list[dict[str, Any]] = field(default_factory=list)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    def upsert_customer(self, profile: CustomerProfileV3) -> None:
        with self._lock:
            self.customers[profile.customer_id] = profile

    def append_user_turn(self, message, trace_id: str) -> None:
        self.append_turn(
            message.conversation_id,
            ConversationTurnV3(
                role=ConversationRoleV3.USER,
                content=message.text,
                trace_id=trace_id,
                sender_id=message.sender_id,
                sender_name=message.sender_name,
                occurred_at=message.sent_at,
            ),
        )

    def append_assistant_turn(self, conversation_id: str, text: str, trace_id: str) -> None:
        self.append_turn(conversation_id, ConversationTurnV3(role=ConversationRoleV3.ASSISTANT, content=text, trace_id=trace_id))

    def append_tool_turn(self, conversation_id: str, text: str, trace_id: str) -> None:
        self.append_turn(conversation_id, ConversationTurnV3(role=ConversationRoleV3.TOOL, content=text, trace_id=trace_id))

    def append_turn(self, conversation_id: str, turn: ConversationTurnV3) -> None:
        with self._lock:
            self.turns.setdefault(conversation_id, []).append(turn)

    def recent_turns(self, conversation_id: str, limit: int = 30) -> list[ConversationTurnV3]:
        with self._lock:
            return list(self.turns.get(conversation_id, []))[-int(limit):]

    def get_conversation_checkpoint(self, conversation_id: str) -> ConversationCheckpointV3 | None:
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
    ) -> tuple[ConversationCheckpointV3, StateTransitionV3]:
        with self._lock:
            previous = self.conversation_checkpoints.get(conversation_id)
            checkpoint = ConversationCheckpointV3(
                conversation_id=conversation_id,
                summary=summary,
                facts=dict(facts),
                open_questions=list(open_questions),
                source_trace_id=trace_id,
            )
            self.conversation_checkpoints[conversation_id] = checkpoint
            transition = StateTransitionV3(
                entity_type="conversation_checkpoint",
                entity_id=conversation_id,
                from_status="exists" if previous else None,
                to_status="updated",
                reason="update_context_checkpoint",
                trace_id=trace_id,
            )
            self.transitions.append(transition)
            return checkpoint, transition

    def active_games(self, conversation_id: str | None = None) -> list[GameV3]:
        with self._lock:
            games = [
                item
                for item in self.games.values()
                if item.status.value in {GameStatusV3.FORMING.value, GameStatusV3.INVITING.value, GameStatusV3.READY.value}
            ]
            if conversation_id:
                scoped = [item for item in games if item.conversation_id == conversation_id]
                return scoped or games
            return games

    def idempotent_result(self, key: str | None) -> ToolResultV3 | None:
        with self._lock:
            return self.idempotency_ledger.get(key or "")

    def claim_idempotent_result(self, key: str | None, claimed_result: ToolResultV3) -> tuple[bool, ToolResultV3 | None]:
        if not key:
            return True, None
        with self._lock:
            existing = self.idempotency_ledger.get(key)
            if existing is not None:
                return False, existing
            self.idempotency_ledger[key] = claimed_result
            return True, None

    def remember_result(self, key: str | None, result: ToolResultV3) -> None:
        if not key:
            return
        with self._lock:
            self.idempotency_ledger[key] = result

    def idempotent_message_result(self, message_id: str | None) -> AgentRuntimeResultV3 | None:
        with self._lock:
            return self.message_results.get(message_id or "")

    def remember_message_result(self, message_id: str | None, result: AgentRuntimeResultV3) -> None:
        if not message_id:
            return
        with self._lock:
            self.message_results.setdefault(message_id, result)

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
        with self._lock:
            scored: list[dict[str, Any]] = []
            for customer in self.customers.values():
                if customer.no_contact or customer.customer_id in excluded:
                    continue
                if self.active_game_for_customer(customer.customer_id):
                    continue
                score, reasons = score_customer(requirement, customer)
                if score <= 0:
                    continue
                scored.append({"customer": customer.to_dict(), "score": score, "reasons": reasons})
            scored.sort(key=lambda item: item["score"], reverse=True)
            return scored[: int(limit)]

    def active_game_for_customer(self, customer_id: str) -> GameV3 | None:
        for game in self.games.values():
            if game.status.value not in {GameStatusV3.FORMING.value, GameStatusV3.INVITING.value, GameStatusV3.READY.value}:
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
    ) -> tuple[GameV3, StateTransitionV3]:
        with self._lock:
            game = GameV3(
                game_id=new_id("game"),
                conversation_id=conversation_id,
                organizer_id=organizer_id,
                organizer_name=organizer_name,
                requirement=dict(requirement),
                participants=[
                    GameParticipantV3(
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
            transition = StateTransitionV3(
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
    ) -> tuple[list[InviteDraftV3], list[StateTransitionV3]]:
        with self._lock:
            game = self.require_game(game_id)
            transitions: list[StateTransitionV3] = []
            if game.status == GameStatusV3.FORMING:
                old = game.status.value
                game.status = GameStatusV3.INVITING
                game.updated_at = now_v3()
                transitions.append(
                    StateTransitionV3("game", game.game_id, old, game.status.value, "create_invite_drafts", trace_id)
                )
            drafts: list[InviteDraftV3] = []
            for raw in invitations:
                if not isinstance(raw, dict):
                    continue
                draft = InviteDraftV3(
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
                    StateTransitionV3("invite_draft", draft.draft_id, None, draft.status.value, "create_invite_drafts", trace_id)
                )
            self.transitions.extend(transitions)
            return drafts, transitions

    def create_outbound_message_drafts(
        self,
        *,
        conversation_id: str,
        drafts: list[dict[str, Any]],
        trace_id: str,
    ) -> tuple[list[OutboundMessageDraftV3], list[StateTransitionV3]]:
        with self._lock:
            created: list[OutboundMessageDraftV3] = []
            transitions: list[StateTransitionV3] = []
            for raw in drafts:
                if not isinstance(raw, dict):
                    continue
                draft = OutboundMessageDraftV3(
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
                    StateTransitionV3(
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
    ) -> tuple[GameV3, list[StateTransitionV3]]:
        with self._lock:
            game = self.require_game(game_id)
            transitions: list[StateTransitionV3] = []
            normalized_status = status.strip()
            for draft in self.invite_drafts.values():
                if draft.game_id == game_id and draft.customer_id == customer_id:
                    old = draft.status.value
                    draft.status = invite_status_from_candidate_status(normalized_status)
                    draft.updated_at = now_v3()
                    transitions.append(StateTransitionV3("invite_draft", draft.draft_id, old, draft.status.value, "record_candidate_reply", trace_id))
            if normalized_status in {"accepted", "confirmed", "arrived"} and not any(
                item.customer_id == customer_id and item.status in {"joined", "confirmed"} for item in game.participants
            ):
                game.participants.append(
                    GameParticipantV3(
                        customer_id=customer_id,
                        display_name=display_name or customer_id,
                        status="confirmed",
                        source="candidate_reply",
                    )
                )
                transitions.append(
                    StateTransitionV3(
                        "game_participant",
                        f"{game.game_id}:{customer_id}",
                        None,
                        "confirmed",
                        "record_candidate_reply",
                        trace_id,
                    )
                )
            if game.remaining_seats() == 0 and game.status != GameStatusV3.READY:
                old = game.status.value
                game.status = GameStatusV3.READY
                transitions.append(StateTransitionV3("game", game.game_id, old, game.status.value, "seats_full", trace_id))
            game.updated_at = now_v3()
            self.transitions.extend(transitions)
            return game, transitions

    def update_game_status(self, *, game_id: str, status: str, reason: str, trace_id: str) -> tuple[GameV3, StateTransitionV3]:
        with self._lock:
            game = self.require_game(game_id)
            target = GameStatusV3(status)
            old = game.status.value
            allowed = ALLOWED_GAME_TRANSITIONS.get(old, set())
            if target.value != old and target.value not in allowed:
                raise ValueError(f"illegal game status transition: {old}->{target.value}")
            game.status = target
            game.updated_at = now_v3()
            transition = StateTransitionV3("game", game.game_id, old, target.value, reason or "update_game_status", trace_id)
            self.transitions.append(transition)
            return game, transition

    def record_badcase(self, payload: dict[str, Any], *, trace_id: str, conversation_id: str) -> dict[str, Any]:
        with self._lock:
            record = {"badcase_id": new_id("badcase"), "trace_id": trace_id, "conversation_id": conversation_id, **dict(payload)}
            self.badcases.append(record)
            return record

    def require_game(self, game_id: str) -> GameV3:
        game = self.games.get(game_id)
        if game is None:
            raise ValueError(f"game not found: {game_id}")
        return game


def score_requirement(query: dict[str, Any], target: dict[str, Any]) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []
    for key, weight in {
        "game_type": 30,
        "stake": 25,
        "smoke_preference": 15,
        "start_time_kind": 10,
        "duration_kind": 10,
    }.items():
        query_value = query.get(key)
        if query_value in {None, "", []}:
            continue
        target_value = target.get(key)
        if value_matches(query_value, target_value):
            score += weight
            reasons.append(f"{key}_matched")
        elif key in {"game_type", "stake", "smoke_preference"}:
            score -= weight
            reasons.append(f"{key}_mismatched")
    return score, reasons


def score_customer(requirement: dict[str, Any], customer: CustomerProfileV3) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []
    if value_matches(requirement.get("game_type"), customer.preferred_games):
        score += 30
        reasons.append("game_type_matched")
    if value_matches(requirement.get("stake"), customer.preferred_stakes):
        score += 25
        reasons.append("stake_matched")
    if smoke_matches(requirement.get("smoke_preference"), customer.smoke_preference):
        score += 10
        reasons.append("smoke_matched")
    gender = requirement.get("preferred_gender") or requirement.get("gender")
    if gender and customer.gender == gender:
        score += 10
        reasons.append("gender_matched")
    score += int(max(0.0, min(1.0, customer.response_score)) * 10)
    score -= int(max(0.0, customer.fatigue_score) * 10)
    return score, reasons


def value_matches(query_value: Any, target_value: Any) -> bool:
    if query_value in {None, "", []}:
        return False
    query_values = set(str(item) for item in query_value) if isinstance(query_value, list) else {str(query_value)}
    target_values = set(str(item) for item in target_value) if isinstance(target_value, list) else {str(target_value)}
    return bool(query_values & target_values)


def smoke_matches(query_value: Any, target_value: Any) -> bool:
    if query_value in {None, "", "any"}:
        return True
    if target_value in {None, "", "any"}:
        return True
    return value_matches(query_value, target_value)


def invite_status_from_candidate_status(status: str) -> InviteStatusV3:
    mapping = {
        "accepted": InviteStatusV3.CONFIRMED,
        "confirmed": InviteStatusV3.CONFIRMED,
        "arrived": InviteStatusV3.CONFIRMED,
        "declined": InviteStatusV3.DECLINED,
        "negotiating": InviteStatusV3.NEGOTIATING,
        "no_reply": InviteStatusV3.NO_REPLY,
    }
    return mapping.get(status, InviteStatusV3.NEGOTIATING)
