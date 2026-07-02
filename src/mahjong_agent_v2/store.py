from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .models import (
    ConversationRoleV2,
    ConversationTurnV2,
    CustomerProfileV2,
    DEFAULT_TZ_V2,
    GameParticipantV2,
    GameStatusV2,
    GameV2,
    InviteDraftV2,
    InviteStatusV2,
    StateTransitionV2,
    ToolResultV2,
    new_id,
)
from .state_policy import StatePolicyV2


@dataclass(slots=True)
class InMemoryAgentStoreV2:
    """A small production-shaped store for the independent V2 runtime.

    It is intentionally not wired to the legacy workflow store. The locking,
    idempotency ledger, state transitions, games, customers and outbox drafts
    are owned by this V2 runtime so tests can prove the new main chain is clean.
    """

    customers: dict[str, CustomerProfileV2] = field(default_factory=dict)
    games: dict[str, GameV2] = field(default_factory=dict)
    invite_drafts: dict[str, InviteDraftV2] = field(default_factory=dict)
    turns_by_conversation: dict[str, list[ConversationTurnV2]] = field(default_factory=dict)
    idempotency_ledger: dict[str, ToolResultV2] = field(default_factory=dict)
    message_result_ledger: dict[str, Any] = field(default_factory=dict)
    transitions: list[StateTransitionV2] = field(default_factory=list)
    state_policy: StatePolicyV2 = field(default_factory=StatePolicyV2.default)
    _lock: threading.RLock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._lock = threading.RLock()

    def upsert_customer(self, profile: CustomerProfileV2) -> None:
        with self._lock:
            self.customers[profile.customer_id] = profile

    def append_turn(self, conversation_id: str, turn: ConversationTurnV2) -> None:
        with self._lock:
            self.turns_by_conversation.setdefault(conversation_id, []).append(turn)

    def append_user_turn(self, message, trace_id: str) -> None:
        self.append_turn(
            message.conversation_id,
            ConversationTurnV2(
                role=ConversationRoleV2.USER,
                content=message.text,
                trace_id=trace_id,
                sender_id=message.sender_id,
                sender_name=message.sender_name,
                occurred_at=message.sent_at,
            ),
        )

    def append_assistant_turn(self, conversation_id: str, content: str, trace_id: str) -> None:
        self.append_turn(
            conversation_id,
            ConversationTurnV2(
                role=ConversationRoleV2.ASSISTANT,
                content=content,
                trace_id=trace_id,
            ),
        )

    def append_tool_turn(self, conversation_id: str, content: str, trace_id: str) -> None:
        self.append_turn(
            conversation_id,
            ConversationTurnV2(
                role=ConversationRoleV2.TOOL,
                content=content,
                trace_id=trace_id,
            ),
        )

    def recent_turns(self, conversation_id: str, limit: int = 12) -> list[ConversationTurnV2]:
        with self._lock:
            return list(self.turns_by_conversation.get(conversation_id, []))[-limit:]

    def active_games(self, conversation_id: str | None = None) -> list[GameV2]:
        with self._lock:
            games = [
                game
                for game in self.games.values()
                if game.status in self.state_policy.active_game_statuses
            ]
            if conversation_id:
                scoped = [game for game in games if game.conversation_id == conversation_id]
                return scoped or games
            return games

    def idempotent_result(self, key: str | None) -> ToolResultV2 | None:
        if not key:
            return None
        with self._lock:
            return self.idempotency_ledger.get(key)

    def remember_result(self, key: str | None, result: ToolResultV2) -> None:
        if not key:
            return
        with self._lock:
            self.idempotency_ledger[key] = result

    def idempotent_message_result(self, message_id: str | None):
        if not message_id:
            return None
        with self._lock:
            return self.message_result_ledger.get(message_id)

    def remember_message_result(self, message_id: str | None, result) -> None:
        if not message_id:
            return
        with self._lock:
            self.message_result_ledger[message_id] = result

    def search_current_games(self, requirement: dict[str, Any], limit: int = 8) -> list[dict[str, Any]]:
        with self._lock:
            scored: list[dict[str, Any]] = []
            for game in self.active_games():
                score, reasons = _score_requirement(requirement, game.requirement)
                if game.remaining_seats() <= 0:
                    continue
                if score <= 0 and requirement:
                    continue
                scored.append(
                    {
                        "game": game.to_dict(),
                        "score": score,
                        "reasons": reasons or ["active_open_game"],
                    }
                )
            scored.sort(key=lambda item: item["score"], reverse=True)
            return scored[:limit]

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
                active_game = self.active_game_for_customer(customer.customer_id)
                if active_game is not None:
                    continue
                score, reasons = _score_customer(requirement, customer)
                if score <= 0:
                    continue
                scored.append({"customer": customer.to_dict(), "score": score, "reasons": reasons})
            scored.sort(key=lambda item: item["score"], reverse=True)
            return scored[:limit]

    def create_game(
        self,
        *,
        conversation_id: str,
        organizer_id: str,
        organizer_name: str,
        requirement: dict[str, Any],
        known_players: list[dict[str, Any]] | None,
        trace_id: str,
    ) -> tuple[GameV2, StateTransitionV2]:
        with self._lock:
            game = GameV2(
                game_id=new_id("gamev2"),
                conversation_id=conversation_id,
                organizer_id=organizer_id,
                organizer_name=organizer_name,
                requirement=dict(requirement),
                participants=[],
            )
            self.state_policy.ensure_game_transition(None, game.status)
            players = list(known_players or [])
            if not players:
                players = [{"customer_id": organizer_id, "display_name": organizer_name, "source": "organizer"}]
            for player in players:
                customer_id = str(player.get("customer_id") or player.get("id") or "").strip()
                display_name = str(player.get("display_name") or player.get("name") or customer_id or "客户").strip()
                if not customer_id:
                    customer_id = new_id("guest")
                if not any(participant.customer_id == customer_id for participant in game.participants):
                    game.participants.append(
                        GameParticipantV2(
                            customer_id=customer_id,
                            display_name=display_name,
                            status=str(player.get("status") or "joined"),
                            source=str(player.get("source") or "organizer"),
                        )
                    )
            self.games[game.game_id] = game
            transition = self._transition(
                entity_type="game",
                entity_id=game.game_id,
                from_status=None,
                to_status=game.status.value,
                reason="create_game",
                trace_id=trace_id,
            )
            return game, transition

    def create_invite_drafts(
        self,
        *,
        game_id: str,
        invitations: list[dict[str, Any]],
        trace_id: str,
    ) -> tuple[list[InviteDraftV2], list[StateTransitionV2]]:
        with self._lock:
            game = self.games.get(game_id)
            if game is None:
                raise ValueError(f"game_id not found: {game_id}")
            self.state_policy.ensure_can_create_invite_drafts(game)
            drafts: list[InviteDraftV2] = []
            transitions: list[StateTransitionV2] = []
            for invitation in invitations:
                customer_id = str(invitation.get("customer_id") or "").strip()
                message_text = str(invitation.get("message_text") or "").strip()
                if not customer_id or not message_text:
                    raise ValueError("each invitation requires customer_id and message_text")
                if self.active_game_for_customer(customer_id) is not None:
                    continue
                profile = self.customers.get(customer_id)
                display_name = str(invitation.get("display_name") or (profile.display_name if profile else customer_id))
                draft = InviteDraftV2(
                    draft_id=new_id("draftv2"),
                    game_id=game_id,
                    customer_id=customer_id,
                    display_name=display_name,
                    message_text=message_text,
                    metadata={"trace_id": trace_id},
                )
                self.invite_drafts[draft.draft_id] = draft
                drafts.append(draft)
            if drafts and game.status == GameStatusV2.FORMING:
                from_status = game.status.value
                self.state_policy.ensure_game_transition(game.status, GameStatusV2.INVITING)
                game.status = GameStatusV2.INVITING
                game.updated_at = datetime.now(DEFAULT_TZ_V2)
                transitions.append(
                    self._transition(
                        entity_type="game",
                        entity_id=game.game_id,
                        from_status=from_status,
                        to_status=game.status.value,
                        reason="invite_drafts_created",
                        trace_id=trace_id,
                    )
                )
            return drafts, transitions

    def record_candidate_reply(
        self,
        *,
        game_id: str,
        customer_id: str,
        status: str,
        trace_id: str,
    ) -> tuple[GameV2, list[StateTransitionV2]]:
        with self._lock:
            game = self.games.get(game_id)
            if game is None:
                raise ValueError(f"game_id not found: {game_id}")
            next_status = InviteStatusV2(status)
            transitions: list[StateTransitionV2] = []
            drafts = [
                draft
                for draft in self.invite_drafts.values()
                if draft.game_id == game_id and draft.customer_id == customer_id
            ]
            self.state_policy.ensure_candidate_reply_allowed(game, drafts)
            for draft in drafts:
                self.state_policy.ensure_invite_transition(draft.status, next_status)
                if draft.status == next_status:
                    continue
                from_status = draft.status.value
                draft.status = next_status
                draft.updated_at = datetime.now(DEFAULT_TZ_V2)
                transitions.append(
                    self._transition(
                        entity_type="invite_draft",
                        entity_id=draft.draft_id,
                        from_status=from_status,
                        to_status=draft.status.value,
                        reason="candidate_reply",
                        trace_id=trace_id,
                    )
                )
            if next_status == InviteStatusV2.CONFIRMED and not any(
                participant.customer_id == customer_id for participant in game.participants
            ):
                profile = self.customers.get(customer_id)
                game.participants.append(
                    GameParticipantV2(
                        customer_id=customer_id,
                        display_name=profile.display_name if profile else customer_id,
                        status="confirmed",
                        source="candidate_reply",
                    )
                )
            if game.remaining_seats() == 0 and game.status != GameStatusV2.READY:
                from_status = game.status.value
                self.state_policy.ensure_game_transition(game.status, GameStatusV2.READY)
                game.status = GameStatusV2.READY
                game.updated_at = datetime.now(DEFAULT_TZ_V2)
                transitions.append(
                    self._transition(
                        entity_type="game",
                        entity_id=game.game_id,
                        from_status=from_status,
                        to_status=game.status.value,
                        reason="all_seats_confirmed",
                        trace_id=trace_id,
                    )
                )
            return game, transitions

    def update_game_status(
        self,
        *,
        game_id: str,
        status: str,
        reason: str,
        trace_id: str,
    ) -> tuple[GameV2, StateTransitionV2]:
        with self._lock:
            game = self.games.get(game_id)
            if game is None:
                raise ValueError(f"game_id not found: {game_id}")
            next_status = GameStatusV2(status)
            from_status = game.status.value
            self.state_policy.ensure_game_transition(game.status, next_status)
            game.status = next_status
            game.updated_at = datetime.now(DEFAULT_TZ_V2)
            transition = self._transition(
                entity_type="game",
                entity_id=game.game_id,
                from_status=from_status,
                to_status=game.status.value,
                reason=reason or "update_game_status",
                trace_id=trace_id,
            )
            return game, transition

    def active_game_for_customer(self, customer_id: str) -> GameV2 | None:
        for game in self.games.values():
            if game.status not in self.state_policy.active_game_statuses:
                continue
            if any(
                participant.customer_id == customer_id and participant.status in {"joined", "confirmed"}
                for participant in game.participants
            ):
                return game
        for draft in self.invite_drafts.values():
            if draft.customer_id == customer_id and draft.status in {
                *self.state_policy.occupied_invite_statuses,
            }:
                game = self.games.get(draft.game_id)
                if game and game.status in self.state_policy.active_game_statuses:
                    return game
        return None

    def _transition(
        self,
        *,
        entity_type: str,
        entity_id: str,
        from_status: str | None,
        to_status: str,
        reason: str,
        trace_id: str,
    ) -> StateTransitionV2:
        transition = StateTransitionV2(
            entity_type=entity_type,
            entity_id=entity_id,
            from_status=from_status,
            to_status=to_status,
            reason=reason,
            trace_id=trace_id,
        )
        self.transitions.append(transition)
        return transition


_REQUIREMENT_FIELD_ALIASES: tuple[tuple[str, tuple[str, ...], int], ...] = (
    ("game_type", ("game_type",), 30),
    ("stake", ("stake", "stake_options"), 25),
    ("smoke_preference", ("smoke_preference",), 15),
    ("start_time_kind", ("start_time_kind",), 10),
    ("duration_kind", ("duration_kind",), 10),
)

CANONICAL_SMOKE_VALUES = {"any", "non_smoking", "smoke_ok", "unknown"}


def _score_requirement(query: dict[str, Any], candidate: dict[str, Any]) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []
    for logical_name, aliases, weight in _REQUIREMENT_FIELD_ALIASES:
        expected = _requirement_value(query, aliases)
        actual = _requirement_value(candidate, aliases)
        if expected in (None, "", [], {}):
            continue
        if actual in (None, "", [], {}):
            continue
        if _compatible(expected, actual):
            score += weight
            reasons.append(f"{logical_name}_matched")
        else:
            return 0, []
    return score, reasons


def _score_customer(requirement: dict[str, Any], customer: CustomerProfileV2) -> tuple[float, list[str]]:
    score = 30.0 * max(0.0, min(1.0, customer.response_score)) - 20.0 * max(0.0, customer.fatigue_score)
    reasons: list[str] = []
    game_type = str(_requirement_value(requirement, ("game_type",)) or "")
    stake = str(_requirement_value(requirement, ("stake",)) or "")
    smoke = str(_requirement_value(requirement, ("smoke_preference",)) or "")
    if game_type and game_type in customer.preferred_games:
        score += 30
        reasons.append("game_preference_matched")
    if stake and stake in customer.preferred_stakes:
        score += 25
        reasons.append("stake_matched")
    if smoke in CANONICAL_SMOKE_VALUES and customer.smoke_preference in {smoke, "any", None}:
        score += 10
        reasons.append("smoke_compatible")
    if not game_type and customer.preferred_games:
        score += 5
        reasons.append("has_game_profile")
    if not stake and customer.preferred_stakes:
        score += 5
        reasons.append("has_stake_profile")
    return score, reasons


def _requirement_value(payload: dict[str, Any], aliases: tuple[str, ...]) -> Any:
    for alias in aliases:
        value = payload.get(alias)
        if value not in (None, "", [], {}):
            return value
    return None


def _compatible(expected: Any, actual: Any) -> bool:
    if isinstance(expected, list):
        return actual in expected or any(_compatible(item, actual) for item in expected)
    if isinstance(actual, list):
        return expected in actual or any(_compatible(expected, item) for item in actual)
    if expected == "any" or actual == "any":
        return True
    return str(expected) == str(actual)
