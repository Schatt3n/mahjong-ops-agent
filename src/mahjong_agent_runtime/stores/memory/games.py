"""InMemory games store operations."""

from __future__ import annotations

from typing import Any
from dataclasses import replace
from ...models import (
    Game,
    GameStatus,
    StateTransition,
    new_id,
    now,
)
from ...store import (
    ALLOWED_GAME_TRANSITIONS,
    PROTECTED_REQUIREMENT_PATCH_FIELDS,
    active_game_participant_ids,
    apply_game_lifecycle,
    apply_game_recruitment_policy,
    expire_game_if_stale,
    game_contains_customer,
    game_for_model_context,
    join_projection,
    normalize_game_participants,
    normalize_game_parties,
    normalize_requirement,
    normalize_requirement_with_party,
    ready_commitment_conflicts,
    refresh_requirement_seat_snapshot,
    requested_seat_count_from_search_requirement,
    requirement_overlaps_game,
    score_requirement,
    seat_count_from_payload,
    task_memory_anchor_ids,
)

class InMemoryGamesStoreMixin:
    """Backend-specific operations extracted from the compatibility store."""

    __slots__ = ()

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
            normalized_requirement = normalize_requirement(requirement)
            duplicate = next(
                (
                    item
                    for item in self.games.values()
                    if item.conversation_id == conversation_id
                    and item.organizer_id == organizer_id
                    and item.status in {GameStatus.FORMING, GameStatus.INVITING, GameStatus.READY}
                    and requirement_overlaps_game(normalized_requirement, item)
                ),
                None,
            )
            if duplicate is not None:
                raise ValueError(f"active game already exists: {duplicate.game_id}")
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
            self._sync_game_recruitment_task_locked(game, trace_id=trace_id)
            return game, transition

    def _expire_stale_games_locked(self, *, trace_id: str) -> list[StateTransition]:
        stamp = now()
        transitions: list[StateTransition] = []
        for game in self.games.values():
            transition = expire_game_if_stale(game, at=stamp, trace_id=trace_id)
            if transition is not None:
                transitions.append(transition)
                _, recruitment_transition = self._sync_game_recruitment_task_locked(game, trace_id=trace_id)
                if recruitment_transition is not None:
                    transitions.append(recruitment_transition)
                transitions.extend(
                    self._release_room_reservations_for_game_locked(
                        game.game_id,
                        trace_id=trace_id,
                        reason="game_lifecycle_closed",
                    )
                )
        if transitions:
            known_transition_ids = {id(item) for item in self.transitions}
            self.transitions.extend(item for item in transitions if id(item) not in known_transition_ids)
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
                "recruitment_opens_at",
                "recruitment_status",
                "recruitment_lead_hours",
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
            game.recruitment_opens_at = prospective.recruitment_opens_at
            game.recruitment_status = prospective.recruitment_status
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
            self._sync_game_recruitment_task_locked(game, trace_id=trace_id)
            return game, transition

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
            apply_game_recruitment_policy(game)
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
                self._sync_game_recruitment_task_locked(game, trace_id=trace_id)
            return game, transition

    def require_game(self, game_id: str) -> Game:
        game = self.games.get(game_id)
        if game is None:
            raise ValueError(f"game not found: {game_id}")
        return game
