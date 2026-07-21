"""SQLite game mutations store operations."""

from __future__ import annotations

from typing import Any
from datetime import datetime
from dataclasses import replace
from ...models import (
    DEFAULT_TZ,
    Game,
    GameStatus,
    StateTransition,
    new_id,
)
from ...store import (
    ALLOWED_GAME_TRANSITIONS,
    PROTECTED_REQUIREMENT_PATCH_FIELDS,
    active_game_participant_ids,
    apply_game_lifecycle,
    apply_game_recruitment_policy,
    expire_game_if_stale,
    normalize_game_participants,
    normalize_game_parties,
    normalize_requirement,
    normalize_requirement_with_party,
    ready_commitment_conflicts,
    refresh_requirement_seat_snapshot,
    requirement_overlaps_game,
    seat_count_from_payload,
)
from .serialization import _loads

class SQLiteGameMutationsStoreMixin:
    """Backend-specific operations extracted from the compatibility store."""

    __slots__ = ()

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
        with self._write_transaction():
            from ...models import new_id

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
            transition = StateTransition("game", game.game_id, None, game.status.value, "create_game", trace_id)
            self._save_game(game)
            self._append_transition(transition)
            _, recruitment_transition = self._sync_game_recruitment_task_in_transaction(game, trace_id=trace_id)
            self._save_game(game)
            if recruitment_transition is not None:
                self._append_transition(recruitment_transition)
            return game, transition

    def _expire_stale_games(self, *, trace_id: str) -> list[StateTransition]:
        with self._lock, self._connection:
            stamp = datetime.now(DEFAULT_TZ)
            transitions: list[StateTransition] = []
            for game in self.games.values():
                transition = expire_game_if_stale(game, at=stamp, trace_id=trace_id)
                if transition is None:
                    continue
                transitions.append(transition)
                self._save_game(game)
                self._append_transition(transition)
                _, recruitment_transition = self._sync_game_recruitment_task_in_transaction(
                    game,
                    trace_id=trace_id,
                )
                self._save_game(game)
                if recruitment_transition is not None:
                    transitions.append(recruitment_transition)
                    self._append_transition(recruitment_transition)
                released = self._release_room_reservations_for_game(
                    game.game_id,
                    trace_id=trace_id,
                    reason="game_lifecycle_closed",
                )
                transitions.extend(released)
            return transitions

    def update_game_requirement(
        self,
        *,
        game_id: str,
        requirement_patch: dict[str, Any],
        reason: str,
        trace_id: str,
    ) -> tuple[Game, StateTransition]:
        """Persist a user-confirmed condition revision in one write transaction."""

        with self._write_transaction():
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
            game.updated_at = datetime.now(DEFAULT_TZ)
            transition = StateTransition(
                "game_requirement",
                game.game_id,
                "configured",
                "configured",
                reason or "update_game_requirement",
                trace_id,
            )
            self._save_game(game)
            self._append_transition(transition)
            _, recruitment_transition = self._sync_game_recruitment_task_in_transaction(game, trace_id=trace_id)
            self._save_game(game)
            if recruitment_transition is not None:
                self._append_transition(recruitment_transition)
            return game, transition

    def join_game(
        self,
        *,
        game_id: str,
        customer_id: str,
        display_name: str,
        seat_count: int = 1,
        trace_id: str,
    ) -> tuple[Game, list[StateTransition]]:
        """Join through the normalized participant table and shared invariants."""

        return self.record_candidate_reply(
            game_id=game_id,
            customer_id=customer_id,
            display_name=display_name,
            status="confirmed",
            seat_count=seat_count,
            trace_id=trace_id,
        )

    def update_game_status(self, *, game_id: str, status: str, reason: str, trace_id: str) -> tuple[Game, StateTransition]:
        with self._lock, self._connection:
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
            game.updated_at = datetime.now(DEFAULT_TZ)
            transition = StateTransition("game", game.game_id, old, target.value, reason or "update_game_status", trace_id)
            self._save_game(game)
            self._append_transition(transition)
            if target in {GameStatus.CANCELLED, GameStatus.FINISHED}:
                self._release_room_reservations_for_game(
                    game.game_id,
                    trace_id=trace_id,
                    reason="game_status_closed",
                )
                _, recruitment_transition = self._sync_game_recruitment_task_in_transaction(
                    game,
                    trace_id=trace_id,
                )
                self._save_game(game)
                if recruitment_transition is not None:
                    self._append_transition(recruitment_transition)
            return game, transition

    def require_game(self, game_id: str) -> Game:
        with self._lock:
            row = self._connection.execute("SELECT payload FROM runtime_games WHERE game_id = ?", (game_id,)).fetchone()
            if row is None:
                raise ValueError(f"game not found: {game_id}")
            return self._hydrate_game(_loads(row["payload"]))
