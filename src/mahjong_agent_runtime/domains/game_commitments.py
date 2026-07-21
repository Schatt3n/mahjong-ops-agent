"""Domain rules for game commitments."""

from __future__ import annotations

from typing import Any
from dataclasses import dataclass
from datetime import datetime
from dataclasses import field
from datetime import timedelta
from ..models import (
    Game,
    GameStatus,
    InviteDraft,
    InviteStatus,
    OPEN_INVITE_STATUSES,
    StateTransition,
    now,
)
from .game_domain import (
    DEFAULT_ASAP_GAME_TTL_HOURS,
    derive_game_lifecycle,
    duration_hours_from_requirement,
    normalize_datetime,
)
from .game_participants import (
    normalize_game_parties,
    refresh_requirement_seat_snapshot,
)

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
