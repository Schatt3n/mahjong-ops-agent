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
    RecruitmentStatus,
    RoomReservation,
    ScheduledAgentTask,
    ScheduledTaskStatus,
    StateTransition,
    TaskMemory,
    ToolResult,
    new_id,
    now,
)
from .stores.idempotency_common import (
    IDEMPOTENCY_CLAIM_LEASE_SECONDS,
    tool_result_is_in_progress,
)
from .stores.memory.idempotency import InMemoryIdempotencyStoreMixin


DEFAULT_ASAP_GAME_TTL_HOURS = 4
DEFAULT_UNKNOWN_DURATION_HOURS = 4
DEFAULT_OVERNIGHT_DURATION_HOURS = 8
START_KIND_SCHEDULED = "scheduled"
START_KIND_ASAP_WHEN_FULL = "asap_" "when_full"
DURATION_KIND_OVERNIGHT = "overnight"
PENDING_INPUT_PROCESSING_LEASE_SECONDS = 120
SCHEDULED_TASK_PROCESSING_LEASE_SECONDS = 120
DEFAULT_RECRUITMENT_LEAD_HOURS = 2
GAME_RECRUITMENT_TASK_TYPE = "activate_game_recruitment"

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
    "recruitment_opens_at",
    "recruitment_status",
    "recruitment_lead_hours",
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
    apply_game_recruitment_policy(game)


def game_recruitment_task_id(game_id: str) -> str:
    """Use a deterministic task id so schedule updates cannot create duplicates."""

    return f"scheduled_recruitment_{game_id}"


def recruitment_open_time(game: Game) -> datetime | None:
    if game.planned_start_at is None:
        return None
    return normalize_datetime(game.planned_start_at) - timedelta(hours=DEFAULT_RECRUITMENT_LEAD_HOURS)


def apply_game_recruitment_policy(game: Game, *, at: datetime | None = None) -> None:
    """Derive the proactive private-outreach boundary from business time.

    This is a temporal invariant, not semantic interpretation: every caller and
    every model is subject to the same T-2h rule. Public game-list visibility is
    unaffected; only candidate outreach is delayed.
    """

    stamp = normalize_datetime(at or now())
    opens_at = recruitment_open_time(game)
    game.recruitment_opens_at = opens_at
    if game.status == GameStatus.CANCELLED:
        status = RecruitmentStatus.CANCELLED
    elif game.status == GameStatus.FINISHED or game.status == GameStatus.READY:
        status = RecruitmentStatus.COMPLETED
    elif opens_at is not None and stamp < opens_at:
        status = RecruitmentStatus.SCHEDULED
    elif game.status == GameStatus.INVITING:
        status = RecruitmentStatus.ACTIVE
    else:
        status = RecruitmentStatus.OPEN
    game.recruitment_status = status
    patch = {
        "recruitment_status": status.value,
        "recruitment_lead_hours": DEFAULT_RECRUITMENT_LEAD_HOURS,
    }
    if opens_at is not None:
        patch["recruitment_opens_at"] = opens_at.isoformat()
    else:
        game.requirement.pop("recruitment_opens_at", None)
    game.requirement = {**dict(game.requirement), **patch}


def game_schedule_sort_key(game: Game) -> tuple[datetime, datetime, str]:
    """Put fixed-time games first in chronological order, then flexible games."""

    distant_future = datetime.max.replace(tzinfo=DEFAULT_TZ)
    return (
        normalize_datetime(game.planned_start_at) if game.planned_start_at else distant_future,
        normalize_datetime(game.created_at),
        game.game_id,
    )


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
    apply_game_recruitment_policy(game, at=at)
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
        "start_time",
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
                status=canonical_game_participant_status(requester_payload.get("status")),
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
                status=canonical_game_participant_status(item.get("status")),
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


def canonical_game_participant_status(value: Any) -> str:
    """Keep role labels out of the participant state machine.

    Role labels such as ``organizer`` belong in ``source`` or party metadata;
    accepting them as participant states would make occupied seats disappear
    from the aggregate. Other persisted states such as ``declined`` and
    ``superseded`` must survive SQLite deserialization unchanged.
    """

    status = str(value or "").strip().lower()
    return "joined" if not status or status == "organizer" else status


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

# Compatibility aggregate: behavior lives in stores/memory/*.py.
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
