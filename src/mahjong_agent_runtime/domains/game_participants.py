"""Domain rules for game participants."""

from __future__ import annotations

from typing import Any
from ..models import (
    Game,
    GameParticipant,
    Party,
)
from .game_domain import normalize_requirement
from .value_utils import is_blank_value

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
