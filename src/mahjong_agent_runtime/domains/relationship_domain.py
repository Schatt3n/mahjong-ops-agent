"""Domain rules for relationship domain."""

from __future__ import annotations

from typing import Any
from ..models import (
    CustomerProfile,
    Game,
    GameParticipant,
    TaskMemory,
)
from .model_context import customer_visible_name
from .value_utils import is_blank_value

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
