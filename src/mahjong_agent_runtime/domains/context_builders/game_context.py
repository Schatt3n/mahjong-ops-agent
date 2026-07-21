"""Build the game-specific slice of an Agent prompt."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...models import Game
from ...stores import AgentStore
from ..model_context import game_for_model_context


@dataclass(slots=True)
class GameContextBundle:
    """Game facts related to one sender, separated from global store state."""

    games: list[Game]
    model_contexts: list[dict[str, Any]]
    visible_summaries: list[dict[str, Any]]
    sender_memberships: list[dict[str, Any]]
    active_parties: list[dict[str, Any]]


def build_game_context(
    store: AgentStore,
    *,
    conversation_id: str,
    sender_id: str,
) -> GameContextBundle:
    """Select only games connected to the current sender or conversation."""

    all_active_games = store.active_games()
    related_game_ids = {
        game.game_id
        for game in all_active_games
        if game.conversation_id == conversation_id
        or game.organizer_id == sender_id
        or any(participant.customer_id == sender_id for participant in game.participants)
    }
    related_game_ids.update(
        draft.game_id
        for draft in store.invite_drafts.values()
        if draft.customer_id == sender_id
    )
    games = [game for game in all_active_games if game.game_id in related_game_ids]
    model_contexts = [compact_game(game_for_model_context(game, store.customers)) for game in games]
    visible_summaries = [active_game_visible_summary(game) for game in games]
    sender_memberships = sender_active_game_memberships(games, sender_id)
    active_parties = [
        {
            "game_id": game_context["game_id"],
            "seat_summary": dict(game_context.get("seat_summary") or {}),
        }
        for game_context in model_contexts
    ]
    return GameContextBundle(
        games=games,
        model_contexts=model_contexts,
        visible_summaries=visible_summaries,
        sender_memberships=sender_memberships,
        active_parties=active_parties,
    )


def sender_active_game_memberships(games: list[Game], sender_id: str) -> list[dict[str, Any]]:
    memberships: list[dict[str, Any]] = []
    for game in games:
        for participant in game.participants:
            if participant.customer_id != sender_id:
                continue
            memberships.append(
                {
                    "game_id": game.game_id,
                    "participant_status": participant.status,
                    "seat_count": participant.seat_count,
                    "participation_already_recorded": participant.status
                    in {"joined", "confirmed", "accepted", "arrived"},
                    "write_instruction": (
                        "Do not call record_candidate_reply with the same participation meaning unless the current "
                        "message explicitly changes status or seat_count."
                    ),
                }
            )
    return memberships


def compact_game(game: Any) -> dict[str, Any]:
    if not isinstance(game, dict):
        return {}
    return {
        "game_id": game.get("game_id"),
        "conversation_id": game.get("conversation_id"),
        "organizer_id": game.get("organizer_id"),
        "organizer_name": game.get("organizer_name"),
        "status": game.get("status"),
        "requirement": compact_requirement(game.get("requirement")),
        "seat_summary": game.get("seat_summary"),
        "remaining_seats": game.get("remaining_seats"),
        "planned_start_at": game.get("planned_start_at"),
        "planned_end_at": game.get("planned_end_at"),
        "expires_at": game.get("expires_at"),
        "participants": [
            {
                "customer_id": item.get("customer_id"),
                "display_name": item.get("display_name"),
                "status": item.get("status"),
                "seat_count": item.get("seat_count"),
                "source": item.get("source"),
            }
            for item in list(game.get("participants") or [])[:8]
            if isinstance(item, dict)
        ],
        "parties": [compact_party(item) for item in list(game.get("parties") or [])[:8]],
    }


def compact_party(party: Any) -> dict[str, Any]:
    if not isinstance(party, dict):
        return {}
    return {
        "party_id": party.get("party_id"),
        "contact_id": party.get("contact_id"),
        "contact_name": party.get("contact_name"),
        "seat_count": party.get("seat_count"),
        "anonymous_seat_count": party.get("anonymous_seat_count"),
        "status": party.get("status"),
        "source": party.get("source"),
    }


def compact_requirement(requirement: Any) -> dict[str, Any]:
    """Remove duplicate party structures while preserving decision facts."""

    if not isinstance(requirement, dict):
        return {}
    structural_duplicates = {
        "requesting_party",
        "seat_claims",
        "parties",
        "participants",
        "known_players",
    }
    return {
        key: compact_context_value(value)
        for key, value in requirement.items()
        if key not in structural_duplicates
    }


def compact_context_value(value: Any, *, max_list_items: int = 12, max_string_chars: int = 1000) -> Any:
    """Bound nested payloads while retaining deterministic JSON shapes."""

    if isinstance(value, dict):
        return {
            str(key): compact_context_value(
                item,
                max_list_items=max_list_items,
                max_string_chars=max_string_chars,
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            compact_context_value(
                item,
                max_list_items=max_list_items,
                max_string_chars=max_string_chars,
            )
            for item in value[:max_list_items]
        ]
    if isinstance(value, str) and len(value) > max_string_chars:
        return value[:max_string_chars] + "...[truncated]"
    return value


def active_game_visible_summary(game: Game) -> dict[str, Any]:
    """Build the only game facts that may be copied into a customer reply."""

    requirement = dict(game.requirement or {})
    public_requirement_keys = (
        "user_visible_summary",
        "game_type",
        "stake",
        "base_stake",
        "cap_score",
        "stake_label",
        "smoke_preference",
        "start_time_kind",
        "start_time",
        "duration_kind",
        "duration_hours",
        "known_player_count",
        "needed_seats",
    )
    visible_summary = str(requirement.get("user_visible_summary") or "")
    return {
        "game_id": game.game_id,
        "status": game.status.value,
        "user_visible_summary": visible_summary,
        "status_query_reply_contract": {
            "when_to_use": "用户问当前局况、现在几个人、还差几人、有没有进展时使用。",
            "preferred_reply_source": "user_visible_summary",
            "preferred_reply_text": visible_summary,
            "preservation_mode": "all_decision_anchors",
            "required_semantic_source": "preferred_reply_text",
            "invalid_rewrite": "只保留人数或缺口，丢失 preferred_reply_text 中的时间、公开昵称、局名或缺口短码。",
            "rule": "如果 user_visible_summary 非空，优先原样使用或轻微口语化；不要只根据 seat_summary 重新概括而丢掉时间、公开昵称、局名或缺口短码。",
        },
        "seat_summary": game.seat_summary(),
        "public_requirement": {
            key: requirement.get(key)
            for key in public_requirement_keys
            if requirement.get(key) is not None
        },
    }


__all__ = [
    "GameContextBundle",
    "active_game_visible_summary",
    "build_game_context",
    "compact_context_value",
    "compact_game",
    "compact_party",
    "compact_requirement",
    "sender_active_game_memberships",
]
