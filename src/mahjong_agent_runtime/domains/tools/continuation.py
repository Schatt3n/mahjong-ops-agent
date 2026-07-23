"""Domain-owned continuation contracts for multi-step tool objectives."""

from __future__ import annotations

from typing import Any

from ...models import Game
from ...stores import AgentStore
from ..customer_domain import current_game_decision_required_fields
from ..game_domain import normalize_requirement


def create_game_continuation(game: Game) -> dict[str, Any]:
    """Describe whether creating a game finished the current turn's objective.

    A tool knows the durable state it created, while the main loop remains
    domain-neutral.  The returned contract lets the model choose the next tool
    but prevents it from declaring success while an immediately recruitable
    game still has empty seats.
    """

    remaining = game.remaining_seats()
    if remaining <= 0:
        return _continuation(
            can_stop=True,
            obligation_id=f"game:{game.game_id}:recruitment",
            reason="The newly created game is already full.",
            game=game,
        )
    if game.recruitment_status.value == "scheduled":
        return _continuation(
            can_stop=True,
            obligation_id=f"game:{game.game_id}:scheduled_recruitment",
            reason="Recruitment is durably delegated to the scheduled trigger window.",
            game=game,
        )
    return _continuation(
        can_stop=False,
        obligation_id=f"game:{game.game_id}:candidate_discovery",
        reason="The game is immediately recruitable and still has open seats.",
        game=game,
        pending_capabilities=["discover_candidates"],
        suggested_tools=["search_customers"],
        instruction=(
            "Do not finish with only an acknowledgement. Continue the existing plan by discovering candidates "
            "for this game; preserve the authoritative game_id, requirement, seat summary, and excluded members."
        ),
    )


def customer_search_continuation(
    store: AgentStore,
    *,
    conversation_id: str,
    sender_id: str,
    requirement: dict[str, Any],
    candidates: list[dict[str, Any]],
    game: Game | None = None,
) -> dict[str, Any] | None:
    """Return follow-up work only when a candidate search belongs to an active game."""

    game = game or find_recruitment_game(
        store,
        conversation_id=conversation_id,
        sender_id=sender_id,
        requirement=requirement,
    )
    if game is None:
        return None
    if not candidates:
        return _continuation(
            can_stop=True,
            allowed_terminal_statuses=["completed", "waiting_user"],
            obligation_id=f"game:{game.game_id}:candidate_outreach",
            reason=(
                "The active game still has open seats, but the current candidate search is exhausted; "
                "the game remains durable for later inbound or scheduled matching."
            ),
            game=game,
            pending_capabilities=[],
            suggested_tools=[],
            instruction=(
                "Do not claim people were contacted. You may stop this synchronous turn with a short acknowledgement "
                "that the active game will keep being matched; no invitation draft can be created without candidates."
            ),
        )
    return _continuation(
        can_stop=False,
        obligation_id=f"game:{game.game_id}:candidate_outreach",
        reason="Candidates were found for an active game with open seats, but no invitation draft exists yet.",
        game=game,
        pending_capabilities=["prepare_candidate_outreach"],
        suggested_tools=["create_invite_drafts"],
        instruction=(
            "Continue with candidate outreach for this game. Use only returned candidates, create reviewed invite "
            "drafts for the needed seats, and do not expose internal candidate details to the requester."
        ),
    )


def invite_draft_continuation(game_id: str, draft_count: int) -> dict[str, Any]:
    """Mark the synchronous recruitment phase complete after drafts are durable."""

    return {
        "version": 1,
        "can_stop": True,
        "allowed_terminal_statuses": ["completed", "waiting_user"],
        "obligation_id": f"game:{game_id}:candidate_outreach",
        "reason": f"Candidate outreach is durably prepared with {max(0, int(draft_count))} draft(s).",
        "authoritative_facts": {"game_id": game_id, "draft_count": max(0, int(draft_count))},
        "pending_capabilities": [],
        "suggested_tools": [],
        "instruction": "The synchronous tool objective may now stop with a short customer-facing acknowledgement.",
    }


def _continuation(
    *,
    can_stop: bool,
    obligation_id: str,
    reason: str,
    game: Game,
    allowed_terminal_statuses: list[str] | None = None,
    pending_capabilities: list[str] | None = None,
    suggested_tools: list[str] | None = None,
    instruction: str = "",
) -> dict[str, Any]:
    seat_summary = game.seat_summary()
    excluded = sorted(
        {
            str(item.get("contact_id") or "")
            for item in game.seat_claims()
            if str(item.get("contact_id") or "").strip()
        }
    )
    return {
        "version": 1,
        "can_stop": bool(can_stop),
        "allowed_terminal_statuses": list(allowed_terminal_statuses or (["completed", "waiting_user"] if can_stop else [])),
        "obligation_id": obligation_id,
        "reason": reason,
        "authoritative_facts": {
            "game_id": game.game_id,
            "requirement": dict(game.requirement),
            "seat_summary": seat_summary,
            "exclude_customer_ids": excluded,
            "recruitment_status": game.recruitment_status.value,
        },
        "pending_capabilities": list(pending_capabilities or []),
        "suggested_tools": list(suggested_tools or []),
        "instruction": instruction,
    }


def bind_candidate_search_requirement(
    store: AgentStore,
    *,
    conversation_id: str,
    sender_id: str,
    requirement: dict[str, Any],
    game_id: str | None = None,
) -> tuple[dict[str, Any], Game | None, str | None]:
    """Bind candidate discovery to durable Game facts when one already exists.

    The model may propose preferences used for ranking, but it must not become
    a second source of truth for occupancy after ``create_game``.  An explicit
    ``game_id`` is preferred.  For resilience, a single compatible active game
    in the authenticated conversation can be inferred.  Multiple compatible
    games are rejected as ambiguous instead of silently searching for the
    wrong table.
    """

    proposed = normalize_requirement(requirement)
    active_games = [
        game
        for game in store.active_games(conversation_id)
        if game.remaining_seats() > 0 and game.recruitment_status.value != "scheduled"
    ]
    requested_game_id = str(game_id or "").strip()
    if requested_game_id:
        game = next((item for item in active_games if item.game_id == requested_game_id), None)
        if game is None:
            return proposed, None, (
                "candidate search game is not an active recruitable game in this conversation: "
                f"{requested_game_id}"
            )
        conflicts = current_game_decision_required_fields(proposed, game.requirement)
        if conflicts:
            return proposed, None, (
                "candidate search requirement conflicts with the authoritative game on: "
                + ",".join(conflicts)
            )
    else:
        compatible = [
            game
            for game in active_games
            if not current_game_decision_required_fields(proposed, game.requirement)
        ]
        if not compatible:
            return proposed, None, None
        if len(compatible) > 1:
            return proposed, None, (
                "candidate search is ambiguous across active games; retry with the game_id returned by create_game"
            )
        game = compatible[0]

    seat_summary = game.seat_summary()
    bound = normalize_requirement({**proposed, **dict(game.requirement)})
    bound.update(
        {
            "known_player_count": seat_summary["claimed_seats"],
            "needed_seats": seat_summary["remaining_seats"],
            "remaining_seats": seat_summary["remaining_seats"],
            "seats_total": seat_summary["seats_total"],
            "seat_format": f"{seat_summary['claimed_seats']}7{seat_summary['remaining_seats']}",
            "seat_claims": game.seat_claims(),
        }
    )
    if game.seat_claims():
        bound["requesting_party"] = game.seat_claims()[0]
    return normalize_requirement(bound), game, None


def find_recruitment_game(
    store: AgentStore,
    *,
    conversation_id: str,
    sender_id: str,
    requirement: dict[str, Any],
) -> Game | None:
    games = [
        game
        for game in store.active_games(conversation_id)
        if game.remaining_seats() > 0
        and game.recruitment_status.value != "scheduled"
        and not current_game_decision_required_fields(requirement, game.requirement)
    ]
    if not games:
        return None
    games.sort(
        key=lambda game: (
            game.organizer_id == sender_id,
            game.updated_at,
            game.created_at,
            game.game_id,
        ),
        reverse=True,
    )
    return games[0]


__all__ = [
    "bind_candidate_search_requirement",
    "create_game_continuation",
    "customer_search_continuation",
    "find_recruitment_game",
    "invite_draft_continuation",
]
