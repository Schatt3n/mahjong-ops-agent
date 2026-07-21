"""Game, room, invite, and outbound-draft persistence contracts."""

from __future__ import annotations

from typing import Any, Protocol

from ..models import (
    Game,
    InviteDraft,
    OutboundMessageDraft,
    RoomReservation,
    StateTransition,
)


class GameStore(Protocol):
    """Persistence operations owned by the game aggregate."""

    @property
    def games(self) -> dict[str, Game]: ...

    @property
    def invite_drafts(self) -> dict[str, InviteDraft]: ...

    @property
    def outbound_message_drafts(self) -> dict[str, OutboundMessageDraft]: ...

    @property
    def room_ids(self) -> list[str]: ...

    @property
    def room_reservations(self) -> dict[str, RoomReservation]: ...

    @property
    def transitions(self) -> list[StateTransition]: ...

    def configure_rooms(self, room_ids: list[str]) -> None: ...

    def search_room_availability(self, *, start_at: Any, end_at: Any) -> dict[str, Any]: ...

    def reserve_room(
        self,
        *,
        conversation_id: str,
        game_id: str | None,
        start_at: Any,
        end_at: Any,
        trace_id: str,
        preferred_room_id: str | None = None,
    ) -> tuple[RoomReservation, StateTransition]: ...

    def active_games(self, conversation_id: str | None = None) -> list[Game]: ...

    def search_current_games(
        self,
        requirement: dict[str, Any],
        *,
        sender_id: str | None = None,
        limit: int = 8,
    ) -> list[dict[str, Any]]: ...

    def active_game_for_customer(self, customer_id: str) -> Game | None: ...

    def create_game(
        self,
        *,
        conversation_id: str,
        organizer_id: str,
        organizer_name: str,
        requirement: dict[str, Any],
        known_players: list[dict[str, Any]],
        trace_id: str,
    ) -> tuple[Game, StateTransition]: ...

    def update_game_requirement(
        self,
        *,
        game_id: str,
        requirement_patch: dict[str, Any],
        trace_id: str,
    ) -> tuple[Game, list[StateTransition]]: ...

    def create_invite_drafts(
        self,
        *,
        game_id: str,
        invitations: list[dict[str, Any]],
        trace_id: str,
    ) -> tuple[list[InviteDraft], list[StateTransition]]: ...

    def create_outbound_message_drafts(
        self,
        *,
        conversation_id: str,
        messages: list[dict[str, Any]],
        trace_id: str,
    ) -> tuple[list[OutboundMessageDraft], list[StateTransition]]: ...

    def update_invite_delivery_status(
        self,
        *,
        draft_id: str,
        status: str,
        trace_id: str,
    ) -> tuple[InviteDraft, StateTransition]: ...

    def record_candidate_reply(
        self,
        *,
        game_id: str,
        customer_id: str,
        display_name: str,
        status: str,
        seat_count: int,
        trace_id: str,
    ) -> tuple[Game, list[StateTransition]]: ...

    def join_game(
        self,
        *,
        game_id: str,
        customer_id: str,
        display_name: str,
        seat_count: int = 1,
        trace_id: str,
    ) -> tuple[Game, list[StateTransition]]: ...

    def update_game_status(
        self,
        *,
        game_id: str,
        status: str,
        reason: str,
        trace_id: str,
    ) -> tuple[Game, StateTransition]: ...

    def require_game(self, game_id: str) -> Game: ...

