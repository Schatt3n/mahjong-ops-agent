"""Persistence contract for the public-room board domain."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Protocol

from ..models import Game, ScheduledAgentTask, StateTransition

if TYPE_CHECKING:
    from ..group_chat.models import (
        BoardSnapshot,
        BoardState,
        ChannelIdentity,
        ChannelSwitch,
        GameClaim,
        GameConversationLink,
        GroupRoomPolicy,
    )


class GroupChatStore(Protocol):
    """State needed to route room messages without sharing room transcripts."""

    def upsert_channel_identity(self, identity: ChannelIdentity) -> ChannelIdentity: ...

    def get_channel_identity(self, channel: str, external_user_id: str) -> ChannelIdentity | None: ...

    def get_channel_identity_for_customer(self, customer_id: str, channel: str = "wechaty") -> ChannelIdentity | None: ...

    def upsert_group_room_policy(self, policy: GroupRoomPolicy) -> GroupRoomPolicy: ...

    def get_group_room_policy(self, room_id: str) -> GroupRoomPolicy | None: ...

    def upsert_group_board_state(self, board_state: BoardState) -> BoardState: ...

    def get_group_board_state(self, room_id: str) -> BoardState | None: ...

    def link_game_conversation(self, link: GameConversationLink) -> GameConversationLink: ...

    def game_conversation_links(self, game_id: str | None = None, room_id: str | None = None) -> list[GameConversationLink]: ...

    def get_board_eligible_games(self, room_id: str) -> list[Game]: ...

    def save_board_snapshot(self, snapshot: BoardSnapshot) -> BoardSnapshot: ...

    def get_board_snapshot(self, snapshot_id: str) -> BoardSnapshot | None: ...

    def get_latest_board_snapshot(self, room_id: str) -> BoardSnapshot | None: ...

    def get_board_snapshot_by_message_id(self, room_id: str, external_message_id: str) -> BoardSnapshot | None: ...

    def get_game_claim_by_source(self, source_conversation_id: str, source_message_id: str) -> GameClaim | None: ...

    def atomic_claim_seat(
        self,
        *,
        room_id: str,
        game_id: str,
        customer_id: str,
        display_name: str,
        source_conversation_id: str,
        source_message_id: str,
        trace_id: str,
    ) -> tuple[GameClaim, Game, list[StateTransition], bool]: ...

    def record_channel_switch(self, switch: ChannelSwitch) -> ChannelSwitch: ...

    def get_recent_active_channel_switch(
        self,
        customer_id: str,
        *,
        room_id: str | None = None,
        at: datetime | None = None,
    ) -> ChannelSwitch | None: ...

    def ensure_group_board_publish_task(
        self,
        *,
        room_id: str,
        due_at: datetime,
        trace_id: str,
        urgent: bool = False,
    ) -> tuple[ScheduledAgentTask, StateTransition | None]: ...


__all__ = ["GroupChatStore"]
