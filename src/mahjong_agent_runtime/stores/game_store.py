"""定义牌局存储相关的存储接口，隔离领域层与具体持久化实现。

"""

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
    """定义牌局存储需要遵守的抽象接口契约。

"""

    @property
    def games(self) -> dict[str, Game]:
        ...

    @property
    def invite_drafts(self) -> dict[str, InviteDraft]:
        ...

    @property
    def outbound_message_drafts(self) -> dict[str, OutboundMessageDraft]:
        ...

    @property
    def room_ids(self) -> list[str]:
        ...

    @property
    def room_reservations(self) -> dict[str, RoomReservation]:
        ...

    @property
    def transitions(self) -> list[StateTransition]:
        ...

    def configure_rooms(self, room_ids: list[str]) -> None:
        ...

    def search_room_availability(self, *, start_at: Any, end_at: Any) -> dict[str, Any]:
        """查询房间可用性；只返回满足当前作用域和约束的结果。"""
        ...

    def reserve_room(
        self,
        *,
        conversation_id: str,
        game_id: str | None,
        start_at: Any,
        end_at: Any,
        trace_id: str,
        preferred_room_id: str | None = None,
    ) -> tuple[RoomReservation, StateTransition]:
        ...

    def active_games(self, conversation_id: str | None = None) -> list[Game]:
        ...

    def search_current_games(
        self,
        requirement: dict[str, Any],
        *,
        sender_id: str | None = None,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        """查询currentgames；只返回满足当前作用域和约束的结果。"""
        ...

    def active_game_for_customer(self, customer_id: str) -> Game | None:
        ...

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
        """创建牌局，并返回持久化后的权威结果。"""
        ...

    def update_game_requirement(
        self,
        *,
        game_id: str,
        requirement_patch: dict[str, Any],
        trace_id: str,
    ) -> tuple[Game, list[StateTransition]]:
        """更新牌局组局条件，同时维护相关状态不变量。"""
        ...

    def create_invite_drafts(
        self,
        *,
        game_id: str,
        invitations: list[dict[str, Any]],
        trace_id: str,
    ) -> tuple[list[InviteDraft], list[StateTransition]]:
        """创建邀约drafts，并返回持久化后的权威结果。"""
        ...

    def create_outbound_message_drafts(
        self,
        *,
        conversation_id: str,
        messages: list[dict[str, Any]],
        trace_id: str,
    ) -> tuple[list[OutboundMessageDraft], list[StateTransition]]:
        """创建外发消息drafts，并返回持久化后的权威结果。"""
        ...

    def update_invite_delivery_status(
        self,
        *,
        draft_id: str,
        status: str,
        trace_id: str,
    ) -> tuple[InviteDraft, StateTransition]:
        """更新邀约delivery状态，同时维护相关状态不变量。"""
        ...

    def record_candidate_reply(
        self,
        *,
        game_id: str,
        customer_id: str,
        display_name: str,
        status: str,
        seat_count: int,
        trace_id: str,
    ) -> tuple[Game, list[StateTransition]]:
        """记录候选人reply，形成后续决策和审计所需的持久证据。"""
        ...

    def join_game(
        self,
        *,
        game_id: str,
        customer_id: str,
        display_name: str,
        seat_count: int = 1,
        trace_id: str,
    ) -> tuple[Game, list[StateTransition]]:
        ...

    def update_game_status(
        self,
        *,
        game_id: str,
        status: str,
        reason: str,
        trace_id: str,
    ) -> tuple[Game, StateTransition]:
        """更新牌局状态，同时维护相关状态不变量。"""
        ...

    def require_game(self, game_id: str) -> Game:
        """读取并确认牌局存在；缺失时立即以领域异常终止。"""
        ...

