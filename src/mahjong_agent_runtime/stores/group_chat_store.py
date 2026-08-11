"""定义群聊聊天存储相关的存储接口，隔离领域层与具体持久化实现。

"""

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
    """定义群聊聊天存储需要遵守的抽象接口契约。

"""

    def upsert_channel_identity(self, identity: ChannelIdentity) -> ChannelIdentity:
        """新增或更新通道身份映射，使重复写入收敛到同一条记录。"""
        ...

    def get_channel_identity(self, channel: str, external_user_id: str) -> ChannelIdentity | None:
        """读取通道身份映射；返回值只反映当前存储或运行快照。"""
        ...

    def get_channel_identity_for_customer(self, customer_id: str, channel: str = "wechaty") -> ChannelIdentity | None:
        """读取通道身份映射for客户；返回值只反映当前存储或运行快照。"""
        ...

    def upsert_group_room_policy(self, policy: GroupRoomPolicy) -> GroupRoomPolicy:
        """新增或更新群聊房间策略，使重复写入收敛到同一条记录。"""
        ...

    def get_group_room_policy(self, room_id: str) -> GroupRoomPolicy | None:
        """读取群聊房间策略；返回值只反映当前存储或运行快照。"""
        ...

    def upsert_group_board_state(self, board_state: BoardState) -> BoardState:
        """新增或更新群聊看板状态，使重复写入收敛到同一条记录。"""
        ...

    def get_group_board_state(self, room_id: str) -> BoardState | None:
        """读取群聊看板状态；返回值只反映当前存储或运行快照。"""
        ...

    def link_game_conversation(self, link: GameConversationLink) -> GameConversationLink:
        ...

    def game_conversation_links(self, game_id: str | None = None, room_id: str | None = None) -> list[GameConversationLink]:
        ...

    def get_board_eligible_games(self, room_id: str) -> list[Game]:
        """读取看板eligiblegames；返回值只反映当前存储或运行快照。"""
        ...

    def save_board_snapshot(self, snapshot: BoardSnapshot) -> BoardSnapshot:
        """保存看板快照，确保后续读取能够得到一致快照。"""
        ...

    def get_board_snapshot(self, snapshot_id: str) -> BoardSnapshot | None:
        """读取看板快照；返回值只反映当前存储或运行快照。"""
        ...

    def get_latest_board_snapshot(self, room_id: str) -> BoardSnapshot | None:
        """读取latest看板快照；返回值只反映当前存储或运行快照。"""
        ...

    def get_board_snapshot_by_message_id(self, room_id: str, external_message_id: str) -> BoardSnapshot | None:
        """读取看板快照by消息id；返回值只反映当前存储或运行快照。"""
        ...

    def get_game_claim_by_source(self, source_conversation_id: str, source_message_id: str) -> GameClaim | None:
        """读取牌局claimbysource；返回值只反映当前存储或运行快照。"""
        ...

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
    ) -> tuple[GameClaim, Game, list[StateTransition], bool]:
        ...

    def record_channel_switch(self, switch: ChannelSwitch) -> ChannelSwitch:
        """记录通道通道切换，形成后续决策和审计所需的持久证据。"""
        ...

    def get_recent_active_channel_switch(
        self,
        customer_id: str,
        *,
        room_id: str | None = None,
        at: datetime | None = None,
    ) -> ChannelSwitch | None:
        """读取recentactive通道通道切换；返回值只反映当前存储或运行快照。"""
        ...

    def ensure_group_board_publish_task(
        self,
        *,
        room_id: str,
        due_at: datetime,
        trace_id: str,
        urgent: bool = False,
    ) -> tuple[ScheduledAgentTask, StateTransition | None]:
        """确保群聊看板publish任务满足后续步骤依赖的不变量。"""
        ...


__all__ = ["GroupChatStore"]
