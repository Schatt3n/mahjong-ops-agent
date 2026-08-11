"""实现群聊场景中的models能力。

"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any, Literal

from ..models import now


@dataclass(slots=True)
class GroupMessage:
    """一条公房消息，尚未路由到业务会话。"""

    room_id: str
    conversation_id: str
    sender_external_id: str
    sender_name: str
    text: str
    message_id: str
    sent_at: datetime = field(default_factory=now)
    quoted_message_id: str | None = None
    channel: str = "wechaty"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换为只包含可序列化字段的字典表示。"""
        payload = asdict(self)
        payload["sent_at"] = self.sent_at.isoformat()
        return payload


@dataclass(slots=True)
class ReplyConstraints:
    """后端发出的输出边界，不包含任何业务结论。"""

    max_length: int
    no_private_info: bool = True

    def to_dict(self) -> dict[str, Any]:
        """转换为只包含可序列化字段的字典表示。"""
        return asdict(self)


@dataclass(slots=True)
class ChannelIdentity:
    """将渠道内本地账号映射到稳定的客户聚合。"""

    channel: str
    external_user_id: str
    customer_id: str
    public_name: str
    private_conversation_id: str
    can_private_message: bool = False
    is_friend: bool = False
    created_at: datetime = field(default_factory=now)
    updated_at: datetime = field(default_factory=now)

    @property
    def identity_key(self) -> str:
        return f"{self.channel}\x1f{self.external_user_id}"

    def to_dict(self) -> dict[str, Any]:
        """转换为只包含可序列化字段的字典表示。"""
        payload = asdict(self)
        payload["created_at"] = self.created_at.isoformat()
        payload["updated_at"] = self.updated_at.isoformat()
        return payload


@dataclass(slots=True)
class GroupRoomPolicy:
    """描述群聊房间策略采用的可配置策略参数。

"""

    room_id: str
    channel: str = "wechaty"
    managed: bool = True
    board_enabled: bool = True
    outbound_enabled: bool = True
    merge_window_seconds: int = 30
    updated_at: datetime = field(default_factory=now)

    def to_dict(self) -> dict[str, Any]:
        """转换为只包含可序列化字段的字典表示。"""
        payload = asdict(self)
        payload["updated_at"] = self.updated_at.isoformat()
        return payload


@dataclass(slots=True)
class GameConversationLink:
    """将牌局聚合与房间/私聊会话关联，但不共享原始回合。"""

    link_id: str
    game_id: str
    conversation_id: str
    room_id: str
    customer_id: str | None
    link_type: str
    created_at: datetime = field(default_factory=now)

    def to_dict(self) -> dict[str, Any]:
        """转换为只包含可序列化字段的字典表示。"""
        payload = asdict(self)
        payload["created_at"] = self.created_at.isoformat()
        return payload


@dataclass(slots=True)
class BoardItem:
    """从房主最新发布的公告板解析出的单个实时条目。"""

    id: str
    display_no: int
    game_type: str
    participant_code: str
    time: str | None
    smoking: str | None
    stakes: str
    special_rules: str | None
    ruleset: str | None = None
    end_time: str | None = None
    duration_hours: float | None = None
    rule_code: str | None = None
    temporary_constraints: list[str] = field(default_factory=list)
    source_message_id: str | None = None
    status: Literal["waiting", "full", "playing"] = "waiting"
    slots_total: int = 4
    slots_filled: int = 0
    participants: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """转换为只包含可序列化字段的字典表示。"""
        return asdict(self)

    @property
    def current_players(self) -> int:
        return self.slots_filled

    @property
    def missing_players(self) -> int:
        return max(0, self.slots_total - self.slots_filled)


@dataclass(slots=True)
class BoardState:
    """某个房间最新由房主编写的公告板，独立于会话持久化。"""

    room_id: str
    items: list[BoardItem]
    source_message_id: str
    last_published_at: datetime

    def to_dict(self) -> dict[str, Any]:
        """转换为只包含可序列化字段的字典表示。"""
        return {
            "room_id": self.room_id,
            "items": [item.to_dict() for item in self.items],
            "source_message_id": self.source_message_id,
            "last_published_at": self.last_published_at.isoformat(),
        }


@dataclass(slots=True)
class BoardSnapshotItem:
    """属于一个对外公告板版本的不可变数字到牌局映射。"""

    item_no: int
    game_id: str
    rendered_text: str

    def to_dict(self) -> dict[str, Any]:
        """转换为只包含可序列化字段的字典表示。"""
        return asdict(self)


@dataclass(slots=True)
class BoardSnapshot:
    """从已发布公告板版本到牌局ID的不可变映射。"""

    snapshot_id: str
    room_id: str
    conversation_id: str
    external_message_id: str
    rendered_text: str
    items: list[BoardSnapshotItem]
    created_at: datetime = field(default_factory=now)

    def to_dict(self) -> dict[str, Any]:
        """转换为只包含可序列化字段的字典表示。"""
        return {
            "snapshot_id": self.snapshot_id,
            "room_id": self.room_id,
            "conversation_id": self.conversation_id,
            "external_message_id": self.external_message_id,
            "rendered_text": self.rendered_text,
            "items": [item.to_dict() for item in self.items],
            "created_at": self.created_at.isoformat(),
        }


@dataclass(slots=True)
class GameClaim:
    """记录群成员对某个看板席位的认领及其幂等来源。"""

    claim_id: str
    room_id: str
    game_id: str
    customer_id: str
    source_conversation_id: str
    source_message_id: str
    status: str = "claimed"
    created_at: datetime = field(default_factory=now)

    def to_dict(self) -> dict[str, Any]:
        """转换为只包含可序列化字段的字典表示。"""
        payload = asdict(self)
        payload["created_at"] = self.created_at.isoformat()
        return payload


@dataclass(slots=True)
class ChannelSwitch:
    """短生命周期指针，证明公开请求已在私聊中继续执行。"""

    switch_id: str
    room_id: str
    customer_id: str
    source_conversation_id: str
    source_message_id: str
    private_conversation_id: str
    trigger_summary: str
    status: str = "active"
    created_at: datetime = field(default_factory=now)
    expires_at: datetime = field(default_factory=lambda: now() + timedelta(minutes=10))

    def to_dict(self) -> dict[str, Any]:
        """转换为只包含可序列化字段的字典表示。"""
        payload = asdict(self)
        payload["created_at"] = self.created_at.isoformat()
        payload["expires_at"] = self.expires_at.isoformat()
        return payload


@dataclass(slots=True)
class PrivateSwitchContext:
    """从群聊切换到私聊时允许携带的最小业务上下文。"""

    private_conversation_id: str
    trigger_summary: str
    customer_profile: dict[str, Any]
    user_original_text: str
    parsed_need: dict[str, Any]
    missing_info: list[str]
    reply_guidelines: str

    def to_dict(self) -> dict[str, Any]:
        """转换为只包含可序列化字段的字典表示。"""
        return asdict(self)


@dataclass(slots=True)
class L1Result:
    """群消息确定性规则层的解析动作与可选业务字段。"""

    action: str
    parsed_game: dict[str, Any] | None = None
    item_no: int | None = None
    needs_ack: bool = False


@dataclass(slots=True)
class RoutingDecision:
    """群消息第二层路由选出的目标通道、确认语和回复约束。"""

    action: str
    channel: str | None = None
    reply_constraints: ReplyConstraints | None = None
    group_ack: str = ""
    switch_context: PrivateSwitchContext | None = None


@dataclass(slots=True)
class ClaimResult:
    """席位认领成功、拒绝或幂等命中的结构化结果。"""

    status: str
    game_id: str | None = None
    reason: str = ""
    deduplicated: bool = False


@dataclass(slots=True)
class GroupHandleResult:
    """群消息入口执行后的动作、回复及关联业务对象。"""

    action: str
    game_id: str | None = None
    reply: str = ""
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ChatSession:
    """有界的一次公房交互单元，而非整段房间记录。"""

    id: str
    room_id: str
    status: Literal["active", "resolved", "merged", "expired"] = "active"
    topic_type: Literal["claim", "formation", "query", "board_update", "unknown"] = "unknown"
    messages: list[GroupMessage] = field(default_factory=list)
    participants: set[str] = field(default_factory=set)
    extracted_state: dict[str, Any] = field(default_factory=dict)
    topic: str = ""
    related_board_item_id: str | None = None
    merged_into: str | None = None
    created_at: datetime = field(default_factory=now)
    last_activity_at: datetime = field(default_factory=now)

    def to_context(self) -> dict[str, Any]:
        return {
            "session_id": self.id,
            "room_id": self.room_id,
            "status": self.status,
            "topic_type": self.topic_type,
            "topic": self.topic,
            "participants": sorted(self.participants),
            "extracted_state": dict(self.extracted_state),
            "related_board_item_id": self.related_board_item_id,
            "messages": [item.to_dict() for item in self.messages],
            "created_at": self.created_at.isoformat(),
            "last_activity_at": self.last_activity_at.isoformat(),
        }


@dataclass(slots=True)
class SessionClassification:
    """由专用群会话模型任务返回的严格语义结果。"""

    intent: Literal["claim", "new_demand", "query", "thread_update", "chitchat"]
    extracted_features: dict[str, Any]
    matched_board_no: int | None
    confidence: float
    reasoning: str
    response: str | None
    channel_action: Literal["private_switch", "group_reply", "ignore"]

    def to_dict(self) -> dict[str, Any]:
        """转换为只包含可序列化字段的字典表示。"""
        return asdict(self)


@dataclass(slots=True)
class GroupSessionOutcome:
    """对单个单元进行缓冲、解析、分类或路由后可观测到的结果。"""

    action: str
    session_id: str | None = None
    classification: SessionClassification | None = None
    detail: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "BoardItem",
    "BoardSnapshotItem",
    "BoardSnapshot",
    "BoardState",
    "ChatSession",
    "ChannelIdentity",
    "ChannelSwitch",
    "ClaimResult",
    "GameClaim",
    "GameConversationLink",
    "GroupHandleResult",
    "GroupMessage",
    "GroupSessionOutcome",
    "GroupRoomPolicy",
    "L1Result",
    "PrivateSwitchContext",
    "ReplyConstraints",
    "RoutingDecision",
    "SessionClassification",
]
