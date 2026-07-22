"""Domain records exchanged by the group-chat routing layer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any, Literal

from ..models import now


@dataclass(slots=True)
class GroupMessage:
    """One public-room message before it is routed to a business conversation."""

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
        payload = asdict(self)
        payload["sent_at"] = self.sent_at.isoformat()
        return payload


@dataclass(slots=True)
class ReplyConstraints:
    """Backend-issued output boundary; it contains no business conclusion."""

    max_length: int
    no_private_info: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ChannelIdentity:
    """Map a channel-local account to the stable customer aggregate."""

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
        payload = asdict(self)
        payload["created_at"] = self.created_at.isoformat()
        payload["updated_at"] = self.updated_at.isoformat()
        return payload


@dataclass(slots=True)
class GroupRoomPolicy:
    """Explicit opt-in policy for a room managed as an operational board."""

    room_id: str
    channel: str = "wechaty"
    managed: bool = True
    board_enabled: bool = True
    outbound_enabled: bool = True
    merge_window_seconds: int = 30
    updated_at: datetime = field(default_factory=now)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["updated_at"] = self.updated_at.isoformat()
        return payload


@dataclass(slots=True)
class GameConversationLink:
    """Link one game aggregate to a room/private conversation without sharing raw turns."""

    link_id: str
    game_id: str
    conversation_id: str
    room_id: str
    customer_id: str | None
    link_type: str
    created_at: datetime = field(default_factory=now)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["created_at"] = self.created_at.isoformat()
        return payload


@dataclass(slots=True)
class BoardItem:
    """One live item parsed from the room owner's latest published board."""

    id: str
    display_no: int
    game_type: str
    table_id: str
    time: str | None
    smoking: str | None
    stakes: str
    special_rules: str | None
    status: Literal["waiting", "full", "playing"] = "waiting"
    slots_total: int = 4
    slots_filled: int = 0
    participants: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BoardState:
    """Latest owner-authored board for one room, persisted independently of sessions."""

    room_id: str
    items: list[BoardItem]
    source_message_id: str
    last_published_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "room_id": self.room_id,
            "items": [item.to_dict() for item in self.items],
            "source_message_id": self.source_message_id,
            "last_published_at": self.last_published_at.isoformat(),
        }


@dataclass(slots=True)
class BoardSnapshotItem:
    """Immutable number-to-game mapping belonging to one outbound board version."""

    item_no: int
    game_id: str
    rendered_text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BoardSnapshot:
    """Immutable mapping from one published board version to game IDs."""

    snapshot_id: str
    room_id: str
    conversation_id: str
    external_message_id: str
    rendered_text: str
    items: list[BoardSnapshotItem]
    created_at: datetime = field(default_factory=now)

    def to_dict(self) -> dict[str, Any]:
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
    claim_id: str
    room_id: str
    game_id: str
    customer_id: str
    source_conversation_id: str
    source_message_id: str
    status: str = "claimed"
    created_at: datetime = field(default_factory=now)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["created_at"] = self.created_at.isoformat()
        return payload


@dataclass(slots=True)
class ChannelSwitch:
    """Short-lived pointer proving that a public request continued in private."""

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
        payload = asdict(self)
        payload["created_at"] = self.created_at.isoformat()
        payload["expires_at"] = self.expires_at.isoformat()
        return payload


@dataclass(slots=True)
class PrivateSwitchContext:
    private_conversation_id: str
    trigger_summary: str
    customer_profile: dict[str, Any]
    user_original_text: str
    parsed_need: dict[str, Any]
    missing_info: list[str]
    reply_guidelines: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class L1Result:
    action: str
    parsed_game: dict[str, Any] | None = None
    item_no: int | None = None
    needs_ack: bool = False


@dataclass(slots=True)
class RoutingDecision:
    action: str
    channel: str | None = None
    reply_constraints: ReplyConstraints | None = None
    group_ack: str = ""
    switch_context: PrivateSwitchContext | None = None


@dataclass(slots=True)
class ClaimResult:
    status: str
    game_id: str | None = None
    reason: str = ""
    deduplicated: bool = False


@dataclass(slots=True)
class GroupHandleResult:
    action: str
    game_id: str | None = None
    reply: str = ""
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ChatSession:
    """A bounded public-room interaction unit, never the whole room transcript."""

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
    """Strict semantic result returned by the dedicated group-session model task."""

    intent: Literal["claim", "new_demand", "query", "thread_update", "chitchat"]
    extracted_features: dict[str, Any]
    matched_board_no: int | None
    confidence: float
    reasoning: str
    response: str | None
    channel_action: Literal["private_switch", "group_reply", "ignore"]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GroupSessionOutcome:
    """Observable result of buffering, parsing, classifying, or routing one unit."""

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
