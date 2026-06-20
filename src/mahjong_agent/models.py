from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo


DEFAULT_TZ = ZoneInfo("Asia/Shanghai")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


class ChannelType(StrEnum):
    WECHAT_GROUP = "wechat_group"
    WECHAT_PRIVATE = "wechat_private"
    WEWORK_GROUP = "wework_group"
    WEWORK_PRIVATE = "wework_private"
    XIAOHONGSHU_COMMENT = "xiaohongshu_comment"
    XIAOHONGSHU_PRIVATE = "xiaohongshu_private"
    DOUYIN_COMMENT = "douyin_comment"
    DOUYIN_PRIVATE = "douyin_private"
    WEB_CONSOLE = "web_console"
    API = "api"
    MANUAL = "manual"


class Intent(StrEnum):
    UNKNOWN = "unknown"
    FIND_PLAYERS = "find_players"
    JOIN_GAME = "join_game"
    CANCEL_OR_FULL = "cancel_or_full"
    UPDATE_GAME = "update_game"


class GameStatus(StrEnum):
    NEED_CLARIFICATION = "need_clarification"
    OPEN = "open"
    NEGOTIATING = "negotiating"
    HOLDING = "holding"
    CONFIRMED = "confirmed"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class RoomHoldStatus(StrEnum):
    ACTIVE = "active"
    CANCELLED = "cancelled"


class InvitationStatus(StrEnum):
    QUEUED = "queued"
    SENT = "sent"
    ACCEPTED = "accepted"
    DECLINED = "declined"
    CANCELLED = "cancelled"
    SUPERSEDED = "superseded"


@dataclass(slots=True)
class Message:
    text: str
    sender_id: str
    sender_name: str
    channel_id: str
    channel_type: ChannelType = ChannelType.MANUAL
    sent_at: datetime = field(default_factory=lambda: datetime.now(DEFAULT_TZ))
    id: str = field(default_factory=lambda: new_id("msg"))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GameRequest:
    organizer_id: str
    organizer_name: str
    channel_id: str
    source_message_id: str | None = None
    id: str = field(default_factory=lambda: new_id("game"))
    status: GameStatus = GameStatus.NEED_CLARIFICATION
    game_type: str = "mahjong"
    ruleset: str | None = None
    variant: str | None = None
    seats_total: int = 4
    current_player_count: int | None = None
    missing_count: int | None = None
    level: str | None = None
    base_score: float | None = None
    cap_score: float | None = None
    start_at: datetime | None = None
    start_time_confidence: float = 0.0
    duration_hours: float | None = None
    play_options: list[str] = field(default_factory=list)
    rules: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    ambiguities: list[str] = field(default_factory=list)
    participant_ids: list[str] = field(default_factory=list)
    reserved_customer_ids: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(DEFAULT_TZ))
    updated_at: datetime = field(default_factory=lambda: datetime.now(DEFAULT_TZ))
    version: int = 0

    @property
    def open_slots(self) -> int | None:
        if self.missing_count is None:
            return None
        return max(0, self.missing_count - len(set(self.reserved_customer_ids)))

    @property
    def is_full(self) -> bool:
        return self.open_slots == 0

    def touch(self) -> None:
        self.updated_at = datetime.now(DEFAULT_TZ)
        self.version += 1


@dataclass(slots=True)
class RoomHold:
    start_at: datetime
    end_at: datetime
    room_id: str | None = None
    source: str = "manual"
    game_id: str | None = None
    status: RoomHoldStatus = RoomHoldStatus.ACTIVE
    id: str = field(default_factory=lambda: new_id("room_hold"))
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RoomAvailability:
    requested_start_at: datetime
    requested_end_at: datetime
    duration_hours: float
    available: bool
    suggested_start_at: datetime | None = None
    suggested_end_at: datetime | None = None
    occupied_rooms: int = 0
    capacity: int | None = None
    reason: str | None = None


@dataclass(slots=True)
class PlayPreference:
    game_type: str
    preferred_levels: list[str] = field(default_factory=list)
    preferred_rulesets: list[str] = field(default_factory=list)
    preferred_variants: list[str] = field(default_factory=list)
    preferred_play_options: list[str] = field(default_factory=list)
    avoid_play_options: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CustomerProfile:
    id: str
    display_name: str
    aliases: list[str] = field(default_factory=list)
    preferred_levels: list[str] = field(default_factory=list)
    play_preferences: list[PlayPreference] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    smoke_free_preference: bool | None = None
    usual_party_size: int | None = None
    usual_party_size_confidence: float = 0.0
    usual_start_hours: list[int] = field(default_factory=list)
    usual_weekdays: list[int] = field(default_factory=list)
    no_contact: bool = False
    last_invited_at: datetime | None = None
    decline_count_30d: int = 0
    max_games_per_day: int = 1
    min_hours_between_games: float = 6.0
    invite_cooldown_hours: float = 6.0
    daily_invite_limit: int = 3
    fatigue_sensitivity: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CustomerFatigue:
    customer_id: str
    games_on_day: int = 0
    invitations_on_day: int = 0
    max_games_per_day: int = 1
    daily_invite_limit: int = 3
    hours_since_last_game: float | None = None
    score_adjustment: float = 0.0
    hard_block: bool = False
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ExtractionResult:
    message_id: str
    intent: Intent
    confidence: float
    game: GameRequest | None = None
    follow_up_questions: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CandidateRecommendation:
    customer_id: str
    display_name: str
    score: float
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class MergeSuggestion:
    game_ids: list[str]
    score: float
    proposed_start_at: datetime | None
    reasons: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Invitation:
    game_id: str
    customer_id: str
    customer_name: str
    status: InvitationStatus = InvitationStatus.QUEUED
    id: str = field(default_factory=lambda: new_id("inv"))
    created_at: datetime = field(default_factory=lambda: datetime.now(DEFAULT_TZ))
    updated_at: datetime = field(default_factory=lambda: datetime.now(DEFAULT_TZ))
    message_text: str | None = None

    def set_status(self, status: InvitationStatus) -> None:
        self.status = status
        self.updated_at = datetime.now(DEFAULT_TZ)
