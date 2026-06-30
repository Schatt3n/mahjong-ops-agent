from __future__ import annotations

"""Controlled workflow contracts.

This module only defines data exchanged between the context builder, LLM
resolver, action validator, tool orchestrator, state machine, and reply policy.
It must not call LLMs, query databases, mutate durable state, or send messages.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, ClassVar
from uuid import uuid4

from .models import ChannelType, DEFAULT_TZ


def new_workflow_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


def _coerce_enum(enum_type: type[StrEnum], value: Any, default: StrEnum) -> StrEnum:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(value or default.value))
    except ValueError:
        return default


class SlotSource(StrEnum):
    EXPLICIT = "explicit"
    CONTEXT = "context"
    PROFILE = "profile"
    REGION_DEFAULT = "region_default"
    INFERRED = "inferred"
    TOOL = "tool"
    UNKNOWN = "unknown"


SLOT_SOURCE_PRIORITY: dict[SlotSource, int] = {
    SlotSource.EXPLICIT: 60,
    SlotSource.TOOL: 50,
    SlotSource.CONTEXT: 40,
    SlotSource.PROFILE: 30,
    SlotSource.REGION_DEFAULT: 20,
    SlotSource.INFERRED: 10,
    SlotSource.UNKNOWN: 0,
}


class UserIntent(StrEnum):
    UNKNOWN = "unknown"
    INQUIRE_EXISTING_GAME = "inquire_existing_game"
    FIND_PLAYERS = "find_players"
    JOIN_GAME = "join_game"
    UPDATE_GAME = "update_game"
    CANCEL_GAME = "cancel_game"
    CANDIDATE_REPLY = "candidate_reply"
    IRRELEVANT = "irrelevant"


class ActionName(StrEnum):
    UNKNOWN = "unknown"
    SEARCH_EXISTING_GAMES = "search_existing_games"
    ASK_CREATE_CONFIRMATION = "ask_create_confirmation"
    ASK_CLARIFICATION = "ask_clarification"
    CREATE_GAME = "create_game"
    QUEUE_INVITES = "queue_invites"
    MATCH_EXISTING_GAME = "match_existing_game"
    JOIN_GAME = "join_game"
    CANCEL_GAME = "cancel_game"
    ACCEPT_SEAT = "accept_seat"
    CLOSE_GAME = "close_game"
    HUMAN_REVIEW = "human_review"
    IGNORE = "ignore"


class ActionSource(StrEnum):
    LLM = "llm"
    RULES = "rules"
    HUMAN = "human"
    TOOL = "tool"


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ToolName(StrEnum):
    UNKNOWN = "unknown"
    SEARCH_CURRENT_OPEN_GAMES = "search_current_open_games"
    SEARCH_CANDIDATE_CUSTOMERS = "search_candidate_customers"
    CREATE_PENDING_OUTBOX = "create_pending_outbox"
    SEND_MESSAGE = "send_message"
    CREATE_GAME = "create_game"
    CLOSE_GAME = "close_game"
    RECORD_SEAT_ACCEPTANCE = "record_seat_acceptance"
    PROFILE_UPDATE = "profile_update"
    RECORD_APPROVAL_DECISION = "record_approval_decision"


class EntityType(StrEnum):
    GAME = "game"
    INVITATION = "invitation"
    OUTBOX = "outbox"
    PROFILE = "profile"


class GameWorkflowStatus(StrEnum):
    NEED_CLARIFICATION = "need_clarification"
    OPEN = "open"
    NEGOTIATING = "negotiating"
    HOLDING = "holding"
    CONFIRMED = "confirmed"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class ToolExecutionMode(StrEnum):
    READ_ONLY = "read_only"
    CREATE_PENDING = "create_pending"
    STATE_WRITE = "state_write"
    DIRECT_SEND = "direct_send"
    NOT_CALLED = "not_called"


class ReplyStatus(StrEnum):
    DRAFT = "draft"
    NEEDS_APPROVAL = "needs_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    GUARDED = "guarded"


@dataclass(slots=True)
class SlotValue:
    name: str
    value: Any
    source: SlotSource = SlotSource.UNKNOWN
    confidence: float = 0.0
    confirmed: bool = False
    needs_confirmation: bool = True
    evidence: str | None = None
    updated_at: datetime = field(default_factory=lambda: datetime.now(DEFAULT_TZ))
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.confidence = max(0.0, min(1.0, float(self.confidence or 0.0)))
        self.source = _coerce_enum(SlotSource, self.source, SlotSource.UNKNOWN)

    @property
    def usable(self) -> bool:
        return self.confirmed and not self.needs_confirmation and self.value not in (None, "", "unknown")

    @property
    def trusted_for_state(self) -> bool:
        return self.usable and self.source in {SlotSource.EXPLICIT, SlotSource.CONTEXT, SlotSource.TOOL}

    def require_confirmation(self, reason: str | None = None) -> SlotValue:
        metadata = dict(self.metadata)
        if reason:
            metadata["confirmation_reason"] = reason
        return SlotValue(
            name=self.name,
            value=self.value,
            source=self.source,
            confidence=self.confidence,
            confirmed=False,
            needs_confirmation=True,
            evidence=self.evidence,
            updated_at=self.updated_at,
            metadata=metadata,
        )

    def confirm(self, *, source: SlotSource | None = None, evidence: str | None = None) -> SlotValue:
        return SlotValue(
            name=self.name,
            value=self.value,
            source=source or self.source,
            confidence=self.confidence,
            confirmed=True,
            needs_confirmation=False,
            evidence=evidence or self.evidence,
            updated_at=datetime.now(DEFAULT_TZ),
            metadata=dict(self.metadata),
        )

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "source": self.source.value,
            "confidence": self.confidence,
            "confirmed": self.confirmed,
            "needs_confirmation": self.needs_confirmation,
            "evidence": self.evidence,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class UserMessage:
    text: str
    sender_id: str
    sender_name: str
    conversation_id: str
    trace_id: str
    message_id: str = field(default_factory=lambda: new_workflow_id("message"))
    channel_type: ChannelType = ChannelType.MANUAL
    sent_at: datetime = field(default_factory=lambda: datetime.now(DEFAULT_TZ))
    modalities: list[str] = field(default_factory=lambda: ["text"])
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.channel_type = _coerce_enum(ChannelType, self.channel_type, ChannelType.MANUAL)

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "sender_id": self.sender_id,
            "sender_name": self.sender_name,
            "conversation_id": self.conversation_id,
            "trace_id": self.trace_id,
            "message_id": self.message_id,
            "channel_type": self.channel_type.value,
            "sent_at": self.sent_at.isoformat(),
            "modalities": list(self.modalities),
        }


@dataclass(slots=True)
class CustomerProfile:
    customer_id: str
    display_name: str
    preferred_slots: dict[str, SlotValue] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    recent_facts: list[str] = field(default_factory=list)
    fatigue: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def slot(self, name: str) -> SlotValue | None:
        return self.preferred_slots.get(name)


@dataclass(slots=True)
class GameRequirement:
    slots: dict[str, SlotValue] = field(default_factory=dict)
    seats_total: int = 4
    organizer_id: str | None = None
    organizer_name: str | None = None
    candidate_composition_preference: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    required_slot_names: ClassVar[tuple[str, ...]] = (
        "game_type",
        "stake",
        "start_time_mode",
        "party_size",
        "smoke",
        "duration_mode",
    )

    def slot(self, name: str) -> SlotValue | None:
        return self.slots.get(name)

    def set_slot(self, slot: SlotValue, *, prefer_confirmed: bool = True) -> None:
        current = self.slots.get(slot.name)
        if current is None:
            self.slots[slot.name] = slot
            return
        if prefer_confirmed and current.usable and not slot.usable:
            return
        if slot.usable and not current.usable:
            self.slots[slot.name] = slot
            return
        if slot.usable and current.usable:
            slot_priority = SLOT_SOURCE_PRIORITY.get(slot.source, 0)
            current_priority = SLOT_SOURCE_PRIORITY.get(current.source, 0)
            if slot_priority > current_priority or (
                slot_priority == current_priority and slot.confidence >= current.confidence
            ):
                self.slots[slot.name] = slot
            return
        if slot.confidence >= current.confidence:
            self.slots[slot.name] = slot

    def missing_required_slots(self, required: tuple[str, ...] | None = None) -> list[str]:
        required_names = required or self.required_slot_names
        return [name for name in required_names if not self._slot_satisfies(name)]

    def is_complete(self, required: tuple[str, ...] | None = None) -> bool:
        return not self.missing_required_slots(required)

    def inherit_confirmed_context(self, previous: GameRequirement) -> None:
        for slot in previous.slots.values():
            if slot.usable:
                inherited = SlotValue(
                    name=slot.name,
                    value=slot.value,
                    source=SlotSource.CONTEXT,
                    confidence=slot.confidence,
                    confirmed=True,
                    needs_confirmation=False,
                    evidence=slot.evidence,
                    metadata={**slot.metadata, "inherited_from_context": True},
                )
                self.set_slot(inherited)

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "slots": {name: slot.to_prompt_dict() for name, slot in self.slots.items()},
            "missing_required_slots": self.missing_required_slots(),
            "complete": self.is_complete(),
            "seats_total": self.seats_total,
            "organizer_id": self.organizer_id,
            "organizer_name": self.organizer_name,
            "candidate_composition_preference": dict(self.candidate_composition_preference),
            "notes": list(self.notes),
        }

    def _slot_satisfies(self, name: str) -> bool:
        slot = self.slots.get(name)
        if slot and slot.usable:
            return True
        if name == "start_time_mode":
            fixed_time = self.slots.get("start_at")
            return bool(fixed_time and fixed_time.usable) or bool(slot and slot.usable)
        if name == "duration_mode":
            duration = self.slots.get("duration_hours")
            return bool(duration and duration.usable) or bool(slot and slot.usable)
        if name == "party_size":
            current_count = self.slots.get("current_player_count")
            missing_count = self.slots.get("missing_count")
            return (
                bool(slot and slot.usable)
                or bool(current_count and current_count.usable)
                or bool(missing_count and missing_count.usable)
            )
        return False


@dataclass(slots=True)
class ToolCallRequest:
    tool_name: ToolName
    arguments: dict[str, Any] = field(default_factory=dict)
    risk_level: RiskLevel = RiskLevel.LOW
    execution_mode: ToolExecutionMode = ToolExecutionMode.READ_ONLY
    idempotency_key: str | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        self.tool_name = _coerce_enum(ToolName, self.tool_name, ToolName.UNKNOWN)
        self.risk_level = _coerce_enum(RiskLevel, self.risk_level, RiskLevel.LOW)
        self.execution_mode = _coerce_enum(
            ToolExecutionMode,
            self.execution_mode,
            ToolExecutionMode.READ_ONLY,
        )


@dataclass(slots=True)
class ToolResult:
    request: ToolCallRequest
    called: bool
    allowed: bool
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    deduplicated: bool = False


ToolCallResult = ToolResult


@dataclass(slots=True)
class WorkflowTurn:
    user_message: UserMessage
    system_reply: str | None = None
    game_requirement: GameRequirement | None = None
    tool_results: list[ToolResult] = field(default_factory=list)
    at: datetime = field(default_factory=lambda: datetime.now(DEFAULT_TZ))

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "user_message": self.user_message.to_prompt_dict(),
            "system_reply": self.system_reply,
            "game_requirement": self.game_requirement.to_prompt_dict() if self.game_requirement else None,
            "tool_results": [
                {
                    "tool_name": result.request.tool_name.value,
                    "called": result.called,
                    "allowed": result.allowed,
                    "result": dict(result.result),
                    "error": result.error,
                    "deduplicated": result.deduplicated,
                }
                for result in self.tool_results
            ],
            "at": self.at.isoformat(),
        }


@dataclass(slots=True)
class ConversationContext:
    current_message: UserMessage
    customer_profile: CustomerProfile | None = None
    recent_turns: list[WorkflowTurn] = field(default_factory=list)
    active_game: GameRequirement | None = None
    open_games: list[GameRequirement] = field(default_factory=list)
    room_state: dict[str, Any] = field(default_factory=dict)
    memory_summary: str | None = None
    followup_context: dict[str, Any] = field(default_factory=dict)
    trace_notes: list[str] = field(default_factory=list)

    def previous_system_reply(self) -> str | None:
        for turn in reversed(self.recent_turns):
            if turn.system_reply:
                return turn.system_reply
        return None

    def previous_game_requirement(self) -> GameRequirement | None:
        for turn in reversed(self.recent_turns):
            if turn.game_requirement:
                return turn.game_requirement
        return None

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "current_message": self.current_message.to_prompt_dict(),
            "customer_profile": {
                "customer_id": self.customer_profile.customer_id,
                "display_name": self.customer_profile.display_name,
                "preferred_slots": {
                    name: slot.to_prompt_dict()
                    for name, slot in self.customer_profile.preferred_slots.items()
                },
                "tags": list(self.customer_profile.tags),
                "recent_facts": list(self.customer_profile.recent_facts),
                "fatigue": dict(self.customer_profile.fatigue),
            }
            if self.customer_profile
            else None,
            "recent_turns": [turn.to_prompt_dict() for turn in self.recent_turns],
            "previous_system_reply": self.previous_system_reply(),
            "previous_game_requirement": self.previous_game_requirement().to_prompt_dict()
            if self.previous_game_requirement()
            else None,
            "active_game": self.active_game.to_prompt_dict() if self.active_game else None,
            "open_games": [game.to_prompt_dict() for game in self.open_games],
            "room_state": dict(self.room_state),
            "memory_summary": self.memory_summary,
            "followup_context": dict(self.followup_context),
            "trace_notes": list(self.trace_notes),
        }


@dataclass(slots=True)
class ProposedAction:
    name: ActionName
    source: ActionSource
    confidence: float
    reason: str
    arguments: dict[str, Any] = field(default_factory=dict)
    risk_level: RiskLevel = RiskLevel.LOW

    def __post_init__(self) -> None:
        self.confidence = max(0.0, min(1.0, float(self.confidence or 0.0)))
        self.name = _coerce_enum(ActionName, self.name, ActionName.UNKNOWN)
        self.source = _coerce_enum(ActionSource, self.source, ActionSource.LLM)
        self.risk_level = _coerce_enum(RiskLevel, self.risk_level, RiskLevel.LOW)


@dataclass(slots=True)
class SemanticResolution:
    intent: UserIntent
    proposed_action: ProposedAction
    game_requirement: GameRequirement = field(default_factory=GameRequirement)
    needs_human_review: bool = False
    reasoning_summary: str = ""
    raw_response: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.intent = _coerce_enum(UserIntent, self.intent, UserIntent.UNKNOWN)


@dataclass(slots=True)
class ValidatedAction:
    proposed_action: ProposedAction
    effective_action: ActionName
    allowed: bool
    code: str
    reason: str
    missing_slots: list[str] = field(default_factory=list)
    approval_required: bool = False
    risk_level: RiskLevel = RiskLevel.LOW
    idempotency_key: str | None = None
    notes: list[str] = field(default_factory=list)
    required_tools: list[ToolName] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.effective_action = _coerce_enum(ActionName, self.effective_action, ActionName.UNKNOWN)
        self.risk_level = _coerce_enum(RiskLevel, self.risk_level, RiskLevel.LOW)


@dataclass(slots=True)
class StateTransition:
    entity_type: str
    entity_id: str
    from_status: str | None
    to_status: str
    reason: str
    allowed: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ReplyDraft:
    text: str
    status: ReplyStatus = ReplyStatus.NEEDS_APPROVAL
    reasoning_summary: str = ""
    source: ActionSource = ActionSource.LLM
    risk_level: RiskLevel = RiskLevel.LOW
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.status = _coerce_enum(ReplyStatus, self.status, ReplyStatus.NEEDS_APPROVAL)
        self.source = _coerce_enum(ActionSource, self.source, ActionSource.LLM)
        self.risk_level = _coerce_enum(RiskLevel, self.risk_level, RiskLevel.LOW)


@dataclass(slots=True)
class GuardedReply:
    draft: ReplyDraft
    final_text: str
    changed: bool = False
    guard_reasons: list[str] = field(default_factory=list)
    status: ReplyStatus = ReplyStatus.GUARDED

    def __post_init__(self) -> None:
        self.status = _coerce_enum(ReplyStatus, self.status, ReplyStatus.GUARDED)


@dataclass(slots=True)
class WorkflowRun:
    trace_id: str
    context: ConversationContext
    semantic_resolution: SemanticResolution | None = None
    validated_action: ValidatedAction | None = None
    tool_results: list[ToolResult] = field(default_factory=list)
    state_transitions: list[StateTransition] = field(default_factory=list)
    reply_draft: ReplyDraft | None = None
    guarded_reply: GuardedReply | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(DEFAULT_TZ))
