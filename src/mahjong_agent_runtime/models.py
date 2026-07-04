from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any
from zoneinfo import ZoneInfo


DEFAULT_TZ = ZoneInfo("Asia/Shanghai")


def now() -> datetime:
    return datetime.now(DEFAULT_TZ)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class ConversationRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class GameStatus(StrEnum):
    FORMING = "forming"
    INVITING = "inviting"
    READY = "ready"
    CANCELLED = "cancelled"
    FINISHED = "finished"


class InviteStatus(StrEnum):
    PENDING_APPROVAL = "pending_approval"
    SENT = "sent"
    CONFIRMED = "confirmed"
    DECLINED = "declined"
    NEGOTIATING = "negotiating"
    NO_REPLY = "no_reply"
    SUPERSEDED = "superseded"


class OutboundDraftStatus(StrEnum):
    PENDING_APPROVAL = "pending_approval"
    SENT = "sent"
    CANCELLED = "cancelled"
    SUPERSEDED = "superseded"


@dataclass(slots=True)
class QuotedMessageRef:
    message_id: str
    sender_id: str | None = None
    sender_name: str | None = None
    text: str = ""
    conversation_id: str | None = None
    business_ref_type: str | None = None
    business_ref_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MessageReference:
    message_id: str
    conversation_id: str
    business_ref_type: str
    business_ref_id: str
    text: str = ""
    channel: str | None = None
    sender_id: str | None = None
    sender_name: str | None = None
    recipient_id: str | None = None
    recipient_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=now)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        return data


@dataclass(slots=True)
class UserMessage:
    conversation_id: str
    sender_id: str
    sender_name: str
    text: str
    message_id: str = field(default_factory=lambda: new_id("msg"))
    sent_at: datetime = field(default_factory=now)
    quoted_message: QuotedMessageRef | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["sent_at"] = self.sent_at.isoformat()
        data["quoted_message"] = self.quoted_message.to_dict() if self.quoted_message else None
        return data


@dataclass(slots=True)
class ConversationTurn:
    role: ConversationRole
    content: str
    trace_id: str
    sender_id: str | None = None
    sender_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    occurred_at: datetime = field(default_factory=now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "content": self.content,
            "trace_id": self.trace_id,
            "sender_id": self.sender_id,
            "sender_name": self.sender_name,
            "metadata": dict(self.metadata),
            "occurred_at": self.occurred_at.isoformat(),
        }


@dataclass(slots=True)
class ConversationCheckpoint:
    conversation_id: str
    summary: str
    facts: dict[str, Any] = field(default_factory=dict)
    open_questions: list[str] = field(default_factory=list)
    source_trace_id: str | None = None
    updated_at: datetime = field(default_factory=now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "summary": self.summary,
            "facts": dict(self.facts),
            "open_questions": list(self.open_questions),
            "source_trace_id": self.source_trace_id,
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass(slots=True)
class CustomerProfile:
    customer_id: str
    display_name: str
    gender: str | None = None
    preferred_games: list[str] = field(default_factory=list)
    preferred_stakes: list[str] = field(default_factory=list)
    preferred_time_tags: list[str] = field(default_factory=list)
    smoke_preference: str | None = None
    response_score: float = 0.5
    fatigue_score: float = 0.0
    no_contact: bool = False
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CustomerRelationship:
    customer_a_id: str
    customer_b_id: str
    played_together_count: int = 0
    avoid_playing: bool = False
    notes: str = ""
    updated_at: datetime = field(default_factory=now)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["updated_at"] = self.updated_at.isoformat()
        return data


@dataclass(slots=True)
class GameParticipant:
    customer_id: str
    display_name: str
    status: str = "joined"
    source: str = "organizer"
    seat_count: int = 1
    party_id: str | None = None
    known_member_ids: list[str] = field(default_factory=list)
    anonymous_seat_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Party:
    party_id: str
    contact_id: str
    contact_name: str
    seat_count: int = 1
    known_member_ids: list[str] = field(default_factory=list)
    anonymous_seat_count: int = 0
    status: str = "joined"
    source: str = "requester"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Game:
    game_id: str
    conversation_id: str
    organizer_id: str
    organizer_name: str
    requirement: dict[str, Any]
    status: GameStatus = GameStatus.FORMING
    participants: list[GameParticipant] = field(default_factory=list)
    parties: list[Party] = field(default_factory=list)
    seats_total: int = 4
    created_at: datetime = field(default_factory=now)
    updated_at: datetime = field(default_factory=now)

    def __post_init__(self) -> None:
        if not self.parties:
            self.parties = parties_from_participants(self.participants)

    def remaining_seats(self) -> int:
        confirmed = sum(
            max(1, int(item.seat_count))
            for item in self.participants
            if item.status in {"joined", "confirmed"}
        )
        return max(0, self.seats_total - confirmed)

    def seat_claims(self) -> list[dict[str, Any]]:
        return [
            {
                "party_id": item.party_id or f"party_{item.customer_id}",
                "contact_id": item.customer_id,
                "contact_name": item.display_name,
                "seat_count": max(1, int(item.seat_count)),
                "known_member_ids": list(item.known_member_ids or [item.customer_id]),
                "anonymous_seat_count": max(0, int(item.anonymous_seat_count)),
                "status": item.status,
                "source": item.source,
            }
            for item in self.participants
            if item.status in {"joined", "confirmed"}
        ]

    def seat_summary(self) -> dict[str, Any]:
        claimed = sum(item["seat_count"] for item in self.seat_claims())
        return {
            "seats_total": self.seats_total,
            "claimed_seats": claimed,
            "remaining_seats": max(0, self.seats_total - claimed),
            "party_count": len(self.parties),
            "known_contact_count": len({item.contact_id for item in self.parties if item.contact_id}),
            "anonymous_seat_count": sum(max(0, int(item.anonymous_seat_count)) for item in self.parties),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "conversation_id": self.conversation_id,
            "organizer_id": self.organizer_id,
            "organizer_name": self.organizer_name,
            "requirement": dict(self.requirement),
            "status": self.status.value,
            "participants": [item.to_dict() for item in self.participants],
            "parties": [item.to_dict() for item in self.parties],
            "seat_claims": self.seat_claims(),
            "seat_summary": self.seat_summary(),
            "seats_total": self.seats_total,
            "remaining_seats": self.remaining_seats(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


def parties_from_participants(participants: list[GameParticipant]) -> list[Party]:
    parties: list[Party] = []
    seen: set[str] = set()
    for participant in participants:
        party_id = participant.party_id or f"party_{participant.customer_id}"
        if party_id in seen:
            continue
        seen.add(party_id)
        known_member_ids = list(participant.known_member_ids or [participant.customer_id])
        seat_count = max(1, int(participant.seat_count))
        parties.append(
            Party(
                party_id=party_id,
                contact_id=participant.customer_id,
                contact_name=participant.display_name,
                seat_count=seat_count,
                known_member_ids=known_member_ids,
                anonymous_seat_count=max(0, int(participant.anonymous_seat_count or max(0, seat_count - len(known_member_ids)))),
                status=participant.status,
                source=participant.source,
            )
        )
    return parties


@dataclass(slots=True)
class InviteDraft:
    draft_id: str
    game_id: str
    customer_id: str
    display_name: str
    message_text: str
    status: InviteStatus = InviteStatus.PENDING_APPROVAL
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=now)
    updated_at: datetime = field(default_factory=now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "draft_id": self.draft_id,
            "game_id": self.game_id,
            "customer_id": self.customer_id,
            "display_name": self.display_name,
            "message_text": self.message_text,
            "status": self.status.value,
            "metadata": dict(self.metadata),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass(slots=True)
class OutboundMessageDraft:
    draft_id: str
    conversation_id: str
    recipient_id: str
    recipient_name: str
    channel: str
    message_text: str
    purpose: str
    status: OutboundDraftStatus = OutboundDraftStatus.PENDING_APPROVAL
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=now)
    updated_at: datetime = field(default_factory=now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "draft_id": self.draft_id,
            "conversation_id": self.conversation_id,
            "recipient_id": self.recipient_id,
            "recipient_name": self.recipient_name,
            "channel": self.channel,
            "message_text": self.message_text,
            "purpose": self.purpose,
            "status": self.status.value,
            "metadata": dict(self.metadata),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass(slots=True)
class StateTransition:
    entity_type: str
    entity_id: str
    from_status: str | None
    to_status: str
    reason: str
    trace_id: str
    occurred_at: datetime = field(default_factory=now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "from_status": self.from_status,
            "to_status": self.to_status,
            "reason": self.reason,
            "trace_id": self.trace_id,
            "occurred_at": self.occurred_at.isoformat(),
        }


@dataclass(slots=True)
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    idempotency_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ToolResult:
    name: str
    called: bool
    allowed: bool
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    idempotency_key: str | None = None
    deduplicated: bool = False
    state_transitions: list[StateTransition] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "called": self.called,
            "allowed": self.allowed,
            "result": dict(self.result),
            "error": self.error,
            "idempotency_key": self.idempotency_key,
            "deduplicated": self.deduplicated,
            "state_transitions": [item.to_dict() for item in self.state_transitions],
        }


@dataclass(slots=True)
class AgentAction:
    goal: str
    objective_status: str
    reasoning_summary: str
    reply_to_user: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    needs_human: bool = False
    stop_reason: dict[str, Any] = field(default_factory=dict)
    badcase: dict[str, Any] | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "AgentAction":
        calls: list[ToolCall] = []
        for raw in payload.get("tool_calls") or []:
            if not isinstance(raw, dict):
                continue
            calls.append(
                ToolCall(
                    name=str(raw.get("name") or ""),
                    arguments=dict(raw.get("arguments") or {}) if isinstance(raw.get("arguments"), dict) else {},
                    reason=str(raw.get("reason") or ""),
                    idempotency_key=(
                        str(raw.get("idempotency_key"))
                        if raw.get("idempotency_key") not in {None, ""}
                        else None
                    ),
                )
            )
        badcase = payload.get("badcase") if isinstance(payload.get("badcase"), dict) else None
        return cls(
            goal=str(payload.get("goal") or ""),
            objective_status=str(payload.get("objective_status") or "unknown"),
            reasoning_summary=str(payload.get("reasoning_summary") or ""),
            reply_to_user=str(payload.get("reply_to_user") or ""),
            tool_calls=calls,
            needs_human=bool(payload.get("needs_human")),
            stop_reason=dict(payload.get("stop_reason") or {}) if isinstance(payload.get("stop_reason"), dict) else {},
            badcase=badcase,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "objective_status": self.objective_status,
            "reasoning_summary": self.reasoning_summary,
            "reply_to_user": self.reply_to_user,
            "tool_calls": [item.to_dict() for item in self.tool_calls],
            "needs_human": self.needs_human,
            "stop_reason": dict(self.stop_reason),
            "badcase": self.badcase,
        }


@dataclass(slots=True)
class AgentRuntimeResult:
    trace_id: str
    conversation_id: str
    final_reply: str
    actions: list[AgentAction] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    state_transitions: list[StateTransition] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "conversation_id": self.conversation_id,
            "final_reply": self.final_reply,
            "actions": [item.to_dict() for item in self.actions],
            "tool_results": [item.to_dict() for item in self.tool_results],
            "state_transitions": [item.to_dict() for item in self.state_transitions],
        }
