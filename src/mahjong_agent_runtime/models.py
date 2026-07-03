from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any
from zoneinfo import ZoneInfo


DEFAULT_TZ_V3 = ZoneInfo("Asia/Shanghai")


def now_v3() -> datetime:
    return datetime.now(DEFAULT_TZ_V3)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class ConversationRoleV3(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class GameStatusV3(StrEnum):
    FORMING = "forming"
    INVITING = "inviting"
    READY = "ready"
    CANCELLED = "cancelled"
    FINISHED = "finished"


class InviteStatusV3(StrEnum):
    PENDING_APPROVAL = "pending_approval"
    SENT = "sent"
    CONFIRMED = "confirmed"
    DECLINED = "declined"
    NEGOTIATING = "negotiating"
    NO_REPLY = "no_reply"


class OutboundDraftStatusV3(StrEnum):
    PENDING_APPROVAL = "pending_approval"
    SENT = "sent"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class UserMessageV3:
    conversation_id: str
    sender_id: str
    sender_name: str
    text: str
    message_id: str = field(default_factory=lambda: new_id("msg"))
    sent_at: datetime = field(default_factory=now_v3)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["sent_at"] = self.sent_at.isoformat()
        return data


@dataclass(slots=True)
class ConversationTurnV3:
    role: ConversationRoleV3
    content: str
    trace_id: str
    sender_id: str | None = None
    sender_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    occurred_at: datetime = field(default_factory=now_v3)

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
class ConversationCheckpointV3:
    conversation_id: str
    summary: str
    facts: dict[str, Any] = field(default_factory=dict)
    open_questions: list[str] = field(default_factory=list)
    source_trace_id: str | None = None
    updated_at: datetime = field(default_factory=now_v3)

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
class CustomerProfileV3:
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
class GameParticipantV3:
    customer_id: str
    display_name: str
    status: str = "joined"
    source: str = "organizer"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GameV3:
    game_id: str
    conversation_id: str
    organizer_id: str
    organizer_name: str
    requirement: dict[str, Any]
    status: GameStatusV3 = GameStatusV3.FORMING
    participants: list[GameParticipantV3] = field(default_factory=list)
    seats_total: int = 4
    created_at: datetime = field(default_factory=now_v3)
    updated_at: datetime = field(default_factory=now_v3)

    def remaining_seats(self) -> int:
        confirmed = sum(1 for item in self.participants if item.status in {"joined", "confirmed"})
        return max(0, self.seats_total - confirmed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "conversation_id": self.conversation_id,
            "organizer_id": self.organizer_id,
            "organizer_name": self.organizer_name,
            "requirement": dict(self.requirement),
            "status": self.status.value,
            "participants": [item.to_dict() for item in self.participants],
            "seats_total": self.seats_total,
            "remaining_seats": self.remaining_seats(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass(slots=True)
class InviteDraftV3:
    draft_id: str
    game_id: str
    customer_id: str
    display_name: str
    message_text: str
    status: InviteStatusV3 = InviteStatusV3.PENDING_APPROVAL
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=now_v3)
    updated_at: datetime = field(default_factory=now_v3)

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
class OutboundMessageDraftV3:
    draft_id: str
    conversation_id: str
    recipient_id: str
    recipient_name: str
    channel: str
    message_text: str
    purpose: str
    status: OutboundDraftStatusV3 = OutboundDraftStatusV3.PENDING_APPROVAL
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=now_v3)
    updated_at: datetime = field(default_factory=now_v3)

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
class StateTransitionV3:
    entity_type: str
    entity_id: str
    from_status: str | None
    to_status: str
    reason: str
    trace_id: str
    occurred_at: datetime = field(default_factory=now_v3)

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
class ToolCallV3:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    idempotency_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ToolResultV3:
    name: str
    called: bool
    allowed: bool
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    idempotency_key: str | None = None
    deduplicated: bool = False
    state_transitions: list[StateTransitionV3] = field(default_factory=list)

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
class AgentActionV3:
    goal: str
    objective_status: str
    reasoning_summary: str
    reply_to_user: str = ""
    tool_calls: list[ToolCallV3] = field(default_factory=list)
    needs_human: bool = False
    stop_reason: dict[str, Any] = field(default_factory=dict)
    badcase: dict[str, Any] | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "AgentActionV3":
        calls: list[ToolCallV3] = []
        for raw in payload.get("tool_calls") or []:
            if not isinstance(raw, dict):
                continue
            calls.append(
                ToolCallV3(
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
class AgentRuntimeResultV3:
    trace_id: str
    conversation_id: str
    final_reply: str
    actions: list[AgentActionV3] = field(default_factory=list)
    tool_results: list[ToolResultV3] = field(default_factory=list)
    state_transitions: list[StateTransitionV3] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "conversation_id": self.conversation_id,
            "final_reply": self.final_reply,
            "actions": [item.to_dict() for item in self.actions],
            "tool_results": [item.to_dict() for item in self.tool_results],
            "state_transitions": [item.to_dict() for item in self.state_transitions],
        }
