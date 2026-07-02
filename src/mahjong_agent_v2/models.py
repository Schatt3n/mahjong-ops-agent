from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any
from zoneinfo import ZoneInfo


DEFAULT_TZ_V2 = ZoneInfo("Asia/Shanghai")


class ConversationRoleV2(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class GameStatusV2(StrEnum):
    FORMING = "forming"
    INVITING = "inviting"
    READY = "ready"
    CANCELLED = "cancelled"
    FINISHED = "finished"


class InviteStatusV2(StrEnum):
    PENDING_APPROVAL = "pending_approval"
    SENT = "sent"
    CONFIRMED = "confirmed"
    DECLINED = "declined"
    NEGOTIATING = "negotiating"
    NO_REPLY = "no_reply"


@dataclass(slots=True)
class UserMessageV2:
    conversation_id: str
    sender_id: str
    sender_name: str
    text: str
    message_id: str = field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:12]}")
    sent_at: datetime = field(default_factory=lambda: datetime.now(DEFAULT_TZ_V2))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["sent_at"] = self.sent_at.isoformat()
        return data


@dataclass(slots=True)
class ConversationTurnV2:
    role: ConversationRoleV2
    content: str
    trace_id: str
    sender_id: str | None = None
    sender_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    occurred_at: datetime = field(default_factory=lambda: datetime.now(DEFAULT_TZ_V2))

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
class CustomerProfileV2:
    customer_id: str
    display_name: str
    gender: str | None = None
    preferred_games: list[str] = field(default_factory=list)
    preferred_stakes: list[str] = field(default_factory=list)
    preferred_time_tags: list[str] = field(default_factory=list)
    smoke_preference: str | None = None
    fatigue_score: float = 0.0
    response_score: float = 0.5
    no_contact: bool = False
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GameParticipantV2:
    customer_id: str
    display_name: str
    status: str = "joined"
    source: str = "organizer"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GameV2:
    game_id: str
    conversation_id: str
    organizer_id: str
    organizer_name: str
    requirement: dict[str, Any]
    status: GameStatusV2 = GameStatusV2.FORMING
    participants: list[GameParticipantV2] = field(default_factory=list)
    seats_total: int = 4
    created_at: datetime = field(default_factory=lambda: datetime.now(DEFAULT_TZ_V2))
    updated_at: datetime = field(default_factory=lambda: datetime.now(DEFAULT_TZ_V2))

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "conversation_id": self.conversation_id,
            "organizer_id": self.organizer_id,
            "organizer_name": self.organizer_name,
            "requirement": dict(self.requirement),
            "status": self.status.value,
            "participants": [participant.to_dict() for participant in self.participants],
            "seats_total": self.seats_total,
            "remaining_seats": self.remaining_seats(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    def remaining_seats(self) -> int:
        confirmed = sum(1 for participant in self.participants if participant.status in {"joined", "confirmed"})
        return max(0, self.seats_total - confirmed)


@dataclass(slots=True)
class InviteDraftV2:
    draft_id: str
    game_id: str
    customer_id: str
    display_name: str
    message_text: str
    status: InviteStatusV2 = InviteStatusV2.PENDING_APPROVAL
    created_at: datetime = field(default_factory=lambda: datetime.now(DEFAULT_TZ_V2))
    updated_at: datetime = field(default_factory=lambda: datetime.now(DEFAULT_TZ_V2))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "draft_id": self.draft_id,
            "game_id": self.game_id,
            "customer_id": self.customer_id,
            "display_name": self.display_name,
            "message_text": self.message_text,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class StateTransitionV2:
    entity_type: str
    entity_id: str
    from_status: str | None
    to_status: str
    reason: str
    trace_id: str
    occurred_at: datetime = field(default_factory=lambda: datetime.now(DEFAULT_TZ_V2))

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

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "StateTransitionV2":
        occurred_at = payload.get("occurred_at")
        return cls(
            entity_type=str(payload.get("entity_type") or ""),
            entity_id=str(payload.get("entity_id") or ""),
            from_status=payload.get("from_status"),
            to_status=str(payload.get("to_status") or ""),
            reason=str(payload.get("reason") or ""),
            trace_id=str(payload.get("trace_id") or ""),
            occurred_at=(
                datetime.fromisoformat(str(occurred_at))
                if occurred_at
                else datetime.now(DEFAULT_TZ_V2)
            ),
        )


@dataclass(slots=True)
class ToolCallV2:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    idempotency_key: str | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ToolResultV2:
    name: str
    called: bool
    allowed: bool
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    idempotency_key: str | None = None
    deduplicated: bool = False
    state_transitions: list[StateTransitionV2] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "called": self.called,
            "allowed": self.allowed,
            "result": dict(self.result),
            "error": self.error,
            "idempotency_key": self.idempotency_key,
            "deduplicated": self.deduplicated,
            "state_transitions": [transition.to_dict() for transition in self.state_transitions],
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ToolResultV2":
        return cls(
            name=str(payload.get("name") or ""),
            called=bool(payload.get("called")),
            allowed=bool(payload.get("allowed")),
            result=dict(payload.get("result") or {}),
            error=payload.get("error"),
            idempotency_key=payload.get("idempotency_key"),
            deduplicated=bool(payload.get("deduplicated", False)),
            state_transitions=[
                StateTransitionV2.from_payload(item)
                for item in payload.get("state_transitions") or []
                if isinstance(item, dict)
            ],
        )


@dataclass(slots=True)
class AgentDecisionV2:
    goal: str
    reasoning_summary: str
    reply_to_user: str
    tool_calls: list[ToolCallV2] = field(default_factory=list)
    needs_human: bool = False
    badcase: dict[str, Any] | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "AgentDecisionV2":
        calls: list[ToolCallV2] = []
        raw_calls = payload.get("tool_calls") or []
        if isinstance(raw_calls, list):
            for item in raw_calls:
                if not isinstance(item, dict):
                    continue
                calls.append(
                    ToolCallV2(
                        name=str(item.get("name") or item.get("tool_name") or ""),
                        arguments=dict(item.get("arguments") or {}) if isinstance(item.get("arguments"), dict) else {},
                        idempotency_key=(
                            str(item.get("idempotency_key"))
                            if item.get("idempotency_key") not in (None, "")
                            else None
                        ),
                        reason=str(item.get("reason") or ""),
                    )
                )
        badcase = payload.get("badcase") if isinstance(payload.get("badcase"), dict) else None
        return cls(
            goal=str(payload.get("goal") or ""),
            reasoning_summary=str(payload.get("reasoning_summary") or ""),
            reply_to_user=str(payload.get("reply_to_user") or ""),
            tool_calls=calls,
            needs_human=bool(payload.get("needs_human", False)),
            badcase=badcase,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "reasoning_summary": self.reasoning_summary,
            "reply_to_user": self.reply_to_user,
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "needs_human": self.needs_human,
            "badcase": self.badcase,
        }


@dataclass(slots=True)
class ReplyReviewV2:
    approved: bool
    reasoning_summary: str
    revised_reply: str = ""
    badcase: dict[str, Any] | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ReplyReviewV2":
        badcase = payload.get("badcase") if isinstance(payload.get("badcase"), dict) else None
        return cls(
            approved=bool(payload.get("approved", False)),
            reasoning_summary=str(payload.get("reasoning_summary") or ""),
            revised_reply=str(payload.get("revised_reply") or ""),
            badcase=badcase,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "approved": self.approved,
            "reasoning_summary": self.reasoning_summary,
            "revised_reply": self.revised_reply,
            "badcase": self.badcase,
        }


@dataclass(slots=True)
class AgentRuntimeResultV2:
    trace_id: str
    final_reply: str
    decisions: list[AgentDecisionV2]
    tool_results: list[ToolResultV2]
    state_transitions: list[StateTransitionV2]
    conversation_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "final_reply": self.final_reply,
            "decisions": [decision.to_dict() for decision in self.decisions],
            "tool_results": [result.to_dict() for result in self.tool_results],
            "state_transitions": [transition.to_dict() for transition in self.state_transitions],
            "conversation_id": self.conversation_id,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "AgentRuntimeResultV2":
        return cls(
            trace_id=str(payload.get("trace_id") or ""),
            final_reply=str(payload.get("final_reply") or ""),
            decisions=[
                AgentDecisionV2.from_payload(item)
                for item in payload.get("decisions") or []
                if isinstance(item, dict)
            ],
            tool_results=[
                ToolResultV2.from_payload(item)
                for item in payload.get("tool_results") or []
                if isinstance(item, dict)
            ],
            state_transitions=[
                StateTransitionV2.from_payload(item)
                for item in payload.get("state_transitions") or []
                if isinstance(item, dict)
            ],
            conversation_id=payload.get("conversation_id"),
        )


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"
