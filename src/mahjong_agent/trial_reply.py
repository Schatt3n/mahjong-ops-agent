from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from .models import CandidateRecommendation, GameRequest


@dataclass(slots=True)
class TrialReplyDraftInput:
    conversation_id: str
    sender_id: str
    sender_name: str
    source_text: str
    effective_text: str
    trace_id: str
    game: GameRequest | None
    workflow_followup_context: dict[str, Any]
    missing_fields: list[str]
    decision_reply: str
    parsed: dict[str, Any]
    recommendations: list[CandidateRecommendation]
    outbox: list[dict[str, Any]]
    pool_matches: list[dict[str, Any]]
    tool_results: dict[str, Any]
    now: datetime


@dataclass(slots=True)
class TrialReplyDraftResult:
    suggested_reply: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrialReplyDraftCallbacks:
    suggested_reply: Callable[..., dict[str, Any]]
    update_sender_memory_after_reply: Callable[..., None]


@dataclass(slots=True)
class TrialReplyDraftAdapter:
    """Runs the legacy trial-page reply stage after tools and state context exist."""

    callbacks: TrialReplyDraftCallbacks

    def draft(self, data: TrialReplyDraftInput) -> TrialReplyDraftResult:
        suggested = self.callbacks.suggested_reply(
            source_text=data.source_text,
            effective_text=data.effective_text,
            trace_id=data.trace_id,
            sender_id=data.sender_id,
            sender_name=data.sender_name,
            game=data.game,
            workflow_followup_context=data.workflow_followup_context,
            missing_fields=data.missing_fields,
            decision_reply=data.decision_reply,
            recommendations=data.recommendations,
            outbox=data.outbox,
            pool_matches=data.pool_matches,
            tool_results=data.tool_results,
            now=data.now,
        )
        self.callbacks.update_sender_memory_after_reply(
            conversation_id=data.conversation_id,
            sender_id=data.sender_id,
            trace_id=data.trace_id,
            suggested_reply=suggested,
            parsed=data.parsed,
            tool_results=data.tool_results,
            pool_matches=data.pool_matches,
            now=data.now,
        )
        return TrialReplyDraftResult(suggested_reply=suggested)
