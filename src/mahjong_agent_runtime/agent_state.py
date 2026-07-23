from __future__ import annotations

"""Durable, resumable state for one bounded Agent loop execution."""

import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any, Callable

from .models import (
    DEFAULT_TZ,
    QuotedMessageRef,
    StateTransition,
    ToolResult,
    UserMessage,
    now,
)
from .runtime_components import TurnBudgets


class AgentRunStatus(StrEnum):
    """Lifecycle of a persisted Agent execution."""

    RUNNING = "running"
    RECOVERABLE = "recoverable"
    COMPLETED = "completed"
    FAILED = "failed"
    SUPERSEDED = "superseded"


@dataclass(slots=True)
class AgentRunState:
    """Minimal state needed to continue after the last completed loop step.

    The checkpoint intentionally excludes rendered prompts and model credentials.
    Context is rebuilt from durable conversation state when execution resumes.
    """

    run_id: str
    trace_id: str
    conversation_id: str
    run_version: int
    message: dict[str, Any]
    status: AgentRunStatus = AgentRunStatus.RUNNING
    turn_prepared: bool = False
    next_step_index: int = 1
    actions: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    pending_tool_results: list[dict[str, Any]] = field(default_factory=list)
    turn_tool_evidence: list[dict[str, Any]] = field(default_factory=list)
    transitions: list[dict[str, Any]] = field(default_factory=list)
    budget_state: dict[str, int] = field(default_factory=dict)
    progress_state: dict[str, Any] = field(default_factory=dict)
    final_reply: str = ""
    runtime_status: str = ""
    last_error: str = ""
    attempts: int = 1
    lease_owner: str = ""
    lease_until: datetime | None = None
    created_at: datetime = field(default_factory=now)
    updated_at: datetime = field(default_factory=now)
    completed_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the checkpoint without adding model context or secrets."""

        return {
            "run_id": self.run_id,
            "trace_id": self.trace_id,
            "conversation_id": self.conversation_id,
            "run_version": self.run_version,
            "message": redact_runtime_payload(self.message),
            "status": self.status.value,
            "turn_prepared": self.turn_prepared,
            "next_step_index": self.next_step_index,
            "actions": redact_runtime_payload(self.actions),
            "tool_results": redact_runtime_payload(self.tool_results),
            "pending_tool_results": redact_runtime_payload(self.pending_tool_results),
            "turn_tool_evidence": redact_runtime_payload(self.turn_tool_evidence),
            "transitions": redact_runtime_payload(self.transitions),
            "budget_state": dict(self.budget_state),
            "progress_state": redact_runtime_payload(self.progress_state),
            "final_reply": self.final_reply,
            "runtime_status": self.runtime_status,
            "last_error": self.last_error,
            "attempts": self.attempts,
            "lease_owner": self.lease_owner,
            "lease_until": self.lease_until.isoformat() if self.lease_until else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AgentRunState":
        """Hydrate a checkpoint loaded from a persistence backend."""

        return cls(
            run_id=str(payload.get("run_id") or ""),
            trace_id=str(payload.get("trace_id") or ""),
            conversation_id=str(payload.get("conversation_id") or ""),
            run_version=int(payload.get("run_version") or 0),
            message=dict(payload.get("message") or {}),
            status=AgentRunStatus(str(payload.get("status") or AgentRunStatus.RECOVERABLE.value)),
            turn_prepared=bool(payload.get("turn_prepared")),
            next_step_index=max(1, int(payload.get("next_step_index") or 1)),
            actions=_dict_list(payload.get("actions")),
            tool_results=_dict_list(payload.get("tool_results")),
            pending_tool_results=_dict_list(payload.get("pending_tool_results")),
            turn_tool_evidence=_dict_list(payload.get("turn_tool_evidence")),
            transitions=_dict_list(payload.get("transitions")),
            budget_state={
                str(key): int(value)
                for key, value in dict(payload.get("budget_state") or {}).items()
            },
            progress_state=dict(payload.get("progress_state") or {}),
            final_reply=str(payload.get("final_reply") or ""),
            runtime_status=str(payload.get("runtime_status") or ""),
            last_error=str(payload.get("last_error") or ""),
            attempts=max(0, int(payload.get("attempts") or 0)),
            lease_owner=str(payload.get("lease_owner") or ""),
            lease_until=_optional_datetime(payload.get("lease_until")),
            created_at=_datetime(payload.get("created_at")),
            updated_at=_datetime(payload.get("updated_at")),
            completed_at=_optional_datetime(payload.get("completed_at")),
        )


@dataclass(slots=True)
class AgentRunRecoveryScheduler:
    """Poll expired/recoverable runs and delegate resumption to AgentRuntime."""

    handler: Callable[[int], list[Any]]
    trace_recorder: Any
    poll_interval_seconds: float = 1.0
    batch_limit: int = 20
    _stop_event: threading.Event = field(init=False, repr=False)
    _thread: threading.Thread | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self.poll_interval_seconds = max(0.1, float(self.poll_interval_seconds))
        self.batch_limit = max(1, int(self.batch_limit))
        self._stop_event = threading.Event()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="agent-run-recovery-scheduler",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout_seconds: float = 2.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.0, float(timeout_seconds)))

    def run_once(self) -> int:
        return len(self.handler(self.batch_limit))

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.run_once()
            except Exception as exc:
                self.trace_recorder.record(
                    "system_agent_run_recovery",
                    "agent_run_recovery_scheduler_error",
                    {"error_type": type(exc).__name__, "error": str(exc)},
                    level="ERROR",
                )
            self._stop_event.wait(self.poll_interval_seconds)


SENSITIVE_RUNTIME_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "cookie",
        "credentials",
        "password",
        "proxy_authorization",
        "secret",
    }
)


def redact_runtime_payload(value: Any) -> Any:
    """Remove credential-shaped values before a run checkpoint is persisted."""

    if isinstance(value, dict):
        return {
            str(key): (
                "[REDACTED]"
                if str(key).strip().lower() in SENSITIVE_RUNTIME_KEYS
                else redact_runtime_payload(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_runtime_payload(item) for item in value]
    return value


def restore_user_message(payload: dict[str, Any]) -> UserMessage:
    """Rebuild the original input identity without re-reading channel payloads."""

    quoted_payload = payload.get("quoted_message")
    quoted = (
        QuotedMessageRef(
            message_id=str(quoted_payload.get("message_id") or ""),
            sender_id=quoted_payload.get("sender_id"),
            sender_name=quoted_payload.get("sender_name"),
            text=str(quoted_payload.get("text") or ""),
            conversation_id=quoted_payload.get("conversation_id"),
            business_ref_type=quoted_payload.get("business_ref_type"),
            business_ref_id=quoted_payload.get("business_ref_id"),
            metadata=dict(quoted_payload.get("metadata") or {}),
        )
        if isinstance(quoted_payload, dict)
        else None
    )
    return UserMessage(
        conversation_id=str(payload.get("conversation_id") or ""),
        sender_id=str(payload.get("sender_id") or ""),
        sender_name=str(payload.get("sender_name") or ""),
        text=str(payload.get("text") or ""),
        message_id=str(payload.get("message_id") or ""),
        sent_at=_datetime(payload.get("sent_at")),
        quoted_message=quoted,
        metadata=dict(payload.get("metadata") or {}),
    )


def restore_tool_result(payload: dict[str, Any]) -> ToolResult:
    return ToolResult(
        name=str(payload.get("name") or ""),
        called=bool(payload.get("called")),
        allowed=bool(payload.get("allowed")),
        call_id=payload.get("call_id"),
        result=dict(payload.get("result") or {}),
        error=payload.get("error"),
        idempotency_key=payload.get("idempotency_key"),
        deduplicated=bool(payload.get("deduplicated")),
        state_transitions=[
            restore_transition(item)
            for item in payload.get("state_transitions") or []
            if isinstance(item, dict)
        ],
    )


def restore_transition(payload: dict[str, Any]) -> StateTransition:
    return StateTransition(
        entity_type=str(payload.get("entity_type") or ""),
        entity_id=str(payload.get("entity_id") or ""),
        from_status=payload.get("from_status"),
        to_status=str(payload.get("to_status") or ""),
        reason=str(payload.get("reason") or ""),
        trace_id=str(payload.get("trace_id") or ""),
        occurred_at=_datetime(payload.get("occurred_at")),
    )


def snapshot_budgets(budgets: TurnBudgets) -> dict[str, int]:
    return {
        "agent_calls": budgets.agent.calls_this_turn,
        "review_calls": budgets.review.calls_this_turn,
        "text_generation_calls": budgets.text_generation.calls_this_turn,
    }


def restore_budgets(budgets: TurnBudgets, state: dict[str, int]) -> None:
    budgets.agent.calls_this_turn = max(0, int(state.get("agent_calls") or 0))
    budgets.review.calls_this_turn = max(0, int(state.get("review_calls") or 0))
    budgets.text_generation.calls_this_turn = max(
        0,
        int(state.get("text_generation_calls") or 0),
    )


def lease_deadline(*, lease_seconds: int, at: datetime | None = None) -> datetime:
    return (at or now()) + timedelta(seconds=max(1, int(lease_seconds)))


def _dict_list(value: Any) -> list[dict[str, Any]]:
    return [dict(item) for item in value or [] if isinstance(item, dict)]


def _datetime(value: Any) -> datetime:
    parsed = _optional_datetime(value)
    return parsed or now()


def _optional_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=DEFAULT_TZ)
    if isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value)
            return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=DEFAULT_TZ)
        except ValueError:
            return None
    return None


__all__ = [
    "AgentRunRecoveryScheduler",
    "AgentRunState",
    "AgentRunStatus",
    "lease_deadline",
    "redact_runtime_payload",
    "restore_budgets",
    "restore_tool_result",
    "restore_transition",
    "restore_user_message",
    "snapshot_budgets",
]
