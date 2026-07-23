from __future__ import annotations

"""Fragmented user-input buffering and delayed re-evaluation infrastructure.

The main Agent loop should reason over a completed utterance, not own timers.
This module therefore sits before AgentRuntime: it persists fragments, builds one
aggregated UserMessage, and runs a small recoverable scheduler. Semantic boundary
decisions remain model-driven at the application edge.
"""

import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

from .models import DEFAULT_TZ, PendingInputBatch, QuotedMessageRef, UserMessage, now


@dataclass(slots=True)
class InputBatchDispatch:
    """Result returned by an input-boundary coordinator to a channel adapter."""

    status: str
    batch: PendingInputBatch
    message: UserMessage | None = None
    decision: dict[str, Any] | None = None
    trace_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "trace_id": self.trace_id,
            "batch": self.batch.to_dict(),
            "message": self.message.to_dict() if self.message else None,
            "decision": dict(self.decision or {}),
        }


def aggregate_pending_input_batch(
    batch: PendingInputBatch,
    *,
    quiet_period_elapsed: bool,
    trigger: str,
) -> UserMessage:
    """Merge ordered fragments while retaining their source ids for audit."""

    fragments = [dict(item) for item in batch.fragments]
    latest = fragments[-1] if fragments else {}
    texts = [str(item.get("text") or "").strip() for item in fragments]
    joined_text = "\n".join(item for item in texts if item)
    metadata = dict(latest.get("metadata") or {}) if isinstance(latest.get("metadata"), dict) else {}
    metadata["input_window"] = {
        "batch_id": batch.batch_id,
        "batch_version": batch.version,
        "fragment_count": len(fragments),
        "source_message_ids": [str(item.get("message_id") or "") for item in fragments],
        "fragments": [
            {
                "message_id": str(item.get("message_id") or ""),
                "text": str(item.get("text") or ""),
                "sent_at": str(item.get("sent_at") or ""),
            }
            for item in fragments
        ],
        "quiet_period_elapsed": bool(quiet_period_elapsed),
        "quiet_deadline": batch.quiet_deadline.isoformat(),
        "trigger": trigger,
    }
    return UserMessage(
        conversation_id=batch.conversation_id,
        sender_id=batch.sender_id,
        sender_name=batch.sender_name,
        text=joined_text,
        message_id=f"{batch.batch_id}:v{batch.version}",
        sent_at=_datetime_from_value(latest.get("sent_at")),
        quoted_message=_quoted_message_from_value(latest.get("quoted_message")),
        metadata=metadata,
    )


class PendingInputScheduler:
    """Poll durable due batches with recoverable leases and idempotent at-least-once dispatch."""

    def __init__(
        self,
        *,
        store: Any,
        handler: Callable[[PendingInputBatch, str], None],
        trace_recorder: Any,
        poll_interval_seconds: float = 0.5,
        batch_limit: int = 50,
    ) -> None:
        self.store = store
        self.handler = handler
        self.trace_recorder = trace_recorder
        self.poll_interval_seconds = max(0.05, float(poll_interval_seconds))
        self.batch_limit = max(1, int(batch_limit))
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="pending-input-scheduler", daemon=True)
        self._thread.start()

    def stop(self, timeout_seconds: float = 2.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.0, float(timeout_seconds)))

    def run_due_once(self, *, at: datetime | None = None) -> int:
        due = self.store.due_pending_input_batches(at=at or now(), limit=self.batch_limit)
        for batch in due:
            trace_id = f"trace_input_wait_{uuid.uuid4().hex[:12]}"
            background_queued = (
                str(batch.decision.get("dispatch_mode") or "") == "background_after_immediate_ack"
            )
            self.trace_recorder.record(
                trace_id,
                "input_background_dispatch_due" if background_queued else "input_quiet_period_elapsed",
                {
                    "batch_id": batch.batch_id,
                    "batch_version": batch.version,
                    "conversation_id": batch.conversation_id,
                    "sender_id": batch.sender_id,
                    "fragment_count": len(batch.fragments),
                    "origin_trace_id": str(batch.decision.get("origin_trace_id") or ""),
                },
            )
            try:
                self.handler(batch, trace_id)
            except Exception as exc:
                self.trace_recorder.record(
                    trace_id,
                    "input_quiet_period_handler_error",
                    {"error_type": type(exc).__name__, "error": str(exc), "batch_id": batch.batch_id},
                    level="ERROR",
                )
        return len(due)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self.run_due_once()
            self._stop_event.wait(self.poll_interval_seconds)


def _datetime_from_value(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value)
            return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=DEFAULT_TZ)
        except ValueError:
            pass
    return now()


def _quoted_message_from_value(value: Any) -> QuotedMessageRef | None:
    if not isinstance(value, dict) or not value.get("message_id"):
        return None
    return QuotedMessageRef(
        message_id=str(value.get("message_id") or ""),
        sender_id=str(value.get("sender_id") or "") or None,
        sender_name=str(value.get("sender_name") or "") or None,
        text=str(value.get("text") or ""),
        conversation_id=str(value.get("conversation_id") or "") or None,
        business_ref_type=str(value.get("business_ref_type") or "") or None,
        business_ref_id=str(value.get("business_ref_id") or "") or None,
        metadata=dict(value.get("metadata") or {}) if isinstance(value.get("metadata"), dict) else {},
    )
