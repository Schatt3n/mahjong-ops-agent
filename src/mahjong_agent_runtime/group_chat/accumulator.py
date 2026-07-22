"""Short quiet-window aggregation for fragmented messages from one room member."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .models import GroupMessage


@dataclass(slots=True)
class AccumulatedMessage:
    message: GroupMessage
    trace_id: str


@dataclass(slots=True)
class _PendingFragments:
    fragments: list[GroupMessage] = field(default_factory=list)
    trace_id: str = ""
    quiet_deadline: datetime | None = None


class MessageAccumulator:
    """Merge same-sender fragments after a five-second quiet period."""

    def __init__(self, *, quiet_seconds: float = 5, continuation_seconds: float = 120) -> None:
        self.quiet_seconds = max(0.0, quiet_seconds)
        self.continuation_seconds = max(self.quiet_seconds, continuation_seconds)
        self._pending: dict[tuple[str, str], _PendingFragments] = {}
        self._ready: list[AccumulatedMessage] = []
        self._lock = threading.RLock()

    def add(self, message: GroupMessage, *, trace_id: str) -> None:
        key = (message.room_id, message.sender_external_id)
        with self._lock:
            pending = self._pending.get(key)
            if pending and pending.fragments:
                elapsed = (message.sent_at - pending.fragments[-1].sent_at).total_seconds()
                if elapsed > self.continuation_seconds:
                    self._ready.append(self._merge(pending))
                    pending = None
            if pending is None:
                pending = _PendingFragments()
                self._pending[key] = pending
            pending.fragments.append(message)
            pending.trace_id = trace_id
            pending.quiet_deadline = message.sent_at + timedelta(seconds=self.quiet_seconds)

    def flush_due(self, *, at: datetime) -> list[AccumulatedMessage]:
        with self._lock:
            ready = list(self._ready)
            self._ready.clear()
            due_keys = [
                key
                for key, pending in self._pending.items()
                if pending.quiet_deadline is not None and pending.quiet_deadline <= at
            ]
            for key in due_keys:
                ready.append(self._merge(self._pending.pop(key)))
            return sorted(ready, key=lambda item: item.message.sent_at)

    @staticmethod
    def _merge(pending: _PendingFragments) -> AccumulatedMessage:
        fragments = sorted(pending.fragments, key=lambda item: item.sent_at)
        first = fragments[0]
        last = fragments[-1]
        metadata = dict(last.metadata)
        metadata["accumulated_source_message_ids"] = [item.message_id for item in fragments]
        metadata["fragment_count"] = len(fragments)
        merged = GroupMessage(
            room_id=first.room_id,
            conversation_id=first.conversation_id,
            sender_external_id=first.sender_external_id,
            sender_name=first.sender_name,
            text="\n".join(item.text.strip() for item in fragments if item.text.strip()),
            message_id=last.message_id,
            sent_at=last.sent_at,
            quoted_message_id=next(
                (item.quoted_message_id for item in reversed(fragments) if item.quoted_message_id),
                None,
            ),
            channel=first.channel,
            metadata=metadata,
        )
        return AccumulatedMessage(message=merged, trace_id=pending.trace_id)


__all__ = ["AccumulatedMessage", "MessageAccumulator"]
