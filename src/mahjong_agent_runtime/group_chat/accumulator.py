"""实现群聊场景中的accumulator能力。

"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .models import GroupMessage


@dataclass(slots=True)
class AccumulatedMessage:
    """静默窗口结束后提交给会话路由器的合并群消息。"""

    message: GroupMessage
    trace_id: str


@dataclass(slots=True)
class _PendingFragments:
    """保存单个群成员尚未结算的碎片消息及静默截止时间。"""

    fragments: list[GroupMessage] = field(default_factory=list)
    trace_id: str = ""
    quiet_deadline: datetime | None = None


class MessageAccumulator:
    """在五秒静默期后，合并同一发送者的消息片段。"""

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

    def invalidate(self, *, room_id: str, source_message_id: str) -> bool:
        """在撤回的源消息进入语义处理之前将其移除。"""

        removed = False
        with self._lock:
            for key, pending in list(self._pending.items()):
                if key[0] != room_id:
                    continue
                retained = [item for item in pending.fragments if item.message_id != source_message_id]
                if len(retained) == len(pending.fragments):
                    continue
                removed = True
                if retained:
                    pending.fragments = retained
                    pending.quiet_deadline = retained[-1].sent_at + timedelta(seconds=self.quiet_seconds)
                else:
                    self._pending.pop(key, None)
            retained_ready: list[AccumulatedMessage] = []
            for item in self._ready:
                source_ids = item.message.metadata.get("accumulated_source_message_ids") or [item.message.message_id]
                if item.message.room_id == room_id and source_message_id in source_ids:
                    removed = True
                    continue
                retained_ready.append(item)
            self._ready = retained_ready
        return removed

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
