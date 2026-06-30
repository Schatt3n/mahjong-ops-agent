from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Protocol

from .models import DEFAULT_TZ
from .workflow_models import GameRequirement, ToolResult, UserMessage, WorkflowTurn


@dataclass(slots=True)
class ShortTermMemoryRecord:
    conversation_id: str
    sender_id: str
    user_message: UserMessage
    system_reply: str | None = None
    game_requirement: GameRequirement | None = None
    tool_results: list[ToolResult] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(DEFAULT_TZ))
    metadata: dict[str, object] = field(default_factory=dict)

    def to_workflow_turn(self) -> WorkflowTurn:
        return WorkflowTurn(
            user_message=self.user_message,
            system_reply=self.system_reply,
            game_requirement=self.game_requirement,
            tool_results=list(self.tool_results),
            at=self.created_at,
        )


class ShortTermMemoryStore(Protocol):
    def load(
        self,
        conversation_id: str,
        sender_id: str,
        now: datetime | None = None,
        limit: int | None = None,
    ) -> list[ShortTermMemoryRecord]:
        ...

    def append(self, record: ShortTermMemoryRecord, now: datetime | None = None) -> None:
        ...

    def clear(self, conversation_id: str, sender_id: str | None = None) -> int:
        ...


class InMemoryShortTermMemoryStore:
    def __init__(self, ttl_seconds: int = 30 * 60, max_records_per_scope: int = 20) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_records_per_scope = max_records_per_scope
        self._records: dict[tuple[str, str], list[ShortTermMemoryRecord]] = {}

    def load(
        self,
        conversation_id: str,
        sender_id: str,
        now: datetime | None = None,
        limit: int | None = None,
    ) -> list[ShortTermMemoryRecord]:
        key = (conversation_id, sender_id)
        effective_now = now or datetime.now(DEFAULT_TZ)
        self._records[key] = self._alive_records(self._records.get(key, []), effective_now)
        records = list(self._records[key])
        if limit is not None and limit >= 0:
            records = records[-limit:]
        return records

    def append(self, record: ShortTermMemoryRecord, now: datetime | None = None) -> None:
        key = (record.conversation_id, record.sender_id)
        effective_now = now or datetime.now(DEFAULT_TZ)
        records = self._alive_records(self._records.get(key, []), effective_now)
        records.append(record)
        self._records[key] = records[-self.max_records_per_scope :]

    def clear(self, conversation_id: str, sender_id: str | None = None) -> int:
        if sender_id is not None:
            key = (conversation_id, sender_id)
            removed = len(self._records.get(key, []))
            self._records.pop(key, None)
            return removed

        keys = [key for key in self._records if key[0] == conversation_id]
        removed = sum(len(self._records.get(key, [])) for key in keys)
        for key in keys:
            self._records.pop(key, None)
        return removed

    def _alive_records(
        self,
        records: list[ShortTermMemoryRecord],
        now: datetime,
    ) -> list[ShortTermMemoryRecord]:
        if self.ttl_seconds <= 0:
            return list(records)
        cutoff = now - timedelta(seconds=self.ttl_seconds)
        return [record for record in records if record.created_at >= cutoff]


def summarize_short_memory(records: list[ShortTermMemoryRecord]) -> str | None:
    if not records:
        return None
    lines: list[str] = []
    for record in records[-6:]:
        user_text = record.user_message.text.strip()
        if user_text:
            lines.append(f"用户：{user_text}")
        if record.system_reply:
            lines.append(f"老板建议回复：{record.system_reply.strip()}")
    return "\n".join(lines) if lines else None
