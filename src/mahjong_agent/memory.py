from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from .models import DEFAULT_TZ
from .observability import to_trace_payload
from .workflow_models import (
    GameRequirement,
    SlotSource,
    SlotValue,
    ToolCallRequest,
    ToolExecutionMode,
    ToolName,
    ToolResult,
    UserMessage,
    WorkflowTurn,
)


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


class SQLiteShortTermMemoryStore:
    """SQLite-backed short-term memory for local production deployments."""

    def __init__(self, path: str | Path, ttl_seconds: int = 30 * 60, max_records_per_scope: int = 20) -> None:
        self.path = Path(path)
        self.ttl_seconds = ttl_seconds
        self.max_records_per_scope = max_records_per_scope
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def load(
        self,
        conversation_id: str,
        sender_id: str,
        now: datetime | None = None,
        limit: int | None = None,
    ) -> list[ShortTermMemoryRecord]:
        effective_now = now or datetime.now(DEFAULT_TZ)
        self._prune_expired(conversation_id=conversation_id, sender_id=sender_id, now=effective_now)
        query_limit = limit if limit is not None and limit >= 0 else self.max_records_per_scope
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM controlled_short_term_memory
                WHERE conversation_id = ? AND sender_id = ?
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (conversation_id, sender_id, query_limit),
            ).fetchall()
        return [_record_from_row(row) for row in reversed(rows)]

    def append(self, record: ShortTermMemoryRecord, now: datetime | None = None) -> None:
        effective_now = now or datetime.now(DEFAULT_TZ)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO controlled_short_term_memory(
                    conversation_id, sender_id, user_message_json, system_reply,
                    game_requirement_json, tool_results_json, metadata_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.conversation_id,
                    record.sender_id,
                    _dump_json(_user_message_payload(record.user_message)),
                    record.system_reply,
                    _dump_json(_game_requirement_payload(record.game_requirement))
                    if record.game_requirement is not None
                    else None,
                    _dump_json([_tool_result_payload(item) for item in record.tool_results]),
                    _dump_json(to_trace_payload(record.metadata)),
                    record.created_at.isoformat(),
                ),
            )
        self._prune_expired(conversation_id=record.conversation_id, sender_id=record.sender_id, now=effective_now)
        self._prune_overflow(conversation_id=record.conversation_id, sender_id=record.sender_id)

    def clear(self, conversation_id: str, sender_id: str | None = None) -> int:
        with self._connect() as conn:
            if sender_id is not None:
                cursor = conn.execute(
                    """
                    DELETE FROM controlled_short_term_memory
                    WHERE conversation_id = ? AND sender_id = ?
                    """,
                    (conversation_id, sender_id),
                )
                return int(cursor.rowcount or 0)
            cursor = conn.execute(
                """
                DELETE FROM controlled_short_term_memory
                WHERE conversation_id = ?
                """,
                (conversation_id,),
            )
            return int(cursor.rowcount or 0)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS controlled_short_term_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    sender_id TEXT NOT NULL,
                    user_message_json TEXT NOT NULL,
                    system_reply TEXT,
                    game_requirement_json TEXT,
                    tool_results_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_controlled_short_term_memory_scope
                    ON controlled_short_term_memory(conversation_id, sender_id, created_at, id);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _prune_expired(self, *, conversation_id: str, sender_id: str, now: datetime) -> None:
        if self.ttl_seconds <= 0:
            return
        cutoff = now - timedelta(seconds=self.ttl_seconds)
        with self._connect() as conn:
            conn.execute(
                """
                DELETE FROM controlled_short_term_memory
                WHERE conversation_id = ? AND sender_id = ? AND created_at < ?
                """,
                (conversation_id, sender_id, cutoff.isoformat()),
            )

    def _prune_overflow(self, *, conversation_id: str, sender_id: str) -> None:
        if self.max_records_per_scope <= 0:
            return
        with self._connect() as conn:
            conn.execute(
                """
                DELETE FROM controlled_short_term_memory
                WHERE conversation_id = ?
                  AND sender_id = ?
                  AND id NOT IN (
                    SELECT id
                    FROM controlled_short_term_memory
                    WHERE conversation_id = ? AND sender_id = ?
                    ORDER BY created_at DESC, id DESC
                    LIMIT ?
                  )
                """,
                (conversation_id, sender_id, conversation_id, sender_id, self.max_records_per_scope),
            )


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


def _user_message_payload(message: UserMessage) -> dict[str, Any]:
    return {
        "text": message.text,
        "sender_id": message.sender_id,
        "sender_name": message.sender_name,
        "conversation_id": message.conversation_id,
        "trace_id": message.trace_id,
        "message_id": message.message_id,
        "channel_type": message.channel_type.value,
        "sent_at": message.sent_at.isoformat(),
        "modalities": list(message.modalities),
        "metadata": to_trace_payload(message.metadata),
    }


def _game_requirement_payload(requirement: GameRequirement | None) -> dict[str, Any] | None:
    if requirement is None:
        return None
    return requirement.to_prompt_dict()


def _tool_result_payload(result: ToolResult) -> dict[str, Any]:
    return {
        "tool_name": result.request.tool_name.value,
        "arguments": to_trace_payload(result.request.arguments),
        "risk_level": result.request.risk_level.value,
        "execution_mode": result.request.execution_mode.value,
        "idempotency_key": result.request.idempotency_key,
        "reason": result.request.reason,
        "called": result.called,
        "allowed": result.allowed,
        "result": to_trace_payload(result.result),
        "error": result.error,
        "deduplicated": result.deduplicated,
    }


def _record_from_row(row: sqlite3.Row) -> ShortTermMemoryRecord:
    user_message = _user_message_from_payload(_loads_dict(str(row["user_message_json"] or "{}")))
    return ShortTermMemoryRecord(
        conversation_id=str(row["conversation_id"]),
        sender_id=str(row["sender_id"]),
        user_message=user_message,
        system_reply=str(row["system_reply"]) if row["system_reply"] is not None else None,
        game_requirement=_game_requirement_from_payload(_loads_dict(str(row["game_requirement_json"] or "{}")))
        if row["game_requirement_json"]
        else None,
        tool_results=[
            _tool_result_from_payload(item)
            for item in _loads_list(str(row["tool_results_json"] or "[]"))
            if isinstance(item, dict)
        ],
        created_at=_parse_datetime(str(row["created_at"])) or datetime.now(DEFAULT_TZ),
        metadata=_loads_dict(str(row["metadata_json"] or "{}")),
    )


def _user_message_from_payload(payload: dict[str, Any]) -> UserMessage:
    return UserMessage(
        text=str(payload.get("text") or ""),
        sender_id=str(payload.get("sender_id") or ""),
        sender_name=str(payload.get("sender_name") or ""),
        conversation_id=str(payload.get("conversation_id") or ""),
        trace_id=str(payload.get("trace_id") or ""),
        message_id=str(payload.get("message_id") or ""),
        channel_type=str(payload.get("channel_type") or "manual"),
        sent_at=_parse_datetime(str(payload.get("sent_at") or "")) or datetime.now(DEFAULT_TZ),
        modalities=[str(item) for item in payload.get("modalities") or ["text"]],
        metadata=dict(payload.get("metadata") or {}) if isinstance(payload.get("metadata"), dict) else {},
    )


def _game_requirement_from_payload(payload: dict[str, Any]) -> GameRequirement:
    requirement = GameRequirement(
        seats_total=int(payload.get("seats_total") or 4),
        organizer_id=str(payload.get("organizer_id")) if payload.get("organizer_id") is not None else None,
        organizer_name=str(payload.get("organizer_name")) if payload.get("organizer_name") is not None else None,
        candidate_composition_preference=dict(payload.get("candidate_composition_preference") or {})
        if isinstance(payload.get("candidate_composition_preference"), dict)
        else {},
        notes=[str(item) for item in payload.get("notes") or []],
    )
    slots = payload.get("slots") if isinstance(payload.get("slots"), dict) else {}
    for name, slot_payload in slots.items():
        if isinstance(slot_payload, dict):
            requirement.set_slot(_slot_from_payload(str(name), slot_payload), prefer_confirmed=False)
    return requirement


def _slot_from_payload(name: str, payload: dict[str, Any]) -> SlotValue:
    return SlotValue(
        name=name,
        value=payload.get("value"),
        source=str(payload.get("source") or SlotSource.UNKNOWN.value),
        confidence=float(payload.get("confidence") or 0.0),
        confirmed=bool(payload.get("confirmed")),
        needs_confirmation=bool(payload.get("needs_confirmation")),
        evidence=str(payload.get("evidence")) if payload.get("evidence") is not None else None,
        metadata=dict(payload.get("metadata") or {}) if isinstance(payload.get("metadata"), dict) else {},
    )


def _tool_result_from_payload(payload: dict[str, Any]) -> ToolResult:
    request = ToolCallRequest(
        tool_name=str(payload.get("tool_name") or ToolName.UNKNOWN.value),
        arguments=dict(payload.get("arguments") or {}) if isinstance(payload.get("arguments"), dict) else {},
        risk_level=str(payload.get("risk_level") or "low"),
        execution_mode=str(payload.get("execution_mode") or ToolExecutionMode.READ_ONLY.value),
        idempotency_key=str(payload.get("idempotency_key")) if payload.get("idempotency_key") is not None else None,
        reason=str(payload.get("reason") or ""),
    )
    return ToolResult(
        request=request,
        called=bool(payload.get("called")),
        allowed=bool(payload.get("allowed")),
        result=dict(payload.get("result") or {}) if isinstance(payload.get("result"), dict) else {},
        error=str(payload.get("error")) if payload.get("error") is not None else None,
        deduplicated=bool(payload.get("deduplicated")),
    )


def _parse_datetime(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=DEFAULT_TZ)
    return parsed


def _dump_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _loads_dict(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _loads_list(text: str) -> list[Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []
