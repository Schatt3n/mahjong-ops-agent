from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from .models import Message


@dataclass(slots=True)
class CachedInputGateResult:
    final_text: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class InputGateDecision:
    """Decision made before a message enters the controlled workflow."""

    accepted: bool
    scope: str
    source_message_id: str
    tenant_id: str = "default"
    sequence: int | None = None
    expected_sequence: int | None = None
    duplicate: bool = False
    out_of_order: bool = False
    waiting_for_sequence: bool = False
    in_progress: bool = False
    reason: str = ""
    cached_result: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "tenant_id": self.tenant_id,
            "scope": self.scope,
            "source_message_id": self.source_message_id,
            "sequence": self.sequence,
            "expected_sequence": self.expected_sequence,
            "duplicate": self.duplicate,
            "out_of_order": self.out_of_order,
            "waiting_for_sequence": self.waiting_for_sequence,
            "in_progress": self.in_progress,
            "reason": self.reason,
            "has_cached_result": self.cached_result is not None,
        }


class InputGate(Protocol):
    def begin(self, message: Message, *, trace_id: str, now: datetime) -> InputGateDecision:
        ...

    def complete(self, message: Message, result: Any, *, trace_id: str, now: datetime) -> None:
        ...

    def fail(self, message: Message, *, trace_id: str, now: datetime) -> None:
        ...


class InMemoryInputGate:
    """Process-local idempotency and ordering gate for controlled workflows.

    The gate deliberately does not understand Mahjong business semantics. It
    only protects the workflow entrance by source message id and optional
    per-conversation sequence.
    """

    def __init__(self) -> None:
        self._completed_by_source: dict[tuple[str, str], Any] = {}
        self._inflight_by_source: dict[tuple[str, str], InputGateDecision] = {}
        self._last_sequence_by_scope: dict[tuple[str, str], int] = {}
        self._source_scope_sequence: dict[tuple[str, str], tuple[str, int | None]] = {}

    def begin(self, message: Message, *, trace_id: str, now: datetime) -> InputGateDecision:
        tenant_id = _tenant_id(message)
        scope = _scope(message)
        source_message_id = _source_message_id(message)
        sequence = _sequence(message)
        source_key = (tenant_id, source_message_id)
        scope_key = (tenant_id, scope)

        cached_result = self._completed_by_source.get(source_key)
        if cached_result is not None:
            return InputGateDecision(
                accepted=False,
                tenant_id=tenant_id,
                scope=scope,
                source_message_id=source_message_id,
                sequence=sequence,
                duplicate=True,
                reason="source_message_id 已完成，直接复用首轮处理结果。",
                cached_result=cached_result,
            )

        inflight = self._inflight_by_source.get(source_key)
        if inflight is not None:
            return InputGateDecision(
                accepted=False,
                tenant_id=tenant_id,
                scope=scope,
                source_message_id=source_message_id,
                sequence=sequence,
                duplicate=True,
                in_progress=True,
                reason="source_message_id 正在处理中，拒绝重复进入 workflow。",
            )

        if sequence is not None:
            last_sequence = self._last_sequence_by_scope.get(scope_key, 0)
            expected = last_sequence + 1
            if sequence <= last_sequence:
                return InputGateDecision(
                    accepted=False,
                    tenant_id=tenant_id,
                    scope=scope,
                    source_message_id=source_message_id,
                    sequence=sequence,
                    expected_sequence=expected,
                    duplicate=True,
                    out_of_order=True,
                    reason="消息 sequence 已落后于会话已处理进度，拒绝重复或过期消息。",
                )
            if sequence > expected:
                return InputGateDecision(
                    accepted=False,
                    tenant_id=tenant_id,
                    scope=scope,
                    source_message_id=source_message_id,
                    sequence=sequence,
                    expected_sequence=expected,
                    out_of_order=True,
                    waiting_for_sequence=True,
                    reason="消息 sequence 超前，等待前序消息处理完成后再进入 workflow。",
                )

        decision = InputGateDecision(
            accepted=True,
            tenant_id=tenant_id,
            scope=scope,
            source_message_id=source_message_id,
            sequence=sequence,
            expected_sequence=sequence,
            reason="消息通过入口幂等和顺序检查。",
        )
        self._inflight_by_source[source_key] = decision
        self._source_scope_sequence[source_key] = (scope, sequence)
        return decision

    def complete(self, message: Message, result: Any, *, trace_id: str, now: datetime) -> None:
        tenant_id = _tenant_id(message)
        source_message_id = _source_message_id(message)
        source_key = (tenant_id, source_message_id)
        scope, sequence = self._source_scope_sequence.get(source_key, (_scope(message), _sequence(message)))
        self._completed_by_source[source_key] = result
        self._inflight_by_source.pop(source_key, None)
        self._source_scope_sequence.pop(source_key, None)
        if sequence is not None:
            scope_key = (tenant_id, scope)
            previous = self._last_sequence_by_scope.get(scope_key, 0)
            if sequence == previous + 1:
                self._last_sequence_by_scope[scope_key] = sequence

    def fail(self, message: Message, *, trace_id: str, now: datetime) -> None:
        tenant_id = _tenant_id(message)
        source_key = (tenant_id, _source_message_id(message))
        self._inflight_by_source.pop(source_key, None)
        self._source_scope_sequence.pop(source_key, None)


class SQLiteInputGate:
    """SQLite-backed input gate for local production deployments."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def begin(self, message: Message, *, trace_id: str, now: datetime) -> InputGateDecision:
        tenant_id = _tenant_id(message)
        scope = _scope(message)
        source_message_id = _source_message_id(message)
        sequence = _sequence(message)
        now_text = now.isoformat()
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            source_row = self._get_source_row(conn, tenant_id, source_message_id)
            if source_row is not None:
                decision = self._decision_for_existing_source(
                    source_row,
                    tenant_id=tenant_id,
                    scope=scope,
                    source_message_id=source_message_id,
                    sequence=sequence,
                )
                if decision is not None:
                    conn.execute("COMMIT")
                    return decision

            if sequence is not None:
                claimed = self._get_sequence_row(conn, tenant_id, scope, sequence, source_message_id)
                if claimed is not None:
                    conn.execute("COMMIT")
                    return InputGateDecision(
                        accepted=False,
                        tenant_id=tenant_id,
                        scope=scope,
                        source_message_id=source_message_id,
                        sequence=sequence,
                        duplicate=True,
                        out_of_order=True,
                        in_progress=str(claimed["status"]) == "inflight",
                        reason="会话 sequence 已被其他 source_message_id 占用，拒绝重复或冲突消息。",
                    )
                last_sequence = self._last_sequence(conn, tenant_id, scope)
                expected = last_sequence + 1
                if sequence <= last_sequence:
                    conn.execute("COMMIT")
                    return InputGateDecision(
                        accepted=False,
                        tenant_id=tenant_id,
                        scope=scope,
                        source_message_id=source_message_id,
                        sequence=sequence,
                        expected_sequence=expected,
                        duplicate=True,
                        out_of_order=True,
                        reason="消息 sequence 已落后于会话已处理进度，拒绝重复或过期消息。",
                    )
                if sequence > expected:
                    conn.execute("COMMIT")
                    return InputGateDecision(
                        accepted=False,
                        tenant_id=tenant_id,
                        scope=scope,
                        source_message_id=source_message_id,
                        sequence=sequence,
                        expected_sequence=expected,
                        out_of_order=True,
                        waiting_for_sequence=True,
                        reason="消息 sequence 超前，等待前序消息处理完成后再进入 workflow。",
                    )

            self._upsert_inflight(
                conn,
                tenant_id=tenant_id,
                source_message_id=source_message_id,
                scope=scope,
                sequence=sequence,
                trace_id=trace_id,
                now_text=now_text,
            )
            conn.execute("COMMIT")
            return InputGateDecision(
                accepted=True,
                tenant_id=tenant_id,
                scope=scope,
                source_message_id=source_message_id,
                sequence=sequence,
                expected_sequence=sequence,
                reason="消息通过 SQLite 入口幂等和顺序检查。",
            )
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    def complete(self, message: Message, result: Any, *, trace_id: str, now: datetime) -> None:
        tenant_id = _tenant_id(message)
        source_message_id = _source_message_id(message)
        now_text = now.isoformat()
        result_payload = _cached_result_payload(result)
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = self._get_source_row(conn, tenant_id, source_message_id)
            scope = str(row["scope"]) if row is not None else _scope(message)
            sequence = int(row["sequence"]) if row is not None and row["sequence"] is not None else _sequence(message)
            if row is None:
                self._insert_message(
                    conn,
                    tenant_id=tenant_id,
                    source_message_id=source_message_id,
                    scope=scope,
                    sequence=sequence,
                    status="completed",
                    trace_id=trace_id,
                    now_text=now_text,
                    result_payload=result_payload,
                )
            else:
                conn.execute(
                    """
                    UPDATE controlled_input_gate_messages
                    SET status = ?, trace_id = ?, final_text = ?, result_json = ?, updated_at = ?
                    WHERE tenant_id = ? AND source_message_id = ?
                    """,
                    (
                        "completed",
                        trace_id,
                        result_payload.get("final_text") or "",
                        _dump_json(result_payload),
                        now_text,
                        tenant_id,
                        source_message_id,
                    ),
                )
            if sequence is not None:
                last_sequence = self._last_sequence(conn, tenant_id, scope)
                if sequence == last_sequence + 1:
                    conn.execute(
                        """
                        INSERT INTO controlled_input_gate_offsets(tenant_id, scope, last_sequence, updated_at)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(tenant_id, scope) DO UPDATE SET
                            last_sequence = excluded.last_sequence,
                            updated_at = excluded.updated_at
                        """,
                        (tenant_id, scope, sequence, now_text),
                    )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    def fail(self, message: Message, *, trace_id: str, now: datetime) -> None:
        tenant_id = _tenant_id(message)
        source_message_id = _source_message_id(message)
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                UPDATE controlled_input_gate_messages
                SET status = ?, trace_id = ?, updated_at = ?
                WHERE tenant_id = ? AND source_message_id = ? AND status = ?
                """,
                ("failed", trace_id, now.isoformat(), tenant_id, source_message_id, "inflight"),
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS controlled_input_gate_messages (
                    tenant_id TEXT NOT NULL,
                    source_message_id TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    sequence INTEGER,
                    status TEXT NOT NULL,
                    trace_id TEXT,
                    final_text TEXT,
                    result_json TEXT,
                    reason TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, source_message_id)
                );

                CREATE UNIQUE INDEX IF NOT EXISTS idx_controlled_input_gate_scope_sequence
                    ON controlled_input_gate_messages(tenant_id, scope, sequence)
                    WHERE sequence IS NOT NULL;

                CREATE INDEX IF NOT EXISTS idx_controlled_input_gate_status
                    ON controlled_input_gate_messages(tenant_id, scope, status, sequence);

                CREATE TABLE IF NOT EXISTS controlled_input_gate_offsets (
                    tenant_id TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    last_sequence INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, scope)
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        return conn

    def _get_source_row(
        self,
        conn: sqlite3.Connection,
        tenant_id: str,
        source_message_id: str,
    ) -> sqlite3.Row | None:
        return conn.execute(
            """
            SELECT *
            FROM controlled_input_gate_messages
            WHERE tenant_id = ? AND source_message_id = ?
            """,
            (tenant_id, source_message_id),
        ).fetchone()

    def _get_sequence_row(
        self,
        conn: sqlite3.Connection,
        tenant_id: str,
        scope: str,
        sequence: int,
        current_source_message_id: str,
    ) -> sqlite3.Row | None:
        return conn.execute(
            """
            SELECT *
            FROM controlled_input_gate_messages
            WHERE tenant_id = ? AND scope = ? AND sequence = ? AND source_message_id != ?
            """,
            (tenant_id, scope, sequence, current_source_message_id),
        ).fetchone()

    def _last_sequence(self, conn: sqlite3.Connection, tenant_id: str, scope: str) -> int:
        row = conn.execute(
            """
            SELECT last_sequence
            FROM controlled_input_gate_offsets
            WHERE tenant_id = ? AND scope = ?
            """,
            (tenant_id, scope),
        ).fetchone()
        return int(row["last_sequence"]) if row else 0

    def _decision_for_existing_source(
        self,
        row: sqlite3.Row,
        *,
        tenant_id: str,
        scope: str,
        source_message_id: str,
        sequence: int | None,
    ) -> InputGateDecision | None:
        status = str(row["status"])
        stored_sequence = int(row["sequence"]) if row["sequence"] is not None else sequence
        stored_scope = str(row["scope"] or scope)
        if status == "completed":
            return InputGateDecision(
                accepted=False,
                tenant_id=tenant_id,
                scope=stored_scope,
                source_message_id=source_message_id,
                sequence=stored_sequence,
                duplicate=True,
                reason="source_message_id 已完成，直接复用 SQLite 中的首轮处理结果。",
                cached_result=_cached_result_from_row(row),
            )
        if status == "inflight":
            return InputGateDecision(
                accepted=False,
                tenant_id=tenant_id,
                scope=stored_scope,
                source_message_id=source_message_id,
                sequence=stored_sequence,
                duplicate=True,
                in_progress=True,
                reason="source_message_id 正在处理中，拒绝重复进入 workflow。",
            )
        if status == "failed":
            return None
        return None

    def _upsert_inflight(
        self,
        conn: sqlite3.Connection,
        *,
        tenant_id: str,
        source_message_id: str,
        scope: str,
        sequence: int | None,
        trace_id: str,
        now_text: str,
    ) -> None:
        existing = self._get_source_row(conn, tenant_id, source_message_id)
        if existing is None:
            self._insert_message(
                conn,
                tenant_id=tenant_id,
                source_message_id=source_message_id,
                scope=scope,
                sequence=sequence,
                status="inflight",
                trace_id=trace_id,
                now_text=now_text,
                result_payload={},
            )
            return
        conn.execute(
            """
            UPDATE controlled_input_gate_messages
            SET scope = ?, sequence = ?, status = ?, trace_id = ?,
                final_text = NULL, result_json = NULL, updated_at = ?
            WHERE tenant_id = ? AND source_message_id = ?
            """,
            (scope, sequence, "inflight", trace_id, now_text, tenant_id, source_message_id),
        )

    def _insert_message(
        self,
        conn: sqlite3.Connection,
        *,
        tenant_id: str,
        source_message_id: str,
        scope: str,
        sequence: int | None,
        status: str,
        trace_id: str,
        now_text: str,
        result_payload: dict[str, Any],
    ) -> None:
        conn.execute(
            """
            INSERT INTO controlled_input_gate_messages(
                tenant_id, source_message_id, scope, sequence, status,
                trace_id, final_text, result_json, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                tenant_id,
                source_message_id,
                scope,
                sequence,
                status,
                trace_id,
                result_payload.get("final_text") or "",
                _dump_json(result_payload) if result_payload else None,
                now_text,
                now_text,
            ),
        )


def _tenant_id(message: Message) -> str:
    value = message.metadata.get("tenant_id") or message.metadata.get("store_id") or "default"
    return str(value).strip() or "default"


def _scope(message: Message) -> str:
    value = message.metadata.get("conversation_id") or message.channel_id
    return str(value).strip() or "default"


def _source_message_id(message: Message) -> str:
    value = (
        message.metadata.get("source_message_id")
        or message.metadata.get("message_id")
        or message.metadata.get("platform_message_id")
        or message.id
    )
    return str(value).strip() or message.id


def _sequence(message: Message) -> int | None:
    value = message.metadata.get("sequence")
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _cached_result_payload(result: Any) -> dict[str, Any]:
    final_text = str(getattr(result, "final_text", "") or "")
    payload: dict[str, Any] = {
        "final_text": final_text,
        "result_type": type(result).__name__,
    }
    run = getattr(result, "run", None)
    validated_action = getattr(run, "validated_action", None)
    if validated_action is not None:
        payload["effective_action"] = getattr(getattr(validated_action, "effective_action", None), "value", None) or str(
            getattr(validated_action, "effective_action", "")
        )
        payload["validation_code"] = getattr(validated_action, "code", None)
    return payload


def _cached_result_from_row(row: sqlite3.Row) -> CachedInputGateResult:
    payload = _loads_dict(str(row["result_json"] or "{}"))
    final_text = str(row["final_text"] or payload.get("final_text") or "")
    return CachedInputGateResult(final_text=final_text, payload=payload)


def _dump_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _loads_dict(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}
