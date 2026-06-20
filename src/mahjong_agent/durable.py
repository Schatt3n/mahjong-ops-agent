from __future__ import annotations

import json
import hashlib
import re
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterator

from .core import InMemoryStore
from .models import (
    DEFAULT_TZ,
    ChannelType,
    CustomerProfile,
    GameRequest,
    GameStatus,
    Invitation,
    InvitationStatus,
    Message,
    PlayPreference,
    RoomHold,
    RoomHoldStatus,
)
from .runtime import (
    AgentRuntime,
    ContextTurn,
    ConversationContext,
    RuntimeResult,
)


@dataclass(slots=True)
class IncomingEnvelope:
    message: Message
    tenant_id: str = "default"
    source_message_id: str | None = None
    sequence: int | None = None
    received_at: datetime = field(default_factory=lambda: datetime.now(DEFAULT_TZ))

    @property
    def conversation_id(self) -> str:
        return f"{self.tenant_id}:{self.message.channel_id}"

    @property
    def trace_id(self) -> str:
        return self.source_message_id or self.message.id


@dataclass(slots=True)
class StoredMessage:
    tenant_id: str
    source_message_id: str
    conversation_id: str
    sequence: int
    message_json: dict[str, Any]
    status: str
    result_json: dict[str, Any] | None
    semantic_fingerprint: str | None = None
    semantic_duplicate_of: str | None = None


@dataclass(slots=True)
class DurableProcessResult:
    status: str
    tenant_id: str
    source_message_id: str
    conversation_id: str
    sequence: int
    runtime_result: RuntimeResult | None
    processed_results: list[RuntimeResult] = field(default_factory=list)
    duplicate: bool = False
    waiting_for_sequence: bool = False
    outbox_created: int = 0

    def to_dict(self) -> dict[str, Any]:
        if self.runtime_result:
            data = self.runtime_result.to_dict()
        else:
            data = {
                "action": "queued",
                "reply_text": "",
                "confidence": 1.0,
                "should_reply": False,
                "needs_human_review": False,
                "game_id": None,
                "draft_group_post": None,
                "invitation_drafts": [],
                "notes": ["消息已持久化，正在等待前序消息。"],
                "runtime": {
                    "ok": True,
                    "latency_ms": 0,
                    "timed_out": False,
                    "error": None,
                    "context": None,
                },
            }
        data["durable"] = {
            "status": self.status,
            "tenant_id": self.tenant_id,
            "source_message_id": self.source_message_id,
            "conversation_id": self.conversation_id,
            "sequence": self.sequence,
            "duplicate": self.duplicate,
            "waiting_for_sequence": self.waiting_for_sequence,
            "processed_message_count": len(self.processed_results),
            "outbox_created": self.outbox_created,
        }
        return data


class SQLiteDurableStore:
    def __init__(self, path: Path | str, semantic_dedupe_window_seconds: float = 20.0) -> None:
        self.path = Path(path)
        self.semantic_dedupe_window_seconds = semantic_dedupe_window_seconds
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        try:
            yield conn
        finally:
            conn.close()

    def init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS inbound_messages (
                    tenant_id TEXT NOT NULL,
                    source_message_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    message_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result_json TEXT,
                    semantic_fingerprint TEXT,
                    semantic_duplicate_of TEXT,
                    received_at TEXT NOT NULL,
                    processing_started_at TEXT,
                    lease_until TEXT,
                    processed_at TEXT,
                    error TEXT,
                    PRIMARY KEY (tenant_id, source_message_id),
                    UNIQUE (conversation_id, sequence)
                );

                CREATE INDEX IF NOT EXISTS idx_inbound_conversation_sequence
                    ON inbound_messages(conversation_id, sequence);

                CREATE INDEX IF NOT EXISTS idx_inbound_semantic_fingerprint
                    ON inbound_messages(tenant_id, conversation_id, semantic_fingerprint, received_at);

                CREATE TABLE IF NOT EXISTS conversation_offsets (
                    conversation_id TEXT PRIMARY KEY,
                    last_sequence INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    source_message_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_audit_trace
                    ON audit_events(trace_id);

                CREATE TABLE IF NOT EXISTS outbox_events (
                    id TEXT PRIMARY KEY,
                    trace_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    output_channel TEXT NOT NULL DEFAULT 'console',
                    target_type TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    original_target_id TEXT,
                    message_text TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    sent_at TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS agent_state_snapshots (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    state_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            self._ensure_column(conn, "outbox_events", "output_channel", "TEXT NOT NULL DEFAULT 'console'")
            self._ensure_column(conn, "outbox_events", "original_target_id", "TEXT")
            self._ensure_column(conn, "outbox_events", "attempt_count", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "outbox_events", "sent_at", "TEXT")
            self._ensure_column(conn, "outbox_events", "error", "TEXT")
            self._ensure_column(conn, "inbound_messages", "semantic_fingerprint", "TEXT")
            self._ensure_column(conn, "inbound_messages", "semantic_duplicate_of", "TEXT")

    def _ensure_column(
        self,
        conn: sqlite3.Connection,
        table_name: str,
        column_name: str,
        column_definition: str,
    ) -> None:
        columns = {
            row["name"]
            for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        }
        if column_name not in columns:
            conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}")

    def reset_all(self) -> None:
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            for table in [
                "inbound_messages",
                "conversation_offsets",
                "audit_events",
                "outbox_events",
                "agent_state_snapshots",
            ]:
                conn.execute(f"DELETE FROM {table}")
            conn.execute("COMMIT")

    def insert_envelope(self, envelope: IncomingEnvelope) -> StoredMessage:
        source_id = envelope.source_message_id or envelope.message.id
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT * FROM inbound_messages
                WHERE tenant_id = ? AND source_message_id = ?
                """,
                (envelope.tenant_id, source_id),
            ).fetchone()
            if row:
                conn.execute("COMMIT")
                return self._stored_from_row(row)

            if envelope.sequence is None:
                sequence = (
                    conn.execute(
                        """
                        SELECT COALESCE(MAX(sequence), 0) + 1
                        FROM inbound_messages
                        WHERE conversation_id = ?
                        """,
                        (envelope.conversation_id,),
                    ).fetchone()[0]
                    or 1
                )
            else:
                sequence = envelope.sequence

            now = _dt(datetime.now(DEFAULT_TZ))
            message_json = _message_to_dict(envelope.message)
            semantic_fingerprint = self._semantic_fingerprint(envelope)
            semantic_duplicate_of = (
                self._find_semantic_duplicate_in_tx(
                    conn,
                    envelope=envelope,
                    source_id=source_id,
                    fingerprint=semantic_fingerprint,
                )
                if semantic_fingerprint
                else None
            )
            conn.execute(
                """
                INSERT OR IGNORE INTO conversation_offsets(conversation_id, last_sequence, updated_at)
                VALUES (?, 0, ?)
                """,
                (envelope.conversation_id, now),
            )
            try:
                conn.execute(
                    """
                    INSERT INTO inbound_messages(
                        tenant_id, source_message_id, conversation_id, sequence,
                        message_json, status, result_json, semantic_fingerprint,
                        semantic_duplicate_of, received_at
                    )
                    VALUES (?, ?, ?, ?, ?, 'pending', NULL, ?, ?, ?)
                    """,
                    (
                        envelope.tenant_id,
                        source_id,
                        envelope.conversation_id,
                        sequence,
                        json.dumps(message_json, ensure_ascii=False),
                        semantic_fingerprint,
                        semantic_duplicate_of,
                        _dt(envelope.received_at),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                conn.execute("ROLLBACK")
                raise ValueError(
                    f"conversation sequence already exists: {envelope.conversation_id}#{sequence}"
                ) from exc

            self._append_audit_in_tx(
                conn,
                trace_id=envelope.trace_id,
                tenant_id=envelope.tenant_id,
                conversation_id=envelope.conversation_id,
                source_message_id=source_id,
                event_type="message_received",
                payload={
                    "sequence": sequence,
                    "message": message_json,
                    "semantic_fingerprint": semantic_fingerprint,
                    "semantic_duplicate_of": semantic_duplicate_of,
                },
            )
            if semantic_duplicate_of:
                self._append_audit_in_tx(
                    conn,
                    trace_id=envelope.trace_id,
                    tenant_id=envelope.tenant_id,
                    conversation_id=envelope.conversation_id,
                    source_message_id=source_id,
                    event_type="semantic_duplicate_detected",
                    payload={
                        "sequence": sequence,
                        "duplicate_of": semantic_duplicate_of,
                        "semantic_fingerprint": semantic_fingerprint,
                        "window_seconds": self.semantic_dedupe_window_seconds,
                    },
                )
            row = conn.execute(
                """
                SELECT * FROM inbound_messages
                WHERE tenant_id = ? AND source_message_id = ?
                """,
                (envelope.tenant_id, source_id),
            ).fetchone()
            conn.execute("COMMIT")
            return self._stored_from_row(row)

    def _semantic_fingerprint(self, envelope: IncomingEnvelope) -> str | None:
        if self.semantic_dedupe_window_seconds <= 0:
            return None
        message = envelope.message
        metadata = message.metadata or {}
        if metadata.get("disable_semantic_dedupe"):
            return None
        normalized_text = _normalize_semantic_text(message.text)
        if not normalized_text:
            return None
        payload = {
            "tenant_id": envelope.tenant_id,
            "conversation_id": envelope.conversation_id,
            "sender_id": message.sender_id,
            "channel_type": message.channel_type.value,
            "text": normalized_text,
            "intent_kind": _semantic_intent_kind(normalized_text),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

    def _find_semantic_duplicate_in_tx(
        self,
        conn: sqlite3.Connection,
        envelope: IncomingEnvelope,
        source_id: str,
        fingerprint: str,
    ) -> str | None:
        cutoff = _dt(envelope.received_at - timedelta(seconds=self.semantic_dedupe_window_seconds))
        row = conn.execute(
            """
            SELECT source_message_id
            FROM inbound_messages
            WHERE tenant_id = ?
              AND conversation_id = ?
              AND source_message_id != ?
              AND semantic_fingerprint = ?
              AND received_at >= ?
              AND status IN ('pending', 'processing', 'processed')
            ORDER BY received_at DESC, sequence DESC
            LIMIT 1
            """,
            (
                envelope.tenant_id,
                envelope.conversation_id,
                source_id,
                fingerprint,
                cutoff,
            ),
        ).fetchone()
        return str(row["source_message_id"]) if row else None

    def get_message(self, tenant_id: str, source_message_id: str) -> StoredMessage | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM inbound_messages
                WHERE tenant_id = ? AND source_message_id = ?
                """,
                (tenant_id, source_message_id),
            ).fetchone()
            return self._stored_from_row(row) if row else None

    def claim_next_ready(
        self,
        conversation_id: str,
        lease_seconds: float,
    ) -> StoredMessage | None:
        now_dt = datetime.now(DEFAULT_TZ)
        now = _dt(now_dt)
        lease_until = _dt(now_dt + timedelta(seconds=lease_seconds))
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                UPDATE inbound_messages
                SET status = 'pending', lease_until = NULL, processing_started_at = NULL
                WHERE conversation_id = ?
                  AND status = 'processing'
                  AND lease_until IS NOT NULL
                  AND lease_until < ?
                """,
                (conversation_id, now),
            )
            offset_row = conn.execute(
                """
                SELECT last_sequence FROM conversation_offsets
                WHERE conversation_id = ?
                """,
                (conversation_id,),
            ).fetchone()
            last_sequence = offset_row["last_sequence"] if offset_row else 0
            expected = last_sequence + 1
            row = conn.execute(
                """
                SELECT * FROM inbound_messages
                WHERE conversation_id = ?
                  AND sequence = ?
                  AND status = 'pending'
                """,
                (conversation_id, expected),
            ).fetchone()
            if not row:
                conn.execute("COMMIT")
                return None
            updated = conn.execute(
                """
                UPDATE inbound_messages
                SET status = 'processing',
                    processing_started_at = ?,
                    lease_until = ?
                WHERE tenant_id = ?
                  AND source_message_id = ?
                  AND status = 'pending'
                """,
                (now, lease_until, row["tenant_id"], row["source_message_id"]),
            ).rowcount
            if updated != 1:
                conn.execute("COMMIT")
                return None
            row = conn.execute(
                """
                SELECT * FROM inbound_messages
                WHERE tenant_id = ? AND source_message_id = ?
                """,
                (row["tenant_id"], row["source_message_id"]),
            ).fetchone()
            self._append_audit_in_tx(
                conn,
                trace_id=row["source_message_id"],
                tenant_id=row["tenant_id"],
                conversation_id=row["conversation_id"],
                source_message_id=row["source_message_id"],
                event_type="message_claimed",
                payload={"sequence": row["sequence"], "lease_until": lease_until},
            )
            conn.execute("COMMIT")
            return self._stored_from_row(row)

    def mark_processed(
        self,
        stored: StoredMessage,
        result: RuntimeResult,
        state_snapshot: dict[str, Any],
    ) -> int:
        result_json = result.to_dict()
        outbox_created = 0
        now = _dt(datetime.now(DEFAULT_TZ))
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                UPDATE inbound_messages
                SET status = 'processed',
                    result_json = ?,
                    processed_at = ?,
                    lease_until = NULL,
                    error = ?
                WHERE tenant_id = ?
                  AND source_message_id = ?
                """,
                (
                    json.dumps(result_json, ensure_ascii=False),
                    now,
                    result.error,
                    stored.tenant_id,
                    stored.source_message_id,
                ),
            )
            conn.execute(
                """
                INSERT INTO conversation_offsets(conversation_id, last_sequence, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(conversation_id) DO UPDATE SET
                    last_sequence = excluded.last_sequence,
                    updated_at = excluded.updated_at
                """,
                (stored.conversation_id, stored.sequence, now),
            )
            self._append_audit_in_tx(
                conn,
                trace_id=stored.source_message_id,
                tenant_id=stored.tenant_id,
                conversation_id=stored.conversation_id,
                source_message_id=stored.source_message_id,
                event_type="decision_made",
                payload=result_json,
            )
            outbox_created += self._insert_outbox_from_result_in_tx(conn, stored, result_json)
            conn.execute(
                """
                INSERT INTO agent_state_snapshots(id, state_json, updated_at)
                VALUES (1, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    state_json = excluded.state_json,
                    updated_at = excluded.updated_at
                """,
                (json.dumps(state_snapshot, ensure_ascii=False), now),
            )
            self._append_audit_in_tx(
                conn,
                trace_id=stored.source_message_id,
                tenant_id=stored.tenant_id,
                conversation_id=stored.conversation_id,
                source_message_id=stored.source_message_id,
                event_type="message_processed",
                payload={
                    "sequence": stored.sequence,
                    "action": result.decision.action.value,
                    "outbox_created": outbox_created,
                },
            )
            conn.execute("COMMIT")
        return outbox_created

    def mark_semantic_duplicate(
        self,
        stored: StoredMessage,
        result: RuntimeResult,
        state_snapshot: dict[str, Any],
    ) -> None:
        result_json = result.to_dict()
        now = _dt(datetime.now(DEFAULT_TZ))
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                UPDATE inbound_messages
                SET status = 'processed',
                    result_json = ?,
                    processed_at = ?,
                    lease_until = NULL,
                    error = NULL
                WHERE tenant_id = ?
                  AND source_message_id = ?
                """,
                (
                    json.dumps(result_json, ensure_ascii=False),
                    now,
                    stored.tenant_id,
                    stored.source_message_id,
                ),
            )
            conn.execute(
                """
                INSERT INTO conversation_offsets(conversation_id, last_sequence, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(conversation_id) DO UPDATE SET
                    last_sequence = excluded.last_sequence,
                    updated_at = excluded.updated_at
                """,
                (stored.conversation_id, stored.sequence, now),
            )
            self._append_audit_in_tx(
                conn,
                trace_id=stored.source_message_id,
                tenant_id=stored.tenant_id,
                conversation_id=stored.conversation_id,
                source_message_id=stored.source_message_id,
                event_type="semantic_duplicate_skipped",
                payload={
                    "sequence": stored.sequence,
                    "duplicate_of": stored.semantic_duplicate_of,
                    "semantic_fingerprint": stored.semantic_fingerprint,
                },
            )
            conn.execute(
                """
                INSERT INTO agent_state_snapshots(id, state_json, updated_at)
                VALUES (1, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    state_json = excluded.state_json,
                    updated_at = excluded.updated_at
                """,
                (json.dumps(state_snapshot, ensure_ascii=False), now),
            )
            conn.execute("COMMIT")

    def mark_failed(self, stored: StoredMessage, error: str) -> None:
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                UPDATE inbound_messages
                SET status = 'failed',
                    processed_at = ?,
                    lease_until = NULL,
                    error = ?
                WHERE tenant_id = ?
                  AND source_message_id = ?
                """,
                (_dt(datetime.now(DEFAULT_TZ)), error, stored.tenant_id, stored.source_message_id),
            )
            self._append_audit_in_tx(
                conn,
                trace_id=stored.source_message_id,
                tenant_id=stored.tenant_id,
                conversation_id=stored.conversation_id,
                source_message_id=stored.source_message_id,
                event_type="message_failed",
                payload={"error": error},
            )
            conn.execute("COMMIT")

    def load_agent_state(self) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT state_json FROM agent_state_snapshots WHERE id = 1"
            ).fetchone()
            return json.loads(row["state_json"]) if row else None

    def snapshot(self) -> dict[str, Any]:
        with self.connect() as conn:
            counts = {}
            for table in ["inbound_messages", "audit_events", "outbox_events"]:
                counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            statuses = {
                row["status"]: row["count"]
                for row in conn.execute(
                    """
                    SELECT status, COUNT(*) AS count
                    FROM inbound_messages
                    GROUP BY status
                    """
                ).fetchall()
            }
            offsets = [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT conversation_id, last_sequence, updated_at
                    FROM conversation_offsets
                    ORDER BY conversation_id
                    """
                ).fetchall()
            ]
            outbox = [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT id, trace_id, output_channel, target_type, target_id,
                           original_target_id, status, attempt_count, sent_at, error, message_text
                    FROM outbox_events
                    ORDER BY created_at DESC
                    LIMIT 50
                    """
                ).fetchall()
            ]
            recent_audit = [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT id, trace_id, conversation_id, event_type, created_at
                    FROM audit_events
                    ORDER BY id DESC
                    LIMIT 50
                    """
                ).fetchall()
            ]
        return {
            "db_path": str(self.path),
            "counts": counts,
            "message_statuses": statuses,
            "offsets": offsets,
            "outbox": outbox,
            "recent_audit": recent_audit,
        }

    def _insert_outbox_from_result_in_tx(
        self,
        conn: sqlite3.Connection,
        stored: StoredMessage,
        result_json: dict[str, Any],
    ) -> int:
        created = 0
        now = _dt(datetime.now(DEFAULT_TZ))
        message_json = stored.message_json
        output_channel = self._output_channel_for_message(message_json)
        reply_text = result_json.get("reply_text") or ""
        if result_json.get("should_reply") and reply_text:
            created += self._insert_outbox_one_in_tx(
                conn,
                stored=stored,
                trace_id=stored.source_message_id,
                output_channel=output_channel,
                target_type="reply",
                target_id=stored.conversation_id,
                original_target_id=stored.conversation_id,
                message_text=reply_text,
                idempotency_key=f"{stored.tenant_id}:{stored.source_message_id}:reply",
                now=now,
            )
        draft_group_post = result_json.get("draft_group_post")
        game_id = result_json.get("game_id") or "unknown_game"
        if draft_group_post:
            created += self._insert_outbox_one_in_tx(
                conn,
                stored=stored,
                trace_id=stored.source_message_id,
                output_channel=output_channel,
                target_type="group",
                target_id=stored.conversation_id,
                original_target_id=stored.conversation_id,
                message_text=draft_group_post,
                idempotency_key=f"{stored.tenant_id}:{game_id}:group_post",
                now=now,
            )
        for draft in result_json.get("invitation_drafts", []):
            created += self._insert_outbox_one_in_tx(
                conn,
                stored=stored,
                trace_id=stored.source_message_id,
                output_channel=output_channel,
                target_type="private",
                target_id=draft["customer_id"],
                original_target_id=draft["customer_id"],
                message_text=draft.get("message_text") or "",
                idempotency_key=f"{stored.tenant_id}:{game_id}:{draft['customer_id']}:private_invite",
                now=now,
            )
        return created

    def _insert_outbox_one_in_tx(
        self,
        conn: sqlite3.Connection,
        stored: StoredMessage,
        trace_id: str,
        output_channel: str,
        target_type: str,
        target_id: str,
        original_target_id: str | None,
        message_text: str,
        idempotency_key: str,
        now: str,
    ) -> int:
        outbox_id = "out_" + hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()[:24]
        before = conn.total_changes
        conn.execute(
            """
            INSERT OR IGNORE INTO outbox_events(
                id, trace_id, tenant_id, conversation_id, output_channel, target_type, target_id,
                original_target_id,
                message_text, idempotency_key, status, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)
            """,
            (
                outbox_id,
                trace_id,
                stored.tenant_id,
                stored.conversation_id,
                output_channel,
                target_type,
                target_id,
                original_target_id,
                message_text,
                idempotency_key,
                now,
                now,
            ),
        )
        inserted = 1 if conn.total_changes > before else 0
        if inserted:
            self._append_audit_in_tx(
                conn,
                trace_id=trace_id,
                tenant_id=stored.tenant_id,
                conversation_id=stored.conversation_id,
                source_message_id=stored.source_message_id,
                event_type="outbox_created",
                payload={
                    "output_channel": output_channel,
                    "target_type": target_type,
                    "target_id": target_id,
                    "original_target_id": original_target_id,
                    "idempotency_key": idempotency_key,
                },
            )
        return inserted

    def _output_channel_for_message(self, message_json: dict[str, Any]) -> str:
        metadata = message_json.get("metadata") or {}
        return str(
            metadata.get("output_channel")
            or metadata.get("reply_output_channel")
            or message_json.get("channel_type")
            or "console"
        )

    def pending_outbox_events(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT id, trace_id, tenant_id, conversation_id, output_channel,
                       target_type, target_id, original_target_id, message_text,
                       idempotency_key, status, attempt_count, sent_at, error,
                       created_at, updated_at
                FROM outbox_events
                WHERE status = 'pending'
                ORDER BY created_at, id
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]

    def mark_outbox_sent(self, outbox_id: str, external_id: str | None = None) -> None:
        now = _dt(datetime.now(DEFAULT_TZ))
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM outbox_events WHERE id = ?", (outbox_id,)).fetchone()
            if row is None:
                conn.execute("COMMIT")
                return
            conn.execute(
                """
                UPDATE outbox_events
                SET status = 'sent',
                    sent_at = ?,
                    error = NULL,
                    updated_at = ?
                WHERE id = ?
                """,
                (now, now, outbox_id),
            )
            self._append_audit_in_tx(
                conn,
                trace_id=row["trace_id"],
                tenant_id=row["tenant_id"],
                conversation_id=row["conversation_id"],
                source_message_id=row["trace_id"],
                event_type="outbox_sent",
                payload={"outbox_id": outbox_id, "external_id": external_id},
            )
            conn.execute("COMMIT")

    def mark_outbox_failed(self, outbox_id: str, error: str) -> None:
        now = _dt(datetime.now(DEFAULT_TZ))
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM outbox_events WHERE id = ?", (outbox_id,)).fetchone()
            if row is None:
                conn.execute("COMMIT")
                return
            conn.execute(
                """
                UPDATE outbox_events
                SET status = 'failed',
                    attempt_count = attempt_count + 1,
                    error = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (error, now, outbox_id),
            )
            self._append_audit_in_tx(
                conn,
                trace_id=row["trace_id"],
                tenant_id=row["tenant_id"],
                conversation_id=row["conversation_id"],
                source_message_id=row["trace_id"],
                event_type="outbox_failed",
                payload={"outbox_id": outbox_id, "error": error},
            )
            conn.execute("COMMIT")

    def _append_audit_in_tx(
        self,
        conn: sqlite3.Connection,
        trace_id: str,
        tenant_id: str,
        conversation_id: str,
        source_message_id: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        conn.execute(
            """
            INSERT INTO audit_events(
                trace_id, tenant_id, conversation_id, source_message_id,
                event_type, payload_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trace_id,
                tenant_id,
                conversation_id,
                source_message_id,
                event_type,
                json.dumps(payload, ensure_ascii=False),
                _dt(datetime.now(DEFAULT_TZ)),
            ),
        )

    def _stored_from_row(self, row: sqlite3.Row) -> StoredMessage:
        return StoredMessage(
            tenant_id=row["tenant_id"],
            source_message_id=row["source_message_id"],
            conversation_id=row["conversation_id"],
            sequence=row["sequence"],
            message_json=json.loads(row["message_json"]),
            status=row["status"],
            result_json=json.loads(row["result_json"]) if row["result_json"] else None,
            semantic_fingerprint=row["semantic_fingerprint"] if "semantic_fingerprint" in row.keys() else None,
            semantic_duplicate_of=row["semantic_duplicate_of"] if "semantic_duplicate_of" in row.keys() else None,
        )


class DurableAgentProcessor:
    def __init__(
        self,
        runtime: AgentRuntime | None = None,
        store: SQLiteDurableStore | None = None,
        processing_lease_seconds: float = 30.0,
    ) -> None:
        self.runtime = runtime or AgentRuntime()
        self.store = store or SQLiteDurableStore(Path("data") / "mahjong_agent.sqlite3")
        self.processing_lease_seconds = processing_lease_seconds
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()
        self._state_lock = threading.Lock()
        state = self.store.load_agent_state()
        if state:
            import_runtime_state(self.runtime, state)

    def process(self, envelope: IncomingEnvelope, now: datetime | None = None) -> DurableProcessResult:
        stored = self.store.insert_envelope(envelope)
        duplicate = stored.status == "processed"
        if duplicate and stored.result_json:
            return DurableProcessResult(
                status="duplicate_processed",
                tenant_id=stored.tenant_id,
                source_message_id=stored.source_message_id,
                conversation_id=stored.conversation_id,
                sequence=stored.sequence,
                runtime_result=runtime_result_from_dict(stored.result_json),
                processed_results=[],
                duplicate=True,
                waiting_for_sequence=False,
                outbox_created=0,
            )

        processed: list[RuntimeResult] = []
        outbox_created = 0
        with self._conversation_lock(stored.conversation_id):
            while True:
                next_message = self.store.claim_next_ready(
                    stored.conversation_id,
                    lease_seconds=self.processing_lease_seconds,
                )
                if next_message is None:
                    break
                try:
                    message = message_from_dict(next_message.message_json)
                    message.metadata["tenant_id"] = next_message.tenant_id
                    message.metadata["conversation_id"] = next_message.conversation_id
                    if next_message.semantic_duplicate_of:
                        result = semantic_duplicate_result(next_message)
                        self.store.mark_semantic_duplicate(
                            next_message,
                            result,
                            export_runtime_state(self.runtime),
                        )
                        processed.append(result)
                        continue
                    with self._state_lock:
                        result = self.runtime.process_message(message, now=now)
                        outbox_created += self.store.mark_processed(
                            next_message,
                            result,
                            export_runtime_state(self.runtime),
                        )
                    processed.append(result)
                except Exception as exc:
                    self.store.mark_failed(next_message, f"{type(exc).__name__}: {exc}")
                    raise

        refreshed = self.store.get_message(stored.tenant_id, stored.source_message_id)
        runtime_result = (
            runtime_result_from_dict(refreshed.result_json)
            if refreshed and refreshed.result_json
            else None
        )
        waiting = runtime_result is None
        return DurableProcessResult(
            status="processed" if runtime_result else "waiting_for_sequence",
            tenant_id=stored.tenant_id,
            source_message_id=stored.source_message_id,
            conversation_id=stored.conversation_id,
            sequence=stored.sequence,
            runtime_result=runtime_result,
            processed_results=processed,
            duplicate=False,
            waiting_for_sequence=waiting,
            outbox_created=outbox_created,
        )

    def snapshot(self) -> dict[str, Any]:
        data = self.runtime.snapshot()
        data["durable"] = self.store.snapshot()
        return data

    def reset_all(self) -> None:
        self.store.reset_all()

    def shutdown(self) -> None:
        self.runtime.shutdown()

    @contextmanager
    def _conversation_lock(self, conversation_id: str) -> Iterator[None]:
        with self._locks_guard:
            lock = self._locks.setdefault(conversation_id, threading.Lock())
        lock.acquire()
        try:
            yield
        finally:
            lock.release()


def export_runtime_state(runtime: AgentRuntime) -> dict[str, Any]:
    store = runtime.responder.core.store
    return {
        "messages": {key: _message_to_dict(value) for key, value in store.messages.items()},
        "games": {key: _game_to_dict(value) for key, value in store.games.items()},
        "customers": {key: _customer_to_dict(value) for key, value in store.customers.items()},
        "invitations": {key: _invitation_to_dict(value) for key, value in store.invitations.items()},
        "room_capacity": store.room_capacity,
        "room_holds": {key: _room_hold_to_dict(value) for key, value in store.room_holds.items()},
        "contexts": {
            key: context.to_dict()
            for key, context in runtime.contexts.items()
        },
    }


def import_runtime_state(runtime: AgentRuntime, state: dict[str, Any]) -> None:
    runtime.responder.core.store = InMemoryStore(
        messages={key: message_from_dict(value) for key, value in state.get("messages", {}).items()},
        games={key: game_from_dict(value) for key, value in state.get("games", {}).items()},
        customers={key: customer_from_dict(value) for key, value in state.get("customers", {}).items()},
        invitations={
            key: invitation_from_dict(value)
            for key, value in state.get("invitations", {}).items()
        },
        room_capacity=state.get("room_capacity"),
        room_holds={key: room_hold_from_dict(value) for key, value in state.get("room_holds", {}).items()},
    )
    runtime.contexts = {}
    for key, value in state.get("contexts", {}).items():
        turns = [
            ContextTurn(
                message_id=turn["message_id"],
                sender_id=turn["sender_id"],
                sender_name=turn["sender_name"],
                text=turn["text"],
                decision_action=turn.get("decision_action"),
                reply_text=turn.get("reply_text"),
                created_at=_parse_dt(turn["created_at"]),
            )
            for turn in value.get("recent_turns", [])
        ]
        context = ConversationContext(
            channel_id=value["channel_id"],
            turns=_deque_from_turns(turns, maxlen=runtime.config.max_recent_messages_per_context),
            created_at=_parse_dt(value["created_at"]),
            updated_at=_parse_dt(value["updated_at"]),
        )
        runtime.contexts[key] = context


def runtime_result_from_dict(data: dict[str, Any]) -> RuntimeResult:
    from .responder import ReplyDecision, ReplyAction

    decision = ReplyDecision(
        action=ReplyAction(data["action"]),
        reply_text=data["reply_text"],
        confidence=data["confidence"],
        should_reply=data["should_reply"],
        needs_human_review=data["needs_human_review"],
        game_id=data.get("game_id"),
        draft_group_post=data.get("draft_group_post"),
        llm_context_digest=data.get("llm_context_digest"),
        llm_context_snapshot=data.get("llm_context_snapshot"),
        invitation_drafts=[
            Invitation(
                game_id=item["game_id"],
                customer_id=item["customer_id"],
                customer_name=item["customer_name"],
                status=InvitationStatus(item["status"]),
                id=item["id"],
                message_text=item.get("message_text"),
            )
            for item in data.get("invitation_drafts", [])
        ],
        notes=data.get("notes", []),
    )
    runtime = data.get("runtime", {})
    return RuntimeResult(
        ok=runtime.get("ok", True),
        decision=decision,
        latency_ms=runtime.get("latency_ms", 0.0),
        timed_out=runtime.get("timed_out", False),
        error=runtime.get("error"),
        context=runtime.get("context"),
    )


def semantic_duplicate_result(stored: StoredMessage) -> RuntimeResult:
    from .responder import ReplyAction, ReplyDecision

    duplicate_of = stored.semantic_duplicate_of or "unknown"
    decision = ReplyDecision(
        action=ReplyAction.IGNORE,
        reply_text="",
        confidence=1.0,
        should_reply=False,
        needs_human_review=False,
        notes=[f"短时间内检测到语义重复消息，已跳过自动处理。duplicate_of={duplicate_of}"],
    )
    return RuntimeResult(
        ok=True,
        decision=decision,
        latency_ms=0.0,
        timed_out=False,
        error=None,
        context=None,
    )


def _message_to_dict(message: Message) -> dict[str, Any]:
    return {
        "id": message.id,
        "text": message.text,
        "sender_id": message.sender_id,
        "sender_name": message.sender_name,
        "channel_id": message.channel_id,
        "channel_type": message.channel_type.value,
        "sent_at": _dt(message.sent_at),
        "metadata": message.metadata,
    }


def message_from_dict(data: dict[str, Any]) -> Message:
    return Message(
        text=data["text"],
        sender_id=data["sender_id"],
        sender_name=data["sender_name"],
        channel_id=data["channel_id"],
        channel_type=ChannelType(data["channel_type"]),
        sent_at=_parse_dt(data["sent_at"]),
        id=data["id"],
        metadata=data.get("metadata", {}),
    )


def _game_to_dict(game: GameRequest) -> dict[str, Any]:
    return {
        "id": game.id,
        "organizer_id": game.organizer_id,
        "organizer_name": game.organizer_name,
        "channel_id": game.channel_id,
        "source_message_id": game.source_message_id,
        "status": game.status.value,
        "game_type": game.game_type,
        "ruleset": game.ruleset,
        "variant": game.variant,
        "seats_total": game.seats_total,
        "current_player_count": game.current_player_count,
        "missing_count": game.missing_count,
        "level": game.level,
        "base_score": game.base_score,
        "cap_score": game.cap_score,
        "start_at": _dt(game.start_at) if game.start_at else None,
        "start_time_confidence": game.start_time_confidence,
        "duration_hours": game.duration_hours,
        "play_options": game.play_options,
        "rules": game.rules,
        "notes": game.notes,
        "ambiguities": game.ambiguities,
        "participant_ids": game.participant_ids,
        "reserved_customer_ids": game.reserved_customer_ids,
        "created_at": _dt(game.created_at),
        "updated_at": _dt(game.updated_at),
        "version": game.version,
    }


def game_from_dict(data: dict[str, Any]) -> GameRequest:
    return GameRequest(
        organizer_id=data["organizer_id"],
        organizer_name=data["organizer_name"],
        channel_id=data["channel_id"],
        source_message_id=data.get("source_message_id"),
        id=data["id"],
        status=GameStatus(data["status"]),
        game_type=data.get("game_type", "mahjong"),
        ruleset=data.get("ruleset"),
        variant=data.get("variant"),
        seats_total=data.get("seats_total", 4),
        current_player_count=data.get("current_player_count"),
        missing_count=data.get("missing_count"),
        level=data.get("level"),
        base_score=data.get("base_score"),
        cap_score=data.get("cap_score"),
        start_at=_parse_dt(data["start_at"]) if data.get("start_at") else None,
        start_time_confidence=data.get("start_time_confidence", 0.0),
        duration_hours=data.get("duration_hours"),
        play_options=data.get("play_options", []),
        rules=data.get("rules", []),
        notes=data.get("notes", []),
        ambiguities=data.get("ambiguities", []),
        participant_ids=data.get("participant_ids", []),
        reserved_customer_ids=data.get("reserved_customer_ids", []),
        created_at=_parse_dt(data["created_at"]),
        updated_at=_parse_dt(data["updated_at"]),
        version=data.get("version", 0),
    )


def _room_hold_to_dict(hold: RoomHold) -> dict[str, Any]:
    return {
        "id": hold.id,
        "start_at": _dt(hold.start_at),
        "end_at": _dt(hold.end_at),
        "room_id": hold.room_id,
        "source": hold.source,
        "game_id": hold.game_id,
        "status": hold.status.value,
        "notes": hold.notes,
    }


def room_hold_from_dict(data: dict[str, Any]) -> RoomHold:
    return RoomHold(
        start_at=_parse_dt(data["start_at"]),
        end_at=_parse_dt(data["end_at"]),
        room_id=data.get("room_id"),
        source=data.get("source", "manual"),
        game_id=data.get("game_id"),
        status=RoomHoldStatus(data.get("status", RoomHoldStatus.ACTIVE.value)),
        id=data["id"],
        notes=data.get("notes", []),
    )


def _customer_to_dict(customer: CustomerProfile) -> dict[str, Any]:
    return {
        "id": customer.id,
        "display_name": customer.display_name,
        "aliases": customer.aliases,
        "preferred_levels": customer.preferred_levels,
        "play_preferences": [_play_preference_to_dict(item) for item in customer.play_preferences],
        "tags": customer.tags,
        "smoke_free_preference": customer.smoke_free_preference,
        "usual_party_size": customer.usual_party_size,
        "usual_party_size_confidence": customer.usual_party_size_confidence,
        "usual_start_hours": customer.usual_start_hours,
        "usual_weekdays": customer.usual_weekdays,
        "no_contact": customer.no_contact,
        "last_invited_at": _dt(customer.last_invited_at) if customer.last_invited_at else None,
        "decline_count_30d": customer.decline_count_30d,
        "max_games_per_day": customer.max_games_per_day,
        "min_hours_between_games": customer.min_hours_between_games,
        "invite_cooldown_hours": customer.invite_cooldown_hours,
        "daily_invite_limit": customer.daily_invite_limit,
        "fatigue_sensitivity": customer.fatigue_sensitivity,
        "metadata": customer.metadata,
    }


def customer_from_dict(data: dict[str, Any]) -> CustomerProfile:
    return CustomerProfile(
        id=data["id"],
        display_name=data["display_name"],
        aliases=data.get("aliases", []),
        preferred_levels=data.get("preferred_levels", []),
        play_preferences=[
            play_preference_from_dict(item)
            for item in data.get("play_preferences", [])
        ],
        tags=data.get("tags", []),
        smoke_free_preference=data.get("smoke_free_preference"),
        usual_party_size=data.get("usual_party_size"),
        usual_party_size_confidence=data.get("usual_party_size_confidence", 0.0),
        usual_start_hours=data.get("usual_start_hours", []),
        usual_weekdays=data.get("usual_weekdays", []),
        no_contact=data.get("no_contact", False),
        last_invited_at=_parse_dt(data["last_invited_at"]) if data.get("last_invited_at") else None,
        decline_count_30d=data.get("decline_count_30d", 0),
        max_games_per_day=data.get("max_games_per_day", 1),
        min_hours_between_games=data.get("min_hours_between_games", 6.0),
        invite_cooldown_hours=data.get("invite_cooldown_hours", 6.0),
        daily_invite_limit=data.get("daily_invite_limit", 3),
        fatigue_sensitivity=data.get("fatigue_sensitivity", 1.0),
        metadata=data.get("metadata", {}),
    )


def _play_preference_to_dict(preference: PlayPreference) -> dict[str, Any]:
    return {
        "game_type": preference.game_type,
        "preferred_levels": preference.preferred_levels,
        "preferred_rulesets": preference.preferred_rulesets,
        "preferred_variants": preference.preferred_variants,
        "preferred_play_options": preference.preferred_play_options,
        "avoid_play_options": preference.avoid_play_options,
    }


def play_preference_from_dict(data: dict[str, Any]) -> PlayPreference:
    return PlayPreference(
        game_type=data["game_type"],
        preferred_levels=data.get("preferred_levels", []),
        preferred_rulesets=data.get("preferred_rulesets", []),
        preferred_variants=data.get("preferred_variants", []),
        preferred_play_options=data.get("preferred_play_options", []),
        avoid_play_options=data.get("avoid_play_options", []),
    )


def _invitation_to_dict(invitation: Invitation) -> dict[str, Any]:
    return {
        "id": invitation.id,
        "game_id": invitation.game_id,
        "customer_id": invitation.customer_id,
        "customer_name": invitation.customer_name,
        "status": invitation.status.value,
        "created_at": _dt(invitation.created_at),
        "updated_at": _dt(invitation.updated_at),
        "message_text": invitation.message_text,
    }


def invitation_from_dict(data: dict[str, Any]) -> Invitation:
    return Invitation(
        game_id=data["game_id"],
        customer_id=data["customer_id"],
        customer_name=data["customer_name"],
        status=InvitationStatus(data["status"]),
        id=data["id"],
        created_at=_parse_dt(data["created_at"]),
        updated_at=_parse_dt(data["updated_at"]),
        message_text=data.get("message_text"),
    )


def _normalize_semantic_text(text: str) -> str:
    normalized = text.strip().lower()
    normalized = normalized.replace("，", ",").replace("。", ".").replace("！", "!").replace("？", "?")
    normalized = re.sub(r"\s+", "", normalized)
    normalized = re.sub(r"[,.!?;:，。！？；：]+$", "", normalized)
    return normalized


def _semantic_intent_kind(normalized_text: str) -> str:
    if re.search(r"(我来|算我|报名|可以来|加我一个|我能来|还有位置吗|还缺人吗|还能来吗)", normalized_text):
        return "join"
    if re.search(r"(不来了|来不了|没空|算了|下次|去不了|不方便)", normalized_text):
        return "decline"
    if re.search(r"(满了|组好了|凑齐了|齐了|不用找了)", normalized_text):
        return "full"
    if re.search(r"(取消|不打了|散了|改天)", normalized_text):
        return "cancel"
    return "general"


def _dt(value: datetime) -> str:
    return value.astimezone(DEFAULT_TZ).isoformat()


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(DEFAULT_TZ)


def _deque_from_turns(turns: list[ContextTurn], maxlen: int):
    from collections import deque

    output = deque(maxlen=maxlen)
    output.extend(turns)
    return output
