from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo


TZ = ZoneInfo("Asia/Shanghai")
TRACE_EVENT_SCHEMA_VERSION = "trace_events.v1"


def default_now() -> datetime:
    return datetime.now(TZ)


def format_io_log_line(trace_id: str, level: str, content: str, *, at: datetime | None = None) -> str:
    stamp = (at or default_now()).strftime("%Y-%m-%d %H:%M:%S")
    safe_trace_id = trace_id or "trace_missing"
    safe_level = (level or "INFO").upper()
    return f"{safe_trace_id}-{stamp}-{safe_level}: {content}"


def trace_payload_from_content(content: str) -> dict[str, Any]:
    parsed = _loads_dict(content)
    if parsed:
        return parsed
    return {"direction": "log", "event": "text_log", "content": content}


def ensure_trace_events_table(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS trace_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trace_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            level TEXT NOT NULL,
            direction TEXT NOT NULL DEFAULT 'log',
            event TEXT NOT NULL DEFAULT '',
            stage TEXT NOT NULL DEFAULT '',
            schema_version TEXT NOT NULL DEFAULT '',
            payload_json TEXT NOT NULL DEFAULT '{}',
            content TEXT NOT NULL DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_trace_events_trace
            ON trace_events(trace_id, id);
        CREATE INDEX IF NOT EXISTS idx_trace_events_kind
            ON trace_events(direction, event, created_at);
        """
    )


class TrialTraceLogger:
    def __init__(
        self,
        *,
        db_path: Path,
        log_path: Path,
        now_factory: Callable[[], datetime] = default_now,
        print_lines: bool = True,
    ) -> None:
        self.db_path = db_path
        self.log_path = log_path
        self.now_factory = now_factory
        self.print_lines = print_lines
        self._log_lock = threading.Lock()
        self._trace_event_lock = threading.Lock()

    def write_trace_event(self, trace_id: str, level: str, content: str) -> None:
        payload = trace_payload_from_content(content)
        direction = str(payload.get("direction") or "log")
        event = str(payload.get("event") or payload.get("path") or direction or "log")
        stage = str(payload.get("stage") or payload.get("tool_stage") or "")
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with self._trace_event_lock:
                conn = sqlite3.connect(self.db_path)
                try:
                    ensure_trace_events_table(conn)
                    conn.execute(
                        """
                        INSERT INTO trace_events (
                            trace_id, created_at, level, direction, event, stage,
                            schema_version, payload_json, content
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            trace_id or "trace_missing",
                            self.now_factory().isoformat(),
                            (level or "INFO").upper(),
                            direction[:80],
                            event[:120],
                            stage[:120],
                            TRACE_EVENT_SCHEMA_VERSION,
                            _dump_json(payload),
                            _truncate_text(content, 8000),
                        ),
                    )
                    conn.commit()
                finally:
                    conn.close()
        except Exception:
            return

    def write_io_log(self, trace_id: str, level: str, content: str) -> None:
        line = format_io_log_line(trace_id, level, content, at=self.now_factory())
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_lock:
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        self.write_trace_event(trace_id, level, content)
        if self.print_lines:
            print(line)

    def write_llm_audit_log(self, trace_id: str, event: str, payload: dict[str, Any]) -> None:
        self.write_io_log(
            trace_id,
            "INFO",
            _dump_json(
                {
                    "direction": "llm",
                    "event": event,
                    **payload,
                }
            ),
        )

    def write_tool_audit_log(self, trace_id: str, event: str, payload: dict[str, Any]) -> None:
        self.write_io_log(
            trace_id,
            "INFO",
            _dump_json(
                {
                    "direction": "tool",
                    "event": event,
                    **payload,
                }
            ),
        )


def _dump_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _loads_dict(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _truncate_text(value: str, limit: int) -> str:
    text = value.replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"
