from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo

from mahjong_agent.trial_observability import (
    TRACE_EVENT_SCHEMA_VERSION,
    TrialTraceLogger,
    format_io_log_line,
    trace_payload_from_content,
)


TZ = ZoneInfo("Asia/Shanghai")


def test_trial_io_log_line_uses_trace_time_level_contract() -> None:
    line = format_io_log_line(
        "trace_trial",
        "warn",
        '{"direction":"input"}',
        at=datetime(2026, 7, 1, 9, 8, 7, tzinfo=TZ),
    )

    assert line == 'trace_trial-2026-07-01 09:08:07-WARN: {"direction":"input"}'


def test_trace_payload_parses_json_or_wraps_text() -> None:
    assert trace_payload_from_content('{"direction":"llm","event":"llm_request"}') == {
        "direction": "llm",
        "event": "llm_request",
    }

    assert trace_payload_from_content("plain text") == {
        "direction": "log",
        "event": "text_log",
        "content": "plain text",
    }


def test_trial_trace_logger_writes_log_file_and_trace_event(tmp_path) -> None:
    db_path = tmp_path / "trial.db"
    log_path = tmp_path / "io.log"
    logger = TrialTraceLogger(
        db_path=db_path,
        log_path=log_path,
        now_factory=lambda: datetime(2026, 7, 1, 10, 11, 12, tzinfo=TZ),
        print_lines=False,
    )

    logger.write_io_log(
        "trace_logger",
        "info",
        '{"direction":"llm","event":"llm_response","stage":"semantic","content":"ok"}',
    )

    assert log_path.read_text(encoding="utf-8") == (
        'trace_logger-2026-07-01 10:11:12-INFO: '
        '{"direction":"llm","event":"llm_response","stage":"semantic","content":"ok"}\n'
    )

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            """
            SELECT trace_id, level, direction, event, stage, schema_version,
                   payload_json, content
            FROM trace_events
            """
        ).fetchone()
    finally:
        conn.close()

    assert row is not None
    assert row[:6] == (
        "trace_logger",
        "INFO",
        "llm",
        "llm_response",
        "semantic",
        TRACE_EVENT_SCHEMA_VERSION,
    )
    assert json.loads(row[6]) == {
        "content": "ok",
        "direction": "llm",
        "event": "llm_response",
        "stage": "semantic",
    }
    assert row[7] == '{"direction":"llm","event":"llm_response","stage":"semantic","content":"ok"}'


def test_trial_trace_logger_records_llm_and_tool_audit_events(tmp_path) -> None:
    logger = TrialTraceLogger(
        db_path=tmp_path / "trial.db",
        log_path=tmp_path / "io.log",
        now_factory=lambda: datetime(2026, 7, 1, 12, 0, 0, tzinfo=TZ),
        print_lines=False,
    )

    logger.write_llm_audit_log("trace_audit", "llm_request", {"stage": "semantic"})
    logger.write_tool_audit_log("trace_audit", "tool_result", {"tool_name": "search_current_open_games"})

    conn = sqlite3.connect(tmp_path / "trial.db")
    try:
        rows = conn.execute(
            "SELECT direction, event, stage, payload_json FROM trace_events ORDER BY id"
        ).fetchall()
    finally:
        conn.close()

    assert [(row[0], row[1], row[2]) for row in rows] == [
        ("llm", "llm_request", "semantic"),
        ("tool", "tool_result", ""),
    ]
    assert json.loads(rows[1][3])["tool_name"] == "search_current_open_games"
