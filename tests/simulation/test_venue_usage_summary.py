"""核对真实用量归档聚合的识别、重试去重和缺失计数边界。"""

from contextlib import closing
from datetime import datetime, timezone
import json
import sqlite3

from mahjong_agent_runtime.tracing import TraceEvent
from scripts.summarize_venue_usage import summarize_usage

from .venue_role_journal import VenueRoleJournal


def test_usage_archive_deduplicates_attempts_and_preserves_cache_evidence(tmp_path):
    """阶段分拆仅认协议元数据，覆盖官方字段、归一化缓存、备援及坏行。"""
    role = {"attempt_id": "private-role-attempt", "usage": {
        "prompt_tokens": 100, "completion_tokens": 10, "prompt_cache_hit_tokens": 70, "prompt_cache_miss_tokens": 30,
    }}
    repaired = {"attempt_id": "repair-attempt", "usage": {
        "prompt_tokens": 50, "completion_tokens": 7, "prompt_tokens_details": {"cached_tokens": 20},
    }}
    shop = {"attempt_id": "shop-attempt", "usage": {"prompt_tokens": 150, "completion_tokens": 21}}
    events = {"decision": {"diagnostics": {"attempts": [role]}, "prior_attempts": [
        {"diagnostics": {"attempts": [repaired]}}
    ], "raw": "这里是私人对话或凭证"}}
    (tmp_path / "events.jsonl").write_text(json.dumps(events) + "\n坏行\n", encoding="utf-8")
    guard = {"diagnostics": {"attempts": [{"attempt_id": "guard-missing-usage", "http_status": 429}]}}
    (tmp_path / "write_authorization.jsonl").write_text(json.dumps(guard) + "\n", encoding="utf-8")
    report = {"semantic_samples": [{"diagnostics": {"attempts": [{"attempt_id": "judge-attempt",
        "usage": {"prompt_tokens": 22, "completion_tokens": 4, "prompt_cache_miss_tokens": 19}}]}}]}
    (tmp_path / "report.json").write_text(json.dumps(report), encoding="utf-8")
    trace = [
        TraceEvent("trace-1", "llm_response", {"completion_diagnostics": {
            "primary": {"attempts": [{"attempt_id": "shop-missing-usage"}]}, "fallback": {"attempts": [shop]},
        }}),
        TraceEvent("trace-1", "llm_response", {"completion_diagnostics": {"attempts": [shop, role]}}),
    ]
    (tmp_path / "agent_trace.log").write_text("\n".join(event.to_log_line() for event in trace) + "\n", encoding="utf-8")

    summary = summarize_usage(tmp_path)
    assert summary["total"] == {
        "observed_attempts": 6, "attempts_with_usage": 4, "missing_usage_attempts": 2,
        "input_tokens": 322, "output_tokens": 42, "cache_hit_input_tokens": 93,
        "cache_miss_input_tokens": 79, "unknown_cache_input_tokens": 150,
    }
    assert summary["stages"]["role"]["observed_attempts"] == 2
    assert summary["stages"]["shop"]["missing_usage_attempts"] == 1
    assert summary["deduplicated_attempt_records"] == 2
    assert summary["unreadable_records_by_source"] == {"events.jsonl": 1}
    content = json.dumps(summary, ensure_ascii=False)
    assert not any(private in content for private in ("private-role-attempt", "trace-1", "私人对话"))


def test_conflicting_duplicate_attempt_usage_is_not_priced(tmp_path):
    """同一 ID 的用量互相矛盾时只报一个不可计价尝试。"""
    rows = [{"decision": {"diagnostics": {"attempts": [
        {"attempt_id": "one-attempt", "usage": {"prompt_tokens": prompt, "completion_tokens": 2}},
        {"usage": {"prompt_tokens": 200, "completion_tokens": 2}},
    ]}}} for prompt in (10, 20)]
    (tmp_path / "events.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    summary = summarize_usage(tmp_path)
    assert summary["conflicting_usage_attempts"] == 1
    assert summary["unidentified_attempt_records"] == 2
    assert summary["total"]["observed_attempts"] == summary["total"]["missing_usage_attempts"] == 1
    assert summary["total"]["input_tokens"] == summary["total"]["output_tokens"] == 0


def test_active_role_journal_counts_saved_decision_and_transport_attempt_read_only(tmp_path):
    """整波未投递且没有导出时，计入已持久化的决定、结构修正及弃用请求。"""
    path = tmp_path / "role_events.sqlite3"
    journal = VenueRoleJournal(path)
    at = datetime(2026, 11, 25, tzinfo=timezone.utc)
    for actor_id in ("private-first-actor", "private-second-actor"):
        journal.enqueue(actor_id=actor_id, kind="session_open", origin_id="timer", due_at=at)
    saved_attempt = {"attempt_id": "private-role-success", "usage": {
        "prompt_tokens": 50, "completion_tokens": 2, "prompt_cache_hit_tokens": 30,
    }}
    repair_attempt = {"attempt_id": "private-role-repair", "usage": {"prompt_tokens": 10, "completion_tokens": 1}}
    with closing(sqlite3.connect(path)) as observer, journal.run_owner():
        observer.execute("SELECT count(*) FROM venue_role_events").fetchone()
        saved, deferred = journal.claim_wave(2)
        journal.save_decision(saved, {"decision": {"diagnostics": {"attempts": [saved_attempt]},
            "prior_attempts": [{"diagnostics": {"attempts": [repair_attempt]}}], "raw": "私人话术"}})
        journal.record_transport_attempt(deferred, {"decision": {"diagnostics": {"attempts": [
            {"attempt_id": "private-role-failure", "http_status": 429},
        ]}}})
        trace = TraceEvent("private-runtime-trace", "llm_response", {"completion_diagnostics": {
            "attempts": [saved_attempt],
        }})
        (tmp_path / "agent_trace.log").write_text(trace.to_log_line() + "\n", encoding="utf-8")
        files_before = set(tmp_path.iterdir())
        persisted = {file: file.read_bytes() for file in files_before if not file.name.endswith("-shm")}

        summary = summarize_usage(tmp_path)

        assert summary["sources"] == ["role_events.sqlite3", "agent_trace.log"]
        assert summary["total"] == summary["stages"]["role"] == {
            "observed_attempts": 3, "attempts_with_usage": 2, "missing_usage_attempts": 1,
            "input_tokens": 60, "output_tokens": 3, "cache_hit_input_tokens": 30,
            "cache_miss_input_tokens": 20, "unknown_cache_input_tokens": 10,
        }
        assert summary["deduplicated_attempt_records"] == 1
        assert summary["unreadable_records_by_source"] == {}
        assert set(tmp_path.iterdir()) == files_before
        assert all(file.read_bytes() == original for file, original in persisted.items())
        assert not any(secret in json.dumps(summary, ensure_ascii=False) for secret in ("private-", "私人话术"))


def test_terminal_role_archive_takes_precedence_over_journal(tmp_path):
    """已经导出角色事件时，用原有的终态来源和计数口径。"""
    (tmp_path / "role_events.sqlite3").write_bytes(b"invalid sqlite journal")
    terminal = {"decision": {"diagnostics": {"attempts": [
        {"attempt_id": "one-attempt", "usage": {"prompt_tokens": 10, "completion_tokens": 1}},
    ]}}}
    (tmp_path / "events.jsonl").write_text(json.dumps(terminal) + "\n", encoding="utf-8")

    summary = summarize_usage(tmp_path)

    assert summary["sources"] == ["events.jsonl"]
    assert summary["unreadable_records_by_source"] == {}
    assert summary["stages"]["role"]["observed_attempts"] == 1
