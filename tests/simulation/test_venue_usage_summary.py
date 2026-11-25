"""核对真实用量归档聚合的识别、重试去重和缺失计数边界。"""

import json

from mahjong_agent_runtime.tracing import TraceEvent
from scripts.summarize_venue_usage import summarize_usage


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
