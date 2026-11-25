#!/usr/bin/env python3
"""汇总仿真归档中可核实的模型用量，仅输出阶段计数与缓存分布。"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterator
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mahjong_agent_runtime.tracing import TraceEvent  # noqa: E402


_SOURCES = ("events.jsonl", "write_authorization.jsonl", "report.json", "agent_trace.log")
_ROLE_JOURNAL = "role_events.sqlite3"
_TOKEN_FIELDS = ("input_tokens", "output_tokens", "cache_hit_input_tokens", "cache_miss_input_tokens",
                 "unknown_cache_input_tokens")
_Usage = tuple[int, int, int, int, int]


def _records(path: Path, unreadable: Counter) -> Iterator[dict]:
    """只解析预定归档，坏行计数，不输出原始日志。"""
    if path.name == _ROLE_JOURNAL:
        yield from _role_journal_records(path, unreadable)
        return
    with path.open(encoding="utf-8") as stream:
        if path.suffix == ".json":
            try:
                row = json.load(stream)
            except json.JSONDecodeError:
                unreadable[path.name] += 1
                return
            if isinstance(row, dict):
                yield row
            else:
                unreadable[path.name] += 1
            return
        for line in stream:
            if not line.strip():
                continue
            if path.name == "agent_trace.log" and (event := TraceEvent.from_log_line(line.rstrip("\n"))) is not None:
                yield {"step": event.step, **event.content}
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                unreadable[path.name] += 1
                continue
            if isinstance(row, dict):
                yield row
            else:
                unreadable[path.name] += 1


def _role_journal_records(path: Path, unreadable: Counter) -> Iterator[dict]:
    """在同一只读快照读取已落盘决定和保留的传输失败，避开调度锁。"""
    uri = path.resolve().as_uri() + "?mode=ro"
    try:
        with closing(sqlite3.connect(uri, uri=True)) as db:
            rows = db.execute("""
                SELECT decision_json FROM venue_role_events WHERE decision_json IS NOT NULL
                UNION ALL
                SELECT decision_json FROM venue_role_transport_attempts
            """)
            for (value,) in rows:
                try:
                    row = json.loads(value)
                except (TypeError, ValueError):
                    unreadable[path.name] += 1
                    continue
                if isinstance(row, dict):
                    yield row
                else:
                    unreadable[path.name] += 1
    except sqlite3.Error:
        unreadable[path.name] += 1


def _diagnostics(row: dict) -> Iterator[dict]:
    """同一流程的最终调用与结构修正都可能独立产生用量。"""
    for field in ("completion_diagnostics", "diagnostics"):
        if isinstance(value := row.get(field), dict):
            yield value
    for prior in row.get("prior_attempts") or ():
        if isinstance(prior, dict):
            yield from _diagnostics(prior)


def _attempts(diagnostics: dict) -> Iterator[dict]:
    """递归读取主路由和备援路由的 HTTP 尝试。"""
    for attempt in diagnostics.get("attempts") or ():
        if isinstance(attempt, dict):
            yield attempt
    for route in ("primary", "fallback"):
        if isinstance(child := diagnostics.get(route), dict):
            yield from _attempts(child)


def _runtime_stage(step: str) -> str:
    """根据稳定追踪事件区分主模型和独立辅助调用。"""
    for prefix, stage in (
        ("customer_visible_content_review_", "reply_review"),
        ("customer_visible_text_generation_", "reply_copy"),
        ("creation_authorization_", "creation_authorization"),
        ("context_summary_", "context_summary"),
        ("group_semantic_routing_", "group_routing"),
        ("llm_", "shop"),
    ):
        if step.startswith(prefix):
            return stage
    return "other_runtime"


def _staged_rows(source: str, record: dict) -> Iterator[tuple[str, dict]]:
    """从各归档提取对应角色、店家、授权及评审的诊断宿主。"""
    if source in ("events.jsonl", _ROLE_JOURNAL):
        if isinstance(decision := record.get("decision"), dict):
            yield "role", decision
    elif source == "write_authorization.jsonl":
        yield "write_guard", record
    elif source == "report.json":
        for row in record.get("semantic_samples") or ():
            if isinstance(row, dict):
                yield "judge", row
    else:
        yield _runtime_stage(str(record.get("step") or "")), record


def _token(value: object) -> int | None:
    """只接受供应商诊断合同允许的非负整数。"""
    return value if type(value) is int and 0 <= value <= 10**12 else None


def _usage(attempt: dict) -> _Usage | None:
    """按供应商凭据区分缓存命中、未命中和无法分辨的输入。"""
    usage = attempt.get("usage")
    if not isinstance(usage, dict):
        return None
    prompt = _token(usage.get("prompt_tokens"))
    completion = _token(usage.get("completion_tokens"))
    if prompt is None or completion is None:
        return None
    details = usage.get("prompt_tokens_details")
    hit = _token(usage.get("prompt_cache_hit_tokens"))
    if hit is None and isinstance(details, dict):
        hit = _token(details.get("cached_tokens"))
    miss = _token(usage.get("prompt_cache_miss_tokens"))
    if hit is None and miss is None:
        return prompt, completion, 0, 0, prompt
    if hit is None:
        hit = prompt - miss
    if miss is None:
        miss = prompt - hit
    if min(hit, miss) < 0 or hit + miss != prompt:
        return prompt, completion, 0, 0, prompt
    return prompt, completion, hit, miss, 0


def summarize_usage(run_dir: Path) -> dict:
    """按独立 HTTP 尝试去重，缓存不明或缺用量时分别记录，避免虚报价格。"""
    files = [filename for filename in _SOURCES if (run_dir / filename).is_file()]
    if "events.jsonl" not in files and (run_dir / _ROLE_JOURNAL).is_file():
        files.insert(0, _ROLE_JOURNAL)
    if not files:
        raise ValueError("运行目录没有可读取的模型归档")
    records: dict[str, tuple[str, _Usage | None]] = {}
    unreadable: Counter = Counter()
    conflicts: set[str] = set()
    repeated = unidentified = 0
    for filename in files:
        for record in _records(run_dir / filename, unreadable):
            for stage, row in _staged_rows(filename, record):
                for diagnostics in _diagnostics(row):
                    for attempt in _attempts(diagnostics):
                        attempt_id = attempt.get("attempt_id")
                        if not isinstance(attempt_id, str) or not attempt_id:
                            unidentified += 1
                            continue
                        usage = _usage(attempt)
                        if attempt_id not in records:
                            records[attempt_id] = stage, usage
                            continue
                        repeated += 1
                        first_stage, first_usage = records[attempt_id]
                        if first_usage is None:
                            records[attempt_id] = first_stage, usage
                        elif usage is not None and usage != first_usage:
                            if first_usage[:2] != usage[:2] or not (first_usage[4] or usage[4]):
                                conflicts.add(attempt_id)
                            elif first_usage[4] > usage[4]:
                                records[attempt_id] = first_stage, usage

    empty = dict.fromkeys(("observed_attempts", "attempts_with_usage", "missing_usage_attempts", *_TOKEN_FIELDS), 0)
    stages = defaultdict(lambda: Counter(empty))
    for attempt_id, (stage, usage) in records.items():
        totals = stages[stage]
        totals["observed_attempts"] += 1
        if usage is None or attempt_id in conflicts:
            totals["missing_usage_attempts"] += 1
            continue
        totals["attempts_with_usage"] += 1
        totals.update(dict(zip(_TOKEN_FIELDS, usage, strict=True)))
    total = Counter(empty)
    for stage_totals in stages.values():
        total.update(stage_totals)
    return {
        "sources": files, "unreadable_records_by_source": dict(unreadable), "total": dict(total),
        "stages": {stage: dict(counts) for stage, counts in sorted(stages.items())},
        "deduplicated_attempt_records": repeated, "unidentified_attempt_records": unidentified,
        "conflicting_usage_attempts": len(conflicts),
    }


def main() -> int:
    """从指定目录读取留档，并把可提交的匿名汇总写到标准输出。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    try:
        report = summarize_usage(args.run_dir)
    except (OSError, UnicodeError, ValueError):
        parser.error("未能读取该运行目录的模型用量归档")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
