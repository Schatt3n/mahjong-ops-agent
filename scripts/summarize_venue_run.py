#!/usr/bin/env python3
"""从沙盒报告生成可提交仓库的最小证据，避免携带个人或请求原文。"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable
import json
from pathlib import Path


def summary(report: dict) -> dict:
    """只保留计数、配置、检查和结论，不复制原始消息、完整请求或角色身份。"""
    evidence = {name: report.get(name) for name in (
        "mode", "scenario", "scenario_date", "seed", "status", "people", "targeted_users",
        "workers", "max_turns", "real_customer_actions", "peak_inflight_deliveries", "chat_events",
        "sandbox_consent_prompt_sha256", "elapsed_seconds",
    )}
    acceptance = report.get("acceptance") or {}
    evidence["checks"] = report.get("checks", {})
    if contacts := report.get("initial_contacts"):
        evidence["initial_contacts"] = contacts
    if initiation := report.get("private_initiation"):
        evidence["private_initiation"] = initiation
    evidence["causal_multi_round_customers"] = acceptance.get("multi_round_private_customers")
    evidence["outcome_counts"] = acceptance.get("outcome_counts", {})
    evidence["technical_failure_counts"] = dict(Counter(row["kind"] for row in acceptance.get("technical_failures", ())))
    judge = report.get("independent_judge")
    evidence["independent_judge"] = {key: judge.get(key) for key in ("status", "verdict")} if judge else None
    if authorization := report.get("write_authorization"):
        evidence["write_authorization"] = authorization
    evidence["judge_samples"] = [
        {
            "status": row["status"], "verdict": row["verdict"],
            "checks": {finding["check"]: finding["verdict"] for finding in row.get("findings", ())},
            "evidence_bytes": (row.get("projection") or {}).get("evidence_bytes"),
            "elapsed_ms": (row.get("diagnostics") or {}).get("total_elapsed_ms"),
        }
        for row in report.get("semantic_samples", ())
    ]
    evidence["models"] = {
        stage: {key: config.get(key) for key in (
            "model", "provider", "protocol", "max_output_tokens", "max_concurrency", "reasoning_effort_on_wire",
        )}
        for stage, config in report.get("model_execution", {}).items()
    }
    if cohort := report.get("cohort"):
        initial, final = cohort.get("initial_game") or {}, cohort.get("final_game") or {}
        evidence["cohort"] = {
            "expected_outcome": cohort.get("expected_outcome"),
            "expected_counts": dict(Counter(cohort.get("expected_outcomes", {}).values())),
            "authorized_transitions": cohort.get("authorized_transitions", {}),
            "initial_remaining_seats": initial.get("remaining_seats"),
            "final_remaining_seats": final.get("remaining_seats"),
            "final_status": final.get("status"),
        }
    return evidence


def actor_observations(events: Iterable[dict]) -> dict:
    """计数真实角色选择、首句差异和格式修正，摘要不携带原话或身份。"""
    first_messages: dict[str, str] = {}
    actions: Counter[str] = Counter()
    schema_repairs = 0
    for row in events:
        decision = row.get("decision") or {}
        if decision.get("source") != "model":
            continue
        schema_repairs += bool(decision.get("prior_attempts"))
        if decision.get("status") != "valid" or not (decision.get("response_source") or {}).get("http_response_complete"):
            continue
        action = decision.get("action") or {}
        kind = action.get("action")
        if kind in {"send", "wait", "arrive", "leave"}:
            actions[kind] += 1
        if kind == "send" and row.get("message_id") and row.get("actor_id"):
            first_messages.setdefault(row["actor_id"], " ".join(action["text"].split()))
    return {
        "valid_actions": dict(actions), "schema_repairs_attempted": schema_repairs,
        "customers_with_delivered_input": len(first_messages),
        "distinct_first_message_texts": len(set(first_messages.values())),
    }


def main() -> int:
    """输出 JSON 摘要，原始报告留在运行目录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    evidence = summary(json.loads(args.report.read_text(encoding="utf-8")))
    events = args.report.with_name("events.jsonl")
    if events.is_file():
        with events.open(encoding="utf-8") as handle:
            evidence["actor_observations"] = actor_observations(json.loads(line) for line in handle if line.strip())
    print(json.dumps({"run": args.report.parent.name, **evidence}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
