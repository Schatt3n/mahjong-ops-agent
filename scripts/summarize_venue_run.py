#!/usr/bin/env python3
"""从沙盒报告生成可提交仓库的最小证据，避免携带个人或请求原文。"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def summary(report: dict) -> dict:
    """只保留计数、配置、检查和结论，不复制原始消息、完整请求或角色身份。"""
    evidence = {name: report.get(name) for name in (
        "mode", "scenario", "scenario_date", "seed", "status", "people", "targeted_users",
        "workers", "max_turns", "real_customer_actions", "peak_inflight_deliveries", "chat_events",
    )}
    acceptance = report.get("acceptance") or {}
    evidence["checks"] = report.get("checks", {})
    evidence["causal_multi_round_customers"] = acceptance.get("multi_round_private_customers")
    evidence["outcome_counts"] = acceptance.get("outcome_counts", {})
    evidence["technical_failure_counts"] = dict(Counter(row["kind"] for row in acceptance.get("technical_failures", ())))
    judge = report.get("independent_judge")
    evidence["independent_judge"] = {key: judge.get(key) for key in ("status", "verdict")} if judge else None
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
            "expected_outcome": cohort["expected_outcome"],
            "initial_remaining_seats": initial.get("remaining_seats"),
            "final_remaining_seats": final.get("remaining_seats"),
            "final_status": final.get("status"),
        }
    return evidence


def main() -> int:
    """输出 JSON 摘要，原始报告留在运行目录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    evidence = summary(json.loads(args.report.read_text(encoding="utf-8")))
    print(json.dumps({"run": args.report.parent.name, **evidence}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
