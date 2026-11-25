#!/usr/bin/env python3
"""核对一对营业仿真实际送入角色的差异，输出可公开的计数和判定。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


_PAIRS = {
    ("smoke-pending", "smoke-confirmed"): "text",
    ("arrival-2100", "arrival-2140"): "earliest_arrival",
    ("question-unquoted", "question-quoted"): "quoted_message_id",
}


def _load(run_dir: Path) -> tuple[dict, dict]:
    """读取完整报告与第一条角色决定，避免将后来的互动当初始条件。"""
    report = json.loads(run_dir.joinpath("report.json").read_text(encoding="utf-8"))
    with run_dir.joinpath("events.jsonl").open(encoding="utf-8") as events:
        opening = json.loads(next(line for line in events if line.strip()))
    return report, opening


def _pop_condition(context: dict, field: str) -> tuple:
    """移除被比较的唯一字段，并验证实际触发和可见群聊一致。"""
    if field == "earliest_arrival":
        return context["events"][0]["details"].pop(field), context["delivery"] is None
    delivery = context["delivery"]
    if not delivery or not delivery["message_id"].startswith("variant:"):
        raise ValueError("缺少场景变体的公开投递")
    received = [row for row in context["history"] if row["message_id"] == delivery["message_id"]]
    if len(received) != 1:
        raise ValueError("场景变体在已见账本中必须恰好一条")
    condition = delivery.pop(field, None)
    return condition, received[0].pop(field, None) == condition


def compare_runs(before: Path, after: Path) -> dict:
    """核对两次模型所见；店家的业务结果或合法沉默各自保留。"""
    a, first = _load(before)
    b, second = _load(after)
    source, changed = a["historical_continuation"], b["historical_continuation"]
    pair = source["variant"], changed["variant"]
    if pair not in _PAIRS:
        raise ValueError("运行报告必须为同一组对照，先传基线，再传变体")
    field = _PAIRS[pair]
    one, two = (row["decision"] for row in (first, second))
    values = [_pop_condition(decision["context"], field) for decision in (one, two)]
    checks = {
        "same_roster_seed_and_run_settings": all(a.get(key) == b.get(key) for key in (
            "mode", "scenario_date", "seed", "people", "targeted_users", "workers", "max_turns", "role_seed_profiles",
            "model_execution", "sandbox_consent_prompt_sha256",
        )),
        "same_human_source_and_holdout": all(source[key] == changed[key] for key in (
            "case", "cutoff", "source_date", "prefix_messages", "held_out_messages"
        )),
        "only_intended_role_context_field_changes": (
            values[0][1] and values[1][1] and values[0][0] != values[1][0]
            and first["actor_id"] == second["actor_id"] and one["context"] == two["context"]
        ),
        "each_variant_was_applied_at_delivery": all(report["checks"]["single_history_variant_applied"]
                                                    for report in (a, b)),
    }
    return {"variants": list(pair), "mode": a["mode"], "compared_field": field, "checks": checks,
            "status": "passed" if all(checks.values()) else "failed", "runs": [
                {"run": path.name, "status": report["status"], "condition": info["condition"],
                 "role_model_actions": report["real_customer_actions"], "actions": info["actions"],
                 "sent_channels": info["sent_channels"], "judge": report["independent_judge"]}
                for path, report, info in ((before, a, source), (after, b, changed))
            ]}


def main() -> int:
    """从一对已完成运行生成去掉聊天原话的证明摘要。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    args = parser.parse_args()
    try:
        evidence = compare_runs(args.before, args.after)
    except (OSError, TypeError, ValueError, KeyError, StopIteration) as exc:
        parser.error(str(exc))
    print(json.dumps(evidence, ensure_ascii=False, indent=2))
    return int(evidence["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
