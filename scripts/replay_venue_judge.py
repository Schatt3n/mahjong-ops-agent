#!/usr/bin/env python3
"""用同一份沙盒会话归档重放 OpenCode Go 评测并汇报请求规模与传输证据。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from mahjong_agent_runtime.llm import llm_conversation_scope  # noqa: E402
from scripts.run_venue_simulation import _select_local_model  # noqa: E402
from tests.simulation.venue_actors import SemanticJudge  # noqa: E402
from tests.simulation.venue_judging import review_dialogue  # noqa: E402
from tests.simulation.venue_models import opencode_auxiliary  # noqa: E402


def _read_rows(path: Path) -> list[dict]:
    """读取已归档的逐行事件或聊天账本。"""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _summary(result: dict) -> dict:
    """投影无原文、无凭证的对照结果，可直接作为仓库的排查记录。"""
    diagnostics = result["diagnostics"]
    attempts = []
    for row in diagnostics.get("attempts", []):
        usage = row.get("usage") or {}
        attempts.append({
            "http_status": row.get("http_status"), "elapsed_ms": row.get("http_elapsed_ms"),
            "timeout_seconds": row.get("timeout_seconds"), "error_type": row.get("error_type"),
            "incomplete_reason": row.get("incomplete_reason"), "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        })
    size = (result.get("projection") or {}).get("evidence_bytes")
    if size is None:
        size = len(json.dumps(result["evidence"], ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    return {"status": result["status"], "verdict": result["verdict"], "error": result.get("error"),
            "serialized_evidence_bytes": size, "elapsed_ms": diagnostics.get("total_elapsed_ms"),
            "reasoning_effort": (diagnostics.get("request") or {}).get("reasoning_effort"), "attempts": attempts}


def main() -> int:
    """从只读本地材料构造对照组，沿用已配置的 OpenCode Go 凭证。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--customer", required=True)
    parser.add_argument("--layout", choices=("compact", "full", "both"), default="compact")
    parser.add_argument("--timeout", type=float, default=45)
    args = parser.parse_args()
    if not 1 <= args.timeout <= 180:
        parser.error("--timeout 必须在 1 到 180 秒之间")
    turns = [row for row in _read_rows(args.run_dir / "events.jsonl") if row.get("actor_id") == args.customer]
    if not turns:
        parser.error("没有该顾客的已归档对话")
    records = _read_rows(args.run_dir / "chat.jsonl")
    report_path = args.run_dir / "report.json"
    outcome = None
    historical = False
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        historical = report.get("scenario") in {"historical-group", "historical-private"}
        outcome = next((row for row in report.get("acceptance", {}).get("outcomes", ())
                        if row.get("customer_id") == args.customer), None)
    config = _select_local_model("opencode-go", None, workers=1)
    judge = SemanticJudge(opencode_auxiliary(config, max_tokens=2600, workers=1), timeout_seconds=args.timeout)
    with llm_conversation_scope(f"venue-judge-replay:{args.run_dir.name}:{args.customer}"):
        if args.layout in {"full", "both"}:
            result = judge.judge(f"dialogue:{args.customer}", turns, trace_id=f"judge-full:{args.customer}",
                checks=("顾客发言符合个人说话方式、已知信息及意向",
                        "续轮真实回应前文，Agent 没有重复追问已知条件或误报结果")).to_dict()
            print(json.dumps({"layout": "full", **_summary(result)}, ensure_ascii=False), flush=True)
        if args.layout in {"compact", "both"}:
            result = review_dialogue(judge, customer_id=args.customer, turns=turns, records=records, outcome=outcome,
                role_only=historical, delivered_history=historical)
            print(json.dumps({"layout": "compact", **_summary(result)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
