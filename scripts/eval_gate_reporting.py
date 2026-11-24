"""为统一门禁记录候选版本、模型配置和可定位的失败证据。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from mahjong_agent_runtime.llm import AgentLLMConfig, optional_positive_env_float
from mahjong_agent_runtime.review_config import build_reply_review_client, reply_review_timeout_seconds
from run_inquiry_grounding_eval import make_semantic_judge_config


def candidate_evidence(root: Path) -> dict[str, Any]:
    """记录实际检出的树和资产摘要，脏工作树的结果不能只凭提交号认领。"""
    assets = {
        str(path.relative_to(root).as_posix()): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((root / "eval").rglob("*"))
        if path.is_file() and path.suffix in {".json", ".jsonl"}
    }
    evidence: dict[str, Any] = {"commit": None, "tree": None, "dirty": None, "asset_sha256": assets}
    if not (root / ".git").exists():
        return evidence
    commit, tree = subprocess.check_output(
        ["git", "rev-parse", "HEAD", "HEAD^{tree}"], cwd=root, text=True,
    ).splitlines()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True))
    evidence.update(commit=commit, tree=tree, dirty=dirty)
    return evidence


def model_evidence() -> dict[str, Any]:
    """在计费和运行测试前解析三套请求配置，报告只使用无凭证字段。"""
    config = AgentLLMConfig.from_env()
    if config is None:
        raise ValueError("a main model and API key are required")
    review = build_reply_review_client(config)
    if review is None:
        raise ValueError("a reply review model is required")
    judge = make_semantic_judge_config(config)
    raw_timeout = optional_positive_env_float("MAHJONG_AGENT_LLM_TIMEOUT_SECONDS")
    fields = (
        "provider", "model", "api_protocol", "response_format", "temperature", "max_tokens",
        "thinking_mode", "reasoning_effort", "attempt_timeout_seconds", "retry_attempts",
        "max_concurrency", "max_estimated_tokens_per_day",
    )
    return {
        "main": {**{key: getattr(config, key) for key in fields},
                 "default_total_timeout_seconds": raw_timeout or 45.0,
                 "topic_total_timeout_seconds": raw_timeout or 90.0},
        "reply_review": {**{key: getattr(review.config, key) for key in fields},
                         "total_timeout_seconds": reply_review_timeout_seconds()},
        "semantic_judge": {**{key: getattr(judge, key) for key in fields},
                           "total_timeout_seconds": judge.attempt_timeout_seconds},
    }


def first_failed_case(phase: dict[str, Any]) -> dict[str, Any]:
    """从子运行器的结构化报告定位首例，保留原日志供受控环境复查。"""
    if not phase.get("report_path"):
        return {}
    path = Path(phase["report_path"])
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(payload, dict):
        return {}
    for collection in ("cases", "reports", "results"):
        rows = payload.get(collection)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict) or (
                row.get("status") not in ("failed", "error") and row.get("passed") is not False
            ):
                continue
            case_id = row.get("case_id") or row.get("scenario_id")
            if not isinstance(case_id, str):
                continue
            checks = row.get("checks")
            failed = []
            if isinstance(checks, dict):
                failed = [key for key, value in checks.items() if value is False]
            elif isinstance(checks, list):
                failed = [check["name"] for check in checks if isinstance(check, dict)
                          and isinstance(check.get("name"), str) and check.get("passed") is False]
            # 恢复评分引用全部业务断言；原始业务失败优先显示，恢复评分仍保留在子报告。
            primary = [key for key in failed if key != "trace_proven_recovery"] or failed
            return {"case_id": case_id, "failed_checks": primary}
    return {}
