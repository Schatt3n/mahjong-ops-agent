"""Read and rerun the fixed test suites exposed by the local operator console."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DETERMINISTIC_REPORT = ROOT / "runtime_data" / "concurrency_eval_deterministic_report.json"
LIVE_REPORT = ROOT / "runtime_data" / "live_eval_multi_option_commitment.json"
FOCUSED_UNIT_REPORT = ROOT / "runtime_data" / "focused_multi_option_pytest.json"
_RUN_LOCK = threading.Lock()


def read_json_report(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "unreadable", "error": f"{type(exc).__name__}: {exc}"}
    if not isinstance(payload, dict):
        return {"status": "unreadable", "error": "report root must be a JSON object"}
    return payload


def report_metadata(path: Path, payload: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "updated_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
        if path.exists()
        else None,
        "payload": payload,
    }


def observability_payload() -> dict[str, Any]:
    """Return reports plus their reproducible source and command provenance."""

    deterministic = read_json_report(DETERMINISTIC_REPORT)
    live = read_json_report(LIVE_REPORT)
    focused_unit = read_json_report(FOCUSED_UNIT_REPORT)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reports": {
            "deterministic": report_metadata(DETERMINISTIC_REPORT, deterministic),
            "live_deepseek": report_metadata(LIVE_REPORT, live),
            "focused_unit": report_metadata(FOCUSED_UNIT_REPORT, focused_unit),
        },
        "test_design": {
            "deterministic": {
                "purpose": "验证并发、幂等和最终落库状态，不依赖模型随机性。",
                "fixture_source": str(ROOT / "scripts" / "run_concurrency_eval.py"),
                "production_entrypoint": "SQLiteAgentStore.record_candidate_reply",
                "candidate_reply_simulation": (
                    "为每个候选人构造 customer_id、game_id、status、seat_count 和 trace_id，"
                    "并发调用与生产相同的 record_candidate_reply。"
                ),
            },
            "live_deepseek": {
                "purpose": "验证候选人自然语言回复进入主 Agent 后，模型是否选择正确工具并得到正确落库结果。",
                "fixture_source": str(ROOT / "scripts" / "run_real_owner_chat_live_eval.py"),
                "golden_source": str(ROOT / "eval" / "golden" / "multi_option_game_commitment.jsonl"),
                "candidate_reply_simulation": (
                    "先创建局、参与者和最近对话，再把候选人的文本作为 UserMessage 送入 AgentRuntime；"
                    "DeepSeek 决定是否调用 record_candidate_reply。"
                ),
            },
        },
        "allowed_suites": ["deterministic", "focused_unit", "live_deepseek"],
    }


def _command_for_suite(suite: str) -> tuple[list[str], Path, int]:
    if suite == "deterministic":
        return (
            [
                sys.executable,
                str(ROOT / "scripts" / "run_concurrency_eval.py"),
                "--mode",
                "deterministic",
                "--operations",
                "12",
                "--workers",
                "8",
                "--strict",
                "--report-path",
                str(DETERMINISTIC_REPORT),
            ],
            DETERMINISTIC_REPORT,
            90,
        )
    if suite == "focused_unit":
        return (
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/test_multi_option_game_commitment.py",
                "tests/test_agent_runtime.py::test_runtime_lets_model_drive_tool_sequence_until_final_reply",
                "tests/test_agent_runtime.py::test_runtime_customer_visible_text_generation_rewrites_invite_text_before_review_and_draft",
                "tests/test_agent_app.py::test_human_approved_invite_is_sent_once_and_persisted",
            ],
            FOCUSED_UNIT_REPORT,
            90,
        )
    if suite == "live_deepseek":
        return (
            [
                sys.executable,
                str(ROOT / "scripts" / "run_real_owner_chat_live_eval.py"),
                "--scenario",
                "accept_existing_offer_marks_game_ready",
                "--strict",
                "--report-path",
                str(LIVE_REPORT),
            ],
            LIVE_REPORT,
            240,
        )
    raise ValueError(f"unsupported test suite: {suite}")


def run_fixed_suite(suite: str) -> dict[str, Any]:
    """Run one allowlisted command; arbitrary shell input is intentionally impossible."""

    if not _RUN_LOCK.acquire(blocking=False):
        return {
            "ok": False,
            "suite": suite,
            "error": "another test suite is already running",
            "return_code": 409,
            "elapsed_ms": 0,
            "stdout_tail": "",
            "stderr_tail": "",
        }
    try:
        return _run_fixed_suite_unlocked(suite)
    finally:
        _RUN_LOCK.release()


def _run_fixed_suite_unlocked(suite: str) -> dict[str, Any]:
    """Execute the selected suite after the process-wide run lock is held."""

    command, report_path, timeout_seconds = _command_for_suite(suite)
    started_at = datetime.now(timezone.utc)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        timed_out = False
        stdout = completed.stdout
        stderr = completed.stderr
        return_code = completed.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return_code = 124

    elapsed_ms = int((datetime.now(timezone.utc) - started_at).total_seconds() * 1000)
    if suite == "focused_unit":
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(
                {
                    "status": "passed" if return_code == 0 else "failed",
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "suite": suite,
                    "command": command,
                    "return_code": return_code,
                    "elapsed_ms": elapsed_ms,
                    "stdout": stdout,
                    "stderr": stderr,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    return {
        "ok": return_code == 0,
        "suite": suite,
        "timed_out": timed_out,
        "return_code": return_code,
        "elapsed_ms": elapsed_ms,
        "command": command,
        "stdout_tail": stdout[-8_000:],
        "stderr_tail": stderr[-8_000:],
        "report": report_metadata(report_path, read_json_report(report_path)),
    }
