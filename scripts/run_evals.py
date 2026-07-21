#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REAL_OWNER_LIVE_EVAL_REPORT = ROOT / "runtime_data" / "real_owner_chat_live_eval_report.json"
DETERMINISTIC_CONCURRENCY_REPORT = ROOT / "runtime_data" / "concurrency_eval_deterministic_report.json"
LIVE_CONCURRENCY_REPORT = ROOT / "runtime_data" / "concurrency_eval_live_report.json"


def run_command(args: list[str]) -> None:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    src_path = str(ROOT / "src")
    env["PYTHONPATH"] = f"{src_path}:{existing_pythonpath}" if existing_pythonpath else src_path
    print(f"\n$ {' '.join(args)}", flush=True)
    subprocess.run(args, cwd=ROOT, env=env, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run current Agent Runtime evals.")
    parser.add_argument(
        "--live-real-owner",
        action="store_true",
        help="also run real owner chat live LLM eval; requires MAHJONG_LLM_MODEL and an API key",
    )
    parser.add_argument(
        "--live-concurrency",
        action="store_true",
        help="also run the real-DeepSeek concurrent eval; requires MAHJONG_LLM_MODEL and an API key",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_command([sys.executable, "scripts/verify_agent_runtime_boundary.py"])
    run_command([sys.executable, "scripts/verify_customer_visible_contract.py"])
    run_command([sys.executable, "scripts/check_badcase_regression_coverage.py"])
    run_command([sys.executable, "scripts/validate_real_owner_chat_golden.py"])
    run_command([sys.executable, "scripts/run_agent_runtime_eval.py"])
    run_command(
        [
            sys.executable,
            "scripts/run_concurrency_eval.py",
            "--mode",
            "deterministic",
            "--operations",
            "40",
            "--workers",
            "8",
            "--strict",
            "--report-path",
            str(DETERMINISTIC_CONCURRENCY_REPORT),
        ]
    )
    run_command([sys.executable, "-m", "pytest", "-q", "tests/test_agent_runtime.py"])
    run_command([sys.executable, "-m", "pytest", "-q", "tests/test_real_owner_chat_golden.py"])
    run_command([sys.executable, "-m", "pytest", "-q", "tests/test_context_summary.py"])
    run_command([sys.executable, "-m", "pytest", "-q", "tests/test_context_summary_quality.py"])
    run_command([sys.executable, "-m", "pytest", "-q", "tests/test_agent_runtime_eval.py"])
    run_command([sys.executable, "-m", "pytest", "-q", "tests/test_agent_app.py"])
    run_command([sys.executable, "-m", "pytest", "-q", "tests/test_input_aggregation.py"])
    run_command([sys.executable, "-m", "pytest", "-q", "tests/test_progress_monitor.py"])
    run_command([sys.executable, "-m", "pytest", "-q", "tests/test_future_game_recruitment.py"])
    run_command([sys.executable, "-m", "pytest", "-q", "tests/test_action_contract_repairs.py"])
    run_command([sys.executable, "-m", "pytest", "-q", "tests/test_agent_runtime_package.py"])
    if args.live_real_owner:
        run_command(
            [
                sys.executable,
                "scripts/run_real_owner_chat_live_eval.py",
                "--strict",
                "--report-path",
                str(REAL_OWNER_LIVE_EVAL_REPORT),
            ]
        )
    if args.live_concurrency:
        run_command(
            [
                sys.executable,
                "scripts/run_concurrency_eval.py",
                "--mode",
                "live",
                "--live-workers",
                "4",
                "--live-repeats",
                "2",
                "--strict",
                "--report-path",
                str(LIVE_CONCURRENCY_REPORT),
            ]
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
