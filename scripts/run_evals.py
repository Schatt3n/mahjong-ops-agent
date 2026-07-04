#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REAL_OWNER_LIVE_EVAL_REPORT = ROOT / "runtime_data" / "real_owner_chat_live_eval_report.json"


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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_command([sys.executable, "scripts/verify_agent_runtime_boundary.py"])
    run_command([sys.executable, "scripts/run_agent_runtime_eval.py"])
    run_command([sys.executable, "-m", "pytest", "-q", "tests/test_agent_runtime.py"])
    run_command([sys.executable, "-m", "pytest", "-q", "tests/test_real_owner_chat_golden.py"])
    run_command([sys.executable, "-m", "pytest", "-q", "tests/test_context_summary.py"])
    run_command([sys.executable, "-m", "pytest", "-q", "tests/test_agent_runtime_eval.py"])
    run_command([sys.executable, "-m", "pytest", "-q", "tests/test_agent_app.py"])
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
