#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_command(args: list[str]) -> None:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    src_path = str(ROOT / "src")
    env["PYTHONPATH"] = f"{src_path}:{existing_pythonpath}" if existing_pythonpath else src_path
    print(f"\n$ {' '.join(args)}", flush=True)
    subprocess.run(args, cwd=ROOT, env=env, check=True)


def main() -> int:
    run_command([sys.executable, "scripts/verify_agent_runtime_boundary.py"])
    run_command([sys.executable, "scripts/run_agent_runtime_eval.py"])
    run_command([sys.executable, "-m", "pytest", "-q", "tests/test_agent_runtime.py"])
    run_command([sys.executable, "-m", "pytest", "-q", "tests/test_real_owner_chat_golden.py"])
    run_command([sys.executable, "-m", "pytest", "-q", "tests/test_context_summary.py"])
    run_command([sys.executable, "-m", "pytest", "-q", "tests/test_agent_runtime_eval.py"])
    run_command([sys.executable, "-m", "pytest", "-q", "tests/test_agent_app.py"])
    run_command([sys.executable, "-m", "pytest", "-q", "tests/test_agent_runtime_package.py"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
