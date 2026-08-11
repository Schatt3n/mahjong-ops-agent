#!/usr/bin/env python3
"""按项目约定顺序运行静态检查、单元测试与评测套件。"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    env = os.environ.copy()
    src_path = str(ROOT / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{src_path}:{existing}" if existing else src_path
    return subprocess.run([sys.executable, "scripts/run_evals.py"], cwd=ROOT, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
