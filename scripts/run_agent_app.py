#!/usr/bin/env python3
"""启动 Mahjong Agent Runtime 本地运营控制台。"""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from agent_runtime_app import main  # noqa: E402


if __name__ == "__main__":
    main()
