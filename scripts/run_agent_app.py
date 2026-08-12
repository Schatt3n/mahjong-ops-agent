#!/usr/bin/env python3
"""启动 ``MahjongAgentApplication`` 本地运营进程。"""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from agent_runtime_app import MahjongAgentApplication  # noqa: E402


def main() -> None:
    MahjongAgentApplication.from_environment().run()


if __name__ == "__main__":
    main()
