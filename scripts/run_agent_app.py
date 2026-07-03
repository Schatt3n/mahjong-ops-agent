#!/usr/bin/env python3
"""Primary local entrypoint for the Mahjong Agent Runtime."""

from __future__ import annotations

import os
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

if "MAHJONG_AGENT_PORT" in os.environ and "MAHJONG_AGENT_V3_PORT" not in os.environ:
    os.environ["MAHJONG_AGENT_V3_PORT"] = os.environ["MAHJONG_AGENT_PORT"]

if "MAHJONG_AGENT_DB_PATH" in os.environ and "MAHJONG_AGENT_V3_DB_PATH" not in os.environ:
    os.environ["MAHJONG_AGENT_V3_DB_PATH"] = os.environ["MAHJONG_AGENT_DB_PATH"]

from run_agent_v3_app import main  # noqa: E402


if __name__ == "__main__":
    main()
