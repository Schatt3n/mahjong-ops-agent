#!/usr/bin/env python3
"""Compatibility entrypoint for historical Agent Runtime V3 commands.

The current implementation lives in ``agent_runtime_app.py`` and exposes the
versionless ``mahjong_agent_runtime`` surface.
"""

from __future__ import annotations

from agent_runtime_app import *  # noqa: F403
from agent_runtime_app import main


if __name__ == "__main__":
    main()
