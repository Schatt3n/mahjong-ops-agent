"""Compatibility SQLite store imports for historical ``mahjong_agent_v3`` users."""

from __future__ import annotations

from mahjong_agent_runtime.sqlite_store import SQLiteAgentStore as SQLiteAgentStoreV3

__all__ = ["SQLiteAgentStoreV3"]
