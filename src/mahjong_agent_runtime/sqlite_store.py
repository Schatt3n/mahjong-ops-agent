"""Backward-compatible import for :class:`stores.sqlite.SQLiteAgentStore`."""

from .stores.sqlite import SQLiteAgentStore

__all__ = ["SQLiteAgentStore"]
