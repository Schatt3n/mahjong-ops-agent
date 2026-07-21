"""Composed storage protocol used by the Agent runtime."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .conversation_store import ConversationStore
from .customer_store import CustomerStore
from .game_store import GameStore
from .idempotency_store import IdempotencyStore
from .task_store import TaskStore


@runtime_checkable
class AgentStore(
    CustomerStore,
    GameStore,
    ConversationStore,
    TaskStore,
    IdempotencyStore,
    Protocol,
):
    """Structural contract shared by in-memory and SQLite backends."""


BaseStore = AgentStore

