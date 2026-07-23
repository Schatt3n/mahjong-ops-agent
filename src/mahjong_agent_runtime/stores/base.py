"""Composed storage protocol used by the Agent runtime."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .agent_run_store import AgentRunStore
from .conversation_store import ConversationStore
from .customer_store import CustomerStore
from .game_store import GameStore
from .group_chat_store import GroupChatStore
from .idempotency_store import IdempotencyStore
from .task_store import TaskStore
from .waiting_store import WaitingDemandStore


@runtime_checkable
class AgentStore(
    AgentRunStore,
    CustomerStore,
    GameStore,
    ConversationStore,
    TaskStore,
    IdempotencyStore,
    WaitingDemandStore,
    GroupChatStore,
    Protocol,
):
    """Structural contract shared by in-memory and SQLite backends."""


BaseStore = AgentStore
