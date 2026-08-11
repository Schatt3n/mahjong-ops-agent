"""定义base相关的存储接口，隔离领域层与具体持久化实现。

"""

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
    """定义智能体存储需要遵守的抽象接口契约。

"""


BaseStore = AgentStore
