"""组合运行时所需的存储能力，隔离上层服务与具体持久化实现。"""

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
    """由八个细分 ``Protocol`` 组合而成的结构化接口，不包含实现代码。

    实现：生产组合根是 ``stores.sqlite.SQLiteAgentStore``，测试/本地替身是
    ``stores.memory.InMemoryAgentStore``。两者通过 Mixin 提供同名方法，无需显式继承
    本类，因此 IDE 的“转到实现”可能找不到；应继续到两个 ``store.py`` 的继承列表。

    边界：``runtime_checkable`` 运行时只检查属性存在，不校验完整参数签名；修改接口时
    必须同时核对两个实现及真实调用方。``BaseStore`` 只是历史导入别名。
    """


BaseStore = AgentStore
