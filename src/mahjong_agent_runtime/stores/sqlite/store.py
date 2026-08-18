"""组合生产 SQLite Store，并提供所有持久化 Mixin 共用的连接与事务。"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from .accessors import SQLiteAccessorsStoreMixin
from .agent_runs import SQLiteAgentRunStoreMixin
from .administration import SQLiteAdministrationStoreMixin
from .channel_observations import SQLiteChannelObservationsStoreMixin
from .conversation import SQLiteConversationStoreMixin
from .customer import SQLiteCustomerStoreMixin
from .drafts import SQLiteDraftsStoreMixin
from .game_mutations import SQLiteGameMutationsStoreMixin
from .game_persistence import SQLiteGamePersistenceStoreMixin
from .game_queries import SQLiteGameQueriesStoreMixin
from .group_chat import SQLiteGroupChatStoreMixin
from .idempotency import SQLiteIdempotencyStoreMixin
from .input_aggregation import SQLiteInputAggregationStoreMixin
from .schema import SQLiteSchemaStoreMixin
from .persistence import SQLitePersistenceStoreMixin
from .references import SQLiteReferencesStoreMixin
from .rooms import SQLiteRoomsStoreMixin
from .scheduling import SQLiteSchedulingStoreMixin
from .task_memory import SQLiteTaskMemoryStoreMixin
from .waiting import SQLiteWaitingDemandStoreMixin


@dataclass(slots=True)
class SQLiteAgentStore(
    SQLiteAgentRunStoreMixin,
    SQLiteAccessorsStoreMixin,
    SQLiteChannelObservationsStoreMixin,
    SQLiteCustomerStoreMixin,
    SQLiteRoomsStoreMixin,
    SQLiteConversationStoreMixin,
    SQLiteTaskMemoryStoreMixin,
    SQLiteSchedulingStoreMixin,
    SQLiteInputAggregationStoreMixin,
    SQLiteReferencesStoreMixin,
    SQLiteAdministrationStoreMixin,
    SQLiteGameQueriesStoreMixin,
    SQLiteGameMutationsStoreMixin,
    SQLiteGroupChatStoreMixin,
    SQLiteDraftsStoreMixin,
    SQLiteWaitingDemandStoreMixin,
    SQLiteGamePersistenceStoreMixin,
    SQLitePersistenceStoreMixin,
    SQLiteSchemaStoreMixin,
    SQLiteIdempotencyStoreMixin,
):
    """正式运行时使用的 Store 组合根。

    导航：``scripts.agent_runtime_app.build_runtime`` 创建本类。具体方法位于继承列表中的
    ``SQLite*StoreMixin``；这些 Mixin 依赖本类提供的连接、锁和写事务，不能独立实例化。
    常见跨 Mixin 协作：``game_mutations`` 会调用 ``game_persistence/scheduling/rooms``，
    ``group_chat`` 会调用 ``drafts/persistence``，会话与输入聚合复用 ``persistence``。
    """

    path: str | Path
    _connection: sqlite3.Connection = field(init=False, repr=False)
    _lock: threading.RLock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """打开数据库，启用 WAL/外键/忙等待，并初始化 schema。"""
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(self.path), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA foreign_keys=ON")
            self._connection.execute("PRAGMA busy_timeout=5000")
            try:
                self._initialize_schema()
            except Exception:
                self._connection.close()
                raise

    @contextmanager
    def _write_transaction(self):
        """用 ``BEGIN IMMEDIATE`` 原子包住“读不变量 -> 修改 -> 写 transition”。

        需要跨表不变量或抢占写锁的 Mixin 路径共用该入口；调用方不应在其内部再次
        开启写事务。简单单表写入也可能直接在 ``_lock`` 下使用连接。
        """

        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield
            except Exception:
                self._connection.rollback()
                raise
            else:
                self._connection.commit()


__all__ = ["SQLiteAgentStore"]
