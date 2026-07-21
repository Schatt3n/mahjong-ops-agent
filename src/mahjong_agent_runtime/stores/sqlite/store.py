"""SQLite aggregate store assembled from focused persistence mixins."""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from .accessors import SQLiteAccessorsStoreMixin
from .administration import SQLiteAdministrationStoreMixin
from .conversation import SQLiteConversationStoreMixin
from .customer import SQLiteCustomerStoreMixin
from .drafts import SQLiteDraftsStoreMixin
from .game_mutations import SQLiteGameMutationsStoreMixin
from .game_persistence import SQLiteGamePersistenceStoreMixin
from .game_queries import SQLiteGameQueriesStoreMixin
from .idempotency import SQLiteIdempotencyStoreMixin
from .input_aggregation import SQLiteInputAggregationStoreMixin
from .migration import SQLiteMigrationStoreMixin
from .persistence import SQLitePersistenceStoreMixin
from .references import SQLiteReferencesStoreMixin
from .rooms import SQLiteRoomsStoreMixin
from .scheduling import SQLiteSchedulingStoreMixin
from .task_memory import SQLiteTaskMemoryStoreMixin


@dataclass(slots=True)
class SQLiteAgentStore(
    SQLiteAccessorsStoreMixin,
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
    SQLiteDraftsStoreMixin,
    SQLiteGamePersistenceStoreMixin,
    SQLitePersistenceStoreMixin,
    SQLiteMigrationStoreMixin,
    SQLiteIdempotencyStoreMixin,
):
    """Persistent AgentStore for one-node deployments and local evaluation."""

    path: str | Path
    _connection: sqlite3.Connection = field(init=False, repr=False)
    _lock: threading.RLock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(self.path), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA foreign_keys=ON")
            self._connection.execute("PRAGMA busy_timeout=5000")
            self._migrate()

    @contextmanager
    def _write_transaction(self):
        """Reserve the SQLite writer before reading mutable invariants."""

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
