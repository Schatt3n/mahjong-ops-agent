from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from .stores.sqlite.idempotency import SQLiteIdempotencyStoreMixin
from .stores.sqlite.accessors import SQLiteAccessorsStoreMixin
from .stores.sqlite.customer import SQLiteCustomerStoreMixin
from .stores.sqlite.rooms import SQLiteRoomsStoreMixin
from .stores.sqlite.conversation import SQLiteConversationStoreMixin
from .stores.sqlite.task_memory import SQLiteTaskMemoryStoreMixin
from .stores.sqlite.scheduling import SQLiteSchedulingStoreMixin
from .stores.sqlite.input_aggregation import SQLiteInputAggregationStoreMixin
from .stores.sqlite.references import SQLiteReferencesStoreMixin
from .stores.sqlite.administration import SQLiteAdministrationStoreMixin
from .stores.sqlite.game_queries import SQLiteGameQueriesStoreMixin
from .stores.sqlite.game_mutations import SQLiteGameMutationsStoreMixin
from .stores.sqlite.drafts import SQLiteDraftsStoreMixin
from .stores.sqlite.game_persistence import SQLiteGamePersistenceStoreMixin
from .stores.sqlite.persistence import SQLitePersistenceStoreMixin
from .stores.sqlite.migration import SQLiteMigrationStoreMixin


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
        """Acquire SQLite's write reservation before reading mutable invariants."""

        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield
            except Exception:
                self._connection.rollback()
                raise
            else:
                self._connection.commit()
