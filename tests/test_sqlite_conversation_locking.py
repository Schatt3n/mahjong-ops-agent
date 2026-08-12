from __future__ import annotations

from mahjong_agent_runtime.stores.sqlite.conversation import SQLiteConversationStoreMixin


def test_task_context_trace_lookup_holds_the_shared_connection_lock() -> None:
    class ObservedLock:
        held = False

        def __enter__(self):
            self.held = True
            return self

        def __exit__(self, *_args) -> None:
            self.held = False

    class Cursor:
        def fetchone(self) -> dict[str, str]:
            return {"task_context_id": "task_context_123"}

    class GuardedConnection:
        def __init__(self, lock: ObservedLock) -> None:
            self.lock = lock

        def execute(self, *_args) -> Cursor:
            assert self.lock.held
            return Cursor()

    class Store(SQLiteConversationStoreMixin):
        def __init__(self) -> None:
            self._lock = ObservedLock()
            self._connection = GuardedConnection(self._lock)

    store = Store()

    assert store._task_context_id_for_trace("conversation_123", "trace_123") == "task_context_123"
