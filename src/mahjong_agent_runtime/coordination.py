from __future__ import annotations

import hashlib
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Protocol


class CoordinationManager(Protocol):
    def lock(self, scope: str) -> Any: ...


class InProcessCoordinationManager:
    """Serialize one logical scope inside a single Python process."""

    def __init__(self) -> None:
        self._locks: dict[str, threading.RLock] = {}
        self._guard = threading.RLock()

    def lock(self, scope: str) -> threading.RLock:
        key = scope or "default"
        with self._guard:
            return self._locks.setdefault(key, threading.RLock())


class FileCoordinationManager:
    """Serialize scopes across threads and local processes on one Mac/host."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._local = InProcessCoordinationManager()

    @contextmanager
    def lock(self, scope: str) -> Iterator[None]:
        import fcntl

        digest = hashlib.sha256((scope or "default").encode("utf-8")).hexdigest()
        path = self.directory / f"{digest}.lock"
        with self._local.lock(scope):
            with path.open("a+", encoding="utf-8") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class RedisCoordinationManager:
    """Optional shared lock for deployments with more than one host."""

    def __init__(self, redis_url: str, *, timeout_seconds: int = 300, blocking_timeout_seconds: int = 30) -> None:
        try:
            import redis
        except ImportError as exc:  # pragma: no cover - only used in distributed deployment
            raise RuntimeError("MAHJONG_REDIS_URL requires the optional 'redis' dependency") from exc
        self.client = redis.Redis.from_url(redis_url)
        self.timeout_seconds = timeout_seconds
        self.blocking_timeout_seconds = blocking_timeout_seconds

    def lock(self, scope: str) -> Any:
        digest = hashlib.sha256((scope or "default").encode("utf-8")).hexdigest()
        return self.client.lock(
            f"mahjong-agent:coordination:{digest}",
            timeout=self.timeout_seconds,
            blocking_timeout=self.blocking_timeout_seconds,
            thread_local=True,
        )


def default_coordination_manager(store: Any) -> CoordinationManager:
    redis_url = os.environ.get("MAHJONG_REDIS_URL", "").strip()
    if redis_url:
        return RedisCoordinationManager(redis_url)
    path = getattr(store, "path", None)
    if path:
        return FileCoordinationManager(Path(path).parent / ".runtime-locks")
    return InProcessCoordinationManager()
