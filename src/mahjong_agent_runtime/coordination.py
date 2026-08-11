"""提供进程内、单机跨进程和 Redis 分布式三种会话协调锁。"""

from __future__ import annotations

import hashlib
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Protocol


class CoordinationManager(Protocol):
    """定义按业务作用域串行化操作的统一锁接口。"""

    def lock(self, scope: str) -> Any:
        """获取指定作用域的互斥锁，调用方通过上下文管理器保证释放。"""
        ...


class InProcessCoordinationManager:
    """在单个 Python 进程内按作用域串行执行。

"""

    def __init__(self) -> None:
        """初始化作用域锁表以及保护锁表自身的可重入锁。"""
        self._locks: dict[str, threading.RLock] = {}
        self._guard = threading.RLock()

    def lock(self, scope: str) -> threading.RLock:
        """返回作用域专属锁，不同会话或局可以并行执行。"""
        key = scope or "default"
        with self._guard:
            return self._locks.setdefault(key, threading.RLock())


class FileCoordinationManager:
    """使用文件锁协调同一台 Mac 上的多个线程和进程。

"""

    def __init__(self, directory: str | Path) -> None:
        """创建锁目录，并准备进程内锁以协调同进程线程。"""
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._local = InProcessCoordinationManager()

    @contextmanager
    def lock(self, scope: str) -> Iterator[None]:
        """先取得进程内锁，再用哈希命名的文件锁协调其他本机进程。"""
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
    """使用 Redis 分布式锁协调部署在多台机器上的运行时。

"""

    def __init__(self, redis_url: str, *, timeout_seconds: int = 300, blocking_timeout_seconds: int = 30) -> None:
        """创建 Redis 客户端，并配置锁租约与最长等待时间。"""
        try:
            import redis
        except ImportError as exc:  # pragma: no cover - 仅分布式部署会进入此分支
            raise RuntimeError("MAHJONG_REDIS_URL requires the optional 'redis' dependency") from exc
        self.client = redis.Redis.from_url(redis_url)
        self.timeout_seconds = timeout_seconds
        self.blocking_timeout_seconds = blocking_timeout_seconds

    def lock(self, scope: str) -> Any:
        """返回以作用域哈希命名的 Redis 锁，避免把原始业务标识暴露为键名。"""
        digest = hashlib.sha256((scope or "default").encode("utf-8")).hexdigest()
        return self.client.lock(
            f"mahjong-agent:coordination:{digest}",
            timeout=self.timeout_seconds,
            blocking_timeout=self.blocking_timeout_seconds,
            thread_local=True,
        )


def default_coordination_manager(store: Any) -> CoordinationManager:
    """按 Redis、多进程文件锁、单进程锁的优先级选择协调实现。"""
    redis_url = os.environ.get("MAHJONG_REDIS_URL", "").strip()
    if redis_url:
        return RedisCoordinationManager(redis_url)
    path = getattr(store, "path", None)
    if path:
        database_path = Path(path)
        # 不同数据库可以合法复用同一个 conversation_id，因此锁目录包含数据库身份；
        # 但所有打开同一数据库的本机进程仍会落到相同锁命名空间。
        return FileCoordinationManager(
            database_path.parent / f".{database_path.name}.runtime-locks"
        )
    return InProcessCoordinationManager()
