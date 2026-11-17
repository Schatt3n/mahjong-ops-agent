"""消费已经持久接收的渠道消息，限制并行会话数并保持会话内顺序。"""

from __future__ import annotations

import logging
import threading
from time import monotonic
from typing import Any, Callable

from .coordination import CoordinationManager


class ChannelIngressWorker:
    """并行槽只分配给不同会话，停止或崩溃后未完成的观察仍留在 Store。"""

    def __init__(
        self,
        *,
        store: Any,
        coordination: CoordinationManager,
        handler: Callable[[dict], None],
        max_workers: int = 4,
    ) -> None:
        """绑定已持久化的队列与应用层路由，worker 不理解通道业务规则。"""
        self.store = store
        self.coordination = coordination
        self.handler = handler
        self.max_workers = max(1, max_workers)
        self._changed = threading.Condition()
        self._active: set[str] = set()
        self._threads: list[threading.Thread] = []
        self._stopping = False

    def start(self) -> None:
        """随 HTTP 应用启动固定数量的消费线程，同时拾取上次留下的记录。"""
        with self._changed:
            for index in range(self.max_workers):
                thread = threading.Thread(target=self._run, name=f"channel-ingress-{index}", daemon=True)
                thread.start()
                self._threads.append(thread)

    def wake(self) -> None:
        """提交事务后通知空闲线程；短轮询也能看见其他进程写入的记录。"""
        with self._changed:
            self._changed.notify_all()

    def stop(self, timeout_seconds: float = 2.0) -> None:
        """停止领取，给已开始的路由有限退出时间；待办依然可从数据库恢复。"""
        with self._changed:
            self._stopping = True
            self._changed.notify_all()
        deadline = monotonic() + timeout_seconds
        for thread in self._threads:
            thread.join(max(0, deadline - monotonic()))

    def _take(self) -> dict | None:
        """选取最早的空闲会话并占用槽位，模型调用期间不占用调度锁。"""
        with self._changed:
            while not self._stopping:
                record = self.store.next_channel_ingress(self._active)
                if record is not None:
                    self._active.add(record["conversation_id"])
                    return record
                self._changed.wait(timeout=0.25)
            return None

    def _run(self) -> None:
        """跨进程锁内复查待办，防止两个实例同时路由同一来源消息。"""
        while True:
            conversation_id = None
            try:
                record = self._take()
                if record is None:
                    return
                conversation_id = record["conversation_id"]
                with self.coordination.lock(f"channel-ingress:{conversation_id}"):
                    if self.store.channel_ingress_pending(record["channel"], record["source_message_id"]):
                        self.handler(record)
            except Exception:
                logging.getLogger(__name__).exception("Channel ingress worker could not complete its storage operation")
                with self._changed:
                    if not self._stopping:
                        self._changed.wait(timeout=0.25)
            finally:
                if conversation_id is not None:
                    with self._changed:
                        self._active.remove(conversation_id)
                        self._changed.notify_all()
