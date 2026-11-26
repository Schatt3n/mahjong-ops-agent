"""实现内存存储层的通道observations数据访问能力，主要用于本地运行和测试。

"""

from __future__ import annotations

from copy import deepcopy
from time import time
from typing import Any


class InMemoryChannelObservationsStoreMixin:
    """镜像持久化观察API，且不将其混入Agent轮次。"""

    __slots__ = ()

    def upsert_channel_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        """新增或更新通道observation，使重复写入收敛到同一条记录。"""
        channel = str(observation.get("channel") or "").strip()
        source_message_id = str(observation.get("source_message_id") or "").strip()
        if not channel:
            raise ValueError("channel observation requires channel")
        if not source_message_id:
            raise ValueError("channel observation requires source_message_id")
        record = dict(observation)
        record["channel"] = channel
        record["source_message_id"] = source_message_id
        record["payload"] = dict(record.get("payload") or {})
        with self._lock:
            self.channel_observations[f"{channel}:{source_message_id}"] = record
        return dict(record)

    def enqueue_channel_ingress(self, observation: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """用相同消息键保留首次入站观察及原有插入顺序。"""
        key = f'{observation["channel"]}:{observation["source_message_id"]}'
        with self._lock:
            existing = self.channel_observations.get(key)
            if existing is not None:
                return deepcopy(existing), False
            return self.upsert_channel_observation(deepcopy(observation)), True

    def next_channel_ingress(
        self, excluded_conversations: set[str], *, route_mode: str | None = None,
    ) -> dict[str, Any] | None:
        """按阶段返回会话头部待办，其他阶段或等待重试的头部保留会话顺序。"""
        blocked = set(excluded_conversations)
        current_time = time()
        with self._lock:
            for record in self.channel_observations.values():
                conversation_id = record["conversation_id"]
                if conversation_id in blocked or record["route_mode"] not in {"ingress_queue", "ingress_reply_pending"}:
                    continue
                if ((route_mode is None or record["route_mode"] == route_mode)
                        and record["payload"].get("deferred_reply", {}).get("retry_at", 0) <= current_time):
                    return deepcopy(record)
                blocked.add(conversation_id)
        return None

    def channel_ingress_record(self, channel: str, source_message_id: str) -> dict[str, Any] | None:
        """持有消费锁后重新读取阶段和重试时间，防止重复路由。"""
        with self._lock:
            record = self.channel_observations.get(f"{channel}:{source_message_id}")
            if (record is None or record["route_mode"] not in {"ingress_queue", "ingress_reply_pending"}
                    or record["payload"].get("deferred_reply", {}).get("retry_at", 0) > time()):
                return None
            return deepcopy(record)

    def get_channel_observation(self, channel: str, source_message_id: str) -> dict[str, Any] | None:
        """按通道和原生消息标识精确读取观测副本，避免外部修改内存记录。"""
        with self._lock:
            record = self.channel_observations.get(f"{channel}:{source_message_id}")
            return deepcopy(record) if record is not None else None

    def has_queued_channel_ingress(self, channel: str, conversation_id: str) -> bool:
        """判断指定会话是否仍有已接收、尚未完成路由的原始消息。"""
        with self._lock:
            return any(
                row["channel"] == channel and row["conversation_id"] == conversation_id
                and row["route_mode"] == "ingress_queue"
                for row in self.channel_observations.values()
            )

    def list_channel_observations(
        self,
        *,
        channel: str | None = None,
        room_id: str | None = None,
        room_topic_keyword: str | None = None,
        semantic_action: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """列出通道observations；结果保持稳定的领域顺序。"""
        with self._lock:
            records = [dict(item) for item in self.channel_observations.values()]
        if channel:
            records = [item for item in records if item.get("channel") == channel]
        if room_id:
            records = [item for item in records if item.get("room_id") == room_id]
        if room_topic_keyword:
            keyword = room_topic_keyword.casefold()
            records = [item for item in records if keyword in str(item.get("room_topic") or "").casefold()]
        if semantic_action:
            records = [item for item in records if item.get("semantic_action") == semantic_action]
        records.sort(key=lambda item: str(item.get("received_at") or ""), reverse=True)
        return records[: max(1, min(int(limit), 1_000))]


__all__ = ["InMemoryChannelObservationsStoreMixin"]
