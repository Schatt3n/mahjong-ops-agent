"""定义群话题的事实账本、来源幂等与个人目标绑定合同。"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol


class GroupTopicStore(Protocol):
    """话题只负责检索与审计；任何建局、入局和取消仍由领域工具执行。"""

    def get_group_topic(self, topic_id: str) -> dict | None:
        """读取包括冷话题在内的事实快照。"""
        ...

    def get_group_topic_message(self, channel: str, room_id: str, customer_id: str, message_id: str) -> dict | None:
        """读取完整来源作用域下不可变的原文与分类结果。"""
        ...

    def get_group_topic_for_message_reference(self, channel: str, room_id: str, message_id: str) -> dict | None:
        """精确引用唯一归属的公开话题；编号歧义或未知时不猜测，不返回原文。"""
        ...

    def record_group_topic_message(
        self, *, channel: str, room_id: str, customer_id: str, message_id: str,
        text: str, sent_at: datetime, intent: str, topic_id: str | None = None,
        public_summary: dict | None = None, classification: dict | None = None,
        expected_version: int | None = None, trace_id: str = '', idle_seconds: int = 1800,
        create_topic: bool = True,
    ) -> dict:
        """原子记录来源与话题 CAS；重放返回原分类且不更新活跃时间。"""
        ...

    def list_group_topics(self, *, room_id: str, customer_id: str | None = None,
                          at: datetime | None = None, limit: int = 5, include_cold: bool = False) -> list[dict]:
        """按活动时间返回有限候选；指定客户时只查询其关联话题。"""
        ...

    def list_group_topic_customer_messages(self, topic_id: str, customer_id: str, *, limit: int = 6) -> list[dict]:
        """仅返回当前客户本人在话题中的来源消息，不暴露他人的原文。"""
        ...

    def get_group_topic_customer(self, topic_id: str, customer_id: str) -> dict | None:
        """读取本人最新绑定，供服务层定位当前业务目标。"""
        ...

    def list_group_topic_customer_bindings(self, topic_id: str, customer_id: str) -> list[dict]:
        """保留本人同话题多目标和显式跨通道后的绑定历史。"""
        ...

    def bind_group_topic_customer(
        self, *, topic_id: str, customer_id: str, conversation_id: str,
        task_context_id: str | None = None, game_ids: tuple[str, ...] = (),
        source_message_id: str | None = None, allow_conversation_change: bool = False,
        at: datetime | None = None,
    ) -> dict:
        """验证任务归属与引用后写入关联；不会把话题发言者变成入局者。"""
        ...
