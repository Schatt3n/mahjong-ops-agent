"""将已持久化的语义判断映射到后端通道策略；不按客户话术匹配关键词。"""

from __future__ import annotations

import os
import random
from collections.abc import Callable, Sequence

from ..stores import AgentStore
from .models import GroupMessage, PrivateSwitchContext, ReplyConstraints, RoutingDecision


class L2IntentRouter:
    """语义模型只提出意图和任务关联，好友权限与回复通道始终由后端决定。"""

    PRIVATE_ACKS = ("好的私聊跟你说", "私聊回你", "好的私聊说")

    def __init__(self, store: AgentStore, *, llm=None, classifier=None, cache=None,
                 topic_service=None, ack_picker: Callable[[Sequence[str]], str] | None = None) -> None:
        """允许测试注入分类器；没有模型时显式报错，禁止退回自然语言规则树。"""
        self.store = store
        self.ack_picker = ack_picker or random.choice
        self.topic_service = topic_service
        self.llm = llm
        self.classifier = classifier
        self.cache = cache

    def _service(self):
        """延迟装配有界路由服务，严格协议输入无需调用分类模型。"""
        if self.topic_service is None:
            from .semantic_topic_router import SemanticTopicClassifier
            from .topic_service import TopicRoutingService
            if self.classifier is None:
                if self.llm is None:
                    raise RuntimeError("semantic group routing requires a classifier or llm")
                self.classifier = SemanticTopicClassifier(self.llm)
            if self.cache is None and os.environ.get("MAHJONG_TOPIC_REDIS_URL"):
                from .topic_cache import RedisGroupTopicCache
                self.cache = RedisGroupTopicCache.from_url(os.environ["MAHJONG_TOPIC_REDIS_URL"])
            self.topic_service = TopicRoutingService(self.store, self.classifier, cache=self.cache)
        return self.topic_service

    def route(self, msg: GroupMessage, *, trace_id: str = "group_route") -> RoutingDecision:
        """先记录语义来源，再选择通道；相同来源重试沿用已有分类。"""
        receipt = self._service().classify(msg, trace_id=trace_id)
        decision = receipt["message"]["classification"]
        intent = decision["intent"]
        if intent == "chitchat":
            return RoutingDecision(action="ignore", topic_routing=receipt)
        identity = self.store.get_channel_identity(msg.channel, msg.sender_external_id)
        topic = receipt.get("topic")
        bindings = self.store.list_group_topic_customer_bindings(topic["topic_id"], identity.customer_id) if topic and identity else []
        binding = next((item for item in bindings
                        if item.get("task_context_id") == decision.get("task_context_id")
                        and item.get("task_context_id")), None)
        # 既有私聊任务的群内补充仍送入该私聊。新的只读询局继续群内公开答复，
        # 不能因为这个人是好友就把所有群输入悄悄归入私聊历史。
        continuing_private = bool(binding and identity and
            binding["conversation_id"] == identity.private_conversation_id and
            decision.get("goal_relation") == "continue")
        private = bool(identity and identity.can_private_message and
                       (intent in {"delegate", "create", "claim"} or continuing_private))
        if private:
            return RoutingDecision(
                action="private_switch", channel="private", group_ack=self.ack_picker(self.PRIVATE_ACKS),
                switch_context=PrivateSwitchContext(
                    private_conversation_id=identity.private_conversation_id,
                    trigger_summary="用户从群内提出本人需求，按来源记录继续处理。",
                    customer_profile={}, user_original_text=msg.text, parsed_need={}, missing_info=[],
                    reply_guidelines="根据本人原话和真实状态推进，不复述已明确信息。",
                ), topic_routing=receipt,
            )
        limit = 20 if intent == "query" else 10 if intent == "cancel" else 50
        return RoutingDecision(action="agent_loop", channel="group",
                               reply_constraints=ReplyConstraints(max_length=limit), topic_routing=receipt)


__all__ = ["L2IntentRouter"]
