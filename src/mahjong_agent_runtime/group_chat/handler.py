"""生产群聊消息入口：协调规则分流、私聊切换和主智能体调用。"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

from ..models import UserMessage, new_id, now
from .board_engine import BoardEngine
from .claim_handler import ClaimHandler
from .intent_router import L2IntentRouter
from .messenger import GroupMessenger
from .models import ChannelSwitch, GroupHandleResult, GroupMessage, RoutingDecision
from .notify_dispatcher import NotifyDispatcher
from .rule_engine import L1RuleEngine
from .topic_context import selected_topic_binding


class GroupMessageHandler:
    """群聊统一入口，在共享房间内隔离公告、认领和用户对话。

    群聊不是一条属于所有成员的连续会话。本类先处理可确定执行的看板协议，
    再判断是否应切到私聊，只有确实需要开放式决策时才进入主 AgentLoop。
    因而不会把整个群的历史消息直接当成某一位用户的上下文。

    导航：Wechaty 托管群由应用层 ``route_wechaty_raw_to_agent`` 进入这里；当前生产链
    没有装配共享原文的 ``GroupSessionPipeline``。决策顺序是 L1 完整发局/编号认领协议
    -> 有界语义路由与持久话题关联 -> 后端通道选择 -> 单成员隔离的主 Agent。
    """

    def __init__(
        self,
        *,
        rule_engine: L1RuleEngine,
        intent_router: L2IntentRouter,
        board_engine: BoardEngine,
        claim_handler: ClaimHandler,
        notify_dispatcher: NotifyDispatcher,
        messenger: GroupMessenger,
        runtime: Any,
        clock: Callable[[], datetime] = now,
    ) -> None:
        self.rule_engine = rule_engine
        self.intent_router = intent_router
        self.board_engine = board_engine
        self.claim_handler = claim_handler
        self.notify_dispatcher = notify_dispatcher
        self.messenger = messenger
        self.runtime = runtime
        self.clock = clock
        self.store = intent_router.store

    def handle(self, msg: GroupMessage, *, trace_id: str) -> GroupHandleResult:
        """按 L1 确定性协议、通道路由、主 Agent 的优先级处理一条群消息。"""

        # 看板导入和编号认领属于高置信度协议，优先执行可减少延迟与模型误判。
        l1_result = self.rule_engine.process(msg)
        if l1_result.action == "ignore":
            return GroupHandleResult(action="ignore")
        if l1_result.action == "board_import":
            game = self.board_engine.import_game_from_post(msg, trace_id=trace_id)
            reply = ""
            if l1_result.needs_ack:
                reply = "好的"
                self.messenger.send_group_message(
                    msg.room_id,
                    reply,
                    metadata={"trace_id": trace_id, "source_message_id": msg.message_id},
                )
            return GroupHandleResult(action="board_import", game_id=game.game_id, reply=reply)
        if l1_result.action == "claim" and l1_result.item_no is not None:
            claim = self.claim_handler.process_claim(
                msg,
                l1_result.item_no,
                trace_id=trace_id,
            )
            return GroupHandleResult(
                action="claim",
                game_id=claim.game_id,
                detail={
                    "status": claim.status,
                    "reason": claim.reason,
                    "deduplicated": claim.deduplicated,
                },
            )

        try:
            routing = self.intent_router.route(msg, trace_id=trace_id)
        except Exception as exc:
            # 语义分类在主循环之前执行，失败也必须保留原始来源与异常证据；否则面板
            # 只能看到 RuntimeError 类型，无法区分供应商超时与合同未恢复。只写内部trace。
            recorder = getattr(self.runtime, "trace_recorder", None)
            if recorder is None:
                recorder = getattr(getattr(self.runtime, "runtime", None), "trace_recorder", None)
            if recorder is not None:
                recorder.record(trace_id, "group_semantic_routing_error", {
                    "source_message": msg.to_dict(), "error_type": type(exc).__name__,
                    "error": str(exc), "audit": getattr(exc, "audit", None),
                }, level="ERROR")
            raise
        if routing.action == "ignore":
            return GroupHandleResult(action="ignore", detail={"topic_routing": routing.topic_routing})
        if routing.action == "private_switch":
            return self._handle_private_switch(msg, routing, trace_id=trace_id)
        if routing.action == "agent_loop":
            return self._handle_agent_loop(msg, routing, trace_id=trace_id)
        raise ValueError("unsupported group routing action")

    def _handle_private_switch(
        self,
        msg: GroupMessage,
        routing: RoutingDecision,
        *,
        trace_id: str,
    ) -> GroupHandleResult:
        """将真实群输入送入本人私聊任务，推进输入版本而不是冒充系统通知。"""
        identity = self.store.get_channel_identity(msg.channel, msg.sender_external_id)
        if identity is None or not identity.can_private_message or routing.switch_context is None:
            raise ValueError("private route has no authorized identity")
        if not routing.topic_routing.get("deduplicated"):
            self.messenger.send_group_message(
                msg.room_id, routing.group_ack,
                metadata={"trace_id": trace_id, "source_message_id": msg.message_id},
            )
        self.store.record_channel_switch(ChannelSwitch(
            switch_id=new_id("channel_switch"), room_id=msg.room_id, customer_id=identity.customer_id,
            source_conversation_id=msg.conversation_id, source_message_id=msg.message_id,
            private_conversation_id=identity.private_conversation_id,
            trigger_summary="从群内接收本人需求，私聊继续处理。",
            created_at=self.clock(), expires_at=self.clock() + timedelta(minutes=10),
        ))
        message = self._user_input(msg, routing, identity.private_conversation_id, identity.customer_id, private=True)
        result = self.runtime.handle_user_message(message, trace_id=trace_id)
        self._bind_result(msg, routing, message, result)
        self.notify_dispatcher.send_private_switch_reply(identity, result, trace_id=trace_id, source_message=msg)
        return GroupHandleResult(action="private_switch", reply=routing.group_ack,
            detail={"private_conversation_id": identity.private_conversation_id, "topic_routing": routing.topic_routing})

    def _handle_agent_loop(self, msg: GroupMessage, routing: RoutingDecision, *, trace_id: str) -> GroupHandleResult:
        """每位群成员拥有独立会话；只把当前原话和可验证的话题引用交给主 Agent。"""
        identity = self.store.get_channel_identity(msg.channel, msg.sender_external_id)
        sender_id = identity.customer_id if identity else f"{msg.channel}:{msg.sender_external_id}"
        conversation_id = f"group:{msg.room_id}:customer:{sender_id}"
        message = self._user_input(msg, routing, conversation_id, sender_id, private=False)
        result = self.runtime.handle_user_message(message, trace_id=trace_id)
        self._bind_result(msg, routing, message, result)
        if result.final_reply:
            self.messenger.send_group_message(msg.room_id, result.final_reply,
                metadata={"trace_id": trace_id, "source_message_id": msg.message_id})
        return GroupHandleResult(action="agent_loop", reply=result.final_reply,
                                 detail={"topic_routing": routing.topic_routing})

    def _user_input(self, msg, routing, conversation_id, customer_id, *, private):
        """保留消息原文、时间和来源；分类不是授权，ContextBuilder 会回查持久凭据。"""
        metadata = {
            "source": "group_private_continuation" if private else "group",
            "source_conversation_id": msg.conversation_id,
            "group_topic_source": {"channel": msg.channel, "room_id": msg.room_id,
                                   "external_user_id": msg.sender_external_id},
        }
        if not private:
            metadata.update(room_id=msg.room_id, reply_constraints=routing.reply_constraints.to_dict())
        return UserMessage(conversation_id=conversation_id, sender_id=customer_id, sender_name=msg.sender_name,
                           text=msg.text, message_id=msg.message_id, sent_at=msg.sent_at, metadata=metadata)

    def _bind_result(self, msg, routing, message, result):
        """保存本人任务与真实牌局引用；话题本身不创建牌局，也不更改参加状态。"""
        topic = routing.topic_routing.get("topic")
        if topic is None:
            return
        binding = selected_topic_binding(self.store, routing.topic_routing.get("message"), message.conversation_id)
        task = self.store.get_task_context(binding["task_context_id"]) if binding else None
        # 真实运行时在接收输入时已选好任务；测试替身没有任务时仍可记录通道关联。
        prior = binding
        game_ids = set(prior["game_ids"] if prior and task and prior.get("task_context_id") == task.task_context_id else [])
        game_ids.update(t.entity_id for t in result.state_transitions if t.entity_type == "game"
                        and t.entity_id in self.store.games and self.store.games[t.entity_id].organizer_id == message.sender_id)
        self.store.bind_group_topic_customer(topic_id=topic["topic_id"], customer_id=message.sender_id,
            conversation_id=message.conversation_id, task_context_id=task.task_context_id if task else None,
            game_ids=sorted(game_ids), source_message_id=msg.message_id, allow_conversation_change=True, at=msg.sent_at)


__all__ = ["GroupMessageHandler"]
