"""生产群聊消息入口：协调规则分流、私聊切换和主智能体调用。"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

from ..models import SystemTriggerMessage, UserMessage, new_id, now
from .board_engine import BoardEngine
from .claim_handler import ClaimHandler
from .intent_router import L2IntentRouter
from .messenger import GroupMessenger
from .models import ChannelSwitch, GroupHandleResult, GroupMessage, RoutingDecision
from .notify_dispatcher import NotifyDispatcher
from .parsing import parse_explicit_need
from .rule_engine import L1RuleEngine


class GroupMessageHandler:
    """群聊统一入口，在共享房间内隔离公告、认领和用户对话。

    群聊不是一条属于所有成员的连续会话。本类先处理可确定执行的看板协议，
    再判断是否应切到私聊，只有确实需要开放式决策时才进入主 AgentLoop。
    因而不会把整个群的历史消息直接当成某一位用户的上下文。

    导航：Wechaty 托管群由应用层 ``route_wechaty_raw_to_agent`` 进入这里；当前生产链
    没有装配 ``GroupSessionPipeline``。决策顺序是 L1 明确发局/编号认领 -> 近期私聊
    任务重定向 -> L2 群答或切私聊 -> 单成员隔离的主 Agent。
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

        # 已经切到私聊的任务若又在群中收到补充，只做通道提醒，避免形成两份状态。
        if self._redirect_private_continuation(msg, trace_id=trace_id):
            return GroupHandleResult(action="redirect_private", reply="私聊回你了，看下哈")

        routing = self.intent_router.route(msg)
        if routing.action == "ignore":
            return GroupHandleResult(action="ignore")
        if routing.action == "private_switch":
            return self._handle_private_switch(msg, routing, trace_id=trace_id)
        if routing.action == "agent_loop":
            return self._handle_agent_loop(msg, routing, trace_id=trace_id)
        return GroupHandleResult(action="ignore")

    def _handle_private_switch(
        self,
        msg: GroupMessage,
        routing: RoutingDecision,
        *,
        trace_id: str,
    ) -> GroupHandleResult:
        """记录群转私聊关系，并以系统事件在目标私聊会话中续接任务。"""
        identity = self.store.get_channel_identity(msg.channel, msg.sender_external_id)
        context = routing.switch_context
        if identity is None or context is None:
            return GroupHandleResult(action="ignore")
        # 群中只发极短确认；需求详情和后续追问都留在用户自己的私聊上下文。
        self.messenger.send_group_message(
            msg.room_id,
            routing.group_ack,
            metadata={"trace_id": trace_id, "source_message_id": msg.message_id},
        )
        # 持久化短期切换关系，供随后到达的群消息识别“应回私聊”的任务。
        self.store.record_channel_switch(
            ChannelSwitch(
                switch_id=new_id("channel_switch"),
                room_id=msg.room_id,
                customer_id=identity.customer_id,
                source_conversation_id=msg.conversation_id,
                source_message_id=msg.message_id,
                private_conversation_id=identity.private_conversation_id,
                trigger_summary=context.trigger_summary,
                created_at=self.clock(),
                expires_at=self.clock() + timedelta(minutes=10),
            )
        )
        # 系统触发沿用正常 AgentLoop、审查和审计链路，但不伪造成新的用户消息。
        trigger = SystemTriggerMessage(
            trigger_id=new_id("private_switch"),
            trigger_type="continue_customer_request",
            conversation_id=context.private_conversation_id,
            sender_id=identity.customer_id,
            sender_name=identity.public_name,
            payload=context.to_dict(),
            created_at=self.clock(),
        )
        result = self.runtime.handle_system_trigger(trigger, trace_id=trace_id)
        self.notify_dispatcher.send_private_switch_reply(
            identity,
            result,
            trace_id=trace_id,
            room_id=msg.room_id,
        )
        return GroupHandleResult(
            action="private_switch",
            reply=routing.group_ack,
            detail={"private_conversation_id": identity.private_conversation_id},
        )

    def _handle_agent_loop(
        self,
        msg: GroupMessage,
        routing: RoutingDecision,
        *,
        trace_id: str,
    ) -> GroupHandleResult:
        """为单个群成员创建隔离会话，并在群聊回复约束下调用主智能体。"""
        constraints = routing.reply_constraints.to_dict() if routing.reply_constraints else {}
        identity = self.store.get_channel_identity(msg.channel, msg.sender_external_id)
        sender_id = identity.customer_id if identity is not None else f"{msg.channel}:{msg.sender_external_id}"
        # 同一个群中的不同成员不能共享 conversation_id，否则画像、追问和临时约束会串线。
        isolated_conversation_id = f"group:{msg.room_id}:customer:{sender_id}"
        result = self.runtime.handle_user_message(
            UserMessage(
                conversation_id=isolated_conversation_id,
                sender_id=sender_id,
                sender_name=msg.sender_name,
                text=msg.text,
                message_id=msg.message_id,
                sent_at=msg.sent_at,
                metadata={
                    "source": "group",
                    "room_id": msg.room_id,
                    "source_conversation_id": msg.conversation_id,
                    "reply_constraints": constraints,
                },
            ),
            trace_id=trace_id,
        )
        if result.final_reply:
            self.messenger.send_group_message(
                msg.room_id,
                result.final_reply,
                metadata={"trace_id": trace_id, "source_message_id": msg.message_id},
            )
        return GroupHandleResult(action="agent_loop", reply=result.final_reply)

    def _redirect_private_continuation(self, msg: GroupMessage, *, trace_id: str) -> bool:
        """识别仍属于近期私聊任务的群内补充，并提醒用户回到私聊。"""
        identity = self.store.get_channel_identity(msg.channel, msg.sender_external_id)
        if identity is None or not identity.can_private_message:
            return False
        switch = self.store.get_recent_active_channel_switch(
            identity.customer_id,
            room_id=msg.room_id,
            at=self.clock(),
        )
        if switch is None or not self._looks_like_business_followup(msg.text, at=msg.sent_at):
            return False
        self.messenger.send_group_message(
            msg.room_id,
            f"@{identity.public_name} 私聊回你了，看下哈",
            metadata={"trace_id": trace_id, "channel_switch_id": switch.switch_id},
        )
        return True

    @staticmethod
    def _looks_like_business_followup(text: str, *, at: datetime) -> bool:
        """保守判断文本是否像组局条件补充，避免把普通闲聊强行导向私聊。"""
        if parse_explicit_need(text, anchor=at):
            return True
        return any(
            signal in text
            for signal in (
                "麻将",
                "杭麻",
                "川麻",
                "红中",
                "无烟",
                "有烟",
                "人齐开",
                "几点",
                "几个人",
                "打",
                "局",
            )
        )


__all__ = ["GroupMessageHandler"]
