"""生产群聊消息入口：协调规则分流、私聊切换和主智能体调用。"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta
import json
from typing import Any

from ..models import GameStatus, UserMessage, new_id, now
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
    没有装配共享原文的 ``GroupSessionPipeline``。决策顺序是 L1 自身回声/编号认领协议
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
        prepare_reply_delivery: Callable[[Any], bool] | None = None,
    ) -> None:
        self.rule_engine = rule_engine
        self.intent_router = intent_router
        self.board_engine = board_engine
        self.claim_handler = claim_handler
        self.notify_dispatcher = notify_dispatcher
        self.messenger = messenger
        self.runtime = runtime
        self.clock = clock
        self.prepare_reply_delivery = prepare_reply_delivery
        self.store = intent_router.store

    def handle(self, msg: GroupMessage, *, trace_id: str) -> GroupHandleResult:
        """同渠道同群的同一发送人依次处理；续发条件可读取前一条建立的牌局与话题绑定。"""

        coordination = getattr(self.runtime, "coordination_manager", None)
        if coordination is None:
            return self._handle_message(msg, trace_id=trace_id)
        # 使用独立入口锁。主 Agent 可以继续获取其会话锁；其他发言人不必等此人完成。
        scope = json.dumps([msg.channel, msg.room_id, msg.sender_external_id], ensure_ascii=False)
        with coordination.lock(f"group-ingress:{scope}"):
            return self._handle_message(msg, trace_id=trace_id)

    def _handle_message(self, msg: GroupMessage, *, trace_id: str) -> GroupHandleResult:
        """按 L1 确定性协议、通道路由、主 Agent 的优先级处理已协调的群消息。"""

        # 编号认领针对已发布看板。群友发出的新牌局条件先经过语义分流。
        l1_result = self.rule_engine.process(msg)
        if l1_result.action == "ignore":
            return GroupHandleResult(action="ignore")
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
        if routing.action == "register_self_recruitment":
            game = self.board_engine.register_self_recruitment(
                msg, routing.topic_routing, trace_id=trace_id, tool_gateway=self.runtime.tool_gateway,
            )
            topic = routing.topic_routing.get("topic")
            if game and topic:
                self.store.bind_group_topic_customer(topic_id=topic["topic_id"], customer_id=game.organizer_id,
                    conversation_id=game.conversation_id, game_ids=[game.game_id],
                    source_message_id=msg.message_id, at=msg.sent_at)
            return GroupHandleResult(action=routing.action, game_id=game.game_id if game else None,
                detail={"topic_routing": routing.topic_routing,
                        "board_arrangement": self.store.game_board_arrangement(game.game_id) if game else None})
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
        """用本人隔离上下文处理委托；实际发送必须等到本轮结果提供联系理由。"""
        identity = self.store.get_channel_identity(msg.channel, msg.sender_external_id)
        if identity is None or not identity.can_private_message or routing.switch_context is None:
            raise ValueError("private route has no authorized identity")
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
        decision = routing.topic_routing["message"]["classification"]
        # 模型调用期间联系人可能撤销权限。已按私聊生成的正文即使无法送达也不能转群。
        latest_identity = self.store.get_channel_identity(msg.channel, msg.sender_external_id)
        still_private = (latest_identity is not None and latest_identity.can_private_message
                         and latest_identity.customer_id == identity.customer_id
                         and latest_identity.private_conversation_id == identity.private_conversation_id)
        can_deliver = still_private and result.status in {"completed", "waiting_user"} and (
            decision["intent"] not in {"delegate", "create"}
            or decision["goal_relation"] == "continue"
            or (not getattr(self.store.customers.get(identity.customer_id), "no_contact", False)
                and any(self._live_option(tool) for tool in result.tool_results))
        )
        delivery_id = ""
        if can_deliver and result.final_reply and self._prepare_delivery(result):
            delivery_id = self.notify_dispatcher.send_private_switch_reply(
                identity, result, trace_id=trace_id, source_message=msg,
            )
        return GroupHandleResult(action="private_switch",
            detail={"private_conversation_id": identity.private_conversation_id, "topic_routing": routing.topic_routing,
                    "private_delivery_id": delivery_id,
                    "private_reply_suppressed": bool(result.final_reply and not can_deliver)})

    @staticmethod
    def _exact_options(tool) -> list:
        """只承认真实查询成功返回的精确选项，排除画像排序及未核实的备选。"""
        if tool.name == "search_current_games" and tool.called and tool.allowed and not tool.error:
            return tool.result.get("matches", [])
        return []

    def _live_option(self, tool, *, fills_table: bool = False) -> bool:
        """只凭真实精确选项联系；群兜底还须有能够补齐桌位的方案。"""
        for match in self._exact_options(tool):
            if fills_table and not (match.get("join_projection") or {}).get("would_fill_game"):
                continue
            snapshot = match["game"]
            try:
                game = self.store.require_game(snapshot["game_id"])
            except ValueError:
                continue
            if (game.status in {GameStatus.FORMING, GameStatus.INVITING} and game.remaining_seats() > 0
                    and game.updated_at.isoformat() == snapshot.get("updated_at")
                    and game.requirement.get("recruitment_mode") != "board_only"
                    and (game.expires_at is None or game.expires_at > self.clock())):
                return True
        return False

    def _handle_agent_loop(self, msg: GroupMessage, routing: RoutingDecision, *, trace_id: str) -> GroupHandleResult:
        """每位群成员拥有独立会话；只把当前原话和可验证的话题引用交给主 Agent。"""
        identity = self.store.get_channel_identity(msg.channel, msg.sender_external_id)
        sender_id = identity.customer_id if identity else f"{msg.channel}:{msg.sender_external_id}"
        conversation_id = f"group:{msg.room_id}:customer:{sender_id}"
        message = self._user_input(msg, routing, conversation_id, sender_id, private=False)
        result = self.runtime.handle_user_message(message, trace_id=trace_id)
        self._bind_result(msg, routing, message, result)
        decision = routing.topic_routing["message"]["classification"]
        # 对公开提问或委托的简短承接仍可公开回复。具体参加征询优先私下；只有
        # 无可用私聊且有补齐方案时，群内才承接这一类面向本人的征询。
        choice_reply = decision["intent"] in {"delegate", "create"} and any(
            self._exact_options(tool) for tool in result.tool_results)
        current_identity = self.store.get_channel_identity(msg.channel, msg.sender_external_id) if choice_reply else None
        allow_group_reply = not choice_reply or (
            current_identity is not None and current_identity.customer_id == sender_id
            and not current_identity.can_private_message
            and not getattr(self.store.customers.get(sender_id), "no_contact", False)
            and any(self._live_option(tool, fills_table=True) for tool in result.tool_results)
        )
        reply = result.final_reply if allow_group_reply and result.final_reply and self._prepare_delivery(result) else ""
        if reply:
            self.messenger.send_group_message(msg.room_id, reply,
                metadata={"trace_id": trace_id, "source_message_id": msg.message_id})
        return GroupHandleResult(action="agent_loop", reply=reply,
                                 detail={"topic_routing": routing.topic_routing})

    def _prepare_delivery(self, result) -> bool:
        """调用通道交付栅栏前，先在业务入口确认本轮确实需要发送。"""
        return self.prepare_reply_delivery is None or self.prepare_reply_delivery(result)

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
