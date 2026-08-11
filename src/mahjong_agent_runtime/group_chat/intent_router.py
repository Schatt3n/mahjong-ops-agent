"""实现群聊场景中的intent路由器能力。

"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence

from ..stores import AgentStore
from .models import GroupMessage, PrivateSwitchContext, ReplyConstraints, RoutingDecision
from .parsing import parse_explicit_need


class L2IntentRouter:
    """根据消息意图与通道身份决定忽略、群内回复、私聊切换或进入 Agent。"""

    SIMPLE_QUERY_SIGNALS = ("齐了吗", "满了吗", "几个人", "几点", "什么时候", "有局吗", "还有位吗", "有人吗")
    COMPLEX_REQUEST_SIGNALS = ("帮我约", "帮我组", "安排一下", "组个局", "约个", "找人")
    CANCEL_SIGNALS = ("不来了", "不打了", "退了", "取消")
    PRIVATE_ACKS = ("好的私聊跟你说", "私聊回你", "好的私聊说")

    def __init__(
        self,
        store: AgentStore,
        *,
        ack_picker: Callable[[Sequence[str]], str] | None = None,
    ) -> None:
        self.store = store
        self.ack_picker = ack_picker or random.choice

    def route(self, msg: GroupMessage) -> RoutingDecision:
        if any(signal in msg.text for signal in self.CANCEL_SIGNALS):
            return RoutingDecision(
                action="agent_loop",
                channel="group",
                reply_constraints=ReplyConstraints(max_length=10),
            )
        if any(signal in msg.text for signal in self.SIMPLE_QUERY_SIGNALS):
            return RoutingDecision(
                action="agent_loop",
                channel="group",
                reply_constraints=ReplyConstraints(max_length=20),
            )
        if any(signal in msg.text for signal in self.COMPLEX_REQUEST_SIGNALS):
            identity = self.store.get_channel_identity(msg.channel, msg.sender_external_id)
            if identity is not None and identity.can_private_message:
                return RoutingDecision(
                    action="private_switch",
                    group_ack=self.ack_picker(self.PRIVATE_ACKS),
                    switch_context=self._build_switch_context(msg, identity),
                )
            return RoutingDecision(
                action="agent_loop",
                channel="group",
                reply_constraints=ReplyConstraints(max_length=50),
            )
        return RoutingDecision(action="ignore")

    def _build_switch_context(self, msg: GroupMessage, identity) -> PrivateSwitchContext:
        parsed = parse_explicit_need(msg.text, anchor=msg.sent_at)
        missing = [
            label
            for key, label in (
                ("requested_game", "玩法"),
                ("stake", "档位"),
                ("start_time_kind", "时间"),
                ("smoke_preference", "烟况"),
                ("known_player_count", "当前人数"),
            )
            if key not in parsed
        ]
        profile = self.store.customers.get(identity.customer_id)
        profile_context = profile.to_model_context() if profile is not None else {}
        guidelines = (
            "直接根据已有事实推进，不寒暄、不重复确认意图；只追问真正缺失且画像无法补足的信息。"
            if missing
            else "信息已较完整，直接查询当前局并继续完成目标。"
        )
        return PrivateSwitchContext(
            private_conversation_id=identity.private_conversation_id,
            trigger_summary="用户刚提出组局请求，需要继续确认细节。",
            customer_profile=profile_context,
            user_original_text=msg.text,
            parsed_need=parsed,
            missing_info=missing,
            reply_guidelines=guidelines,
        )


__all__ = ["L2IntentRouter"]
