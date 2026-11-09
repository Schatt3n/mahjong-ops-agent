"""实现群聊场景中的notify分发器能力。

"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from ..models import Game, SystemTriggerMessage, new_id
from ..stores import AgentStore
from .messenger import GroupMessenger
from .models import GroupMessage
from .projections import public_game_start_display, public_group_game_summary


class NotifyDispatcher:
    """优先通过私聊发送经审查的领域通知，必要时降级为群内提醒。"""

    def __init__(self, *, store: AgentStore, messenger: GroupMessenger, runtime: Any) -> None:
        self.store = store
        self.messenger = messenger
        self.runtime = runtime

    def notify_claim_success(self, identity, game: Game, *, trace_id: str) -> str:
        room_id = self._room_id(game)
        if identity.can_private_message:
            result = self.runtime.handle_system_trigger(
                SystemTriggerMessage(
                    trigger_id=new_id("group_claim_confirmed"),
                    trigger_type="group_claim_confirmed",
                    conversation_id=identity.private_conversation_id,
                    sender_id=identity.customer_id,
                    sender_name=identity.public_name,
                    payload={
                        "trigger_summary": "用户刚在群看板认领了一个座位，后端已原子占位。",
                        "claimed_game": public_group_game_summary(game),
                        "reply_guidelines": "只简短确认已经占上，不透露其他参与者身份。",
                    },
                ),
                trace_id=trace_id,
            )
            if result.final_reply:
                return self.messenger.send_private_message(
                    identity.external_user_id,
                    result.final_reply,
                    metadata={
                        "trace_id": trace_id,
                        "game_id": game.game_id,
                        "origin_room_id": room_id,
                    },
                )
            return ""
        return self.messenger.send_group_message(
            room_id,
            f"@{identity.public_name} 占上了",
            metadata={"trace_id": trace_id, "game_id": game.game_id},
        )

    def notify_rejection(
        self,
        identity,
        game: Game | None,
        reason: str,
        *,
        trace_id: str,
        room_id: str | None = None,
    ) -> str:
        public_reason = {
            "game_full": "这局已经满了",
            "already_joined": "你已经在这局里了",
            "time_conflict": "这个时间和你已有的局冲突了",
            "game_not_found": "这个编号现在对不上了",
            "self_recruitment_requires_contact": "这桌是群友自招，先问一下发局人",
        }.get(reason, "这次没占上")
        if identity.can_private_message:
            result = self.runtime.handle_system_trigger(
                SystemTriggerMessage(
                    trigger_id=new_id("group_claim_rejected"),
                    trigger_type="group_claim_rejected",
                    conversation_id=identity.private_conversation_id,
                    sender_id=identity.customer_id,
                    sender_name=identity.public_name,
                    payload={
                        "trigger_summary": public_reason,
                        "game": public_group_game_summary(game) if game is not None else None,
                        "reply_guidelines": "简短说明认领没有成功，只说必要原因。",
                    },
                ),
                trace_id=trace_id,
            )
            if result.final_reply:
                return self.messenger.send_private_message(
                    identity.external_user_id,
                    result.final_reply,
                    metadata={
                        "trace_id": trace_id,
                        "game_id": game.game_id if game else None,
                        "origin_room_id": room_id or (self._room_id(game) if game else None),
                    },
                )
        return ""

    def notify_game_full(self, game: Game, *, trace_id: str) -> None:
        """仅对确已补齐的局通知参加者；没有可用私聊时才允许群内定向告知。"""
        if game.remaining_seats() != 0:
            return
        room_id = self._room_id(game)
        customers = self.store.customers
        for participant in game.participants:
            if participant.status not in {"joined", "confirmed"}:
                continue
            identity = self.store.get_channel_identity_for_customer(participant.customer_id)
            if identity is None or getattr(customers.get(participant.customer_id), "no_contact", False):
                continue
            if not identity.can_private_message:
                self.messenger.send_group_message(
                    room_id,
                    f"@{identity.public_name} 人齐了，{public_game_start_display(game)}开",
                    metadata={"trace_id": trace_id, "game_id": game.game_id},
                )
                continue
            result = self.runtime.handle_system_trigger(
                SystemTriggerMessage(
                    trigger_id=f"group_game_full:{game.game_id}:{identity.customer_id}",
                    trigger_type="group_game_full",
                    conversation_id=identity.private_conversation_id,
                    sender_id=identity.customer_id,
                    sender_name=identity.public_name,
                    payload={
                        "trigger_summary": "用户参与的局已经满员。",
                        "game": public_group_game_summary(game),
                        "reply_guidelines": "简短通知人齐了和开局时间，不透露其他参与者身份。",
                    },
                ),
                trace_id=trace_id,
            )
            if result.final_reply:
                self.messenger.send_private_message(
                    identity.external_user_id,
                    result.final_reply,
                    metadata={
                        "trace_id": trace_id,
                        "game_id": game.game_id,
                        "origin_room_id": room_id,
                    },
                )

    def send_private_switch_reply(
        self,
        identity,
        result,
        *,
        trace_id: str,
        source_message: GroupMessage,
    ) -> str:
        """为同一群来源的私聊正文生成稳定交付键，重试换 trace 不得重复外发。

        桥接以 draft_id 优先且跨通道去重，因此正文不能复用群确认的 source_message_id
        作为交付键；保留原消息号供引用，另以完整来源和 private_reply 角色生成唯一键。
        """
        if not result.final_reply:
            return ""
        source = [source_message.channel, source_message.room_id, identity.customer_id,
                  source_message.message_id, "private_reply"]
        delivery_id = "group_private_reply:" + hashlib.sha256(
            json.dumps(source, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return self.messenger.send_private_message(
            identity.external_user_id,
            result.final_reply,
            metadata={
                "trace_id": trace_id,
                "draft_id": delivery_id,
                "source_message_id": source_message.message_id,
                "conversation_id": identity.private_conversation_id,
                "origin_room_id": source_message.room_id,
            },
        )

    def _room_id(self, game: Game) -> str:
        links = self.store.game_conversation_links(game_id=game.game_id)
        return links[0].room_id if links else game.conversation_id.removeprefix("wechaty:room:")

__all__ = ["NotifyDispatcher"]
