"""实现群聊场景中的notify分发器能力。

"""

from __future__ import annotations

from typing import Any

from ..models import Game, SystemTriggerMessage, new_id
from ..stores import AgentStore
from .messenger import GroupMessenger
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
        room_id = self._room_id(game)
        for participant in game.participants:
            if participant.status not in {"joined", "confirmed"}:
                continue
            identity = self.store.get_channel_identity_for_customer(participant.customer_id)
            if identity is None:
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
        room_id: str,
    ) -> str:
        if not result.final_reply:
            return ""
        return self.messenger.send_private_message(
            identity.external_user_id,
            result.final_reply,
            metadata={
                "trace_id": trace_id,
                "conversation_id": identity.private_conversation_id,
                "origin_room_id": room_id,
            },
        )

    def _room_id(self, game: Game) -> str:
        links = self.store.game_conversation_links(game_id=game.game_id)
        return links[0].room_id if links else game.conversation_id.removeprefix("wechaty:room:")

__all__ = ["NotifyDispatcher"]
