"""实现群聊场景中的看板触发器能力。

"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from ..hooks import HookEvent
from ..models import new_id, now
from ..stores import AgentStore
from .models import GameConversationLink


_GAME_MUTATION_TOOLS = {
    "create_game",
    "join_game",
    "record_candidate_reply",
    "update_game_requirement",
    "update_game_status",
}


@dataclass(slots=True)
class GroupBoardTrigger:
    """在牌局变更成功后调度面板投影更新。

    该钩子特意调度持久化工作而不是发送消息。
    这样将传输关注点置于 ToolGateway 之外，并让重启后的进程能在稍后完成投影。

    导航：``ToolExecutionService`` emit ``after_tool_execute`` -> ``HookManager`` -> 本类
    ``__call__`` -> Store 写持久化发布任务。后续由定时任务 Scheduler 消费，并在房间、
    外发策略和可发布牌局都满足时尝试发布看板。
    """

    store: AgentStore
    trace_recorder: Any | None = None
    clock: Callable[[], datetime] = now

    def __call__(self, event: HookEvent) -> None:
        """筛选成功的牌局变更工具，并为相关托管群安排看板刷新。"""
        call = event.payload.get("call") if isinstance(event.payload.get("call"), dict) else {}
        result = event.payload.get("result") if isinstance(event.payload.get("result"), dict) else {}
        tool_name = str(call.get("name") or result.get("name") or "")
        if tool_name not in _GAME_MUTATION_TOOLS or not self._succeeded(result):
            return
        game_id = self._game_id(result)
        if not game_id:
            return

        links = self.store.game_conversation_links(game_id=game_id)
        if tool_name == "create_game" and not links:
            link = self._link_private_switch_game(event, call=call, game_id=game_id)
            if link is not None:
                links = [link]
        if not links:
            return

        event_type, urgent = self._board_event(tool_name, call)
        scheduled_rooms: list[str] = []
        for room_id in sorted({item.room_id for item in links if item.room_id}):
            policy = self.store.get_group_room_policy(room_id)
            if policy is None or not policy.managed or not policy.board_enabled:
                continue
            merge_seconds = max(0, int(policy.merge_window_seconds))
            due_at = self.clock() if urgent else self.clock() + timedelta(seconds=merge_seconds)
            self.store.ensure_group_board_publish_task(
                room_id=room_id,
                due_at=due_at,
                trace_id=event.trace_id,
                urgent=urgent,
            )
            scheduled_rooms.append(room_id)
        self._trace(
            event.trace_id,
            {
                "tool_name": tool_name,
                "game_id": game_id,
                "event_type": event_type,
                "urgent": urgent,
                "scheduled_room_ids": scheduled_rooms,
            },
        )

    def _link_private_switch_game(
        self,
        event: HookEvent,
        *,
        call: dict[str, Any],
        game_id: str,
    ) -> GameConversationLink | None:
        arguments = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
        customer_id = str(arguments.get("organizer_id") or event.payload.get("sender_id") or "").strip()
        if not customer_id:
            return None
        switch = self.store.get_recent_active_channel_switch(customer_id, at=self.clock())
        if switch is None:
            return None
        link = GameConversationLink(
            link_id=new_id("game_conversation_link"),
            game_id=game_id,
            conversation_id=str(event.payload.get("conversation_id") or switch.private_conversation_id),
            room_id=switch.room_id,
            customer_id=customer_id,
            link_type="private_switch_created",
        )
        return self.store.link_game_conversation(link)

    @staticmethod
    def _succeeded(result: dict[str, Any]) -> bool:
        return bool(result.get("called")) and bool(result.get("allowed")) and not result.get("error") and not bool(
            result.get("deduplicated")
        )

    @staticmethod
    def _game_id(result: dict[str, Any]) -> str:
        value = result.get("result") if isinstance(result.get("result"), dict) else {}
        game = value.get("game") if isinstance(value.get("game"), dict) else {}
        if game.get("game_id"):
            return str(game["game_id"])
        for transition in result.get("state_transitions") or []:
            if isinstance(transition, dict) and transition.get("entity_type") == "game":
                return str(transition.get("entity_id") or "")
        return ""

    @staticmethod
    def _board_event(tool_name: str, call: dict[str, Any]) -> tuple[str, bool]:
        arguments = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
        if tool_name == "join_game":
            return "seat_claimed", True
        if tool_name == "record_candidate_reply":
            status = str(arguments.get("status") or "").lower()
            if status in {"confirmed", "accepted", "arrived", "joined"}:
                return "seat_claimed", True
            if status in {"declined", "cancelled", "canceled", "superseded"}:
                return "seat_released", True
        if tool_name == "update_game_status":
            return "game_status_changed", True
        return "game_updated" if tool_name != "create_game" else "game_created", False

    def _trace(self, trace_id: str, payload: dict[str, Any]) -> None:
        if self.trace_recorder is not None:
            self.trace_recorder.record(trace_id, "group_board_refresh_scheduled", payload)


__all__ = ["GroupBoardTrigger"]
