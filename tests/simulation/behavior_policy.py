"""Layer 2: deterministic personas and message behavior policy."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

try:  # Package import under pytest, direct import when the CLI file is executed.
    from .sim_factory import (
        GROUP_ID,
        PERSONA_ACTIVE_GAMBLER,
        PERSONA_LURKER,
        PERSONA_TROUBLEMAKER,
        VirtualUser,
    )
except ImportError:  # pragma: no cover - direct script execution path
    from sim_factory import (  # type: ignore
        GROUP_ID,
        PERSONA_ACTIVE_GAMBLER,
        PERSONA_LURKER,
        PERSONA_TROUBLEMAKER,
        VirtualUser,
    )


QUESTION_POOL: tuple[str, ...] = (
    "今晚有局吗",
    "川麻三缺一？",
    "包间多少钱",
    "还有位置吗",
    "现在0.5有人吗",
    "一块无烟有人齐开吗",
    "下午两点能组一桌吗",
    "财敲还有位置吗",
    "红中麻将有人打吗",
    "通宵局还有吗",
    "现在最快几点能开",
    "帮我约个无烟局",
    "我一个人，晚上七点可以",
    "272还缺人吗",
    "371我可以来",
    "川麻1-32有人吗",
    "能打四个小时吗",
    "有烟无烟都可以",
    "今天房间满了吗",
    "能不能帮我找两个人",
)


@dataclass(slots=True, order=True, frozen=True)
class SimulationAction:
    """A scheduled synthetic WeChat event ordered by simulated time."""

    due_simulated_seconds: float
    sequence: int
    channel: str = field(compare=False)
    conversation_id: str = field(compare=False)
    sender_id: str = field(compare=False)
    sender_name: str = field(compare=False)
    text: str = field(compare=False)
    event_type: str = field(default="text", compare=False)
    recalled_message_id: str | None = field(default=None, compare=False)

    @property
    def message_id(self) -> str:
        return f"sim_message_{self.sequence:06d}"

    def to_wechat_payload(self) -> dict[str, Any]:
        """Return the unified envelope plus auditable WeChat-like raw fields."""

        is_room = self.channel == "group"
        raw_wechat_payload = {
            "platform_name": "wechaty",
            "source_message_id": self.message_id,
            "message_type": self.event_type,
            "is_room": is_room,
            "room": {"id": GROUP_ID, "topic": "百人麻将测试群"} if is_room else None,
            "talker": {"id": self.sender_id, "name": self.sender_name},
            "raw_text": self.text,
            "recalled_message_id": self.recalled_message_id,
        }
        return {
            "conversation_id": self.conversation_id,
            "sender_id": self.sender_id,
            "sender_name": self.sender_name,
            "message_id": self.message_id,
            "trace_id": f"trace_sim_{self.sequence:06d}",
            "text": self.text,
            "aggregate_fragments": False,
            "metadata": {
                "channel": "wechaty",
                "simulation": True,
                "simulation_event_type": self.event_type,
                "group_id": GROUP_ID if is_room else None,
                "raw_wechat_payload": raw_wechat_payload,
            },
        }


class BehaviorPolicy:
    """Generate actions from personas instead of choosing arbitrary users."""

    ACTIVE_INTERVAL_SECONDS = 10.0
    TROUBLE_INTERVAL_SECONDS = 8.0

    def __init__(self, users: list[VirtualUser], *, seed: int = 42) -> None:
        self.users = list(users)
        self.random = random.Random(seed)
        self._last_text_message: dict[str, SimulationAction] = {}
        self._trouble_turn: dict[str, int] = {}

    def speaking_users(self) -> list[VirtualUser]:
        return [user for user in self.users if user.persona != PERSONA_LURKER]

    @staticmethod
    def interval_for(user: VirtualUser) -> float:
        if user.persona == PERSONA_ACTIVE_GAMBLER:
            return BehaviorPolicy.ACTIVE_INTERVAL_SECONDS
        if user.persona == PERSONA_TROUBLEMAKER:
            return BehaviorPolicy.TROUBLE_INTERVAL_SECONDS
        raise ValueError("lurkers never receive scheduled speaking actions")

    def get_next_action(
        self,
        user: VirtualUser,
        *,
        sequence: int,
        due_simulated_seconds: float,
    ) -> SimulationAction | None:
        if user.persona == PERSONA_LURKER:
            return None
        channel = "group" if self.random.random() < 0.80 else "private"
        conversation_id = (
            f"sim:group:{GROUP_ID}"
            if channel == "group"
            else f"sim:private:{user.customer_id}"
        )

        event_type = "text"
        recalled_message_id: str | None = None
        if user.persona == PERSONA_ACTIVE_GAMBLER:
            text = "还有位置吗"
        else:
            turn = self._trouble_turn.get(user.customer_id, 0) + 1
            self._trouble_turn[user.customer_id] = turn
            recalled_action = self._last_text_message.get(user.customer_id)
            if turn % 3 == 0 and recalled_action:
                event_type = "recall"
                recalled_message_id = recalled_action.message_id
                channel = recalled_action.channel
                conversation_id = recalled_action.conversation_id
                text = "撤回了一条消息"
            else:
                text = self._with_typo(self.random.choice(QUESTION_POOL))

        action = SimulationAction(
            due_simulated_seconds=due_simulated_seconds,
            sequence=sequence,
            channel=channel,
            conversation_id=conversation_id,
            sender_id=user.customer_id,
            sender_name=user.display_name,
            text=text,
            event_type=event_type,
            recalled_message_id=recalled_message_id,
        )
        if event_type == "text":
            self._last_text_message[user.customer_id] = action
        return action

    def first_action(self, user: VirtualUser, *, sequence: int) -> SimulationAction | None:
        return self.get_next_action(
            user,
            sequence=sequence,
            due_simulated_seconds=self.interval_for(user),
        ) if user.persona != PERSONA_LURKER else None

    def following_action(self, user: VirtualUser, previous: SimulationAction, *, sequence: int) -> SimulationAction:
        action = self.get_next_action(
            user,
            sequence=sequence,
            due_simulated_seconds=previous.due_simulated_seconds + self.interval_for(user),
        )
        if action is None:  # Defensive: callers only schedule speaking personas.
            raise ValueError("cannot schedule a following action for a lurker")
        return action

    @staticmethod
    def _with_typo(text: str) -> str:
        replacements = (
            ("麻将", "麻酱"),
            ("位置", "位子"),
            ("三缺一", "三却一"),
            ("无烟", "无言"),
            ("可以", "可一"),
            ("吗", "嘛"),
        )
        for source, target in replacements:
            if source in text:
                return text.replace(source, target, 1)
        return f"{text}。。"
