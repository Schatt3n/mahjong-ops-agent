"""Layer 2: deterministic personas and message behavior policy."""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

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

# The first-turn pool keeps the original 20 high-frequency Mahjong questions.
# Follow-up messages are deliberately short fragments, because they represent a
# user answering the Agent rather than opening another independent topic.
FIRST_TURN_QUESTION_POOL = QUESTION_POOL
FOLLOW_UP_REPLY_POOL: tuple[str, ...] = (
    "晚上七点左右",
    "现在就可以",
    "我一个人",
    "我们两个",
    "0.5，无烟",
    "一块也可以",
    "四个小时",
    "五个小时都行",
    "确认，可以",
    "对，就按这个来",
    "可以，帮我问问",
    "都可以，你看着安排",
)
NEW_TOPIC_POOL: tuple[str, ...] = (
    "对了，今晚还有别的局吗",
    "明天中午有人打吗",
    "有无烟的再叫我",
    "川麻那桌还缺人吗",
    "现在最快几点能开",
    "包间还有空的吗",
)
CHITCHAT_POOL: tuple[str, ...] = (
    "你们今天忙不忙",
    "最近生意怎么样",
    "今天外面也太热了",
    "对了，你吃饭了吗",
)
BUSINESS_RESUME_POOL: tuple[str, ...] = (
    "对了，刚才那个局有消息了吗",
    "说回刚才那个局，继续帮我看看",
    "刚才说的那个还在帮我问吗",
    "对了，刚才那个现在什么情况",
)
QUESTION_CUES = ("?", "？", "几点", "几人", "确认")

DIALOG_PHASE_BUSINESS = "business"
DIALOG_PHASE_CHITCHAT = "chitchat"
DIALOG_PHASE_BUSINESS_RESUME = "business_resume"


class DialogStateView(Protocol):
    """Read-only shape consumed by the behavior layer.

    ``DialogState`` itself lives in the orchestrator so one simulation run owns
    one consistent state registry.  A protocol avoids coupling the behavior
    layer back to the scheduler module.
    """

    turn_count: int
    pending_response_to: str | None
    last_agent_reply: str
    last_conversation_id: str | None
    last_channel: str | None
    business_active: bool
    business_anchor: str
    business_conversation_id: str | None
    business_channel: str | None
    business_turn_count: int
    business_turns_since_chitchat: int
    chitchat_pending_resume: bool
    chitchat_interruption_count: int


@dataclass(slots=True, frozen=True)
class MessageGenerationRequest:
    """Business-safe context supplied to a synthetic-message generator."""

    sender_id: str
    sender_name: str
    persona: str
    preferred_game: str
    channel: str
    conversation_id: str
    turn_count: int
    last_agent_reply: str
    fallback_text: str
    is_follow_up: bool
    dialog_phase: str
    business_anchor: str


@dataclass(slots=True, frozen=True)
class MessageGenerationResult:
    """One generated utterance plus audit metadata for the run report."""

    text: str
    source: str
    model: str | None = None
    trace_id: str | None = None
    latency_ms: float | None = None
    error: str | None = None


class SimulationMessageGenerator(Protocol):
    """Generate a user-visible synthetic message without touching Agent state."""

    def generate(self, request: MessageGenerationRequest) -> MessageGenerationResult:
        ...


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
    generation_source: str = field(default="rule", compare=False)
    generator_model: str | None = field(default=None, compare=False)
    generation_trace_id: str | None = field(default=None, compare=False)
    generation_latency_ms: float | None = field(default=None, compare=False)
    generation_error: str | None = field(default=None, compare=False)
    dialog_phase: str = field(default=DIALOG_PHASE_BUSINESS, compare=False)

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
                "simulation_generation": {
                    "source": self.generation_source,
                    "model": self.generator_model,
                    "trace_id": self.generation_trace_id,
                    "latency_ms": self.generation_latency_ms,
                    "error": self.generation_error,
                },
                "simulation_dialog_phase": self.dialog_phase,
                "group_id": GROUP_ID if is_room else None,
                "raw_wechat_payload": raw_wechat_payload,
            },
        }


class BehaviorPolicy:
    """Generate actions from personas instead of choosing arbitrary users."""

    ACTIVE_INTERVAL_SECONDS = 10.0
    TROUBLE_INTERVAL_SECONDS = 8.0

    def __init__(
        self,
        users: list[VirtualUser],
        *,
        seed: int = 42,
        message_generator: SimulationMessageGenerator | None = None,
        chitchat_probability: float = 0.35,
        business_return_probability: float = 0.5,
        max_chitchat_interruptions_per_dialog: int = 2,
    ) -> None:
        if not 0.0 <= chitchat_probability <= 1.0:
            raise ValueError("chitchat_probability must be between 0 and 1")
        if not 0.0 <= business_return_probability <= 1.0:
            raise ValueError("business_return_probability must be between 0 and 1")
        if max_chitchat_interruptions_per_dialog < 0:
            raise ValueError("max_chitchat_interruptions_per_dialog must not be negative")
        self.users = list(users)
        self.random = random.Random(seed)
        self.message_generator = message_generator
        self.chitchat_probability = float(chitchat_probability)
        self.business_return_probability = float(business_return_probability)
        self.max_chitchat_interruptions_per_dialog = int(max_chitchat_interruptions_per_dialog)
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
        dialog_state: DialogStateView | None = None,
        channel_override: str | None = None,
    ) -> SimulationAction | None:
        if user.persona == PERSONA_LURKER:
            return None
        if channel_override not in {None, "group", "private"}:
            raise ValueError("channel_override must be group, private, or None")

        turn_count = int(getattr(dialog_state, "turn_count", 0))
        last_agent_reply = str(getattr(dialog_state, "last_agent_reply", "") or "")
        is_answering_agent = turn_count >= 1 and reply_requires_user(last_agent_reply)
        business_anchor = str(getattr(dialog_state, "business_anchor", "") or "")
        chitchat_pending_resume = bool(
            getattr(dialog_state, "chitchat_pending_resume", False)
        )
        dialog_phase = DIALOG_PHASE_BUSINESS

        if turn_count == 0:
            text = self.random.choice(FIRST_TURN_QUESTION_POOL)
        elif chitchat_pending_resume and business_anchor:
            text = self.random.choice(BUSINESS_RESUME_POOL)
            dialog_phase = DIALOG_PHASE_BUSINESS_RESUME
        elif self._should_interrupt_with_chitchat(user, dialog_state):
            text = self.random.choice(CHITCHAT_POOL)
            dialog_phase = DIALOG_PHASE_CHITCHAT
        elif is_answering_agent:
            text = self._follow_up_reply(last_agent_reply)
        elif bool(getattr(dialog_state, "business_active", False)):
            # A background matching task can become quiet without being
            # complete. Some users return later for a status check; others
            # naturally stop messaging for this simulation cycle.
            if self.random.random() >= self.business_return_probability:
                return None
            text = self.random.choice(BUSINESS_RESUME_POOL)
            dialog_phase = DIALOG_PHASE_BUSINESS_RESUME
        else:
            # A non-question reply closes the current exchange.  The user may
            # start a fresh topic, or remain silent for this scheduling cycle.
            if self.random.random() < 0.50:
                return None
            text = self.random.choice(NEW_TOPIC_POOL)

        previous_conversation_id = getattr(dialog_state, "last_conversation_id", None)
        previous_channel = getattr(dialog_state, "last_channel", None)
        business_conversation_id = getattr(dialog_state, "business_conversation_id", None)
        business_channel = getattr(dialog_state, "business_channel", None)
        continues_business_thread = dialog_phase in {
            DIALOG_PHASE_CHITCHAT,
            DIALOG_PHASE_BUSINESS_RESUME,
        }
        if continues_business_thread and business_conversation_id and business_channel:
            channel = str(business_channel)
            conversation_id = str(business_conversation_id)
        elif is_answering_agent and previous_conversation_id and previous_channel:
            channel = str(previous_channel)
            conversation_id = str(previous_conversation_id)
        elif channel_override is not None:
            channel = channel_override
            conversation_id = (
                f"sim:group:{GROUP_ID}"
                if channel == "group"
                else f"sim:private:{user.customer_id}"
            )
        else:
            channel = "group" if self.random.random() < 0.80 else "private"
            conversation_id = (
                f"sim:group:{GROUP_ID}"
                if channel == "group"
                else f"sim:private:{user.customer_id}"
            )

        event_type = "text"
        recalled_message_id: str | None = None
        if user.persona == PERSONA_TROUBLEMAKER:
            turn = self._trouble_turn.get(user.customer_id, 0) + 1
            self._trouble_turn[user.customer_id] = turn
            recalled_action = self._last_text_message.get(user.customer_id)
            if dialog_phase == DIALOG_PHASE_BUSINESS and turn % 3 == 0 and recalled_action:
                event_type = "recall"
                recalled_message_id = recalled_action.message_id
                channel = recalled_action.channel
                conversation_id = recalled_action.conversation_id
                text = "撤回了一条消息"
            elif dialog_phase == DIALOG_PHASE_BUSINESS:
                text = self._with_typo(self.random.choice(QUESTION_POOL))
            else:
                text = self._with_typo(text)

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
            dialog_phase=dialog_phase,
        )
        if event_type == "text":
            self._last_text_message[user.customer_id] = action
        return action

    def first_action(
        self,
        user: VirtualUser,
        *,
        sequence: int,
        dialog_state: DialogStateView | None = None,
        channel_override: str | None = None,
    ) -> SimulationAction | None:
        return self.get_next_action(
            user,
            sequence=sequence,
            due_simulated_seconds=self.interval_for(user),
            dialog_state=dialog_state,
            channel_override=channel_override,
        ) if user.persona != PERSONA_LURKER else None

    def following_action(
        self,
        user: VirtualUser,
        previous: SimulationAction,
        *,
        sequence: int,
        dialog_state: DialogStateView | None = None,
    ) -> SimulationAction | None:
        action = self.get_next_action(
            user,
            sequence=sequence,
            due_simulated_seconds=previous.due_simulated_seconds + self.interval_for(user),
            dialog_state=dialog_state,
        )
        return action

    def materialize_action(
        self,
        action: SimulationAction,
        *,
        user: VirtualUser,
        dialog_state: DialogStateView | None = None,
    ) -> SimulationAction:
        """Generate text only after the scheduler has selected this action.

        Scheduling twenty possible speakers must not spend tokens for actions
        that never leave the priority queue. The deterministic text already on
        ``action`` remains the fail-open fallback.
        """

        if action.event_type != "text" or self.message_generator is None:
            return action
        turn_count = int(getattr(dialog_state, "turn_count", 0))
        last_agent_reply = str(getattr(dialog_state, "last_agent_reply", "") or "")
        request = MessageGenerationRequest(
            sender_id=user.customer_id,
            sender_name=user.display_name,
            persona=user.persona,
            preferred_game=user.preferred_game,
            channel=action.channel,
            conversation_id=action.conversation_id,
            turn_count=turn_count,
            last_agent_reply=last_agent_reply,
            fallback_text=action.text,
            is_follow_up=turn_count >= 1 and reply_requires_user(last_agent_reply),
            dialog_phase=action.dialog_phase,
            business_anchor=str(getattr(dialog_state, "business_anchor", "") or action.text),
        )
        try:
            generated = self.message_generator.generate(request)
            generation = generated if generated.text.strip() else MessageGenerationResult(
                text=action.text,
                source="rule_fallback",
                model=generated.model,
                trace_id=generated.trace_id,
                latency_ms=generated.latency_ms,
                error="generator_returned_empty_text",
            )
        except Exception as exc:
            generation = MessageGenerationResult(
                text=action.text,
                source="rule_fallback",
                error=f"{type(exc).__name__}: {str(exc)[:160]}",
            )
        materialized = replace(
            action,
            text=generation.text,
            generation_source=generation.source,
            generator_model=generation.model,
            generation_trace_id=generation.trace_id,
            generation_latency_ms=generation.latency_ms,
            generation_error=generation.error,
        )
        if materialized.event_type == "text":
            self._last_text_message[user.customer_id] = materialized
        return materialized

    def _should_interrupt_with_chitchat(
        self,
        user: VirtualUser,
        dialog_state: DialogStateView | None,
    ) -> bool:
        """Select a chatty detour without losing the active business thread."""

        if dialog_state is None or not user.interleaves_chitchat:
            return False
        if not bool(getattr(dialog_state, "business_active", False)):
            return False
        if bool(getattr(dialog_state, "chitchat_pending_resume", False)):
            return False
        if int(getattr(dialog_state, "business_turn_count", 0)) < 1:
            return False
        if int(getattr(dialog_state, "business_turns_since_chitchat", 0)) < 1:
            return False
        if (
            int(getattr(dialog_state, "chitchat_interruption_count", 0))
            >= self.max_chitchat_interruptions_per_dialog
        ):
            return False
        return self.random.random() < self.chitchat_probability

    def _follow_up_reply(self, agent_reply: str) -> str:
        if "几点" in agent_reply:
            candidates = FOLLOW_UP_REPLY_POOL[0:2]
        elif "几人" in agent_reply:
            candidates = FOLLOW_UP_REPLY_POOL[2:4]
        elif "确认" in agent_reply:
            candidates = FOLLOW_UP_REPLY_POOL[8:10]
        else:
            candidates = FOLLOW_UP_REPLY_POOL[4:]
        return self.random.choice(candidates)

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


def reply_requires_user(reply: str) -> bool:
    """Whether an Agent reply explicitly asks the user for another turn."""

    normalized = str(reply or "")
    return any(cue in normalized for cue in QUESTION_CUES)
