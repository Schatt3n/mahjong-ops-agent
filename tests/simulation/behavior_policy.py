"""Layer 2: deterministic personas and message behavior policy."""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

try:  # pytest 使用包导入，直接执行命令行文件时使用本地导入。
    from .sim_factory import (
        GROUP_ID,
        PERSONA_ACTIVE_GAMBLER,
        PERSONA_LURKER,
        PERSONA_TROUBLEMAKER,
        VirtualUser,
    )
except ImportError:  # pragma: no cover - 直接执行脚本时使用本地导入
    from sim_factory import (  # type: ignore
        GROUP_ID,
        PERSONA_ACTIVE_GAMBLER,
        PERSONA_LURKER,
        PERSONA_TROUBLEMAKER,
        VirtualUser,
    )


QUESTION_POOL: tuple[str, ...] = (
    "今晚有局吗",
    "{game}还有位吗",
    "包间多少钱",
    "还有位置吗",
    "现在{stake}有人吗",
    "{stake}{smoke}有人齐开吗",
    "帮我约个{start}的",
    "{game}还有位置吗",
    "{game}有人打吗",
    "人齐开还有吗",
    "现在最快几点能开",
    "帮我约个{smoke}局",
    "我这边{party}个人，{start}可以",
    "还有局能拼吗",
    "{game}我可以来",
    "{game}{stake}有人吗",
    "能打{duration}个小时吗",
    "{smoke}的可以",
    "今天房间满了吗",
    "能不能帮我找人",
)

# 剧本只渲染客户本人固定条件，规则回退不再随机抽取确认短句。
FIRST_TURN_QUESTION_POOL = QUESTION_POOL
NEW_TOPIC_POOL: tuple[str, ...] = (
    "对了，还有别的{game}局吗",
    "{start}有人打吗",
    "有{smoke}的再叫我",
    "{game}那桌还缺人吗",
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
    "对了，刚才说的有消息了吗",
    "说回刚才那事，继续帮我看看",
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
    last_thread_id: str | None
    last_channel: str | None
    business_active: bool
    business_anchor: str
    business_conversation_id: str | None
    business_thread_id: str | None
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
    customer_facts: dict[str, Any] = field(default_factory=dict)
    visible_history: tuple[dict[str, str], ...] = ()


@dataclass(slots=True, frozen=True)
class MessageGenerationResult:
    """One generated utterance plus audit metadata for the run report."""

    text: str
    source: str
    model: str | None = None
    trace_id: str | None = None
    latency_ms: float | None = None
    error: str | None = None
    reference_case_ids: tuple[str, ...] = ()
    wants_to_continue: bool = True
    decision_reason: str = ""


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
    thread_id: str = field(default="", compare=False)
    event_type: str = field(default="text", compare=False)
    recalled_message_id: str | None = field(default=None, compare=False)
    generation_source: str = field(default="rule", compare=False)
    generator_model: str | None = field(default=None, compare=False)
    generation_trace_id: str | None = field(default=None, compare=False)
    generation_latency_ms: float | None = field(default=None, compare=False)
    generation_error: str | None = field(default=None, compare=False)
    generation_reference_case_ids: tuple[str, ...] = field(default=(), compare=False)
    dialog_phase: str = field(default=DIALOG_PHASE_BUSINESS, compare=False)
    generation_context: dict[str, Any] = field(default_factory=dict, compare=False)
    customer_wants_to_continue: bool = field(default=True, compare=False)
    customer_decision_reason: str = field(default="", compare=False)

    @property
    def message_id(self) -> str:
        return f"sim_message_{self.sequence:06d}"

    @property
    def coordination_scope(self) -> str:
        """Serialize only this logical topic, not every message in the room."""

        return f"{self.conversation_id}::{self.thread_id or 'legacy'}"

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
            "text": self.text if self.event_type == "text" else "",
            "aggregate_fragments": False,
            "metadata": {
                "channel": "wechaty",
                "simulation": True,
                "simulation_event_type": self.event_type,
                "event_type": "message_recalled" if self.event_type == "recall" else "text",
                "target_message_id": self.recalled_message_id,
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
        inject_recall_events: bool = False,
    ) -> None:
        if not 0.0 <= chitchat_probability <= 1.0:
            raise ValueError("chitchat_probability must be between 0 and 1")
        if not 0.0 <= business_return_probability <= 1.0:
            raise ValueError("business_return_probability must be between 0 and 1")
        if max_chitchat_interruptions_per_dialog < 0:
            raise ValueError("max_chitchat_interruptions_per_dialog must not be negative")
        self.inject_recall_events = bool(inject_recall_events)
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
            text = self._follow_up_reply(user)
        elif bool(getattr(dialog_state, "business_active", False)):
                # 后台匹配任务可能暂时沉默但尚未完成；部分用户稍后会回来查询进度，
                # 另一些用户则会在本轮模拟中自然停止发送消息。
            if self.random.random() >= self.business_return_probability:
                return None
            text = self.random.choice(BUSINESS_RESUME_POOL)
            dialog_phase = DIALOG_PHASE_BUSINESS_RESUME
        else:
            # 非疑问式回复结束当前交流；用户随后可能开启新话题，
            # 也可能在本轮调度中保持沉默。
            if self.random.random() < 0.50:
                return None
            text = self.random.choice(NEW_TOPIC_POOL)

        previous_conversation_id = getattr(dialog_state, "last_conversation_id", None)
        previous_thread_id = getattr(dialog_state, "last_thread_id", None)
        previous_channel = getattr(dialog_state, "last_channel", None)
        business_conversation_id = getattr(dialog_state, "business_conversation_id", None)
        business_thread_id = getattr(dialog_state, "business_thread_id", None)
        business_channel = getattr(dialog_state, "business_channel", None)
        continues_business_thread = dialog_phase in {
            DIALOG_PHASE_CHITCHAT,
            DIALOG_PHASE_BUSINESS_RESUME,
        }
        if continues_business_thread and business_conversation_id and business_channel:
            channel = str(business_channel)
            conversation_id = str(business_conversation_id)
            thread_id = str(business_thread_id or previous_thread_id or "")
        elif is_answering_agent and previous_conversation_id and previous_channel:
            channel = str(previous_channel)
            conversation_id = str(previous_conversation_id)
            thread_id = str(previous_thread_id or "")
        elif channel_override is not None:
            channel = channel_override
            conversation_id = (
                f"sim:group:{GROUP_ID}"
                if channel == "group"
                else f"sim:private:{user.customer_id}"
            )
            thread_id = f"sim:thread:{user.customer_id}:{sequence:06d}"
        else:
            channel = "group" if self.random.random() < 0.80 else "private"
            conversation_id = (
                f"sim:group:{GROUP_ID}"
                if channel == "group"
                else f"sim:private:{user.customer_id}"
            )
            thread_id = f"sim:thread:{user.customer_id}:{sequence:06d}"

        if not thread_id:
            thread_id = f"sim:thread:{user.customer_id}:{sequence:06d}"

        event_type = "text"
        recalled_message_id: str | None = None
        if user.persona == PERSONA_TROUBLEMAKER:
            turn = self._trouble_turn.get(user.customer_id, 0) + 1
            self._trouble_turn[user.customer_id] = turn
            recalled_action = self._last_text_message.get(user.customer_id)
            if self.inject_recall_events and dialog_phase == DIALOG_PHASE_BUSINESS and turn % 3 == 0 and recalled_action:
                event_type = "recall"
                recalled_message_id = recalled_action.message_id
                channel = recalled_action.channel
                conversation_id = recalled_action.conversation_id
                thread_id = recalled_action.thread_id
                text = "撤回了一条消息"
            else:
                text = self._with_typo(text)

        if event_type == "text":
            text = self._format_scenario_text(text, user)

        action = SimulationAction(
            due_simulated_seconds=due_simulated_seconds,
            sequence=sequence,
            channel=channel,
            conversation_id=conversation_id,
            sender_id=user.customer_id,
            sender_name=user.display_name,
            text=text,
            thread_id=thread_id,
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
        客户模型失败时保留错误且不发送规则替身，避免把固定话术伪装成正常客户。
        """

        turn_count = int(getattr(dialog_state, "turn_count", 0))
        last_agent_reply = str(getattr(dialog_state, "last_agent_reply", "") or "")
        context = {
            "scenario_text": action.text if self.message_generator is None or turn_count == 0 else None,
            "fixed_reply_used": self.message_generator is None,
            "last_agent_reply": last_agent_reply,
            "business_anchor": str(getattr(dialog_state, "business_anchor", "") or action.text),
            "persona": user.persona,
            "preferred_game": user.preferred_game,
            "turn_count": turn_count,
            "dialog_phase": action.dialog_phase,
            "customer_facts": user.scenario_facts(),
            "visible_history": list(getattr(dialog_state, "visible_history", []))[-12:],
        }
        if action.event_type != "text" or self.message_generator is None:
            return replace(action, generation_context=context)
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
            customer_facts=user.scenario_facts(),
            visible_history=tuple(getattr(dialog_state, "visible_history", []))[-12:],
        )
        try:
            generated = self.message_generator.generate(request)
            generation = generated if generated.text.strip() or generated.source == "generation_failed" else MessageGenerationResult(
                text="",
                source="generation_failed",
                model=generated.model,
                trace_id=generated.trace_id,
                latency_ms=generated.latency_ms,
                error="generator_returned_empty_text",
            )
        except Exception as exc:
            generation = MessageGenerationResult(
                text="",
                source="generation_failed",
                error=f"{type(exc).__name__}: {str(exc)[:160]}",
            )
        materialized = replace(
            action,
            text=generation.text,
            generation_context=context,
            generation_source=generation.source,
            generator_model=generation.model,
            generation_trace_id=generation.trace_id,
            generation_latency_ms=generation.latency_ms,
            generation_error=generation.error,
            generation_reference_case_ids=generation.reference_case_ids,
            customer_wants_to_continue=generation.wants_to_continue,
            customer_decision_reason=generation.decision_reason,
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

        if self.message_generator is not None or dialog_state is None or not user.interleaves_chitchat:
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

    def _follow_up_reply(self, user: VirtualUser) -> str:
        """规则模式明确给出固定场景条件，避免随机短句冒充对追问的回答。

        自然地只回答当前问题由一次性客户模型完成；离线回退宁可多给条件，
        也不从“确认/都行”池中抽签，更不读取主 Agent 的内部 missing_facts。
        """

        return self._format_scenario_text(
            "我打{game}，我这边{party}个人，{stake}，{smoke}，{start}，{duration}个小时", user,
        )

    @staticmethod
    def _format_scenario_text(template: str, user: VirtualUser) -> str:
        """渲染模拟剧本的显式占位符，事实只来自客户本人固定设定。"""

        game = {"sichuan_mahjong": "川麻", "national_standard_mahjong": "国标", "hangzhou_mahjong": "杭麻"}.get(user.preferred_game, user.preferred_game)
        smoke = {"no_smoking": "无烟", "smoking": "有烟", "any": "烟都可以"}.get(user.smoke_preference, user.smoke_preference)
        return template.format(game=game, stake=user.preferred_stake, smoke=smoke,
                               party=user.party_size, start=user.start_time_text, duration=user.duration_hours)

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
