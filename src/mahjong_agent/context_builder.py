from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .core import ACTIVE_GAME_STATUSES, AgentCore
from .memory import ShortTermMemoryStore, summarize_short_memory
from .models import (
    ChannelType,
    CustomerProfile as LegacyCustomerProfile,
    DEFAULT_TZ,
    GameRequest,
    Message,
    RoomHoldStatus,
)
from .workflow_models import (
    ConversationContext,
    CustomerProfile,
    GameRequirement,
    SlotSource,
    SlotValue,
    UserMessage,
    WorkflowTurn,
    new_workflow_id,
)


@dataclass(slots=True)
class WorkflowContextBuilderConfig:
    max_recent_turns: int = 8
    max_memory_records: int = 6
    max_open_games: int = 12
    max_room_holds: int = 8
    include_raw_conversation_history: bool = True


@dataclass(slots=True)
class WorkflowContextBuildResult:
    context: ConversationContext
    used_short_memory: bool
    followup_context: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


class WorkflowContextBuilder:
    """Builds the controlled-agent ConversationContext contract.

    This builder reads operational state and memory, but it does not parse user
    intent, choose actions, call tools, mutate state, or generate replies.
    """

    def __init__(
        self,
        core: AgentCore,
        memory_store: ShortTermMemoryStore | None = None,
        config: WorkflowContextBuilderConfig | None = None,
    ) -> None:
        self.core = core
        self.memory_store = memory_store
        self.config = config or WorkflowContextBuilderConfig()

    def build(
        self,
        message: Message,
        now: datetime | None = None,
        trace_id: str | None = None,
    ) -> WorkflowContextBuildResult:
        effective_now = now or datetime.now(DEFAULT_TZ)
        conversation_id = self._conversation_id(message)
        effective_trace_id = trace_id or str(message.metadata.get("trace_id") or new_workflow_id("trace"))
        current_message = self._user_message_from_message(message, conversation_id, effective_trace_id)
        memory_records = (
            self.memory_store.load(
                conversation_id=conversation_id,
                sender_id=message.sender_id,
                now=effective_now,
                limit=self.config.max_memory_records,
            )
            if self.memory_store
            else []
        )
        memory_turns = [record.to_workflow_turn() for record in memory_records]
        history_turns = self._conversation_history_turns(
            current=message,
            conversation_id=conversation_id,
            trace_id=effective_trace_id,
        )
        recent_turns = self._merge_recent_turns(history_turns, memory_turns)
        active_game = self._latest_game_requirement(memory_turns)
        followup_context = self._followup_context(current_message, memory_turns)
        context = ConversationContext(
            current_message=current_message,
            customer_profile=self._customer_profile(message.sender_id),
            recent_turns=recent_turns,
            active_game=active_game,
            open_games=self._open_games(),
            room_state=self._room_state(),
            memory_summary=summarize_short_memory(memory_records),
            followup_context=followup_context,
            trace_notes=[
                "context_builder.workflow_contract.v1",
                "上下文构建器只提供事实和候选信号，不决定最终业务动作。",
                "上一轮系统回复和短期记忆已显式放入 followup_context，供 LLM 判断当前消息是否为追问回答。",
            ],
        )
        return WorkflowContextBuildResult(
            context=context,
            used_short_memory=bool(memory_records),
            followup_context=followup_context,
            notes=[
                f"conversation_id={conversation_id}",
                f"recent_turns={len(recent_turns)}",
                f"open_games={len(context.open_games)}",
                f"used_short_memory={bool(memory_records)}",
            ],
        )

    def _conversation_history_turns(
        self,
        current: Message,
        conversation_id: str,
        trace_id: str,
    ) -> list[WorkflowTurn]:
        if not self.config.include_raw_conversation_history:
            return []
        messages = [
            item
            for item in self.core.store.messages.values()
            if item.id != current.id
            and self._conversation_id(item) == conversation_id
            and item.channel_type == current.channel_type
        ]
        messages = sorted(messages, key=lambda item: (item.sent_at, item.id))[-self.config.max_recent_turns :]
        return [
            WorkflowTurn(
                user_message=self._user_message_from_message(item, conversation_id, trace_id),
                at=item.sent_at,
            )
            for item in messages
        ]

    def _merge_recent_turns(
        self,
        history_turns: list[WorkflowTurn],
        memory_turns: list[WorkflowTurn],
    ) -> list[WorkflowTurn]:
        by_message_id: dict[str, WorkflowTurn] = {}
        for turn in [*history_turns, *memory_turns]:
            message_id = turn.user_message.message_id
            existing = by_message_id.get(message_id)
            if existing is None or (turn.system_reply and not existing.system_reply):
                by_message_id[message_id] = turn
        turns = sorted(by_message_id.values(), key=lambda turn: (turn.at, turn.user_message.message_id))
        return turns[-self.config.max_recent_turns :]

    def _followup_context(
        self,
        current_message: UserMessage,
        memory_turns: list[WorkflowTurn],
    ) -> dict[str, Any]:
        previous_turn = next((turn for turn in reversed(memory_turns) if turn.system_reply), None)
        if previous_turn is None:
            return {
                "schema_version": "followup_context.v1",
                "has_previous_system_reply": False,
                "previous_turn": None,
                "unresolved_questions": [],
                "expected_answer_type": "none",
                "current_message_response_type": self._current_message_response_type(current_message.text, []),
                "should_treat_current_message_as_followup": False,
                "current_message_may_answer_previous_reply": False,
                "signals": {},
            }
        current_text = current_message.text.strip()
        previous_reply = previous_turn.system_reply or ""
        unresolved_questions = self._unresolved_questions(previous_reply)
        response_type = self._current_message_response_type(current_text, unresolved_questions)
        short_ack = response_type == "short_ack"
        asks_create_confirmation = "create_confirmation" in unresolved_questions
        asks_clarification = any(item != "create_confirmation" for item in unresolved_questions)
        should_treat_as_followup = bool(
            response_type in {"short_ack", "slot_fill", "correction", "negative"}
            and (asks_create_confirmation or asks_clarification)
        )
        return {
            "schema_version": "followup_context.v1",
            "has_previous_system_reply": True,
            "previous_turn": {
                "message_id": previous_turn.user_message.message_id,
                "user_text": previous_turn.user_message.text,
                "system_reply": previous_reply,
                "at": previous_turn.at.isoformat(),
            },
            "previous_user_text": previous_turn.user_message.text,
            "previous_system_reply": previous_reply,
            "previous_game_requirement": previous_turn.game_requirement.to_prompt_dict()
            if previous_turn.game_requirement
            else None,
            "unresolved_questions": unresolved_questions,
            "expected_answer_type": self._expected_answer_type(unresolved_questions),
            "current_message_response_type": response_type,
            "should_treat_current_message_as_followup": should_treat_as_followup,
            "current_message_may_answer_previous_reply": should_treat_as_followup,
            "signals": {
                "current_message_is_short_ack": short_ack,
                "current_message_is_slot_fill": response_type == "slot_fill",
                "current_message_is_correction": response_type == "correction",
                "current_message_is_negative": response_type == "negative",
                "previous_reply_asked_create_confirmation": asks_create_confirmation,
                "previous_reply_asked_clarification": asks_clarification,
            },
            "instruction": (
                "这是上下文信号，不是最终动作。LLM 需要结合 current_message、previous_turn、"
                "previous_game_requirement 和 unresolved_questions 判断本轮是在确认、拒绝、纠正还是补充上一轮老板建议。"
            ),
        }

    def _unresolved_questions(self, previous_reply: str) -> list[str]:
        questions: list[str] = []
        if any(token in previous_reply for token in ["要组一个吗", "要不要帮你组", "帮你组一个", "组一个吗"]):
            questions.append("create_confirmation")
        if any(token in previous_reply for token in ["几点", "什么时候", "大概时间", "几点能", "几点开"]):
            questions.append("start_time")
        if any(token in previous_reply for token in ["多大", "打多大", "档位"]):
            questions.append("stake")
        if any(token in previous_reply for token in ["几个人", "一个人吗", "你这边几人", "现在几人"]):
            questions.append("party_size")
        if any(token in previous_reply for token in ["烟况", "无烟", "有烟", "烟都可"]):
            questions.append("smoke")
        if any(token in previous_reply for token in ["几个小时", "打多久", "多久", "通宵"]):
            questions.append("duration")
        return list(dict.fromkeys(questions))

    def _expected_answer_type(self, unresolved_questions: list[str]) -> str:
        if not unresolved_questions:
            return "none"
        if unresolved_questions == ["create_confirmation"]:
            return "yes_no_confirmation"
        if "create_confirmation" in unresolved_questions:
            return "confirmation_or_slot_fill"
        return "slot_fill"

    def _current_message_response_type(self, text: str, unresolved_questions: list[str]) -> str:
        normalized = str(text or "").strip()
        if not normalized:
            return "empty"
        if normalized in {"组", "组吧", "可以", "好", "好的", "行", "要", "来", "打", "可以的", "行的"}:
            return "short_ack"
        if normalized in {"不", "不要", "不组", "算了", "不打了", "取消", "不用了"}:
            return "negative"
        if any(token in normalized for token in ["不是", "不对", "我说的是", "改成", "改到"]):
            return "correction"
        if unresolved_questions and self._looks_like_slot_fill(normalized):
            return "slot_fill"
        return "unknown"

    def _looks_like_slot_fill(self, text: str) -> bool:
        if any(token in text for token in ["点", "半", "小时", "通宵", "人", "个", "烟", "无烟", "有烟", "都可"]):
            return True
        if any(ch.isdigit() for ch in text):
            return True
        return False

    def _customer_profile(self, sender_id: str) -> CustomerProfile | None:
        profile = self.core.store.customers.get(sender_id)
        if profile is None:
            return None
        preferred_slots: dict[str, SlotValue] = {}
        if profile.preferred_levels:
            preferred_slots["stake_preferences"] = self._slot(
                "stake_preferences",
                list(profile.preferred_levels),
                source=SlotSource.PROFILE,
                confidence=0.75,
                evidence="customer_profile.preferred_levels",
            )
        if profile.smoke_free_preference is not None:
            preferred_slots["smoke"] = self._slot(
                "smoke",
                "no_smoke" if profile.smoke_free_preference else "smoke_ok",
                source=SlotSource.PROFILE,
                confidence=0.75,
                evidence="customer_profile.smoke_free_preference",
            )
        if profile.usual_party_size is not None:
            preferred_slots["party_size"] = self._slot(
                "party_size",
                profile.usual_party_size,
                source=SlotSource.PROFILE,
                confidence=profile.usual_party_size_confidence,
                evidence="customer_profile.usual_party_size",
                confirmed=profile.usual_party_size_confidence >= 0.75,
            )
        if profile.usual_start_hours:
            preferred_slots["usual_start_hours"] = self._slot(
                "usual_start_hours",
                list(profile.usual_start_hours),
                source=SlotSource.PROFILE,
                confidence=0.65,
                evidence="customer_profile.usual_start_hours",
            )
        self._add_play_preference_slots(preferred_slots, profile)
        return CustomerProfile(
            customer_id=profile.id,
            display_name=profile.display_name,
            preferred_slots=preferred_slots,
            tags=list(profile.tags),
            recent_facts=self._profile_recent_facts(profile),
            fatigue={
                "no_contact": profile.no_contact,
                "max_games_per_day": profile.max_games_per_day,
                "min_hours_between_games": profile.min_hours_between_games,
                "invite_cooldown_hours": profile.invite_cooldown_hours,
                "daily_invite_limit": profile.daily_invite_limit,
                "decline_count_30d": profile.decline_count_30d,
            },
            metadata={"legacy_profile_id": profile.id},
        )

    def _add_play_preference_slots(
        self,
        preferred_slots: dict[str, SlotValue],
        profile: LegacyCustomerProfile,
    ) -> None:
        game_types: list[str] = []
        rulesets: list[str] = []
        variants: list[str] = []
        play_options: list[str] = []
        for preference in profile.play_preferences:
            if preference.game_type:
                game_types.append(preference.game_type)
            rulesets.extend(preference.preferred_rulesets)
            variants.extend(preference.preferred_variants)
            play_options.extend(preference.preferred_play_options)
        if game_types:
            preferred_slots["game_type_preferences"] = self._slot(
                "game_type_preferences",
                sorted(set(game_types)),
                source=SlotSource.PROFILE,
                confidence=0.8,
                evidence="customer_profile.play_preferences.game_type",
            )
        if rulesets:
            preferred_slots["ruleset_preferences"] = self._slot(
                "ruleset_preferences",
                sorted(set(rulesets)),
                source=SlotSource.PROFILE,
                confidence=0.75,
                evidence="customer_profile.play_preferences.rulesets",
            )
        if variants:
            preferred_slots["variant_preferences"] = self._slot(
                "variant_preferences",
                sorted(set(variants)),
                source=SlotSource.PROFILE,
                confidence=0.75,
                evidence="customer_profile.play_preferences.variants",
            )
        if play_options:
            preferred_slots["play_option_preferences"] = self._slot(
                "play_option_preferences",
                sorted(set(play_options)),
                source=SlotSource.PROFILE,
                confidence=0.7,
                evidence="customer_profile.play_preferences.play_options",
            )

    def _profile_recent_facts(self, profile: LegacyCustomerProfile) -> list[str]:
        facts: list[str] = []
        if profile.preferred_levels:
            facts.append(f"常打档位：{'/'.join(profile.preferred_levels[:4])}")
        if profile.smoke_free_preference is True:
            facts.append("偏好无烟")
        elif profile.smoke_free_preference is False:
            facts.append("可接受有烟")
        if profile.usual_party_size is not None:
            facts.append(f"常见同行人数：{profile.usual_party_size}")
        for preference in profile.play_preferences[:4]:
            label = preference.game_type
            if preference.preferred_variants:
                label += f"({','.join(preference.preferred_variants[:3])})"
            facts.append(f"玩法偏好：{label}")
        for observation in self._controlled_profile_observations(profile)[:8]:
            field = observation.get("field")
            value = observation.get("value")
            evidence = observation.get("evidence")
            if field and value is not None:
                fact = f"画像观察：{field}={value}"
                if evidence:
                    fact += f"；证据：{evidence}"
                facts.append(fact)
        return facts

    def _controlled_profile_observations(self, profile: LegacyCustomerProfile) -> list[dict[str, object]]:
        raw = profile.metadata.get("controlled_profile_observations")
        if not isinstance(raw, list):
            return []
        return [item for item in raw if isinstance(item, dict)]

    def _open_games(self) -> list[GameRequirement]:
        games = [
            game
            for game in self.core.store.games.values()
            if game.status in ACTIVE_GAME_STATUSES or str(game.status.value) == "need_clarification"
        ]
        games = sorted(games, key=lambda game: (game.updated_at, game.id), reverse=True)
        return [self._game_requirement_from_game(game) for game in games[: self.config.max_open_games]]

    def _latest_game_requirement(self, memory_turns: list[WorkflowTurn]) -> GameRequirement | None:
        for turn in reversed(memory_turns):
            if turn.game_requirement:
                return turn.game_requirement
        return None

    def _game_requirement_from_game(self, game: GameRequest) -> GameRequirement:
        requirement = GameRequirement(
            seats_total=game.seats_total,
            organizer_id=game.organizer_id,
            organizer_name=game.organizer_name,
            notes=list(game.notes),
        )
        if game.game_type:
            requirement.set_slot(self._slot("game_type", game.game_type, source=SlotSource.TOOL))
        if game.ruleset:
            requirement.set_slot(self._slot("ruleset", game.ruleset, source=SlotSource.TOOL))
        if game.variant:
            requirement.set_slot(self._slot("variant", game.variant, source=SlotSource.TOOL))
        if game.level:
            requirement.set_slot(self._slot("stake", game.level, source=SlotSource.TOOL))
        if game.base_score is not None:
            requirement.set_slot(self._slot("base_score", game.base_score, source=SlotSource.TOOL))
        if game.cap_score is not None:
            requirement.set_slot(self._slot("cap_score", game.cap_score, source=SlotSource.TOOL))
        if game.start_at is not None:
            requirement.set_slot(self._slot("start_at", game.start_at.isoformat(), source=SlotSource.TOOL))
            requirement.set_slot(self._slot("start_time_mode", "fixed", source=SlotSource.TOOL))
        elif "人齐开" in game.rules:
            requirement.set_slot(self._slot("start_time_mode", "people_ready", source=SlotSource.TOOL))
        if game.current_player_count is not None:
            requirement.set_slot(self._slot("current_player_count", game.current_player_count, source=SlotSource.TOOL))
        if game.missing_count is not None:
            requirement.set_slot(self._slot("missing_count", game.missing_count, source=SlotSource.TOOL))
        if game.current_player_count is not None or game.missing_count is not None:
            requirement.set_slot(
                self._slot(
                    "party_size",
                    {
                        "current_player_count": game.current_player_count,
                        "missing_count": game.missing_count,
                        "seats_total": game.seats_total,
                    },
                    source=SlotSource.TOOL,
                )
            )
        smoke = self._smoke_value(game.rules)
        if smoke:
            requirement.set_slot(self._slot("smoke", smoke, source=SlotSource.TOOL))
        if game.duration_hours is not None:
            requirement.set_slot(self._slot("duration_hours", game.duration_hours, source=SlotSource.TOOL))
            requirement.set_slot(self._slot("duration_mode", "fixed", source=SlotSource.TOOL))
        elif "通宵" in game.rules:
            requirement.set_slot(self._slot("duration_mode", "overnight", source=SlotSource.TOOL))
        if game.play_options:
            requirement.set_slot(self._slot("play_options", list(game.play_options), source=SlotSource.TOOL))
        if game.rules:
            requirement.set_slot(self._slot("rules", list(game.rules), source=SlotSource.TOOL))
        requirement.notes.append(f"status={game.status.value}")
        requirement.notes.append(f"game_id={game.id}")
        return requirement

    def _room_state(self) -> dict[str, Any]:
        holds = [
            hold
            for hold in self.core.store.room_holds.values()
            if hold.status == RoomHoldStatus.ACTIVE
        ]
        holds = sorted(holds, key=lambda hold: (hold.start_at, hold.id))[: self.config.max_room_holds]
        return {
            "capacity": self.core.store.room_capacity,
            "active_holds": [
                {
                    "room_id": hold.room_id,
                    "game_id": hold.game_id,
                    "start_at": hold.start_at.isoformat(),
                    "end_at": hold.end_at.isoformat(),
                    "source": hold.source,
                }
                for hold in holds
            ],
        }

    def _user_message_from_message(
        self,
        message: Message,
        conversation_id: str,
        trace_id: str,
    ) -> UserMessage:
        return UserMessage(
            text=message.text,
            sender_id=message.sender_id,
            sender_name=message.sender_name,
            conversation_id=conversation_id,
            trace_id=trace_id,
            message_id=message.id,
            channel_type=message.channel_type if isinstance(message.channel_type, ChannelType) else ChannelType.MANUAL,
            sent_at=message.sent_at,
            modalities=list(message.metadata.get("modalities") or ["text"]),
            metadata=dict(message.metadata),
        )

    def _slot(
        self,
        name: str,
        value: Any,
        *,
        source: SlotSource,
        confidence: float = 0.9,
        evidence: str | None = None,
        confirmed: bool = True,
    ) -> SlotValue:
        return SlotValue(
            name=name,
            value=value,
            source=source,
            confidence=confidence,
            confirmed=confirmed,
            needs_confirmation=not confirmed,
            evidence=evidence,
        )

    def _smoke_value(self, rules: list[str]) -> str | None:
        if "烟况都可" in rules:
            return "any"
        if "无烟" in rules:
            return "no_smoke"
        if "可吸烟" in rules or "有烟" in rules:
            return "smoke_ok"
        return None

    def _conversation_id(self, message: Message) -> str:
        return str(message.metadata.get("conversation_id") or message.channel_id)
