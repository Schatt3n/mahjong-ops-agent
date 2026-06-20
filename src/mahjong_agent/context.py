from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .core import AgentCore
from .models import DEFAULT_TZ, GameRequest, GameStatus, Message, RoomHoldStatus
from .signals import extract_intent_evidence


CONTEXT_SCHEMA_VERSION = "mahjong_context.v1"
CONTEXT_BUILDER_VERSION = "context_builder.2026-06-20"


@dataclass(slots=True)
class ContextBuilderConfig:
    max_context_chars: int = 12000
    max_recent_messages: int = 12
    max_open_games: int = 8
    max_room_holds: int = 8
    max_string_chars: int = 500
    redact_sensitive: bool = True
    include_recent_messages: bool = True
    stable_hash_salt: str = "mahjong-agent-context-v1"


@dataclass(slots=True)
class ContextBuildResult:
    context: dict[str, Any]
    notes: list[str] = field(default_factory=list)
    estimated_chars: int = 0
    redaction_counts: dict[str, int] = field(default_factory=dict)
    context_digest: str | None = None


class ContextBuilder:
    """Builds audited, budgeted, privacy-aware LLM context for one workflow run.

    The builder owns what the model can see. It does not let the model mutate
    state, produce ids for writes, or decide which side-effect tools are
    executable.
    """

    sensitive_patterns = (
        ("phone", re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"), "[手机号]"),
        ("long_number", re.compile(r"(?<!\d)\d{13,19}(?!\d)"), "[长数字]"),
        (
            "wechat_id",
            re.compile(
                r"(?i)(wxid_[a-z0-9_]{6,}|微信(?:号)?[:：]?\s*[a-z][-_a-z0-9]{5,20}|wechat[:：]?\s*[a-z][-_a-z0-9]{5,20})"
            ),
            "[微信号]",
        ),
        (
            "payment",
            re.compile(r"(支付宝|银行卡|收款码|付款码|身份证|转账|代收|代付)[^\s，。,.]{0,24}"),
            "[敏感信息]",
        ),
    )

    known_local_aliases = {
        "cq": "杭麻财敲",
        "371": "三缺一",
        "272": "二缺二",
        "173": "一缺三",
        "216": "2-16档，底注2，封顶16",
        "1-32": "1-32档，底注1，封顶32",
        "半块": "0.5档",
        "五毛": "0.5档",
        "不抽": "无烟局",
    }

    safety_boundaries = [
        "LLM 只能解释语义、归一化事实、提出建议，不能直接改状态。",
        "占座、取消局、锁客户、发送消息必须由后端状态机和 outbox 提交。",
        "涉及抽水、赌资、输赢结算、代收代付、借码、上分下分必须转人工。",
        "低置信度、身份不确定、房态冲突、纠纷类消息必须转人工或追问。",
        "trace_id、message_ref、customer_ref、game_ref 是后端生成的只读引用，模型返回时不得作为可信写入依据。",
    ]

    def __init__(self, core: AgentCore, config: ContextBuilderConfig | None = None) -> None:
        self.core = core
        self.config = config or ContextBuilderConfig()

    def build(
        self,
        message: Message,
        now: datetime | None = None,
        goal: str = "semantic_resolver",
        stage: str = "interpret_message",
    ) -> ContextBuildResult:
        effective_now = now or datetime.now(DEFAULT_TZ)
        redaction_counts: dict[str, int] = {}
        evidence = extract_intent_evidence(message)
        conversation_id = self._conversation_id(message)
        trace_id = self._trace_id(message)

        context: dict[str, Any] = {
            "schema_version": CONTEXT_SCHEMA_VERSION,
            "builder_version": CONTEXT_BUILDER_VERSION,
            "runtime": {
                "trace_id": trace_id,
                "built_at": effective_now.isoformat(),
                "conversation_ref": self._stable_ref("conversation", conversation_id),
                "message_ref": self._stable_ref("message", message.id),
                "read_only_fields": [
                    "trace_id",
                    "conversation_ref",
                    "message_ref",
                    "customer_ref",
                    "game_ref",
                ],
            },
            "goal": goal,
            "stage": stage,
            "current_message": self._current_message(message, evidence, redaction_counts),
            "conversation_summary": self._conversation_summary(message, conversation_id, redaction_counts),
            "customer_profile_summary": self._customer_profile(message, redaction_counts),
            "game_state_snapshot": self._game_state_snapshot(redaction_counts),
            "room_state_snapshot": self._room_state_snapshot(redaction_counts),
            "tool_policy": self._tool_policy(stage),
            "allowed_tools": [],
            "rag_snippets": self._rag_snippets(),
            "known_local_aliases": dict(self.known_local_aliases),
            "output_schema": self._output_schema(),
            "safety_boundaries": list(self.safety_boundaries),
            "context_budget": {
                "max_context_chars": self.config.max_context_chars,
                "estimated_chars": 0,
                "trimmed_sections": [],
                "max_recent_messages": self.config.max_recent_messages,
                "max_open_games": self.config.max_open_games,
            },
            "privacy": {
                "redact_sensitive": self.config.redact_sensitive,
                "redaction_counts": {},
                "identity_policy": "stable_refs_only",
            },
            "audit": {
                "sources": [
                    "current_message",
                    "conversation_summary",
                    "customer_profile_summary",
                    "game_state_snapshot",
                    "room_state_snapshot",
                    "builtin_ruleset_aliases",
                ],
                "requires_persistence": True,
            },
        }

        # Backward-compatible aliases for the existing semantic resolver prompt.
        context["sender_profile"] = context["customer_profile_summary"]
        context["recent_open_games"] = context["game_state_snapshot"]["recent_open_games"]

        estimated_chars, trimmed_sections = self._fit_budget(context)
        context["context_budget"] = {
            "max_context_chars": self.config.max_context_chars,
            "estimated_chars": estimated_chars,
            "trimmed_sections": trimmed_sections,
            "max_recent_messages": self.config.max_recent_messages,
            "max_open_games": self.config.max_open_games,
        }
        context["privacy"] = {
            "redact_sensitive": self.config.redact_sensitive,
            "redaction_counts": dict(redaction_counts),
            "identity_policy": "stable_refs_only",
        }
        estimated_chars, extra_trimmed_sections = self._fit_budget(context)
        if extra_trimmed_sections:
            trimmed_sections = sorted(set([*trimmed_sections, *extra_trimmed_sections]))
        context["context_budget"]["estimated_chars"] = estimated_chars
        context["context_budget"]["trimmed_sections"] = trimmed_sections

        estimated_chars = self._estimate_chars(context)
        context["context_budget"]["estimated_chars"] = estimated_chars
        context_digest = self._context_digest(context)
        context["audit"]["context_digest"] = context_digest
        notes = [
            f"ContextBuilder schema={CONTEXT_SCHEMA_VERSION}, digest={context_digest}, estimated_chars={estimated_chars}",
        ]
        if redaction_counts:
            notes.append(f"ContextBuilder redacted={dict(redaction_counts)}")
        if trimmed_sections:
            notes.append(f"ContextBuilder trimmed_sections={trimmed_sections}")

        return ContextBuildResult(
            context=context,
            notes=notes,
            estimated_chars=estimated_chars,
            redaction_counts=dict(redaction_counts),
            context_digest=context_digest,
        )

    def _current_message(
        self,
        message: Message,
        evidence,
        redaction_counts: dict[str, int],
    ) -> dict[str, Any]:
        return {
            "text": self._redact_text(evidence.combined_text or message.text, redaction_counts),
            "sent_at": message.sent_at.isoformat(),
            "channel_type": message.channel_type.value,
            "sender_ref": self._stable_ref("customer", message.sender_id),
            "sender_display_name": self._redact_text(message.sender_name, redaction_counts),
            "modalities": evidence.modalities,
            "evidence": {
                key: self._redact_text(value, redaction_counts)
                for key, value in evidence.evidence.items()
            },
            "lead_score": evidence.lead_score,
            "lead_reasons": evidence.lead_reasons,
            "source": {
                "message_ref": self._stable_ref("message", message.id),
                "timestamp": message.sent_at.isoformat(),
                "confidence": 1.0,
                "sensitivity": "user_content",
                "version": "message.v1",
            },
        }

    def _conversation_summary(
        self,
        message: Message,
        conversation_id: str,
        redaction_counts: dict[str, int],
    ) -> dict[str, Any]:
        if not self.config.include_recent_messages:
            recent_messages: list[dict[str, Any]] = []
        else:
            candidates = [
                item
                for item in self.core.store.messages.values()
                if self._conversation_id(item) == conversation_id
                and item.channel_type == message.channel_type
            ]
            if not any(item.id == message.id for item in candidates):
                candidates.append(message)
            candidates = sorted(candidates, key=lambda item: (item.sent_at, item.id))
            recent_messages = [
                self._message_summary(item, redaction_counts)
                for item in candidates[-self.config.max_recent_messages :]
            ]

        return {
            "conversation_ref": self._stable_ref("conversation", conversation_id),
            "channel_type": message.channel_type.value,
            "recent_messages": recent_messages,
            "source": {
                "timestamp": message.sent_at.isoformat(),
                "confidence": 0.9,
                "sensitivity": "conversation_history",
                "version": "conversation_summary.v1",
            },
        }

    def _message_summary(self, message: Message, redaction_counts: dict[str, int]) -> dict[str, Any]:
        evidence = extract_intent_evidence(message)
        return {
            "message_ref": self._stable_ref("message", message.id),
            "sender_ref": self._stable_ref("customer", message.sender_id),
            "sender_display_name": self._redact_text(message.sender_name, redaction_counts),
            "sent_at": message.sent_at.isoformat(),
            "text": self._redact_text(evidence.combined_text or message.text, redaction_counts),
            "modalities": evidence.modalities,
        }

    def _customer_profile(self, message: Message, redaction_counts: dict[str, int]) -> dict[str, Any] | None:
        customer = self.core.store.customers.get(message.sender_id)
        if customer is None:
            return None
        return {
            "customer_ref": self._stable_ref("customer", customer.id),
            "display_name": self._redact_text(customer.display_name, redaction_counts),
            "aliases": [self._redact_text(alias, redaction_counts) for alias in customer.aliases[:6]],
            "preferred_levels": list(customer.preferred_levels[:8]),
            "tags": list(customer.tags[:12]),
            "smoke_free_preference": customer.smoke_free_preference,
            "usual_party_size": customer.usual_party_size,
            "usual_party_size_confidence": customer.usual_party_size_confidence,
            "usual_start_hours": list(customer.usual_start_hours[:8]),
            "usual_weekdays": list(customer.usual_weekdays[:7]),
            "fatigue_policy": {
                "max_games_per_day": customer.max_games_per_day,
                "min_hours_between_games": customer.min_hours_between_games,
                "invite_cooldown_hours": customer.invite_cooldown_hours,
                "daily_invite_limit": customer.daily_invite_limit,
                "fatigue_sensitivity": customer.fatigue_sensitivity,
                "no_contact": customer.no_contact,
            },
            "play_preferences": [
                {
                    "game_type": preference.game_type,
                    "preferred_levels": list(preference.preferred_levels[:8]),
                    "preferred_rulesets": list(preference.preferred_rulesets[:8]),
                    "preferred_variants": list(preference.preferred_variants[:8]),
                    "preferred_play_options": list(preference.preferred_play_options[:12]),
                    "avoid_play_options": list(preference.avoid_play_options[:12]),
                }
                for preference in customer.play_preferences[:8]
            ],
            "source": {
                "timestamp": message.sent_at.isoformat(),
                "confidence": 0.85,
                "sensitivity": "profile_summary",
                "version": "customer_profile.v1",
            },
        }

    def _game_state_snapshot(self, redaction_counts: dict[str, int]) -> dict[str, Any]:
        open_games = [
            self._game_summary(game, redaction_counts)
            for game in list(self.core.store.games.values())[-self.config.max_open_games :]
            if game.status
            in {GameStatus.OPEN, GameStatus.NEED_CLARIFICATION, GameStatus.NEGOTIATING, GameStatus.HOLDING}
        ]
        return {
            "recent_open_games": open_games,
            "source": {
                "timestamp": datetime.now(DEFAULT_TZ).isoformat(),
                "confidence": 0.95,
                "sensitivity": "operational_state",
                "version": "game_state.v1",
            },
        }

    def _game_summary(self, game: GameRequest, redaction_counts: dict[str, int]) -> dict[str, Any]:
        return {
            "game_ref": self._stable_ref("game", game.id),
            "organizer_ref": self._stable_ref("customer", game.organizer_id),
            "organizer_name": self._redact_text(game.organizer_name, redaction_counts),
            "channel_ref": self._stable_ref("conversation", game.channel_id),
            "status": game.status.value,
            "game_type": game.game_type,
            "ruleset": game.ruleset,
            "variant": game.variant,
            "level": game.level,
            "base_score": game.base_score,
            "cap_score": game.cap_score,
            "start_at": game.start_at.isoformat() if game.start_at else None,
            "start_time_confidence": game.start_time_confidence,
            "duration_hours": game.duration_hours,
            "current_player_count": game.current_player_count,
            "missing_count": game.missing_count,
            "open_slots": game.open_slots,
            "rules": list(game.rules[:12]),
            "play_options": list(game.play_options[:12]),
            "ambiguities": list(game.ambiguities[:8]),
            "updated_at": game.updated_at.isoformat(),
            "version": game.version,
        }

    def _room_state_snapshot(self, redaction_counts: dict[str, int]) -> dict[str, Any]:
        active_holds = [
            hold
            for hold in self.core.store.room_holds.values()
            if hold.status == RoomHoldStatus.ACTIVE
        ]
        active_holds = sorted(active_holds, key=lambda hold: (hold.start_at, hold.id))[: self.config.max_room_holds]
        return {
            "capacity": self.core.store.room_capacity,
            "active_holds": [
                {
                    "hold_ref": self._stable_ref("room_hold", hold.id),
                    "room_ref": self._stable_ref("room", hold.room_id) if hold.room_id else None,
                    "start_at": hold.start_at.isoformat(),
                    "end_at": hold.end_at.isoformat(),
                    "source": self._redact_text(hold.source, redaction_counts),
                    "game_ref": self._stable_ref("game", hold.game_id) if hold.game_id else None,
                }
                for hold in active_holds
            ],
            "source": {
                "timestamp": datetime.now(DEFAULT_TZ).isoformat(),
                "confidence": 0.95,
                "sensitivity": "room_state",
                "version": "room_state.v1",
            },
        }

    def _tool_policy(self, stage: str) -> dict[str, Any]:
        return {
            "tool_calling_enabled": False,
            "stage": stage,
            "available_tools": [],
            "reason": "当前 LLMResolver 是语义解析器；生产 tool call 必须由后端 ToolRouter 按状态注入，并经 ToolGateway 校验执行。",
            "side_effect_tools_require_backend_commit": True,
        }

    def _rag_snippets(self) -> list[dict[str, Any]]:
        return [
            {
                "source": "builtin_ruleset_aliases",
                "version": "local_aliases.v1",
                "confidence": 0.9,
                "sensitivity": "business_dictionary",
                "content": dict(self.known_local_aliases),
            }
        ]

    def _output_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["is_mahjong_related", "intent", "confidence"],
            "properties": {
                "is_mahjong_related": "boolean",
                "intent": "find_players|join_game|cancel_or_full|update_game|irrelevant|uncertain",
                "confidence": "number between 0 and 1",
                "normalized_text": "optional normalized Chinese text",
                "reply_text": "optional follow-up text",
                "needs_human_review": "boolean",
                "facts": "object",
            },
        }

    def _fit_budget(self, context: dict[str, Any]) -> tuple[int, list[str]]:
        trimmed_sections: list[str] = []
        estimated = self._estimate_chars(context)
        if self.config.max_context_chars <= 0:
            return estimated, trimmed_sections

        recent_messages = context["conversation_summary"].get("recent_messages") or []
        open_games = context["game_state_snapshot"].get("recent_open_games") or []
        room_holds = context["room_state_snapshot"].get("active_holds") or []

        while estimated > self.config.max_context_chars:
            if len(recent_messages) > 1:
                recent_messages.pop(0)
                trimmed_sections.append("conversation_summary.recent_messages")
            elif len(open_games) > 1:
                open_games.pop(0)
                trimmed_sections.append("game_state_snapshot.recent_open_games")
            elif len(room_holds) > 0:
                room_holds.pop()
                trimmed_sections.append("room_state_snapshot.active_holds")
            elif context.get("rag_snippets"):
                context["rag_snippets"] = []
                trimmed_sections.append("rag_snippets")
            else:
                break
            estimated = self._estimate_chars(context)
        return estimated, sorted(set(trimmed_sections))

    def _conversation_id(self, message: Message) -> str:
        return str(message.metadata.get("conversation_id") or message.channel_id)

    def _trace_id(self, message: Message) -> str:
        raw = str(message.metadata.get("trace_id") or "")
        if re.fullmatch(r"[A-Za-z0-9_.:-]{8,128}", raw):
            return raw
        return self._stable_ref("trace", message.id)

    def _stable_ref(self, prefix: str, raw: str | None) -> str | None:
        if raw is None:
            return None
        digest = hashlib.sha256(f"{self.config.stable_hash_salt}:{prefix}:{raw}".encode("utf-8")).hexdigest()[:16]
        return f"{prefix}_{digest}"

    def _redact_text(self, text: str | None, redaction_counts: dict[str, int]) -> str:
        if text is None:
            return ""
        value = str(text)
        if self.config.redact_sensitive:
            for category, pattern, replacement in self.sensitive_patterns:
                value, count = pattern.subn(replacement, value)
                if count:
                    redaction_counts[category] = redaction_counts.get(category, 0) + count
        return self._truncate(value)

    def _truncate(self, value: str) -> str:
        limit = self.config.max_string_chars
        if limit <= 0 or len(value) <= limit:
            return value
        return value[: max(0, limit - 12)] + "...[truncated]"

    def _estimate_chars(self, context: dict[str, Any]) -> int:
        return len(json.dumps(context, ensure_ascii=False, sort_keys=True))

    def _context_digest(self, context: dict[str, Any]) -> str:
        payload = json.dumps(context, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
        return f"ctx_{digest}"
