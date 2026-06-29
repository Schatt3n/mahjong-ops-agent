from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .core import AgentCore
from .models import DEFAULT_TZ, GameRequest, GameStatus, Message, RoomHoldStatus
from .normalization import normalize_mahjong_text
from .skills import select_relevant_skills
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
        "cq": "杭麻财敲；财敲是杭麻的一种细分玩法，不是独立大类",
        "default_region": "杭州",
        "default_ambiguous_mahjong": "未明确玩法时默认按杭麻理解；四川门店可改为默认川麻",
        "371": "三缺一",
        "272": "二缺二",
        "173": "一缺三",
        "216": "2-16档，底注2，封顶16",
        "1-32": "1-32档，底注1，封顶32",
        "半块": "0.5档",
        "五毛": "0.5档",
        "不抽": "无烟局",
        "人齐开": "开局时间策略：人够后尽快开，不要求固定开局时间",
        "尽快开": "开局时间策略：能早点开就早点开，时间可以协商",
        "时间可以商量": "开局时间策略：人齐后再协商具体时间，不是缺少时间",
        "通宵": "时长策略：通宵局，不是缺少时长",
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
            "text_normalization": self._text_normalization(message, evidence, redaction_counts),
            "workflow_followup_context": self._workflow_followup_context(message, redaction_counts),
            "conversation_summary": self._conversation_summary(message, conversation_id, redaction_counts),
            "customer_profile_summary": self._customer_profile(message, redaction_counts),
            "game_state_snapshot": self._game_state_snapshot(redaction_counts),
            "room_state_snapshot": self._room_state_snapshot(redaction_counts),
            "tool_policy": self._tool_policy(stage),
            "allowed_tools": [],
            "rag_snippets": self._rag_snippets(),
            "active_skills": self._active_skills(stage, message.text),
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
                    "text_normalization",
                    "workflow_followup_context",
                    "conversation_summary",
                    "customer_profile_summary",
                    "game_state_snapshot",
                    "room_state_snapshot",
                    "builtin_ruleset_aliases",
                    "skill_library",
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

    def _text_normalization(
        self,
        message: Message,
        evidence,
        redaction_counts: dict[str, int],
    ) -> dict[str, Any]:
        source_text = evidence.combined_text or message.text
        result = normalize_mahjong_text(source_text)
        return {
            "raw_text": self._redact_text(result.raw_text, redaction_counts),
            "normalized_text": self._redact_text(result.text, redaction_counts),
            "changed": result.text != result.raw_text,
            "changed_rule_ids": result.changed_rule_ids(),
            "changes": [
                {
                    "rule_id": change.rule_id,
                    "before": self._redact_text(change.before, redaction_counts),
                    "after": self._redact_text(change.after, redaction_counts),
                    "reason": change.reason,
                }
                for change in result.changes[:8]
            ],
            "policy": "这是低风险文本标准化证据，不是业务事实；金额、人数、时间等槽位仍需结合原文、画像和上下文判断。",
        }

    def _workflow_followup_context(
        self,
        message: Message,
        redaction_counts: dict[str, int],
    ) -> dict[str, Any]:
        payload = message.metadata.get("workflow_followup_context")
        if not isinstance(payload, dict):
            return {}
        if not payload:
            return {}
        return {
            **self._redact_value(payload, redaction_counts),
            "policy": "这是上一轮工作流上下文；模型需要判断当前消息是否是在确认、拒绝或补充上一轮建议，后端只校验模型提出的动作。",
        }

    def _redact_value(self, value: Any, redaction_counts: dict[str, int]) -> Any:
        if isinstance(value, str):
            return self._redact_text(value, redaction_counts)
        if isinstance(value, list):
            return [self._redact_value(item, redaction_counts) for item in value[:20]]
        if isinstance(value, dict):
            return {
                str(key): self._redact_value(item, redaction_counts)
                for key, item in value.items()
                if len(str(key)) <= 80
            }
        return value

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
            omitted_messages = candidates[: -self.config.max_recent_messages] if self.config.max_recent_messages else candidates
            recent_messages = [
                self._message_summary(item, redaction_counts)
                for item in candidates[-self.config.max_recent_messages :]
            ]
        compressed_history = self._compressed_message_history(
            omitted_messages if self.config.include_recent_messages else [],
            redaction_counts,
            reason="recent_window_limit",
        )

        return {
            "conversation_ref": self._stable_ref("conversation", conversation_id),
            "channel_type": message.channel_type.value,
            "recent_messages": recent_messages,
            "compressed_history": compressed_history,
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

    def _compressed_message_history(
        self,
        messages: list[Message],
        redaction_counts: dict[str, int],
        *,
        reason: str,
    ) -> dict[str, Any]:
        if not messages:
            return {
                "strategy": "deterministic_lossy_summary",
                "reason": reason,
                "omitted_count": 0,
                "budget_trimmed_count": 0,
                "recent_omitted_messages": [],
                "extracted_hints": {},
                "policy": "原始消息仍在日志/消息表中；上下文只保留压缩摘要用于本轮推理。",
            }
        sorted_messages = sorted(messages, key=lambda item: (item.sent_at, item.id))
        return {
            "strategy": "deterministic_lossy_summary",
            "reason": reason,
            "omitted_count": len(sorted_messages),
            "budget_trimmed_count": 0,
            "time_range": {
                "start": sorted_messages[0].sent_at.isoformat(),
                "end": sorted_messages[-1].sent_at.isoformat(),
            },
            "recent_omitted_messages": [
                {
                    "sent_at": item.sent_at.isoformat(),
                    "sender_ref": self._stable_ref("customer", item.sender_id),
                    "sender_display_name": self._redact_text(item.sender_name, redaction_counts),
                    "text_excerpt": self._redact_text(item.text, redaction_counts)[:120],
                    "intent_hints": extract_intent_evidence(item).lead_reasons[:4],
                }
                for item in sorted_messages[-3:]
            ],
            "extracted_hints": self._history_hints(sorted_messages),
            "policy": "这是有损摘要，只用于帮助模型理解长期上下文；状态推进必须以当前消息、结构化状态和工具结果为准。",
        }

    def _history_hints(self, messages: list[Message]) -> dict[str, Any]:
        text = "\n".join(item.text for item in messages[-20:])
        levels = sorted(set(re.findall(r"(?<!\d)(?:0\.5|0。5|0，5|0,5|1|2|1-32|2-16)(?!\d)", text)))[:8]
        party_terms = sorted(set(re.findall(r"(?:371|三缺一|173|一缺三|272|二缺二|缺[一二三123])", text)))[:8]
        time_terms = sorted(set(re.findall(r"(?:通宵|人齐开|尽快|下班|下午|晚上|[一二两三四五六七八九十\d]{1,2}点(?:半)?)", text)))[:12]
        intent_terms = sorted(set(re.findall(r"(?:有人吗|组一桌|帮我组|摇人|不打了|取消|可以|来不了|打)", text)))[:12]
        return {
            "levels": levels,
            "party_terms": party_terms,
            "time_terms": time_terms,
            "intent_terms": intent_terms,
        }

    def _record_budget_trimmed_message(self, conversation_summary: dict[str, Any], message_summary: dict[str, Any]) -> None:
        compressed = conversation_summary.setdefault(
            "compressed_history",
            {
                "strategy": "deterministic_lossy_summary",
                "reason": "context_budget_trim",
                "omitted_count": 0,
                "budget_trimmed_count": 0,
                "recent_omitted_messages": [],
                "extracted_hints": {},
                "policy": "原始消息仍在日志/消息表中；上下文只保留压缩摘要用于本轮推理。",
            },
        )
        compressed["omitted_count"] = int(compressed.get("omitted_count") or 0) + 1
        compressed["budget_trimmed_count"] = int(compressed.get("budget_trimmed_count") or 0) + 1
        examples = compressed.setdefault("recent_omitted_messages", [])
        examples.append(
            {
                "sent_at": message_summary.get("sent_at"),
                "sender_ref": message_summary.get("sender_ref"),
                "sender_display_name": message_summary.get("sender_display_name"),
                "text_excerpt": str(message_summary.get("text") or "")[:120],
                "intent_hints": [],
            }
        )
        del examples[:-3]

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
        compressed_history = context["conversation_summary"].get("compressed_history") or {}
        open_games = context["game_state_snapshot"].get("recent_open_games") or []
        room_holds = context["room_state_snapshot"].get("active_holds") or []

        while estimated > self.config.max_context_chars:
            if len(recent_messages) > 1:
                self._record_budget_trimmed_message(context["conversation_summary"], recent_messages.pop(0))
                trimmed_sections.append("conversation_summary.recent_messages")
            elif len(open_games) > 1:
                open_games.pop(0)
                trimmed_sections.append("game_state_snapshot.recent_open_games")
            elif len(room_holds) > 0:
                room_holds.pop()
                trimmed_sections.append("room_state_snapshot.active_holds")
            elif compressed_history.get("recent_omitted_messages"):
                compressed_history["recent_omitted_messages"] = []
                trimmed_sections.append("conversation_summary.compressed_history.examples")
            elif compressed_history.get("extracted_hints"):
                compressed_history["extracted_hints"] = {}
                trimmed_sections.append("conversation_summary.compressed_history.hints")
            elif context.get("rag_snippets"):
                context["rag_snippets"] = []
                trimmed_sections.append("rag_snippets")
            elif context.get("active_skills"):
                context["active_skills"] = []
                trimmed_sections.append("active_skills")
            else:
                break
            estimated = self._estimate_chars(context)
        return estimated, sorted(set(trimmed_sections))

    def _active_skills(self, stage: str, text: str) -> list[dict[str, Any]]:
        skill_stage = "semantic_resolution" if stage in {"interpret_message", "semantic_resolution"} else stage
        return select_relevant_skills(stage=skill_stage, text=text, limit=5)

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
