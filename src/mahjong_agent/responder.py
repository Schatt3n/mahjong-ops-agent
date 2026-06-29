from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any

from .context import ContextBuilder, ContextBuildResult
from .core import AgentCore, IngestOutcome, PARTY_SIZE_PROFILE_CONFIDENCE_THRESHOLD
from .llm import LLMResolver
from .models import (
    ChannelType,
    CustomerProfile,
    GameRequest,
    GameStatus,
    Invitation,
    InvitationStatus,
    Message,
)
from .messages import GAME_RULE_LABELS, GAME_TYPE_LABELS, VARIANT_LABELS
from .signals import extract_intent_evidence, message_for_intent


class ReplyAction(StrEnum):
    ASK_CLARIFICATION = "ask_clarification"
    CREATE_PENDING_GAME = "create_pending_game"
    CREATE_GAME = "create_game"
    QUEUE_INVITES = "queue_invites"
    ACCEPT_SEAT = "accept_seat"
    DECLINE_INVITE = "decline_invite"
    CLOSE_GAME = "close_game"
    IGNORE = "ignore"
    HUMAN_REVIEW = "human_review"


@dataclass(slots=True)
class ReplyDecision:
    action: ReplyAction
    reply_text: str
    confidence: float
    should_reply: bool = True
    needs_human_review: bool = False
    game_id: str | None = None
    draft_group_post: str | None = None
    invitation_drafts: list[Invitation] = field(default_factory=list)
    llm_context_digest: str | None = None
    llm_context_snapshot: dict | None = None
    semantic_proposal: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "action": self.action.value,
            "reply_text": self.reply_text,
            "confidence": self.confidence,
            "should_reply": self.should_reply,
            "needs_human_review": self.needs_human_review,
            "game_id": self.game_id,
            "draft_group_post": self.draft_group_post,
            "llm_context_digest": self.llm_context_digest,
            "llm_context_snapshot": self.llm_context_snapshot,
            "semantic_proposal": self.semantic_proposal,
            "invitation_drafts": [
                {
                    "id": invitation.id,
                    "game_id": invitation.game_id,
                    "customer_id": invitation.customer_id,
                    "customer_name": invitation.customer_name,
                    "status": invitation.status.value,
                    "message_text": invitation.message_text,
                }
                for invitation in self.invitation_drafts
            ],
            "notes": self.notes,
        }


class AgentResponder:
    """Decides what the operator agent should say or do for one incoming message."""

    sensitive_pattern = re.compile(
        r"(抽水|赌资|赌博|洗钱|上分|下分|代收|代付|结算输赢|输赢结算|放贷|借码)"
    )
    decline_pattern = re.compile(r"(不来了|来不了|没空|算了|下次|去不了|不方便)")
    full_pattern = re.compile(r"(满了|组好了|凑齐了|齐了|不用找了)")
    cancel_pattern = re.compile(r"(取消|不打了|散了|改天)")
    join_pattern = re.compile(r"(我来|算我|报名|可以来|加我一个|我能来|还有位置吗|还缺人吗|还能来吗)")
    soft_lead_pattern = re.compile(
        r"(有人.*(?:打|玩).*(?:麻将|牌)|(?:打|玩).*(?:麻将|牌).*吗|麻将.*(?:有人|有局|约吗|来吗))"
    )
    llm_hint_pattern = re.compile(
        r"(麻|麻将|牌|局|桌|缺|差|约|搓|打|玩|来|开|人|搭子|雀|杭|川|红中|捉鸡|幺鸡|妖鸡|财敲|cq|\d{3})"
    )

    def __init__(
        self,
        core: AgentCore | None = None,
        invite_limit: int = 5,
        llm_resolver: LLMResolver | None = None,
        context_builder: ContextBuilder | None = None,
        fragment_window_seconds: float = 120.0,
        fragment_max_messages: int = 8,
    ) -> None:
        self.core = core or AgentCore()
        self.invite_limit = invite_limit
        self.llm_resolver = llm_resolver
        self.context_builder = context_builder or ContextBuilder(self.core)
        self.fragment_window_seconds = fragment_window_seconds
        self.fragment_max_messages = fragment_max_messages

    def respond(self, message: Message, now: datetime | None = None) -> ReplyDecision:
        self.core.advance_game_lifecycle(now)
        evidence = extract_intent_evidence(message)
        message = message_for_intent(message, evidence)
        evidence_notes = self._evidence_notes(evidence)
        text = message.text.strip()
        normalized = self._normalize(text)
        self.core.store.messages[message.id] = message

        if self.sensitive_pattern.search(normalized):
            return ReplyDecision(
                action=ReplyAction.HUMAN_REVIEW,
                reply_text="这个我先转人工确认一下。",
                confidence=0.95,
                needs_human_review=True,
                notes=["命中敏感经营/资金相关词，线上自动回复应停止继续处理。", *evidence_notes],
            )

        active_invitation = self._latest_active_invitation(message.sender_id)
        if active_invitation and self.decline_pattern.search(normalized):
            invitation = self.core.decline_invitation(active_invitation.id)
            return ReplyDecision(
                action=ReplyAction.DECLINE_INVITE,
                reply_text="收到，那我先不帮你占这桌，后面有合适的再问你。",
                confidence=0.9,
                game_id=invitation.game_id,
                notes=[f"邀约 {invitation.id} 已标记为 declined。"],
            )

        if active_invitation and self.join_pattern.search(normalized):
            outcome = self.core.accept_invitation(active_invitation.id, now=now)
            if not outcome.accepted:
                return ReplyDecision(
                    action=ReplyAction.DECLINE_INVITE,
                    reply_text=outcome.message_to_customer,
                    confidence=0.9,
                    game_id=outcome.game.id,
                    notes=[
                        "客户已被另一个有效局占用，本邀约已废弃。",
                        f"conflict_game_id={outcome.conflict_game_id}",
                    ],
                )
            return ReplyDecision(
                action=ReplyAction.ACCEPT_SEAT,
                reply_text=outcome.message_to_customer,
                confidence=0.92,
                game_id=outcome.game.id,
                notes=[
                    f"邀约 {active_invitation.id} 已接受。",
                    f"已废弃该客户其他待确认邀约 {len(outcome.cancelled_invitations)} 个。",
                ],
            )

        if self.full_pattern.search(normalized) or self.cancel_pattern.search(normalized):
            game = self._latest_game_by_organizer(message.sender_id)
            if game:
                status = GameStatus.CONFIRMED if self.full_pattern.search(normalized) else GameStatus.CANCELLED
                cancelled = self.core.set_game_status(game.id, status)
                reply = "好的，这桌我标记为已组好。" if status == GameStatus.CONFIRMED else "收到，这桌我标记为取消。"
                return ReplyDecision(
                    action=ReplyAction.CLOSE_GAME,
                    reply_text=reply,
                    confidence=0.88,
                    game_id=game.id,
                    notes=[f"已同步取消 {len(cancelled)} 个待处理邀约。"],
                )

        if self.join_pattern.search(normalized):
            game = self._latest_joinable_game(message)
            if game:
                accepted, cancelled, conflict_game_id = self.core.reserve_customer_for_game(
                    game.id,
                    message.sender_id,
                    now=now,
                )
                if not accepted:
                    if conflict_game_id:
                        return ReplyDecision(
                            action=ReplyAction.DECLINE_INVITE,
                            reply_text="你已经在另一桌有效局里了，我先不重复帮你安排。",
                            confidence=0.86,
                            game_id=game.id,
                            notes=[f"客户已被 {conflict_game_id} 占用。"],
                        )
                    return ReplyDecision(
                        action=ReplyAction.ASK_CLARIFICATION,
                        reply_text="这桌刚刚已经满了，我先给你记个候补，有合适的再问你。",
                        confidence=0.82,
                        game_id=game.id,
                        notes=["客户报名时目标局已满。"],
                    )
                reply = "给你占上了，这桌人数已齐。" if game.is_full else "给你先占上了，我继续确认剩余人数。"
                return ReplyDecision(
                    action=ReplyAction.ACCEPT_SEAT,
                    reply_text=reply,
                    confidence=0.82,
                    game_id=game.id,
                    notes=[
                        "根据同频道唯一可加入局自动占位。",
                        f"已废弃该客户其他待确认邀约 {len(cancelled)} 个。",
                    ],
                )
            return ReplyDecision(
                action=ReplyAction.ASK_CLARIFICATION,
                reply_text="可以，你想加入哪一桌？把时间或发起人告诉我一下。",
                confidence=0.74,
            )

        if not message.metadata.get("workflow_followup_context"):
            fragmented_decision = self._maybe_resolve_fragmented_request(
                message=message,
                normalized=normalized,
                current_evidence=evidence,
                current_evidence_notes=evidence_notes,
                now=now,
            )
            if fragmented_decision:
                return fragmented_decision

        llm_decision = self._maybe_resolve_with_llm(message, normalized, evidence_notes, now)
        if llm_decision:
            return llm_decision

        outcome = self.core.ingest_message(message, now=now)
        extraction = outcome.extraction
        if extraction.game:
            return self._decision_for_game_outcome(
                outcome=outcome,
                message=message,
                normalized=normalized,
                evidence=evidence,
                evidence_notes=evidence_notes,
                now=now,
            )

        if evidence.is_potential_lead:
            self._mark_potential_customer(message, normalized, evidence)
            return ReplyDecision(
                action=ReplyAction.ASK_CLARIFICATION,
                reply_text=self._potential_customer_reply(message, normalized),
                confidence=max(evidence.lead_score, 0.68),
                should_reply=True,
                needs_human_review=False,
                notes=[
                    "已识别为潜在客户/组局意向。",
                    *evidence_notes,
                ],
            )

        if message.channel_type in {ChannelType.WECHAT_GROUP, ChannelType.WEWORK_GROUP}:
            return ReplyDecision(
                action=ReplyAction.IGNORE,
                reply_text="",
                confidence=0.75,
                should_reply=False,
                notes=["群聊无关消息默认静默，避免打扰。"],
            )

        return ReplyDecision(
            action=ReplyAction.IGNORE,
            reply_text="我这边可以帮你处理组局、缺人、改时间和确认人数。你可以直接说：今晚5点 0.5 三缺一 无烟。",
            confidence=0.7,
        )

    def _decision_for_game_outcome(
        self,
        outcome: IngestOutcome,
        message: Message,
        normalized: str,
        evidence,
        evidence_notes: list[str],
        now: datetime | None,
        extra_notes: list[str] | None = None,
    ) -> ReplyDecision:
        extraction = outcome.extraction
        game = extraction.game
        if game is None:
            raise ValueError("game outcome requires extraction.game")
        extra_notes = extra_notes or []

        if outcome.room_conflict_text:
            return ReplyDecision(
                action=ReplyAction.ASK_CLARIFICATION,
                reply_text=outcome.room_conflict_text,
                confidence=max(extraction.confidence, 0.76),
                game_id=game.id,
                notes=[
                    "目标开局时间无可用房间，已暂停邀约并进入时间协商。",
                    *extra_notes,
                    *evidence_notes,
                    *extraction.follow_up_questions,
                ],
            )

        if outcome.clarification_text:
            if self._is_pending_queue_candidate(game):
                return ReplyDecision(
                    action=ReplyAction.CREATE_PENDING_GAME,
                    reply_text=self._pending_queue_reply(game, extraction.follow_up_questions),
                    confidence=max(extraction.confidence, 0.62),
                    game_id=game.id,
                    notes=[
                        "已进入待组局队列，等待补齐关键信息后再匹配/邀约。",
                        *extra_notes,
                        *evidence_notes,
                        *extraction.follow_up_questions,
                    ],
                )
            if self.soft_lead_pattern.search(normalized) or evidence.is_potential_lead:
                self._mark_potential_customer(message, normalized, evidence)
                return ReplyDecision(
                    action=ReplyAction.ASK_CLARIFICATION,
                    reply_text=self._potential_customer_reply(message, normalized),
                    confidence=max(extraction.confidence, evidence.lead_score, 0.72),
                    game_id=game.id,
                    notes=[
                        "已识别为潜在客户/组局意向。",
                        *extra_notes,
                        *evidence_notes,
                        *extraction.follow_up_questions,
                    ],
                )
            return ReplyDecision(
                action=ReplyAction.ASK_CLARIFICATION,
                reply_text=outcome.clarification_text,
                confidence=extraction.confidence,
                game_id=game.id,
                notes=[*extra_notes, *extraction.follow_up_questions],
            )

        invitations = self.core.queue_invitations(
            game.id,
            outcome.candidates,
            limit=self.invite_limit,
            now=now,
        )
        summary = self._game_summary(game)
        if invitations:
            names = "、".join(invitation.customer_name for invitation in invitations[:5])
            reply = f"可以，我先记录：{summary}。建议先私聊 {names}，群发草稿也已经生成。"
            action = ReplyAction.QUEUE_INVITES
            needs_human_review = True
            notes = ["邀约草稿已生成，建议人工确认后再发出。", *extra_notes, *evidence_notes]
        else:
            reply = f"可以，我先记录：{summary}。目前客户画像里没有高匹配候选人，建议先发群。"
            action = ReplyAction.CREATE_GAME
            needs_human_review = outcome.draft_group_post is not None
            notes = ["没有生成私聊邀约草稿。", *extra_notes, *evidence_notes]

        return ReplyDecision(
            action=action,
            reply_text=reply,
            confidence=extraction.confidence,
            needs_human_review=needs_human_review,
            game_id=game.id,
            draft_group_post=outcome.draft_group_post,
            invitation_drafts=invitations,
            notes=notes,
        )

    def _maybe_resolve_with_llm(
        self,
        message: Message,
        normalized: str,
        evidence_notes: list[str],
        now: datetime | None,
    ) -> ReplyDecision | None:
        if self.llm_resolver is None or not self._should_consult_llm(message, normalized):
            return None

        context_result = self.context_builder.build(
            message,
            now=now,
            goal="interpret_mahjong_operator_message",
            stage="semantic_resolution",
        )
        resolution = self.llm_resolver.resolve(message, context=context_result.context)
        llm_notes = [
            *context_result.notes,
            *resolution.notes,
            f"LLM intent={resolution.intent}, confidence={resolution.confidence}",
            f"LLM proposed_action={resolution.proposed_action}",
            *evidence_notes,
        ]
        if resolution.facts:
            llm_notes.append(f"LLM facts={resolution.facts}")
        if resolution.slots:
            llm_notes.append(f"LLM slots={resolution.slots}")

        if resolution.needs_human_review:
            if self._llm_resolution_is_infra_failure(resolution):
                return None
            return self._with_llm_context(
                ReplyDecision(
                    action=ReplyAction.HUMAN_REVIEW,
                    reply_text="这个我先转人工确认一下。",
                    confidence=max(resolution.confidence, 0.72),
                    needs_human_review=True,
                    notes=llm_notes,
                ),
                context_result,
                semantic_proposal=self._semantic_proposal_from_resolution(resolution),
            )

        slot_normalized_text = self._normalized_text_from_llm_slots(
            resolution.slots,
            original_text=message.text,
            notes=llm_notes,
        )
        normalized_candidate = resolution.normalized_text or slot_normalized_text

        if (
            normalized_candidate
            and resolution.confidence >= 0.55
            and resolution.intent in {"find_players", "update_game", "uncertain"}
        ):
            normalized_text = self._guard_llm_normalized_text(
                original_text=message.text,
                normalized_text=normalized_candidate,
                notes=llm_notes,
            )
            llm_message = Message(
                text=normalized_text,
                sender_id=message.sender_id,
                sender_name=message.sender_name,
                channel_id=message.channel_id,
                channel_type=message.channel_type,
                sent_at=message.sent_at,
                id=message.id,
                metadata={
                    **message.metadata,
                    "llm_original_text": message.text,
                    "llm_normalized_text": resolution.normalized_text,
                    "llm_slot_normalized_text": slot_normalized_text,
                    "llm_guarded_normalized_text": normalized_text,
                    "llm_intent": resolution.intent,
                    "llm_confidence": resolution.confidence,
                    "llm_slots": resolution.slots,
                },
            )
            outcome = self.core.ingest_message(llm_message, now=now)
            if outcome.extraction.game:
                return self._with_llm_context(
                    self._decision_for_game_outcome(
                        outcome=outcome,
                        message=llm_message,
                        normalized=self._normalize(llm_message.text),
                        evidence=type("LLMEvidence", (), {"is_potential_lead": False, "lead_score": 0.0})(),
                        evidence_notes=[],
                        now=now,
                        extra_notes=llm_notes,
                    ),
                    context_result,
                    semantic_proposal=self._semantic_proposal_from_resolution(resolution),
                )

        if resolution.is_mahjong_related and resolution.confidence >= 0.45:
            self._mark_potential_customer(message, normalized)
            return self._with_llm_context(
                ReplyDecision(
                    action=ReplyAction.ASK_CLARIFICATION,
                    reply_text=resolution.reply_text or self._potential_customer_reply(message, normalized),
                    confidence=max(resolution.confidence, 0.62),
                    needs_human_review=False,
                    notes=["LLM 判断为麻将相关但信息不足。", *llm_notes],
                ),
                context_result,
                semantic_proposal=self._semantic_proposal_from_resolution(resolution),
            )

        if message.channel_type in {ChannelType.WECHAT_GROUP, ChannelType.WEWORK_GROUP}:
            return self._with_llm_context(
                ReplyDecision(
                    action=ReplyAction.IGNORE,
                    reply_text="",
                    confidence=max(resolution.confidence, 0.75),
                    should_reply=False,
                    notes=["LLM 判断为无关或低置信度，群聊静默。", *llm_notes],
                ),
                context_result,
                semantic_proposal=self._semantic_proposal_from_resolution(resolution),
            )
        return None

    def _normalized_text_from_llm_slots(
        self,
        slots: dict[str, Any],
        *,
        original_text: str,
        notes: list[str],
    ) -> str | None:
        if not slots:
            return None
        parts: list[str] = []
        query_mode = self._slot_value(slots.get("query_mode"))
        if query_mode in {"search_existing", "join_existing"} and self._slot_is_usable(slots.get("query_mode"), min_confidence=0.65):
            parts.append("现在有没有现成局 人齐开")
        elif query_mode == "create_new" and self._slot_is_usable(slots.get("query_mode"), min_confidence=0.65):
            parts.append("帮我组一桌")

        game_type = self._slot_value(slots.get("game_type"))
        if isinstance(game_type, str) and game_type not in {"", "unknown"} and self._slot_is_usable(slots.get("game_type"), min_confidence=0.7):
            label = GAME_TYPE_LABELS.get(game_type, game_type)
            if label != "麻将":
                parts.append(label)

        variant = self._slot_value(slots.get("variant"))
        if isinstance(variant, str) and variant not in {"", "unknown"} and self._slot_is_usable(slots.get("variant"), min_confidence=0.75):
            parts.append(VARIANT_LABELS.get(variant, variant))

        level_options = self._slot_value(slots.get("level_options"))
        if isinstance(level_options, list) and self._slot_is_usable(slots.get("level_options"), min_confidence=0.75):
            clean_levels = [str(item).strip() for item in level_options if str(item).strip() and str(item).strip() != "unknown"]
            if clean_levels:
                parts.append("或者".join(clean_levels) + "都行" if len(clean_levels) > 1 else clean_levels[0])
        else:
            level = self._slot_value(slots.get("level"))
            if isinstance(level, str) and level not in {"", "unknown"} and self._slot_is_usable(slots.get("level"), min_confidence=0.75):
                parts.append(level)

        start_time_slot = slots.get("start_time")
        start_time = self._slot_value(start_time_slot)
        start_time_mode_slot = slots.get("start_time_mode")
        start_time_mode = self._slot_value(start_time_mode_slot)
        if (
            isinstance(start_time, str)
            and re.fullmatch(r"\d{1,2}:\d{2}", start_time)
            and self._slot_is_usable(start_time_slot, min_confidence=0.75)
            and not self._slot_needs_confirmation(start_time_slot)
        ):
            parts.append(start_time)
        elif (
            start_time_mode == "people_ready"
            and self._slot_is_usable(start_time_mode_slot, min_confidence=0.7)
        ):
            parts.append("人齐开")
        elif self._slot_needs_confirmation(start_time_slot):
            notes.append("LLM start_time 槽位需要确认，未作为确定时间写入解析文本。")

        duration_slot = slots.get("duration_hours")
        duration = self._slot_value(duration_slot)
        duration_mode_slot = slots.get("duration_mode")
        duration_mode = self._slot_value(duration_mode_slot)
        if isinstance(duration, (int, float)) and duration > 0 and self._slot_is_usable(duration_slot, min_confidence=0.75):
            value = int(duration) if float(duration).is_integer() else duration
            parts.append(f"{value}小时")
        elif duration_mode == "overnight" and self._slot_is_usable(duration_mode_slot, min_confidence=0.7):
            parts.append("通宵")

        if self._has_explicit_party_size(self._normalize(original_text)):
            missing_count = self._slot_value(slots.get("missing_count"))
            known_players = self._slot_value(slots.get("known_players"))
            if isinstance(missing_count, int) and 0 <= missing_count <= 3 and self._slot_is_usable(slots.get("missing_count"), min_confidence=0.75, explicit_only=True):
                parts.append(f"缺{missing_count}人")
            elif isinstance(known_players, int) and 1 <= known_players <= 4 and self._slot_is_usable(slots.get("known_players"), min_confidence=0.75, explicit_only=True):
                parts.append(f"我这边{known_players}个人")
        elif slots.get("missing_count") or slots.get("known_players"):
            notes.append("LLM 人数槽位不是原文明确人数，未作为确定人数写入解析文本。")

        smoke = self._slot_value(slots.get("smoke"))
        if isinstance(smoke, str) and self._slot_is_usable(slots.get("smoke"), min_confidence=0.75):
            smoke_labels = {
                "no_smoke": "无烟",
                "smoke_ok": "可吸烟",
                "any": "烟都可",
            }
            if smoke in smoke_labels and self._slot_source(slots.get("smoke")) in {"explicit", "inferred"}:
                parts.append(smoke_labels[smoke])

        text = " ".join(part for part in parts if part)
        return text or None

    def _slot_value(self, slot: Any) -> Any:
        if isinstance(slot, dict):
            return slot.get("value")
        return slot

    def _slot_confidence(self, slot: Any) -> float:
        if not isinstance(slot, dict):
            return 0.0
        try:
            return float(slot.get("confidence") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _slot_source(self, slot: Any) -> str:
        if not isinstance(slot, dict):
            return ""
        return str(slot.get("source") or "").strip().lower()

    def _slot_needs_confirmation(self, slot: Any) -> bool:
        return isinstance(slot, dict) and bool(slot.get("needs_confirmation"))

    def _slot_is_usable(self, slot: Any, *, min_confidence: float, explicit_only: bool = False) -> bool:
        if not isinstance(slot, dict):
            return False
        if self._slot_confidence(slot) < min_confidence:
            return False
        if self._slot_needs_confirmation(slot):
            return False
        source = self._slot_source(slot)
        if explicit_only and source != "explicit":
            return False
        return source not in {"", "unknown"}

    def _guard_llm_normalized_text(
        self,
        *,
        original_text: str,
        normalized_text: str,
        notes: list[str],
    ) -> str:
        if self._has_explicit_party_size(self._normalize(original_text)):
            return normalized_text
        guarded = re.sub(
            r"(?:371|3\s*缺\s*1|三\s*缺\s*一|3\s*差\s*1|三\s*差\s*一|3\s*等\s*1|三\s*等\s*一|"
            r"173|1\s*缺\s*3|一\s*缺\s*三|1\s*差\s*3|一\s*差\s*三|1\s*等\s*3|一\s*等\s*三|"
            r"272|2\s*缺\s*2|二\s*缺\s*二|两\s*缺\s*两|2\s*差\s*2|二\s*差\s*二|2\s*等\s*2|二\s*等\s*二|"
            r"缺\s*[一二两三123]\s*(?:个|位|人)?)",
            "",
            normalized_text,
        )
        guarded = re.sub(r"[，,]\s*[，,]+", "，", guarded)
        guarded = re.sub(r"\s{2,}", " ", guarded).strip(" ，,。")
        if guarded != normalized_text:
            notes.append("已移除 LLM 归一化中新增的人数/缺口信息，原文未明确人数。")
        return guarded or normalized_text

    def _llm_resolution_is_infra_failure(self, resolution) -> bool:
        joined_notes = " ".join(str(note) for note in resolution.notes)
        return any(
            marker in joined_notes
            for marker in [
                "LLM 调用失败",
                "LLM 预算不足",
                "LLM 未返回可解析 JSON",
                "LLM 返回的 JSON 片段无法解析",
            ]
        )

    def _semantic_proposal_from_resolution(self, resolution) -> dict[str, Any]:
        reasoning_summary = ""
        if isinstance(resolution.facts, dict):
            reasoning_summary = str(resolution.facts.get("reasoning_summary") or resolution.facts.get("reason") or "")
        return {
            "source": "llm",
            "intent": resolution.intent,
            "proposed_action": resolution.proposed_action,
            "confidence": resolution.confidence,
            "needs_human_review": resolution.needs_human_review,
            "reasoning_summary": reasoning_summary[:240],
            "slots": resolution.slots,
            "facts": resolution.facts,
            "budget": resolution.budget,
        }

    def _with_llm_context(
        self,
        decision: ReplyDecision,
        context_result: ContextBuildResult,
        semantic_proposal: dict[str, Any] | None = None,
    ) -> ReplyDecision:
        decision.llm_context_digest = context_result.context_digest
        decision.llm_context_snapshot = context_result.context
        decision.semantic_proposal = semantic_proposal
        return decision

    def _maybe_resolve_fragmented_request(
        self,
        message: Message,
        normalized: str,
        current_evidence,
        current_evidence_notes: list[str],
        now: datetime | None,
    ) -> ReplyDecision | None:
        combined_text = self._combined_recent_sender_text(message, now)
        if not combined_text or self._normalize(combined_text) == normalized:
            return None

        combined_message = Message(
            text=combined_text,
            sender_id=message.sender_id,
            sender_name=message.sender_name,
            channel_id=message.channel_id,
            channel_type=message.channel_type,
            sent_at=message.sent_at,
            id=message.id,
            metadata={
                **message.metadata,
                "fragment_combined_text": combined_text,
                "fragment_window_seconds": self.fragment_window_seconds,
            },
        )
        combined_evidence = extract_intent_evidence(combined_message)
        combined_notes = self._evidence_notes(combined_evidence)
        combined_extraction = self.core.parser.parse(combined_message, now=now)
        if combined_extraction.game:
            if self._should_reply_as_fragmented_lead(combined_extraction.game, combined_evidence):
                self._mark_potential_customer(combined_message, self._normalize(combined_text), combined_evidence)
                return ReplyDecision(
                    action=ReplyAction.ASK_CLARIFICATION,
                    reply_text=self._fragmented_potential_reply(combined_text, message),
                    confidence=max(combined_extraction.confidence, combined_evidence.lead_score, 0.7),
                    should_reply=True,
                    needs_human_review=False,
                    game_id=combined_extraction.game.id,
                    notes=[
                        "已合并同一用户短时间内的多条碎片消息。",
                        "已识别为潜在客户/组局意向。",
                        *combined_notes,
                        *current_evidence_notes,
                        *combined_extraction.follow_up_questions,
                    ],
                )
            combined_outcome = self.core.ingest_message(combined_message, now=now)
            return self._decision_for_game_outcome(
                outcome=combined_outcome,
                message=combined_message,
                normalized=self._normalize(combined_text),
                evidence=combined_evidence,
                evidence_notes=combined_notes,
                now=now,
                extra_notes=["已合并同一用户短时间内的多条碎片消息。"],
            )

        if combined_evidence.is_potential_lead and combined_evidence.lead_score >= current_evidence.lead_score:
            self._mark_potential_customer(combined_message, self._normalize(combined_text), combined_evidence)
            return ReplyDecision(
                action=ReplyAction.ASK_CLARIFICATION,
                reply_text=self._fragmented_potential_reply(combined_text, message),
                confidence=max(combined_evidence.lead_score, 0.7),
                should_reply=True,
                needs_human_review=False,
                notes=[
                    "已合并同一用户短时间内的多条碎片消息。",
                    "已识别为潜在客户/组局意向。",
                    *combined_notes,
                    *current_evidence_notes,
                ],
            )
        return None

    def _combined_recent_sender_text(self, message: Message, now: datetime | None) -> str | None:
        if self.fragment_window_seconds <= 0:
            return None
        reference_time = message.sent_at
        if now is not None and abs((now - message.sent_at).total_seconds()) < self.fragment_window_seconds * 10:
            reference_time = now
        cutoff = reference_time - timedelta(seconds=self.fragment_window_seconds)
        candidates: list[Message] = []
        for item in self.core.store.messages.values():
            if item.sender_id != message.sender_id:
                continue
            if item.channel_id != message.channel_id:
                continue
            if item.channel_type != message.channel_type:
                continue
            if item.sent_at < cutoff or item.sent_at > reference_time + timedelta(seconds=1):
                continue
            if not item.text.strip():
                continue
            candidates.append(item)

        if not any(item.id == message.id for item in candidates):
            candidates.append(message)
        candidates = sorted(candidates, key=lambda item: item.sent_at)[-self.fragment_max_messages :]
        parts: list[str] = []
        for item in candidates:
            text = item.text.strip()
            if text and (not parts or parts[-1] != text):
                parts.append(text)
        if len(parts) < 2:
            return None
        return " ".join(parts)

    def _should_consult_llm(self, message: Message, normalized: str) -> bool:
        if message.metadata.get("disable_llm"):
            return False
        if message.channel_type in {ChannelType.WECHAT_PRIVATE, ChannelType.WEWORK_PRIVATE, ChannelType.MANUAL}:
            return True
        return bool(self.llm_hint_pattern.search(normalized))

    def _llm_context(self, message: Message) -> dict:
        return self.context_builder.build(message).context

    def _latest_active_invitation(self, customer_id: str) -> Invitation | None:
        active = [
            invitation
            for invitation in self.core.store.invitations.values()
            if invitation.customer_id == customer_id
            and invitation.status in {InvitationStatus.QUEUED, InvitationStatus.SENT}
        ]
        return max(active, key=lambda item: item.created_at, default=None)

    def _latest_joinable_game(self, message: Message) -> GameRequest | None:
        games = [
            game
            for game in self.core.store.games.values()
            if game.channel_id == message.channel_id
            and game.organizer_id != message.sender_id
            and game.open_slots is not None
            and game.open_slots > 0
            and game.status in {GameStatus.OPEN, GameStatus.NEGOTIATING, GameStatus.HOLDING}
        ]
        return max(games, key=lambda item: item.created_at, default=None)

    def _latest_game_by_organizer(self, organizer_id: str) -> GameRequest | None:
        games = [
            game
            for game in self.core.store.games.values()
            if game.organizer_id == organizer_id
            and game.status
            in {GameStatus.OPEN, GameStatus.NEED_CLARIFICATION, GameStatus.NEGOTIATING, GameStatus.HOLDING}
        ]
        return max(games, key=lambda item: item.created_at, default=None)

    def _game_summary(self, game: GameRequest) -> str:
        parts: list[str] = []
        if game.start_at:
            parts.append(game.start_at.strftime("%m-%d %H:%M"))
        parts.extend(self._game_labels(game))
        if game.level:
            parts.append(self._stake_label(game))
        if game.current_player_count is not None and game.missing_count is not None:
            parts.append(f"{game.current_player_count}缺{game.missing_count}")
        elif game.missing_count is not None:
            parts.append(f"缺{game.missing_count}位")
        play_options = self._visible_play_options(game)
        if play_options:
            parts.append("、".join(play_options))
        if game.duration_hours:
            hours = int(game.duration_hours) if game.duration_hours.is_integer() else game.duration_hours
            parts.append(f"预计{hours}小时")
        rules = self._visible_rules(game)
        if rules:
            parts.append("、".join(rules))
        return "，".join(parts) if parts else "一桌待确认的局"

    def _is_pending_queue_candidate(self, game: GameRequest) -> bool:
        if game.start_at is not None:
            return False
        if game.current_player_count is None or game.missing_count is None:
            return False
        if self._has_profile_inferred_party_size(game) and game.start_at is None:
            return False
        if self._has_only_regional_default_play_type(game) and not game.level:
            return False
        return bool(game.level or game.game_type != "mahjong" or game.rules)

    def _should_reply_as_fragmented_lead(self, game: GameRequest, evidence) -> bool:
        if not evidence.is_potential_lead:
            return False
        if game.current_player_count is not None or game.missing_count is not None:
            return False
        return any("按当前地区默认玩法" in note for note in game.notes)

    def _has_profile_inferred_party_size(self, game: GameRequest) -> bool:
        return any("根据客户画像推断" in note for note in game.notes)

    def _has_only_regional_default_play_type(self, game: GameRequest) -> bool:
        if not any("按当前地区默认玩法" in note for note in game.notes):
            return False
        visible_rules = self._visible_rules(game)
        return not game.level and not game.variant and not game.play_options and not visible_rules

    def _pending_queue_reply(self, game: GameRequest, questions: list[str]) -> str:
        summary = self._game_summary(game)
        question_text = " ".join(questions) if questions else "我再确认一下开局时间。"
        return f"收到，已先放入待组局队列：{summary}。{question_text}"

    def _game_type_label(self, game: GameRequest) -> str | None:
        if game.game_type == "mahjong":
            return None
        return GAME_TYPE_LABELS.get(game.game_type, game.game_type)

    def _variant_label(self, game: GameRequest) -> str | None:
        if not game.variant:
            return None
        return VARIANT_LABELS.get(game.variant, game.variant)

    def _game_labels(self, game: GameRequest) -> list[str]:
        labels = []
        game_type = self._game_type_label(game)
        variant = self._variant_label(game)
        if game_type:
            labels.append(game_type)
        if variant and variant not in labels:
            labels.append(variant)
        return labels

    def _stake_label(self, game: GameRequest) -> str:
        if game.base_score is not None and game.cap_score is not None:
            return f"{game.level}档(底注{game.base_score:g}/封顶{game.cap_score:g})"
        return f"{game.level}档"

    def _visible_rules(self, game: GameRequest) -> list[str]:
        hidden = set(self._game_labels(game)) | set(game.play_options)
        if game.game_type in GAME_RULE_LABELS:
            hidden.add(GAME_RULE_LABELS[game.game_type])
        return [rule for rule in game.rules if rule not in hidden]

    def _visible_play_options(self, game: GameRequest) -> list[str]:
        variant = self._variant_label(game)
        if not variant:
            return game.play_options
        return [option for option in game.play_options if option != variant]

    def _normalize(self, text: str) -> str:
        return text.strip().replace("，", ",").replace("。", ".").lower()

    def _mark_potential_customer(self, message: Message, normalized_text: str, evidence=None) -> None:
        customer = self.core.store.customers.get(message.sender_id)
        if customer is None:
            customer = CustomerProfile(
                id=message.sender_id,
                display_name=message.sender_name,
                tags=[],
                metadata={},
            )
            self.core.upsert_customer(customer)

        for tag in ["潜在客户", "组局意向"]:
            if tag not in customer.tags:
                customer.tags.append(tag)
        customer.metadata["last_lead_text"] = message.text
        customer.metadata["last_lead_channel_id"] = message.channel_id
        customer.metadata["lead_signal"] = "soft_mahjong_inquiry"
        if evidence is not None:
            customer.metadata["last_lead_modalities"] = getattr(evidence, "modalities", [])
            customer.metadata["last_lead_evidence"] = getattr(evidence, "evidence", [])
            customer.metadata["last_lead_score"] = getattr(evidence, "lead_score", 0.0)
        if "下班" in normalized_text and "下班后活跃" not in customer.tags:
            customer.tags.append("下班后活跃")

    def _potential_customer_reply(self, message: Message | None = None, normalized_text: str = "") -> str:
        normalized = normalized_text or (self._normalize(message.text) if message else "")
        known_party = self._known_party_size(message.sender_id) if message else None
        has_party = self._has_explicit_party_size(normalized) or known_party is not None

        if known_party:
            prefix = f"可以的，我先按{self._party_size_label(known_party[0])}来记。"
        elif "下班" in normalized:
            prefix = "可以的，你今天下班后想打吗？"
        else:
            prefix = "可以的，我先帮你看看。"

        questions: list[str] = []
        if not self._has_specific_clock_time(normalized):
            questions.append("大概几点到")
        if not has_party:
            questions.append("这边现在几个人")
        if not self._fragment_level_labels(normalized):
            questions.append("想打多大的")

        correction = "如果今天人数不一样你跟我说下，" if known_party else ""
        if questions:
            return prefix + "你" + "、".join(questions) + f"？{correction}我这边帮你看看能不能拼一桌。"
        return prefix + correction + "我这边帮你看看能不能拼一桌。"

    def _fragmented_potential_reply(self, combined_text: str, message: Message | None = None) -> str:
        normalized = self._normalize(combined_text)
        known_party = self._known_party_size(message.sender_id) if message else None
        explicit_party = self._has_explicit_party_size(normalized)
        facts: list[str] = []
        if "今天下午" in normalized:
            facts.append("今天下午")
        elif "下午" in normalized:
            facts.append("下午")
        elif "今晚" in normalized or "晚上" in normalized:
            facts.append("晚上")
        levels = self._fragment_level_labels(normalized)
        if levels:
            facts.append("或".join(levels) + "都可以")
        if re.search(r"(烟也都可|烟都可|烟都行|烟都可以|有烟无烟都可|有烟无烟都行)", normalized):
            facts.append("烟况不限")
        elif "无烟" in normalized or "不抽烟" in normalized or "不抽" in normalized:
            facts.append("无烟")
        elif "有烟" in normalized or "可抽烟" in normalized:
            facts.append("可吸烟")

        prefix_parts = [
            f"可以，我先按{','.join(facts)}来记。" if facts else "可以，我先帮你看看。"
        ]
        if known_party and not explicit_party:
            prefix_parts.append(f"你这边我先按{self._party_size_label(known_party[0])}。")
        prefix = " ".join(prefix_parts)
        questions: list[str] = []
        if not explicit_party and not known_party:
            questions.append("你这边现在几个人")
        if not self._has_specific_clock_time(normalized):
            questions.append("大概几点能到")
        if not levels:
            questions.append("想打多大的")
        correction = "如果今天人数不一样你跟我说下，" if known_party and not explicit_party else ""
        if questions:
            return prefix + " " + "，".join(questions) + f"？{correction}我帮你看看能不能拼一桌。"
        return prefix + " " + correction + "我帮你看看能不能拼一桌。"

    def _known_party_size(self, sender_id: str) -> tuple[int, float] | None:
        customer = self.core.store.customers.get(sender_id)
        if customer is None:
            return None

        party_size = customer.usual_party_size
        confidence = customer.usual_party_size_confidence
        if party_size is None:
            raw_party_size = customer.metadata.get("usual_party_size") or customer.metadata.get("default_party_size")
            if raw_party_size is None:
                return None
            try:
                party_size = int(raw_party_size)
            except (TypeError, ValueError):
                return None
            raw_confidence = customer.metadata.get("usual_party_size_confidence") or customer.metadata.get(
                "party_size_confidence"
            )
            try:
                confidence = float(raw_confidence) if raw_confidence is not None else confidence
            except (TypeError, ValueError):
                return None

        if confidence < PARTY_SIZE_PROFILE_CONFIDENCE_THRESHOLD:
            return None
        if party_size < 1 or party_size > 4:
            return None
        return party_size, confidence

    def _party_size_label(self, party_size: int) -> str:
        if party_size == 1:
            return "你一个人"
        return f"你们{party_size}个人"

    def _has_explicit_party_size(self, normalized: str) -> bool:
        return bool(
            re.search(
                r"(371|三\s*(?:缺|差|等)\s*一|3\s*(?:缺|差|等)\s*1|272|二\s*(?:缺|差|等)\s*二|两\s*(?:缺|差|等)\s*两|2\s*(?:缺|差|等)\s*2|"
                r"173|一\s*(?:缺|差|等)\s*三|1\s*(?:缺|差|等)\s*3|缺\s*[一二两三123])",
                normalized,
            )
            or re.search(r"(?<!\d)(1|一)\s*(?:个|位)?\s*人(?!\d)", normalized)
            or re.search(r"(?<!\d)(2|二|两|俩)\s*(?:个|位)?\s*人(?!\d)", normalized)
            or re.search(r"(?<!\d)(3|三)\s*(?:个|位)?\s*人(?!\d)", normalized)
            or re.search(r"(?<!\d)(4|四)\s*(?:个|位)?\s*人(?!\d)", normalized)
        )

    def _fragment_level_labels(self, normalized: str) -> list[str]:
        levels: list[str] = []
        if re.search(r"(0\.5|半块|半|五毛)", normalized):
            levels.append("0.5")
        if re.search(r"(?<!\d)(1|一)(?:块|档)?(?!\d)", normalized):
            levels.append("1")
        if re.search(r"(?<!\d)(2|二|两)(?:块|档)?(?!\d)", normalized):
            levels.append("2")
        return levels

    def _has_specific_clock_time(self, normalized: str) -> bool:
        return bool(
            re.search(r"(?<!\d)\d{1,2}\s*点(?:半|\d{1,2}分?)?(?!\d)", normalized)
            or re.search(r"(?<!\d)\d{1,2}\s*[:：]\s*\d{2}(?!\d)", normalized)
            or re.search(r"(?<!\d)\d{1,2}\.\d{2}(?!\d)", normalized)
        )

    def _evidence_notes(self, evidence) -> list[str]:
        if not evidence.evidence:
            return []
        if not evidence.lead_reasons and evidence.modalities == ["text"]:
            return []
        modalities = "、".join(evidence.modalities) if evidence.modalities else "unknown"
        return [
            f"意图证据: {modalities}，lead_score={evidence.lead_score}",
            *evidence.lead_reasons,
        ]
