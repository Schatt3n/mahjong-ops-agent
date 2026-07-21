from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .llm import AgentLLMClient
from .models import ConversationCheckpoint, ConversationTurn, StateTransition
from .domains import game_for_model_context, invite_draft_for_model_context, outbound_message_draft_for_model_context
from .token_estimation import estimate_tokens as shared_estimate_tokens


DEFAULT_CONTEXT_SUMMARY_PROMPT_PATH = Path(__file__).with_name("prompts").joinpath("context_summary.md")


@dataclass(slots=True)
class ContextSummaryPolicy:
    min_turns_before_summary: int = 12
    min_turns_since_last_summary: int = 6
    max_recent_tokens_before_summary: int = 3_000
    max_turns_considered: int = 80
    max_summary_input_tokens: int = 6_000
    max_summary_chars: int = 800
    max_open_questions: int = 10
    min_confidence: float = 0.6
    timeout_seconds: float = 30.0


@dataclass(slots=True)
class ContextSummaryDecision:
    should_summarize: bool
    reason: str
    turn_count: int
    turns_since_last_summary: int
    estimated_recent_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "should_summarize": self.should_summarize,
            "reason": self.reason,
            "turn_count": self.turn_count,
            "turns_since_last_summary": self.turns_since_last_summary,
            "estimated_recent_tokens": self.estimated_recent_tokens,
        }


@dataclass(slots=True)
class ContextSummaryResult:
    summarized: bool
    reason: str
    checkpoint: ConversationCheckpoint | None = None
    transition: StateTransition | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "summarized": self.summarized,
            "reason": self.reason,
            "checkpoint": self.checkpoint.to_dict() if self.checkpoint else None,
            "transition": self.transition.to_dict() if self.transition else None,
        }


@dataclass(slots=True)
class ContextSummaryManager:
    store: Any
    llm_client: AgentLLMClient
    trace_recorder: Any | None = None
    policy: ContextSummaryPolicy = field(default_factory=ContextSummaryPolicy)
    prompt_path: Path = DEFAULT_CONTEXT_SUMMARY_PROMPT_PATH

    def maybe_summarize_after_turn(self, *, conversation_id: str, trace_id: str) -> ContextSummaryResult:
        """在一轮处理结束后按常规阈值尝试生成 checkpoint。"""

        decision = self.should_summarize(conversation_id)
        self._record(trace_id, "context_summary_checked", decision.to_dict())
        if not decision.should_summarize:
            return ContextSummaryResult(False, decision.reason)

        return self._summarize(conversation_id=conversation_id, trace_id=trace_id, trigger="after_turn")

    def summarize_for_context_budget(
        self,
        *,
        conversation_id: str,
        trace_id: str,
        estimated_context_tokens: int,
        max_context_tokens: int,
        trigger_threshold_tokens: int,
    ) -> ContextSummaryResult:
        """在主模型调用前，因为上下文接近或超过预算而强制尝试摘要。

        这个入口绕过 should_summarize 的轮数阈值，因为预算压力比“够不够轮数”优先级更高。
        如果摘要失败，调用方仍会回到原预算检查并按原有安全策略处理。
        """

        self._record(
            trace_id,
            "context_summary_budget_triggered",
            {
                "conversation_id": conversation_id,
                "estimated_context_tokens": estimated_context_tokens,
                "max_context_tokens": max_context_tokens,
                "trigger_threshold_tokens": trigger_threshold_tokens,
            },
        )
        return self._summarize(conversation_id=conversation_id, trace_id=trace_id, trigger="context_budget")

    def summarize_for_quality_evaluation(
        self,
        *,
        conversation_id: str,
        trace_id: str,
    ) -> ContextSummaryResult:
        """Force one checkpoint for an offline decision-consistency evaluation."""

        self._record(
            trace_id,
            "context_summary_quality_evaluation_triggered",
            {"conversation_id": conversation_id},
        )
        return self._summarize(
            conversation_id=conversation_id,
            trace_id=trace_id,
            trigger="quality_evaluation",
        )

    def _summarize(self, *, conversation_id: str, trace_id: str, trigger: str) -> ContextSummaryResult:
        """执行一次摘要模型调用并保存 checkpoint。"""

        payload = self._build_summary_payload(conversation_id)
        messages = [
            {"role": "system", "content": self.prompt_path.read_text(encoding="utf-8")},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
        ]
        estimated = estimate_tokens(messages)
        self._record(
            trace_id,
            "context_summary_prompt",
            {
                "messages": messages,
                "estimated_tokens": estimated,
                "included_turn_count": len(payload.get("recent_conversation") or []),
                "omitted_turn_count": payload.get("summary_input_budget", {}).get("omitted_turn_count", 0),
                "trigger": trigger,
            },
        )
        if estimated > self.policy.max_summary_input_tokens:
            self._record(
                trace_id,
                "context_summary_skipped",
                {
                    "reason": "summary prompt token estimate exceeded",
                    "estimated_tokens": estimated,
                    "max_summary_input_tokens": self.policy.max_summary_input_tokens,
                },
                level="WARN",
            )
            return ContextSummaryResult(False, "summary prompt token estimate exceeded")

        raw_response = self.llm_client.complete(messages, trace_id=trace_id, timeout_seconds=self.policy.timeout_seconds)
        self._record(trace_id, "context_summary_response", {"content": raw_response})
        summary, errors = parse_context_summary(raw_response)
        if errors:
            self._record(trace_id, "context_summary_contract_error", {"errors": errors, "raw_response": raw_response}, level="WARN")
            return ContextSummaryResult(False, "context summary contract invalid: " + "; ".join(errors))

        confidence = float(summary.get("confidence") or 0.0)
        if confidence < self.policy.min_confidence:
            self._record(
                trace_id,
                "context_summary_rejected",
                {"reason": "confidence below threshold", "confidence": confidence, "min_confidence": self.policy.min_confidence},
                level="WARN",
            )
            return ContextSummaryResult(False, "confidence below threshold")

        facts = dict(summary.get("facts") or {})
        validation_error = self._validate_summary_facts(facts)
        if validation_error:
            self._record(trace_id, "context_summary_rejected", {"reason": validation_error, "facts": facts}, level="WARN")
            return ContextSummaryResult(False, validation_error)

        checkpoint, transition = self.store.upsert_conversation_checkpoint(
            conversation_id=conversation_id,
            summary=clamp_text(str(summary.get("summary") or ""), self.policy.max_summary_chars),
            facts=facts,
            open_questions=[str(item)[:200] for item in summary.get("open_questions") or []][: self.policy.max_open_questions],
            trace_id=trace_id,
        )
        self._record(trace_id, "context_summary_saved", {"checkpoint": checkpoint.to_dict(), "transition": transition.to_dict()})
        self._record(trace_id, "state_transition", transition.to_dict())
        return ContextSummaryResult(True, "checkpoint updated", checkpoint=checkpoint, transition=transition)

    def should_summarize(self, conversation_id: str) -> ContextSummaryDecision:
        turns = self.store.recent_turns(conversation_id, self.policy.max_turns_considered)
        checkpoint = self.store.get_conversation_checkpoint(conversation_id)
        estimated = estimate_tokens([turn.to_dict() for turn in turns])
        turns_since_checkpoint = count_turns_since_checkpoint(turns, checkpoint)
        if len(turns) < self.policy.min_turns_before_summary:
            return ContextSummaryDecision(False, "turn count below threshold", len(turns), turns_since_checkpoint, estimated)
        if turns_since_checkpoint < self.policy.min_turns_since_last_summary:
            return ContextSummaryDecision(False, "turns since last summary below threshold", len(turns), turns_since_checkpoint, estimated)
        if estimated < self.policy.max_recent_tokens_before_summary:
            return ContextSummaryDecision(False, "recent conversation tokens below threshold", len(turns), turns_since_checkpoint, estimated)
        return ContextSummaryDecision(True, "summary thresholds exceeded", len(turns), turns_since_checkpoint, estimated)

    def _build_summary_payload(self, conversation_id: str) -> dict[str, Any]:
        raw_turns = self.store.recent_turns(conversation_id, self.policy.max_turns_considered)
        checkpoint = self.store.get_conversation_checkpoint(conversation_id)
        active_games = self.store.active_games(conversation_id)
        active_game_ids = {item.game_id for item in active_games}
        payload = {
            "conversation_id": conversation_id,
            "existing_checkpoint": checkpoint.to_dict() if checkpoint else None,
            "recent_conversation": [],
            "active_games": [
                game_for_model_context(item, self.store.customers)
                for item in active_games
            ],
            "invite_drafts": [
                invite_draft_for_model_context(item, self.store.customers)
                for item in self.store.invite_drafts.values()
                if item.game_id in active_game_ids
            ],
            "outbound_message_drafts": [
                outbound_message_draft_for_model_context(item, self.store.customers)
                for item in self.store.outbound_message_drafts.values()
                if item.conversation_id == conversation_id
            ],
            "summary_input_budget": {},
            "summary_contract": {
                "format": "json_object",
                "required_keys": ["summary", "facts", "open_questions", "confidence"],
                "confidence_min_to_save": self.policy.min_confidence,
                "max_summary_chars": self.policy.max_summary_chars,
                "max_open_questions": self.policy.max_open_questions,
                "decision_critical_fact_hints": [
                    "current_objective",
                    "confirmed_facts",
                    "failed_attempts",
                    "temporary_constraints",
                    "completed_steps",
                    "pending_work",
                    "candidate_progress",
                    "active_game_id",
                ],
            },
        }
        # The policy limits the entire model request, not just raw turns. Reserve
        # room for the system prompt and structured state before packing history.
        fixed_messages = [
            {"role": "system", "content": self.prompt_path.read_text(encoding="utf-8")},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
        ]
        fixed_tokens = estimate_tokens(fixed_messages)
        # Keep headroom for the packing audit fields and provider chat framing.
        turn_budget = max(1, self.policy.max_summary_input_tokens - fixed_tokens - 512)
        turns, budget = pack_turns_for_summary(raw_turns, turn_budget)
        payload["recent_conversation"] = [turn.to_dict() for turn in turns]
        payload["summary_input_budget"] = {
            **budget,
            "fixed_prompt_tokens": fixed_tokens,
            "total_prompt_token_limit": self.policy.max_summary_input_tokens,
        }
        return payload

    def _validate_summary_facts(self, facts: dict[str, Any]) -> str | None:
        game_id = facts.get("active_game_id")
        if game_id is not None and str(game_id) not in self.store.games:
            return f"facts.active_game_id does not exist: {game_id}"
        game_ids = facts.get("active_game_ids")
        if isinstance(game_ids, list):
            missing = [str(item) for item in game_ids if str(item) not in self.store.games]
            if missing:
                return "facts.active_game_ids contain missing games: " + ", ".join(missing)
        return None

    def _record(self, trace_id: str, step: str, content: dict[str, Any], *, level: str = "INFO") -> None:
        if self.trace_recorder is not None:
            self.trace_recorder.record(trace_id, step, content, level=level)


def parse_context_summary(raw_response: str) -> tuple[dict[str, Any], list[str]]:
    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        return {}, [f"context summary is not valid JSON: {exc.msg}"]
    if not isinstance(payload, dict):
        return {}, ["context summary JSON root must be object"]
    errors: list[str] = []
    summary = payload.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        errors.append("summary must be non-empty string")
    if not isinstance(payload.get("facts"), dict):
        errors.append("facts must be object")
    open_questions = payload.get("open_questions")
    if not isinstance(open_questions, list):
        errors.append("open_questions must be array")
    elif any(not isinstance(item, str) for item in open_questions):
        errors.append("open_questions items must be string")
    confidence = payload.get("confidence")
    if not isinstance(confidence, int | float):
        errors.append("confidence must be number")
    elif not 0 <= float(confidence) <= 1:
        errors.append("confidence must be between 0 and 1")
    return payload, errors


def count_turns_since_checkpoint(turns: list[ConversationTurn], checkpoint: ConversationCheckpoint | None) -> int:
    if checkpoint is None:
        return len(turns)
    return sum(1 for turn in turns if turn.occurred_at > checkpoint.updated_at)


def pack_turns_for_summary(turns: list[ConversationTurn], max_tokens: int) -> tuple[list[ConversationTurn], dict[str, Any]]:
    included_reversed: list[ConversationTurn] = []
    estimated = 0
    omitted = 0
    for turn in reversed(turns):
        turn_tokens = estimate_tokens(turn.to_dict())
        if included_reversed and estimated + turn_tokens > max_tokens:
            omitted += 1
            continue
        included_reversed.append(turn)
        estimated += turn_tokens
    included = list(reversed(included_reversed))
    return included, {
        "total_turn_count": len(turns),
        "included_turn_count": len(included),
        "omitted_turn_count": omitted,
        "estimated_tokens": estimated,
        "max_tokens": max_tokens,
    }


def clamp_text(text: str, max_chars: int) -> str:
    stripped = text.strip()
    if len(stripped) <= max_chars:
        return stripped
    return stripped[: max(0, max_chars - 1)].rstrip() + "…"


def estimate_tokens(value: Any) -> int:
    """Compatibility wrapper around the shared conservative estimator."""

    return shared_estimate_tokens(value)
