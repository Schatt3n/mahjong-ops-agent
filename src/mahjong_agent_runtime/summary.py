"""管理长对话的结构化摘要与检查点，避免上下文超预算时遗失关键事实。"""

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
    """集中配置摘要触发阈值、输入预算、输出上限和最低可信度。"""
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
    """记录是否需要摘要以及作出判断时使用的轮次和 token 指标。"""
    should_summarize: bool
    reason: str
    turn_count: int
    turns_since_last_summary: int
    estimated_recent_tokens: int

    def to_dict(self) -> dict[str, Any]:
        """转换为只包含可序列化字段的字典表示。"""
        return {
            "should_summarize": self.should_summarize,
            "reason": self.reason,
            "turn_count": self.turn_count,
            "turns_since_last_summary": self.turns_since_last_summary,
            "estimated_recent_tokens": self.estimated_recent_tokens,
        }


@dataclass(slots=True)
class ContextSummaryResult:
    """返回摘要是否落库，以及对应 checkpoint 和状态变更。"""
    summarized: bool
    reason: str
    checkpoint: ConversationCheckpoint | None = None
    transition: StateTransition | None = None

    def to_dict(self) -> dict[str, Any]:
        """转换为只包含可序列化字段的字典表示。"""
        return {
            "summarized": self.summarized,
            "reason": self.reason,
            "checkpoint": self.checkpoint.to_dict() if self.checkpoint else None,
            "transition": self.transition.to_dict() if self.transition else None,
        }


@dataclass(slots=True)
class ContextSummaryManager:
    """根据 token 和轮次阈值生成结构化检查点，并校验摘要没有丢失关键事实。"""
    store: Any
    llm_client: AgentLLMClient
    trace_recorder: Any | None = None
    policy: ContextSummaryPolicy = field(default_factory=ContextSummaryPolicy)
    prompt_path: Path = DEFAULT_CONTEXT_SUMMARY_PROMPT_PATH

    def maybe_summarize_after_turn(
        self,
        *,
        conversation_id: str,
        trace_id: str,
        task_context_id: str | None = None,
    ) -> ContextSummaryResult:
        """在一轮处理结束后按常规阈值尝试生成 checkpoint。"""

        decision = self.should_summarize(conversation_id, task_context_id=task_context_id)
        self._record(trace_id, "context_summary_checked", decision.to_dict())
        if not decision.should_summarize:
            return ContextSummaryResult(False, decision.reason)

        return self._summarize(
            conversation_id=conversation_id,
            trace_id=trace_id,
            trigger="after_turn",
            task_context_id=task_context_id,
        )

    def summarize_for_context_budget(
        self,
        *,
        conversation_id: str,
        trace_id: str,
        estimated_context_tokens: int,
        max_context_tokens: int,
        trigger_threshold_tokens: int,
        task_context_id: str | None = None,
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
        return self._summarize(
            conversation_id=conversation_id,
            trace_id=trace_id,
            trigger="context_budget",
            task_context_id=task_context_id,
        )

    def summarize_for_quality_evaluation(
        self,
        *,
        conversation_id: str,
        trace_id: str,
        task_context_id: str | None = None,
    ) -> ContextSummaryResult:
        """为离线决策一致性评测强制生成一次 checkpoint。

"""

        self._record(
            trace_id,
            "context_summary_quality_evaluation_triggered",
            {"conversation_id": conversation_id},
        )
        return self._summarize(
            conversation_id=conversation_id,
            trace_id=trace_id,
            trigger="quality_evaluation",
            task_context_id=task_context_id,
        )

    def _summarize(
        self,
        *,
        conversation_id: str,
        trace_id: str,
        trigger: str,
        task_context_id: str | None = None,
    ) -> ContextSummaryResult:
        """执行一次摘要模型调用并保存 checkpoint。"""

        payload = self._build_summary_payload(
            conversation_id,
            task_context_id=task_context_id,
        )
        resolved_task_context_id = str(payload.get("task_context_id") or "") or None
        messages = [
            {"role": "system", "content": self.prompt_path.read_text(encoding="utf-8")},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
        ]
        # 摘要本身也会消耗上下文；调用前必须先估算完整请求，而不能只统计聊天轮次。
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
        # 摘要是后续决策的输入，因此合同不合法时宁可保留原历史，也不能保存半结构化结果。
        summary, errors = parse_context_summary(raw_response)
        if errors:
            self._record(trace_id, "context_summary_contract_error", {"errors": errors, "raw_response": raw_response}, level="WARN")
            return ContextSummaryResult(False, "context summary contract invalid: " + "; ".join(errors))

        # 置信度是模型的自评，只作为第一道门槛；关键业务 ID 仍要由后端事实校验。
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

        # 只有合同、置信度和事实引用全部通过后才原子更新 checkpoint。
        checkpoint, transition = self.store.upsert_conversation_checkpoint(
            conversation_id=conversation_id,
            summary=clamp_text(str(summary.get("summary") or ""), self.policy.max_summary_chars),
            facts=facts,
            open_questions=[str(item)[:200] for item in summary.get("open_questions") or []][: self.policy.max_open_questions],
            trace_id=trace_id,
            task_context_id=resolved_task_context_id,
        )
        self._record(trace_id, "context_summary_saved", {"checkpoint": checkpoint.to_dict(), "transition": transition.to_dict()})
        self._record(trace_id, "state_transition", transition.to_dict())
        return ContextSummaryResult(True, "checkpoint updated", checkpoint=checkpoint, transition=transition)

    def should_summarize(
        self,
        conversation_id: str,
        *,
        task_context_id: str | None = None,
    ) -> ContextSummaryDecision:
        """综合估算 token、累计轮数和距上次摘要的增量判断是否需要压缩。"""
        # 三个条件同时满足才做常规摘要：历史足够长、上次摘要后有新信息、
        # 且近期原文已经形成明显 token 压力，避免每几轮就反复调用摘要模型。
        _, turns, checkpoint = self._summary_scope(
            conversation_id,
            task_context_id=task_context_id,
        )
        estimated = estimate_tokens([turn.to_dict() for turn in turns])
        turns_since_checkpoint = count_turns_since_checkpoint(turns, checkpoint)
        if len(turns) < self.policy.min_turns_before_summary:
            return ContextSummaryDecision(False, "turn count below threshold", len(turns), turns_since_checkpoint, estimated)
        if turns_since_checkpoint < self.policy.min_turns_since_last_summary:
            return ContextSummaryDecision(False, "turns since last summary below threshold", len(turns), turns_since_checkpoint, estimated)
        if estimated < self.policy.max_recent_tokens_before_summary:
            return ContextSummaryDecision(False, "recent conversation tokens below threshold", len(turns), turns_since_checkpoint, estimated)
        return ContextSummaryDecision(True, "summary thresholds exceeded", len(turns), turns_since_checkpoint, estimated)

    def _build_summary_payload(
        self,
        conversation_id: str,
        *,
        task_context_id: str | None = None,
    ) -> dict[str, Any]:
        """构造任务级摘要输入，并给固定结构和历史轮次分别预留预算。"""
        resolved_task_context_id, raw_turns, checkpoint = self._summary_scope(
            conversation_id,
            task_context_id=task_context_id,
        )
        active_games = self.store.active_games(conversation_id)
        active_game_ids = {item.game_id for item in active_games}
        payload = {
            "conversation_id": conversation_id,
            "task_context_id": resolved_task_context_id,
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
                "scope_contract": (
                    "Summarize only this task_context_id. Do not merge facts from other business episodes "
                    "that share the same conversation_id."
                ),
            },
        }
        # 策略限制的是完整模型请求，不只是原始对话；先为系统提示词和结构化状态预留空间。
        fixed_messages = [
            {"role": "system", "content": self.prompt_path.read_text(encoding="utf-8")},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
        ]
        fixed_tokens = estimate_tokens(fixed_messages)
        # 再为预算审计字段和模型供应商的消息封装保留余量。
        turn_budget = max(1, self.policy.max_summary_input_tokens - fixed_tokens - 512)
        turns, budget = pack_turns_for_summary(raw_turns, turn_budget)
        payload["recent_conversation"] = [turn.to_dict() for turn in turns]
        payload["summary_input_budget"] = {
            **budget,
            "fixed_prompt_tokens": fixed_tokens,
            "total_prompt_token_limit": self.policy.max_summary_input_tokens,
        }
        return payload

    def _summary_scope(
        self,
        conversation_id: str,
        *,
        task_context_id: str | None,
    ) -> tuple[str | None, list[ConversationTurn], ConversationCheckpoint | None]:
        """先确定任务级历史范围，再应用轮数限制。

        """

        requested_task_context_id = str(task_context_id or "") or None
        task_context = (
            self.store.get_task_context(requested_task_context_id)
            if requested_task_context_id
            else self.store.latest_task_context(conversation_id)
        )
        if requested_task_context_id and (
            task_context is None or task_context.conversation_id != conversation_id
        ):
            return requested_task_context_id, [], None
        if task_context is None:
            return (
                None,
                [turn for turn in self.store.recent_turns(conversation_id, self.policy.max_turns_considered) if not turn.metadata.get("context_invalidated")],
                self.store.get_conversation_checkpoint(conversation_id),
            )

        resolved_task_context_id = task_context.task_context_id
        turns = self.store.task_context_turns(
            conversation_id,
            resolved_task_context_id,
            self.policy.max_turns_considered,
        )
        # 旧数据库可能存在没有 task_context_id 的历史轮次。这里只补入当前任务时间窗内的
        # 旧轮次；已经明确属于其他任务的轮次仍然排除，避免跨场组局串话。
        known_turns = {
            (turn.trace_id, turn.role.value, turn.occurred_at.isoformat(), turn.content)
            for turn in turns
        }
        for turn in self.store.recent_turns(conversation_id, self.policy.max_turns_considered):
            identity = (turn.trace_id, turn.role.value, turn.occurred_at.isoformat(), turn.content)
            if (
                not turn.metadata.get("task_context_id")
                and turn.occurred_at >= task_context.started_at
                and (task_context.closed_at is None or turn.occurred_at <= task_context.closed_at)
                and identity not in known_turns
            ):
                turns.append(turn)
                known_turns.add(identity)
        turns.sort(key=lambda turn: turn.occurred_at)
        checkpoint = self.store.get_task_context_checkpoint(resolved_task_context_id)
        return resolved_task_context_id, [turn for turn in turns if not turn.metadata.get("context_invalidated")], checkpoint

    def _validate_summary_facts(self, facts: dict[str, Any]) -> str | None:
        """校验摘要引用的业务实体仍然真实存在，拒绝模型虚构的局 ID。"""
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
        """在启用追踪器时记录摘要链路事件。"""
        if self.trace_recorder is not None:
            self.trace_recorder.record(trace_id, step, content, level=level)


def parse_context_summary(raw_response: str) -> tuple[dict[str, Any], list[str]]:
    """解析上下文摘要，并把非结构化输入转换为稳定合同。"""
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
    """统计上次 checkpoint 之后新增的原始对话轮次。"""
    if checkpoint is None:
        return len(turns)
    return sum(1 for turn in turns if turn.occurred_at > checkpoint.updated_at)


def pack_turns_for_summary(turns: list[ConversationTurn], max_tokens: int) -> tuple[list[ConversationTurn], dict[str, Any]]:
    """从最近一轮向前装入历史，直到达到摘要输入预算。"""
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
    """把摘要文本限制在持久化上限内，并用省略号标记截断。"""
    stripped = text.strip()
    if len(stripped) <= max_chars:
        return stripped
    return stripped[: max(0, max_chars - 1)].rstrip() + "…"


def estimate_tokens(value: Any) -> int:
    """估算tokens，用于保守的预算和容量判断。

"""

    return shared_estimate_tokens(value)
