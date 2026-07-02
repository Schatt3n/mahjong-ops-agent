from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .tool_orchestrator import ToolOrchestrationResult
from .workflow_models import (
    ActionName,
    ActionSource,
    ConversationContext,
    ReplyDraft,
    ReplyStatus,
    RiskLevel,
    SemanticResolution,
    StateTransition,
    ToolName,
    ValidatedAction,
)


DEFAULT_REPLY_PROMPT_PATH = Path(__file__).with_name("prompts") / "reply_draft.md"
INVITE_PROMISE_PATTERN = re.compile(r"帮你问|问问|问人|摇人|帮你摇")
ROOM_PROMISE_PATTERN = re.compile(r"房间.*(确认|留|定|安排)|留着|留座")


class ReplyDraftLLMClient(Protocol):
    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        trace_id: str,
        timeout_seconds: float,
    ) -> str | dict[str, Any]:
        ...


MISSING_SLOT_QUESTIONS: dict[str, str] = {
    "game_type": "打杭麻吗？",
    "stake": "打多大？",
    "start_time_mode": "大概什么时候开？",
    "party_size": "你这边几个人？",
    "smoke": "烟况有要求吗？",
    "duration_mode": "大概要打多久？",
    "duration_hours": "大概要打几个小时？",
}


@dataclass(slots=True)
class ReplyPolicyConfig:
    prompt_path: Path = DEFAULT_REPLY_PROMPT_PATH
    timeout_seconds: float = 8.0
    include_prompt_in_metadata: bool = True
    allow_json_fragment_extraction: bool = False


@dataclass(slots=True)
class ReplyPolicyInput:
    context: ConversationContext
    semantic_resolution: SemanticResolution
    validated_action: ValidatedAction
    tool_result: ToolOrchestrationResult
    state_transitions: list[StateTransition] = field(default_factory=list)


class ReplyPolicy:
    """Generate boss-facing drafts from final action results.

    This layer intentionally uses ValidatedAction and ToolResult only. It does
    not re-parse the user message, call tools, or mutate state.
    """

    def __init__(
        self,
        llm_client: ReplyDraftLLMClient | None = None,
        config: ReplyPolicyConfig | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.config = config or ReplyPolicyConfig()
        self._prompt_cache: str | None = None

    def draft(
        self,
        *,
        context: ConversationContext,
        semantic_resolution: SemanticResolution,
        validated_action: ValidatedAction,
        tool_result: ToolOrchestrationResult,
        state_transitions: list[StateTransition] | None = None,
    ) -> ReplyDraft:
        data = ReplyPolicyInput(
            context=context,
            semantic_resolution=semantic_resolution,
            validated_action=validated_action,
            tool_result=tool_result,
            state_transitions=state_transitions or [],
        )
        llm_contract: dict[str, Any] | None = None
        if self.llm_client is not None:
            llm_draft, llm_contract = self._draft_with_llm(data)
            if llm_draft is not None:
                return llm_draft

        action = validated_action.effective_action
        if action == ActionName.HUMAN_REVIEW:
            return self._draft(
                "这个我先转人工确认一下。",
                data,
                "高风险或不确定，转人工。",
                risk=RiskLevel.HIGH,
                llm_contract=llm_contract,
            )
        if action == ActionName.IGNORE:
            return self._draft("", data, "无需回复。", status=ReplyStatus.DRAFT, llm_contract=llm_contract)
        if action == ActionName.ASK_CLARIFICATION:
            return self._draft(
                self._clarification_text(validated_action.missing_slots),
                data,
                validated_action.reason,
                llm_contract=llm_contract,
            )
        if action == ActionName.ASK_CREATE_CONFIRMATION:
            return self._draft("现在没有合适的，要组一个吗？", data, validated_action.reason, llm_contract=llm_contract)
        if action == ActionName.MATCH_EXISTING_GAME:
            return self._draft(self._existing_game_text(tool_result), data, validated_action.reason, llm_contract=llm_contract)
        if action == ActionName.QUEUE_INVITES:
            return self._draft(self._queue_invites_text(tool_result), data, validated_action.reason, llm_contract=llm_contract)
        if action == ActionName.ACCEPT_SEAT:
            return self._draft(
                self._accept_seat_text(data),
                data,
                validated_action.reason,
                llm_contract=llm_contract,
            )
        if action == ActionName.CLOSE_GAME:
            return self._draft("收到，我先标记这桌需要处理。", data, validated_action.reason, llm_contract=llm_contract)
        return self._draft("我先确认一下。", data, f"未覆盖的有效动作：{action.value}", llm_contract=llm_contract)

    def build_messages(self, data: ReplyPolicyInput) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self._prompt_text()},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": "reply_draft_contract_v1",
                        "input": self._llm_input_payload(data),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            },
        ]

    def _draft_with_llm(self, data: ReplyPolicyInput) -> tuple[ReplyDraft | None, dict[str, Any] | None]:
        messages = self.build_messages(data)
        trace_id = data.context.current_message.trace_id
        try:
            raw_output = self.llm_client.complete(
                messages,
                trace_id=trace_id,
                timeout_seconds=self.config.timeout_seconds,
            )
        except Exception as exc:
            return None, self._llm_contract_audit(
                messages=messages,
                accepted=False,
                error=f"{type(exc).__name__}: {exc}",
            )
        raw, parse_error = _parse_reply_contract(
            raw_output,
            allow_json_fragment_extraction=self.config.allow_json_fragment_extraction,
        )
        if parse_error:
            return None, self._llm_contract_audit(
                messages=messages,
                accepted=False,
                parse_error=parse_error,
                raw_output=raw_output,
            )
        contract_errors = [
            *_validate_reply_contract(raw),
            *self._validate_reply_against_execution(raw, data),
        ]
        if contract_errors:
            return None, self._llm_contract_audit(
                messages=messages,
                accepted=False,
                contract_errors=contract_errors,
                raw_output=raw,
            )
        text = str(raw["text"])
        llm_contract = self._llm_contract_audit(
            messages=messages,
            accepted=True,
            raw_output=raw,
        )
        metadata: dict[str, Any] = {
            "schema": "reply_draft_contract_v1",
            "model_output": raw,
            "llm_contract": llm_contract,
            "effective_action": data.validated_action.effective_action.value,
            "validation_code": data.validated_action.code,
        }
        if self.config.include_prompt_in_metadata:
            metadata["prompt_messages"] = list(messages)
        return ReplyDraft(
            text=text,
            status=ReplyStatus.NEEDS_APPROVAL if text else ReplyStatus.DRAFT,
            reasoning_summary=str(raw["reasoning_summary"]).strip(),
            source=ActionSource.LLM,
            risk_level=RiskLevel(str(raw["risk_level"])),
            metadata=metadata,
        ), llm_contract

    def _validate_reply_against_execution(self, raw: dict[str, Any], data: ReplyPolicyInput) -> list[str]:
        text = str(raw.get("text") or "")
        errors: list[str] = []
        if INVITE_PROMISE_PATTERN.search(text) and not _has_pending_outbox(data.tool_result):
            errors.append("reply promises inviting players before create_pending_outbox succeeded")
        if ROOM_PROMISE_PATTERN.search(text):
            errors.append("reply promises room reservation without room availability confirmation")
        return errors

    def _prompt_text(self) -> str:
        if self._prompt_cache is None:
            self._prompt_cache = self.config.prompt_path.read_text(encoding="utf-8")
        return self._prompt_cache

    def _llm_input_payload(self, data: ReplyPolicyInput) -> dict[str, Any]:
        return {
            "current_message": data.context.current_message.to_prompt_dict(),
            "previous_system_reply": data.context.previous_system_reply(),
            "semantic_resolution": {
                "intent": data.semantic_resolution.intent.value,
                "proposed_action": data.semantic_resolution.proposed_action.name.value,
                "confidence": data.semantic_resolution.proposed_action.confidence,
                "reasoning_summary": data.semantic_resolution.reasoning_summary,
                "game_requirement": data.semantic_resolution.game_requirement.to_prompt_dict(),
            },
            "validated_action": {
                "effective_action": data.validated_action.effective_action.value,
                "allowed": data.validated_action.allowed,
                "code": data.validated_action.code,
                "reason": data.validated_action.reason,
                "missing_slots": list(data.validated_action.missing_slots),
                "approval_required": data.validated_action.approval_required,
                "risk_level": data.validated_action.risk_level.value,
                "required_tools": [tool.value for tool in data.validated_action.required_tools],
            },
            "tool_results": [
                {
                    "tool_name": item.request.tool_name.value,
                    "called": item.called,
                    "allowed": item.allowed,
                    "error": item.error,
                    "result": dict(item.result),
                }
                for item in data.tool_result.tool_results
            ],
            "state_transitions": [
                {
                    "entity_type": item.entity_type,
                    "entity_id": item.entity_id,
                    "from_status": item.from_status,
                    "to_status": item.to_status,
                    "allowed": item.allowed,
                    "reason": item.reason,
                    "metadata": dict(item.metadata),
                }
                for item in data.state_transitions
            ],
        }

    def _draft(
        self,
        text: str,
        data: ReplyPolicyInput,
        reasoning_summary: str,
        *,
        status: ReplyStatus = ReplyStatus.NEEDS_APPROVAL,
        risk: RiskLevel | None = None,
        llm_contract: dict[str, Any] | None = None,
    ) -> ReplyDraft:
        metadata: dict[str, Any] = {
            "effective_action": data.validated_action.effective_action.value,
            "validation_code": data.validated_action.code,
            "tool_results": [
                {
                    "tool_name": item.request.tool_name.value,
                    "called": item.called,
                    "allowed": item.allowed,
                    "error": item.error,
                }
                for item in data.tool_result.tool_results
            ],
            "state_transitions": [
                {
                    "entity_type": item.entity_type,
                    "entity_id": item.entity_id,
                    "from_status": item.from_status,
                    "to_status": item.to_status,
                    "allowed": item.allowed,
                    "metadata": dict(item.metadata),
                }
                for item in data.state_transitions
            ],
        }
        if llm_contract is not None:
            metadata["llm_contract"] = llm_contract
        return ReplyDraft(
            text=text,
            status=status,
            reasoning_summary=reasoning_summary,
            source=ActionSource.RULES,
            risk_level=risk or data.validated_action.risk_level,
            metadata=metadata,
        )

    def _llm_contract_audit(
        self,
        *,
        messages: list[dict[str, str]],
        accepted: bool,
        raw_output: Any | None = None,
        parse_error: str | None = None,
        contract_errors: list[str] | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        audit: dict[str, Any] = {
            "schema": "reply_draft_contract_v1",
            "attempted": True,
            "accepted": accepted,
            "strict_json": not self.config.allow_json_fragment_extraction,
        }
        if raw_output is not None:
            audit["raw_output"] = raw_output
        if parse_error:
            audit["parse_error"] = parse_error
        if contract_errors:
            audit["contract_errors"] = list(contract_errors)
        if error:
            audit["error"] = error
        if self.config.include_prompt_in_metadata:
            audit["prompt_messages"] = list(messages)
        return audit

    def _clarification_text(self, missing_slots: list[str]) -> str:
        questions = [MISSING_SLOT_QUESTIONS.get(slot, f"{slot} 再确认一下？") for slot in missing_slots[:3]]
        if not questions:
            return "我再确认一下。"
        return " ".join(questions)

    def _existing_game_text(self, tool_result: ToolOrchestrationResult) -> str:
        result = tool_result.result_for(ToolName.SEARCH_CURRENT_OPEN_GAMES)
        matches = (result.result.get("matches") if result and result.called and result.allowed else []) or []
        if not matches:
            return "我先看一下有没有合适的。"
        summary = str(matches[0].get("summary") or "有一桌合适的")
        return f"{summary}，要不要加？"

    def _queue_invites_text(self, tool_result: ToolOrchestrationResult) -> str:
        outbox = tool_result.result_for(ToolName.CREATE_PENDING_OUTBOX)
        drafts = (outbox.result.get("drafts") if outbox and outbox.called and outbox.allowed else []) or []
        if drafts:
            return "好的，我帮你问问。"
        candidate_result = tool_result.result_for(ToolName.SEARCH_CANDIDATE_CUSTOMERS)
        if candidate_result and candidate_result.called and candidate_result.allowed:
            return "我先看下合适的人选。"
        return "我先确认一下。"

    def _accept_seat_text(self, data: ReplyPolicyInput) -> str:
        for transition in reversed(data.state_transitions):
            seat_delta = transition.metadata.get("seat_delta")
            if not isinstance(seat_delta, dict):
                continue
            current_count = _coerce_int(seat_delta.get("current_player_count"))
            missing_count = _coerce_int(seat_delta.get("missing_count"))
            if current_count is None or missing_count is None:
                continue
            label = _party_progress_label(current_count, missing_count)
            if label == "人齐":
                return "好的，加你了，人齐了。"
            return f"好的，加你{label}了。"

        accept_result = data.tool_result.result_for(ToolName.RECORD_SEAT_ACCEPTANCE)
        intent = accept_result.result.get("state_write_intent") if accept_result and accept_result.result else {}
        seat_delta = intent.get("seat_delta") if isinstance(intent, dict) else {}
        if isinstance(seat_delta, dict):
            current_count = _coerce_int(seat_delta.get("current_player_count"))
            missing_count = _coerce_int(seat_delta.get("missing_count"))
            if current_count is not None and missing_count is not None:
                label = _party_progress_label(current_count, missing_count)
                if label == "人齐":
                    return "好的，加你了，人齐了。"
                return f"好的，加你{label}了。"
        return "好的，加你了。"


REPLY_REQUIRED_FIELDS: tuple[str, ...] = ("text", "reasoning_summary", "risk_level")
REPLY_ALLOWED_RISK_LEVELS = frozenset(item.value for item in RiskLevel)


def _validate_reply_contract(raw: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REPLY_REQUIRED_FIELDS:
        if field not in raw:
            errors.append(f"missing required field {field!r}")

    if "text" in raw and not isinstance(raw.get("text"), str):
        errors.append("text must be a string")

    if "reasoning_summary" in raw and not _optional_str(raw.get("reasoning_summary")):
        errors.append("reasoning_summary must be a non-empty string")

    risk_level = raw.get("risk_level")
    if "risk_level" in raw and str(risk_level or "").strip() not in REPLY_ALLOWED_RISK_LEVELS:
        errors.append(f"invalid risk_level {risk_level!r}")

    return errors


def _parse_reply_contract(
    raw_output: str | dict[str, Any],
    *,
    allow_json_fragment_extraction: bool = False,
) -> tuple[dict[str, Any], str | None]:
    if isinstance(raw_output, dict):
        return raw_output, None
    text = str(raw_output or "").strip()
    if not text:
        return {}, "reply draft LLM returned empty output."
    try:
        raw = json.loads(text)
        if not isinstance(raw, dict):
            return {}, "reply draft LLM JSON root is not an object."
        return raw, None
    except json.JSONDecodeError:
        if not allow_json_fragment_extraction:
            return {}, "reply draft LLM output must be a single JSON object with no surrounding text."
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            return {}, "reply draft LLM returned no JSON object."
        try:
            raw = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            return {}, f"reply draft LLM returned invalid JSON: {exc}"
        if not isinstance(raw, dict):
            return {}, "reply draft LLM JSON fragment is not an object."
        return raw, None


def _risk_from_raw(value: Any, *, default: RiskLevel) -> RiskLevel:
    try:
        return RiskLevel(str(value or default.value))
    except ValueError:
        return default


def _has_pending_outbox(tool_result: ToolOrchestrationResult) -> bool:
    result = tool_result.result_for(ToolName.CREATE_PENDING_OUTBOX)
    if not result or not result.called or not result.allowed:
        return False
    drafts = result.result.get("drafts")
    return isinstance(drafts, list) and bool(drafts)


def _optional_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _party_progress_label(current_count: int, missing_count: int) -> str:
    if missing_count <= 0:
        return "人齐"
    mapping = {
        (1, 3): "173",
        (2, 2): "272",
        (3, 1): "371",
    }
    return mapping.get((current_count, missing_count), f"{current_count}缺{missing_count}")
