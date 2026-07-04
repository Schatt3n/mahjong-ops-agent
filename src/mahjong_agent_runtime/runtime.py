from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .context import AgentContextBuilder, estimate_tokens
from .llm import AgentLLMClient
from .models import AgentAction, AgentRuntimeResult, ToolResult, UserMessage
from .store import InMemoryAgentStore
from .summary import ContextSummaryManager
from .tools import ToolGateway
from .tracing import InMemoryTraceRecorder


DEFAULT_REPLY_SELF_REVIEW_PROMPT_PATH = Path(__file__).with_name("prompts").joinpath("agent_runtime_reply_self_review.md")
CUSTOMER_VISIBLE_CONTENT_REVIEW_TOOL_NAME = "customer_visible_content_review"
REPLY_SELF_REVIEW_TOOL_NAME = CUSTOMER_VISIBLE_CONTENT_REVIEW_TOOL_NAME


@dataclass(slots=True)
class BudgetDecision:
    allowed: bool
    reason: str
    estimated_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return {"allowed": self.allowed, "reason": self.reason, "estimated_tokens": self.estimated_tokens}


@dataclass(slots=True)
class TokenBudget:
    max_tokens_per_call: int = 24_000
    max_calls_per_turn: int = 8
    calls_this_turn: int = 0

    def reserve(self, messages: list[dict[str, str]]) -> BudgetDecision:
        self.calls_this_turn += 1
        estimated = sum(estimate_tokens(item.get("content", "")) for item in messages)
        if self.calls_this_turn > self.max_calls_per_turn:
            return BudgetDecision(False, f"turn llm call limit exceeded: {self.max_calls_per_turn}", estimated)
        if estimated > self.max_tokens_per_call:
            return BudgetDecision(False, f"single call token estimate exceeded: {estimated}>{self.max_tokens_per_call}", estimated)
        return BudgetDecision(True, "budget_reserved", estimated)


@dataclass(slots=True)
class AgentRuntime:
    llm_client: AgentLLMClient
    store: InMemoryAgentStore = field(default_factory=InMemoryAgentStore)
    tool_gateway: ToolGateway | None = None
    trace_recorder: Any = field(default_factory=InMemoryTraceRecorder)
    token_budget: TokenBudget = field(default_factory=TokenBudget)
    review_token_budget: TokenBudget = field(default_factory=TokenBudget)
    max_steps: int = 8
    llm_timeout_seconds: float = 45.0
    reply_self_review_enabled: bool = False
    reply_self_review_client: AgentLLMClient | None = None
    reply_self_review_prompt_path: Path = DEFAULT_REPLY_SELF_REVIEW_PROMPT_PATH
    context_summary_manager: ContextSummaryManager | None = None
    context_builder: AgentContextBuilder = field(init=False)
    _conversation_locks: dict[str, threading.RLock] = field(default_factory=dict, init=False, repr=False)
    _conversation_locks_guard: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.tool_gateway is None:
            self.tool_gateway = ToolGateway(self.store)
        if self.tool_gateway.trace_recorder is None:
            self.tool_gateway.trace_recorder = self.trace_recorder
        self.context_builder = AgentContextBuilder(self.store, self.tool_gateway)

    def handle_user_message(self, message: UserMessage, *, trace_id: str | None = None) -> AgentRuntimeResult:
        with self._conversation_lock(message.conversation_id):
            actual_trace_id = trace_id or f"trace_{uuid.uuid4().hex[:12]}"
            message_key = message_idempotency_key(message)
            cached = self.store.idempotent_message_result(message_key)
            if cached is not None:
                self.trace_recorder.record(actual_trace_id, "user_input", {"message": message.to_dict()})
                self.trace_recorder.record(
                    actual_trace_id,
                    "message_deduplicated",
                    {
                        "message_id": message.message_id,
                        "message_idempotency_key": message_key,
                        "original_trace_id": cached.trace_id,
                    },
                )
                self.trace_recorder.record(actual_trace_id, "final_output", {"reply": cached.final_reply, "reason": "message_deduplicated"})
                return cached
            result = self._handle_once(message, trace_id=actual_trace_id)
            if self.context_summary_manager is not None:
                try:
                    summary_result = self.context_summary_manager.maybe_summarize_after_turn(
                        conversation_id=message.conversation_id,
                        trace_id=actual_trace_id,
                    )
                    if summary_result.transition is not None:
                        result.state_transitions.append(summary_result.transition)
                except Exception as exc:
                    self.trace_recorder.record(
                        actual_trace_id,
                        "context_summary_error",
                        {"error_type": type(exc).__name__, "error": str(exc)},
                        level="ERROR",
                    )
            self.store.remember_message_result(message_key, result)
            return result

    def _handle_once(self, message: UserMessage, *, trace_id: str) -> AgentRuntimeResult:
        turn_budget = TokenBudget(
            max_tokens_per_call=self.token_budget.max_tokens_per_call,
            max_calls_per_turn=self.token_budget.max_calls_per_turn,
        )
        review_turn_budget = TokenBudget(
            max_tokens_per_call=self.review_token_budget.max_tokens_per_call,
            max_calls_per_turn=self.review_token_budget.max_calls_per_turn,
        )
        self.store.append_user_turn(message, trace_id)
        self.trace_recorder.record(trace_id, "user_input", {"message": message.to_dict()})
        actions: list[AgentAction] = []
        tool_results: list[ToolResult] = []
        pending_tool_results: list[ToolResult] = []
        final_reply = ""

        for step_index in range(1, self.max_steps + 1):
            built = self.context_builder.build(message, trace_id=trace_id, previous_tool_results=pending_tool_results)
            self.trace_recorder.record(trace_id, "context_packed", built.audit)
            self.trace_recorder.record(trace_id, "context_built", built.payload)
            self.trace_recorder.record(trace_id, "llm_prompt", {"messages": built.messages, "step_index": step_index})
            budget = turn_budget.reserve(built.messages)
            self.trace_recorder.record(trace_id, "budget_checked", budget.to_dict())
            if not budget.allowed:
                final_reply = "这个我先转人工确认一下。"
                self.trace_recorder.record(trace_id, "final_output", {"reply": final_reply, "reason": budget.reason}, level="WARN")
                self.store.append_assistant_turn(message.conversation_id, final_reply, trace_id)
                break

            started = time.perf_counter()
            try:
                raw_response = self.llm_client.complete(built.messages, trace_id=trace_id, timeout_seconds=self.llm_timeout_seconds)
            except Exception as exc:
                final_reply = "这个我先转人工确认一下。"
                self.trace_recorder.record(
                    trace_id,
                    "llm_error",
                    {"error_type": type(exc).__name__, "error": str(exc), "elapsed_ms": int((time.perf_counter() - started) * 1000)},
                    level="ERROR",
                )
                self.trace_recorder.record(trace_id, "final_output", {"reply": final_reply, "reason": "llm_error"}, level="WARN")
                self.store.append_assistant_turn(message.conversation_id, final_reply, trace_id)
                break
            self.trace_recorder.record(
                trace_id,
                "llm_response",
                {"content": raw_response, "elapsed_ms": int((time.perf_counter() - started) * 1000), "step_index": step_index},
            )
            action, errors = parse_action(raw_response)
            actions.append(action)
            if errors:
                self.trace_recorder.record(trace_id, "action_contract_error", {"errors": errors, "step_index": step_index}, level="WARN")
                feedback = ToolResult(
                    name="agent_action_contract",
                    called=False,
                    allowed=False,
                    result={
                        "errors": list(errors),
                        "raw_response": raw_response,
                        "instruction": "Fix the AgentAction JSON contract. If waiting for user, use objective_status=waiting_user with non-empty reply_to_user. If tools are needed, use objective_status=needs_tool with at least one tool_call.",
                    },
                    error="AgentAction contract invalid: " + "; ".join(errors),
                )
                pending_tool_results = [feedback]
                self.trace_recorder.record(trace_id, "contract_error_feedback", feedback.to_dict(), level="WARN")
                self.store.append_tool_turn(message.conversation_id, json.dumps([feedback.to_dict()], ensure_ascii=False), trace_id)
                continue
            self.trace_recorder.record(trace_id, "action_proposed", action.to_dict())

            if action.tool_calls:
                review_items = customer_visible_items_for_action(action)
                review_result = self._run_customer_visible_content_review(
                    message=message,
                    trace_id=trace_id,
                    action=action,
                    review_items=review_items,
                    context_payload=built.payload,
                    turn_budget=review_turn_budget,
                    review_scope="tool_calls",
                )
                if review_result is not None:
                    tool_results.append(review_result)
                    self.trace_recorder.record(trace_id, "tool_result", review_result.to_dict())
                    self.store.append_tool_turn(message.conversation_id, json.dumps([review_result.to_dict()], ensure_ascii=False), trace_id)
                    if not customer_visible_content_review_approved(review_result):
                        pending_tool_results = [review_result]
                        continue
                pending_tool_results = []
                for call_index, call in enumerate(action.tool_calls, start=1):
                    self.trace_recorder.record(trace_id, "tool_called", {"call": call.to_dict(), "step_index": step_index})
                    result = self.tool_gateway.execute(
                        call,
                        trace_id=trace_id,
                        conversation_id=message.conversation_id,
                        sender_id=message.sender_id,
                        sender_name=message.sender_name,
                        step_index=step_index * 100 + call_index,
                        source_message_id=message.message_id,
                    )
                    tool_results.append(result)
                    pending_tool_results.append(result)
                    self.trace_recorder.record(trace_id, "tool_result", result.to_dict())
                    for transition in result.state_transitions:
                        step = "state_transition_replayed" if result.deduplicated else "state_transition"
                        self.trace_recorder.record(trace_id, step, transition.to_dict())
                self.store.append_tool_turn(message.conversation_id, json.dumps([item.to_dict() for item in pending_tool_results], ensure_ascii=False), trace_id)
                continue

            proposed_reply = action.reply_to_user.strip()
            if action.needs_human and not proposed_reply:
                proposed_reply = "这个我先转人工确认一下。"
            review_result = self._run_customer_visible_content_review(
                message=message,
                trace_id=trace_id,
                action=action,
                review_items=[
                    {
                        "item_id": "reply_to_user",
                        "source": "reply_to_user",
                        "recipient_id": message.sender_id,
                        "recipient_name": message.sender_name,
                        "text": proposed_reply,
                    }
                ],
                context_payload=built.payload,
                turn_budget=review_turn_budget,
                review_scope="reply_to_user",
            )
            if review_result is not None:
                tool_results.append(review_result)
                self.trace_recorder.record(trace_id, "tool_result", review_result.to_dict())
                self.store.append_tool_turn(message.conversation_id, json.dumps([review_result.to_dict()], ensure_ascii=False), trace_id)
                if not customer_visible_content_review_approved(review_result):
                    pending_tool_results = [review_result]
                    continue
            final_reply = proposed_reply
            self.store.append_assistant_turn(message.conversation_id, final_reply, trace_id)
            self.trace_recorder.record(trace_id, "final_output", {"reply": final_reply, "objective_status": action.objective_status})
            break
        else:
            final_reply = "这个我先转人工确认一下。"
            self.store.append_assistant_turn(message.conversation_id, final_reply, trace_id)
            self.trace_recorder.record(trace_id, "final_output", {"reply": final_reply, "reason": "max_steps_exceeded"}, level="WARN")

        transitions = [transition for result in tool_results if not result.deduplicated for transition in result.state_transitions]
        return AgentRuntimeResult(
            trace_id=trace_id,
            conversation_id=message.conversation_id,
            final_reply=final_reply,
            actions=actions,
            tool_results=tool_results,
            state_transitions=transitions,
        )

    def _conversation_lock(self, conversation_id: str) -> threading.RLock:
        key = conversation_id or "default"
        with self._conversation_locks_guard:
            lock = self._conversation_locks.get(key)
            if lock is None:
                lock = threading.RLock()
                self._conversation_locks[key] = lock
            return lock

    def _run_customer_visible_content_review(
        self,
        *,
        message: UserMessage,
        trace_id: str,
        action: AgentAction,
        review_items: list[dict[str, Any]],
        context_payload: dict[str, Any],
        turn_budget: TokenBudget,
        review_scope: str,
    ) -> ToolResult | None:
        if not self.reply_self_review_enabled or not review_items:
            return None
        client = self.reply_self_review_client or self.llm_client
        review_payload = build_reply_self_review_payload(
            message=message,
            action=action,
            review_items=review_items,
            context_payload=context_payload,
            review_scope=review_scope,
        )
        messages = [
            {"role": "system", "content": self.reply_self_review_prompt_path.read_text(encoding="utf-8")},
            {"role": "user", "content": json.dumps(review_payload, ensure_ascii=False, sort_keys=True)},
        ]
        self.trace_recorder.record(trace_id, "customer_visible_content_review_prompt", {"messages": messages})
        budget = turn_budget.reserve(messages)
        self.trace_recorder.record(trace_id, "customer_visible_content_review_budget_checked", budget.to_dict())
        if not budget.allowed:
            self.trace_recorder.record(trace_id, "customer_visible_content_review_failed", {"reason": budget.reason}, level="WARN")
            return ToolResult(
                name=CUSTOMER_VISIBLE_CONTENT_REVIEW_TOOL_NAME,
                called=False,
                allowed=False,
                result={
                    "review_scope": review_scope,
                    "items": review_items,
                    "instruction": "Customer-visible content review could not run because the turn budget was exhausted. Use needs_human with a short safe handoff reply.",
                },
                error=budget.reason,
            )
        started = time.perf_counter()
        try:
            raw_response = client.complete(messages, trace_id=trace_id, timeout_seconds=self.llm_timeout_seconds)
        except Exception as exc:
            self.trace_recorder.record(
                trace_id,
                "customer_visible_content_review_error",
                {"error_type": type(exc).__name__, "error": str(exc), "elapsed_ms": int((time.perf_counter() - started) * 1000)},
                    level="ERROR",
            )
            return ToolResult(
                name=CUSTOMER_VISIBLE_CONTENT_REVIEW_TOOL_NAME,
                called=False,
                allowed=False,
                result={
                    "review_scope": review_scope,
                    "items": review_items,
                    "instruction": "Customer-visible content review failed. Use needs_human with a short safe handoff reply.",
                },
                error=f"{type(exc).__name__}: {exc}",
            )
        self.trace_recorder.record(
            trace_id,
            "customer_visible_content_review_response",
            {"content": raw_response, "elapsed_ms": int((time.perf_counter() - started) * 1000)},
        )
        review, errors = parse_reply_self_review(raw_response)
        if errors:
            self.trace_recorder.record(trace_id, "customer_visible_content_review_contract_error", {"errors": errors}, level="WARN")
            return ToolResult(
                name=CUSTOMER_VISIBLE_CONTENT_REVIEW_TOOL_NAME,
                called=True,
                allowed=False,
                result={
                    "review_scope": review_scope,
                    "items": review_items,
                    "errors": list(errors),
                    "raw_response": raw_response,
                    "instruction": "Customer-visible content review returned an invalid contract. Use needs_human with a short safe handoff reply.",
                },
                error="Customer-visible content review contract invalid: " + "; ".join(errors),
            )
        item_reviews = normalize_item_reviews(review, review_items)
        approved = bool(review.get("approved")) and not bool(review.get("needs_human")) and item_reviews_approved(item_reviews, review_items)
        needs_human = bool(review.get("needs_human"))
        instruction = (
            "Review approved all customer-visible content. You may continue with the original action."
            if approved
            else "Review rejected customer-visible content. Read violations and suggested_safe_text values, then generate a corrected AgentAction; do not expose review details to any customer."
        )
        self.trace_recorder.record(
            trace_id,
            "customer_visible_content_review_result",
            {
                "approved": approved,
                "raw_approved": bool(review.get("approved")),
                "needs_human": needs_human,
                "review_scope": review_scope,
                "item_reviews": item_reviews,
                "reasoning_summary": str(review.get("reasoning_summary") or ""),
                "violations": list(review.get("violations") or []) if isinstance(review.get("violations"), list) else [],
            },
            level="WARN" if not approved else "INFO",
        )
        return ToolResult(
            name=CUSTOMER_VISIBLE_CONTENT_REVIEW_TOOL_NAME,
            called=True,
            allowed=True,
            result={
                "approved": approved,
                "raw_approved": bool(review.get("approved")),
                "needs_human": needs_human,
                "review_scope": review_scope,
                "items": review_items,
                "item_reviews": item_reviews,
                "reasoning_summary": str(review.get("reasoning_summary") or ""),
                "violations": list(review.get("violations") or []) if isinstance(review.get("violations"), list) else [],
                "instruction": instruction,
            },
        )


def parse_action(raw_response: str) -> tuple[AgentAction, list[str]]:
    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        return contract_error_action(), [f"response is not valid JSON: {exc.msg}"]
    if not isinstance(payload, dict):
        return contract_error_action(), ["response JSON root must be object"]
    errors = validate_action_contract(payload)
    if errors:
        return contract_error_action(), errors
    return AgentAction.from_payload(payload), []


def customer_visible_content_review_approved(result: ToolResult) -> bool:
    return (
        result.name == CUSTOMER_VISIBLE_CONTENT_REVIEW_TOOL_NAME
        and result.called
        and result.allowed
        and result.error is None
        and bool(result.result.get("approved"))
    )


def customer_visible_items_for_action(action: AgentAction) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for call_index, call in enumerate(action.tool_calls, start=1):
        if call.name == "create_invite_drafts":
            for item_index, raw in enumerate(call.arguments.get("invitations") or [], start=1):
                if not isinstance(raw, dict):
                    continue
                text = str(raw.get("message_text") or "").strip()
                if not text:
                    continue
                items.append(
                    {
                        "item_id": f"tool_calls[{call_index}].arguments.invitations[{item_index}].message_text",
                        "source": "create_invite_drafts",
                        "recipient_id": str(raw.get("customer_id") or ""),
                        "recipient_name": str(raw.get("display_name") or raw.get("customer_id") or ""),
                        "text": text,
                    }
                )
        if call.name == "create_outbound_message_drafts":
            for item_index, raw in enumerate(call.arguments.get("drafts") or [], start=1):
                if not isinstance(raw, dict):
                    continue
                text = str(raw.get("message_text") or "").strip()
                if not text:
                    continue
                items.append(
                    {
                        "item_id": f"tool_calls[{call_index}].arguments.drafts[{item_index}].message_text",
                        "source": "create_outbound_message_drafts",
                        "recipient_id": str(raw.get("recipient_id") or ""),
                        "recipient_name": str(raw.get("recipient_name") or raw.get("recipient_id") or ""),
                        "channel": str(raw.get("channel") or ""),
                        "purpose": str(raw.get("purpose") or ""),
                        "text": text,
                    }
                )
    return items


def normalize_item_reviews(review: dict[str, Any], review_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw_reviews = review.get("item_reviews")
    if not isinstance(raw_reviews, list):
        legacy_safe_text = str(review.get("final_reply") or "").strip()
        return [
            {
                "item_id": str(item.get("item_id") or ""),
                "approved": bool(review.get("approved")) and (not legacy_safe_text or legacy_safe_text == str(item.get("text") or "")),
                "suggested_safe_text": legacy_safe_text or str(item.get("text") or ""),
                "reasoning_summary": str(review.get("reasoning_summary") or ""),
                "violations": list(review.get("violations") or []) if isinstance(review.get("violations"), list) else [],
            }
            for item in review_items
        ]
    by_item_id = {str(item.get("item_id") or ""): item for item in raw_reviews if isinstance(item, dict)}
    normalized: list[dict[str, Any]] = []
    for item in review_items:
        item_id = str(item.get("item_id") or "")
        raw = by_item_id.get(item_id, {})
        suggested = str(raw.get("suggested_safe_text") or raw.get("safe_text") or raw.get("final_text") or item.get("text") or "")
        violations = raw.get("violations")
        normalized.append(
            {
                "item_id": item_id,
                "approved": bool(raw.get("approved")),
                "suggested_safe_text": suggested,
                "reasoning_summary": str(raw.get("reasoning_summary") or ""),
                "violations": list(violations) if isinstance(violations, list) else [],
            }
        )
    return normalized


def item_reviews_approved(item_reviews: list[dict[str, Any]], review_items: list[dict[str, Any]]) -> bool:
    original_by_id = {str(item.get("item_id") or ""): str(item.get("text") or "") for item in review_items}
    if len(item_reviews) != len(review_items):
        return False
    for item in item_reviews:
        item_id = str(item.get("item_id") or "")
        if item_id not in original_by_id:
            return False
        if not bool(item.get("approved")):
            return False
        if str(item.get("suggested_safe_text") or "") != original_by_id[item_id]:
            return False
    return True


def validate_action_contract(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in ["goal", "objective_status", "reasoning_summary", "reply_to_user", "tool_calls", "needs_human", "stop_reason"]:
        if key not in payload:
            errors.append(f"missing required key: {key}")
    for key in ["goal", "objective_status", "reasoning_summary", "reply_to_user"]:
        if key in payload and not isinstance(payload.get(key), str):
            errors.append(f"{key} must be string")
    if "needs_human" in payload and not isinstance(payload.get("needs_human"), bool):
        errors.append("needs_human must be boolean")
    stop_reason = payload.get("stop_reason")
    if not isinstance(stop_reason, dict):
        errors.append("stop_reason must be object")
        stop_reason = {}
    errors.extend(validate_stop_reason_contract(stop_reason, payload.get("objective_status")))
    if "badcase" in payload and payload.get("badcase") is not None:
        errors.append("badcase side-channel is not allowed; call record_badcase tool instead")
    if payload.get("objective_status") not in {"needs_tool", "waiting_user", "completed", "needs_human", "unknown"}:
        errors.append("objective_status is invalid")
    if not isinstance(payload.get("tool_calls", []), list):
        errors.append("tool_calls must be array")
    for index, call in enumerate(payload.get("tool_calls") or [], start=1):
        if not isinstance(call, dict):
            errors.append(f"tool_calls[{index}] must be object")
            continue
        if not isinstance(call.get("name"), str) or not call.get("name"):
            errors.append(f"tool_calls[{index}].name is required")
        if "arguments" not in call:
            errors.append(f"tool_calls[{index}].arguments is required")
        elif not isinstance(call.get("arguments"), dict):
            errors.append(f"tool_calls[{index}].arguments must be object")
        if not isinstance(call.get("reason"), str) or not call.get("reason", "").strip():
            errors.append(f"tool_calls[{index}].reason is required")
        if "idempotency_key" in call and call.get("idempotency_key") is not None and not isinstance(call.get("idempotency_key"), str):
            errors.append(f"tool_calls[{index}].idempotency_key must be string or null")
    status = payload.get("objective_status")
    tool_calls = payload.get("tool_calls") or []
    reply = payload.get("reply_to_user")
    terminal_statuses = {"waiting_user", "completed", "needs_human", "unknown"}
    if status == "needs_tool" and not tool_calls:
        errors.append("needs_tool requires at least one tool_call")
    if status == "needs_tool" and isinstance(reply, str) and reply.strip():
        errors.append("needs_tool requires empty reply_to_user")
    if status in terminal_statuses and tool_calls:
        errors.append(f"{status} must not include tool_calls")
    if status in terminal_statuses and isinstance(reply, str) and not reply.strip():
        errors.append(f"{status} requires non-empty reply_to_user")
    if status == "needs_human" and payload.get("needs_human") is not True:
        errors.append("needs_human objective_status requires needs_human=true")
    if payload.get("needs_human") is True and status != "needs_human":
        errors.append("needs_human=true requires objective_status=needs_human")
    return errors


def validate_stop_reason_contract(stop_reason: dict[str, Any], status: Any) -> list[str]:
    errors: list[str] = []
    for key in ["can_stop", "why", "pending_work", "depends_on_tool_results"]:
        if key not in stop_reason:
            errors.append(f"stop_reason.{key} is required")
    can_stop = stop_reason.get("can_stop")
    if "can_stop" in stop_reason and not isinstance(can_stop, bool):
        errors.append("stop_reason.can_stop must be boolean")
    why = stop_reason.get("why")
    if "why" in stop_reason and (not isinstance(why, str) or not why.strip()):
        errors.append("stop_reason.why must be non-empty string")
    pending_work = stop_reason.get("pending_work")
    if "pending_work" in stop_reason and not isinstance(pending_work, list):
        errors.append("stop_reason.pending_work must be array")
        pending_work = []
    if isinstance(pending_work, list) and any(not isinstance(item, str) or not item.strip() for item in pending_work):
        errors.append("stop_reason.pending_work items must be non-empty strings")
    depends_on_tool_results = stop_reason.get("depends_on_tool_results")
    if "depends_on_tool_results" in stop_reason and not isinstance(depends_on_tool_results, bool):
        errors.append("stop_reason.depends_on_tool_results must be boolean")
    if status == "needs_tool":
        if can_stop is not False:
            errors.append("needs_tool requires stop_reason.can_stop=false")
        if isinstance(pending_work, list) and not pending_work:
            errors.append("needs_tool requires non-empty stop_reason.pending_work")
    if status in {"waiting_user", "completed", "needs_human", "unknown"} and can_stop is not True:
        errors.append(f"{status} requires stop_reason.can_stop=true")
    return errors


def build_reply_self_review_payload(
    *,
    message: UserMessage,
    action: AgentAction,
    review_items: list[dict[str, Any]],
    context_payload: dict[str, Any],
    review_scope: str,
) -> dict[str, Any]:
    return {
        "current_message": message.to_dict(),
        "sender_profile": context_payload.get("sender_profile"),
        "active_games": context_payload.get("active_games") or [],
        "previous_tool_results": context_payload.get("previous_tool_results") or [],
        "recent_conversation_tail": list(context_payload.get("recent_conversation") or [])[-8:],
        "proposed_action": action.to_dict(),
        "review_scope": review_scope,
        "review_items": review_items,
        "review_goal": "只审查 review_items 中的客户可见文本是否泄露系统信息、后台流程、其他用户信息或未发生动作；不负责润色文风。",
        "review_contract": {
            "format": "json_object",
            "required_keys": ["approved", "needs_human", "reasoning_summary", "violations", "item_reviews"],
            "field_types": {
                "approved": "boolean",
                "needs_human": "boolean",
                "reasoning_summary": "string",
                "violations": "array of string labels",
                "item_reviews": "array; one item per review_items entry",
                "item_reviews[].item_id": "must equal original review_items[].item_id",
                "item_reviews[].approved": "boolean",
                "item_reviews[].suggested_safe_text": "string; same as original text when approved=true, safe rewrite when approved=false",
            },
            "invariants": [
                "approved=true means every review item has no information leakage or unverified external action",
                "approved=true requires every item_reviews[].approved=true and suggested_safe_text equals the original text",
                "approved=false and needs_human=false means at least one item_reviews entry contains a safe rewrite",
                "needs_human=true means the main agent should use a short human-handoff reply or stop the unsafe tool action",
                "suggested_safe_text must not expose tool names, JSON, traceId, internal process, draft/approval state, or candidate identities",
            ],
            "available_tools": [],
        },
    }


def parse_reply_self_review(raw_response: str) -> tuple[dict[str, Any], list[str]]:
    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        return {}, [f"reply self review is not valid JSON: {exc.msg}"]
    if not isinstance(payload, dict):
        return {}, ["reply self review JSON root must be object"]
    errors: list[str] = []
    for key in ["approved", "needs_human", "reasoning_summary", "violations"]:
        if key not in payload:
            errors.append(f"missing required review key: {key}")
    if "approved" in payload and not isinstance(payload.get("approved"), bool):
        errors.append("approved must be boolean")
    if "needs_human" in payload and not isinstance(payload.get("needs_human"), bool):
        errors.append("needs_human must be boolean")
    if "reasoning_summary" in payload and not isinstance(payload.get("reasoning_summary"), str):
        errors.append("reasoning_summary must be string")
    if "violations" in payload and not isinstance(payload.get("violations"), list):
        errors.append("violations must be array")
    if "item_reviews" in payload and not isinstance(payload.get("item_reviews"), list):
        errors.append("item_reviews must be array")
    if "item_reviews" in payload and isinstance(payload.get("item_reviews"), list):
        for index, item in enumerate(payload.get("item_reviews") or [], start=1):
            if not isinstance(item, dict):
                errors.append(f"item_reviews[{index}] must be object")
                continue
            if not isinstance(item.get("item_id"), str) or not item.get("item_id"):
                errors.append(f"item_reviews[{index}].item_id is required")
            if "approved" in item and not isinstance(item.get("approved"), bool):
                errors.append(f"item_reviews[{index}].approved must be boolean")
            if not isinstance(item.get("suggested_safe_text"), str):
                errors.append(f"item_reviews[{index}].suggested_safe_text must be string")
    return payload, errors


def contract_error_action() -> AgentAction:
    return AgentAction(
        goal="contract_error",
        objective_status="needs_human",
        reasoning_summary="模型输出不符合 AgentAction 合同，后端拒绝执行。",
        reply_to_user="这个我先转人工确认一下。",
        needs_human=True,
        stop_reason={
            "can_stop": True,
            "why": "模型输出合同错误，后端不能安全继续执行。",
            "pending_work": [],
            "depends_on_tool_results": False,
        },
    )


def message_idempotency_key(message: UserMessage) -> str:
    return f"conversation:{message.conversation_id}:sender:{message.sender_id}:message:{message.message_id}"
