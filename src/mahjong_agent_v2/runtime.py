from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from .context import ContextBuilderV2
from .llm import AgentLLMClientV2
from .models import AgentDecisionV2, AgentRuntimeResultV2, ToolResultV2, UserMessageV2
from .store import InMemoryAgentStoreV2
from .tools import ToolGatewayV2
from .tracing import InMemoryTraceRecorderV2


@dataclass(slots=True)
class BudgetDecisionV2:
    allowed: bool
    reason: str
    estimated_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "estimated_tokens": self.estimated_tokens,
        }


@dataclass(slots=True)
class TokenBudgetV2:
    max_tokens_per_call: int = 24_000
    max_calls_per_turn: int = 6
    calls_this_turn: int = 0

    def reserve(self, messages: list[dict[str, str]]) -> BudgetDecisionV2:
        self.calls_this_turn += 1
        estimated_tokens = estimate_tokens(messages)
        if self.calls_this_turn > self.max_calls_per_turn:
            return BudgetDecisionV2(
                allowed=False,
                reason=f"turn llm call limit exceeded: {self.max_calls_per_turn}",
                estimated_tokens=estimated_tokens,
            )
        if estimated_tokens > self.max_tokens_per_call:
            return BudgetDecisionV2(
                allowed=False,
                reason=f"single call token estimate exceeded: {estimated_tokens}>{self.max_tokens_per_call}",
                estimated_tokens=estimated_tokens,
            )
        return BudgetDecisionV2(allowed=True, reason="budget_reserved", estimated_tokens=estimated_tokens)


@dataclass(slots=True)
class AgentRuntimeV2:
    llm_client: AgentLLMClientV2
    store: InMemoryAgentStoreV2 = field(default_factory=InMemoryAgentStoreV2)
    tool_gateway: ToolGatewayV2 | None = None
    trace_recorder: Any = field(default_factory=InMemoryTraceRecorderV2)
    max_steps: int = 6
    llm_timeout_seconds: float = 45.0
    token_budget: TokenBudgetV2 = field(default_factory=TokenBudgetV2)
    context_builder: ContextBuilderV2 = field(init=False)
    _conversation_locks: dict[str, threading.RLock] = field(default_factory=dict, init=False, repr=False)
    _conversation_locks_guard: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.tool_gateway is None:
            self.tool_gateway = ToolGatewayV2(self.store)
        self.context_builder = ContextBuilderV2(self.store, self.tool_gateway)

    def handle_user_message(self, message: UserMessageV2, *, trace_id: str | None = None) -> AgentRuntimeResultV2:
        with self._conversation_lock(message.conversation_id):
            cached_result = self._idempotent_message_result(message.message_id)
            actual_trace_id = trace_id or f"trace_v2_{uuid.uuid4().hex[:12]}"
            if cached_result is not None:
                self.trace_recorder.record(
                    actual_trace_id,
                    "message_deduplicated",
                    {
                        "message_id": message.message_id,
                        "original_trace_id": cached_result.trace_id,
                    },
                )
                return cached_result
            result = self._handle_user_message_once(message, trace_id=actual_trace_id)
            self._remember_message_result(message.message_id, result)
            return result

    def _handle_user_message_once(self, message: UserMessageV2, *, trace_id: str | None = None) -> AgentRuntimeResultV2:
        actual_trace_id = trace_id or f"trace_v2_{uuid.uuid4().hex[:12]}"
        self.token_budget.calls_this_turn = 0
        self.trace_recorder.record(actual_trace_id, "user_input", {"message": message.to_dict()})
        self.store.append_user_turn(message, actual_trace_id)

        decisions: list[AgentDecisionV2] = []
        tool_results: list[ToolResultV2] = []
        pending_tool_results: list[ToolResultV2] = []
        final_reply = ""

        for step_index in range(1, self.max_steps + 1):
            built = self.context_builder.build(message, trace_id=actual_trace_id, previous_tool_results=pending_tool_results)
            self.trace_recorder.record(actual_trace_id, "context_packed", built.audit)
            self.trace_recorder.record(actual_trace_id, "context_built", built.payload)
            self.trace_recorder.record(actual_trace_id, "llm_prompt", {"messages": built.messages})
            budget_decision = self.token_budget.reserve(built.messages)
            self.trace_recorder.record(actual_trace_id, "budget_checked", budget_decision.to_dict())
            if not budget_decision.allowed:
                final_reply = "这个我先转人工确认一下。"
                self.trace_recorder.record(actual_trace_id, "final_output", {"reply": final_reply, "reason": budget_decision.reason})
                break

            llm_started = time.perf_counter()
            try:
                raw_response = self.llm_client.complete(
                    built.messages,
                    trace_id=actual_trace_id,
                    timeout_seconds=self.llm_timeout_seconds,
                )
            except Exception as exc:
                elapsed_ms = int((time.perf_counter() - llm_started) * 1000)
                final_reply = "这个我先转人工确认一下。"
                self.trace_recorder.record(
                    actual_trace_id,
                    "llm_error",
                    {
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "elapsed_ms": elapsed_ms,
                        "timeout_seconds": self.llm_timeout_seconds,
                        "step_index": step_index,
                    },
                    level="ERROR",
                )
                self.store.append_assistant_turn(message.conversation_id, final_reply, actual_trace_id)
                self.trace_recorder.record(
                    actual_trace_id,
                    "final_output",
                    {"reply": final_reply, "reason": "llm_error"},
                )
                break
            elapsed_ms = int((time.perf_counter() - llm_started) * 1000)
            self.trace_recorder.record(
                actual_trace_id,
                "llm_response",
                {"content": raw_response, "elapsed_ms": elapsed_ms, "step_index": step_index},
            )
            decision, contract_errors = self._parse_decision(raw_response)
            decisions.append(decision)
            if contract_errors:
                self.trace_recorder.record(
                    actual_trace_id,
                    "decision_contract_error",
                    {"errors": contract_errors, "step_index": step_index},
                    level="WARN",
                )
            self.trace_recorder.record(actual_trace_id, "action_proposed", decision.to_dict())

            if decision.badcase:
                badcase_call = {
                    "name": "record_badcase",
                    "arguments": decision.badcase,
                    "reason": "model reported badcase",
                }
                decision.tool_calls.append(self._tool_call_from_dict(badcase_call))

            if decision.tool_calls:
                pending_tool_results = []
                for call_index, call in enumerate(decision.tool_calls, start=1):
                    result = self.tool_gateway.execute(
                        call,
                        trace_id=actual_trace_id,
                        conversation_id=message.conversation_id,
                        sender_id=message.sender_id,
                        sender_name=message.sender_name,
                        step_index=(step_index * 100 + call_index),
                    )
                    tool_results.append(result)
                    pending_tool_results.append(result)
                    self.trace_recorder.record(actual_trace_id, "tool_called", {"call": call.to_dict()})
                    self.trace_recorder.record(actual_trace_id, "tool_result", result.to_dict())
                    for transition in result.state_transitions:
                        self.trace_recorder.record(actual_trace_id, "state_transition", transition.to_dict())
                self.store.append_tool_turn(
                    message.conversation_id,
                    json.dumps([result.to_dict() for result in pending_tool_results], ensure_ascii=False),
                    actual_trace_id,
                )
                continue

            final_reply = decision.reply_to_user.strip()
            if decision.needs_human and not final_reply:
                final_reply = "这个我先转人工确认一下。"
            if not final_reply:
                final_reply = "我先确认一下。"
            self.store.append_assistant_turn(message.conversation_id, final_reply, actual_trace_id)
            self.trace_recorder.record(actual_trace_id, "final_output", {"reply": final_reply})
            break
        else:
            final_reply = "我先确认一下，稍后回复你。"
            self.store.append_assistant_turn(message.conversation_id, final_reply, actual_trace_id)
            self.trace_recorder.record(actual_trace_id, "final_output", {"reply": final_reply, "reason": "max_steps_exceeded"})

        transitions = [transition for result in tool_results for transition in result.state_transitions]
        return AgentRuntimeResultV2(
            trace_id=actual_trace_id,
            final_reply=final_reply,
            decisions=decisions,
            tool_results=tool_results,
            state_transitions=transitions,
            conversation_id=message.conversation_id,
        )

    def _conversation_lock(self, conversation_id: str) -> threading.RLock:
        key = conversation_id or "default"
        with self._conversation_locks_guard:
            lock = self._conversation_locks.get(key)
            if lock is None:
                lock = threading.RLock()
                self._conversation_locks[key] = lock
            return lock

    def _idempotent_message_result(self, message_id: str | None) -> AgentRuntimeResultV2 | None:
        getter = getattr(self.store, "idempotent_message_result", None)
        if not callable(getter):
            return None
        return getter(message_id)

    def _remember_message_result(self, message_id: str | None, result: AgentRuntimeResultV2) -> None:
        remember = getattr(self.store, "remember_message_result", None)
        if callable(remember):
            remember(message_id, result)

    def _parse_decision(self, raw_response: str) -> tuple[AgentDecisionV2, list[str]]:
        try:
            payload = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            return self._contract_error_decision([f"response is not valid JSON: {exc.msg}"])
        if not isinstance(payload, dict):
            return self._contract_error_decision(["response JSON root must be object"])
        errors = validate_decision_contract(payload)
        if errors:
            return self._contract_error_decision(errors)
        return AgentDecisionV2.from_payload(payload), []

    def _contract_error_decision(self, errors: list[str]) -> tuple[AgentDecisionV2, list[str]]:
        return (
            AgentDecisionV2(
                goal="decision_contract_invalid",
                reasoning_summary="模型输出不符合 AgentDecisionV2 合同，后端拒绝执行工具。",
                reply_to_user="这个我先转人工确认一下。",
                tool_calls=[],
                needs_human=True,
            ),
            errors,
        )

    def _tool_call_from_dict(self, raw: dict[str, Any]):
        from .models import ToolCallV2

        return ToolCallV2(
            name=str(raw.get("name") or ""),
            arguments=dict(raw.get("arguments") or {}),
            idempotency_key=str(raw.get("idempotency_key")) if raw.get("idempotency_key") else None,
            reason=str(raw.get("reason") or ""),
        )


def estimate_tokens(value: Any) -> int:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True) if not isinstance(value, str) else value
    return max(1, len(text) // 4)


def validate_decision_contract(payload: dict[str, Any]) -> list[str]:
    """Validate the model-facing decision contract before any tool execution.

    This is a generic agent boundary, not mahjong semantic logic. A malformed
    decision must not be coerced into tool calls because that would let bad model
    output mutate state.
    """

    errors: list[str] = []
    required_types = {
        "goal": str,
        "reasoning_summary": str,
        "reply_to_user": str,
        "tool_calls": list,
        "needs_human": bool,
    }
    for key, expected_type in required_types.items():
        if key not in payload:
            errors.append(f"{key} is required")
            continue
        if not isinstance(payload[key], expected_type):
            errors.append(f"{key} must be {expected_type.__name__}")

    raw_calls = payload.get("tool_calls")
    if isinstance(raw_calls, list):
        for index, call in enumerate(raw_calls):
            path = f"tool_calls[{index}]"
            if not isinstance(call, dict):
                errors.append(f"{path} must be object")
                continue
            name = call.get("name", call.get("tool_name"))
            if not isinstance(name, str) or not name.strip():
                errors.append(f"{path}.name must be non-empty string")
            arguments = call.get("arguments")
            if not isinstance(arguments, dict):
                errors.append(f"{path}.arguments must be object")
            if "idempotency_key" in call and call.get("idempotency_key") is not None:
                if not isinstance(call.get("idempotency_key"), str):
                    errors.append(f"{path}.idempotency_key must be string when provided")
            if "reason" in call and not isinstance(call.get("reason"), str):
                errors.append(f"{path}.reason must be string when provided")

    if "badcase" in payload and payload.get("badcase") is not None and not isinstance(payload.get("badcase"), dict):
        errors.append("badcase must be object or null")
    return errors
