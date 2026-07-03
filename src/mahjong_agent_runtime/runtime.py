from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from .context import AgentContextBuilderV3, estimate_tokens
from .llm import AgentLLMClientV3
from .models import AgentActionV3, AgentRuntimeResultV3, ToolResultV3, UserMessageV3
from .store import InMemoryAgentStoreV3
from .tools import ToolGatewayV3
from .tracing import InMemoryTraceRecorderV3


@dataclass(slots=True)
class BudgetDecisionV3:
    allowed: bool
    reason: str
    estimated_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return {"allowed": self.allowed, "reason": self.reason, "estimated_tokens": self.estimated_tokens}


@dataclass(slots=True)
class TokenBudgetV3:
    max_tokens_per_call: int = 24_000
    max_calls_per_turn: int = 8
    calls_this_turn: int = 0

    def reserve(self, messages: list[dict[str, str]]) -> BudgetDecisionV3:
        self.calls_this_turn += 1
        estimated = sum(estimate_tokens(item.get("content", "")) for item in messages)
        if self.calls_this_turn > self.max_calls_per_turn:
            return BudgetDecisionV3(False, f"turn llm call limit exceeded: {self.max_calls_per_turn}", estimated)
        if estimated > self.max_tokens_per_call:
            return BudgetDecisionV3(False, f"single call token estimate exceeded: {estimated}>{self.max_tokens_per_call}", estimated)
        return BudgetDecisionV3(True, "budget_reserved", estimated)


@dataclass(slots=True)
class AgentRuntimeV3:
    llm_client: AgentLLMClientV3
    store: InMemoryAgentStoreV3 = field(default_factory=InMemoryAgentStoreV3)
    tool_gateway: ToolGatewayV3 | None = None
    trace_recorder: Any = field(default_factory=InMemoryTraceRecorderV3)
    token_budget: TokenBudgetV3 = field(default_factory=TokenBudgetV3)
    max_steps: int = 8
    llm_timeout_seconds: float = 45.0
    context_builder: AgentContextBuilderV3 = field(init=False)
    _conversation_locks: dict[str, threading.RLock] = field(default_factory=dict, init=False, repr=False)
    _conversation_locks_guard: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.tool_gateway is None:
            self.tool_gateway = ToolGatewayV3(self.store)
        if self.tool_gateway.trace_recorder is None:
            self.tool_gateway.trace_recorder = self.trace_recorder
        self.context_builder = AgentContextBuilderV3(self.store, self.tool_gateway)

    def handle_user_message(self, message: UserMessageV3, *, trace_id: str | None = None) -> AgentRuntimeResultV3:
        with self._conversation_lock(message.conversation_id):
            actual_trace_id = trace_id or f"trace_v3_{uuid.uuid4().hex[:12]}"
            cached = self.store.idempotent_message_result(message.message_id)
            if cached is not None:
                self.trace_recorder.record(actual_trace_id, "user_input", {"message": message.to_dict()})
                self.trace_recorder.record(actual_trace_id, "message_deduplicated", {"message_id": message.message_id, "original_trace_id": cached.trace_id})
                self.trace_recorder.record(actual_trace_id, "final_output", {"reply": cached.final_reply, "reason": "message_deduplicated"})
                return cached
            result = self._handle_once(message, trace_id=actual_trace_id)
            self.store.remember_message_result(message.message_id, result)
            return result

    def _handle_once(self, message: UserMessageV3, *, trace_id: str) -> AgentRuntimeResultV3:
        self.token_budget.calls_this_turn = 0
        self.store.append_user_turn(message, trace_id)
        self.trace_recorder.record(trace_id, "user_input", {"message": message.to_dict()})
        actions: list[AgentActionV3] = []
        tool_results: list[ToolResultV3] = []
        pending_tool_results: list[ToolResultV3] = []
        final_reply = ""

        for step_index in range(1, self.max_steps + 1):
            built = self.context_builder.build(message, trace_id=trace_id, previous_tool_results=pending_tool_results)
            self.trace_recorder.record(trace_id, "context_packed", built.audit)
            self.trace_recorder.record(trace_id, "context_built", built.payload)
            self.trace_recorder.record(trace_id, "llm_prompt", {"messages": built.messages, "step_index": step_index})
            budget = self.token_budget.reserve(built.messages)
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
                final_reply = "这个我先转人工确认一下。"
                self.trace_recorder.record(trace_id, "action_contract_error", {"errors": errors, "step_index": step_index}, level="WARN")
                self.trace_recorder.record(trace_id, "final_output", {"reply": final_reply, "reason": "contract_error"}, level="WARN")
                self.store.append_assistant_turn(message.conversation_id, final_reply, trace_id)
                break
            self.trace_recorder.record(trace_id, "action_proposed", action.to_dict())

            if action.tool_calls:
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

            final_reply = action.reply_to_user.strip()
            if action.needs_human and not final_reply:
                final_reply = "这个我先转人工确认一下。"
            self.store.append_assistant_turn(message.conversation_id, final_reply, trace_id)
            self.trace_recorder.record(trace_id, "final_output", {"reply": final_reply, "objective_status": action.objective_status})
            break
        else:
            final_reply = "这个我先转人工确认一下。"
            self.store.append_assistant_turn(message.conversation_id, final_reply, trace_id)
            self.trace_recorder.record(trace_id, "final_output", {"reply": final_reply, "reason": "max_steps_exceeded"}, level="WARN")

        transitions = [transition for result in tool_results if not result.deduplicated for transition in result.state_transitions]
        return AgentRuntimeResultV3(
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


def parse_action(raw_response: str) -> tuple[AgentActionV3, list[str]]:
    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        return contract_error_action(), [f"response is not valid JSON: {exc.msg}"]
    if not isinstance(payload, dict):
        return contract_error_action(), ["response JSON root must be object"]
    errors = validate_action_contract(payload)
    if errors:
        return contract_error_action(), errors
    return AgentActionV3.from_payload(payload), []


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


def contract_error_action() -> AgentActionV3:
    return AgentActionV3(
        goal="contract_error",
        objective_status="needs_human",
        reasoning_summary="模型输出不符合 AgentActionV3 合同，后端拒绝执行。",
        reply_to_user="这个我先转人工确认一下。",
        needs_human=True,
        stop_reason={
            "can_stop": True,
            "why": "模型输出合同错误，后端不能安全继续执行。",
            "pending_work": [],
            "depends_on_tool_results": False,
        },
    )
