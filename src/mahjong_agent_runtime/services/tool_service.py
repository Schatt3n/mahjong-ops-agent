from __future__ import annotations

"""Trusted execution boundary for model-proposed tool calls."""

import json
from dataclasses import dataclass, field
from typing import Any

from ..hooks import HookManager
from ..models import AgentAction, ToolCall, ToolResult, UserMessage
from ..runtime_components import ActionProcessingResult
from ..stores import AgentStore
from ..tool_consistency import (
    latest_read_requirement,
    validate_explicit_task_fact_consistency,
    validate_tool_call_consistency,
)
from ..tools import ToolGateway
from .contracts import SingleToolExecution
from .tool_scheduler import ToolCallScheduler


def input_batch_run_is_stale(store: Any, message: UserMessage) -> bool:
    """Return whether a newer fragment batch superseded this run."""

    metadata = message.metadata if isinstance(message.metadata, dict) else {}
    window = metadata.get("input_window") if isinstance(metadata.get("input_window"), dict) else {}
    batch_id = str(window.get("batch_id") or "")
    try:
        batch_version = int(window.get("batch_version"))
    except (TypeError, ValueError):
        return False
    if not batch_id:
        return False
    current = store.pending_input_batch(message.conversation_id, message.sender_id)
    return current is None or current.batch_id != batch_id or current.version != batch_version


@dataclass(slots=True)
class ToolExecutionService:
    """Validate, execute, persist, and trace tool calls proposed by the model."""

    store: AgentStore
    tool_gateway: ToolGateway
    trace_recorder: Any
    hook_manager: HookManager | None = None
    max_parallel_read_tools: int = 4
    _scheduler: ToolCallScheduler = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.max_parallel_read_tools = max(1, int(self.max_parallel_read_tools))
        self._scheduler = ToolCallScheduler(
            tool_gateway=self.tool_gateway,
            trace_recorder=self.trace_recorder,
            max_parallel_read_tools=self.max_parallel_read_tools,
        )

    def execute_tool_calls(
        self,
        action: AgentAction,
        *,
        message: UserMessage,
        trace_id: str,
        previous_step_tool_results: list[ToolResult],
        step_index: int,
        run_id: str,
        run_version: int,
        context_payload: dict[str, Any] | None = None,
    ) -> ActionProcessingResult:
        """Execute calls, append ordered observations, and return loop signals."""

        results_by_index, consistency_blocked, stale_blocked = self._scheduler.execute(
            action,
            execute_one=self._execute_one,
            message=message,
            trace_id=trace_id,
            previous_step_tool_results=previous_step_tool_results,
            step_index=step_index,
            run_id=run_id,
            run_version=run_version,
            context_payload=context_payload,
        )
        tool_results = [results_by_index[index] for index in sorted(results_by_index)]
        pending_tool_results = list(tool_results)
        self.store.append_tool_turn(
            message.conversation_id,
            json.dumps([item.to_dict() for item in pending_tool_results], ensure_ascii=False),
            trace_id,
        )
        if stale_blocked:
            self.trace_recorder.record(
                trace_id,
                "final_output",
                {
                    "reply": "",
                    "reason": "conversation_run_stale",
                    "run_id": run_id,
                    "run_version": run_version,
                    "current_version": self.store.conversation_version(message.conversation_id),
                },
                level="WARN",
            )
            return ActionProcessingResult(
                action=action,
                tool_results=tool_results,
                pending_tool_results=pending_tool_results,
                final_reply="",
                stop_loop=True,
            )
        if consistency_blocked:
            self.trace_recorder.record(
                trace_id,
                "tool_argument_consistency_feedback",
                {"results": [item.to_dict() for item in pending_tool_results]},
                level="WARN",
            )
        return ActionProcessingResult(
            action=action,
            tool_results=tool_results,
            pending_tool_results=pending_tool_results,
        )

    def _execute_one(
        self,
        call: ToolCall,
        *,
        call_index: int,
        call_id: str | None,
        observed_results: list[ToolResult],
        message: UserMessage,
        trace_id: str,
        step_index: int,
        run_id: str,
        run_version: int,
        context_payload: dict[str, Any] | None,
    ) -> SingleToolExecution:
        """Apply consistency and staleness guards before one gateway call."""

        explicit_fact_error, explicit_fact_reference = validate_explicit_task_fact_consistency(
            call,
            context_payload,
        )
        if explicit_fact_error:
            result = ToolResult(
                name=call.name,
                called=False,
                allowed=False,
                call_id=call_id,
                result={
                    "instruction": (
                        "Fix the tool arguments and call the tool again. Explicit facts from the current task "
                        "are authoritative until the user explicitly changes them."
                    ),
                    "call": call.to_dict(),
                    "reference_tool_name": "explicit_task_facts",
                    "reference_requirement": explicit_fact_reference,
                },
                error=explicit_fact_error,
            )
            self.trace_recorder.record(
                trace_id,
                "tool_explicit_fact_consistency_error",
                {"call": call.to_dict(), "error": explicit_fact_error, "step_index": step_index},
                level="WARN",
            )
            self.trace_recorder.record(trace_id, "tool_result", result.to_dict(), level="WARN")
            return SingleToolExecution(result=result, blocked_by_consistency=True)

        consistency_error = validate_tool_call_consistency(call, observed_results)
        if consistency_error:
            result = ToolResult(
                name=call.name,
                called=False,
                allowed=False,
                call_id=call_id,
                result={
                    "instruction": (
                        "Fix the tool arguments and call the tool again. Preserve explicit requirement fields "
                        "from previous read-only tool results unless the user has clearly changed them."
                    ),
                    "call": call.to_dict(),
                    "reference_tool_name": "search_current_games",
                    "reference_requirement": latest_read_requirement(
                        observed_results, tool_name="search_current_games"
                    ) or {},
                },
                error=consistency_error,
            )
            self.trace_recorder.record(
                trace_id,
                "tool_argument_consistency_error",
                {"call": call.to_dict(), "error": consistency_error, "step_index": step_index},
                level="WARN",
            )
            self.trace_recorder.record(trace_id, "tool_result", result.to_dict(), level="WARN")
            return SingleToolExecution(result=result, blocked_by_consistency=True)

        stale_result = self.stale_write_tool_result(
            call_name=call.name,
            message=message,
            run_id=run_id,
            run_version=run_version,
        )
        if stale_result is not None:
            stale_result.call_id = call_id
            self.trace_recorder.record(trace_id, "conversation_run_stale", stale_result.to_dict(), level="WARN")
            self.trace_recorder.record(trace_id, "tool_result", stale_result.to_dict(), level="WARN")
            return SingleToolExecution(result=stale_result, blocked_by_stale_run=True)

        self._emit(
            "before_tool_execute",
            trace_id=trace_id,
            payload={"call": call.to_dict(), "step_index": step_index, "call_index": call_index},
        )
        self.trace_recorder.record(
            trace_id,
            "tool_called",
            {"call": call.to_dict(), "step_index": step_index, "call_index": call_index},
        )
        result = self.tool_gateway.execute(
            call,
            trace_id=trace_id,
            conversation_id=message.conversation_id,
            sender_id=message.sender_id,
            sender_name=message.sender_name,
            step_index=step_index * 100 + call_index,
            source_message_id=message.message_id,
            message_reference_contract=dict((context_payload or {}).get("message_reference_contract") or {}),
        )
        result.call_id = call_id
        self.trace_recorder.record(trace_id, "tool_result", result.to_dict())
        self._emit(
            "after_tool_execute",
            trace_id=trace_id,
            payload={
                "call": call.to_dict(),
                "result": result.to_dict(),
                "conversation_id": message.conversation_id,
                "sender_id": message.sender_id,
                "source_message_id": message.message_id,
                "step_index": step_index,
                "call_index": call_index,
            },
        )
        for transition in result.state_transitions:
            event = "state_transition_replayed" if result.deduplicated else "state_transition"
            self.trace_recorder.record(trace_id, event, transition.to_dict())
        return SingleToolExecution(result=result)

    def stale_write_tool_result(
        self,
        *,
        call_name: str,
        message: UserMessage,
        run_id: str,
        run_version: int,
    ) -> ToolResult | None:
        """Reject writes from a superseded conversation or input batch."""

        definition = self.tool_gateway.tools.get(call_name) if self.tool_gateway else None
        if definition is None or definition.execution_mode not in {"state_write", "draft_write"}:
            return None
        current_version = self.store.conversation_version(message.conversation_id)
        stale_input_batch = input_batch_run_is_stale(self.store, message)
        if current_version == int(run_version) and not stale_input_batch:
            return None
        return ToolResult(
            name=call_name,
            called=False,
            allowed=False,
            result={
                "run_id": run_id,
                "run_version": run_version,
                "current_version": current_version,
                "stale_input_batch": stale_input_batch,
                "instruction": (
                    "This run is stale because a newer user fragment or conversation turn arrived. "
                    "Do not write state or create drafts from the old version; rebuild context from the latest user input."
                ),
            },
            error="stale run: input batch or conversation version changed before a state-writing tool could execute",
        )

    def _emit(self, event_name: str, *, trace_id: str, payload: dict[str, Any]) -> None:
        if self.hook_manager is not None:
            self.hook_manager.emit(event_name, trace_id=trace_id, payload=payload)


__all__ = ["ToolExecutionService", "input_batch_run_is_stale"]
