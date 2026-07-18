from __future__ import annotations

"""Action processing and tool execution services for the agent loop."""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .action_contract import parse_action
from .budget import TokenBudget
from .copywriting import DEFAULT_CUSTOMER_VISIBLE_TEXT_PROMPT_PATH, action_with_customer_visible_rewrites
from .hooks import HookManager
from .llm import AgentLLMClient
from .models import AgentAction, ToolResult, UserMessage
from .runtime_components import ActionProcessingResult, ModelActionStep, TurnBudgets
from .store import InMemoryAgentStore
from .tool_consistency import latest_read_requirement, validate_tool_call_consistency
from .tools import ToolGateway
from .visibility import (
    DEFAULT_REPLY_SELF_REVIEW_PROMPT_PATH,
    CustomerVisibleProcessor,
    customer_visible_content_review_approved,
    customer_visible_items_for_action,
)


def input_batch_run_is_stale(store: Any, message: UserMessage) -> bool:
    """Compare a running aggregate with the latest durable fragment batch.

    The check is generic concurrency control: a newly arrived fragment advances
    the batch version, so an older run may still read but cannot write or send.
    Messages that did not come through the aggregation layer are unaffected.
    """

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
    """Execute model-proposed tool calls behind production boundaries."""

    store: InMemoryAgentStore
    tool_gateway: ToolGateway
    trace_recorder: Any
    hook_manager: HookManager | None = None

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
    ) -> ActionProcessingResult:
        """Execute tools sequentially and append tool results to short-term memory."""

        tool_results: list[ToolResult] = []
        pending_tool_results: list[ToolResult] = []
        blocked_by_consistency = False
        blocked_by_stale_run = False
        for call_index, call in enumerate(action.tool_calls, start=1):
            consistency_error = validate_tool_call_consistency(
                call,
                previous_step_tool_results + pending_tool_results,
            )
            if consistency_error:
                reference_requirement = latest_read_requirement(
                    previous_step_tool_results + pending_tool_results,
                    tool_name="search_current_games",
                )
                result = ToolResult(
                    name=call.name,
                    called=False,
                    allowed=False,
                    result={
                        "instruction": (
                            "Fix the tool arguments and call the tool again. Preserve explicit requirement fields "
                            "from previous read-only tool results unless the user has clearly changed them."
                        ),
                        "call": call.to_dict(),
                        "reference_tool_name": "search_current_games",
                        "reference_requirement": reference_requirement or {},
                    },
                    error=consistency_error,
                )
                tool_results.append(result)
                pending_tool_results.append(result)
                self.trace_recorder.record(
                    trace_id,
                    "tool_argument_consistency_error",
                    {"call": call.to_dict(), "error": consistency_error, "step_index": step_index},
                    level="WARN",
                )
                self.trace_recorder.record(trace_id, "tool_result", result.to_dict(), level="WARN")
                blocked_by_consistency = True
                break

            stale_result = self.stale_write_tool_result(
                call_name=call.name,
                message=message,
                run_id=run_id,
                run_version=run_version,
            )
            if stale_result is not None:
                tool_results.append(stale_result)
                pending_tool_results.append(stale_result)
                self.trace_recorder.record(trace_id, "conversation_run_stale", stale_result.to_dict(), level="WARN")
                self.trace_recorder.record(trace_id, "tool_result", stale_result.to_dict(), level="WARN")
                blocked_by_stale_run = True
                break

            self._emit(
                "before_tool_execute",
                trace_id=trace_id,
                payload={"call": call.to_dict(), "step_index": step_index, "call_index": call_index},
            )
            self.trace_recorder.record(
                trace_id,
                "tool_called",
                {"call": call.to_dict(), "step_index": step_index},
            )
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
            self._emit(
                "after_tool_execute",
                trace_id=trace_id,
                payload={"result": result.to_dict(), "step_index": step_index, "call_index": call_index},
            )
            for transition in result.state_transitions:
                step = "state_transition_replayed" if result.deduplicated else "state_transition"
                self.trace_recorder.record(trace_id, step, transition.to_dict())

        self.store.append_tool_turn(
            message.conversation_id,
            json.dumps([item.to_dict() for item in pending_tool_results], ensure_ascii=False),
            trace_id,
        )
        if blocked_by_stale_run:
            final_reply = ""
            self.trace_recorder.record(
                trace_id,
                "final_output",
                {
                    "reply": final_reply,
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
                final_reply=final_reply,
                stop_loop=True,
            )
        if blocked_by_consistency:
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

    def stale_write_tool_result(
        self,
        *,
        call_name: str,
        message: UserMessage,
        run_id: str,
        run_version: int,
    ) -> ToolResult | None:
        """Reject writes from a superseded conversation or input-batch version."""

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


@dataclass(slots=True)
class ActionProcessor:
    """Validate model actions and route them to tools or final replies."""

    llm_client: AgentLLMClient
    store: InMemoryAgentStore
    trace_recorder: Any
    tool_execution_service: ToolExecutionService
    llm_timeout_seconds: float = 45.0
    customer_visible_text_generation_enabled: bool = False
    customer_visible_text_generation_client: AgentLLMClient | None = None
    customer_visible_text_generation_prompt_path: Path = DEFAULT_CUSTOMER_VISIBLE_TEXT_PROMPT_PATH
    reply_self_review_enabled: bool = False
    reply_self_review_client: AgentLLMClient | None = None
    reply_self_review_prompt_path: Path = DEFAULT_REPLY_SELF_REVIEW_PROMPT_PATH
    hook_manager: HookManager | None = None

    def call_agent_action(
        self,
        message: UserMessage,
        *,
        trace_id: str,
        built_messages: list[dict[str, str]],
        step_index: int,
        budget: TokenBudget,
        run_id: str,
        run_version: int,
    ) -> ModelActionStep:
        """Call the main model and parse its response into an AgentAction."""

        self._emit(
            "before_llm_call",
            trace_id=trace_id,
            payload={"step_index": step_index, "message": message.to_dict(), "run_id": run_id},
        )
        budget_decision = budget.reserve(built_messages)
        self.trace_recorder.record(trace_id, "budget_checked", budget_decision.to_dict())
        if not budget_decision.allowed:
            final_reply = "这个我先转人工确认一下。"
            self.trace_recorder.record(
                trace_id,
                "final_output",
                {"reply": final_reply, "reason": budget_decision.reason},
                level="WARN",
            )
            self.append_pending_assistant_turn(
                message.conversation_id,
                final_reply,
                trace_id,
                run_id=run_id,
                run_version=run_version,
            )
            return ModelActionStep(action=None, final_reply=final_reply, stop_loop=True)

        started = time.perf_counter()
        try:
            raw_response = self.llm_client.complete(
                built_messages,
                trace_id=trace_id,
                timeout_seconds=self.llm_timeout_seconds,
            )
        except Exception as exc:
            final_reply = "这个我先转人工确认一下。"
            self.trace_recorder.record(
                trace_id,
                "llm_error",
                {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "elapsed_ms": int((time.perf_counter() - started) * 1000),
                },
                level="ERROR",
            )
            self.trace_recorder.record(
                trace_id,
                "final_output",
                {"reply": final_reply, "reason": "llm_error"},
                level="WARN",
            )
            self.append_pending_assistant_turn(
                message.conversation_id,
                final_reply,
                trace_id,
                run_id=run_id,
                run_version=run_version,
            )
            return ModelActionStep(action=None, final_reply=final_reply, stop_loop=True)

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        self.trace_recorder.record(
            trace_id,
            "llm_response",
            {"content": raw_response, "elapsed_ms": elapsed_ms, "step_index": step_index},
        )
        self._emit(
            "after_llm_response",
            trace_id=trace_id,
            payload={"step_index": step_index, "elapsed_ms": elapsed_ms, "raw_response": raw_response},
        )
        action, errors = parse_action(raw_response)
        return ModelActionStep(action=action, raw_response=raw_response, errors=errors)

    def record_action_contract_feedback(
        self,
        message: UserMessage,
        *,
        trace_id: str,
        raw_response: str,
        errors: list[str],
        step_index: int,
    ) -> list[ToolResult]:
        """Feed AgentAction contract errors back to the model as a virtual tool result."""

        self.trace_recorder.record(
            trace_id,
            "action_contract_error",
            {"errors": errors, "step_index": step_index},
            level="WARN",
        )
        feedback = ToolResult(
            name="agent_action_contract",
            called=False,
            allowed=False,
            result={
                "errors": list(errors),
                "raw_response": raw_response,
                "instruction": (
                    "Fix the AgentAction JSON contract. If waiting for user, use objective_status=waiting_user "
                    "with non-empty reply_to_user. If tools are needed, use objective_status=needs_tool with at least one tool_call."
                ),
            },
            error="AgentAction contract invalid: " + "; ".join(errors),
        )
        self.trace_recorder.record(trace_id, "contract_error_feedback", feedback.to_dict(), level="WARN")
        self.store.append_tool_turn(message.conversation_id, json.dumps([feedback.to_dict()], ensure_ascii=False), trace_id)
        return [feedback]

    def trace_action_plan(
        self,
        action: AgentAction,
        *,
        trace_id: str,
        step_index: int,
        previous_tool_result_count: int,
    ) -> None:
        self.trace_recorder.record(trace_id, "action_proposed", action.to_dict())
        self.trace_recorder.record(
            trace_id,
            "objective_plan_proposed",
            {
                "step_index": step_index,
                "goal": action.goal,
                "objective_status": action.objective_status,
                "objective_state": dict(action.objective_state),
                "objective_plan": [dict(item) for item in action.objective_plan],
                "plan_revision_reason": action.plan_revision_reason,
                "previous_tool_result_count": previous_tool_result_count,
                "tool_call_names": [call.name for call in action.tool_calls],
            },
        )
        self._emit(
            "after_action_proposed",
            trace_id=trace_id,
            payload={"step_index": step_index, "action": action.to_dict()},
        )

    def process_action(
        self,
        action: AgentAction,
        *,
        message: UserMessage,
        trace_id: str,
        context_payload: dict[str, Any],
        previous_pending_tool_results: list[ToolResult],
        step_index: int,
        budgets: TurnBudgets,
        run_id: str,
        run_version: int,
    ) -> ActionProcessingResult:
        if action.tool_calls:
            return self.process_tool_action(
                action,
                message=message,
                trace_id=trace_id,
                context_payload=context_payload,
                previous_pending_tool_results=previous_pending_tool_results,
                step_index=step_index,
                budgets=budgets,
                run_id=run_id,
                run_version=run_version,
            )
        return self.process_reply_action(
            action,
            message=message,
            trace_id=trace_id,
            context_payload=context_payload,
            budgets=budgets,
            run_id=run_id,
            run_version=run_version,
        )

    def process_tool_action(
        self,
        action: AgentAction,
        *,
        message: UserMessage,
        trace_id: str,
        context_payload: dict[str, Any],
        previous_pending_tool_results: list[ToolResult],
        step_index: int,
        budgets: TurnBudgets,
        run_id: str,
        run_version: int,
    ) -> ActionProcessingResult:
        processor = self.customer_visible_processor()
        collected_results: list[ToolResult] = []
        review_items = customer_visible_items_for_action(action)
        text_generation_result = processor.run_text_generation(
            message=message,
            trace_id=trace_id,
            action=action,
            items=review_items,
            context_payload=context_payload,
            turn_budget=budgets.text_generation,
            generation_scope="tool_calls",
        )
        if text_generation_result is not None:
            collected_results.append(text_generation_result)
            action = self.apply_customer_visible_rewrites(action, text_generation_result, trace_id=trace_id)
            review_items = customer_visible_items_for_action(action)

        review_result = processor.run_content_review(
            message=message,
            trace_id=trace_id,
            action=action,
            review_items=review_items,
            context_payload=context_payload,
            turn_budget=budgets.review,
            review_scope="tool_calls",
        )
        if review_result is not None:
            collected_results.append(review_result)
            self.trace_recorder.record(trace_id, "tool_result", review_result.to_dict())
            self.store.append_tool_turn(message.conversation_id, json.dumps([review_result.to_dict()], ensure_ascii=False), trace_id)
            if not customer_visible_content_review_approved(review_result):
                return ActionProcessingResult(
                    action=action,
                    tool_results=collected_results,
                    pending_tool_results=[review_result],
                    continue_loop=True,
                )
            action = self.attach_content_review_proof(action, trace_id=trace_id)

        execution = self.tool_execution_service.execute_tool_calls(
            action,
            message=message,
            trace_id=trace_id,
            previous_step_tool_results=list(previous_pending_tool_results),
            step_index=step_index,
            run_id=run_id,
            run_version=run_version,
        )
        collected_results.extend(execution.tool_results)
        if execution.stop_loop:
            return ActionProcessingResult(
                action=action,
                tool_results=collected_results,
                pending_tool_results=execution.pending_tool_results,
                final_reply=execution.final_reply,
                stop_loop=True,
            )
        return ActionProcessingResult(
            action=action,
            tool_results=collected_results,
            pending_tool_results=execution.pending_tool_results,
            continue_loop=True,
        )

    @staticmethod
    def attach_content_review_proof(action: AgentAction, *, trace_id: str) -> AgentAction:
        """Stamp reviewed customer-visible drafts with backend-owned evidence.

        The model cannot grant approval to itself. This marker is added only
        after the independent review contract has approved every visible item,
        and the delivery endpoint requires it before any external send.
        """

        payload = action.to_dict()
        for call in payload.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            arguments = call.get("arguments")
            if not isinstance(arguments, dict):
                continue
            item_key = {
                "create_invite_drafts": "invitations",
                "create_outbound_message_drafts": "drafts",
            }.get(str(call.get("name") or ""))
            if not item_key:
                continue
            for item in arguments.get(item_key) or []:
                if not isinstance(item, dict):
                    continue
                metadata = dict(item.get("metadata") or {}) if isinstance(item.get("metadata"), dict) else {}
                metadata.update(
                    {
                        "content_review_approved": True,
                        "content_review_trace_id": trace_id,
                    }
                )
                item["metadata"] = metadata
        return AgentAction.from_payload(payload)

    def process_reply_action(
        self,
        action: AgentAction,
        *,
        message: UserMessage,
        trace_id: str,
        context_payload: dict[str, Any],
        budgets: TurnBudgets,
        run_id: str,
        run_version: int,
    ) -> ActionProcessingResult:
        processor = self.customer_visible_processor()
        collected_results: list[ToolResult] = []
        proposed_reply = action.reply_to_user.strip()
        if action.needs_human and not proposed_reply:
            proposed_reply = "这个我先转人工确认一下。"

        review_item = {
            "item_id": "reply_to_user",
            "source": "reply_to_user",
            "recipient_id": message.sender_id,
            "recipient_name": message.sender_name,
            "text": proposed_reply,
        }
        text_generation_result = processor.run_text_generation(
            message=message,
            trace_id=trace_id,
            action=action,
            items=[review_item],
            context_payload=context_payload,
            turn_budget=budgets.text_generation,
            generation_scope="reply_to_user",
        )
        if text_generation_result is not None:
            collected_results.append(text_generation_result)
            rewrites = self.customer_visible_rewrites(text_generation_result)
            if rewrites.get("reply_to_user"):
                proposed_reply = rewrites["reply_to_user"].strip()
                action = action_with_customer_visible_rewrites(action, rewrites)
                self.trace_recorder.record(trace_id, "action_after_customer_visible_text_generation", action.to_dict())
                review_item = {**review_item, "text": proposed_reply}

        review_result = processor.run_content_review(
            message=message,
            trace_id=trace_id,
            action=action,
            review_items=[review_item],
            context_payload=context_payload,
            turn_budget=budgets.review,
            review_scope="reply_to_user",
        )
        if review_result is not None:
            collected_results.append(review_result)
            self.trace_recorder.record(trace_id, "tool_result", review_result.to_dict())
            self.store.append_tool_turn(message.conversation_id, json.dumps([review_result.to_dict()], ensure_ascii=False), trace_id)
            if not customer_visible_content_review_approved(review_result):
                return ActionProcessingResult(
                    action=action,
                    tool_results=collected_results,
                    pending_tool_results=[review_result],
                    continue_loop=True,
                )

        if self.run_is_stale(message, run_version):
            final_reply = ""
            self.trace_recorder.record(
                trace_id,
                "conversation_run_stale",
                {
                    "run_id": run_id,
                    "run_version": run_version,
                    "current_version": self.store.conversation_version(message.conversation_id),
                    "blocked": "final_reply",
                },
                level="WARN",
            )
            self.trace_recorder.record(
                trace_id,
                "final_output",
                {
                    "reply": final_reply,
                    "reason": "conversation_run_stale",
                    "run_id": run_id,
                    "run_version": run_version,
                    "current_version": self.store.conversation_version(message.conversation_id),
                },
                level="WARN",
            )
            return ActionProcessingResult(
                action=action,
                tool_results=collected_results,
                final_reply=final_reply,
                stop_loop=True,
            )

        self.append_pending_assistant_turn(
            message.conversation_id,
            proposed_reply,
            trace_id,
            run_id=run_id,
            run_version=run_version,
        )
        self.trace_recorder.record(
            trace_id,
            "final_output",
            {"reply": proposed_reply, "objective_status": action.objective_status},
        )
        self._emit(
            "before_reply_send",
            trace_id=trace_id,
            payload={"reply": proposed_reply, "action": action.to_dict()},
        )
        return ActionProcessingResult(
            action=action,
            tool_results=collected_results,
            final_reply=proposed_reply,
            stop_loop=True,
        )

    def apply_customer_visible_rewrites(self, action: AgentAction, result: ToolResult, *, trace_id: str) -> AgentAction:
        rewrites = self.customer_visible_rewrites(result)
        if not rewrites:
            return action
        rewritten = action_with_customer_visible_rewrites(action, rewrites)
        self.trace_recorder.record(trace_id, "action_after_customer_visible_text_generation", rewritten.to_dict())
        return rewritten

    @staticmethod
    def customer_visible_rewrites(result: ToolResult) -> dict[str, str]:
        return {
            str(item.get("item_id") or ""): str(item.get("final_text") or "")
            for item in result.result.get("item_rewrites", [])
            if isinstance(item, dict)
        }

    def run_is_stale(self, message: UserMessage, run_version: int) -> bool:
        return (
            self.store.conversation_version(message.conversation_id) != int(run_version)
            or input_batch_run_is_stale(self.store, message)
        )

    def append_pending_assistant_turn(
        self,
        conversation_id: str,
        text: str,
        trace_id: str,
        *,
        run_id: str,
        run_version: int,
    ) -> None:
        self.store.append_assistant_turn(
            conversation_id,
            text,
            trace_id,
            metadata={
                "delivery_status": "pending_operator_send",
                "run_id": run_id,
                "conversation_version": run_version,
            },
        )

    def customer_visible_processor(self) -> CustomerVisibleProcessor:
        return CustomerVisibleProcessor(
            llm_client=self.llm_client,
            trace_recorder=self.trace_recorder,
            timeout_seconds=self.llm_timeout_seconds,
            text_generation_enabled=self.customer_visible_text_generation_enabled,
            text_generation_client=self.customer_visible_text_generation_client,
            text_generation_prompt_path=self.customer_visible_text_generation_prompt_path,
            review_enabled=self.reply_self_review_enabled,
            review_client=self.reply_self_review_client,
            review_prompt_path=self.reply_self_review_prompt_path,
        )

    def _emit(self, event_name: str, *, trace_id: str, payload: dict[str, Any]) -> None:
        if self.hook_manager is not None:
            self.hook_manager.emit(event_name, trace_id=trace_id, payload=payload)
