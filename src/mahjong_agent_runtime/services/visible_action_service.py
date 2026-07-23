from __future__ import annotations

"""Customer-visible generation, review, and reply persistence."""

import json
from dataclasses import dataclass
from typing import Any

from ..copywriting import action_with_customer_visible_rewrites
from ..hooks import HookManager
from ..models import AgentAction, ToolResult, UserMessage
from ..runtime_components import ActionProcessingResult, TurnBudgets
from ..stores import AgentStore
from ..visibility import (
    CustomerVisibleProcessor,
    customer_visible_content_review_approved,
    customer_visible_items_for_action,
)
from ..customer_visible_review import external_action_evidence_from_tool_results
from .tool_service import ToolExecutionService, input_batch_run_is_stale


def attach_content_review_proof(action: AgentAction, *, trace_id: str) -> AgentAction:
    """Stamp approved outbound drafts with backend-owned review evidence."""

    payload = action.to_dict()
    for call in payload.get("tool_calls") or []:
        if not isinstance(call, dict) or not isinstance(call.get("arguments"), dict):
            continue
        item_key = {
            "create_invite_drafts": "invitations",
            "create_outbound_message_drafts": "drafts",
        }.get(str(call.get("name") or ""))
        if not item_key:
            continue
        for item in call["arguments"].get(item_key) or []:
            if not isinstance(item, dict):
                continue
            metadata = dict(item.get("metadata") or {}) if isinstance(item.get("metadata"), dict) else {}
            metadata.update({"content_review_approved": True, "content_review_trace_id": trace_id})
            item["metadata"] = metadata
    return AgentAction.from_payload(payload)


def customer_visible_rewrites(result: ToolResult) -> dict[str, str]:
    """Read item-id to final-text rewrites from the generation tool result."""

    return {
        str(item.get("item_id") or ""): str(item.get("final_text") or "")
        for item in result.result.get("item_rewrites", [])
        if isinstance(item, dict)
    }


@dataclass(slots=True)
class CustomerVisibleActionService:
    """Process external text through generation, review, and durable pending-send state."""

    store: AgentStore
    trace_recorder: Any
    tool_execution_service: ToolExecutionService
    hook_manager: HookManager | None = None

    def process_tool_action(
        self,
        action: AgentAction,
        *,
        processor: CustomerVisibleProcessor,
        message: UserMessage,
        trace_id: str,
        context_payload: dict[str, Any],
        previous_pending_tool_results: list[ToolResult],
        step_index: int,
        budgets: TurnBudgets,
        run_id: str,
        run_version: int,
    ) -> ActionProcessingResult:
        """Review all customer-visible tool arguments before executing any tool."""

        collected_results: list[ToolResult] = []
        review_items = customer_visible_items_for_action(action)
        original_text = {
            str(item.get("item_id") or ""): str(item.get("text") or "") for item in review_items
        }
        generation_result = processor.run_text_generation(
            message=message,
            trace_id=trace_id,
            action=action,
            items=review_items,
            context_payload=context_payload,
            turn_budget=budgets.text_generation,
            generation_scope="tool_calls",
        )
        if generation_result is not None:
            collected_results.append(generation_result)
            action = self.apply_customer_visible_rewrites(action, generation_result, trace_id=trace_id)
            review_items = customer_visible_items_for_action(action)
        review_items = [
            {
                **item,
                "source_text": original_text.get(
                    str(item.get("item_id") or ""),
                    str(item.get("text") or ""),
                ),
            }
            for item in review_items
        ]

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
            self.store.append_tool_turn(
                message.conversation_id,
                json.dumps([review_result.to_dict()], ensure_ascii=False),
                trace_id,
            )
            if not customer_visible_content_review_approved(review_result):
                return ActionProcessingResult(
                    action=action,
                    tool_results=collected_results,
                    pending_tool_results=[review_result],
                    continue_loop=True,
                )
            action = attach_content_review_proof(action, trace_id=trace_id)

        execution = self.tool_execution_service.execute_tool_calls(
            action,
            message=message,
            trace_id=trace_id,
            previous_step_tool_results=list(previous_pending_tool_results),
            step_index=step_index,
            run_id=run_id,
            run_version=run_version,
            context_payload=context_payload,
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

    def process_reply_action(
        self,
        action: AgentAction,
        *,
        processor: CustomerVisibleProcessor,
        message: UserMessage,
        trace_id: str,
        context_payload: dict[str, Any],
        budgets: TurnBudgets,
        run_id: str,
        run_version: int,
    ) -> ActionProcessingResult:
        """Generate, review, and persist one final customer-visible reply."""

        collected_results: list[ToolResult] = []
        proposed_reply = action.reply_to_user.strip()
        if action.needs_human and not proposed_reply:
            proposed_reply = "这个我先转人工确认一下。"

        if bool(message.metadata.get("internal_event")):
            return self._complete_internal_event(
                action,
                message=message,
                trace_id=trace_id,
                proposed_reply=proposed_reply,
                run_id=run_id,
                run_version=run_version,
            )

        review_item = {
            "item_id": "reply_to_user",
            "source": "reply_to_user",
            "recipient_id": message.sender_id,
            "recipient_name": message.sender_name,
            "text": proposed_reply,
            "source_text": proposed_reply,
            "action_evidence": external_action_evidence_from_tool_results(
                list(
                    context_payload.get("turn_tool_evidence")
                    or context_payload.get("previous_tool_results")
                    or []
                )
            ),
        }
        generation_result = processor.run_text_generation(
            message=message,
            trace_id=trace_id,
            action=action,
            items=[review_item],
            context_payload=context_payload,
            turn_budget=budgets.text_generation,
            generation_scope="reply_to_user",
        )
        if generation_result is not None:
            collected_results.append(generation_result)
            rewrites = customer_visible_rewrites(generation_result)
            if rewrites.get("reply_to_user"):
                proposed_reply = rewrites["reply_to_user"].strip()
                action = action_with_customer_visible_rewrites(action, rewrites)
                self.trace_recorder.record(
                    trace_id, "action_after_customer_visible_text_generation", action.to_dict()
                )
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
            self.store.append_tool_turn(
                message.conversation_id,
                json.dumps([review_result.to_dict()], ensure_ascii=False),
                trace_id,
            )
            if not customer_visible_content_review_approved(review_result):
                return ActionProcessingResult(
                    action=action,
                    tool_results=collected_results,
                    pending_tool_results=[review_result],
                    continue_loop=True,
                )

        if self.run_is_stale(message, run_version):
            self._trace_stale_reply(message, trace_id=trace_id, run_id=run_id, run_version=run_version)
            return ActionProcessingResult(
                action=action,
                tool_results=collected_results,
                final_reply="",
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

    def apply_customer_visible_rewrites(
        self, action: AgentAction, result: ToolResult, *, trace_id: str
    ) -> AgentAction:
        rewrites = customer_visible_rewrites(result)
        if not rewrites:
            return action
        rewritten = action_with_customer_visible_rewrites(action, rewrites)
        self.trace_recorder.record(
            trace_id, "action_after_customer_visible_text_generation", rewritten.to_dict()
        )
        return rewritten

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

    def _complete_internal_event(
        self,
        action: AgentAction,
        *,
        message: UserMessage,
        trace_id: str,
        proposed_reply: str,
        run_id: str,
        run_version: int,
    ) -> ActionProcessingResult:
        summary = proposed_reply or "内部定时任务已完成。"
        self.store.append_assistant_turn(
            message.conversation_id,
            summary,
            trace_id,
            metadata={
                "internal_event": True,
                "delivery_mode": "internal_only",
                "run_id": run_id,
                "run_version": run_version,
            },
        )
        self.trace_recorder.record(
            trace_id,
            "final_output",
            {
                "reply": summary,
                "delivery_mode": "internal_only",
                "run_id": run_id,
                "run_version": run_version,
            },
        )
        return ActionProcessingResult(action=action, final_reply=summary, stop_loop=True)

    def _trace_stale_reply(
        self, message: UserMessage, *, trace_id: str, run_id: str, run_version: int
    ) -> None:
        payload = {
            "run_id": run_id,
            "run_version": run_version,
            "current_version": self.store.conversation_version(message.conversation_id),
        }
        self.trace_recorder.record(
            trace_id,
            "conversation_run_stale",
            {**payload, "blocked": "final_reply"},
            level="WARN",
        )
        self.trace_recorder.record(
            trace_id,
            "final_output",
            {**payload, "reply": "", "reason": "conversation_run_stale"},
            level="WARN",
        )

    def _emit(self, event_name: str, *, trace_id: str, payload: dict[str, Any]) -> None:
        if self.hook_manager is not None:
            self.hook_manager.emit(event_name, trace_id=trace_id, payload=payload)


__all__ = [
    "CustomerVisibleActionService",
    "attach_content_review_proof",
    "customer_visible_rewrites",
]
