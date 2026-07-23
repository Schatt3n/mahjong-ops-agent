from __future__ import annotations

"""Context build, trace, budget compression, and checkpoint lifecycle."""

from dataclasses import dataclass
from typing import Any

from ..budget import TokenBudget
from ..context import AgentContextBuilder, BuiltContext, estimate_tokens
from ..hooks import HookManager
from ..models import StateTransition, ToolResult, UserMessage
from ..summary import ContextSummaryManager, ContextSummaryResult


@dataclass(slots=True)
class ContextLifecycleManager:
    """Build and compress model context without leaking lifecycle details into the loop."""

    context_builder: AgentContextBuilder
    trace_recorder: Any
    context_summary_manager: ContextSummaryManager | None = None
    context_summary_preemptive_ratio: float = 0.85
    hook_manager: HookManager | None = None

    def build_and_trace_context(
        self,
        message: UserMessage,
        *,
        trace_id: str,
        pending_tool_results: list[ToolResult],
        turn_tool_evidence: list[ToolResult],
        run_id: str,
        run_version: int,
        step_index: int,
        progress_hint: str | None = None,
    ) -> BuiltContext:
        """Build model context and record replayable prompt artifacts."""

        built = self.context_builder.build(
            message,
            trace_id=trace_id,
            previous_tool_results=pending_tool_results,
            turn_tool_evidence=turn_tool_evidence,
            run_id=run_id,
            run_version=run_version,
        )
        if progress_hint:
            if built.messages and built.messages[0].get("role") == "system":
                built.messages[0] = {
                    **built.messages[0],
                    "content": built.messages[0].get("content", "").rstrip() + "\n\n" + progress_hint,
                }
            else:
                built.messages.insert(0, {"role": "system", "content": progress_hint})
            self.trace_recorder.record(
                trace_id,
                "agent_progress_hint_injected",
                {
                    "step_index": step_index,
                    "line_count": len(progress_hint.splitlines()),
                    "hint": progress_hint,
                },
            )
        self.trace_recorder.record(trace_id, "context_packed", built.audit)
        self.trace_recorder.record(trace_id, "context_built", built.payload)
        self.trace_recorder.record(
            trace_id,
            "llm_prompt",
            {"messages": built.messages, "step_index": step_index},
        )
        self._emit(
            "after_context_built",
            trace_id=trace_id,
            payload={"step_index": step_index, "audit": built.audit, "payload": built.payload},
        )
        return built

    def summarize_and_rebuild_context_if_needed(
        self,
        message: UserMessage,
        *,
        built: BuiltContext,
        trace_id: str,
        pending_tool_results: list[ToolResult],
        turn_tool_evidence: list[ToolResult],
        run_id: str,
        run_version: int,
        step_index: int,
        budget: TokenBudget,
        progress_hint: str | None = None,
    ) -> tuple[BuiltContext, StateTransition | None]:
        """Summarize and rebuild before the model call when prompt budget is pressured."""

        estimated = sum(estimate_tokens(item.get("content", "")) for item in built.messages)
        threshold = max(1, int(budget.max_tokens_per_call * self.context_summary_preemptive_ratio))
        self.trace_recorder.record(
            trace_id,
            "context_budget_precheck",
            {
                "estimated_tokens": estimated,
                "max_tokens_per_call": budget.max_tokens_per_call,
                "trigger_threshold_tokens": threshold,
                "context_summary_enabled": self.context_summary_manager is not None,
                "step_index": step_index,
            },
        )
        if self.context_summary_manager is None or estimated < threshold:
            return built, None
        checkpoint = built.payload.get("conversation_checkpoint") if isinstance(built.payload, dict) else None
        if isinstance(checkpoint, dict) and checkpoint.get("source_trace_id") == trace_id:
            self.trace_recorder.record(
                trace_id,
                "context_summary_budget_already_applied",
                {
                    "estimated_tokens": estimated,
                    "max_tokens_per_call": budget.max_tokens_per_call,
                    "trigger_threshold_tokens": threshold,
                    "step_index": step_index,
                },
            )
            return built, None
        summary_result = self._summarize(
            message,
            built=built,
            trace_id=trace_id,
            estimated=estimated,
            threshold=threshold,
            budget=budget,
        )
        if summary_result is None or summary_result.transition is None:
            if summary_result is not None:
                self.trace_recorder.record(
                    trace_id,
                    "context_summary_budget_not_applied",
                    summary_result.to_dict(),
                    level="WARN",
                )
            return built, None

        rebuilt = self.build_and_trace_context(
            message,
            trace_id=trace_id,
            pending_tool_results=pending_tool_results,
            turn_tool_evidence=turn_tool_evidence,
            run_id=run_id,
            run_version=run_version,
            step_index=step_index,
            progress_hint=progress_hint,
        )
        rebuilt_estimated = sum(estimate_tokens(item.get("content", "")) for item in rebuilt.messages)
        self.trace_recorder.record(
            trace_id,
            "context_rebuilt_after_summary",
            {
                "previous_estimated_tokens": estimated,
                "rebuilt_estimated_tokens": rebuilt_estimated,
                "checkpoint": summary_result.checkpoint.to_dict() if summary_result.checkpoint else None,
                "transition": summary_result.transition.to_dict(),
                "step_index": step_index,
            },
        )
        self._emit(
            "after_context_rebuilt",
            trace_id=trace_id,
            payload={
                "step_index": step_index,
                "previous_estimated_tokens": estimated,
                "rebuilt_estimated_tokens": rebuilt_estimated,
            },
        )
        return rebuilt, summary_result.transition

    def maybe_summarize_after_turn(
        self,
        *,
        conversation_id: str,
        trace_id: str,
        task_context_id: str | None = None,
    ) -> ContextSummaryResult | None:
        """Run the lower-priority post-turn checkpoint policy when configured."""

        if self.context_summary_manager is None:
            return None
        return self.context_summary_manager.maybe_summarize_after_turn(
            conversation_id=conversation_id,
            trace_id=trace_id,
            task_context_id=task_context_id,
        )

    def _summarize(
        self,
        message: UserMessage,
        *,
        built: BuiltContext,
        trace_id: str,
        estimated: int,
        threshold: int,
        budget: TokenBudget,
    ) -> ContextSummaryResult | None:
        assert self.context_summary_manager is not None
        task_context_id = self.summary_task_context_id(message, built=built)
        try:
            return self.context_summary_manager.summarize_for_context_budget(
                conversation_id=message.conversation_id,
                trace_id=trace_id,
                estimated_context_tokens=estimated,
                max_context_tokens=budget.max_tokens_per_call,
                trigger_threshold_tokens=threshold,
                task_context_id=task_context_id,
            )
        except Exception as exc:
            self.trace_recorder.record(
                trace_id,
                "context_summary_error",
                {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "trigger": "context_budget",
                },
                level="ERROR",
            )
            return None

    def summary_task_context_id(
        self,
        message: UserMessage,
        *,
        built: BuiltContext | None = None,
    ) -> str | None:
        """Return the backend-resolved summary scope for this turn.

        Trusted scheduled/system events may resume an older task explicitly.
        Ordinary messages use the task selected by ``TaskContextManager`` and
        exposed in the already-built context; user text never chooses this id.
        """

        metadata = message.metadata if isinstance(message.metadata, dict) else {}
        trusted_source = str(metadata.get("_trusted_source_task_context_id") or "")
        if trusted_source:
            task_context = (
                self.context_summary_manager.store.get_task_context(trusted_source)
                if self.context_summary_manager
                else None
            )
            if task_context is not None and task_context.conversation_id == message.conversation_id:
                return trusted_source
        if built is not None:
            window = built.payload.get("task_context_window") if isinstance(built.payload, dict) else None
            if isinstance(window, dict):
                resolved = str(window.get("task_context_id") or "")
                if resolved:
                    return resolved
        if self.context_summary_manager is None:
            return None
        task_context = self.context_summary_manager.store.current_task_context(
            message.conversation_id,
            message.sender_id,
        )
        return task_context.task_context_id if task_context is not None else None

    def _emit(self, event_name: str, *, trace_id: str, payload: dict[str, Any]) -> None:
        if self.hook_manager is not None:
            self.hook_manager.emit(event_name, trace_id=trace_id, payload=payload)


__all__ = ["ContextLifecycleManager"]
