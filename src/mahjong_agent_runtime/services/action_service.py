"""解析并分派模型提出的动作，将模型建议转换为受控的应用层操作。

"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..action_contract import parse_action_with_repairs
from ..budget import TokenBudget
from ..copywriting import DEFAULT_CUSTOMER_VISIBLE_TEXT_PROMPT_PATH
from ..hooks import HookManager
from ..llm import AgentLLMClient
from ..models import AgentAction, ToolResult, UserMessage
from ..runtime_components import ActionProcessingResult, ModelActionStep, TurnBudgets
from ..stores import AgentStore
from ..visibility import DEFAULT_REPLY_SELF_REVIEW_PROMPT_PATH, CustomerVisibleProcessor
from .tool_service import ToolExecutionService
from .objective_continuation import continuation_feedback
from .visible_action_service import (
    CustomerVisibleActionService,
    attach_content_review_proof,
    customer_visible_rewrites,
)


@dataclass(slots=True)
class ActionProcessor:
    """调用模型，验证其契约，并分发产生的动作。"""

    llm_client: AgentLLMClient
    store: AgentStore
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
        """调用主模型并将一个响应解析为 AgentAction。"""

        self._emit(
            "before_llm_call",
            trace_id=trace_id,
            payload={"step_index": step_index, "message": message.to_dict(), "run_id": run_id},
        )
        budget_decision = budget.reserve(built_messages)
        self.trace_recorder.record(trace_id, "budget_checked", budget_decision.to_dict())
        if not budget_decision.allowed:
            return self._stop_for_human(
                message,
                trace_id=trace_id,
                reason=budget_decision.reason,
                run_id=run_id,
                run_version=run_version,
            )

        started = time.perf_counter()
        try:
            raw_response = self.llm_client.complete(
                built_messages,
                trace_id=trace_id,
                timeout_seconds=self.llm_timeout_seconds,
            )
        except Exception as exc:
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
            return self._stop_for_human(
                message,
                trace_id=trace_id,
                reason="llm_error",
                run_id=run_id,
                run_version=run_version,
            )

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
        action, errors, repairs = parse_action_with_repairs(raw_response)
        if repairs:
            self.trace_recorder.record(
                trace_id,
                "action_contract_repaired",
                {"repairs": repairs, "step_index": step_index},
                level="WARN",
            )
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
        """记录动作结构化合同feedback，形成后续决策和审计所需的持久证据。

"""

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
                    "Regenerate one complete AgentAction JSON object from scratch; do not copy malformed fragments. "
                    "Every JSON object member must be key:value. objective_state.known_facts must be an object such as "
                    "{\"fact_name\": \"fact value\"}; never use set-like {\"fact A\", \"fact B\"}. "
                    "All list fields, including objective_plan[].depends_on, must use arrays. "
                    "If waiting for user, use objective_status=waiting_user with non-empty reply_to_user. "
                    "If tools are needed, use objective_status=needs_tool with at least one tool_call."
                ),
            },
            error="AgentAction contract invalid: " + "; ".join(errors),
        )
        self.trace_recorder.record(trace_id, "contract_error_feedback", feedback.to_dict(), level="WARN")
        self.store.append_tool_turn(
            message.conversation_id,
            json.dumps([feedback.to_dict()], ensure_ascii=False),
            trace_id,
        )
        return [feedback]

    def record_objective_continuation_feedback(
        self,
        message: UserMessage,
        *,
        trace_id: str,
        action: AgentAction,
        continuation: dict[str, Any],
        step_index: int,
    ) -> ToolResult:
        """记录objectivecontinuationfeedback，形成后续决策和审计所需的持久证据。

"""

        feedback = continuation_feedback(action, continuation)
        payload = {
            "step_index": step_index,
            "action_status": action.objective_status,
            "continuation": continuation,
        }
        self.trace_recorder.record(trace_id, "objective_continuation_rejected", payload, level="WARN")
        self.trace_recorder.record(trace_id, "tool_result", feedback.to_dict(), level="WARN")
        self.store.append_tool_turn(
            message.conversation_id,
            json.dumps([feedback.to_dict()], ensure_ascii=False),
            trace_id,
        )
        return feedback

    def trace_action_plan(
        self,
        action: AgentAction,
        *,
        trace_id: str,
        step_index: int,
        previous_tool_result_count: int,
    ) -> None:
        """记录模型的目标、计划和选定的工具以供审计。"""

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
        """处理动作，将输入转换为受控的下一步结果。

"""

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

    def process_tool_action(self, action: AgentAction, **kwargs: Any) -> ActionProcessingResult:
        """处理工具动作，将输入转换为受控的下一步结果。

"""

        return self._visible_service().process_tool_action(
            action,
            processor=self.customer_visible_processor(),
            **kwargs,
        )

    def process_reply_action(self, action: AgentAction, **kwargs: Any) -> ActionProcessingResult:
        """处理reply动作，将输入转换为受控的下一步结果。

"""

        return self._visible_service().process_reply_action(
            action,
            processor=self.customer_visible_processor(),
            **kwargs,
        )

    @staticmethod
    def attach_content_review_proof(action: AgentAction, *, trace_id: str) -> AgentAction:
        return attach_content_review_proof(action, trace_id=trace_id)

    def apply_customer_visible_rewrites(
        self, action: AgentAction, result: ToolResult, *, trace_id: str
    ) -> AgentAction:
        """应用客户客户可见rewrites规则，并保持领域状态合法。"""
        return self._visible_service().apply_customer_visible_rewrites(
            action, result, trace_id=trace_id
        )

    @staticmethod
    def customer_visible_rewrites(result: ToolResult) -> dict[str, str]:
        return customer_visible_rewrites(result)

    def run_is_stale(self, message: UserMessage, run_version: int) -> bool:
        return self._visible_service().run_is_stale(message, run_version)

    def append_pending_assistant_turn(
        self,
        conversation_id: str,
        text: str,
        trace_id: str,
        *,
        run_id: str,
        run_version: int,
    ) -> None:
        """追加pendingassistantturn，保留原有时间顺序和审计信息。"""
        self._visible_service().append_pending_assistant_turn(
            conversation_id,
            text,
            trace_id,
            run_id=run_id,
            run_version=run_version,
        )

    def customer_visible_processor(self) -> CustomerVisibleProcessor:
        """为此动作构建一次性文本生成/审核处理器。"""

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

    def _visible_service(self) -> CustomerVisibleActionService:
        return CustomerVisibleActionService(
            store=self.store,
            trace_recorder=self.trace_recorder,
            tool_execution_service=self.tool_execution_service,
            hook_manager=self.hook_manager,
        )

    def _stop_for_human(
        self,
        message: UserMessage,
        *,
        trace_id: str,
        reason: str,
        run_id: str,
        run_version: int,
    ) -> ModelActionStep:
        reply = "这个我先转人工确认一下。"
        self.trace_recorder.record(
            trace_id,
            "final_output",
            {"reply": reply, "reason": reason},
            level="WARN",
        )
        self.append_pending_assistant_turn(
            message.conversation_id,
            reply,
            trace_id,
            run_id=run_id,
            run_version=run_version,
        )
        return ModelActionStep(action=None, final_reply=reply, stop_loop=True)

    def _emit(self, event_name: str, *, trace_id: str, payload: dict[str, Any]) -> None:
        if self.hook_manager is not None:
            self.hook_manager.emit(event_name, trace_id=trace_id, payload=payload)


__all__ = ["ActionProcessor"]
