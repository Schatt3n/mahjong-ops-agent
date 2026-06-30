from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .action_validator import ActionValidator
from .context_builder import WorkflowContextBuildResult, WorkflowContextBuilder
from .core import AgentCore
from .memory import ShortTermMemoryRecord, ShortTermMemoryStore
from .models import DEFAULT_TZ, Message
from .observability import (
    InMemoryTraceRecorder,
    TraceEvent,
    TraceRecorder,
    TraceStep,
    validate_controlled_trace_completeness,
)
from .reply_guard import ReplyGuard
from .reply_policy import ReplyPolicy
from .semantic_resolver import SemanticResolver
from .state_machine import InMemoryWorkflowStateStore, StateMachine, WorkflowStateStore
from .tool_orchestrator import ToolOrchestrationResult, ToolOrchestrator, ToolOrchestratorConfig
from .workflow_models import (
    ActionName,
    EntityType,
    GameWorkflowStatus,
    GuardedReply,
    ReplyDraft,
    SemanticResolution,
    StateTransition,
    ToolName,
    ToolResult,
    ValidatedAction,
    WorkflowRun,
    new_workflow_id,
)


@dataclass(slots=True)
class ControlledWorkflowConfig:
    persist_short_memory: bool = True
    record_llm_prompt: bool = True
    record_llm_response: bool = True
    record_context_payload: bool = True


@dataclass(slots=True)
class ControlledWorkflowResult:
    run: WorkflowRun
    context_build: WorkflowContextBuildResult
    tool_orchestration: ToolOrchestrationResult
    trace_events: list[TraceEvent] = field(default_factory=list)

    @property
    def final_text(self) -> str:
        return self.run.guarded_reply.final_text if self.run.guarded_reply else ""


class ControlledWorkflowService:
    """Composes the controlled agent workflow in one auditable pipeline.

    The service is deliberately thin: it wires existing stages, records trace
    events, and writes short-term conversation memory. It does not put business
    if-else patches back into the legacy trial app.
    """

    def __init__(
        self,
        *,
        core: AgentCore,
        context_builder: WorkflowContextBuilder,
        semantic_resolver: SemanticResolver,
        action_validator: ActionValidator | None = None,
        tool_orchestrator: ToolOrchestrator | None = None,
        state_machine: StateMachine | None = None,
        state_store: WorkflowStateStore | None = None,
        reply_policy: ReplyPolicy | None = None,
        reply_guard: ReplyGuard | None = None,
        memory_store: ShortTermMemoryStore | None = None,
        trace_recorder: TraceRecorder | None = None,
        config: ControlledWorkflowConfig | None = None,
    ) -> None:
        self.core = core
        self.context_builder = context_builder
        self.semantic_resolver = semantic_resolver
        self.action_validator = action_validator or ActionValidator()
        self.tool_orchestrator = tool_orchestrator or ToolOrchestrator(
            core,
            ToolOrchestratorConfig(allow_state_write=True),
        )
        self.state_machine = state_machine or StateMachine()
        self.state_store = state_store or InMemoryWorkflowStateStore()
        self.reply_policy = reply_policy or ReplyPolicy()
        self.reply_guard = reply_guard or ReplyGuard()
        self.memory_store = memory_store
        self.trace_recorder = trace_recorder or InMemoryTraceRecorder()
        self.config = config or ControlledWorkflowConfig()

    def handle_message(
        self,
        message: Message,
        *,
        now: datetime | None = None,
        trace_id: str | None = None,
    ) -> ControlledWorkflowResult:
        effective_now = now or datetime.now(DEFAULT_TZ)
        effective_trace_id = trace_id or str(message.metadata.get("trace_id") or new_workflow_id("trace"))
        self._record(
            effective_trace_id,
            TraceStep.USER_INPUT,
            {
                "message_id": message.id,
                "conversation_id": message.metadata.get("conversation_id") or message.channel_id,
                "sender_id": message.sender_id,
                "sender_name": message.sender_name,
                "channel_type": message.channel_type,
                "text": message.text,
            },
            now=effective_now,
        )

        context_build = self.context_builder.build(message, now=effective_now, trace_id=effective_trace_id)
        context = context_build.context
        run = WorkflowRun(trace_id=effective_trace_id, context=context, created_at=effective_now)
        self._record_context(context_build, now=effective_now)

        semantic_resolution = self.semantic_resolver.resolve(context)
        run.semantic_resolution = semantic_resolution
        self._record_semantic_resolution(effective_trace_id, semantic_resolution, now=effective_now)

        validated_action = self.action_validator.validate(context, semantic_resolution)
        run.validated_action = validated_action
        self._record_validated_action(effective_trace_id, validated_action, now=effective_now)

        tool_orchestration = self.tool_orchestrator.run(
            context=context,
            semantic_resolution=semantic_resolution,
            validated_action=validated_action,
            now=effective_now,
        )
        run.tool_results = list(tool_orchestration.tool_results)
        self._record_tool_results(effective_trace_id, tool_orchestration.tool_results, now=effective_now)

        state_transitions = self._apply_state_transitions(
            self._plan_state_transitions(
                validated_action,
                semantic_resolution,
                tool_orchestration,
                trace_id=effective_trace_id,
            )
        )
        run.state_transitions = state_transitions
        self._record_state_transitions(effective_trace_id, state_transitions, now=effective_now)

        reply_draft = self.reply_policy.draft(
            context=context,
            semantic_resolution=semantic_resolution,
            validated_action=validated_action,
            tool_result=tool_orchestration,
            state_transitions=state_transitions,
        )
        run.reply_draft = reply_draft
        self._record_reply_draft(effective_trace_id, reply_draft, now=effective_now)

        guarded_reply = self.reply_guard.guard(
            draft=reply_draft,
            validated_action=validated_action,
            tool_result=tool_orchestration,
        )
        run.guarded_reply = guarded_reply
        self._record_guarded_reply(effective_trace_id, guarded_reply, now=effective_now)

        if self.config.persist_short_memory:
            self._write_short_memory(
                context_build,
                semantic_resolution=semantic_resolution,
                tool_results=tool_orchestration.tool_results,
                guarded_reply=guarded_reply,
                now=effective_now,
            )

        trace_before_final = self.trace_recorder.get_trace(effective_trace_id)
        completeness = validate_controlled_trace_completeness(
            [
                *trace_before_final,
                TraceEvent(
                    trace_id=effective_trace_id,
                    step=TraceStep.FINAL_OUTPUT,
                    content={},
                    occurred_at=effective_now,
                ),
            ]
        )
        self._record(
            effective_trace_id,
            TraceStep.FINAL_OUTPUT,
            {
                "final_text": guarded_reply.final_text,
                "reply_status": guarded_reply.status,
                "approval_required": validated_action.approval_required,
                "effective_action": validated_action.effective_action,
                "validation_code": validated_action.code,
                "trace_completeness": completeness.to_dict(),
            },
            now=effective_now,
        )

        return ControlledWorkflowResult(
            run=run,
            context_build=context_build,
            tool_orchestration=tool_orchestration,
            trace_events=self.trace_recorder.get_trace(effective_trace_id),
        )

    def _record_context(self, context_build: WorkflowContextBuildResult, *, now: datetime) -> None:
        context = context_build.context
        content: dict[str, Any] = {
            "used_short_memory": context_build.used_short_memory,
            "followup_context": context_build.followup_context,
            "notes": list(context_build.notes),
        }
        if self.config.record_context_payload:
            content["context"] = context.to_prompt_dict()
        self._record(context.current_message.trace_id, TraceStep.CONTEXT_BUILT, content, now=now)

    def _record_semantic_resolution(
        self,
        trace_id: str,
        semantic_resolution: SemanticResolution,
        *,
        now: datetime,
    ) -> None:
        prompt_messages = semantic_resolution.raw_response.get("prompt_messages")
        if self.config.record_llm_prompt and prompt_messages:
            self._record(trace_id, TraceStep.LLM_PROMPT, {"messages": prompt_messages}, now=now)
        raw_response = dict(semantic_resolution.raw_response)
        raw_response.pop("prompt_messages", None)
        if self.config.record_llm_response:
            llm_contract = raw_response.get("llm_contract")
            response_level = "WARN" if semantic_resolution.needs_human_review else "INFO"
            if isinstance(llm_contract, dict) and llm_contract.get("accepted") is False:
                response_level = "WARN"
            self._record(
                trace_id,
                TraceStep.LLM_RESPONSE,
                {
                    "intent": semantic_resolution.intent,
                    "reasoning_summary": semantic_resolution.reasoning_summary,
                    "needs_human_review": semantic_resolution.needs_human_review,
                    "raw_response": raw_response,
                },
                level=response_level,
                now=now,
            )
        self._record(
            trace_id,
            TraceStep.ACTION_PROPOSED,
            {
                "proposed_action": semantic_resolution.proposed_action,
                "game_requirement": semantic_resolution.game_requirement,
            },
            now=now,
        )

    def _record_validated_action(
        self,
        trace_id: str,
        validated_action: ValidatedAction,
        *,
        now: datetime,
    ) -> None:
        self._record(
            trace_id,
            TraceStep.ACTION_VALIDATED,
            {"validated_action": validated_action},
            level="INFO" if validated_action.allowed else "WARN",
            now=now,
        )

    def _record_tool_results(
        self,
        trace_id: str,
        tool_results: list[ToolResult],
        *,
        now: datetime,
    ) -> None:
        if not tool_results:
            self._record(trace_id, TraceStep.TOOL_CALLED, {"tool_results": []}, now=now)
            return
        for result in tool_results:
            self._record(
                trace_id,
                TraceStep.TOOL_CALLED,
                {"tool_result": result},
                level="INFO" if result.allowed else "WARN",
                now=now,
            )

    def _record_state_transitions(
        self,
        trace_id: str,
        state_transitions: list[StateTransition],
        *,
        now: datetime,
    ) -> None:
        self._record(
            trace_id,
            TraceStep.STATE_TRANSITION,
            {"state_transitions": state_transitions},
            level="INFO" if all(item.allowed for item in state_transitions) else "WARN",
            now=now,
        )

    def _record_reply_draft(self, trace_id: str, reply_draft: ReplyDraft, *, now: datetime) -> None:
        llm_contract = reply_draft.metadata.get("llm_contract") if isinstance(reply_draft.metadata, dict) else None
        level = "WARN" if isinstance(llm_contract, dict) and llm_contract.get("accepted") is False else "INFO"
        self._record(trace_id, TraceStep.REPLY_DRAFTED, {"reply_draft": reply_draft}, level=level, now=now)

    def _record_guarded_reply(self, trace_id: str, guarded_reply: GuardedReply, *, now: datetime) -> None:
        self._record(
            trace_id,
            TraceStep.REPLY_GUARDED,
            {"guarded_reply": guarded_reply},
            level="WARN" if guarded_reply.changed else "INFO",
            now=now,
        )

    def _plan_state_transitions(
        self,
        validated_action: ValidatedAction,
        semantic_resolution: SemanticResolution,
        tool_orchestration: ToolOrchestrationResult,
        *,
        trace_id: str,
    ) -> list[StateTransition]:
        if validated_action.effective_action == ActionName.QUEUE_INVITES:
            create_result = tool_orchestration.result_for(ToolName.CREATE_GAME)
            if not self._successful_tool_result(create_result):
                return []
            entity_id = self._state_entity_id(
                validated_action,
                semantic_resolution,
                tool_result=create_result,
                trace_id=trace_id,
            )
            outbox_result = tool_orchestration.result_for(ToolName.CREATE_PENDING_OUTBOX)
            has_outbox = bool(outbox_result and outbox_result.called and outbox_result.allowed)
            current_status = self.state_store.current_status(EntityType.GAME.value, entity_id)
            transitions: list[StateTransition] = []
            if current_status is None:
                transitions.append(
                    self.state_machine.validate_game_transition(
                        entity_id=entity_id,
                        from_status=None,
                        to_status=GameWorkflowStatus.OPEN,
                        reason=validated_action.reason,
                    )
                )
                current_status = GameWorkflowStatus.OPEN.value
            if has_outbox and current_status == GameWorkflowStatus.OPEN.value:
                transitions.append(
                    self.state_machine.validate_game_transition(
                        entity_id=entity_id,
                        from_status=GameWorkflowStatus.OPEN,
                        to_status=GameWorkflowStatus.NEGOTIATING,
                        reason="已创建待审批邀约，进入邀约中。",
                    )
                )
            return transitions
        if validated_action.effective_action == ActionName.CLOSE_GAME:
            close_result = tool_orchestration.result_for(ToolName.CLOSE_GAME)
            if not self._successful_tool_result(close_result):
                return []
            entity_id = self._state_entity_id(
                validated_action,
                semantic_resolution,
                tool_result=close_result,
                trace_id=trace_id,
            )
            current_status = self.state_store.current_status(EntityType.GAME.value, entity_id) or GameWorkflowStatus.OPEN.value
            return [
                self.state_machine.validate_game_transition(
                    entity_id=entity_id,
                    from_status=current_status,
                    to_status=GameWorkflowStatus.CANCELLED,
                    reason=validated_action.reason,
                )
            ]
        return []

    def _state_entity_id(
        self,
        validated_action: ValidatedAction,
        semantic_resolution: SemanticResolution,
        *,
        tool_result: ToolResult | None = None,
        trace_id: str,
    ) -> str:
        intent = (tool_result.result.get("state_write_intent") if tool_result and tool_result.result else None) or {}
        if isinstance(intent, dict) and intent.get("entity_id"):
            return str(intent["entity_id"])
        action_game_id = semantic_resolution.proposed_action.arguments.get("game_id")
        if action_game_id:
            return str(action_game_id)
        return validated_action.idempotency_key or f"pending_game:{trace_id}"

    def _successful_tool_result(self, result: ToolResult | None) -> bool:
        return bool(result and result.called and result.allowed)

    def _apply_state_transitions(self, transitions: list[StateTransition]) -> list[StateTransition]:
        return [self.state_store.apply_transition(transition) for transition in transitions]

    def _write_short_memory(
        self,
        context_build: WorkflowContextBuildResult,
        *,
        semantic_resolution: SemanticResolution,
        tool_results: list[ToolResult],
        guarded_reply: GuardedReply,
        now: datetime,
    ) -> None:
        if not self.memory_store:
            return
        context = context_build.context
        self.memory_store.append(
            ShortTermMemoryRecord(
                conversation_id=context.current_message.conversation_id,
                sender_id=context.current_message.sender_id,
                user_message=context.current_message,
                system_reply=guarded_reply.final_text,
                game_requirement=semantic_resolution.game_requirement,
                tool_results=list(tool_results),
                created_at=now,
                metadata={
                    "trace_id": context.current_message.trace_id,
                    "workflow": "controlled_workflow.v1",
                },
            ),
            now=now,
        )
        self._record(
            context.current_message.trace_id,
            TraceStep.MEMORY_WRITTEN,
            {
                "conversation_id": context.current_message.conversation_id,
                "sender_id": context.current_message.sender_id,
                "system_reply": guarded_reply.final_text,
            },
            now=now,
        )

    def _record(
        self,
        trace_id: str,
        step: TraceStep | str,
        content: dict[str, Any],
        *,
        level: str = "INFO",
        now: datetime | None = None,
    ) -> None:
        self.trace_recorder.record(
            trace_id,
            step,
            content,
            level=level,
            occurred_at=now,
        )
