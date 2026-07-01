from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any

from .action_validator import ActionValidator
from .context_builder import WorkflowContextBuildResult, WorkflowContextBuilder
from .core import AgentCore
from .input_gate import InputGate, InputGateDecision
from .memory import ShortTermMemoryRecord, ShortTermMemoryStore
from .models import DEFAULT_TZ, Message
from .observability import (
    InMemoryTraceRecorder,
    TraceEvent,
    TraceRecorder,
    TraceStep,
    validate_controlled_trace_completeness,
)
from .reply_approval import ReplyApprovalQueue, ReplyApprovalQueueResult
from .reply_guard import ReplyGuard
from .reply_policy import ReplyPolicy
from .semantic_resolver import SemanticResolver
from .state_write_contract import StateWriteIntent, parse_state_write_intent
from .state_machine import InMemoryWorkflowStateStore, StateMachine, WorkflowStateStore
from .tool_orchestrator import ToolOrchestrationResult, ToolOrchestrator, ToolOrchestratorConfig
from .workflow_models import (
    ActionName,
    ActionSource,
    ConversationContext,
    EntityType,
    GameWorkflowStatus,
    GuardedReply,
    ProposedAction,
    ReplyDraft,
    ReplyStatus,
    RiskLevel,
    SemanticResolution,
    StateTransition,
    ToolName,
    ToolResult,
    UserIntent,
    UserMessage,
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
    reply_approval: ReplyApprovalQueueResult | None = None
    trace_events: list[TraceEvent] = field(default_factory=list)

    @property
    def final_text(self) -> str:
        return self.run.guarded_reply.final_text if self.run.guarded_reply else ""


@dataclass(slots=True)
class StateTransitionPlan:
    transitions: list[StateTransition] = field(default_factory=list)
    rejected_state_write_intents: list[dict[str, Any]] = field(default_factory=list)


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
        reply_approval_queue: ReplyApprovalQueue | None = None,
        memory_store: ShortTermMemoryStore | None = None,
        input_gate: InputGate | None = None,
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
        self.reply_approval_queue = reply_approval_queue
        self.memory_store = memory_store
        self.input_gate = input_gate
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
            self._user_input_trace_payload(message),
            now=effective_now,
        )
        input_gate_decision: InputGateDecision | None = None
        if self.input_gate is not None:
            input_gate_decision = self.input_gate.begin(message, trace_id=effective_trace_id, now=effective_now)
            self._record(
                effective_trace_id,
                "input_gate",
                input_gate_decision.to_dict(),
                level="INFO" if input_gate_decision.accepted else "WARN",
                now=effective_now,
            )
            if not input_gate_decision.accepted:
                return self._input_gate_short_circuit(
                    message,
                    decision=input_gate_decision,
                    now=effective_now,
                    trace_id=effective_trace_id,
                )

        try:
            result = self._handle_accepted_message(message, now=effective_now, trace_id=effective_trace_id)
        except Exception:
            if self.input_gate is not None and input_gate_decision and input_gate_decision.accepted:
                self.input_gate.fail(message, trace_id=effective_trace_id, now=effective_now)
            raise
        if self.input_gate is not None and input_gate_decision and input_gate_decision.accepted:
            self.input_gate.complete(message, result, trace_id=effective_trace_id, now=effective_now)
        return result

    def _user_input_trace_payload(self, message: Message) -> dict[str, Any]:
        metadata = message.metadata or {}
        source_message_id = (
            metadata.get("source_message_id")
            or metadata.get("message_id")
            or metadata.get("platform_message_id")
            or message.id
        )
        return {
            "message_id": message.id,
            "source_message_id": source_message_id,
            "platform_message_id": metadata.get("platform_message_id"),
            "tenant_id": metadata.get("tenant_id") or metadata.get("store_id") or "default",
            "conversation_id": metadata.get("conversation_id") or message.channel_id,
            "channel_id": message.channel_id,
            "channel_type": message.channel_type,
            "sequence": metadata.get("sequence"),
            "sender_id": message.sender_id,
            "sender_name": message.sender_name,
            "text": message.text,
            "input_refs": {
                "source": metadata.get("source"),
                "store_id": metadata.get("store_id"),
                "channel_ref": metadata.get("channel_ref"),
                "platform": metadata.get("platform"),
            },
        }

    def _handle_accepted_message(
        self,
        message: Message,
        *,
        now: datetime,
        trace_id: str,
    ) -> ControlledWorkflowResult:
        context_build = self.context_builder.build(message, now=now, trace_id=trace_id)
        context = context_build.context
        run = WorkflowRun(trace_id=trace_id, context=context, created_at=now)
        self._record_context(context_build, now=now)

        semantic_resolution = self.semantic_resolver.resolve(context)
        run.semantic_resolution = semantic_resolution
        self._record_semantic_resolution(trace_id, semantic_resolution, now=now)

        validated_action = self.action_validator.validate(context, semantic_resolution)
        run.validated_action = validated_action
        self._record_validated_action(trace_id, validated_action, now=now)

        tool_orchestration = self.tool_orchestrator.run(
            context=context,
            semantic_resolution=semantic_resolution,
            validated_action=validated_action,
            now=now,
        )
        run.tool_results = list(tool_orchestration.tool_results)
        self._record_tool_results(trace_id, tool_orchestration.tool_results, now=now)

        state_transition_plan = self._plan_state_transitions(
            validated_action,
            semantic_resolution,
            tool_orchestration,
            trace_id=trace_id,
        )
        state_transitions = self._apply_state_transitions(state_transition_plan.transitions)
        run.state_transitions = state_transitions
        self._record_state_transitions(
            trace_id,
            state_transitions,
            rejected_state_write_intents=state_transition_plan.rejected_state_write_intents,
            now=now,
        )

        reply_draft = self.reply_policy.draft(
            context=context,
            semantic_resolution=semantic_resolution,
            validated_action=validated_action,
            tool_result=tool_orchestration,
            state_transitions=state_transitions,
        )
        run.reply_draft = reply_draft
        self._record_reply_draft(trace_id, reply_draft, now=now)

        guarded_reply = self.reply_guard.guard(
            draft=reply_draft,
            validated_action=validated_action,
            tool_result=tool_orchestration,
        )
        run.guarded_reply = guarded_reply
        self._record_guarded_reply(trace_id, guarded_reply, now=now)

        reply_approval = self._queue_reply_approval(
            context=context,
            reply_draft=reply_draft,
            guarded_reply=guarded_reply,
            validated_action=validated_action,
            now=now,
        )
        self._record_reply_approval(trace_id, reply_approval, now=now)

        if self.config.persist_short_memory:
            self._write_short_memory(
                context_build,
                semantic_resolution=semantic_resolution,
                tool_results=tool_orchestration.tool_results,
                guarded_reply=guarded_reply,
                now=now,
            )

        trace_before_final = self.trace_recorder.get_trace(trace_id)
        completeness = validate_controlled_trace_completeness(
            [
                *trace_before_final,
                TraceEvent(
                    trace_id=trace_id,
                    step=TraceStep.FINAL_OUTPUT,
                    content={},
                    occurred_at=now,
                ),
            ]
        )
        self._record(
            trace_id,
            TraceStep.FINAL_OUTPUT,
            {
                "final_text": guarded_reply.final_text,
                "reply_status": guarded_reply.status,
                "reply_approval": reply_approval.to_dict(),
                "approval_required": validated_action.approval_required,
                "effective_action": validated_action.effective_action,
                "validation_code": validated_action.code,
                "trace_completeness": completeness.to_dict(),
            },
            now=now,
        )

        result = ControlledWorkflowResult(
            run=run,
            context_build=context_build,
            tool_orchestration=tool_orchestration,
            reply_approval=reply_approval,
            trace_events=self.trace_recorder.get_trace(trace_id),
        )
        return result

    def _input_gate_short_circuit(
        self,
        message: Message,
        *,
        decision: InputGateDecision,
        now: datetime,
        trace_id: str,
    ) -> ControlledWorkflowResult:
        context_build = self._input_gate_context_build(message, decision=decision, now=now, trace_id=trace_id)
        context = context_build.context
        run = WorkflowRun(trace_id=trace_id, context=context, created_at=now)
        proposed_action = ProposedAction(
            name=ActionName.IGNORE,
            source=ActionSource.RULES,
            confidence=1.0,
            reason=decision.reason,
            arguments={"input_gate": decision.to_dict()},
            risk_level=RiskLevel.LOW,
        )
        semantic_resolution = SemanticResolution(
            intent=UserIntent.UNKNOWN,
            proposed_action=proposed_action,
            reasoning_summary="入口幂等/顺序 gate 拒绝进入 LLM 语义解析。",
            raw_response={"input_gate": decision.to_dict()},
        )
        validated_action = ValidatedAction(
            proposed_action=proposed_action,
            effective_action=ActionName.IGNORE,
            allowed=False,
            code=self._input_gate_validation_code(decision),
            reason=decision.reason,
            approval_required=False,
            risk_level=RiskLevel.LOW,
        )
        tool_orchestration = ToolOrchestrationResult()
        reply_text = self._input_gate_reply_text(decision)
        reply_draft = ReplyDraft(
            text=reply_text,
            status=ReplyStatus.DRAFT,
            reasoning_summary=decision.reason,
            source=ActionSource.RULES,
            risk_level=RiskLevel.LOW,
            metadata={"input_gate": decision.to_dict()},
        )
        guarded_reply = GuardedReply(
            draft=reply_draft,
            final_text=reply_text,
            changed=False,
            guard_reasons=["input_gate_short_circuit"],
            status=ReplyStatus.GUARDED,
        )
        run.semantic_resolution = semantic_resolution
        run.validated_action = validated_action
        run.reply_draft = reply_draft
        run.guarded_reply = guarded_reply
        reply_approval = ReplyApprovalQueueResult(queued=False, reason="input_gate_short_circuit")

        self._record_context(context_build, now=now)
        self._record(
            trace_id,
            TraceStep.LLM_PROMPT,
            {"skipped": True, "reason": "input_gate_short_circuit", "input_gate": decision.to_dict()},
            level="WARN",
            now=now,
        )
        self._record(
            trace_id,
            TraceStep.LLM_RESPONSE,
            {"skipped": True, "reason": "input_gate_short_circuit", "raw_response": semantic_resolution.raw_response},
            level="WARN",
            now=now,
        )
        self._record(
            trace_id,
            TraceStep.ACTION_PROPOSED,
            {"proposed_action": proposed_action, "source": "input_gate"},
            level="WARN",
            now=now,
        )
        self._record_validated_action(trace_id, validated_action, now=now)
        self._record_tool_results(trace_id, [], now=now)
        self._record_state_transitions(trace_id, [], now=now)
        self._record_reply_draft(trace_id, reply_draft, now=now)
        self._record_guarded_reply(trace_id, guarded_reply, now=now)
        self._record_reply_approval(trace_id, reply_approval, now=now)
        trace_before_final = self.trace_recorder.get_trace(trace_id)
        completeness = validate_controlled_trace_completeness(
            [
                *trace_before_final,
                TraceEvent(
                    trace_id=trace_id,
                    step=TraceStep.FINAL_OUTPUT,
                    content={},
                    occurred_at=now,
                ),
            ]
        )
        self._record(
            trace_id,
            TraceStep.FINAL_OUTPUT,
            {
                "final_text": guarded_reply.final_text,
                "reply_status": guarded_reply.status,
                "approval_required": False,
                "effective_action": ActionName.IGNORE,
                "validation_code": validated_action.code,
                "input_gate": decision.to_dict(),
                "short_circuited": True,
                "reply_approval": reply_approval.to_dict(),
                "trace_completeness": completeness.to_dict(),
            },
            level="WARN",
            now=now,
        )
        return ControlledWorkflowResult(
            run=run,
            context_build=context_build,
            tool_orchestration=tool_orchestration,
            reply_approval=reply_approval,
            trace_events=self.trace_recorder.get_trace(trace_id),
        )

    def _input_gate_context_build(
        self,
        message: Message,
        *,
        decision: InputGateDecision,
        now: datetime,
        trace_id: str,
    ) -> WorkflowContextBuildResult:
        conversation_id = str(message.metadata.get("conversation_id") or message.channel_id or decision.scope)
        user_message = UserMessage(
            text=message.text,
            sender_id=message.sender_id,
            sender_name=message.sender_name,
            conversation_id=conversation_id,
            trace_id=trace_id,
            message_id=message.id,
            channel_type=message.channel_type,
            sent_at=message.sent_at or now,
            metadata={**dict(message.metadata), "input_gate": decision.to_dict()},
        )
        context = ConversationContext(
            current_message=user_message,
            trace_notes=[
                "input_gate.short_circuit",
                "入口 gate 只处理幂等、去重和顺序，不做业务语义判断。",
                decision.reason,
            ],
        )
        return WorkflowContextBuildResult(
            context=context,
            used_short_memory=False,
            followup_context={},
            notes=[
                "input_gate_short_circuit=True",
                f"source_message_id={decision.source_message_id}",
                f"sequence={decision.sequence}",
                f"expected_sequence={decision.expected_sequence}",
            ],
        )

    def _input_gate_validation_code(self, decision: InputGateDecision) -> str:
        if decision.waiting_for_sequence:
            return "input_gate_waiting_for_sequence"
        if decision.in_progress:
            return "input_gate_duplicate_in_progress"
        if decision.duplicate:
            return "input_gate_duplicate"
        if decision.out_of_order:
            return "input_gate_out_of_order"
        return "input_gate_rejected"

    def _input_gate_reply_text(self, decision: InputGateDecision) -> str:
        cached_result = decision.cached_result
        cached_text = getattr(cached_result, "final_text", "") if cached_result is not None else ""
        if cached_text:
            return str(cached_text)
        if decision.waiting_for_sequence:
            expected = f"第 {decision.expected_sequence} 条" if decision.expected_sequence else "前序"
            return f"这条消息顺序有点乱，我先等{expected}消息处理完再继续。"
        if decision.in_progress:
            return "这条消息正在处理中，我先不重复处理。"
        if decision.duplicate:
            return "这条消息已经处理过了，我先不重复处理。"
        return "这条消息暂时没有进入自动处理，我先转人工确认一下。"

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
        rejected_state_write_intents: list[dict[str, Any]] | None = None,
        now: datetime,
    ) -> None:
        rejected = list(rejected_state_write_intents or [])
        self._record(
            trace_id,
            TraceStep.STATE_TRANSITION,
            {
                "state_transitions": state_transitions,
                "rejected_state_write_intents": rejected,
            },
            level="WARN" if rejected or not all(item.allowed for item in state_transitions) else "INFO",
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

    def _queue_reply_approval(
        self,
        *,
        context: ConversationContext,
        reply_draft: ReplyDraft,
        guarded_reply: GuardedReply,
        validated_action: ValidatedAction,
        now: datetime,
    ) -> ReplyApprovalQueueResult:
        if self.reply_approval_queue is None:
            return ReplyApprovalQueueResult(queued=False, reason="reply_approval_queue_not_configured")
        return self.reply_approval_queue.enqueue(
            context=context,
            reply_draft=reply_draft,
            guarded_reply=guarded_reply,
            validated_action=validated_action,
            now=now,
        )

    def _record_reply_approval(
        self,
        trace_id: str,
        reply_approval: ReplyApprovalQueueResult,
        *,
        now: datetime,
    ) -> None:
        self._record(
            trace_id,
            TraceStep.REPLY_APPROVAL,
            {"reply_approval": reply_approval.to_dict()},
            level="INFO" if reply_approval.queued else "WARN",
            now=now,
        )

    def _plan_state_transitions(
        self,
        validated_action: ValidatedAction,
        semantic_resolution: SemanticResolution,
        tool_orchestration: ToolOrchestrationResult,
        *,
        trace_id: str,
    ) -> StateTransitionPlan:
        if validated_action.effective_action == ActionName.QUEUE_INVITES:
            create_result = tool_orchestration.result_for(ToolName.CREATE_GAME)
            if not self._successful_tool_result(create_result):
                return StateTransitionPlan()
            intent, rejected = self._state_write_intent(create_result)
            if intent is None:
                return StateTransitionPlan(rejected_state_write_intents=[rejected] if rejected else [])
            entity_id = intent.entity_id
            target_status = intent.target_status
            current_status = self.state_store.current_status(EntityType.GAME.value, entity_id)
            transitions: list[StateTransition] = []
            if current_status is None:
                bootstrap = self._transition_from_state_intent(
                    intent,
                    semantic_resolution,
                    from_status=None,
                    to_status=GameWorkflowStatus.OPEN.value,
                    reason_fallback=validated_action.reason,
                    trace_id=trace_id,
                )
                if bootstrap is not None:
                    transitions.append(bootstrap)
                current_status = GameWorkflowStatus.OPEN.value
            if target_status != current_status:
                transition = self._transition_from_state_intent(
                    intent,
                    semantic_resolution,
                    from_status=current_status,
                    reason_fallback=validated_action.reason,
                    trace_id=trace_id,
                )
                if transition is not None:
                    transitions.append(transition)
            return StateTransitionPlan(transitions=transitions)
        if validated_action.effective_action == ActionName.CLOSE_GAME:
            close_result = tool_orchestration.result_for(ToolName.CLOSE_GAME)
            if not self._successful_tool_result(close_result):
                return StateTransitionPlan()
            intent, rejected = self._state_write_intent(close_result)
            if intent is None:
                return StateTransitionPlan(rejected_state_write_intents=[rejected] if rejected else [])
            entity_id = intent.entity_id
            current_status = self.state_store.current_status(EntityType.GAME.value, entity_id) or GameWorkflowStatus.OPEN.value
            transition = self._transition_from_state_intent(
                intent,
                semantic_resolution,
                from_status=current_status,
                reason_fallback=validated_action.reason,
                trace_id=trace_id,
            )
            return StateTransitionPlan(transitions=[transition] if transition is not None else [])
        if validated_action.effective_action == ActionName.ACCEPT_SEAT:
            accept_result = tool_orchestration.result_for(ToolName.RECORD_SEAT_ACCEPTANCE)
            if not self._successful_tool_result(accept_result):
                return StateTransitionPlan()
            intent, rejected = self._state_write_intent(accept_result)
            if intent is None:
                return StateTransitionPlan(rejected_state_write_intents=[rejected] if rejected else [])
            entity_id = intent.entity_id
            current_status = self.state_store.current_status(EntityType.GAME.value, entity_id)
            transition = self._transition_from_state_intent(
                intent,
                semantic_resolution,
                from_status=current_status,
                reason_fallback=validated_action.reason,
                trace_id=trace_id,
            )
            return StateTransitionPlan(transitions=[transition] if transition is not None else [])
        return StateTransitionPlan()

    def _state_write_intent(self, result: ToolResult | None) -> tuple[StateWriteIntent | None, dict[str, Any] | None]:
        intent = result.result.get("state_write_intent") if result and result.result else None
        parsed, errors = parse_state_write_intent(intent)
        if errors:
            return None, {
                "schema": "state_write_intent.v1",
                "tool_name": result.request.tool_name if result else None,
                "idempotency_key": result.request.idempotency_key if result else None,
                "errors": errors,
                "raw_intent": intent,
            }
        return parsed, None

    def _transition_from_state_intent(
        self,
        intent: StateWriteIntent,
        semantic_resolution: SemanticResolution,
        *,
        from_status: str | None,
        reason_fallback: str,
        trace_id: str,
        to_status: str | None = None,
    ) -> StateTransition | None:
        entity_id = intent.entity_id
        target_status = str(to_status or intent.target_status).strip()
        if not entity_id or not target_status:
            return None
        transition = self.state_machine.validate_game_transition(
            entity_id=entity_id,
            from_status=from_status,
            to_status=target_status,
            reason=intent.reason or reason_fallback,
        )
        return self._transition_with_metadata(
            transition,
            **self._state_intent_metadata(intent, semantic_resolution, trace_id=trace_id),
        )

    def _state_intent_metadata(
        self,
        intent: StateWriteIntent,
        semantic_resolution: SemanticResolution,
        *,
        trace_id: str,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "trace_id": trace_id,
            "requirement": intent.requirement or semantic_resolution.game_requirement.to_prompt_dict(),
            "tool_intent_kind": intent.kind,
            "tool_intent_target_status": intent.target_status,
            "state_write_intent_contract": "state_write_intent.v1",
        }
        if intent.metadata:
            metadata["tool_intent_metadata"] = dict(intent.metadata)
        if intent.participant is not None:
            metadata["participant"] = dict(intent.participant)
        if intent.seat_delta is not None:
            metadata["seat_delta"] = dict(intent.seat_delta)
        return metadata

    def _transition_with_metadata(self, transition: StateTransition, **metadata: Any) -> StateTransition:
        return replace(transition, metadata={**transition.metadata, **metadata})

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
