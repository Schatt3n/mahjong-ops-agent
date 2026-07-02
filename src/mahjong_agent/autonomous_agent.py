from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from .controlled_workflow import ControlledWorkflowResult, ControlledWorkflowService, StateTransitionPlan
from .context_builder import WorkflowContextBuildResult
from .models import DEFAULT_TZ, Message
from .observability import TraceEvent, TraceStep, validate_controlled_trace_completeness
from .reply_approval import ReplyApprovalQueueResult
from .state_write_contract import parse_state_write_intent
from .workflow_models import (
    ActionName,
    ActionSource,
    ConversationContext,
    EntityType,
    GameRequirement,
    GameWorkflowStatus,
    GuardedReply,
    ProposedAction,
    ReplyDraft,
    ReplyStatus,
    RiskLevel,
    SemanticResolution,
    SlotSource,
    SlotValue,
    ToolCallRequest,
    ToolExecutionMode,
    ToolName,
    ToolResult,
    UserIntent,
    ValidatedAction,
    WorkflowRun,
    new_workflow_id,
)


DEFAULT_AGENT_PROMPT_PATH = Path(__file__).with_name("prompts") / "autonomous_agent.md"


class AgentLLMClient(Protocol):
    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        trace_id: str,
        timeout_seconds: float,
    ) -> str | dict[str, Any]:
        ...


@dataclass(slots=True)
class AutonomousAgentConfig:
    prompt_path: Path = DEFAULT_AGENT_PROMPT_PATH
    timeout_seconds: float = 8.0
    max_steps: int = 6
    include_prompt_in_raw_response: bool = True


@dataclass(slots=True)
class AgentStep:
    index: int
    decision: str
    goal_status: str
    intent: UserIntent
    reasoning_summary: str
    requirement: GameRequirement
    tool_name: ToolName | None = None
    tool_arguments: dict[str, Any] = field(default_factory=dict)
    reply_text: str = ""
    raw_output: dict[str, Any] = field(default_factory=dict)


class AutonomousAgentService(ControlledWorkflowService):
    """Goal-driven agent loop.

    Unlike ControlledWorkflowService, this service does not ask the backend to
    advance a fixed semantic workflow. The LLM decides the next tool or final
    reply each step; the backend only enforces tool boundaries, state machine
    transitions, idempotency, approval, and traceability.
    """

    def __init__(
        self,
        *,
        agent_llm_client: AgentLLMClient,
        agent_config: AutonomousAgentConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.agent_llm_client = agent_llm_client
        self.agent_config = agent_config or AutonomousAgentConfig()
        self._agent_prompt_cache: str | None = None

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

        agent_steps: list[AgentStep] = []
        tool_results: list[ToolResult] = []
        state_transitions = []
        current_requirement = self._initial_requirement(context)
        scratch: dict[str, Any] = {}
        final_step: AgentStep | None = None

        for step_index in range(1, self.agent_config.max_steps + 1):
            step = self._plan_step(
                context=context,
                trace_id=trace_id,
                step_index=step_index,
                current_requirement=current_requirement,
                tool_results=tool_results,
                agent_steps=agent_steps,
                scratch=scratch,
            )
            agent_steps.append(step)
            current_requirement = self._merge_requirement(current_requirement, step.requirement)
            self._record(
                trace_id,
                "agent_step_planned",
                self._agent_step_payload(step),
                now=now,
            )
            if step.decision == "tool_call" and step.tool_name is not None:
                tool_result = self._execute_agent_tool(
                    step,
                    context=context,
                    trace_id=trace_id,
                    current_requirement=current_requirement,
                    scratch=scratch,
                    now=now,
                )
                tool_results.append(tool_result)
                self._record_tool_results(trace_id, [tool_result], now=now)
                continue
            final_step = step
            break

        if final_step is None:
            final_step = AgentStep(
                index=self.agent_config.max_steps,
                decision="human_review",
                goal_status="needs_human",
                intent=UserIntent.UNKNOWN,
                reasoning_summary="Agent 达到最大步骤数仍未完成目标。",
                requirement=current_requirement,
                reply_text="这个我先转人工确认一下。",
            )

        semantic_resolution = self._semantic_from_agent(final_step, current_requirement, agent_steps)
        run.semantic_resolution = semantic_resolution
        self._record_semantic_resolution(trace_id, semantic_resolution, now=now)

        validated_action = self._validated_from_agent(final_step, tool_results, trace_id=trace_id)
        run.validated_action = validated_action
        self._record_validated_action(trace_id, validated_action, now=now)

        tool_orchestration = self._tool_orchestration_result(tool_results)
        run.tool_results = list(tool_results)

        transition_plan = self._plan_state_transitions(
            validated_action,
            semantic_resolution,
            tool_orchestration,
            trace_id=trace_id,
        )
        state_transitions = self._apply_state_transitions(transition_plan.transitions)
        run.state_transitions = state_transitions
        self._record_state_transitions(
            trace_id,
            state_transitions,
            rejected_state_write_intents=transition_plan.rejected_state_write_intents,
            now=now,
        )

        reply_draft = ReplyDraft(
            text=final_step.reply_text,
            status=ReplyStatus.NEEDS_APPROVAL,
            reasoning_summary=final_step.reasoning_summary,
            source=ActionSource.LLM,
            risk_level=validated_action.risk_level,
            metadata={
                "runtime": "autonomous_agent.v1",
                "goal_status": final_step.goal_status,
                "agent_steps": [self._agent_step_payload(step) for step in agent_steps],
            },
        )
        if final_step.decision == "ignore" and not reply_draft.text:
            reply_draft.status = ReplyStatus.DRAFT
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
                tool_results=tool_results,
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
                "runtime": "autonomous_agent.v1",
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

        return ControlledWorkflowResult(
            run=run,
            context_build=context_build,
            tool_orchestration=tool_orchestration,
            reply_approval=reply_approval,
            trace_events=self.trace_recorder.get_trace(trace_id),
        )

    def _plan_step(
        self,
        *,
        context: ConversationContext,
        trace_id: str,
        step_index: int,
        current_requirement: GameRequirement,
        tool_results: list[ToolResult],
        agent_steps: list[AgentStep],
        scratch: dict[str, Any],
    ) -> AgentStep:
        messages = self._build_agent_messages(
            context=context,
            step_index=step_index,
            current_requirement=current_requirement,
            tool_results=tool_results,
            agent_steps=agent_steps,
            scratch=scratch,
        )
        self._record(
            trace_id,
            "agent_llm_prompt",
            {"messages": messages, "step_index": step_index},
        )
        try:
            raw_output = self.agent_llm_client.complete(
                messages,
                trace_id=trace_id,
                timeout_seconds=self.agent_config.timeout_seconds,
            )
            raw, error = _parse_agent_output(raw_output)
        except TimeoutError as exc:
            raw, error = {}, f"Agent LLM timeout: {exc}"
        except Exception as exc:
            raw, error = {}, f"Agent LLM error: {type(exc).__name__}: {exc}"
        if error:
            return AgentStep(
                index=step_index,
                decision="human_review",
                goal_status="needs_human",
                intent=UserIntent.UNKNOWN,
                reasoning_summary=error,
                requirement=current_requirement,
                reply_text="这个我先转人工确认一下。",
                raw_output={"error": error},
            )
        self._record(
            trace_id,
            "agent_llm_response",
            {"raw_output": raw, "step_index": step_index},
        )
        return self._agent_step_from_raw(step_index, raw, current_requirement)

    def _build_agent_messages(
        self,
        *,
        context: ConversationContext,
        step_index: int,
        current_requirement: GameRequirement,
        tool_results: list[ToolResult],
        agent_steps: list[AgentStep],
        scratch: dict[str, Any],
    ) -> list[dict[str, str]]:
        payload = {
            "task": "autonomous_agent_loop_v1",
            "step_index": step_index,
            "context": context.to_prompt_dict(),
            "current_requirement": current_requirement.to_prompt_dict(),
            "previous_agent_steps": [self._agent_step_payload(step) for step in agent_steps[-4:]],
            "tool_results": [_tool_result_prompt_payload(item) for item in tool_results[-8:]],
            "scratch_summary": {
                "has_candidates": bool(scratch.get("candidates") or []),
                "has_outbox_drafts": bool(scratch.get("outbox_drafts") or []),
                "has_current_game_matches": bool(scratch.get("current_game_matches") or []),
            },
        }
        return [
            {"role": "system", "content": self._agent_prompt_text()},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
        ]

    def _agent_prompt_text(self) -> str:
        if self._agent_prompt_cache is None:
            self._agent_prompt_cache = self.agent_config.prompt_path.read_text(encoding="utf-8")
        return self._agent_prompt_cache

    def _agent_step_from_raw(
        self,
        step_index: int,
        raw: dict[str, Any],
        fallback_requirement: GameRequirement,
    ) -> AgentStep:
        decision = str(raw.get("decision") or "human_review").strip()
        if decision not in {"tool_call", "final_reply", "wait_user", "human_review", "ignore"}:
            decision = "human_review"
        tool_call = raw.get("tool_call") if isinstance(raw.get("tool_call"), dict) else {}
        tool_name = _coerce_tool_name(tool_call.get("tool_name")) if decision == "tool_call" else None
        if decision == "tool_call" and tool_name == ToolName.UNKNOWN:
            decision = "human_review"
        return AgentStep(
            index=step_index,
            decision=decision,
            goal_status=str(raw.get("goal_status") or "in_progress"),
            intent=_coerce_intent(raw.get("intent")),
            reasoning_summary=str(raw.get("reasoning_summary") or ""),
            requirement=_requirement_from_raw(raw.get("requirement"), fallback=fallback_requirement),
            tool_name=tool_name,
            tool_arguments=dict(tool_call.get("arguments") or {}) if isinstance(tool_call.get("arguments"), dict) else {},
            reply_text=str(raw.get("reply_text") or ""),
            raw_output=raw,
        )

    def _execute_agent_tool(
        self,
        step: AgentStep,
        *,
        context: ConversationContext,
        trace_id: str,
        current_requirement: GameRequirement,
        scratch: dict[str, Any],
        now: datetime,
    ) -> ToolResult:
        assert step.tool_name is not None
        request = ToolCallRequest(
            tool_name=step.tool_name,
            arguments={
                **dict(step.tool_arguments),
                "requirement": current_requirement.to_prompt_dict(),
                "conversation_id": context.current_message.conversation_id,
                "trace_id": trace_id,
                "game_id": str(step.tool_arguments.get("game_id") or f"agent_game_{trace_id[-8:]}"),
            },
            risk_level=_risk_for_tool(step.tool_name),
            execution_mode=_mode_for_tool(step.tool_name),
            idempotency_key=f"{trace_id}:agent_step_{step.index}:{step.tool_name.value}",
            reason=step.reasoning_summary,
        )
        permission_error = self._agent_tool_permission_error(request)
        if permission_error:
            return self.tool_orchestrator.execution_ledger.record(
                ToolResult(
                    request=request,
                    called=False,
                    allowed=False,
                    error=permission_error,
                )
            )
        deduplicated = self.tool_orchestrator._deduplicated_result(request)
        if deduplicated is not None:
            return self.tool_orchestrator.execution_ledger.record(deduplicated)
        semantic_resolution = self._semantic_from_agent(step, current_requirement, [step])
        result = self.tool_orchestrator._execute(
            request,
            context=context,
            semantic_resolution=semantic_resolution,
            scratch=scratch,
            now=now,
        )
        return self.tool_orchestrator.execution_ledger.record(result)

    def _agent_tool_permission_error(self, request: ToolCallRequest) -> str | None:
        if request.tool_name == ToolName.SEND_MESSAGE:
            return "Agent 不允许直接发送真实消息，请使用 create_pending_outbox。"
        if request.tool_name == ToolName.UNKNOWN:
            return "Unknown tool."
        if request.execution_mode == ToolExecutionMode.DIRECT_SEND:
            return "Direct send is disabled."
        if request.execution_mode == ToolExecutionMode.STATE_WRITE and not self.tool_orchestrator.config.allow_state_write:
            return "State write tools are disabled."
        if request.execution_mode == ToolExecutionMode.CREATE_PENDING and not self.tool_orchestrator.config.allow_create_pending:
            return "Create-pending tools are disabled."
        if request.execution_mode == ToolExecutionMode.READ_ONLY and not self.tool_orchestrator.config.allow_read_only:
            return "Read-only tools are disabled."
        return None

    def _semantic_from_agent(
        self,
        step: AgentStep,
        requirement: GameRequirement,
        agent_steps: list[AgentStep],
    ) -> SemanticResolution:
        action = _action_from_step(step)
        proposed = ProposedAction(
            name=action,
            source=ActionSource.LLM,
            confidence=0.85 if step.decision != "human_review" else 0.0,
            reason=step.reasoning_summary or "Agent 自主决策",
            arguments=step.tool_arguments,
            risk_level=RiskLevel.HIGH if step.decision == "human_review" else RiskLevel.LOW,
        )
        return SemanticResolution(
            intent=step.intent,
            proposed_action=proposed,
            game_requirement=requirement,
            needs_human_review=step.decision == "human_review",
            reasoning_summary=step.reasoning_summary,
            raw_response={
                "runtime": "autonomous_agent.v1",
                "model_output": step.raw_output,
                "agent_steps": [self._agent_step_payload(item) for item in agent_steps],
            },
        )

    def _validated_from_agent(
        self,
        final_step: AgentStep,
        tool_results: list[ToolResult],
        *,
        trace_id: str,
    ) -> ValidatedAction:
        tools = [item.request.tool_name for item in tool_results]
        effective = _effective_action(final_step, tools)
        risk = RiskLevel.HIGH if final_step.decision == "human_review" else RiskLevel.LOW
        return ValidatedAction(
            proposed_action=ProposedAction(
                name=_action_from_step(final_step),
                source=ActionSource.LLM,
                confidence=0.85 if risk != RiskLevel.HIGH else 0.0,
                reason=final_step.reasoning_summary or "Agent 自主决策",
                risk_level=risk,
            ),
            effective_action=effective,
            allowed=risk != RiskLevel.HIGH,
            code="autonomous_agent_decision" if risk != RiskLevel.HIGH else "human_review_required",
            reason=final_step.reasoning_summary or "Agent 自主决策完成。",
            approval_required=risk == RiskLevel.HIGH,
            risk_level=risk,
            idempotency_key=f"{trace_id}:agent:{effective.value}",
            required_tools=tools,
        )

    def _initial_requirement(self, context: ConversationContext) -> GameRequirement:
        requirement = GameRequirement()
        previous = context.previous_game_requirement()
        if previous is not None:
            requirement.inherit_confirmed_context(previous)
        if context.active_game is not None:
            requirement.inherit_confirmed_context(context.active_game)
        return requirement

    def _merge_requirement(self, base: GameRequirement, incoming: GameRequirement) -> GameRequirement:
        merged = GameRequirement(
            slots=dict(base.slots),
            seats_total=base.seats_total,
            organizer_id=base.organizer_id,
            organizer_name=base.organizer_name,
            candidate_composition_preference=dict(base.candidate_composition_preference),
            notes=list(base.notes),
        )
        for slot in incoming.slots.values():
            merged.set_slot(slot)
        if incoming.candidate_composition_preference:
            merged.candidate_composition_preference.update(incoming.candidate_composition_preference)
        for note in incoming.notes:
            if note not in merged.notes:
                merged.notes.append(note)
        return merged

    def _tool_orchestration_result(self, tool_results: list[ToolResult]) -> Any:
        from .tool_orchestrator import ToolOrchestrationResult

        return ToolOrchestrationResult(tool_results=list(tool_results))

    def _agent_step_payload(self, step: AgentStep) -> dict[str, Any]:
        return {
            "index": step.index,
            "decision": step.decision,
            "goal_status": step.goal_status,
            "intent": step.intent.value,
            "reasoning_summary": step.reasoning_summary,
            "tool_name": step.tool_name.value if step.tool_name else None,
            "reply_text": step.reply_text,
            "requirement": step.requirement.to_prompt_dict(),
        }


def _parse_agent_output(raw_output: str | dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    if isinstance(raw_output, dict):
        return raw_output, None
    text = str(raw_output or "").strip()
    if not text:
        return {}, "Agent LLM returned empty output."
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        return {}, f"Agent LLM output must be a JSON object: {exc}"
    if not isinstance(raw, dict):
        return {}, "Agent LLM JSON root is not an object."
    return raw, None


def _requirement_from_raw(raw: Any, *, fallback: GameRequirement) -> GameRequirement:
    requirement = GameRequirement()
    requirement.inherit_confirmed_context(fallback)
    if not isinstance(raw, dict):
        return requirement
    slots = raw.get("slots")
    if isinstance(slots, dict):
        for name, value in slots.items():
            slot = _slot_from_raw(str(name), value)
            if slot is not None:
                requirement.set_slot(slot)
    preference = raw.get("candidate_composition_preference")
    if isinstance(preference, dict):
        requirement.candidate_composition_preference.update(preference)
    notes = raw.get("notes")
    if isinstance(notes, list):
        requirement.notes.extend(str(item) for item in notes if item not in (None, ""))
    return requirement


def _slot_from_raw(name: str, raw: Any) -> SlotValue | None:
    if isinstance(raw, dict):
        value = raw.get("value")
        if value in (None, "", "unknown"):
            return None
        return SlotValue(
            name=name,
            value=value,
            source=_coerce_slot_source(raw.get("source")),
            confidence=float(raw.get("confidence") or 0.8),
            confirmed=bool(raw.get("confirmed", True)),
            needs_confirmation=bool(raw.get("needs_confirmation", False)),
            evidence=str(raw.get("evidence") or "") or None,
            metadata=dict(raw.get("metadata") or {}) if isinstance(raw.get("metadata"), dict) else {},
        )
    if raw in (None, "", "unknown"):
        return None
    return SlotValue(
        name=name,
        value=raw,
        source=SlotSource.INFERRED,
        confidence=0.7,
        confirmed=True,
        needs_confirmation=False,
    )


def _coerce_slot_source(value: Any) -> SlotSource:
    try:
        return SlotSource(str(value or SlotSource.INFERRED.value))
    except ValueError:
        return SlotSource.INFERRED


def _coerce_intent(value: Any) -> UserIntent:
    try:
        return UserIntent(str(value or UserIntent.UNKNOWN.value))
    except ValueError:
        return UserIntent.UNKNOWN


def _coerce_tool_name(value: Any) -> ToolName:
    try:
        return ToolName(str(value or ToolName.UNKNOWN.value))
    except ValueError:
        return ToolName.UNKNOWN


def _action_from_step(step: AgentStep) -> ActionName:
    if step.decision == "wait_user":
        return ActionName.ASK_CLARIFICATION
    if step.decision == "human_review":
        return ActionName.HUMAN_REVIEW
    if step.decision == "ignore":
        return ActionName.IGNORE
    if step.tool_name == ToolName.SEARCH_CURRENT_OPEN_GAMES:
        return ActionName.SEARCH_EXISTING_GAMES
    if step.tool_name in {ToolName.CREATE_GAME, ToolName.SEARCH_CANDIDATE_CUSTOMERS, ToolName.CREATE_PENDING_OUTBOX}:
        return ActionName.QUEUE_INVITES
    if step.tool_name == ToolName.RECORD_SEAT_ACCEPTANCE:
        return ActionName.ACCEPT_SEAT
    if step.tool_name == ToolName.CLOSE_GAME:
        return ActionName.CLOSE_GAME
    return ActionName.UNKNOWN if step.decision == "tool_call" else ActionName.ASK_CLARIFICATION


def _effective_action(final_step: AgentStep, tools: list[ToolName]) -> ActionName:
    if final_step.decision == "human_review":
        return ActionName.HUMAN_REVIEW
    if final_step.decision == "ignore":
        return ActionName.IGNORE
    if ToolName.RECORD_SEAT_ACCEPTANCE in tools:
        return ActionName.ACCEPT_SEAT
    if ToolName.CLOSE_GAME in tools:
        return ActionName.CLOSE_GAME
    if ToolName.CREATE_GAME in tools:
        return ActionName.QUEUE_INVITES
    if ToolName.SEARCH_CURRENT_OPEN_GAMES in tools:
        return ActionName.SEARCH_EXISTING_GAMES
    if final_step.decision == "wait_user":
        return ActionName.ASK_CLARIFICATION
    return ActionName.ASK_CLARIFICATION


def _mode_for_tool(tool_name: ToolName) -> ToolExecutionMode:
    if tool_name in {ToolName.SEARCH_CURRENT_OPEN_GAMES, ToolName.SEARCH_CANDIDATE_CUSTOMERS}:
        return ToolExecutionMode.READ_ONLY
    if tool_name in {ToolName.CREATE_PENDING_OUTBOX}:
        return ToolExecutionMode.CREATE_PENDING
    if tool_name in {ToolName.CREATE_GAME, ToolName.CLOSE_GAME, ToolName.RECORD_SEAT_ACCEPTANCE, ToolName.PROFILE_UPDATE}:
        return ToolExecutionMode.STATE_WRITE
    return ToolExecutionMode.NOT_CALLED


def _risk_for_tool(tool_name: ToolName) -> RiskLevel:
    if tool_name in {ToolName.CREATE_GAME, ToolName.CLOSE_GAME, ToolName.RECORD_SEAT_ACCEPTANCE, ToolName.PROFILE_UPDATE}:
        return RiskLevel.MEDIUM
    if tool_name == ToolName.CREATE_PENDING_OUTBOX:
        return RiskLevel.HIGH
    return RiskLevel.LOW


def _tool_result_prompt_payload(result: ToolResult) -> dict[str, Any]:
    visibility = _tool_visibility_contract(result)
    payload: dict[str, Any] = {
        "tool_name": result.request.tool_name.value,
        "called": result.called,
        "allowed": result.allowed,
        "error": result.error,
        "visibility_contract": visibility,
    }
    return payload


def _tool_visibility_contract(result: ToolResult) -> dict[str, Any]:
    tool_name = result.request.tool_name
    if not result.allowed or not result.called:
        return {
            "agent_observation": "工具未执行成功。",
            "customer_visible_facts": [],
            "private_facts_not_for_customer": [result.error] if result.error else [],
        }
    if tool_name == ToolName.SEARCH_CURRENT_OPEN_GAMES:
        matches = result.result.get("matches")
        visible_matches = []
        if isinstance(matches, list):
            visible_matches = [
                {
                    "summary": item.get("summary"),
                    "game_id": item.get("game_id") or item.get("id"),
                }
                for item in matches[:3]
                if isinstance(item, dict)
            ]
        return {
            "agent_observation": "已查询当前可拼局。",
            "customer_visible_facts": visible_matches,
            "private_facts_not_for_customer": [],
            "status_flags": {"has_matches": bool(visible_matches)},
        }
    if tool_name == ToolName.SEARCH_CANDIDATE_CUSTOMERS:
        candidates = result.result.get("candidates")
        has_candidates = isinstance(candidates, list) and bool(candidates)
        return {
            "agent_observation": "已找到可邀约候选人。" if has_candidates else "没有找到可邀约候选人。",
            "customer_visible_facts": [],
            "private_facts_not_for_customer": ["候选人数量", "候选人名单", "候选人评分"],
            "status_flags": {"has_candidates": has_candidates},
        }
    if tool_name == ToolName.CREATE_PENDING_OUTBOX:
        drafts = result.result.get("drafts")
        created = isinstance(drafts, list) and bool(drafts)
        return {
            "agent_observation": "已创建待审批邀约草稿。" if created else "没有创建待审批邀约草稿。",
            "customer_visible_facts": ["已按要求帮忙问了"] if created else [],
            "private_facts_not_for_customer": ["草稿数量", "待审批", "outbox", "工具执行细节"],
            "status_flags": {"invite_drafts_created": created},
        }
    if tool_name == ToolName.CREATE_GAME:
        return {
            "agent_observation": "已登记待组局状态写入意图。",
            "customer_visible_facts": [],
            "private_facts_not_for_customer": ["内部局ID", "状态写入意图"],
            "status_flags": {"game_intent_created": True},
        }
    if tool_name == ToolName.RECORD_SEAT_ACCEPTANCE:
        return {
            "agent_observation": "已登记候选人确认入局意图。",
            "customer_visible_facts": ["已帮对方确认入局"],
            "private_facts_not_for_customer": ["内部局ID", "状态写入意图"],
            "status_flags": {"seat_acceptance_recorded": True},
        }
    if tool_name == ToolName.CLOSE_GAME:
        return {
            "agent_observation": "已登记局关闭意图。",
            "customer_visible_facts": [],
            "private_facts_not_for_customer": ["内部局ID", "状态写入意图"],
            "status_flags": {"close_intent_created": True},
        }
    return {
        "agent_observation": "工具已执行。",
        "customer_visible_facts": [],
        "private_facts_not_for_customer": ["工具执行细节"],
        "status_flags": {},
    }
