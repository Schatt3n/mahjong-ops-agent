from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .core import AgentCore
from .tools import CandidateSearchTool, CurrentGameSearchTool, PendingOutboxTool
from .workflow_models import (
    ConversationContext,
    RiskLevel,
    SemanticResolution,
    ToolCallRequest,
    ToolExecutionMode,
    ToolName,
    ToolResult,
    ValidatedAction,
)


@dataclass(slots=True)
class ToolOrchestratorConfig:
    allow_read_only: bool = True
    allow_create_pending: bool = True
    allow_state_write: bool = False
    allow_direct_send: bool = False


@dataclass(slots=True)
class ToolOrchestrationResult:
    tool_results: list[ToolResult] = field(default_factory=list)
    skipped_tools: list[str] = field(default_factory=list)

    def result_for(self, tool_name: ToolName) -> ToolResult | None:
        for result in reversed(self.tool_results):
            if result.request.tool_name == tool_name:
                return result
        return None


class ToolOrchestrator:
    """Runs backend-approved tools and normalizes results.

    The orchestrator enforces permissions and idempotency keys. It does not let
    LLM output call arbitrary tools or directly send messages.
    """

    def __init__(
        self,
        core: AgentCore,
        config: ToolOrchestratorConfig | None = None,
        current_games_tool: CurrentGameSearchTool | None = None,
        candidate_tool: CandidateSearchTool | None = None,
        outbox_tool: PendingOutboxTool | None = None,
    ) -> None:
        self.core = core
        self.config = config or ToolOrchestratorConfig()
        self.current_games_tool = current_games_tool or CurrentGameSearchTool()
        self.candidate_tool = candidate_tool or CandidateSearchTool(core)
        self.outbox_tool = outbox_tool or PendingOutboxTool()

    def run(
        self,
        *,
        context: ConversationContext,
        semantic_resolution: SemanticResolution,
        validated_action: ValidatedAction,
        now: datetime | None = None,
    ) -> ToolOrchestrationResult:
        results: list[ToolResult] = []
        scratch: dict[str, Any] = {}
        for tool_name in validated_action.required_tools:
            request = self._request_for_tool(
                tool_name,
                context=context,
                semantic_resolution=semantic_resolution,
                validated_action=validated_action,
                scratch=scratch,
            )
            permission_error = self._permission_error(request, validated_action)
            if permission_error:
                results.append(
                    ToolResult(
                        request=request,
                        called=False,
                        allowed=False,
                        error=permission_error,
                    )
                )
                continue
            result = self._execute(
                request,
                context=context,
                semantic_resolution=semantic_resolution,
                scratch=scratch,
                now=now,
            )
            results.append(result)
        return ToolOrchestrationResult(tool_results=results)

    def _request_for_tool(
        self,
        tool_name: ToolName,
        *,
        context: ConversationContext,
        semantic_resolution: SemanticResolution,
        validated_action: ValidatedAction,
        scratch: dict[str, Any],
    ) -> ToolCallRequest:
        mode = self._mode_for_tool(tool_name)
        arguments: dict[str, Any] = {
            "effective_action": validated_action.effective_action.value,
            "requirement": semantic_resolution.game_requirement.to_prompt_dict(),
        }
        if tool_name == ToolName.CREATE_PENDING_OUTBOX:
            arguments["candidate_count"] = len(scratch.get("candidates") or [])
            arguments["conversation_id"] = context.current_message.conversation_id
        return ToolCallRequest(
            tool_name=tool_name,
            arguments=arguments,
            risk_level=self._risk_for_tool(tool_name, validated_action),
            execution_mode=mode,
            idempotency_key=f"{validated_action.idempotency_key}:{tool_name.value}"
            if validated_action.idempotency_key
            else None,
            reason=validated_action.reason,
        )

    def _execute(
        self,
        request: ToolCallRequest,
        *,
        context: ConversationContext,
        semantic_resolution: SemanticResolution,
        scratch: dict[str, Any],
        now: datetime | None,
    ) -> ToolResult:
        if request.tool_name == ToolName.SEARCH_CURRENT_OPEN_GAMES:
            payload = self.current_games_tool.search(context, semantic_resolution.game_requirement)
            scratch["current_game_matches"] = payload.get("matches") or []
            return ToolResult(request=request, called=True, allowed=True, result=payload)
        if request.tool_name == ToolName.SEARCH_CANDIDATE_CUSTOMERS:
            payload = self.candidate_tool.search(semantic_resolution.game_requirement, now=now)
            scratch["candidates"] = payload.get("candidates") or []
            return ToolResult(request=request, called=True, allowed=True, result=payload)
        if request.tool_name == ToolName.CREATE_PENDING_OUTBOX:
            candidates = list(scratch.get("candidates") or [])
            if not candidates:
                return ToolResult(
                    request=request,
                    called=False,
                    allowed=False,
                    error="CREATE_PENDING_OUTBOX requires candidate search results.",
                )
            payload = self.outbox_tool.create_pending_invites(
                semantic_resolution.game_requirement,
                candidates,
                conversation_id=context.current_message.conversation_id,
                trace_id=context.current_message.trace_id,
            )
            scratch["outbox_drafts"] = payload.get("drafts") or []
            return ToolResult(request=request, called=True, allowed=True, result=payload)
        return ToolResult(
            request=request,
            called=False,
            allowed=False,
            error=f"Tool {request.tool_name.value} is not implemented in controlled orchestrator.",
        )

    def _permission_error(self, request: ToolCallRequest, validated_action: ValidatedAction) -> str | None:
        if request.risk_level == RiskLevel.HIGH:
            return "High risk tool call requires human review."
        if request.execution_mode == ToolExecutionMode.READ_ONLY and not self.config.allow_read_only:
            return "Read-only tools are disabled."
        if request.execution_mode == ToolExecutionMode.CREATE_PENDING and not self.config.allow_create_pending:
            return "Create-pending tools are disabled."
        if request.execution_mode == ToolExecutionMode.STATE_WRITE and not self.config.allow_state_write:
            return "State-write tools are disabled in this orchestrator."
        if request.execution_mode == ToolExecutionMode.DIRECT_SEND:
            return "Direct-send tools are not allowed without explicit human approval." if not self.config.allow_direct_send else None
        if validated_action.risk_level == RiskLevel.HIGH:
            return "Validated action is high risk and cannot call tools automatically."
        return None

    def _mode_for_tool(self, tool_name: ToolName) -> ToolExecutionMode:
        if tool_name in {ToolName.SEARCH_CURRENT_OPEN_GAMES, ToolName.SEARCH_CANDIDATE_CUSTOMERS}:
            return ToolExecutionMode.READ_ONLY
        if tool_name == ToolName.CREATE_PENDING_OUTBOX:
            return ToolExecutionMode.CREATE_PENDING
        if tool_name in {ToolName.CREATE_GAME, ToolName.CLOSE_GAME, ToolName.PROFILE_UPDATE}:
            return ToolExecutionMode.STATE_WRITE
        if tool_name == ToolName.SEND_MESSAGE:
            return ToolExecutionMode.DIRECT_SEND
        return ToolExecutionMode.NOT_CALLED

    def _risk_for_tool(self, tool_name: ToolName, validated_action: ValidatedAction) -> RiskLevel:
        if tool_name == ToolName.SEND_MESSAGE:
            return RiskLevel.HIGH
        if tool_name in {ToolName.CREATE_PENDING_OUTBOX, ToolName.CREATE_GAME, ToolName.CLOSE_GAME, ToolName.PROFILE_UPDATE}:
            return RiskLevel.MEDIUM
        return validated_action.risk_level if validated_action.risk_level == RiskLevel.HIGH else RiskLevel.LOW
