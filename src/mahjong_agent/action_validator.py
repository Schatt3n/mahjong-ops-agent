from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from .action_arguments_contract import validate_action_arguments_contract
from .slot_matching import slot_values_compatible
from .state_machine import StateMachine
from .tool_permissions import validate_required_tools_for_action
from .workflow_models import (
    ActionName,
    ConversationContext,
    GameRequirement,
    RiskLevel,
    SemanticResolution,
    SlotSource,
    ToolName,
    UserIntent,
    ValidatedAction,
)


CRITICAL_CREATE_GAME_SLOTS: tuple[str, ...] = (
    "game_type",
    "stake",
    "start_time_mode",
    "party_size",
    "smoke",
    "duration_mode",
)


@dataclass(slots=True)
class ActionValidatorConfig:
    min_confidence_for_state_action: float = 0.62
    min_confidence_for_contextual_create: float = 0.72
    require_explicit_or_context_create: bool = True
    high_risk_requires_human: bool = True
    create_game_approval_required: bool = True
    queue_invites_approval_required: bool = True


@dataclass(slots=True)
class ActionValidationInput:
    context: ConversationContext
    semantic_resolution: SemanticResolution
    tool_hints: dict[str, Any] = field(default_factory=dict)


class ActionValidator:
    """Backend guardrail for LLM proposed actions.

    It validates action legality, required slots, state-machine preconditions,
    risk level, and tool permissions. It does not execute tools or mutate state.
    """

    def __init__(
        self,
        config: ActionValidatorConfig | None = None,
        state_machine: StateMachine | None = None,
    ) -> None:
        self.config = config or ActionValidatorConfig()
        self.state_machine = state_machine or StateMachine()

    def validate(
        self,
        context: ConversationContext,
        semantic_resolution: SemanticResolution,
        *,
        tool_hints: dict[str, Any] | None = None,
    ) -> ValidatedAction:
        validation_input = ActionValidationInput(
            context=context,
            semantic_resolution=semantic_resolution,
            tool_hints=tool_hints or {},
        )
        if semantic_resolution.needs_human_review or semantic_resolution.proposed_action.risk_level == RiskLevel.HIGH:
            return self._validated(
                validation_input,
                effective_action=ActionName.HUMAN_REVIEW,
                allowed=False,
                code="human_review_required",
                reason="语义解析标记为高风险或需要人工审核，后端拒绝自动执行。",
                risk_level=RiskLevel.HIGH,
                approval_required=True,
            )

        proposed = semantic_resolution.proposed_action.name
        confidence = semantic_resolution.proposed_action.confidence
        action_argument_errors = validate_action_arguments_contract(
            proposed,
            semantic_resolution.proposed_action.arguments,
        )
        if action_argument_errors:
            return self._validated(
                validation_input,
                effective_action=ActionName.HUMAN_REVIEW,
                allowed=False,
                code="action_arguments_contract_invalid",
                reason="LLM 动作参数 contract 不合法：" + "；".join(action_argument_errors),
                risk_level=RiskLevel.HIGH,
                approval_required=True,
            )
        if proposed in self._stateful_actions() and confidence < self.config.min_confidence_for_state_action:
            return self._validated(
                validation_input,
                effective_action=ActionName.ASK_CLARIFICATION,
                allowed=False,
                code="low_confidence_downgrade",
                reason=f"LLM 动作置信度 {confidence:.2f} 低于阈值，降级为追问。",
                missing_slots=semantic_resolution.game_requirement.missing_required_slots(),
            )

        if proposed == ActionName.SEARCH_EXISTING_GAMES:
            return self._validate_search_existing_games(validation_input)
        if proposed in {ActionName.CREATE_GAME, ActionName.QUEUE_INVITES}:
            return self._validate_create_game(validation_input)
        if proposed == ActionName.ASK_CREATE_CONFIRMATION:
            if semantic_resolution.intent == UserIntent.FIND_PLAYERS:
                missing_slots = semantic_resolution.game_requirement.missing_required_slots(CRITICAL_CREATE_GAME_SLOTS)
                if missing_slots:
                    return self._validated(
                        validation_input,
                        effective_action=ActionName.ASK_CLARIFICATION,
                        allowed=True,
                        code="confirmed_create_missing_slots",
                        reason="用户已经确认要新组局，但组局关键信息不足，后端改为追问缺失信息。",
                        missing_slots=missing_slots,
                    )
                return self._validate_create_game(validation_input)
            return self._validated(
                validation_input,
                effective_action=ActionName.ASK_CREATE_CONFIRMATION,
                allowed=True,
                code="ask_create_confirmation",
                reason="用户只是咨询现有局，当前应确认是否要新组。",
            )
        if proposed == ActionName.ASK_CLARIFICATION:
            return self._validated(
                validation_input,
                effective_action=ActionName.ASK_CLARIFICATION,
                allowed=True,
                code="ask_clarification",
                reason="LLM 提案为补充关键信息，后端允许追问。",
                missing_slots=semantic_resolution.game_requirement.missing_required_slots(),
            )
        if proposed == ActionName.MATCH_EXISTING_GAME:
            return self._validate_match_existing_game(validation_input)
        if proposed in {ActionName.JOIN_GAME, ActionName.ACCEPT_SEAT}:
            return self._validated(
                validation_input,
                effective_action=ActionName.ACCEPT_SEAT,
                allowed=True,
                code="candidate_accept_seat",
                reason="候选人表达确认加入，后续需由状态机和邀约记录校验座位。",
                required_tools=[ToolName.RECORD_SEAT_ACCEPTANCE],
                approval_required=True,
                risk_level=RiskLevel.MEDIUM,
            )
        if proposed in {ActionName.CANCEL_GAME, ActionName.CLOSE_GAME}:
            return self._validated(
                validation_input,
                effective_action=ActionName.CLOSE_GAME,
                allowed=True,
                code="cancel_or_close_game",
                reason="用户表达取消或关闭相关意图，后续需校验发起人身份和当前局状态。",
                required_tools=[ToolName.CLOSE_GAME],
                approval_required=True,
                risk_level=RiskLevel.MEDIUM,
            )
        if proposed == ActionName.HUMAN_REVIEW:
            return self._validated(
                validation_input,
                effective_action=ActionName.HUMAN_REVIEW,
                allowed=False,
                code="llm_requested_human_review",
                reason="LLM 明确建议转人工。",
                risk_level=RiskLevel.HIGH,
                approval_required=True,
            )
        if proposed == ActionName.IGNORE:
            return self._validated(
                validation_input,
                effective_action=ActionName.IGNORE,
                allowed=True,
                code="ignore",
                reason="用户消息与麻将馆运营无关或无需回复。",
            )
        return self._validated(
            validation_input,
            effective_action=ActionName.ASK_CLARIFICATION,
            allowed=False,
            code="unknown_action",
            reason="LLM 动作不在后端白名单内，拒绝执行并降级为追问。",
        )

    def _validate_search_existing_games(self, data: ActionValidationInput) -> ValidatedAction:
        match = self._find_existing_match(data.context, data.semantic_resolution.game_requirement)
        if match is not None:
            return self._validated(
                data,
                effective_action=ActionName.MATCH_EXISTING_GAME,
                allowed=True,
                code="existing_game_matched",
                reason="当前局池存在可承接局，优先匹配现有局。",
                required_tools=[ToolName.SEARCH_CURRENT_OPEN_GAMES],
                notes=["后续回复应基于匹配局结果生成，不应创建新局。"],
            )
        return self._validated(
            data,
            effective_action=ActionName.ASK_CREATE_CONFIRMATION,
            allowed=True,
            code="no_existing_match_ask_create",
            reason="未找到匹配现有局，且用户尚未确认新组。",
            required_tools=[ToolName.SEARCH_CURRENT_OPEN_GAMES],
        )

    def _validate_create_game(self, data: ActionValidationInput) -> ValidatedAction:
        requirement = data.semantic_resolution.game_requirement
        missing_slots = requirement.missing_required_slots(CRITICAL_CREATE_GAME_SLOTS)
        if missing_slots:
            return self._validated(
                data,
                effective_action=ActionName.ASK_CLARIFICATION,
                allowed=False,
                code="critical_slots_missing",
                reason="组局关键信息不足，后端拒绝创建局。",
                missing_slots=missing_slots,
            )
        match = self._find_existing_match(data.context, requirement)
        if match is not None:
            return self._validated(
                data,
                effective_action=ActionName.MATCH_EXISTING_GAME,
                allowed=True,
                code="existing_game_preferred",
                reason="当前已有可承接局，优先匹配现有局而不是新建。",
                required_tools=[ToolName.SEARCH_CURRENT_OPEN_GAMES],
                notes=["这可以避免 search_existing_game 和 create_game 抢优先级。"],
            )
        if self.config.require_explicit_or_context_create and not self._has_create_authority(data):
            return self._validated(
                data,
                effective_action=ActionName.ASK_CREATE_CONFIRMATION,
                allowed=False,
                code="create_not_explicit_or_contextual",
                reason="用户没有明确要求老板新组，画像或推断不足以创建局。",
            )
        return self._validated(
            data,
            effective_action=ActionName.QUEUE_INVITES,
            allowed=True,
            code="queue_invites_after_create_validation",
            reason="LLM 提出新组局，后端校验关键槽位和局池后，允许进入待审批邀约。",
            required_tools=[
                ToolName.SEARCH_CURRENT_OPEN_GAMES,
                ToolName.SEARCH_CANDIDATE_CUSTOMERS,
                ToolName.CREATE_PENDING_OUTBOX,
                ToolName.CREATE_GAME,
            ],
            approval_required=self.config.queue_invites_approval_required,
            risk_level=RiskLevel.MEDIUM,
        )

    def _validate_match_existing_game(self, data: ActionValidationInput) -> ValidatedAction:
        match = self._find_existing_match(data.context, data.semantic_resolution.game_requirement)
        if match is None:
            return self._validated(
                data,
                effective_action=ActionName.SEARCH_EXISTING_GAMES,
                allowed=False,
                code="match_without_existing_game",
                reason="LLM 提出匹配现有局，但上下文中没有可匹配局，降级为查局。",
                required_tools=[ToolName.SEARCH_CURRENT_OPEN_GAMES],
            )
        return self._validated(
            data,
            effective_action=ActionName.MATCH_EXISTING_GAME,
            allowed=True,
            code="match_existing_game",
            reason="上下文中存在可匹配局，允许后续生成确认回复。",
            required_tools=[ToolName.SEARCH_CURRENT_OPEN_GAMES],
        )

    def _has_create_authority(self, data: ActionValidationInput) -> bool:
        resolution = data.semantic_resolution
        if resolution.intent == UserIntent.FIND_PLAYERS:
            return True
        slots = resolution.game_requirement.slots.values()
        explicit_slots = [slot for slot in slots if slot.source == SlotSource.EXPLICIT and slot.confidence >= 0.75]
        return len(explicit_slots) >= 3 and resolution.proposed_action.confidence >= self.config.min_confidence_for_contextual_create

    def _find_existing_match(
        self,
        context: ConversationContext,
        requirement: GameRequirement,
    ) -> GameRequirement | None:
        for open_game in context.open_games:
            if self._game_matches(requirement, open_game):
                return open_game
        return None

    def _game_matches(self, requirement: GameRequirement, open_game: GameRequirement) -> bool:
        for slot_name in ("game_type", "stake", "smoke"):
            requested = requirement.slot(slot_name)
            offered = open_game.slot(slot_name)
            if (
                requested
                and requested.usable
                and offered
                and offered.usable
                and not slot_values_compatible(requested, offered, slot_name=slot_name)
            ):
                return False
        missing = open_game.slot("missing_count")
        if missing is None or not missing.usable:
            return True
        try:
            return int(missing.value) > 0
        except (TypeError, ValueError):
            return True

    def _stateful_actions(self) -> set[ActionName]:
        return {
            ActionName.CREATE_GAME,
            ActionName.QUEUE_INVITES,
            ActionName.MATCH_EXISTING_GAME,
            ActionName.JOIN_GAME,
            ActionName.ACCEPT_SEAT,
            ActionName.CANCEL_GAME,
            ActionName.CLOSE_GAME,
        }

    def _validated(
        self,
        data: ActionValidationInput,
        *,
        effective_action: ActionName,
        allowed: bool,
        code: str,
        reason: str,
        missing_slots: list[str] | None = None,
        approval_required: bool = False,
        risk_level: RiskLevel | None = None,
        required_tools: list[ToolName] | None = None,
        notes: list[str] | None = None,
    ) -> ValidatedAction:
        risk = risk_level or data.semantic_resolution.proposed_action.risk_level
        if self.config.high_risk_requires_human and risk == RiskLevel.HIGH:
            approval_required = True
        final_required_tools = list(required_tools or [])
        if allowed and risk != RiskLevel.HIGH and self._has_profile_observations(data):
            if ToolName.PROFILE_UPDATE not in final_required_tools:
                final_required_tools.append(ToolName.PROFILE_UPDATE)
        tool_permission_errors = validate_required_tools_for_action(effective_action, final_required_tools)
        if tool_permission_errors:
            return ValidatedAction(
                proposed_action=data.semantic_resolution.proposed_action,
                effective_action=ActionName.HUMAN_REVIEW,
                allowed=False,
                code="tool_permission_denied",
                reason="后端工具权限校验失败：" + "；".join(tool_permission_errors),
                missing_slots=missing_slots or [],
                approval_required=True,
                risk_level=RiskLevel.HIGH,
                idempotency_key=self._idempotency_key(data, ActionName.HUMAN_REVIEW),
                notes=[*(notes or []), *tool_permission_errors],
                required_tools=[],
            )
        return ValidatedAction(
            proposed_action=data.semantic_resolution.proposed_action,
            effective_action=effective_action,
            allowed=allowed,
            code=code,
            reason=reason,
            missing_slots=missing_slots or [],
            approval_required=approval_required,
            risk_level=risk,
            idempotency_key=self._idempotency_key(data, effective_action),
            notes=notes or [],
            required_tools=final_required_tools,
        )

    def _has_profile_observations(self, data: ActionValidationInput) -> bool:
        model_output = data.semantic_resolution.raw_response.get("model_output")
        if not isinstance(model_output, dict):
            return False
        observations = model_output.get("profile_observations")
        return isinstance(observations, list) and any(isinstance(item, dict) for item in observations)

    def _idempotency_key(self, data: ActionValidationInput, effective_action: ActionName) -> str:
        payload = {
            "trace_id": data.context.current_message.trace_id,
            "message_id": data.context.current_message.message_id,
            "conversation_id": data.context.current_message.conversation_id,
            "sender_id": data.context.current_message.sender_id,
            "effective_action": effective_action.value,
            "proposed_action": data.semantic_resolution.proposed_action.name.value,
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:24]
        return f"action_{digest}"
