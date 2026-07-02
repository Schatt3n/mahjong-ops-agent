from __future__ import annotations

from typing import Any

from .controlled_workflow import ControlledWorkflowResult
from .observability import to_trace_payload
from .workflow_models import (
    ActionName,
    GameRequirement,
    ReplyStatus,
    SlotValue,
    ToolName,
    ToolResult,
)


def project_controlled_result_for_trial(result: ControlledWorkflowResult) -> dict[str, Any]:
    """Project controlled workflow output into the current trial-console shape.

    This adapter is intentionally read-only. It does not parse user text, choose
    actions, call tools, or mutate state. It lets the legacy trial page migrate
    to ControlledWorkflowService without forcing a frontend rewrite first.
    """

    run = result.run
    semantic = run.semantic_resolution
    validated = run.validated_action
    draft = run.reply_draft
    guarded = run.guarded_reply
    requirement = semantic.game_requirement if semantic else GameRequirement()
    return {
        "workflow": {
            "engine": _workflow_engine(result),
            "trace_id": run.trace_id,
            "approval_required": bool(validated.approval_required) if validated else False,
        },
        "parsed": _parsed_payload(result, requirement),
        "suggested_reply": {
            "text": guarded.final_text if guarded else "",
            "status": _reply_status_cn(draft.status if draft else None),
            "source": draft.source.value if draft else "unknown",
            "reasoning_summary": draft.reasoning_summary if draft else "",
            "guard_changed": guarded.changed if guarded else False,
            "guard_reasons": list(guarded.guard_reasons) if guarded else [],
            "approval_required": bool(validated.approval_required) if validated else False,
        },
        "group_draft": "",
        "outbox": _outbox_payload(result),
        "pool_matches": _pool_matches_payload(result),
        "tool_results": _tool_results_payload(run.tool_results),
        "state": {
            "games": _state_games_payload(result),
            "state_transitions": [to_trace_payload(item) for item in run.state_transitions],
        },
        "agent_actions": _agent_actions_payload(result),
        "trace": [event.to_dict() for event in result.trace_events],
    }


def _parsed_payload(result: ControlledWorkflowResult, requirement: GameRequirement) -> dict[str, Any]:
    semantic = result.run.semantic_resolution
    validated = result.run.validated_action
    current_message = result.run.context.current_message
    return {
        "conversation_id": current_message.conversation_id,
        "message_id": current_message.message_id,
        "sender_id": current_message.sender_id,
        "sender_name": current_message.sender_name,
        "user_intent": semantic.intent.value if semantic else "unknown",
        "intent_action": validated.effective_action.value if validated else "unknown",
        "raw_action": semantic.proposed_action.name.value if semantic else "unknown",
        "confidence": semantic.proposed_action.confidence if semantic else 0.0,
        "reasoning_summary": semantic.reasoning_summary if semantic else "",
        "missing_fields": list(validated.missing_slots) if validated else [],
        "semantic_action": {
            "source": semantic.proposed_action.source.value if semantic else "unknown",
            "proposed_action": semantic.proposed_action.name.value if semantic else "unknown",
            "effective_action": validated.effective_action.value if validated else "unknown",
            "allowed": validated.allowed if validated else False,
            "code": validated.code if validated else "",
            "reason": validated.reason if validated else "",
            "risk_level": validated.risk_level.value if validated else "low",
            "approval_required": validated.approval_required if validated else False,
            "idempotency_key": validated.idempotency_key if validated else None,
            "required_tools": [tool.value for tool in validated.required_tools] if validated else [],
        },
        "game_type": _slot_value(requirement, "game_type"),
        "variant": _slot_value(requirement, "variant"),
        "level": _slot_value(requirement, "stake"),
        "stake": _slot_value(requirement, "stake"),
        "start_time": _slot_value(requirement, "start_at") or _slot_value(requirement, "start_time_mode"),
        "start_time_mode": _slot_value(requirement, "start_time_mode"),
        "duration_hours": _slot_value(requirement, "duration_hours"),
        "duration_mode": _slot_value(requirement, "duration_mode"),
        "current_player_count": _slot_value(requirement, "current_player_count"),
        "missing_count": _slot_value(requirement, "missing_count"),
        "smoke": _slot_value(requirement, "smoke"),
        "rules": _rules_from_requirement(requirement),
        "slots": {
            name: slot.to_prompt_dict()
            for name, slot in requirement.slots.items()
        },
        "candidate_composition_preference": dict(requirement.candidate_composition_preference),
        "summary": _requirement_summary(requirement),
    }


def _workflow_engine(result: ControlledWorkflowResult) -> str:
    semantic = result.run.semantic_resolution
    raw = semantic.raw_response if semantic else {}
    runtime = raw.get("runtime") if isinstance(raw, dict) else None
    return str(runtime or "controlled_workflow.v1")


def _outbox_payload(result: ControlledWorkflowResult) -> list[dict[str, Any]]:
    tool_result = result.tool_orchestration.result_for(ToolName.CREATE_PENDING_OUTBOX)
    drafts = (tool_result.result.get("drafts") if tool_result and tool_result.allowed else []) or []
    outbox: list[dict[str, Any]] = []
    for draft in drafts:
        metadata = draft.get("metadata") if isinstance(draft.get("metadata"), dict) else {}
        outbox.append(
            {
                "id": draft.get("id"),
                "trace_id": draft.get("trace_id"),
                "conversation_id": draft.get("conversation_id"),
                "customer_id": draft.get("target_customer_id"),
                "customer_name": draft.get("target_display_name"),
                "message_text": draft.get("message_text"),
                "status": "待审批" if draft.get("status") == "pending_approval" else draft.get("status"),
                "approval_status": "待审批",
                "approval_required": True,
                "direct_send_executed": False,
                "draft_source": draft.get("source") or "tool_orchestrator",
                "score": metadata.get("candidate_score"),
                "reasons": list(metadata.get("candidate_reasons") or []),
                "warnings": list(metadata.get("candidate_warnings") or []),
                "metadata": metadata,
            }
        )
    return outbox


def _pool_matches_payload(result: ControlledWorkflowResult) -> list[dict[str, Any]]:
    tool_result = result.tool_orchestration.result_for(ToolName.SEARCH_CURRENT_OPEN_GAMES)
    matches = (tool_result.result.get("matches") if tool_result and tool_result.allowed else []) or []
    payload: list[dict[str, Any]] = []
    for match in matches:
        game = match.get("game_requirement") if isinstance(match.get("game_requirement"), dict) else {}
        slots = game.get("slots") if isinstance(game.get("slots"), dict) else {}
        payload.append(
            {
                "summary": match.get("summary"),
                "score": match.get("score"),
                "reasons": list(match.get("reasons") or []),
                "game_requirement": game,
                "game_type": _slot_prompt_value(slots.get("game_type")),
                "level": _slot_prompt_value(slots.get("stake")),
                "stake": _slot_prompt_value(slots.get("stake")),
                "start_time": _slot_prompt_value(slots.get("start_at")) or _slot_prompt_value(slots.get("start_time_mode")),
                "start_time_mode": _slot_prompt_value(slots.get("start_time_mode")),
                "missing_count": _slot_prompt_value(slots.get("missing_count")),
                "smoke": _slot_prompt_value(slots.get("smoke")),
            }
        )
    return payload


def _tool_results_payload(tool_results: list[ToolResult]) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    for item in tool_results:
        name = item.request.tool_name.value
        payload[name] = {
            "tool_name": name,
            "called": item.called,
            "allowed": item.allowed,
            "error": item.error,
            "deduplicated": item.deduplicated,
            "risk_level": item.request.risk_level.value,
            "execution_mode": item.request.execution_mode.value,
            "idempotency_key": item.request.idempotency_key,
            "result_count": item.result.get("result_count"),
            "result": to_trace_payload(item.result),
        }
    return payload


def _state_games_payload(result: ControlledWorkflowResult) -> list[dict[str, Any]]:
    semantic = result.run.semantic_resolution
    validated = result.run.validated_action
    if not semantic or not validated:
        return []
    if validated.effective_action not in {ActionName.QUEUE_INVITES, ActionName.MATCH_EXISTING_GAME}:
        return []
    latest_transition = result.run.state_transitions[-1] if result.run.state_transitions else None
    return [
        {
            "id": latest_transition.entity_id if latest_transition else validated.idempotency_key,
            "status": latest_transition.to_status if latest_transition else "pending",
            "requirement": semantic.game_requirement.to_prompt_dict(),
            "summary": _requirement_summary(semantic.game_requirement),
            "outbox_count": len(_outbox_payload(result)),
        }
    ]


def _agent_actions_payload(result: ControlledWorkflowResult) -> list[dict[str, Any]]:
    validated = result.run.validated_action
    if not validated:
        return []
    return [
        {
            "protocol": "controlled_workflow.v1",
            "stage": "action_validation",
            "proposed_action": validated.proposed_action.name.value,
            "effective_action": validated.effective_action.value,
            "allowed": validated.allowed,
            "approval_required": validated.approval_required,
            "risk_level": validated.risk_level.value,
            "idempotency_key": validated.idempotency_key,
            "required_tools": [tool.value for tool in validated.required_tools],
            "validated_actions": [
                {
                    "tool_name": tool.value,
                    "allowed": True,
                    "approval_required": validated.approval_required,
                    "idempotency_key": f"{validated.idempotency_key}:{tool.value}" if validated.idempotency_key else None,
                    "ledger_status": "projected",
                }
                for tool in validated.required_tools
            ],
        }
    ]


def _slot_value(requirement: GameRequirement, name: str) -> Any:
    slot = requirement.slot(name)
    return slot.value if slot else None


def _slot_prompt_value(slot: Any) -> Any:
    return slot.get("value") if isinstance(slot, dict) else None


def _rules_from_requirement(requirement: GameRequirement) -> list[str]:
    rules: list[str] = []
    for name in ("game_type", "variant", "smoke", "duration_mode"):
        value = _slot_value(requirement, name)
        if value:
            rules.append(str(value))
    raw_rules = _slot_value(requirement, "rules")
    if isinstance(raw_rules, list):
        rules.extend(str(item) for item in raw_rules)
    return list(dict.fromkeys(rules))


def _requirement_summary(requirement: GameRequirement) -> str:
    parts = [
        _slot_value(requirement, "game_type"),
        _slot_value(requirement, "variant"),
        _slot_value(requirement, "stake"),
        _slot_value(requirement, "start_at") or _slot_value(requirement, "start_time_mode"),
        f"缺{_slot_value(requirement, 'missing_count')}" if _slot_value(requirement, "missing_count") is not None else None,
        _slot_value(requirement, "smoke"),
    ]
    return " ".join(str(part) for part in parts if part not in (None, ""))


def _reply_status_cn(status: ReplyStatus | None) -> str:
    if status == ReplyStatus.NEEDS_APPROVAL:
        return "待审批"
    if status == ReplyStatus.APPROVED:
        return "已审批"
    if status == ReplyStatus.REJECTED:
        return "已拒绝"
    if status == ReplyStatus.DRAFT:
        return "草稿"
    if status == ReplyStatus.GUARDED:
        return "已安全改写"
    return "待审批"
