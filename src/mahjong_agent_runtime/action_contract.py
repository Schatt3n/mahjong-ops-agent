from __future__ import annotations

import json
from typing import Any

from .models import AgentAction


def parse_action(raw_response: str) -> tuple[AgentAction, list[str]]:
    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        return contract_error_action(), [f"response is not valid JSON: {exc.msg}"]
    if not isinstance(payload, dict):
        return contract_error_action(), ["response JSON root must be object"]
    errors = validate_action_contract(payload)
    if errors:
        return contract_error_action(), errors
    return AgentAction.from_payload(payload), []


def validate_action_contract(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in [
        "goal",
        "objective_status",
        "reasoning_summary",
        "reply_to_user",
        "tool_calls",
        "needs_human",
        "stop_reason",
    ]:
        if key not in payload:
            errors.append(f"missing required key: {key}")
    for key in ["goal", "objective_status", "reasoning_summary", "plan_revision_reason", "reply_to_user"]:
        if key in payload and not isinstance(payload.get(key), str):
            errors.append(f"{key} must be string")
    if "objective_state" in payload and not isinstance(payload.get("objective_state"), dict):
        errors.append("objective_state must be object")
    if "objective_plan" in payload and not isinstance(payload.get("objective_plan"), list):
        errors.append("objective_plan must be array")
    errors.extend(validate_objective_plan_contract(payload.get("objective_plan")))
    if "needs_human" in payload and not isinstance(payload.get("needs_human"), bool):
        errors.append("needs_human must be boolean")
    stop_reason = payload.get("stop_reason")
    if not isinstance(stop_reason, dict):
        errors.append("stop_reason must be object")
        stop_reason = {}
    errors.extend(validate_stop_reason_contract(stop_reason, payload.get("objective_status")))
    if "badcase" in payload and payload.get("badcase") is not None:
        errors.append("badcase side-channel is not allowed; call record_badcase tool instead")
    if payload.get("objective_status") not in {"needs_tool", "waiting_user", "completed", "needs_human", "unknown"}:
        errors.append("objective_status is invalid")
    if not isinstance(payload.get("tool_calls", []), list):
        errors.append("tool_calls must be array")
    for index, call in enumerate(payload.get("tool_calls") or [], start=1):
        if not isinstance(call, dict):
            errors.append(f"tool_calls[{index}] must be object")
            continue
        if not isinstance(call.get("name"), str) or not call.get("name"):
            errors.append(f"tool_calls[{index}].name is required")
        if "arguments" not in call:
            errors.append(f"tool_calls[{index}].arguments is required")
        elif not isinstance(call.get("arguments"), dict):
            errors.append(f"tool_calls[{index}].arguments must be object")
        if not isinstance(call.get("reason"), str) or not call.get("reason", "").strip():
            errors.append(f"tool_calls[{index}].reason is required")
        if "idempotency_key" in call and call.get("idempotency_key") is not None and not isinstance(call.get("idempotency_key"), str):
            errors.append(f"tool_calls[{index}].idempotency_key must be string or null")
    status = payload.get("objective_status")
    tool_calls = payload.get("tool_calls") or []
    reply = payload.get("reply_to_user")
    terminal_statuses = {"waiting_user", "completed", "needs_human", "unknown"}
    if status == "needs_tool" and not tool_calls:
        errors.append("needs_tool requires at least one tool_call")
    if status == "needs_tool" and isinstance(reply, str) and reply.strip():
        errors.append("needs_tool requires empty reply_to_user")
    if status in terminal_statuses and tool_calls:
        errors.append(f"{status} must not include tool_calls")
    if status in terminal_statuses and isinstance(reply, str) and not reply.strip():
        errors.append(f"{status} requires non-empty reply_to_user")
    if status == "needs_human" and payload.get("needs_human") is not True:
        errors.append("needs_human objective_status requires needs_human=true")
    if payload.get("needs_human") is True and status != "needs_human":
        errors.append("needs_human=true requires objective_status=needs_human")
    return errors


def validate_objective_plan_contract(raw_plan: Any) -> list[str]:
    if raw_plan is None:
        return []
    if not isinstance(raw_plan, list):
        return []
    errors: list[str] = []
    valid_statuses = {"pending", "in_progress", "done", "blocked", "skipped"}
    for index, raw_step in enumerate(raw_plan, start=1):
        if not isinstance(raw_step, dict):
            errors.append(f"objective_plan[{index}] must be object")
            continue
        step_id = raw_step.get("step_id")
        title = raw_step.get("title")
        status = raw_step.get("status")
        if not isinstance(step_id, str) or not step_id.strip():
            errors.append(f"objective_plan[{index}].step_id is required")
        if not isinstance(title, str) or not title.strip():
            errors.append(f"objective_plan[{index}].title is required")
        if status not in valid_statuses:
            errors.append(f"objective_plan[{index}].status is invalid")
        depends_on = raw_step.get("depends_on")
        if depends_on is not None and not isinstance(depends_on, list):
            errors.append(f"objective_plan[{index}].depends_on must be array when present")
    return errors


def validate_stop_reason_contract(stop_reason: dict[str, Any], status: Any) -> list[str]:
    errors: list[str] = []
    for key in ["can_stop", "why", "pending_work", "depends_on_tool_results"]:
        if key not in stop_reason:
            errors.append(f"stop_reason.{key} is required")
    can_stop = stop_reason.get("can_stop")
    if "can_stop" in stop_reason and not isinstance(can_stop, bool):
        errors.append("stop_reason.can_stop must be boolean")
    why = stop_reason.get("why")
    if "why" in stop_reason and (not isinstance(why, str) or not why.strip()):
        errors.append("stop_reason.why must be non-empty string")
    pending_work = stop_reason.get("pending_work")
    if "pending_work" in stop_reason and not isinstance(pending_work, list):
        errors.append("stop_reason.pending_work must be array")
        pending_work = []
    if isinstance(pending_work, list) and any(not isinstance(item, str) or not item.strip() for item in pending_work):
        errors.append("stop_reason.pending_work items must be non-empty strings")
    depends_on_tool_results = stop_reason.get("depends_on_tool_results")
    if "depends_on_tool_results" in stop_reason and not isinstance(depends_on_tool_results, bool):
        errors.append("stop_reason.depends_on_tool_results must be boolean")
    if status == "needs_tool":
        if can_stop is not False:
            errors.append("needs_tool requires stop_reason.can_stop=false")
        if isinstance(pending_work, list) and not pending_work:
            errors.append("needs_tool requires non-empty stop_reason.pending_work")
    if status in {"waiting_user", "completed", "needs_human", "unknown"} and can_stop is not True:
        errors.append(f"{status} requires stop_reason.can_stop=true")
    return errors


def contract_error_action() -> AgentAction:
    return AgentAction(
        goal="contract_error",
        objective_status="needs_human",
        reasoning_summary="模型输出不符合 AgentAction 合同，后端拒绝执行。",
        reply_to_user="这个我先转人工确认一下。",
        needs_human=True,
        stop_reason={
            "can_stop": True,
            "why": "模型输出合同错误，后端不能安全继续执行。",
            "pending_work": [],
            "depends_on_tool_results": False,
        },
    )
