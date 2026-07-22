from __future__ import annotations

"""主模型 AgentAction 输出合同校验。

设计理念：
- LLM 可以决定下一步做什么，但必须用后端规定的结构化合同表达。
- 合同校验只检查结构、状态约束和安全不变量，不写麻将业务 if-else。
- 合同不合法时，后端不执行任何工具，而是把错误作为工具结果回喂模型修正。
"""

import json
from typing import Any

from .models import AgentAction


def parse_action(raw_response: str) -> tuple[AgentAction, list[str]]:
    """把模型原始文本解析为 AgentAction。

    返回值始终包含一个 AgentAction；如果解析失败，返回 contract_error_action 和错误列表，
    这样主 loop 可以统一处理，不需要在每个调用点写异常分支。
    """

    action, errors, _ = parse_action_with_repairs(raw_response)
    return action, errors


def parse_action_with_repairs(raw_response: str) -> tuple[AgentAction, list[str], list[dict[str, Any]]]:
    """Parse an action and apply only unambiguous, domain-neutral contract repairs.

    A model can occasionally reason that a turn should stop while accidentally leaving `objective_status=needs_tool`.
    When every other structural signal agrees that the action is terminal, normalizing that one discriminator is safer
    and cheaper than another model call. Repairs never invent tool arguments, business facts or customer-visible text.
    """

    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        return contract_error_action(), [f"response is not valid JSON: {exc.msg}"], []
    if not isinstance(payload, dict):
        return contract_error_action(), ["response JSON root must be object"], []
    payload, repairs = normalize_action_contract(payload)
    errors = validate_action_contract(payload)
    if errors:
        return contract_error_action(), errors, repairs
    return AgentAction.from_payload(payload), [], repairs


def normalize_action_contract(payload: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Repair a single contradictory status only when all terminal invariants agree.

    This is deliberately narrower than semantic fallback logic. It does not inspect Mahjong terms or infer intent; it
    only reconciles the enum discriminator with existing `tool_calls`, `reply_to_user`, `needs_human` and `stop_reason`.
    """

    normalized = dict(payload)
    repairs: list[dict[str, Any]] = []
    raw_tool_calls = normalized.get("tool_calls")
    if isinstance(raw_tool_calls, list):
        graph_mode = any(isinstance(raw_call, dict) and "depends_on" in raw_call for raw_call in raw_tool_calls)
        if graph_mode:
            normalized_calls: list[Any] = []
            for index, raw_call in enumerate(raw_tool_calls, start=1):
                if not isinstance(raw_call, dict):
                    normalized_calls.append(raw_call)
                    continue
                call = dict(raw_call)
                if call.get("depends_on") is None:
                    call["depends_on"] = []
                    repairs.append(
                        {
                            "field": f"tool_calls[{index}].depends_on",
                            "from": None,
                            "to": [],
                            "reason": "an omitted tool dependency is structurally equivalent to an independent call",
                        }
                    )
                normalized_calls.append(call)
            normalized["tool_calls"] = normalized_calls
    raw_plan = normalized.get("objective_plan")
    if isinstance(raw_plan, list):
        normalized_plan: list[Any] = []
        for index, raw_step in enumerate(raw_plan, start=1):
            if not isinstance(raw_step, dict):
                normalized_plan.append(raw_step)
                continue
            step = dict(raw_step)
            step_id = step.get("step_id")
            if not isinstance(step_id, str) or not step_id.strip():
                step["step_id"] = str(index)
                repairs.append(
                    {
                        "field": f"objective_plan[{index}].step_id",
                        "from": step_id,
                        "to": str(index),
                        "reason": "plan step identifiers are structural metadata and can be assigned deterministically",
                    }
                )
            depends_on = step.get("depends_on")
            if depends_on is None:
                step["depends_on"] = []
                repairs.append(
                    {
                        "field": f"objective_plan[{index}].depends_on",
                        "from": None,
                        "to": [],
                        "reason": "an absent plan dependency is structurally equivalent to an empty dependency list",
                    }
                )
            elif isinstance(depends_on, str):
                step["depends_on"] = [depends_on] if depends_on.strip() else []
                repairs.append(
                    {
                        "field": f"objective_plan[{index}].depends_on",
                        "from": depends_on,
                        "to": list(step["depends_on"]),
                        "reason": "a single dependency identifier can be wrapped as the required array without changing meaning",
                    }
                )
            normalized_plan.append(step)
        normalized["objective_plan"] = normalized_plan
    stop_reason = normalized.get("stop_reason")
    tool_calls = normalized.get("tool_calls")
    reply = normalized.get("reply_to_user")
    structurally_terminal = (
        normalized.get("objective_status") == "needs_tool"
        and isinstance(tool_calls, list)
        and not tool_calls
        and isinstance(reply, str)
        and bool(reply.strip())
        and normalized.get("needs_human") is False
        and isinstance(stop_reason, dict)
        and stop_reason.get("can_stop") is True
        and isinstance(stop_reason.get("pending_work"), list)
        and not stop_reason.get("pending_work")
    )
    if structurally_terminal:
        normalized["objective_status"] = "completed"
        repairs.append(
            {
                "field": "objective_status",
                "from": "needs_tool",
                "to": "completed",
                "reason": "all other contract fields describe a terminal reply with no tool work",
            }
        )
    return normalized, repairs


def validate_action_contract(payload: dict[str, Any]) -> list[str]:
    """校验主模型输出是否符合 AgentAction 合同。

    重点检查 required keys、字段类型、tool_calls 结构、objective_status 与 reply/tool 的互斥关系，
    以及 stop_reason 是否能解释为什么继续或停止。
    """

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
    self_assessment = payload.get("self_assessment")
    if self_assessment is not None:
        if not isinstance(self_assessment, dict):
            errors.append("self_assessment must be object or null")
        else:
            progress = self_assessment.get("progress")
            should_escalate = self_assessment.get("should_escalate")
            if progress not in {"advancing", "stalled", "regressing"}:
                errors.append("self_assessment.progress is invalid")
            if not isinstance(should_escalate, bool):
                errors.append("self_assessment.should_escalate must be boolean")
            if should_escalate is True and progress != "stalled":
                errors.append("self_assessment.should_escalate=true requires progress=stalled")
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
        if "call_id" in call and (not isinstance(call.get("call_id"), str) or not call.get("call_id", "").strip()):
            errors.append(f"tool_calls[{index}].call_id must be non-empty string")
        if "depends_on" in call:
            dependencies = call.get("depends_on")
            if not isinstance(dependencies, list):
                errors.append(f"tool_calls[{index}].depends_on must be array")
            elif any(not isinstance(item, str) or not item.strip() for item in dependencies):
                errors.append(f"tool_calls[{index}].depends_on items must be non-empty strings")
    errors.extend(validate_tool_dependency_contract(payload.get("tool_calls") or []))
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


def validate_tool_dependency_contract(raw_calls: Any) -> list[str]:
    """Validate the optional per-action tool dependency graph.

    Legacy actions without graph metadata remain valid and execute
    sequentially. Once any call declares dependencies, every call must have a
    unique call_id. Missing dependency arrays are normalized to [] before this
    validation because an omitted edge means the call declared no prerequisite.
    """

    if not isinstance(raw_calls, list) or not raw_calls:
        return []
    calls = [item for item in raw_calls if isinstance(item, dict)]
    # A model may attach a diagnostic call_id to a single legacy call while
    # omitting an empty dependency list. That does not authorize parallelism
    # and is safe to execute sequentially. Graph mode starts only when the
    # model explicitly declares at least one depends_on field.
    graph_mode = any("depends_on" in item for item in calls)
    if not graph_mode:
        return []

    errors: list[str] = []
    call_ids: list[str] = []
    for index, call in enumerate(calls, start=1):
        call_id = call.get("call_id")
        depends_on = call.get("depends_on")
        if not isinstance(call_id, str) or not call_id.strip():
            errors.append(f"tool_calls[{index}].call_id is required in dependency graph mode")
        else:
            call_ids.append(call_id)
        if not isinstance(depends_on, list):
            errors.append(f"tool_calls[{index}].depends_on is required in dependency graph mode")

    duplicate_ids = sorted({item for item in call_ids if call_ids.count(item) > 1})
    if duplicate_ids:
        errors.append("tool call_id values must be unique: " + ",".join(duplicate_ids))
    if errors:
        return errors
    known_ids = set(call_ids)
    dependencies_by_id: dict[str, set[str]] = {}
    for call in calls:
        call_id = call.get("call_id")
        dependencies = call.get("depends_on")
        if not isinstance(call_id, str) or not call_id.strip() or not isinstance(dependencies, list):
            continue
        dependency_ids = {item for item in dependencies if isinstance(item, str) and item.strip()}
        unknown = sorted(dependency_ids - known_ids)
        if unknown:
            errors.append(f"tool call {call_id} depends on unknown call_id values: {','.join(unknown)}")
        if call_id in dependency_ids:
            errors.append(f"tool call {call_id} must not depend on itself")
        # Unknown identifiers have already produced a precise error above. Do
        # not leave them in the cycle detector, otherwise one malformed edge
        # would also be reported as an unrelated graph cycle.
        dependencies_by_id[call_id] = dependency_ids & known_ids

    remaining = {key: set(value) for key, value in dependencies_by_id.items()}
    resolved: set[str] = set()
    while remaining:
        ready = [key for key, dependencies in remaining.items() if dependencies <= resolved]
        if not ready:
            errors.append("tool dependency graph contains a cycle")
            break
        for key in ready:
            resolved.add(key)
            remaining.pop(key, None)
    return errors


def validate_objective_plan_contract(raw_plan: Any) -> list[str]:
    """校验模型输出的目标计划列表。

    计划用于可观测和多步任务推进，不直接驱动数据库写入；
    因此这里只做结构约束，具体是否执行仍由 tool_calls 和 ToolGateway 决定。
    """

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
    """校验模型声明的停止原因。

    这个字段要求模型显式说明“为什么能停”或“为什么还要继续调用工具”，
    方便排查模型过早停止、漏调用工具或无意义循环。
    """

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
    """生成合同错误时的安全兜底 AgentAction。

    该 action 不会触发工具执行；它只是给主 loop 一个稳定对象，
    实际上主 loop 会优先把错误回喂给模型修正。
    """

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
