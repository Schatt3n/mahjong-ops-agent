"""主模型 AgentAction 输出合同校验。

设计理念：
- LLM 可以决定下一步做什么，但必须用后端规定的结构化合同表达。
- 合同校验只检查结构、状态约束和安全不变量，不写麻将业务 if-else。
- 合同不合法时，后端不执行任何工具，而是把错误作为工具结果回喂模型修正。
"""

from __future__ import annotations

import json
from typing import Any

from .customer_visible_contract import CUSTOMER_EXECUTION_FAILURE_REPLY

from .models import AgentAction, ToolResult


def parse_action(raw_response: str) -> tuple[AgentAction, list[str]]:
    """把模型原始文本解析为 AgentAction。

    返回值始终包含一个 AgentAction；如果解析失败，返回 contract_error_action 和错误列表，
    这样主 loop 可以统一处理，不需要在每个调用点写异常分支。
    """

    action, errors, _ = parse_action_with_repairs(raw_response)
    return action, errors


def _unique_action_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """拒绝任意层级的重复 JSON 字段，避免后值静默覆盖工具或状态参数。"""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _repair_premature_action_close(raw_response: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """仅修复误放状态元字段后提前闭合根对象、其后继续同一动作尾字段的结构。

    位置由 JSONDecoder 给出，不能通过括号扫描猜测，因此引号内括号不受影响。
    只删除这一枚根闭括号，再完整解析余下全部文本；不得截取首对象或合并两份动作。
    """
    decoder = json.JSONDecoder(object_pairs_hook=_unique_action_object)
    start = len(raw_response) - len(raw_response.lstrip(" \t\r\n"))
    prefix, end = decoder.raw_decode(raw_response, idx=start)
    header_fields = {"goal", "objective_status", "reasoning_summary", "objective_state", "objective_plan"}
    if not isinstance(prefix, dict) or not header_fields <= prefix.keys():
        raise ValueError("first segment is not an action header with its state and plan")
    if not isinstance(prefix["objective_state"], dict):
        raise ValueError("objective_state must be an object before structural repair")
    if not isinstance(prefix["objective_plan"], list):
        raise ValueError("objective_plan must be an array before structural repair")
    misplaced = [field for field in ("reply_scope", "active_game_id") if field in prefix]
    if not misplaced:
        raise ValueError("first segment contains no declared misplaced state metadata")
    if "reply_scope" in prefix and not isinstance(prefix["reply_scope"], dict):
        raise ValueError("misplaced reply_scope must be an object")
    if "active_game_id" in prefix and prefix["active_game_id"] is not None and not isinstance(prefix["active_game_id"], str):
        raise ValueError("misplaced active_game_id must be a string or null")
    action_fields = {"goal", "objective_status", "reasoning_summary", "reply_to_user", "tool_calls", "needs_human", "stop_reason"}
    if action_fields <= prefix.keys():
        raise ValueError("a complete first action cannot absorb trailing content")

    suffix_start = end
    while suffix_start < len(raw_response) and raw_response[suffix_start] in " \t\r\n":
        suffix_start += 1
    if suffix_start >= len(raw_response) or raw_response[suffix_start] != ",":
        raise ValueError("suffix must continue the same object with a comma and field name")
    key_start = suffix_start + 1
    while key_start < len(raw_response) and raw_response[key_start] in " \t\r\n":
        key_start += 1
    suffix_key, key_end = decoder.raw_decode(raw_response, idx=key_start)
    tail_fields = {"plan_revision_reason", "reply_to_user", "tool_calls", "needs_human", "stop_reason",
                   "self_assessment", "response_mode", "silence_reason", "badcase"}
    if not isinstance(suffix_key, str) or suffix_key not in tail_fields:
        raise ValueError("suffix must resume a declared action tail field")
    while key_end < len(raw_response) and raw_response[key_end] in " \t\r\n":
        key_end += 1
    if key_end >= len(raw_response) or raw_response[key_end] != ":" or raw_response[end - 1] != "}":
        raise ValueError("suffix is not an object field continuation")

    candidate_text = raw_response[:end - 1] + raw_response[end:]
    candidate = json.loads(candidate_text, object_pairs_hook=_unique_action_object)
    if not isinstance(candidate, dict) or not action_fields <= candidate.keys():
        raise ValueError("removing one closing brace did not produce a complete action")
    if not isinstance(candidate["tool_calls"], list):
        raise ValueError("candidate tool_calls must be an array before structural normalization")
    state = dict(candidate["objective_state"])
    for field in misplaced:
        value = candidate[field]
        # JSON 类型也属于值的一部分，不能把 True 与 1 等 Python 等值结果当成无冲突。
        if field in state and json.dumps(state[field], sort_keys=True) != json.dumps(value, sort_keys=True):
            raise ValueError(f"misplaced state metadata conflicts with objective_state.{field}")
        state[field] = candidate.pop(field)
    candidate["objective_state"] = state
    repairs = [{
        "field": "response_json", "removed_character_offset": end - 1, "from": "}", "to": "",
        "reason": "removed exactly the premature root closing brace before the remaining fields of the same action",
    }]
    repairs.extend({
        "field": f"objective_state.{field}", "from": f"root.{field}", "to": f"objective_state.{field}",
        "reason": "relocated declared state metadata without changing its value",
    } for field in misplaced)
    return candidate, repairs


def parse_action_with_repairs(
    raw_response: str,
    *,
    turn_tool_evidence: list[ToolResult] | None = None,
    previous_step_tool_results: list[ToolResult] | None = None,
) -> tuple[AgentAction, list[str], list[dict[str, Any]]]:
    """解析动作并规范化结构；历史依赖只信服务端本回合的真实执行证据。"""

    structural_repairs: list[dict[str, Any]] = []
    try:
        payload = json.loads(raw_response, object_pairs_hook=_unique_action_object)
    except json.JSONDecodeError as exc:
        if exc.msg != "Extra data":
            return contract_error_action(), [f"response is not valid JSON: {exc.msg}"], []
        try:
            payload, structural_repairs = _repair_premature_action_close(raw_response)
        except ValueError as repair_error:
            return contract_error_action(), [f"response is not valid JSON: {exc.msg}; structural repair rejected: {repair_error}"], []
    except ValueError as exc:
        return contract_error_action(), [f"response is not valid JSON: {exc}"], []
    if not isinstance(payload, dict):
        return contract_error_action(), ["response JSON root must be object"], []
    payload, repairs = normalize_action_contract(
        payload, turn_tool_evidence=turn_tool_evidence,
        previous_step_tool_results=previous_step_tool_results,
    )
    repairs = structural_repairs + repairs
    errors = validate_action_contract(payload)
    if errors:
        return contract_error_action(), errors, repairs
    return AgentAction.from_payload(payload), [], repairs


def _satisfied_historical_call_ids(
    turn_tool_evidence: list[ToolResult], previous_step_tool_results: list[ToolResult],
) -> set[str]:
    """只认可本回合唯一且在最新批成功的调用，不猜历史重名或失败依赖的指代。

    call_id 只在单批内唯一，不能从旧结果中取首个或最后一个同名成功来凑依据。
    最新批还须与完整证据中的唯一结果一致，避免把更老的成功跨过失败批次重用。
    """
    history: dict[str, list[ToolResult]] = {}
    latest: dict[str, list[ToolResult]] = {}
    for evidence, grouped in ((turn_tool_evidence, history), (previous_step_tool_results, latest)):
        for item in evidence:
            if isinstance(item.call_id, str) and item.call_id:
                grouped.setdefault(item.call_id, []).append(item)
    satisfied: set[str] = set()
    for call_id, matches in history.items():
        current_matches = latest.get(call_id, [])
        if len(matches) != 1 or len(current_matches) != 1:
            continue
        result = matches[0]
        if result.called and result.allowed and not result.error and current_matches[0] == result:
            satisfied.add(call_id)
    return satisfied


def normalize_action_contract(
    payload: dict[str, Any],
    *,
    turn_tool_evidence: list[ToolResult] | None = None,
    previous_step_tool_results: list[ToolResult] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """规范化结构元数据，不改工具选择、实参或任何业务决策。

    独立的计划步骤命名空间误用沿用既有修复；一旦标识出现在本回合执行历史，
    就必须按真实结果证明已满足，不能靠计划中的 done 等模型自述兜底。
    当前批标识始终优先，历史成功也不能抹掉真正的同批依赖或自引用。
    """

    normalized = dict(payload)
    repairs: list[dict[str, Any]] = []
    raw_plan = normalized.get("objective_plan")
    plan_step_ids = {
        str(step.get("step_id")).strip()
        for step in raw_plan or []
        if isinstance(step, dict) and step.get("step_id") is not None and str(step.get("step_id")).strip()
    }
    raw_tool_calls = normalized.get("tool_calls")
    historical_results = turn_tool_evidence or []
    observed_historical_ids = {item.call_id for item in historical_results if isinstance(item.call_id, str)}
    satisfied_historical_ids = _satisfied_historical_call_ids(historical_results, previous_step_tool_results or [])
    if isinstance(raw_tool_calls, list):
        current_call_ids = {call.get("call_id") for call in raw_tool_calls if isinstance(call, dict) and isinstance(call.get("call_id"), str)}
        # 计划编号匹配会去空白并转字符串，保护集合也须同样规范化，避免类型错用绕过真实调用检查。
        protected_plan_dependency_ids = {call_id.strip() for call_id in current_call_ids | observed_historical_ids}
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
                elif (
                    len(raw_tool_calls) == 1
                    and isinstance(call.get("depends_on"), list)
                    and call["depends_on"]
                    and all(str(item).strip() in plan_step_ids for item in call["depends_on"])
                    and not any(str(item).strip() in protected_plan_dependency_ids for item in call["depends_on"])
                ):
                    original_dependencies = list(call["depends_on"])
                    call["depends_on"] = []
                    repairs.append(
                        {
                            "field": f"tool_calls[{index}].depends_on",
                            "from": original_dependencies,
                            "to": [],
                            "reason": (
                                "a single tool call has no same-action prerequisite, so external plan-step "
                                "references are redundant"
                            ),
                        }
                    )
                elif len(raw_tool_calls) == 1 and isinstance(call.get("depends_on"), list):
                    original_dependencies = list(call["depends_on"])
                    # 实参、工具选择和同批依赖均不变；仅消去已由后端证实满足的旧边。
                    removed = [item for item in original_dependencies if isinstance(item, str)
                               and item not in current_call_ids and item in satisfied_historical_ids]
                    if removed:
                        call["depends_on"] = [item for item in original_dependencies if item not in removed]
                        repairs.append({
                            "field": f"tool_calls[{index}].depends_on", "from": original_dependencies,
                            "to": list(call["depends_on"]), "satisfied_historical_call_ids": removed,
                            "reason": "unique historical dependencies succeeded in the latest executed batch and are already satisfied",
                        })
                normalized_calls.append(call)
            normalized["tool_calls"] = normalized_calls
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
    response_mode = payload.get("response_mode", "reply")
    if not isinstance(response_mode, str) or response_mode not in {"reply", "silent"}:
        errors.append("response_mode must be reply or silent")
    if response_mode == "silent":
        if status not in {"completed", "waiting_user"}:
            errors.append("silent requires completed or waiting_user")
        if reply:
            errors.append("silent requires empty reply_to_user")
        if not isinstance(payload.get("silence_reason"), str) or not payload["silence_reason"].strip():
            errors.append("silent requires non-empty silence_reason")
    if response_mode != "silent" and status in terminal_statuses and isinstance(reply, str) and not reply.strip():
        errors.append(f"{status} requires non-empty reply_to_user")
    if status == "needs_human" and payload.get("needs_human") is not True:
        errors.append("needs_human objective_status requires needs_human=true")
    if payload.get("needs_human") is True and status != "needs_human":
        errors.append("needs_human=true requires objective_status=needs_human")
    return errors


def validate_tool_dependency_contract(raw_calls: Any) -> list[str]:
    """校验工具dependency结构化合同；失败时返回或抛出可审计的原因。

    """

    if not isinstance(raw_calls, list) or not raw_calls:
        return []
    calls = [item for item in raw_calls if isinstance(item, dict)]
    # 模型可能给单个旧式调用附带诊断用的 call_id，却省略空依赖列表。
    # 这不代表允许并行，仍应按顺序执行；只有模型显式声明至少一个
    # depends_on 字段时，才启用依赖图模式。
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
        # 未知标识符已在上方产生精确错误，不再送入环检测器；
        # 否则一条格式错误的边还会被重复报告成无关的依赖环。
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
        reply_to_user=CUSTOMER_EXECUTION_FAILURE_REPLY,
        needs_human=True,
        stop_reason={
            "can_stop": True,
            "why": "模型输出合同错误，后端不能安全继续执行。",
            "pending_work": [],
            "depends_on_tool_results": False,
        },
    )
