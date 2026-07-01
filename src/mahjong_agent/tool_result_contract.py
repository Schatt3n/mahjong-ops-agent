from __future__ import annotations

from typing import Any

from .workflow_models import ToolCallRequest, ToolExecutionMode, ToolName


TOOL_RESULT_CONTRACT_VERSION = "tool_result.v1"


def with_tool_result_contract(
    payload: dict[str, Any],
    *,
    request: ToolCallRequest,
    result_type: str,
    side_effect: str,
    audit_policy: str,
) -> dict[str, Any]:
    """Attach the stable audit contract used by traces, evals and projections."""

    wrapped = dict(payload)
    wrapped["contract"] = {
        "schema_version": TOOL_RESULT_CONTRACT_VERSION,
        "tool_name": request.tool_name.value,
        "result_type": str(result_type),
        "execution_mode": request.execution_mode.value,
        "risk_level": request.risk_level.value,
        "side_effect": str(side_effect),
        "audit_policy": str(audit_policy),
        "idempotency_key": request.idempotency_key,
    }
    return wrapped


def tool_result_contract(payload: dict[str, Any]) -> dict[str, Any]:
    contract = payload.get("contract") if isinstance(payload, dict) else None
    return dict(contract) if isinstance(contract, dict) else {}


def expected_side_effect_for_mode(mode: ToolExecutionMode | str) -> str:
    execution_mode = mode if isinstance(mode, ToolExecutionMode) else ToolExecutionMode(str(mode))
    if execution_mode == ToolExecutionMode.READ_ONLY:
        return "none"
    if execution_mode == ToolExecutionMode.CREATE_PENDING:
        return "pending_approval_record"
    if execution_mode == ToolExecutionMode.STATE_WRITE:
        return "state_write_intent_or_low_risk_profile_observation"
    if execution_mode == ToolExecutionMode.DIRECT_SEND:
        return "external_message_send"
    return "not_called"


def tool_result_audit_policy(tool_name: ToolName | str) -> str:
    name = tool_name if isinstance(tool_name, ToolName) else ToolName(str(tool_name))
    if name in {ToolName.SEARCH_CURRENT_OPEN_GAMES, ToolName.SEARCH_CANDIDATE_CUSTOMERS}:
        return "只读查询，可自动执行，必须记录 query 和 result_count。"
    if name == ToolName.CREATE_PENDING_OUTBOX:
        return "只创建待老板审批草稿，不自动发送。"
    if name in {ToolName.CREATE_GAME, ToolName.CLOSE_GAME, ToolName.RECORD_SEAT_ACCEPTANCE}:
        return "只生成状态写入意图，由状态机校验后落库。"
    if name == ToolName.PROFILE_UPDATE:
        return "只写入低风险画像观察事实，要求字段白名单和证据。"
    if name == ToolName.SEND_MESSAGE:
        return "高风险真实发送，必须人工审批和发送网关幂等。"
    return "未知工具结果，不允许产生副作用。"
