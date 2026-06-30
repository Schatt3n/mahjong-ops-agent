from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


ValidatedActionLookup = Callable[[dict[str, Any] | None, str], dict[str, Any] | None]
ControlledActionExecutor = Callable[[dict[str, Any], Callable[[], dict[str, Any]]], dict[str, Any]]


@dataclass(slots=True)
class TrialToolGateway:
    """Executes trial-page tools through backend-validated controlled actions.

    This adapter does not decide whether a tool should be used. That decision
    must already exist in the validated action plan. Its only job is to prevent
    unvalidated tool execution and to preserve action/idempotency metadata in
    the tool result.
    """

    validated_action_lookup: ValidatedActionLookup
    action_executor: ControlledActionExecutor

    def execute(
        self,
        *,
        tool_name: str,
        tool_plan: dict[str, Any] | None,
        request: dict[str, Any],
        operation: Callable[[], dict[str, Any]],
        rejected_result: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        gateway_action = self.validated_action_lookup(tool_plan, tool_name)
        if gateway_action is None:
            result = {
                **request,
                **(rejected_result or {}),
                "called": False,
                "rejected": True,
                "validation_error": f"{tool_name} 未通过后端动作校验，拒绝执行。",
            }
            return result, None

        result = self.action_executor(gateway_action, operation)
        return {
            **result,
            "action_id": gateway_action.get("action_id"),
            "idempotency_key": gateway_action.get("idempotency_key"),
        }, gateway_action
