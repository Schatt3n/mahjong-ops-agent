from __future__ import annotations

from typing import Any

from mahjong_agent.trial_tool_gateway import TrialToolGateway


def test_trial_tool_gateway_rejects_unvalidated_tool_without_running_operation() -> None:
    calls: list[str] = []

    gateway = TrialToolGateway(
        validated_action_lookup=lambda tool_plan, tool_name: None,
        action_executor=lambda action, operation: operation(),
    )

    result, action = gateway.execute(
        tool_name="send_message",
        tool_plan={"validated_actions": []},
        request={"tool_name": "send_message", "called": True},
        rejected_result={"result_count": 0, "outbox": []},
        operation=lambda: calls.append("ran") or {"ok": True},
    )

    assert calls == []
    assert action is None
    assert result == {
        "tool_name": "send_message",
        "called": False,
        "result_count": 0,
        "outbox": [],
        "rejected": True,
        "validation_error": "send_message 未通过后端动作校验，拒绝执行。",
    }


def test_trial_tool_gateway_executes_validated_action_and_preserves_control_metadata() -> None:
    action_record = {
        "action_id": "act_123",
        "idempotency_key": "trace:stage:send_message:123",
        "tool_name": "send_message",
    }
    executed: list[dict[str, Any]] = []

    def execute(action: dict[str, Any], operation) -> dict[str, Any]:
        executed.append(action)
        return operation()

    gateway = TrialToolGateway(
        validated_action_lookup=lambda tool_plan, tool_name: action_record,
        action_executor=execute,
    )

    result, action = gateway.execute(
        tool_name="send_message",
        tool_plan={"validated_actions": [action_record]},
        request={"tool_name": "send_message", "called": True},
        operation=lambda: {"ok": True, "result_count": 2, "outbox": [{"id": "out_1"}]},
    )

    assert executed == [action_record]
    assert action is action_record
    assert result == {
        "ok": True,
        "result_count": 2,
        "outbox": [{"id": "out_1"}],
        "action_id": "act_123",
        "idempotency_key": "trace:stage:send_message:123",
    }
