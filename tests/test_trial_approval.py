from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from mahjong_agent.trial_approval import TrialApprovalDecisionAdapter


TZ = ZoneInfo("Asia/Shanghai")


def test_trial_approval_adapter_records_action_and_loads_state() -> None:
    calls: dict[str, object] = {}

    def action_record_factory(**kwargs):
        calls["action_kwargs"] = kwargs
        return {
            "action_id": "action_approval",
            "tool_name": kwargs["action_name"],
            "trace_id": kwargs["trace_id"],
            "idempotency_key": "approval_key",
            **kwargs,
        }

    def action_executor(action, fn):
        calls["executed_action"] = action
        result = fn()
        return {**result, "deduplicated": False}

    def action_plan_projector(**kwargs):
        return {"stage": kwargs["stage"], "validated_actions": [{"tool_name": kwargs["action"]["tool_name"]}]}

    def approval_executor(payload):
        calls["approval_payload"] = payload
        return {
            "ok": True,
            "approval": {
                "id": "approval_outbox_001",
                "status": "approved",
                "metadata": {"game_id": "game_001"},
            },
        }

    cached: list[str] = []
    state_times: list[datetime] = []
    adapter = TrialApprovalDecisionAdapter(
        approval_executor=approval_executor,
        action_record_factory=action_record_factory,
        action_executor=action_executor,
        action_plan_projector=action_plan_projector,
        state_loader=lambda now: state_times.append(now) or {"games": []},
        trace_id_factory=lambda: "trace_generated",
        now_factory=lambda: datetime(2026, 7, 1, 15, 0, tzinfo=TZ),
        parse_datetime=lambda value: None,
        game_cache_updater=cached.append,
    )

    result = adapter.decide(
        {
            "approval_id": "approval_outbox_001",
            "decision": "approved",
        }
    )

    assert result["ok"] is True
    assert result["agent_actions"][0]["stage"] == "approval_decision"
    assert result["agent_actions"][0]["validated_actions"][0]["tool_name"] == "record_approval_decision"
    assert result["state"] == {"games": []}
    assert cached == ["game_001"]
    assert calls["action_kwargs"]["trace_id"] == "trace_generated"
    assert calls["action_kwargs"]["approval_required"] is True
    assert calls["approval_payload"]["trace_id"] == "trace_generated"
    assert calls["approval_payload"]["now"] == "2026-07-01T15:00:00+08:00"
    assert state_times == [datetime(2026, 7, 1, 15, 0, tzinfo=TZ)]


def test_trial_approval_adapter_preserves_explicit_trace_and_now() -> None:
    parsed_now = datetime(2026, 7, 1, 16, 30, tzinfo=TZ)

    adapter = TrialApprovalDecisionAdapter(
        approval_executor=lambda payload: {"ok": True, "approval": {"status": "rejected", "metadata": {}}},
        action_record_factory=lambda **kwargs: {"tool_name": kwargs["action_name"], **kwargs},
        action_executor=lambda action, fn: fn(),
        action_plan_projector=lambda **kwargs: {"stage": kwargs["stage"], "action": kwargs["action"]["tool_name"]},
        state_loader=lambda now: {"now": now.isoformat()},
        trace_id_factory=lambda: "trace_generated",
        now_factory=lambda: datetime(2026, 7, 1, 15, 0, tzinfo=TZ),
        parse_datetime=lambda value: parsed_now if value == "custom_now" else None,
    )

    result = adapter.decide({"trace_id": "trace_given", "now": "custom_now", "decision": "rejected"})

    assert result["state"] == {"now": "2026-07-01T16:30:00+08:00"}
    assert result["agent_actions"][0]["action"] == "record_approval_decision"
