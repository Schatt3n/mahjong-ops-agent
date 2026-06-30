from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from mahjong_agent.trial_followup import TrialOrganizerFollowupAdapter


TZ = ZoneInfo("Asia/Shanghai")


def classification() -> dict:
    return {
        "feedback_type": "candidate_negotiation",
        "requested_start_time_label": "四点半",
    }


def outbox_item() -> dict:
    return {
        "id": "outbox_001",
        "game_id": "game_001",
        "customer_id": "amy",
        "customer_name": "Amy",
    }


def game() -> dict:
    return {
        "id": "game_001",
        "organizer_id": "zhang",
        "organizer_name": "张哥",
        "status": "邀约中",
    }


def build_adapter(*, allow_tool: bool = True) -> tuple[TrialOrganizerFollowupAdapter, dict]:
    calls: dict[str, object] = {"audits": []}

    def tool_plan_validator(**kwargs):
        calls["tool_plan_input"] = kwargs
        plan = dict(kwargs["plan"])
        tool_call = dict(plan["tool_calls"][0])
        tool_call["action_id"] = "tool_call_action"
        tool_call["idempotency_key"] = "tool_call_key"
        plan["tool_calls"] = [tool_call]
        action = {
            "action_id": "act_followup",
            "tool_name": "send_message",
            "idempotency_key": "idem_followup",
            "risk_level": "high",
            "approval_required": True,
            "validation": {"allowed": allow_tool, "code": "allowed" if allow_tool else "blocked"},
        }
        if allow_tool:
            plan["validated_actions"] = [action]
            plan["rejected_actions"] = []
        else:
            plan["validated_actions"] = []
            plan["rejected_actions"] = [action]
        return plan

    def validated_action_lookup(plan, tool_name):
        for item in plan.get("validated_actions") or []:
            if item.get("tool_name") == tool_name:
                return item
        return None

    def action_executor(action, fn):
        calls["executed_action"] = action
        return fn()

    def followup_state_writer(**kwargs):
        calls["state_write"] = kwargs
        return {
            "ok": True,
            "id": "followup_001",
            "recipient_id": kwargs["recipient_id"],
            "recipient_name": kwargs["recipient_name"],
            "message_text": kwargs["message_text"],
            "status": "待审批",
            "approval": {"status": "pending", "target_type": "followup"},
            "approval_status": "待审批",
        }

    adapter = TrialOrganizerFollowupAdapter(
        fallback_factory=lambda **kwargs: "张哥，Amy最快四点半到，你们四点半开可以吗？",
        draft_factory=lambda **kwargs: {
            "should_create_message": True,
            "text": "张哥，Amy最快四点半到，你们四点半开可以吗？",
            "source": "llm",
            "model": "test-model",
            "reasoning_summary": "候选人改时间，需要发起人确认。",
        },
        text_guard=lambda text, **kwargs: text,
        tool_plan_validator=tool_plan_validator,
        validated_action_lookup=validated_action_lookup,
        action_executor=action_executor,
        followup_state_writer=followup_state_writer,
        plan_projector=lambda plan: {
            "stage": plan["stage"],
            "validated_actions": plan.get("validated_actions") or [],
            "rejected_actions": plan.get("rejected_actions") or [],
        },
        tool_audit_logger=lambda trace_id, event, payload: calls["audits"].append((trace_id, event, payload)),
    )
    return adapter, calls


def test_trial_organizer_followup_adapter_creates_pending_followup() -> None:
    adapter, calls = build_adapter()

    result = adapter.create(
        trace_id="trace_followup",
        classification=classification(),
        candidate_text="可以倒是可以，但是我最快要四点半",
        suggested_candidate_reply="我先问下这桌其他人。",
        outbox_item=outbox_item(),
        game=game(),
        now=datetime(2026, 7, 1, 16, 0, tzinfo=TZ),
    )

    assert result["id"] == "followup_001"
    assert result["source"] == "llm"
    assert result["model"] == "test-model"
    assert result["recipient_name"] == "张哥"
    assert result["message_text"] == "张哥，Amy最快四点半到，你们四点半开可以吗？"
    assert result["direct_send_executed"] is False
    assert result["needs_approval"] is True
    assert calls["state_write"]["message_text"] == result["message_text"]
    assert calls["state_write"]["draft_source"] == "llm"
    assert [event for _, event, _ in calls["audits"]] == ["tool_request", "tool_response"]
    assert calls["audits"][0][2]["direct_send_allowed"] is False


def test_trial_organizer_followup_adapter_returns_rejection_when_tool_validation_blocks() -> None:
    adapter, calls = build_adapter(allow_tool=False)

    result = adapter.create(
        trace_id="trace_followup",
        classification=classification(),
        candidate_text="可以倒是可以，但是我最快要四点半",
        suggested_candidate_reply="我先问下这桌其他人。",
        outbox_item=outbox_item(),
        game=game(),
        now=datetime(2026, 7, 1, 16, 0, tzinfo=TZ),
    )

    assert result["skipped"] is True
    assert result["status"] == "已拦截"
    assert result["direct_send_executed"] is False
    assert "state_write" not in calls
    assert calls["audits"] == []
    assert result["agent_actions"][0]["rejected_actions"]


def test_trial_organizer_followup_adapter_ignores_non_negotiation_or_missing_game() -> None:
    adapter, _ = build_adapter()

    assert adapter.create(
        trace_id="trace_followup",
        classification={"feedback_type": "accepted"},
        candidate_text="可以",
        suggested_candidate_reply="好的。",
        outbox_item=outbox_item(),
        game=game(),
        now=datetime(2026, 7, 1, 16, 0, tzinfo=TZ),
    ) is None
    assert adapter.create(
        trace_id="trace_followup",
        classification=classification(),
        candidate_text="可以倒是可以，但是我最快要四点半",
        suggested_candidate_reply="我先问下这桌其他人。",
        outbox_item=outbox_item(),
        game=None,
        now=datetime(2026, 7, 1, 16, 0, tzinfo=TZ),
    ) is None
