from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from mahjong_agent.candidate_feedback_action import CandidateFeedbackActionService


TZ = ZoneInfo("Asia/Shanghai")
NOW = datetime(2026, 7, 1, 14, 0, tzinfo=TZ)


def outbox_item() -> dict:
    return {
        "id": "outbox_001",
        "game_id": "game_001",
        "customer_id": "ran",
    }


def base_kwargs(**overrides) -> dict:
    kwargs = {
        "trace_id": "trace_001",
        "proposal": {
            "source": "llm",
            "proposed_action": "mark_candidate_confirmed",
            "reasoning_summary": "候选人确认来。",
        },
        "validation": {
            "validated_action": "mark_candidate_confirmed",
            "validation": {"accepted": True, "notes": []},
        },
        "classification": {"feedback_type": "accepted"},
        "outbox_item": outbox_item(),
        "game": {"id": "game_001", "status": "邀约中"},
        "now": NOW,
    }
    kwargs.update(overrides)
    return kwargs


def test_candidate_feedback_action_builds_allowed_controlled_record() -> None:
    audits: list[tuple[str, str, dict]] = []
    service = CandidateFeedbackActionService(
        protocol_version="controlled_agent.v1",
        action_compactor=lambda action: {"tool_name": action["tool_name"], "code": action["validation"]["code"]},
        tool_audit_logger=lambda trace_id, event, payload: audits.append((trace_id, event, payload)),
    )

    action = service.build(**base_kwargs())

    assert action["stage"] == "candidate_feedback"
    assert action["tool_name"] == "record_candidate_feedback"
    assert action["idempotency_key"].startswith("trace_001:candidate_feedback:record_candidate_feedback:")
    assert action["risk_level"] == "medium"
    assert action["validation"]["allowed"] is True
    assert action["validation"]["code"] == "allowed"
    assert audits[0][1] == "action_validation"
    assert audits[0][2]["allowed_count"] == 1
    assert audits[0][2]["validated_actions"] == [{"tool_name": "record_candidate_feedback", "code": "allowed"}]


def test_candidate_feedback_action_marks_model_downgrade_without_rejecting_safe_write() -> None:
    service = CandidateFeedbackActionService(protocol_version="controlled_agent.v1")

    action = service.build(
        **base_kwargs(
            proposal={"source": "llm", "proposed_action": "mark_candidate_confirmed"},
            validation={
                "validated_action": "answer_candidate_question",
                "validation": {"accepted": False, "notes": ["置信度不足"]},
            },
            classification={"feedback_type": "candidate_question"},
        )
    )

    assert action["validation"]["allowed"] is True
    assert action["validation"]["code"] == "downgraded_to_safe_feedback"
    assert action["validation"]["notes"] == ["置信度不足"]
    assert action["arguments"]["validated_action"] == "answer_candidate_question"


def test_candidate_feedback_action_rejects_final_game_confirmation() -> None:
    service = CandidateFeedbackActionService(
        protocol_version="controlled_agent.v1",
        final_game_statuses={"已成局", "已取消"},
    )

    action = service.build(**base_kwargs(game={"id": "game_001", "status": "已成局"}))

    assert action["validation"]["allowed"] is False
    assert action["validation"]["code"] == "final_game_reject"


def test_candidate_feedback_action_applies_runtime_policy_rejection() -> None:
    audits: list[dict] = []
    service = CandidateFeedbackActionService(
        protocol_version="controlled_agent.v1",
        runtime_policy_validator=lambda **kwargs: {
            "allowed": False,
            "code": "runtime_policy_read_only",
            "reason": "只读模式",
            "notes": ["test"],
        },
        action_compactor=lambda action: {"code": action["validation"]["code"]},
        tool_audit_logger=lambda trace_id, event, payload: audits.append(payload),
    )

    action = service.build(**base_kwargs())

    assert action["validation"]["allowed"] is False
    assert action["validation"]["code"] == "runtime_policy_read_only"
    assert action["validation"]["notes"] == ["test"]
    assert audits[0]["allowed_count"] == 0
    assert audits[0]["rejected_actions"] == [{"code": "runtime_policy_read_only"}]
