from __future__ import annotations

import pytest

from mahjong_agent.trial_state_policy import (
    STATE_MACHINE_VERSION,
    approval_status_label,
    require_state_transition,
    state_transition_verdict,
)


def test_trial_state_policy_allows_expected_game_transition() -> None:
    verdict = state_transition_verdict(
        entity_type="game",
        current_status="待组局",
        next_status="邀约中",
        event="candidate_outbox_created",
    )

    assert verdict["allowed"] is True
    assert verdict["code"] == "state_transition_allowed"
    assert verdict["from_status"] == "待组局"
    assert verdict["to_status"] == "邀约中"
    assert verdict["state_machine_version"] == STATE_MACHINE_VERSION


def test_trial_state_policy_rejects_final_game_reopen() -> None:
    verdict = state_transition_verdict(
        entity_type="game",
        current_status="已成局",
        next_status="邀约中",
        event="late_candidate_reply",
    )

    assert verdict["allowed"] is False
    assert verdict["code"] == "state_transition_rejected"
    assert "不允许" in verdict["reason"]


def test_trial_state_policy_validates_outbox_and_followup_transitions() -> None:
    outbox = require_state_transition(
        entity_type="outbox",
        current_status="待确认",
        next_status="已确认",
        event="candidate_accepts",
    )
    followup = require_state_transition(
        entity_type="followup",
        current_status="待审批",
        next_status="已审批",
        event="boss_approves_followup",
    )

    assert outbox["allowed"] is True
    assert followup["allowed"] is True


def test_trial_state_policy_rejects_unknown_entity_and_missing_target() -> None:
    unknown = state_transition_verdict(
        entity_type="room",
        current_status=None,
        next_status="open",
        event="test",
    )
    missing = state_transition_verdict(
        entity_type="game",
        current_status="待组局",
        next_status="",
        event="test",
    )

    assert unknown["code"] == "unknown_entity_type"
    assert missing["code"] == "missing_next_status"


def test_require_state_transition_raises_when_rejected() -> None:
    with pytest.raises(ValueError, match="不允许"):
        require_state_transition(
            entity_type="outbox",
            current_status="拒绝",
            next_status="已确认",
            event="invalid_retry",
        )


def test_approval_status_label_normalizes_known_statuses() -> None:
    assert approval_status_label("pending") == "待审批"
    assert approval_status_label("approved") == "已审批"
    assert approval_status_label("rejected") == "审批拒绝"
    assert approval_status_label("自定义") == "自定义"
    assert approval_status_label(None) == "待审批"
