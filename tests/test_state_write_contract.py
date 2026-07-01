from __future__ import annotations

from mahjong_agent.state_write_contract import parse_state_write_intent, validate_state_write_intent_contract
from mahjong_agent.workflow_models import GameWorkflowStatus


def valid_create_intent() -> dict:
    return {
        "kind": "create_game",
        "entity_type": "game",
        "entity_id": "game_001",
        "target_status": GameWorkflowStatus.NEGOTIATING.value,
        "enter_negotiating_if_outbox_created": True,
        "reason": "候选邀约已创建，进入邀约中。",
        "requirement": {"slots": {}},
    }


def test_parse_state_write_intent_accepts_valid_create_contract() -> None:
    parsed, errors = parse_state_write_intent(valid_create_intent())

    assert errors == []
    assert parsed is not None
    assert parsed.kind == "create_game"
    assert parsed.entity_id == "game_001"
    assert parsed.target_status == GameWorkflowStatus.NEGOTIATING.value
    assert parsed.metadata == {"enter_negotiating_if_outbox_created": True}


def test_state_write_intent_rejects_invalid_shape_and_status() -> None:
    errors = validate_state_write_intent_contract(
        {
            "kind": "create_game",
            "entity_type": "customer",
            "entity_id": "",
            "target_status": GameWorkflowStatus.CANCELLED.value,
            "reason": "",
            "requirement": [],
            "participant": {"customer_id": "ran"},
        }
    )

    assert "state_write_intent.entity_type invalid 'customer'" in errors
    assert "state_write_intent.entity_id must be a non-empty string" in errors
    assert "state_write_intent.target_status 'cancelled' is not allowed for create_game" in errors
    assert "state_write_intent.reason must be a non-empty string" in errors
    assert "state_write_intent.requirement must be an object" in errors
    assert "state_write_intent.participant is not allowed for create_game" in errors


def test_state_write_intent_requires_acceptance_participant_and_seat_delta() -> None:
    errors = validate_state_write_intent_contract(
        {
            "kind": "record_seat_acceptance",
            "entity_type": "game",
            "entity_id": "game_001",
            "target_status": GameWorkflowStatus.NEGOTIATING.value,
            "reason": "候选人确认加入。",
            "requirement": {"slots": {}},
        }
    )

    assert "state_write_intent.participant must be an object for record_seat_acceptance" in errors
    assert "state_write_intent.seat_delta must be an object for record_seat_acceptance" in errors
