from __future__ import annotations

from mahjong_agent.candidate_validation import CandidateActionProposalValidator


def build_validator(
    *,
    negotiation=None,
    full: bool = False,
    final_statuses: set[str] | None = None,
    facts: list[dict] | None = None,
) -> CandidateActionProposalValidator:
    facts = facts if facts is not None else []

    def fallback_classifier(text, game):
        return {"intent": "candidate_question", "feedback_type": "candidate_question", "status": "待确认"}

    def negotiation_classifier(text, game):
        return negotiation

    def extracted_fact_applier(classification, proposal, game):
        facts.append({"classification": classification, "proposal": proposal, "game": game})
        if proposal.get("extracted_facts"):
            classification["applied_facts"] = proposal["extracted_facts"]

    return CandidateActionProposalValidator(
        fallback_classifier=fallback_classifier,
        negotiation_classifier=negotiation_classifier,
        game_full_checker=lambda game: full,
        extracted_fact_applier=extracted_fact_applier,
        final_game_statuses=final_statuses or set(),
    )


def llm_proposal(action: str, *, confidence: float = 0.9, **extra) -> dict:
    return {
        "source": "llm",
        "semantic_type": "accepted",
        "proposed_action": action,
        "confidence": confidence,
        "reply_text": "好的，加你272了。",
        **extra,
    }


def test_candidate_validator_accepts_valid_llm_state_change() -> None:
    validator = build_validator()

    result = validator.validate(
        llm_proposal("mark_candidate_confirmed"),
        candidate_text="可以",
        outbox_item={"status": "已发送"},
        game={"status": "邀约中"},
        fallback={},
    )

    assert result["classification"]["feedback_type"] == "accepted"
    assert result["validated_action"] == "mark_candidate_confirmed"
    assert result["validation"]["accepted"] is True
    assert result["validation"]["mode"] == "llm_proposal_backend_validated"


def test_candidate_validator_uses_fallback_contract_when_llm_unavailable() -> None:
    validator = build_validator()

    result = validator.validate(
        {"source": "rules"},
        candidate_text="可以",
        outbox_item={"status": "已发送"},
        game={"status": "邀约中"},
        fallback={
            "backend_fallback_classification": {
                "intent": "accepted",
                "feedback_type": "accepted",
                "status": "已确认",
            }
        },
    )

    assert result["classification"]["feedback_type"] == "accepted"
    assert result["validated_action"] == "mark_candidate_confirmed"
    assert result["validation"]["mode"] == "fallback_rules"


def test_candidate_validator_downgrades_low_confidence_state_change() -> None:
    validator = build_validator()

    result = validator.validate(
        llm_proposal("mark_candidate_confirmed", confidence=0.4),
        candidate_text="可以吧",
        outbox_item={"status": "已发送"},
        game={"status": "邀约中"},
        fallback={},
    )

    assert result["classification"]["feedback_type"] == "candidate_question"
    assert result["validated_action"] == "answer_candidate_question"
    assert result["validation"]["accepted"] is False
    assert "低于状态提交阈值" in result["validation"]["notes"][0]


def test_candidate_validator_turns_acceptance_into_negotiation_when_conditions_changed() -> None:
    facts: list[dict] = []
    validator = build_validator(
        negotiation={
            "intent": "candidate_negotiation",
            "feedback_type": "candidate_negotiation",
            "status": "待协商",
        },
        facts=facts,
    )

    result = validator.validate(
        llm_proposal(
            "mark_candidate_confirmed",
            extracted_facts={"requested_duration_hours": 6},
        ),
        candidate_text="可以，不过我想打六个小时",
        outbox_item={"status": "已发送"},
        game={"status": "邀约中"},
        fallback={},
    )

    assert result["classification"]["feedback_type"] == "candidate_negotiation"
    assert result["classification"]["applied_facts"] == {"requested_duration_hours": 6}
    assert result["validated_action"] == "start_negotiation"
    assert result["validation"]["accepted"] is False
    assert facts


def test_candidate_validator_blocks_terminal_or_full_game_confirmation() -> None:
    archived_validator = build_validator(final_statuses={"已取消"})
    archived = archived_validator.validate(
        llm_proposal("mark_candidate_confirmed"),
        candidate_text="可以",
        outbox_item={"status": "已发送"},
        game={"status": "已取消"},
        fallback={},
    )

    assert archived["classification"]["feedback_type"] == "candidate_question"
    assert archived["validation"]["accepted"] is False
    assert "已归档" in archived["validation"]["notes"][0]

    full_validator = build_validator(full=True)
    full = full_validator.validate(
        llm_proposal("mark_candidate_confirmed"),
        candidate_text="可以",
        outbox_item={"status": "已发送"},
        game={"status": "邀约中"},
        fallback={},
    )

    assert full["classification"]["feedback_type"] == "candidate_question"
    assert full["validation"]["accepted"] is False
    assert "缺口已满" in full["validation"]["notes"][0]
