from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from mahjong_agent.candidate_semantics import (
    CandidateSemanticProposalAdapter,
    candidate_action_for_feedback_type,
    feedback_type_for_candidate_action,
    normalize_candidate_proposed_action,
    normalize_candidate_semantic_type,
)


TZ = ZoneInfo("Asia/Shanghai")


def fallback_contract(text: str, outbox_item: dict, game: dict | None) -> dict:
    return {
        "source": "rules",
        "semantic_type": "accepted",
        "proposed_action": "mark_candidate_confirmed",
        "confidence": 0.65,
        "reply_text": "",
        "reasoning_summary": "fallback",
        "notes": [],
        "backend_fallback_classification": {
            "intent": "accepted",
            "feedback_type": "accepted",
            "status": "已确认",
        },
        "outbox_id": outbox_item.get("id"),
    }


def test_candidate_semantic_adapter_returns_llm_contract_with_fallback() -> None:
    calls: dict[str, object] = {}

    def llm_contract(**kwargs):
        calls["kwargs"] = kwargs
        return {
            **kwargs["fallback"],
            "source": "llm",
            "model": "test-model",
            "confidence": 0.93,
            "reply_text": "好的，加你272了。",
        }

    adapter = CandidateSemanticProposalAdapter(
        fallback_proposal_factory=fallback_contract,
        llm_proposal_factory=llm_contract,
    )

    result = adapter.propose(
        trace_id="trace_1",
        candidate_text="可以",
        outbox_item={"id": "outbox_1"},
        game={"id": "game_1"},
        now=datetime(2026, 7, 1, 18, 0, tzinfo=TZ),
    )

    assert result.proposal["source"] == "llm"
    assert result.proposal["model"] == "test-model"
    assert result.fallback["source"] == "rules"
    assert calls["kwargs"]["fallback"] == result.fallback


def test_candidate_semantic_adapter_degrades_to_fallback_on_llm_error() -> None:
    def broken_llm(**kwargs):
        raise TimeoutError("slow")

    adapter = CandidateSemanticProposalAdapter(
        fallback_proposal_factory=fallback_contract,
        llm_proposal_factory=broken_llm,
    )

    result = adapter.propose(
        trace_id="trace_1",
        candidate_text="可以",
        outbox_item={"id": "outbox_1"},
        game={"id": "game_1"},
        now=datetime(2026, 7, 1, 18, 0, tzinfo=TZ),
    )

    assert result.proposal["source"] == "rules"
    assert result.proposal["semantic_type"] == "accepted"
    assert "TimeoutError" in result.proposal["reasoning_summary"]
    assert result.proposal["notes"]


def test_candidate_semantic_adapter_degrades_to_fallback_on_invalid_llm_contract() -> None:
    adapter = CandidateSemanticProposalAdapter(
        fallback_proposal_factory=fallback_contract,
        llm_proposal_factory=lambda **kwargs: "ok",  # type: ignore[return-value]
    )

    result = adapter.propose(
        trace_id="trace_1",
        candidate_text="可以",
        outbox_item={"id": "outbox_1"},
        game={"id": "game_1"},
        now=datetime(2026, 7, 1, 18, 0, tzinfo=TZ),
    )

    assert result.proposal["source"] == "rules"
    assert "expected dict" in result.proposal["reasoning_summary"]


def test_candidate_contract_normalizes_model_aliases() -> None:
    assert normalize_candidate_semantic_type("candidate-accept") == "accepted"
    assert normalize_candidate_semantic_type("do not disturb") == "do_not_disturb"
    assert normalize_candidate_semantic_type("unknown thing") == "uncertain"
    assert normalize_candidate_proposed_action(
        "confirm candidate",
        semantic_type="accepted",
    ) == "mark_candidate_confirmed"
    assert normalize_candidate_proposed_action(
        "",
        semantic_type="candidate_negotiation",
    ) == "start_negotiation"
    assert feedback_type_for_candidate_action("mark_candidate_confirmed") == "accepted"
    assert feedback_type_for_candidate_action("missing") == ""
    assert candidate_action_for_feedback_type("candidate_question") == "answer_candidate_question"
    assert candidate_action_for_feedback_type("missing") == "request_human_review"
