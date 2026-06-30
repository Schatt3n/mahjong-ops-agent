from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from mahjong_agent.candidate_reply_facts import CandidateReplyFactService


TZ = ZoneInfo("Asia/Shanghai")


def parse_dt(value):
    if isinstance(value, datetime):
        return value
    if not value:
        return None
    return datetime.fromisoformat(str(value))


def service() -> CandidateReplyFactService:
    return CandidateReplyFactService(parse_datetime=parse_dt)


def game_state(*, start_at: str = "2026-07-01T14:00:00+08:00", duration_hours: float = 4) -> dict:
    return {
        "id": "game_001",
        "parsed": {
            "start_at": start_at,
            "duration_hours": duration_hours,
        },
    }


def test_candidate_reply_fact_service_classifies_simple_acceptance() -> None:
    result = service().classify_reply("可以", game_state())

    assert result == {"intent": "accepted", "feedback_type": "accepted", "status": "已确认"}


def test_candidate_reply_fact_service_detects_changed_start_time() -> None:
    result = service().classify_reply("可以，但是我最快四点半", game_state())

    assert result["feedback_type"] == "candidate_negotiation"
    assert result["requested_start_time"] == "16:30"
    assert result["requested_start_time_label"] == "四点半"
    assert result["current_start_time"] == "14:00"


def test_candidate_reply_fact_service_resolves_ambiguous_hour_near_game_start() -> None:
    result = service().classify_negotiation("我4点可以到", game_state())

    assert result is not None
    assert result["requested_start_time"] == "16:00"
    assert result["requested_start_time_label"] == "四点"


def test_candidate_reply_fact_service_detects_changed_duration() -> None:
    result = service().classify_reply("可以，但想打六小时", game_state(duration_hours=4))

    assert result["feedback_type"] == "candidate_negotiation"
    assert result["requested_duration_hours"] == 6.0
    assert result["current_duration_hours"] == 4.0


def test_candidate_reply_fact_service_applies_llm_extracted_duration() -> None:
    classification = {"intent": "candidate_negotiation", "feedback_type": "candidate_negotiation", "status": "待协商"}

    service().apply_extracted_negotiation_facts(
        classification,
        {"extracted_facts": {"requested_duration_hours": 5}},
        game_state(duration_hours=4),
    )

    assert classification["requested_duration_hours"] == 5.0
    assert classification["current_duration_hours"] == 4.0


def test_candidate_reply_fact_service_applies_llm_extracted_start_time() -> None:
    classification = {"intent": "candidate_negotiation", "feedback_type": "candidate_negotiation", "status": "待协商"}

    service().apply_extracted_negotiation_facts(
        classification,
        {"extracted_facts": {"requested_start_time": "16:30"}},
        game_state(),
    )

    assert classification["requested_start_time"] == "16:30"
    assert classification["requested_start_time_label"] == "四点半"
    assert classification["current_start_time"] == "14:00"
