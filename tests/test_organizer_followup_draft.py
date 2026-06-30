from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo

from mahjong_agent import LLMConfig, LLMBudgetManager
from mahjong_agent.organizer_followup_draft import OrganizerFollowupDraftService


TZ = ZoneInfo("Asia/Shanghai")


def classification() -> dict:
    return {
        "feedback_type": "candidate_negotiation",
        "requested_start_time_label": "四点半",
        "requested_start_time": "16:30",
    }


def outbox_item() -> dict:
    return {
        "id": "outbox_001",
        "game_id": "game_001",
        "customer_id": "amy",
        "customer_name": "Amy",
        "message_text": "Amy，16:00，0.5无烟，约5小时，打吗？",
    }


def game() -> dict:
    return {
        "id": "game_001",
        "organizer_id": "zhang",
        "organizer_name": "张哥",
        "live_summary": "杭麻 0.5档 16:00 缺1 无烟",
        "parsed": {"start_time": "16:00", "duration_hours": 5},
        "participants": [{"customer_name": "张哥"}],
    }


def test_organizer_followup_draft_service_uses_rules_without_llm() -> None:
    service = OrganizerFollowupDraftService()
    fallback = service.fallback_message(
        classification=classification(),
        candidate_name="Amy",
        organizer_name="张哥",
    )

    result = service.draft(
        trace_id="trace_followup",
        classification=classification(),
        candidate_text="可以倒是可以，但是我最快要四点半",
        suggested_candidate_reply="我先问下这桌其他人。",
        outbox_item=outbox_item(),
        game=game(),
        fallback=fallback,
        now=datetime(2026, 7, 1, 16, 0, tzinfo=TZ),
    )

    assert fallback == "张哥，Amy最快四点半到，你们四点半开可以吗？"
    assert result["source"] == "rules"
    assert result["text"] == fallback


def test_organizer_followup_guard_rejects_unsafe_or_incomplete_text() -> None:
    service = OrganizerFollowupDraftService()
    fallback = "张哥，Amy最快四点半到，你们四点半开可以吗？"

    assert service.guard_message(
        "已经安排 Amy 直接来了",
        fallback=fallback,
        classification=classification(),
        organizer_name="张哥",
    ) == fallback
    assert service.guard_message(
        "张哥，Amy晚点到可以吗？",
        fallback=fallback,
        classification=classification(),
        organizer_name="张哥",
    ) == fallback
    assert service.guard_message(
        "Amy最快四点半到，你们四点半开可以吗？",
        fallback=fallback,
        classification=classification(),
        organizer_name="张哥",
    ) == "张哥，Amy最快四点半到，你们四点半开可以吗？"


def test_organizer_followup_draft_service_calls_llm_and_parses_contract() -> None:
    captured: dict[str, object] = {}
    audits: list[tuple[str, str, dict]] = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            content = {
                "should_create_message": True,
                "message_text": "张哥，Amy最快四点半到，你们四点半开可以吗？",
                "risk_level": "low",
                "reasoning_summary": "候选人改时间，需要发起人确认。",
                "notes": [],
            }
            return json.dumps(
                {
                    "choices": [{"message": {"content": json.dumps(content, ensure_ascii=False)}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
                },
                ensure_ascii=False,
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse()

    service = OrganizerFollowupDraftService(
        llm_config=LLMConfig(
            api_key="test-key",
            model="test-model",
            base_url="https://example.invalid/v1",
            timeout_seconds=3,
            max_completion_tokens=200,
        ),
        budget_manager=LLMBudgetManager(),
        audit_logger=lambda trace_id, event, payload: audits.append((trace_id, event, payload)),
        urlopen=fake_urlopen,
    )

    result = service.draft(
        trace_id="trace_followup",
        classification=classification(),
        candidate_text="可以倒是可以，但是我最快要四点半",
        suggested_candidate_reply="我先问下这桌其他人。",
        outbox_item=outbox_item(),
        game=game(),
        fallback="张哥，Amy最快四点半到，你们四点半开可以吗？",
        now=datetime(2026, 7, 1, 16, 0, tzinfo=TZ),
    )

    payload = captured["payload"]
    prompt = json.loads(payload["messages"][1]["content"])
    assert "协商消息起草助手" in payload["messages"][0]["content"]
    assert prompt["organizer"]["customer_name"] == "张哥"
    assert prompt["candidate"]["customer_name"] == "Amy"
    assert prompt["backend_classification"]["requested_start_time"] == "16:30"
    assert captured["timeout"] == 3
    assert result["source"] == "llm"
    assert result["model"] == "test-model"
    assert result["text"] == "张哥，Amy最快四点半到，你们四点半开可以吗？"
    assert [event for _, event, _ in audits] == ["llm_request", "llm_response", "llm_parsed"]
