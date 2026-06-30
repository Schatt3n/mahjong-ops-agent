from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from mahjong_agent.trial_candidate import TrialCandidateMessageAdapter


TZ = ZoneInfo("Asia/Shanghai")


def outbox_item() -> dict:
    return {
        "id": "outbox_001",
        "game_id": "game_001",
        "customer_id": "ran",
        "customer_name": "冉姐",
        "message_text": "冉姐，14:00，0.5无烟，打吗？",
        "status": "已发送",
    }


def build_adapter(*, outbox: dict | None = None, feedback_result: dict | None = None) -> TrialCandidateMessageAdapter:
    outbox = outbox if outbox is not None else outbox_item()
    feedback_result = feedback_result if feedback_result is not None else {"ok": True, "auto_success": False}
    calls: dict[str, object] = {}

    def fallback_proposal(text, outbox_item, game):
        calls["fallback"] = (text, outbox_item, game)
        return {
            "source": "rules",
            "semantic_type": "accepted",
            "proposed_action": "record_candidate_feedback",
            "confidence": 0.7,
            "reply_text": "",
            "reasoning_summary": "候选人明确接受。",
        }

    def llm_proposal(**kwargs):
        calls["llm"] = kwargs
        return {
            **kwargs["fallback"],
            "source": "llm",
            "model": "test-model",
            "reply_text": "好的，加你272了。",
            "confidence": 0.92,
        }

    def validate(proposal, **kwargs):
        calls["validation"] = kwargs
        return {
            "classification": {
                "feedback_type": "accepted",
                "status": "已确认",
            },
            "validated_action": "record_candidate_feedback",
            "validation": {"accepted": True, "notes": []},
        }

    def action_factory(**kwargs):
        calls["action"] = kwargs
        return {
            "tool_name": "record_candidate_feedback",
            "validation": {"allowed": True, "code": "allowed"},
            "idempotency_key": "candidate_feedback_once",
        }

    def action_executor(action, fn):
        calls["executed_action"] = action
        result = fn()
        return {**result, "deduplicated": False}

    def feedback_recorder(payload):
        calls["feedback_payload"] = payload
        return feedback_result

    adapter = TrialCandidateMessageAdapter(
        outbox_lookup=lambda outbox_id: outbox if outbox_id == "outbox_001" else None,
        game_lookup=lambda game_id: {"id": game_id, "status": "邀约中"},
        fallback_proposal_factory=fallback_proposal,
        llm_proposal_factory=llm_proposal,
        proposal_validator=validate,
        candidate_reply_factory=lambda classification, text, item, game: "好的，加你272了。",
        candidate_reply_guard=lambda text, **kwargs: text,
        candidate_action_factory=action_factory,
        organizer_followup_factory=lambda **kwargs: None,
        action_executor=action_executor,
        action_plan_projector=lambda **kwargs: {
            "stage": kwargs["stage"],
            "validated_actions": [{"tool_name": kwargs["action"]["tool_name"]}],
        },
        feedback_recorder=feedback_recorder,
        state_loader=lambda now: {"now": now.isoformat()},
        trace_id_factory=lambda: "trace_generated",
        now_factory=lambda: datetime(2026, 7, 1, 18, 0, tzinfo=TZ),
        parse_datetime=lambda value: None,
        customer_reloader=lambda: calls.setdefault("reloaded", True),
        game_cache_updater=lambda game_id: calls.setdefault("cached_game", game_id),
        json_dumper=lambda value: json.dumps(value, ensure_ascii=False, separators=(",", ":")),
    )
    adapter._test_calls = calls  # type: ignore[attr-defined]
    return adapter


def test_trial_candidate_message_adapter_handles_accepted_reply() -> None:
    adapter = build_adapter()

    result = adapter.handle({"outbox_id": "outbox_001", "text": "可以", "now": "raw_now"})
    calls = adapter._test_calls  # type: ignore[attr-defined]
    notes = json.loads(calls["feedback_payload"]["notes"])

    assert result["ok"] is True
    assert result["candidate_message"]["feedback_type"] == "accepted"
    assert result["candidate_message"]["suggested_boss_reply"] == "好的，加你272了。"
    assert result["candidate_message"]["reply_source"] == "llm"
    assert result["candidate_message"]["model"] == "test-model"
    assert result["agent_actions"][0]["stage"] == "candidate_feedback"
    assert result["outbox_item"]["id"] == "outbox_001"
    assert result["state"] == {"now": "2026-07-01T18:00:00+08:00"}
    assert calls["reloaded"] is True
    assert calls["cached_game"] == "game_001"
    assert calls["feedback_payload"]["feedback_type"] == "accepted"
    assert calls["feedback_payload"]["profile_note"] == "邀约回复：可以"
    assert calls["feedback_payload"]["now"] == "raw_now"
    assert notes["kind"] == "candidate_message"
    assert notes["boss_reply"] == "好的，加你272了。"


def test_trial_candidate_message_adapter_extends_organizer_followup_actions() -> None:
    adapter = build_adapter()
    adapter.organizer_followup_factory = lambda **kwargs: {
        "message_text": "张哥，冉姐想晚点到，可以吗？",
        "agent_actions": [{"stage": "organizer_followup_draft"}],
    }

    result = adapter.handle({"outbox_id": "outbox_001", "text": "晚点到可以吗"})

    assert result["organizer_followup"]["message_text"] == "张哥，冉姐想晚点到，可以吗？"
    assert [item["stage"] for item in result["agent_actions"]] == [
        "candidate_feedback",
        "organizer_followup_draft",
    ]


def test_trial_candidate_message_adapter_rejects_missing_input() -> None:
    adapter = build_adapter()

    with pytest.raises(ValueError, match="缺少 outbox_id"):
        adapter.handle({"text": "可以"})

    with pytest.raises(ValueError, match="候选人回复不能为空"):
        adapter.handle({"outbox_id": "outbox_001", "text": ""})

    with pytest.raises(ValueError, match="找不到这条候选人邀约"):
        adapter.handle({"outbox_id": "missing", "text": "可以"})
