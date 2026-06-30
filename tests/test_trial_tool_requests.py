from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from mahjong_agent.trial_tool_requests import TrialToolRequestFactory


TZ = ZoneInfo("Asia/Shanghai")


def test_trial_tool_request_factory_builds_current_open_games_contract() -> None:
    factory = TrialToolRequestFactory()
    window = [
        datetime(2026, 7, 1, 17, 0, tzinfo=TZ),
        datetime(2026, 7, 1, 18, 30, tzinfo=TZ),
    ]

    request = factory.current_open_games(
        called=True,
        requested_by="llm",
        tool_plan_source="llm",
        decision_action="join_game",
        call_reason="用户在问现成局。",
        sender_id="zhang",
        game_type="杭麻",
        level_options=["0.5", "1"],
        smoke_preference="any",
        smoke_options=["no_smoke", "smoke_ok"],
        time_window=window,
        source_text="现在有人吗",
    )

    assert request["tool_name"] == "search_current_open_games"
    assert request["called"] is True
    assert request["requested_by"] == "llm"
    assert request["query"] == {
        "sender_id": "zhang",
        "game_type": "杭麻",
        "level_options": ["0.5", "1"],
        "smoke_preference": "any",
        "smoke_options": ["no_smoke", "smoke_ok"],
        "time_window": ["2026-07-01T17:00:00+08:00", "2026-07-01T18:30:00+08:00"],
        "source_text": "现在有人吗",
    }
    assert "不返回当前发送人自己发起的局" in request["hard_filters"]


def test_trial_tool_request_factory_builds_candidate_search_contract() -> None:
    request = TrialToolRequestFactory().candidate_customers(
        requested_by="llm",
        tool_plan_source="llm",
        game_id="game_1",
        game_type="杭麻",
        game_label="杭麻 0.5档 16:00",
        level="0.5",
        start_at=datetime(2026, 7, 1, 16, 0, tzinfo=TZ),
        rules=["无烟"],
        missing_count=3,
        organizer_id="zhang",
        candidate_composition_preference={"desired_gender_counts": {"male": 1, "female": 1}},
    )

    assert request["tool_name"] == "search_candidate_customers"
    assert request["called"] is True
    assert request["risk_level"] == "low"
    assert request["approval_required"] is False
    assert request["query"]["start_at"] == "2026-07-01T16:00:00+08:00"
    assert request["query"]["candidate_composition_preference"] == {"desired_gender_counts": {"male": 1, "female": 1}}
    assert "性别等候选组合偏好是推荐排序约束，不是外发话术内容" in request["hard_filters"]


def test_trial_tool_request_factory_builds_pending_outbox_contract() -> None:
    request = TrialToolRequestFactory().pending_outbox_message(
        called=True,
        requested_by="llm",
        tool_plan_source="llm",
        game_id="game_1",
        recipient_count=5,
    )

    assert request["tool_name"] == "send_message"
    assert request["called"] is True
    assert request["risk_level"] == "high"
    assert request["approval_required"] is True
    assert request["direct_send_allowed"] is False
    assert request["execution_mode"] == "create_pending_outbox"
    assert request["query"] == {
        "game_id": "game_1",
        "recipient_count": 5,
        "channel_policy": "老板审批后复制/发送",
    }
    assert "禁止直接发送真实消息" in request["hard_filters"]
