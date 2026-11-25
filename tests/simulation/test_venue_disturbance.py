"""核对试用演练的渠道、独立模型配置和持久化状态验证。"""

from datetime import date
import json

import pytest

from mahjong_agent_runtime.llm import AgentLLMConfig, OPENCODE_GO_BASE_URL
from scripts.run_venue_simulation import _real_roles, run_simulation

from .venue_actors import ModelActor
from .venue_channel import AGENT_ID, OWNER_ID
from .venue_disturbance import DISTURBANCE_CASES


def _jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_mock_disturbances_check_real_chat_delivery_and_domain_state(tmp_path):
    """离线角色不冒充 OpenCode，取消、拒邀和两场牌局核对的仍是正式业务库。"""
    report = run_simulation(
        tmp_path / "exercise", mode="mock", seed=42, scenario_date=date(2026, 11, 25),
        scenario_name="disturbances", background_events=0,
    )
    assert report["status"] == "passed", report
    assert report["model_execution"]["role"]["provider"] == "mock"
    assert report["real_customer_actions"] == 0
    assert {entry["case"] for entry in report["disturbances"]} == set(DISTURBANCE_CASES)
    assert {entry["game_type"] for entry in report["disturbances"]} == {"杭州麻将", "四川麻将", "红中麻将"}
    assert report["checks"]["rain.customer_seat_released"]
    assert report["checks"]["traffic.customer_seat_retained"]
    assert report["checks"]["long_commute.original_start_time_intact"]
    assert report["checks"]["out_of_town.invite_decline_recorded"]
    assert report["checks"]["out_of_town.invite_decline_tool_evidence"]
    assert report["checks"]["repeat_visit.morning_play_stayed_finished"]
    states = {row["kind"]: row["persisted_game_state"] for row in _jsonl(tmp_path / "exercise" / "events.jsonl")}
    assert (states["rain"]["customer_seat_before"], states["rain"]["customer_seat_after"]) == (True, False)
    assert (states["traffic"]["customer_seat_before"], states["traffic"]["customer_seat_after"]) == (True, True)
    rows = _jsonl(tmp_path / "exercise" / "chat.jsonl")
    assert any("三四点" in row["text"] for row in rows if row["direction"] == "inbound")
    assert not any("event_condition" in row["text"] for row in rows)
    receipt, = [row for row in rows if row["message_id"] == "shop-receipt:morning-repeat-visit"]
    repeat, = [row for row in _jsonl(tmp_path / "exercise" / "events.jsonl") if row["kind"] == "repeat_visit"]
    assert receipt["kind"] == "private" and receipt["recipient_id"] == repeat["actor_id"]
    assert receipt["sent_at"] < repeat["virtual_at"]
    afternoon_text, = [row["text"] for row in rows if row["message_id"] == repeat["message_id"]]
    assert "三四点" in afternoon_text


def test_operator_invitation_keeps_archival_sender_and_role_answers_shop_alias(tmp_path):
    """店家同一账号由渠道归一；不能把角色模型指向内部店员 ID 或其他人。"""
    root = tmp_path / "alias"
    report = run_simulation(
        root, mode="mock", seed=42, scenario_date=date(2026, 11, 25),
        scenario_name="disturbances", disturbance_case="out_of_town", background_events=0,
    )
    assert report["status"] == "passed", report
    event, = _jsonl(root / "events.jsonl")
    archived = {record["message_id"]: record for record in _jsonl(root / "chat.jsonl")}
    assert archived[event["caused_by"]]["sender_id"] == OWNER_ID
    context = event["decision"]["context"]
    assert context["delivery"]["message_id"] == event["caused_by"]
    assert context["delivery"]["sender_id"] == AGENT_ID
    assert context["private_targets"] == [AGENT_ID]
    assert archived[event["message_id"]]["recipient_id"] == AGENT_ID


@pytest.mark.parametrize("explicit", [True, False])
def test_opencode_role_override_is_respected_only_in_auto_mode(monkeypatch, explicit):
    """显式 OpenCode 测试不受其他角色凭证影响；裁判仍能独立配置。"""
    monkeypatch.setenv("MAHJONG_VENUE_ROLE_API_KEY", "offline-role-token")
    monkeypatch.setenv("MAHJONG_VENUE_ROLE_MODEL", "separate-role")
    monkeypatch.setenv("MAHJONG_VENUE_ROLE_BASE_URL", "https://fixture.invalid/v1")
    monkeypatch.setenv("MAHJONG_VENUE_JUDGE_API_KEY", "offline-judge-token")
    monkeypatch.setenv("MAHJONG_VENUE_JUDGE_MODEL", "separate-judge")
    monkeypatch.setenv("MAHJONG_VENUE_JUDGE_BASE_URL", "https://fixture.invalid/v1")
    config = AgentLLMConfig(model="opencode-fixture", base_url=OPENCODE_GO_BASE_URL, api_key="offline",
                            provider="opencode_go")
    role, judge = _real_roles(config, force_opencode_role=explicit)
    assert isinstance(role, ModelActor)
    assert role.client.config.model == ("opencode-fixture" if explicit else "separate-role")
    assert judge.client.config.model == "separate-judge"
