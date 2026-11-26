"""核验并发营业会话的因果续轮和合理未成局验收。"""

from datetime import date, datetime
import json
from types import SimpleNamespace

import pytest

from scripts.run_venue_simulation import run_simulation
from tests.simulation.venue_actors import ActorContext, ModelActor, generate_personas
from tests.simulation.venue_dialogues import VenueDialogues, _model_failure
from tests.simulation.venue_judging import AGENT_CHECK, ROLE_CHECK


def test_open_dialogues_accept_grounded_no_match_and_preserve_each_persons_causality(tmp_path):
    """空局池是一种合法营业结果，私聊续轮需来自本人已送达的回复。"""
    run_dir = tmp_path / "open-dialogues"
    report = run_simulation(run_dir, mode="mock", seed=42, scenario_date=date(2026, 11, 25),
                            background_events=2, scenario_name="dialogues", active_users=6, workers=3,
                            dialogue_turns=3)
    assert report["status"] == "passed", report
    outcome = report["acceptance"]
    assert report["people"] == 500 and report["targeted_users"] == 6
    assert outcome["multi_round_private_customers"] == 6
    assert outcome["outcome_counts"] == {"no_match": 6}
    assert outcome["valid_non_formation_count"] == 6
    assert 2 <= report["peak_inflight_deliveries"] <= 3

    archived = {row["message_id"]: row for line in (run_dir / "chat.jsonl").read_text().splitlines()
                if (row := json.loads(line))}
    followups = [row for line in (run_dir / "events.jsonl").read_text().splitlines()
                 if (row := json.loads(line)).get("caused_by") and row.get("message_id")]
    assert len(followups) >= 6
    for row in followups:
        cause = archived[row["caused_by"]]
        assert cause["kind"] == "private" and cause["recipient_id"] == row["actor_id"]
        assert cause["sequence"] < archived[row["message_id"]]["sequence"]
        assert archived[row["message_id"]]["quoted_message_id"] is None


def test_provider_incomplete_and_timeout_have_distinct_failure_evidence():
    """收到 HTTP 200 后的输出截断和等待连接时超时有不同归因。"""
    assert _model_failure([{"completion_diagnostics": {"attempts": [
        {"http_status": 200, "incomplete_reason": "max_output_tokens"},
    ]}}]) == {"error_type": "model_output_incomplete", "http_statuses": [200]}
    assert _model_failure([{"completion_diagnostics": {"attempts": [
        {"http_status": None, "error_type": "TimeoutError"},
    ]}}]) == {"error_type": "timeout", "http_statuses": []}


@pytest.mark.parametrize(("check", "citations", "expected"), [
    (AGENT_CHECK, ("ar1",), "pending"),
    (AGENT_CHECK, ("a1",), "review_required"),
    (ROLE_CHECK, ("ar1",), "review_required"),
])
def test_open_dialogue_only_promotes_pending_from_the_linked_shop_finding(tmp_path, monkeypatch, check, citations, expected):
    """顾客项和其他引用不能替代最后一轮实质答复的店家评审。"""
    person = generate_personas(42, count=1)[0]
    incoming = {"message_id": "a1", "kind": "private", "direction": "inbound", "conversation_id": "dm",
                "sender_id": person.sender_id, "recipient_id": "simulation-agent", "text": "刚才那桌我还能来吗"}
    outgoing = {**incoming, "message_id": "ar1", "direction": "outbound", "source_message_id": "a1",
                "sender_id": "simulation-agent", "recipient_id": person.sender_id, "text": "你说的是群里1号吗？"}
    scenario = VenueDialogues(people=[person], actor=ModelActor(None), judge=object(), store=None,
        channel=SimpleNamespace(events=lambda: [incoming, outgoing]), run_dir=tmp_path, seed=42,
        start_at=datetime.fromisoformat("2026-11-25T19:00:00+08:00"), active_users=1,
        workers=1, dialogue_turns=2, background_events=0)

    def turn(session, at, index):
        result = {"event_id": str(index), "actor_id": person.sender_id, "acted": True, "decision": {
            "source": "model", "status": "valid", "response_source": {"http_response_complete": True},
        }}
        if index == 1:
            result.update(message_id="a1", runtime_status="waiting_user", reply_message_id="ar1")
        return result

    monkeypatch.setattr(scenario, "_turn", turn)
    monkeypatch.setattr(scenario, "_review_samples", lambda records, outcomes: [
        {"status": "valid", "verdict": "pass", "findings": [{"check": check, "evidence_ids": citations, "verdict": "pass"}],
         "evidence": [{"kind": "agent_reply", "evidence_id": "ar1"}, {"kind": "customer_message", "evidence_id": "a1"}]},
    ])
    report = scenario.run()
    assert report["acceptance"]["outcomes"][0]["kind"] == "clarification_requested"
    assert report["acceptance"]["outcomes"][0]["status"] == expected
    assert report["checks"]["outcomes_grounded"] is (expected == "pending")


def test_private_role_can_read_its_group_with_no_permission_to_send_there():
    """私聊角色看得到所属群的历史，同时仍受私聊演练的发言目标限制。"""
    person = generate_personas(0, count=1)[0]
    events = (
        {"message_id": "ours", "kind": "group", "recipient_id": "venue-group"},
        {"message_id": "other", "kind": "group", "recipient_id": "other-group"},
        {"message_id": "foreign-dm", "kind": "private", "sender_id": "other", "recipient_id": "simulation-agent"},
    )
    projected = ActorContext(person, "2026-11-25T18:00:00+08:00", history=events,
                             visible_groups=("venue-group",)).to_dict()
    assert projected["group_targets"] == []
    assert [item["message_id"] for item in projected["history"]] == ["ours"]
