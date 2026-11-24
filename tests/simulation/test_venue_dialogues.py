"""核验并发营业会话的因果续轮和合理未成局验收。"""

from datetime import date
import json

from scripts.run_venue_simulation import run_simulation
from tests.simulation.venue_actors import ActorContext, generate_personas
from tests.simulation.venue_dialogues import _model_failure


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
