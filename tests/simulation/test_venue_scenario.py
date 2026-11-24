"""用离线模型合同跑通 500 人门店验收场景并核对证据文件。"""

from datetime import date, datetime, timedelta
import json
import sqlite3

from scripts.run_venue_simulation import run_simulation
from scripts import run_venue_simulation as runner
from tests.simulation.venue_actors import ActorDecision


def test_venue_full_game_keeps_role_truth_chat_and_actual_play_separate(tmp_path):
    """真实群和私聊入口处理模拟发言，并以办结事务和可复订证明散场。"""
    run_dir = tmp_path / "isolated-run"
    report = run_simulation(run_dir, mode="mock", seed=42, scenario_date=date(2026, 11, 25), background_events=4)

    assert report["status"] == "passed", report
    assert report["people"] == 500
    assert report["independent_judge"] is None
    assert report["main_model"] == "offline fixture"
    assert report["room_id"] == "东二包间"
    assert all(report["checks"].values())
    assert report["checks"]["memory_from_visible_message"]
    assert report["checks"]["peer_delivery_isolation"]
    assert report["checks"]["board_delivered"]
    personas = json.loads((run_dir / "personas.json").read_text(encoding="utf-8"))["personas"]
    assert len(personas) == len({person["sender_id"] for person in personas}) == 500
    assert "venue-persona-" not in (run_dir / "agent_trace.log").read_text(encoding="utf-8")

    archive = [json.loads(line) for line in (run_dir / "chat.jsonl").read_text(encoding="utf-8").splitlines()]
    with sqlite3.connect(run_dir / "chat.sqlite3") as database:
        assert database.execute("SELECT COUNT(*) FROM venue_messages").fetchone()[0] == len(archive)
    assert {"inbound", "peer", "outbound"} <= {row["direction"] for row in archive}

    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    started = next(row for row in events if row["kind"] == "start")
    finished = next(row for row in events if row["kind"] == "finish")
    assert datetime.fromisoformat(finished["game"]["actual_ended_at"]) - datetime.fromisoformat(
        started["game"]["actual_started_at"]
    ) == timedelta(hours=4)
    assert len(set(started["game"]["actual_attendee_ids"])) == 4
    assert finished["released"]["status"] == "released"
    assert next(row for row in events if row["kind"] == "rebook")["next_booking"]["room_id"] == report["room_id"]


def test_venue_reports_unavailable_model_without_inventing_traffic(tmp_path, monkeypatch):
    """角色模型传输失败应保留原始原因及报告，且不能补发固定话术。"""

    class BlockedRoleModel:
        def decide(self, context, *, trace_id):
            return ActorDecision(trace_id, "model", "transport_error", None, None, context.to_dict(),
                                 {"attempts": [{"http_status": 402}]}, error="RuntimeError")

    class UnreachableJudge:
        def judge(self, *args, **kwargs):
            raise AssertionError("没有双向证据时不应调用裁判")

    build = runner.build_runtime
    monkeypatch.setattr(runner, "load_dotenv_defaults", lambda path: None)
    monkeypatch.setattr(runner, "_real_roles", lambda: (BlockedRoleModel(), UnreachableJudge()))
    monkeypatch.setattr(runner, "build_runtime", lambda store, mode, **kwargs: build(store, "mock", **kwargs))
    run_dir = tmp_path / "model-blocked"
    report = run_simulation(run_dir, mode="real", seed=42, scenario_date=date(2026, 11, 25),
                            background_events=0, model_source="project")

    assert report["status"] == "inconclusive", report
    assert report["chat_events"] == 0
    assert report["game_id"] is None
    assert report["independent_judge"]["status"] == "not_run"
    assert all(row["http_statuses"] == [402] for row in report["model_transport_errors"])
    assert (run_dir / "events.jsonl").stat().st_size > 0
