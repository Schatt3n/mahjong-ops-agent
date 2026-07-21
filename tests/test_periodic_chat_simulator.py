from __future__ import annotations

import json
import random
import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_periodic_chat_simulator import (  # noqa: E402
    PeriodicChatSimulationScheduler,
    PeriodicSimulationConfig,
    SimulationRunSpec,
    atomic_write_json,
    load_json,
    process_alive,
    update_control,
)


def _config(tmp_path: Path) -> PeriodicSimulationConfig:
    return PeriodicSimulationConfig(
        runtime_dir=tmp_path / "periodic",
        mode="mock",
        min_interval_seconds=3600,
        max_interval_seconds=7200,
        min_messages=3,
        max_messages=7,
        initial_dialog_limit=3,
        max_duration_seconds=5,
        max_workers=1,
        rate_limit=1,
        speed=100,
        poll_seconds=0.1,
    )


def test_interval_and_run_parameters_are_random_but_bounded(tmp_path: Path) -> None:
    scheduler = PeriodicChatSimulationScheduler(_config(tmp_path), rng=random.Random(17))

    intervals = [scheduler.interval_seconds() for _ in range(20)]

    assert all(3600 <= item <= 7200 for item in intervals)
    assert len({round(item, 3) for item in intervals}) > 1


def test_run_once_persists_status_events_and_full_transcript_path(tmp_path: Path) -> None:
    captured: list[SimulationRunSpec] = []

    def fake_runner(spec: SimulationRunSpec) -> dict[str, object]:
        captured.append(spec)
        report = {
            "status": "completed",
            "total_messages": spec.message_limit,
            "transcript": [
                {
                    "user": {"customer_id": "u1", "text": "今晚有人吗"},
                    "agent": {"reply": "有个一块的，打吗？"},
                }
            ],
        }
        atomic_write_json(spec.report_path, report)
        return report

    scheduler = PeriodicChatSimulationScheduler(
        _config(tmp_path),
        runner=fake_runner,
        rng=random.Random(23),
        time_fn=lambda: 1_700_000_000.0,
    )

    report = scheduler.run_once()
    state = scheduler.status()
    events = [json.loads(line) for line in scheduler.config.event_log_path.read_text(encoding="utf-8").splitlines()]

    assert report["status"] == "completed"
    assert len(captured) == 1
    assert 3 <= captured[0].message_limit <= 7
    assert captured[0].database_path.name == "test_sim.db"
    assert captured[0].report_path.exists()
    assert state["last_run_status"] == "completed"
    assert state["last_transcript_turns"] == 1
    assert state["last_report_path"] == str(captured[0].report_path)
    assert [item["event"] for item in events] == ["run_started", "run_completed"]


def test_pause_resume_and_run_now_are_persistent_control_signals(tmp_path: Path) -> None:
    config = _config(tmp_path)
    scheduler = PeriodicChatSimulationScheduler(config)

    update_control(config, enabled=False)
    assert scheduler.status()["enabled"] is False

    update_control(config, enabled=True, run_request_id="request-1")
    status = scheduler.status()
    assert status["enabled"] is True
    assert status["run_requested"] is True

    atomic_write_json(config.state_path, {"handled_run_request_id": "request-1"})
    assert scheduler.status()["run_requested"] is False
    assert load_json(config.control_path)["run_request_id"] == "request-1"


def test_process_alive_treats_permission_denied_as_existing(monkeypatch) -> None:
    def deny_signal(_pid: int, _signal: int) -> None:
        raise PermissionError

    monkeypatch.setattr("run_periodic_chat_simulator.os.kill", deny_signal)

    assert process_alive(12345) is True
