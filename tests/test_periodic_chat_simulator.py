from __future__ import annotations

import json
import fcntl
import os
import random
import subprocess
import sys
import time
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


def test_real_simulation_defaults_to_model_customers_while_mock_stays_offline() -> None:
    assert PeriodicSimulationConfig(mode="real").message_mode == "model"
    assert PeriodicSimulationConfig(mode="mock").message_mode == "rule"


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


def test_run_is_visible_before_first_model_reply_and_failure_remains_inspectable(tmp_path) -> None:
    config = _config(tmp_path)

    def failed_runner(spec):
        initial = load_json(spec.report_path)
        assert initial['status'] == 'running'
        assert initial['llm_mode'] == 'mock'
        assert initial['configuration']['seed'] == spec.seed
        assert 'transcript' not in initial
        raise RuntimeError('injected startup failure')

    scheduler = PeriodicChatSimulationScheduler(config, runner=failed_runner, rng=random.Random(42))
    import pytest
    with pytest.raises(RuntimeError, match='injected startup failure'):
        scheduler.run_single()
    state = scheduler.status()
    report = load_json(Path(state['last_report_path']))
    assert state['status'] == 'stopped' and state['pid'] is None
    assert report['status'] == 'failed' and report['quality_status'] == 'failed'
    assert report['error'] == 'RuntimeError: injected startup failure'


def test_manual_request_runs_once_while_periodic_schedule_stays_paused(tmp_path) -> None:
    config = _config(tmp_path)
    executions = []

    def fake_runner(spec):
        executions.append(spec.run_id)
        scheduler.stop()
        return {'status': 'completed', 'transcript': []}

    scheduler = PeriodicChatSimulationScheduler(config, runner=fake_runner, rng=random.Random(42))
    update_control(config, enabled=False, run_request_id='manual-request')
    scheduler.run_forever(run_immediately=False)
    assert len(executions) == 1
    assert load_json(config.control_path)['enabled'] is False
    assert scheduler.status()['handled_run_request_id'] == 'manual-request'
    assert scheduler.status()['pid'] is None


def test_run_now_starts_real_single_process_without_existing_scheduler(tmp_path) -> None:
    runtime_dir = tmp_path / 'single'
    runtime_dir.mkdir()
    atomic_write_json(runtime_dir / 'control.json', {'enabled': False})
    script = SCRIPTS / 'run_periodic_chat_simulator.py'
    result = subprocess.run(
        [sys.executable, str(script), 'run-now', '--runtime-dir', str(runtime_dir),
         '--mode', 'mock', '--message-mode', 'rule', '--min-messages', '3', '--max-messages', '3',
         '--workers', '1', '--rate', '1', '--duration', '10'],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
    accepted = json.loads(result.stdout)
    assert accepted['ok'] and accepted['accepted']
    assert accepted['execution_mode'] == 'once'
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        state = load_json(runtime_dir / 'state.json')
        if state.get('status') == 'stopped':
            break
        time.sleep(0.05)
    else:
        raise AssertionError('single simulation did not finish')
    report = load_json(Path(state['last_report_path']))
    assert report['status'] == 'completed'
    assert report['llm_mode'] == 'mock'
    assert report['total_messages'] == 3
    assert report['transcript']
    assert len(list((runtime_dir / 'runs').iterdir())) == 1
    assert state['pid'] is None
    assert not state.get('next_run_at')
    assert load_json(runtime_dir / 'control.json')['enabled'] is False


def test_single_worker_shares_exclusive_lock_and_repeated_request_cannot_spawn_again(tmp_path) -> None:
    script = SCRIPTS / 'run_periodic_chat_simulator.py'
    with (tmp_path / 'scheduler.lock').open('a+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run([sys.executable, str(script), 'once', '--runtime-dir', str(tmp_path)],
                                capture_output=True, text=True, timeout=10)
    assert result.returncode == 2
    assert not (tmp_path / 'runs').exists()
    atomic_write_json(tmp_path / 'state.json', {'pid': os.getpid(), 'worker_mode': 'once',
                                              'status': 'running', 'current_run_id': 'active-run'})
    result = subprocess.run([sys.executable, str(script), 'run-now', '--runtime-dir', str(tmp_path)],
                            capture_output=True, text=True, timeout=10)
    response = json.loads(result.stdout)
    assert result.returncode == 0
    assert response['already_running'] is True and response['run_id'] == 'active-run'
    assert not (tmp_path / 'runs').exists()
    assert not (tmp_path / 'control.json').exists()
