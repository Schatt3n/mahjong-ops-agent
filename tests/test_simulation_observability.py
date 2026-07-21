from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "simulation_observability.py"


def load_module():
    spec = importlib.util.spec_from_file_location("simulation_observability_test", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _turn(sequence: int, *, channel: str = "group") -> dict:
    return {
        "observed_at": f"2026-07-21T10:00:0{sequence}+00:00",
        "sequence": sequence,
        "conversation_id": "sim:group:g1" if channel == "group" else "sim:private:u1",
        "channel": channel,
        "event_type": "text",
        "user": {
            "customer_id": "u1",
            "display_name": "王哥",
            "text": "今晚有局吗",
            "generation": {
                "source": "glm",
                "model": "glm-4.7-flash",
                "trace_id": f"trace_gen_{sequence}",
                "latency_ms": 1200,
                "error": None,
            },
        },
        "agent": {
            "reply": "现在没有，要组吗？",
            "trace_id": f"trace_agent_{sequence}",
            "objective_status": "waiting_user",
        },
        "tool_calls": ["search_current_games"],
        "latency_ms": 2300,
        "status_code": 200,
        "error": "",
    }


def test_dashboard_payload_groups_conversations_and_never_exposes_api_key(tmp_path) -> None:
    module = load_module()
    runtime_dir = tmp_path / "periodic"
    run_dir = runtime_dir / "runs" / "20260721T100000Z_abcd1234"
    run_dir.mkdir(parents=True)
    transcript = [_turn(1), _turn(2, channel="private")]
    (run_dir / "report.json").write_text(
        json.dumps(
            {
                "run_id": run_dir.name,
                "status": "completed",
                "quality_status": "passed",
                "total_messages": 2,
                "group_messages": 1,
                "private_messages": 1,
                "message_generation_mode": "glm",
                "message_generation_model": "glm-4.7-flash",
                "transcript": transcript,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (runtime_dir / "state.json").write_text(
        json.dumps({"last_run_id": run_dir.name}),
        encoding="utf-8",
    )

    payload = module.simulation_observability_payload(runtime_dir=runtime_dir)

    assert payload["selected_run"]["turn_count"] == 2
    assert len(payload["selected_run"]["conversations"]) == 2
    assert payload["runs"][0]["message_generation_model"] == "glm-4.7-flash"
    assert payload["runs"][0]["quality_status"] == "passed"
    assert payload["selected_run"]["quality_status"] == "passed"
    assert "API_KEY" not in json.dumps(payload, ensure_ascii=False)
    assert payload["generator"]["api_key_exposed"] is False


def test_dashboard_reads_live_turns_before_report_is_finished(tmp_path) -> None:
    module = load_module()
    runtime_dir = tmp_path / "periodic"
    run_id = "20260721T100000Z_live1234"
    run_dir = runtime_dir / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "live_events.jsonl").write_text(
        json.dumps({"event": "chat_turn", "run_id": run_id, **_turn(1)}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (runtime_dir / "state.json").write_text(
        json.dumps({"current_run_id": run_id, "status": "running"}),
        encoding="utf-8",
    )

    payload = module.simulation_observability_payload(runtime_dir=runtime_dir)

    assert payload["selected_run"]["status"] == "running"
    assert payload["selected_run"]["transcript"][0]["user"]["text"] == "今晚有局吗"


def test_control_only_accepts_allowlisted_actions(tmp_path) -> None:
    module = load_module()

    try:
        module.update_simulation_control("rm -rf /", runtime_dir=tmp_path)
    except ValueError as exc:
        assert "unsupported" in str(exc)
    else:
        raise AssertionError("arbitrary controls must be rejected")

    result = module.update_simulation_control("run_now", runtime_dir=tmp_path)
    assert result["ok"] is True
    assert result["control"]["enabled"] is True
    assert result["control"]["run_request_id"]


def test_start_control_uses_real_agent_mode_from_private_env(tmp_path, monkeypatch) -> None:
    module = load_module()
    env_file = tmp_path / ".env.simulation.local"
    env_file.write_text(
        "MAHJONG_PERIODIC_SIM_LLM_MODE=real\n"
        "MAHJONG_PERIODIC_SIM_MESSAGE_MODE=glm\n",
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout="started", stderr="")

    monkeypatch.setattr(module, "DEFAULT_ENV_FILE", env_file)
    monkeypatch.setattr(module.subprocess, "run", fake_run)

    result = module.update_simulation_control("start", runtime_dir=tmp_path / "runtime")

    command = captured["command"]
    assert isinstance(command, list)
    assert command[command.index("--mode") + 1] == "real"
    assert command[command.index("--message-mode") + 1] == "glm"
    assert command[command.index("--workers") + 1] == "1"
    assert result["ok"] is True
