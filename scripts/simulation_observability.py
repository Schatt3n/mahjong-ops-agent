"""Read-only aggregation and allowlisted controls for chat simulations."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME_DIR = ROOT / "runtime_data" / "periodic_chat_simulation"
DEFAULT_ENV_FILE = ROOT / ".env.simulation.local"
STATIC_PAGE_PATH = ROOT / "scripts" / "static" / "simulation_observability.html"
_CONTROL_LOCK = threading.Lock()


def simulation_dashboard_html() -> str:
    return STATIC_PAGE_PATH.read_text(encoding="utf-8")


def simulation_observability_payload(
    *,
    runtime_dir: Path | None = None,
    selected_run_id: str | None = None,
    max_runs: int = 30,
) -> dict[str, Any]:
    """Combine scheduler state, completed reports, and the current live stream."""

    base = (runtime_dir or DEFAULT_RUNTIME_DIR).expanduser().resolve()
    state = _load_json(base / "state.json")
    control = _load_json(base / "control.json", {"enabled": True})
    run_summaries = _list_run_summaries(base, max_runs=max_runs)
    selected_id = _safe_run_id(selected_run_id)
    if not selected_id:
        selected_id = str(state.get("current_run_id") or state.get("last_run_id") or "")
    if not selected_id and run_summaries:
        selected_id = str(run_summaries[0]["run_id"])

    selected = _load_run(base, selected_id) if selected_id else _empty_run()
    generator_env = _read_env_metadata(DEFAULT_ENV_FILE)
    pid = int(state.get("pid") or 0)
    scheduler = {
        **state,
        "enabled": bool(control.get("enabled", True)),
        "run_requested": bool(control.get("run_request_id"))
        and control.get("run_request_id") != state.get("handled_run_request_id"),
        "process_alive": _process_alive(pid),
        "runtime_dir": str(base),
        "event_log_path": str(base / "events.jsonl"),
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scheduler": scheduler,
        "generator": {
            "configured": bool(generator_env.get("MAHJONG_SIM_GENERATOR_API_KEY")),
            "mode": generator_env.get("MAHJONG_PERIODIC_SIM_MESSAGE_MODE", "rule"),
            "model": generator_env.get("MAHJONG_SIM_GENERATOR_MODEL", "glm-4.7-flash"),
            "base_url": generator_env.get(
                "MAHJONG_SIM_GENERATOR_BASE_URL",
                "https://open.bigmodel.cn/api/paas/v4",
            ),
            "api_key_exposed": False,
        },
        "runs": run_summaries,
        "selected_run": selected,
        "scheduler_events": _tail_jsonl(base / "events.jsonl", 80),
    }


def update_simulation_control(
    action: str,
    *,
    runtime_dir: Path | None = None,
) -> dict[str, Any]:
    """Apply an allowlisted scheduler command without accepting shell input."""

    normalized = str(action or "").strip().lower()
    if normalized not in {"pause", "resume", "run_now", "start"}:
        raise ValueError("unsupported simulation control action")
    base = (runtime_dir or DEFAULT_RUNTIME_DIR).expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)
    if normalized == "start":
        simulator_env = _read_env_metadata(DEFAULT_ENV_FILE)
        llm_mode = simulator_env.get("MAHJONG_PERIODIC_SIM_LLM_MODE", "real")
        message_mode = simulator_env.get("MAHJONG_PERIODIC_SIM_MESSAGE_MODE", "glm")
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "run_periodic_chat_simulator.py"),
                "start",
                "--runtime-dir",
                str(base),
                "--env-file",
                str(DEFAULT_ENV_FILE),
                "--message-mode",
                message_mode,
                "--mode",
                llm_mode,
                "--workers",
                "1",
                "--rate",
                "1",
                "--request-timeout",
                "180",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        return {
            "ok": completed.returncode == 0,
            "action": normalized,
            "return_code": completed.returncode,
            "stdout": completed.stdout[-2000:],
            "stderr": completed.stderr[-2000:],
        }

    with _CONTROL_LOCK:
        path = base / "control.json"
        control = _load_json(path, {"enabled": True})
        if normalized == "pause":
            control["enabled"] = False
        elif normalized == "resume":
            control["enabled"] = True
        else:
            control["enabled"] = True
            control["run_request_id"] = uuid.uuid4().hex
        control["updated_at"] = datetime.now(timezone.utc).isoformat()
        _atomic_write_json(path, control)
    return {"ok": True, "action": normalized, "control": control}


def _list_run_summaries(runtime_dir: Path, *, max_runs: int) -> list[dict[str, Any]]:
    runs_dir = runtime_dir / "runs"
    if not runs_dir.exists():
        return []
    summaries: list[dict[str, Any]] = []
    for run_dir in sorted(
        (item for item in runs_dir.iterdir() if item.is_dir()),
        key=lambda item: item.name,
        reverse=True,
    )[: max(1, min(int(max_runs), 100))]:
        report = _load_json(run_dir / "report.json")
        live_turns = _tail_jsonl(run_dir / "live_events.jsonl", 10000)
        transcript = report.get("transcript") if isinstance(report.get("transcript"), list) else live_turns
        summaries.append(
            {
                "run_id": run_dir.name,
                "status": str(report.get("status") or ("running" if live_turns else "starting")),
                "started_at": report.get("started_at"),
                "finished_at": report.get("finished_at"),
                "total_messages": int(report.get("total_messages") or len(transcript or [])),
                "group_messages": int(report.get("group_messages") or _count_channel(transcript, "group")),
                "private_messages": int(report.get("private_messages") or _count_channel(transcript, "private")),
                "agent_response_latency_ms": report.get("agent_response_latency_ms") or {},
                "llm_mode": report.get("llm_mode"),
                "message_generation_mode": report.get("message_generation_mode") or _generation_mode(transcript),
                "message_generation_model": report.get("message_generation_model") or _generation_model(transcript),
            }
        )
    return summaries


def _load_run(runtime_dir: Path, run_id: str) -> dict[str, Any]:
    safe_id = _safe_run_id(run_id)
    if not safe_id:
        return _empty_run()
    run_dir = runtime_dir / "runs" / safe_id
    report = _load_json(run_dir / "report.json")
    live_events = _tail_jsonl(run_dir / "live_events.jsonl", 10000)
    transcript = report.get("transcript") if isinstance(report.get("transcript"), list) else live_events
    transcript = [item for item in transcript or [] if isinstance(item, dict)]
    conversations = _group_conversations(transcript)
    return {
        "run_id": safe_id,
        "status": str(report.get("status") or ("running" if live_events else "starting")),
        "report": report,
        "report_path": str(run_dir / "report.json"),
        "live_event_path": str(run_dir / "live_events.jsonl"),
        "transcript": transcript,
        "conversations": conversations,
        "turn_count": len(transcript),
    }


def _group_conversations(transcript: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for turn in transcript:
        conversation_id = str(turn.get("conversation_id") or "unknown")
        user = turn.get("user") if isinstance(turn.get("user"), dict) else {}
        record = grouped.setdefault(
            conversation_id,
            {
                "conversation_id": conversation_id,
                "channel": str(turn.get("channel") or "private"),
                "participants": {},
                "turns": [],
                "latest_observed_at": "",
            },
        )
        customer_id = str(user.get("customer_id") or "unknown")
        record["participants"][customer_id] = str(user.get("display_name") or customer_id)
        record["turns"].append(turn)
        record["latest_observed_at"] = str(turn.get("observed_at") or record["latest_observed_at"])
    result: list[dict[str, Any]] = []
    for record in grouped.values():
        record["participants"] = [
            {"customer_id": customer_id, "display_name": display_name}
            for customer_id, display_name in record["participants"].items()
        ]
        record["turn_count"] = len(record["turns"])
        result.append(record)
    return sorted(
        result,
        key=lambda item: (
            str(item.get("latest_observed_at") or ""),
            int((item.get("turns") or [{}])[-1].get("sequence") or 0),
        ),
        reverse=True,
    )


def _read_env_metadata(path: Path) -> dict[str, str]:
    allowed = {
        "MAHJONG_PERIODIC_SIM_LLM_MODE",
        "MAHJONG_SIM_GENERATOR_API_KEY",
        "MAHJONG_SIM_GENERATOR_MODEL",
        "MAHJONG_SIM_GENERATOR_BASE_URL",
        "MAHJONG_PERIODIC_SIM_MESSAGE_MODE",
    }
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key in allowed:
            values[key] = value.strip().strip("'").strip('"')
    for key in allowed:
        if os.getenv(key):
            values[key] = str(os.getenv(key))
    return values


def _safe_run_id(run_id: str | None) -> str:
    value = str(run_id or "").strip()
    return value if value and value.replace("_", "").isalnum() else ""


def _count_channel(transcript: Any, channel: str) -> int:
    return sum(
        isinstance(item, dict) and str(item.get("channel") or "") == channel
        for item in transcript or []
    )


def _generation_mode(transcript: Any) -> str:
    sources = {
        str(((item.get("user") or {}).get("generation") or {}).get("source") or "")
        for item in transcript or []
        if isinstance(item, dict)
    }
    return "glm" if "glm" in sources else ("rule_fallback" if "rule_fallback" in sources else "rule")


def _generation_model(transcript: Any) -> str | None:
    for item in transcript or []:
        if not isinstance(item, dict):
            continue
        model = ((item.get("user") or {}).get("generation") or {}).get("model")
        if model:
            return str(model)
    return None


def _load_json(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return dict(default or {})
    return value if isinstance(value, dict) else dict(default or {})


def _tail_jsonl(path: Path, limit: int) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()[-max(1, limit) :]
    except OSError:
        return []
    values: list[dict[str, Any]] = []
    for line in lines:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            values.append(item)
    return values


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _empty_run() -> dict[str, Any]:
    return {
        "run_id": "",
        "status": "empty",
        "report": {},
        "report_path": "",
        "live_event_path": "",
        "transcript": [],
        "conversations": [],
        "turn_count": 0,
    }
