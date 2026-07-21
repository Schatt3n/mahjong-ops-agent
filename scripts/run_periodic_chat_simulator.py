#!/usr/bin/env python3
"""Run small randomized chat simulations every one to two hours.

This is operational test infrastructure, not a customer message channel. Every
run receives its own SQLite database and report directory, so a simulation can
never mutate the development or WeChat runtime state.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import random
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
SIMULATION_DIR = ROOT / "tests" / "simulation"
if str(SIMULATION_DIR) not in sys.path:
    sys.path.insert(0, str(SIMULATION_DIR))

from hundred_user_simulator import HundredUserSimulator  # noqa: E402
from message_generation import build_message_generator  # noqa: E402
from sim_adapter import StaticAgentLLMClient, build_runtime, running_http_backend  # noqa: E402
from sim_factory import build_population  # noqa: E402
from mahjong_agent_runtime.env import load_dotenv_defaults  # noqa: E402


DEFAULT_RUNTIME_DIR = ROOT / "runtime_data" / "periodic_chat_simulation"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_time(timestamp: float | None = None) -> str:
    value = utc_now() if timestamp is None else datetime.fromtimestamp(timestamp, timezone.utc)
    return value.isoformat()


def load_json(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return dict(default or {})
    return value if isinstance(value, dict) else dict(default or {})


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


@dataclass(slots=True, frozen=True)
class PeriodicSimulationConfig:
    runtime_dir: Path = DEFAULT_RUNTIME_DIR
    mode: str = "mock"
    message_mode: str = "rule"
    env_file: Path | None = None
    min_interval_seconds: float = 3600.0
    max_interval_seconds: float = 7200.0
    min_messages: int = 6
    max_messages: int = 12
    max_duration_seconds: float = 300.0
    max_workers: int = 3
    rate_limit: int = 2
    speed: float = 100.0
    request_timeout_seconds: float = 60.0
    poll_seconds: float = 2.0

    def __post_init__(self) -> None:
        if self.mode not in {"mock", "real"}:
            raise ValueError("mode must be mock or real")
        if self.message_mode not in {"rule", "glm"}:
            raise ValueError("message_mode must be rule or glm")
        if self.min_interval_seconds <= 0 or self.max_interval_seconds < self.min_interval_seconds:
            raise ValueError("interval range is invalid")
        if self.min_messages <= 0 or self.max_messages < self.min_messages:
            raise ValueError("message range is invalid")
        if not 1 <= self.max_workers <= 10:
            raise ValueError("max_workers must be between 1 and 10")
        if not 1 <= self.rate_limit <= 5:
            raise ValueError("rate_limit must be between 1 and 5")

    @property
    def state_path(self) -> Path:
        return self.runtime_dir / "state.json"

    @property
    def control_path(self) -> Path:
        return self.runtime_dir / "control.json"

    @property
    def event_log_path(self) -> Path:
        return self.runtime_dir / "events.jsonl"

    @property
    def lock_path(self) -> Path:
        return self.runtime_dir / "scheduler.lock"

    @property
    def daemon_log_path(self) -> Path:
        return self.runtime_dir / "daemon.log"


@dataclass(slots=True, frozen=True)
class SimulationRunSpec:
    run_id: str
    seed: int
    message_limit: int
    database_path: Path
    report_path: Path
    config: PeriodicSimulationConfig


SimulationRunner = Callable[[SimulationRunSpec], dict[str, Any]]


def run_randomized_simulation(spec: SimulationRunSpec) -> dict[str, Any]:
    """Execute one isolated mini simulation using the normal Agent Runtime."""

    load_dotenv_defaults(ROOT / ".env")
    if spec.config.env_file is not None:
        load_dotenv_defaults(spec.config.env_file)
    store, users = build_population(spec.database_path, seed=spec.seed)
    runtime, llm_client = build_runtime(store, spec.config.mode, seed=spec.seed)
    message_generator = build_message_generator(spec.config.message_mode)
    live_event_path = spec.report_path.parent / "live_events.jsonl"

    def record_live_turn(turn: dict[str, Any]) -> None:
        append_jsonl(
            live_event_path,
            {"event": "chat_turn", "run_id": spec.run_id, **turn},
        )

    with running_http_backend(runtime) as base_url:
        simulator = HundredUserSimulator(
            users=users,
            base_url=base_url,
            seed=spec.seed,
            max_messages=spec.message_limit,
            max_duration_seconds=spec.config.max_duration_seconds,
            max_workers=spec.config.max_workers,
            rate_limit=spec.config.rate_limit,
            speed=spec.config.speed,
            request_timeout_seconds=spec.config.request_timeout_seconds,
            report_path=spec.report_path,
            message_generator=message_generator,
            event_sink=record_live_turn,
        )
        report = simulator.run()
    report.update(
        {
            "run_id": spec.run_id,
            "llm_mode": spec.config.mode,
            "message_generation_mode": spec.config.message_mode,
            "message_generation_model": (
                message_generator.config.model if message_generator is not None else None
            ),
            "database_path": str(spec.database_path),
            "report_path": str(spec.report_path),
            "live_event_path": str(live_event_path),
            "mock_llm_call_count": (
                llm_client.call_count if isinstance(llm_client, StaticAgentLLMClient) else None
            ),
        }
    )
    atomic_write_json(spec.report_path, report)
    return report


class PeriodicChatSimulationScheduler:
    """Persistent scheduler with pause, resume, run-now, and run history."""

    def __init__(
        self,
        config: PeriodicSimulationConfig,
        *,
        runner: SimulationRunner = run_randomized_simulation,
        rng: random.Random | random.SystemRandom | None = None,
        time_fn: Callable[[], float] = time.time,
    ) -> None:
        self.config = config
        self.runner = runner
        self.rng = rng or random.SystemRandom()
        self.time_fn = time_fn
        self.stop_event = threading.Event()
        self.config.runtime_dir.mkdir(parents=True, exist_ok=True)

    def interval_seconds(self) -> float:
        return self.rng.uniform(
            self.config.min_interval_seconds,
            self.config.max_interval_seconds,
        )

    def status(self) -> dict[str, Any]:
        state = load_json(self.config.state_path)
        control = load_json(self.config.control_path, {"enabled": True})
        pid = int(state.get("pid") or 0)
        state["process_alive"] = process_alive(pid)
        state["enabled"] = bool(control.get("enabled", True))
        state["run_requested"] = bool(control.get("run_request_id")) and (
            control.get("run_request_id") != state.get("handled_run_request_id")
        )
        state["state_path"] = str(self.config.state_path)
        state["event_log_path"] = str(self.config.event_log_path)
        return state

    def run_once(self) -> dict[str, Any]:
        timestamp = self.time_fn()
        run_id = datetime.fromtimestamp(timestamp, timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{run_id}_{uuid.uuid4().hex[:8]}"
        seed = self.rng.randrange(1, 2**31)
        message_limit = self.rng.randint(self.config.min_messages, self.config.max_messages)
        run_dir = self.config.runtime_dir / "runs" / run_id
        spec = SimulationRunSpec(
            run_id=run_id,
            seed=seed,
            message_limit=message_limit,
            database_path=run_dir / "test_sim.db",
            report_path=run_dir / "report.json",
            config=self.config,
        )
        self._merge_state(
            status="running",
            current_run_id=run_id,
            current_run_started_at=iso_time(timestamp),
            current_seed=seed,
            current_message_limit=message_limit,
            llm_mode=self.config.mode,
            message_generation_mode=self.config.message_mode,
        )
        self._event(
            "run_started",
            run_id=run_id,
            seed=seed,
            message_limit=message_limit,
            llm_mode=self.config.mode,
            message_generation_mode=self.config.message_mode,
        )
        try:
            report = self.runner(spec)
        except Exception as exc:
            self._event("run_failed", run_id=run_id, error_type=type(exc).__name__, error=str(exc))
            self._merge_state(
                status="idle",
                current_run_id=None,
                last_run_id=run_id,
                last_run_status="failed",
                last_run_finished_at=iso_time(self.time_fn()),
                last_error=f"{type(exc).__name__}: {exc}",
                last_report_path=str(spec.report_path),
            )
            raise
        self._event(
            "run_completed",
            run_id=run_id,
            total_messages=report.get("total_messages"),
            report_path=str(spec.report_path),
        )
        self._merge_state(
            status="idle",
            current_run_id=None,
            last_run_id=run_id,
            last_run_status=str(report.get("status") or "completed"),
            last_run_finished_at=iso_time(self.time_fn()),
            last_error="",
            last_report_path=str(spec.report_path),
            last_transcript_turns=len(report.get("transcript") or []),
        )
        return report

    def run_forever(self, *, run_immediately: bool = True) -> None:
        state = load_json(self.config.state_path)
        next_run_at = float(state.get("next_run_timestamp") or 0.0)
        if next_run_at <= 0:
            next_run_at = self.time_fn() if run_immediately else self.time_fn() + self.interval_seconds()
        handled_request_id = str(state.get("handled_run_request_id") or "")
        self._merge_state(status="idle", pid=os.getpid(), started_at=iso_time(), stopped_at=None)
        self._event(
            "scheduler_started",
            pid=os.getpid(),
            mode=self.config.mode,
            message_generation_mode=self.config.message_mode,
        )
        while not self.stop_event.is_set():
            control = load_json(self.config.control_path, {"enabled": True})
            enabled = bool(control.get("enabled", True))
            request_id = str(control.get("run_request_id") or "")
            if not enabled:
                self._merge_state(status="paused", next_run_timestamp=next_run_at, next_run_at=iso_time(next_run_at))
                self.stop_event.wait(self.config.poll_seconds)
                continue

            now_timestamp = self.time_fn()
            requested = bool(request_id and request_id != handled_request_id)
            if requested or now_timestamp >= next_run_at:
                if requested:
                    handled_request_id = request_id
                    self._merge_state(handled_run_request_id=handled_request_id)
                try:
                    self.run_once()
                except Exception:
                    pass
                next_run_at = self.time_fn() + self.interval_seconds()
                self._merge_state(
                    status="idle",
                    next_run_timestamp=next_run_at,
                    next_run_at=iso_time(next_run_at),
                )
                self._event("next_run_scheduled", next_run_at=iso_time(next_run_at))
                continue

            self._merge_state(
                status="idle",
                next_run_timestamp=next_run_at,
                next_run_at=iso_time(next_run_at),
            )
            self.stop_event.wait(min(self.config.poll_seconds, max(0.05, next_run_at - now_timestamp)))
        self._merge_state(status="stopped", stopped_at=iso_time(), pid=None)
        self._event("scheduler_stopped")

    def stop(self) -> None:
        self.stop_event.set()

    def _merge_state(self, **updates: Any) -> None:
        state = load_json(self.config.state_path)
        state.update(updates)
        state["updated_at"] = iso_time(self.time_fn())
        atomic_write_json(self.config.state_path, state)

    def _event(self, event: str, **payload: Any) -> None:
        append_jsonl(
            self.config.event_log_path,
            {"time": iso_time(self.time_fn()), "event": event, **payload},
        )


def process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # The process exists, but this observer is not allowed to signal it.
        return True
    return True


def update_control(config: PeriodicSimulationConfig, **updates: Any) -> dict[str, Any]:
    control = load_json(config.control_path, {"enabled": True})
    control.update(updates)
    control["updated_at"] = iso_time()
    atomic_write_json(config.control_path, control)
    return control


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("daemon", "once", "start", "status", "pause", "resume", "run-now"))
    parser.add_argument("--runtime-dir", type=Path, default=Path(os.getenv("MAHJONG_PERIODIC_SIM_RUNTIME_DIR", DEFAULT_RUNTIME_DIR)))
    parser.add_argument("--mode", choices=("mock", "real"), default=os.getenv("MAHJONG_PERIODIC_SIM_LLM_MODE", "mock"))
    parser.add_argument(
        "--message-mode",
        choices=("rule", "glm"),
        default=os.getenv("MAHJONG_PERIODIC_SIM_MESSAGE_MODE", "rule"),
        help="Generate synthetic customer speech with deterministic rules or GLM.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(os.getenv("MAHJONG_PERIODIC_SIM_ENV_FILE", ROOT / ".env.simulation.local")),
        help="Local secrets for the simulation generator; values are never copied into reports.",
    )
    parser.add_argument("--min-interval", type=float, default=env_float("MAHJONG_PERIODIC_SIM_MIN_INTERVAL_SECONDS", 3600.0))
    parser.add_argument("--max-interval", type=float, default=env_float("MAHJONG_PERIODIC_SIM_MAX_INTERVAL_SECONDS", 7200.0))
    parser.add_argument("--min-messages", type=int, default=env_int("MAHJONG_PERIODIC_SIM_MIN_MESSAGES", 6))
    parser.add_argument("--max-messages", type=int, default=env_int("MAHJONG_PERIODIC_SIM_MAX_MESSAGES", 12))
    parser.add_argument("--duration", type=float, default=env_float("MAHJONG_PERIODIC_SIM_DURATION_SECONDS", 300.0))
    parser.add_argument("--workers", type=int, default=env_int("MAHJONG_PERIODIC_SIM_WORKERS", 3))
    parser.add_argument("--rate", type=int, default=env_int("MAHJONG_PERIODIC_SIM_RATE_LIMIT", 2))
    parser.add_argument("--speed", type=float, default=env_float("MAHJONG_PERIODIC_SIM_SPEED", 100.0))
    parser.add_argument("--request-timeout", type=float, default=env_float("MAHJONG_PERIODIC_SIM_REQUEST_TIMEOUT_SECONDS", 60.0))
    parser.add_argument("--poll", type=float, default=env_float("MAHJONG_PERIODIC_SIM_POLL_SECONDS", 2.0))
    parser.add_argument("--wait-first", action="store_true", help="Do not run immediately when the daemon first starts.")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> PeriodicSimulationConfig:
    return PeriodicSimulationConfig(
        runtime_dir=args.runtime_dir.expanduser().resolve(),
        mode=args.mode,
        message_mode=args.message_mode,
        env_file=args.env_file.expanduser().resolve() if args.env_file else None,
        min_interval_seconds=args.min_interval,
        max_interval_seconds=args.max_interval,
        min_messages=args.min_messages,
        max_messages=args.max_messages,
        max_duration_seconds=args.duration,
        max_workers=args.workers,
        rate_limit=args.rate,
        speed=args.speed,
        request_timeout_seconds=args.request_timeout,
        poll_seconds=max(0.1, args.poll),
    )


def daemon_cli_args(args: argparse.Namespace) -> list[str]:
    values = [
        sys.executable,
        str(Path(__file__).resolve()),
        "daemon",
        "--runtime-dir", str(args.runtime_dir),
        "--mode", args.mode,
        "--message-mode", args.message_mode,
        "--env-file", str(args.env_file),
        "--min-interval", str(args.min_interval),
        "--max-interval", str(args.max_interval),
        "--min-messages", str(args.min_messages),
        "--max-messages", str(args.max_messages),
        "--duration", str(args.duration),
        "--workers", str(args.workers),
        "--rate", str(args.rate),
        "--speed", str(args.speed),
        "--request-timeout", str(args.request_timeout),
        "--poll", str(args.poll),
    ]
    if args.wait_first:
        values.append("--wait-first")
    return values


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = config_from_args(args)
    scheduler = PeriodicChatSimulationScheduler(config)

    if args.command == "status":
        print(json.dumps(scheduler.status(), ensure_ascii=False, indent=2))
        return 0
    if args.command == "pause":
        update_control(config, enabled=False)
        print(json.dumps(scheduler.status(), ensure_ascii=False, indent=2))
        return 0
    if args.command == "resume":
        update_control(config, enabled=True)
        print(json.dumps(scheduler.status(), ensure_ascii=False, indent=2))
        return 0
    if args.command == "run-now":
        update_control(config, enabled=True, run_request_id=uuid.uuid4().hex)
        print(json.dumps(scheduler.status(), ensure_ascii=False, indent=2))
        return 0
    if args.command == "once":
        report = scheduler.run_once()
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1 if report.get("status") == "failed" else 0
    if args.command == "start":
        status = scheduler.status()
        if status.get("process_alive"):
            print(json.dumps({"started": False, "reason": "already_running", **status}, ensure_ascii=False, indent=2))
            return 0
        config.runtime_dir.mkdir(parents=True, exist_ok=True)
        update_control(config, enabled=True)
        with config.daemon_log_path.open("a", encoding="utf-8") as output:
            process = subprocess.Popen(
                daemon_cli_args(args),
                cwd=ROOT,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        print(json.dumps({"started": True, "pid": process.pid, "daemon_log": str(config.daemon_log_path)}, ensure_ascii=False, indent=2))
        return 0

    config.runtime_dir.mkdir(parents=True, exist_ok=True)
    with config.lock_path.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("periodic chat simulator is already running", file=sys.stderr)
            return 2

        def request_stop(_signum: int, _frame: Any) -> None:
            scheduler.stop()

        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        scheduler.run_forever(run_immediately=not args.wait_first)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
