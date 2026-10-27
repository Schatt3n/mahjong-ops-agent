#!/usr/bin/env python3
"""每隔一至两小时运行隔离的随机聊天模拟，并持久化可观测报告。"""

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
from sim_state import build_simulation_state_backend  # noqa: E402
from mahjong_agent_runtime.env import load_dotenv_defaults  # noqa: E402


DEFAULT_RUNTIME_DIR = ROOT / "runtime_data" / "periodic_chat_simulation"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_time(timestamp: float | None = None) -> str:
    value = utc_now() if timestamp is None else datetime.fromtimestamp(timestamp, timezone.utc)
    return value.isoformat()


def load_json(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    """加载json，并恢复为领域层可使用的结构。"""
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
    """追加jsonl，保留原有时间顺序和审计信息。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


@dataclass(slots=True, frozen=True)
class PeriodicSimulationConfig:
    """描述periodicsimulation配置采用的可配置策略参数。"""
    runtime_dir: Path = DEFAULT_RUNTIME_DIR
    mode: str = "mock"
    message_mode: str | None = None
    state_backend: str = "memory"
    redis_url: str = "redis://127.0.0.1:6379/0"
    env_file: Path | None = None
    min_interval_seconds: float = 3600.0
    max_interval_seconds: float = 7200.0
    min_messages: int = 6
    max_messages: int = 12
    initial_dialog_limit: int = 3
    max_duration_seconds: float = 300.0
    max_workers: int = 3
    rate_limit: int = 2
    speed: float = 100.0
    request_timeout_seconds: float = 60.0
    poll_seconds: float = 2.0

    def __post_init__(self) -> None:
        """完成数据类初始化后的依赖组装和不变量校验。"""
        if self.mode not in {"mock", "real"}:
            raise ValueError("mode must be mock or real")
        if self.message_mode is None:
            # 真实模拟默认使用有独立事实与可见历史的模型客户；离线 mock 才默认规则夹具。
            object.__setattr__(self, "message_mode", "model" if self.mode == "real" else "rule")
        if self.message_mode not in {"rule", "glm", "model"}:
            raise ValueError("message_mode must be rule, glm or model")
        if self.state_backend not in {"memory", "redis"}:
            raise ValueError("state_backend must be memory or redis")
        if self.min_interval_seconds <= 0 or self.max_interval_seconds < self.min_interval_seconds:
            raise ValueError("interval range is invalid")
        if self.min_messages <= 0 or self.max_messages < self.min_messages:
            raise ValueError("message range is invalid")
        if not 1 <= self.initial_dialog_limit <= 20:
            raise ValueError("initial_dialog_limit must be between 1 and 20")
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
    """一次周期聊天模拟使用的隔离数据库、报告路径与随机种子配置。"""

    run_id: str
    seed: int
    message_limit: int
    database_path: Path
    report_path: Path
    config: PeriodicSimulationConfig


SimulationRunner = Callable[[SimulationRunSpec], dict[str, Any]]


def run_randomized_simulation(spec: SimulationRunSpec) -> dict[str, Any]:
    """使用常规Agent运行时执行一次隔离的迷你模拟。"""

    load_dotenv_defaults(ROOT / ".env")
    if spec.config.env_file is not None:
        load_dotenv_defaults(spec.config.env_file)
    store, users = build_population(spec.database_path, seed=spec.seed)
    runtime, llm_client = build_runtime(store, spec.config.mode, seed=spec.seed)
    message_generator = build_message_generator(spec.config.message_mode)
    state_backend = build_simulation_state_backend(
        spec.config.state_backend,
        user_ids=(user.customer_id for user in users),
        redis_url=spec.config.redis_url,
        namespace=f"periodic:{spec.run_id}",
    )
    live_event_path = spec.report_path.parent / "live_events.jsonl"

    def record_live_turn(turn: dict[str, Any]) -> None:
        """记录liveturn，形成后续决策和审计所需的持久证据。"""
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
            initial_dialog_limit=spec.config.initial_dialog_limit,
            request_timeout_seconds=spec.config.request_timeout_seconds,
            report_path=spec.report_path,
            message_generator=message_generator,
            event_sink=record_live_turn,
            state_backend=state_backend,
        )
        report = simulator.run()
    report.update(
        {
            "run_id": spec.run_id,
            "llm_mode": spec.config.mode,
            "main_agent_model": getattr(getattr(llm_client, "config", None), "model", None),
            "main_agent_runtime_config": {"max_tokens_per_call": runtime.token_budget.max_tokens_per_call,
                                          "max_calls_per_turn": runtime.token_budget.max_calls_per_turn,
                                          "max_steps": runtime.max_steps},
            "message_generation_mode": spec.config.message_mode,
            "simulation_state_backend": spec.config.state_backend,
            "message_generation_model": (
                message_generator.config.model if message_generator is not None else None
            ),
            "database_path": str(spec.database_path),
            "report_path": str(spec.report_path),
            "live_event_path": str(live_event_path),
            "pending_follow_up_count": sum(task.task_type == "conversation_follow_up" and task.status.value in {"pending", "processing"} for task in runtime.store.scheduled_tasks.values()),
            "mock_llm_call_count": (
                llm_client.call_count if isinstance(llm_client, StaticAgentLLMClient) else None
            ),
        }
    )
    atomic_write_json(spec.report_path, report)
    return report


class PeriodicChatSimulationScheduler:
    """定时创建隔离模拟运行，并管理并发上限、停止信号与结果归档。"""

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
        run_dir.mkdir(parents=True, exist_ok=True)
        spec = SimulationRunSpec(
            run_id=run_id,
            seed=seed,
            message_limit=message_limit,
            database_path=run_dir / "test_sim.db",
            report_path=run_dir / "report.json",
            config=self.config,
        )
        # 启动时立即留下运行记录；模型首轮尚未返回时，页面也能展示新运行。
        atomic_write_json(spec.report_path, {
            "run_id": run_id, "status": "running", "quality_status": "unknown",
            "started_at": iso_time(timestamp), "llm_mode": self.config.mode,
            "message_generation_mode": self.config.message_mode,
            "configuration": {"seed": seed, "max_messages": message_limit},
        })
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
            failed_report = load_json(spec.report_path)
            failed_report.update(status="failed", quality_status="failed", finished_at=iso_time(self.time_fn()),
                                 error=f"{type(exc).__name__}: {exc}")
            atomic_write_json(spec.report_path, failed_report)
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

    def run_single(self) -> dict[str, Any]:
        """在已有进程锁保护下执行一次，并在结束后退出，不开启周期调度。"""
        control = load_json(self.config.control_path)
        self._merge_state(status="starting", worker_mode="once", pid=os.getpid(),
                          started_at=iso_time(), stopped_at=None,
                          handled_run_request_id=control.get("run_request_id"))
        try:
            return self.run_once()
        finally:
            self._merge_state(status="stopped", pid=None, stopped_at=iso_time())

    def run_forever(self, *, run_immediately: bool = True) -> None:
        state = load_json(self.config.state_path)
        next_run_at = float(state.get("next_run_timestamp") or 0.0)
        if next_run_at <= 0:
            next_run_at = self.time_fn() if run_immediately else self.time_fn() + self.interval_seconds()
        handled_request_id = str(state.get("handled_run_request_id") or "")
        self._merge_state(status="idle", worker_mode="daemon", pid=os.getpid(), started_at=iso_time(), stopped_at=None)
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
            requested = bool(request_id and request_id != handled_request_id)
            # 手动一轮独立于周期启停；暂停期间也可手动执行，结束后仍保持暂停。
            if not enabled and not requested:
                self._merge_state(status="paused", next_run_timestamp=next_run_at, next_run_at=iso_time(next_run_at))
                self.stop_event.wait(self.config.poll_seconds)
                continue

            now_timestamp = self.time_fn()
            if requested or (enabled and now_timestamp >= next_run_at):
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
    """处理alive，将输入转换为受控的下一步结果。"""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # 进程仍然存在，但当前观测进程无权向它发送信号。
        return True
    return True


def update_control(config: PeriodicSimulationConfig, **updates: Any) -> dict[str, Any]:
    """更新control，同时维护相关状态不变量。"""
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
    """解析args，并把非结构化输入转换为稳定合同。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("daemon", "once", "start", "status", "pause", "resume", "run-now"))
    parser.add_argument("--runtime-dir", type=Path, default=Path(os.getenv("MAHJONG_PERIODIC_SIM_RUNTIME_DIR", DEFAULT_RUNTIME_DIR)))
    parser.add_argument("--mode", choices=("mock", "real"), default=os.getenv("MAHJONG_PERIODIC_SIM_LLM_MODE", "mock"))
    parser.add_argument(
        "--message-mode",
        choices=("rule", "glm", "model"),
        default=os.getenv("MAHJONG_PERIODIC_SIM_MESSAGE_MODE"),
        help="Generate synthetic customer speech with deterministic scenarios, dedicated GLM or an isolated model customer.",
    )
    parser.add_argument(
        "--state-backend",
        choices=("memory", "redis"),
        default=os.getenv("MAHJONG_PERIODIC_SIM_STATE_BACKEND", "memory"),
        help="Store disposable inboxes, reply gates, and events in memory or local Redis.",
    )
    parser.add_argument(
        "--redis-url",
        default=os.getenv("MAHJONG_PERIODIC_SIM_REDIS_URL", "redis://127.0.0.1:6379/0"),
        help="Redis URL used only when --state-backend=redis.",
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
    parser.add_argument(
        "--initial-dialogs",
        type=int,
        default=env_int("MAHJONG_PERIODIC_SIM_INITIAL_DIALOGS", 3),
        help="Number of dialogs seeded per scenario run; pressure runs still seed all speakers.",
    )
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
        state_backend=args.state_backend,
        redis_url=args.redis_url,
        env_file=args.env_file.expanduser().resolve() if args.env_file else None,
        min_interval_seconds=args.min_interval,
        max_interval_seconds=args.max_interval,
        min_messages=args.min_messages,
        max_messages=args.max_messages,
        initial_dialog_limit=args.initial_dialogs,
        max_duration_seconds=args.duration,
        max_workers=args.workers,
        rate_limit=args.rate,
        speed=args.speed,
        request_timeout_seconds=args.request_timeout,
        poll_seconds=max(0.1, args.poll),
    )


def daemon_cli_args(args: argparse.Namespace, *, command: str = "daemon") -> list[str]:
    """复用同一配置构造后台执行命令，单次运行不会隐式开启定时任务。"""
    values = [
        sys.executable,
        str(Path(__file__).resolve()),
        command,
        "--runtime-dir", str(args.runtime_dir),
        "--mode", args.mode,
        "--message-mode", args.message_mode or ("model" if args.mode == "real" else "rule"),
        "--state-backend", args.state_backend,
        "--redis-url", args.redis_url,
        "--env-file", str(args.env_file),
        "--min-interval", str(args.min_interval),
        "--max-interval", str(args.max_interval),
        "--min-messages", str(args.min_messages),
        "--max-messages", str(args.max_messages),
        "--initial-dialogs", str(args.initial_dialogs),
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


def launch_worker(args: argparse.Namespace, scheduler: PeriodicChatSimulationScheduler, *, single: bool) -> dict[str, Any]:
    """启动隔离子进程并确认其接管状态；启动失败不能报告为已提交成功。"""
    config = scheduler.config
    previous_run = scheduler.status().get("last_run_id")
    log_path = config.runtime_dir / ("once.log" if single else "daemon.log")
    with log_path.open("a", encoding="utf-8") as output:
        process = subprocess.Popen(daemon_cli_args(args, command="once" if single else "daemon"),
                                   cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        state = scheduler.status()
        code = process.poll()
        if state.get("pid") == process.pid and state.get("process_alive"):
            return {"ok": True, "accepted": True, "pid": process.pid,
                    "execution_mode": "once" if single else "daemon", "run_id": state.get("current_run_id")}
        if code is not None:
            if code == 0 and state.get("last_run_id") != previous_run:
                return {"ok": True, "accepted": True, "execution_mode": "once",
                        "run_id": state.get("last_run_id")}
            # 同时启动的另一个进程已拿到锁时，返回它的运行，不重复创建。
            if code == 2 and state.get("process_alive"):
                return {"ok": True, "accepted": True, "already_running": True,
                        "run_id": state.get("current_run_id")}
            return {"ok": False, "error": f"模拟进程启动失败（退出码 {code}），请查看 {log_path.name}"}
        time.sleep(0.05)
    return {"ok": False, "error": f"模拟进程未在 5 秒内确认启动，请查看 {log_path.name}"}


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
        status = scheduler.status()
        if status.get("process_alive"):
            if status.get("worker_mode") == "once" or status.get("status") in {"starting", "running"}:
                result = {"ok": True, "accepted": True, "already_running": True,
                          "run_id": status.get("current_run_id")}
            else:
                control = update_control(config, run_request_id=uuid.uuid4().hex)
                result = {"ok": True, "accepted": True, "execution_mode": "daemon",
                          "run_request_id": control["run_request_id"]}
        else:
            result = launch_worker(args, scheduler, single=True)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("ok") else 1
    if args.command == "start":
        status = scheduler.status()
        if status.get("process_alive"):
            print(json.dumps({"ok": True, "accepted": True, "started": False, "reason": "already_running", **status}, ensure_ascii=False, indent=2))
            return 0
        config.runtime_dir.mkdir(parents=True, exist_ok=True)
        update_control(config, enabled=True)
        result = launch_worker(args, scheduler, single=False)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("ok") else 1

    config.runtime_dir.mkdir(parents=True, exist_ok=True)
    with config.lock_path.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("periodic chat simulator is already running", file=sys.stderr)
            return 2

        if args.command == "once":
            report = scheduler.run_single()
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 1 if report.get("status") == "failed" else 0

        def request_stop(_signum: int, _frame: Any) -> None:
            """收到终止信号后通知调度器停止领取新的模拟任务。"""
            scheduler.stop()

        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        scheduler.run_forever(run_immediately=not args.wait_first)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
