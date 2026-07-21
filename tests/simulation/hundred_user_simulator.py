#!/usr/bin/env python3
"""Four-layer HTTP pressure simulation for 100 Mahjong customers.

The executable is deliberately a thin composition root:

1. ``sim_factory`` creates 100 customers and one group in ``test_sim.db``.
2. ``behavior_policy`` assigns 80 lurkers, 15 active gamblers, and 5 troublemakers.
3. ``sim_orchestrator`` schedules events, limits concurrency/rate, and reports.
4. ``sim_adapter`` sends WeChat-shaped HTTP payloads and records inbox replies.

``SIM_LLM_MODE`` is mandatory. ``mock`` never constructs a provider client;
only the explicit value ``real`` can load model credentials.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

try:
    from .behavior_policy import (
        BehaviorPolicy,
        QUESTION_POOL,
        SimulationAction,
        SimulationMessageGenerator,
    )
    from .sim_adapter import (
        StaticAgentLLMClient,
        SimulationAdapter,
        build_runtime,
        required_llm_mode,
        running_http_backend,
    )
    from .sim_factory import (
        DEFAULT_DB_PATH,
        DEFAULT_SEED,
        DEFAULT_USER_COUNT,
        VirtualUser,
        build_population,
        ensure_isolated_database,
    )
    from .sim_orchestrator import (
        DEFAULT_DURATION_SECONDS,
        DEFAULT_MESSAGE_LIMIT,
        DEFAULT_RATE_LIMIT,
        DEFAULT_REPORT_PATH,
        DEFAULT_SPEED,
        DEFAULT_WORKERS,
        RateLimiter,
        SimulationOrchestrator,
    )
except ImportError:  # pragma: no cover - direct script execution path
    from behavior_policy import (  # type: ignore
        BehaviorPolicy,
        QUESTION_POOL,
        SimulationAction,
        SimulationMessageGenerator,
    )
    from sim_adapter import (  # type: ignore
        StaticAgentLLMClient,
        SimulationAdapter,
        build_runtime,
        required_llm_mode,
        running_http_backend,
    )
    from sim_factory import (  # type: ignore
        DEFAULT_DB_PATH,
        DEFAULT_SEED,
        DEFAULT_USER_COUNT,
        VirtualUser,
        build_population,
        ensure_isolated_database,
    )
    from sim_orchestrator import (  # type: ignore
        DEFAULT_DURATION_SECONDS,
        DEFAULT_MESSAGE_LIMIT,
        DEFAULT_RATE_LIMIT,
        DEFAULT_REPORT_PATH,
        DEFAULT_SPEED,
        DEFAULT_WORKERS,
        RateLimiter,
        SimulationOrchestrator,
    )


class HundredUserSimulator:
    """Facade exposing the requested simulator while preserving four layers."""

    def __init__(
        self,
        *,
        users: list[VirtualUser],
        base_url: str,
        seed: int = DEFAULT_SEED,
        max_messages: int = DEFAULT_MESSAGE_LIMIT,
        max_duration_seconds: float = DEFAULT_DURATION_SECONDS,
        max_workers: int = DEFAULT_WORKERS,
        rate_limit: int = DEFAULT_RATE_LIMIT,
        speed: float = DEFAULT_SPEED,
        request_timeout_seconds: float = 30.0,
        report_path: Path = DEFAULT_REPORT_PATH,
        initial_dialog_limit: int | None = None,
        message_generator: SimulationMessageGenerator | None = None,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.users = list(users)
        self.behavior_policy = BehaviorPolicy(
            users,
            seed=seed,
            message_generator=message_generator,
        )
        self.adapter = SimulationAdapter(
            base_url=base_url,
            users=users,
            request_timeout_seconds=request_timeout_seconds,
        )
        self.orchestrator = SimulationOrchestrator(
            users=users,
            behavior_policy=self.behavior_policy,
            adapter=self.adapter,
            seed=seed,
            max_messages=max_messages,
            max_duration_seconds=max_duration_seconds,
            max_workers=max_workers,
            rate_limit=rate_limit,
            speed=speed,
            report_path=report_path,
            initial_dialog_limit=initial_dialog_limit,
            event_sink=event_sink,
        )
        self._preview_index = 0

    def get_next_action(self) -> SimulationAction:
        """Preview the next persona action using the same deterministic policy."""

        speakers = self.behavior_policy.speaking_users()
        user = speakers[self._preview_index % len(speakers)]
        cycle = self._preview_index // len(speakers) + 1
        self._preview_index += 1
        action = self.behavior_policy.get_next_action(
            user,
            sequence=self._preview_index,
            due_simulated_seconds=self.behavior_policy.interval_for(user) * cycle,
        )
        if action is None:
            raise RuntimeError("speaking persona unexpectedly produced no action")
        return action

    def run(self) -> dict[str, object]:
        return self.orchestrator.run()


def parse_speed(value: str) -> float:
    raw = str(value).strip().lower()
    if raw.endswith("x"):
        raw = raw[:-1]
    try:
        speed = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("speed must look like 10x or 10") from exc
    if speed <= 0:
        raise argparse.ArgumentTypeError("speed must be positive")
    return speed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--messages", type=int, default=DEFAULT_MESSAGE_LIMIT)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_SECONDS)
    parser.add_argument("--workers", type=int, choices=range(1, DEFAULT_WORKERS + 1), default=DEFAULT_WORKERS)
    parser.add_argument("--rate", type=int, choices=range(1, DEFAULT_RATE_LIMIT + 1), default=DEFAULT_RATE_LIMIT)
    parser.add_argument("--speed", type=parse_speed, default=DEFAULT_SPEED, help="simulated clock multiplier, e.g. 10x")
    parser.add_argument("--request-timeout", type=float, default=30.0)
    return parser.parse_args(argv)


def _write_enriched_report(path: Path, report: dict[str, object], **extra: Any) -> None:
    report.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    mode = required_llm_mode()
    db_path = ensure_isolated_database(args.db_path)
    store, users = build_population(db_path, seed=args.seed)
    runtime, llm_client = build_runtime(store, mode, seed=args.seed)
    with running_http_backend(runtime) as base_url:
        simulator = HundredUserSimulator(
            users=users,
            base_url=base_url,
            seed=args.seed,
            max_messages=args.messages,
            max_duration_seconds=args.duration,
            max_workers=args.workers,
            rate_limit=args.rate,
            speed=args.speed,
            request_timeout_seconds=args.request_timeout,
            report_path=args.report_path,
        )
        report = simulator.run()

    _write_enriched_report(
        args.report_path,
        report,
        llm_mode=mode,
        database_path=str(db_path),
        report_path=str(args.report_path.expanduser().resolve()),
        static_llm_call_count=(
            llm_client.call_count if isinstance(llm_client, StaticAgentLLMClient) else None
        ),
        question_pool_size=len(QUESTION_POOL),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
