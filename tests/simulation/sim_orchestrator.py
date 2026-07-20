"""Layer 3: priority-queue scheduler, concurrency, rate limits, and reports."""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, PriorityQueue
from typing import Callable

try:
    from .behavior_policy import BehaviorPolicy, SimulationAction
    from .sim_adapter import RequestOutcome, SimulationAdapter
    from .sim_factory import (
        DEFAULT_USER_COUNT,
        PERSONA_ACTIVE_GAMBLER,
        PERSONA_LURKER,
        PERSONA_TROUBLEMAKER,
        VirtualUser,
    )
except ImportError:  # pragma: no cover - direct script execution path
    from behavior_policy import BehaviorPolicy, SimulationAction  # type: ignore
    from sim_adapter import RequestOutcome, SimulationAdapter  # type: ignore
    from sim_factory import (  # type: ignore
        DEFAULT_USER_COUNT,
        PERSONA_ACTIVE_GAMBLER,
        PERSONA_LURKER,
        PERSONA_TROUBLEMAKER,
        VirtualUser,
    )


DEFAULT_MESSAGE_LIMIT = 500
DEFAULT_DURATION_SECONDS = 120.0
DEFAULT_RATE_LIMIT = 5
DEFAULT_WORKERS = 10
DEFAULT_SPEED = 1.0
DEFAULT_REPORT_PATH = Path(__file__).with_name("sim_report.json")


class RateLimiter:
    """Thread-safe global sliding-window limit of at most five calls/second."""

    HARD_MAX_CALLS_PER_SECOND = 5

    def __init__(
        self,
        max_calls: int = DEFAULT_RATE_LIMIT,
        period_seconds: float = 1.0,
        *,
        monotonic_fn: Callable[[], float] = time.monotonic,
        wait_fn: Callable[[float, threading.Event | None], bool] | None = None,
    ) -> None:
        if not 0 < max_calls <= self.HARD_MAX_CALLS_PER_SECOND:
            raise ValueError("max_calls must be between 1 and 5")
        if period_seconds <= 0:
            raise ValueError("period_seconds must be positive")
        self.max_calls = int(max_calls)
        self.period_seconds = float(period_seconds)
        self._monotonic = monotonic_fn
        self._wait_fn = wait_fn or self._default_wait
        self._events: deque[float] = deque()
        self._grant_history: list[float] = []
        self._lock = threading.Lock()

    @staticmethod
    def _default_wait(delay: float, stop_event: threading.Event | None) -> bool:
        if stop_event is None:
            time.sleep(delay)
            return False
        return stop_event.wait(delay)

    def acquire(
        self,
        *,
        deadline: float | None = None,
        stop_event: threading.Event | None = None,
    ) -> bool:
        while True:
            if stop_event is not None and stop_event.is_set():
                return False
            with self._lock:
                current = self._monotonic()
                cutoff = current - self.period_seconds
                while self._events and self._events[0] <= cutoff:
                    self._events.popleft()
                if len(self._events) < self.max_calls:
                    self._events.append(current)
                    self._grant_history.append(current)
                    return True
                delay = max(0.0, self.period_seconds - (current - self._events[0]))
            if deadline is not None and current + delay > deadline:
                return False
            if self._wait_fn(delay, stop_event):
                return False

    def grant_history(self) -> list[float]:
        with self._lock:
            return list(self._grant_history)


class SimulationOrchestrator:
    """Drive persona events in simulated-time order and execute HTTP concurrently."""

    def __init__(
        self,
        *,
        users: list[VirtualUser],
        behavior_policy: BehaviorPolicy,
        adapter: SimulationAdapter,
        seed: int = 42,
        max_messages: int = DEFAULT_MESSAGE_LIMIT,
        max_duration_seconds: float = DEFAULT_DURATION_SECONDS,
        max_workers: int = DEFAULT_WORKERS,
        rate_limit: int = DEFAULT_RATE_LIMIT,
        speed: float = DEFAULT_SPEED,
        report_path: Path = DEFAULT_REPORT_PATH,
    ) -> None:
        if len(users) != DEFAULT_USER_COUNT:
            raise ValueError("SimulationOrchestrator requires exactly 100 virtual users")
        if not 1 <= max_workers <= DEFAULT_WORKERS:
            raise ValueError("max_workers must be between 1 and 10")
        if speed <= 0:
            raise ValueError("speed must be positive")
        self.users = list(users)
        self.users_by_id = {user.customer_id: user for user in users}
        self.behavior_policy = behavior_policy
        self.adapter = adapter
        self.seed = seed
        self.max_messages = max(1, int(max_messages))
        self.max_duration_seconds = max(0.1, float(max_duration_seconds))
        self.max_workers = int(max_workers)
        self.speed = float(speed)
        self.report_path = report_path
        self.rate_limiter = RateLimiter(max_calls=rate_limit)
        self.stop_event = threading.Event()
        self._schedule: PriorityQueue[SimulationAction] = PriorityQueue()
        self._next_sequence = 1

    def run(self) -> dict[str, object]:
        started_at = datetime.now(timezone.utc)
        started_monotonic = time.monotonic()
        deadline = started_monotonic + self.max_duration_seconds
        outcomes: list[RequestOutcome] = []
        futures: dict[Future[RequestOutcome], SimulationAction] = {}
        submitted = 0
        stop_reason = "message_limit"
        consecutive_lock_errors = 0
        max_consecutive_lock_errors = 0
        self._seed_schedule()

        executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="sim-sender")
        try:
            while True:
                current = time.monotonic()
                if current >= deadline and not self.stop_event.is_set():
                    stop_reason = "duration_limit"
                    self.stop_event.set()

                while (
                    not self.stop_event.is_set()
                    and submitted < self.max_messages
                    and len(futures) < self.max_workers
                ):
                    try:
                        action = self._schedule.get_nowait()
                    except Empty:
                        break
                    due_at = started_monotonic + action.due_simulated_seconds / self.speed
                    if time.monotonic() < due_at:
                        self._schedule.put(action)
                        break
                    future = executor.submit(self._send_with_rate_limit, action, deadline)
                    futures[future] = action
                    submitted += 1
                    if submitted < self.max_messages:
                        self._schedule_following(action)

                if submitted >= self.max_messages and not futures:
                    stop_reason = "message_limit"
                    break
                if self.stop_event.is_set() and not futures:
                    break

                timeout = self._next_wait_timeout(started_monotonic, deadline, bool(futures))
                if futures:
                    done, _ = wait(set(futures), timeout=timeout, return_when=FIRST_COMPLETED)
                else:
                    self.stop_event.wait(timeout)
                    done = set()

                completed: list[RequestOutcome] = []
                for future in done:
                    action = futures.pop(future)
                    try:
                        outcome = future.result()
                    except Exception as exc:  # Keep the run observable instead of crashing the reporter.
                        outcome = RequestOutcome(
                            action=action,
                            sent=True,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    if outcome.sent:
                        completed.append(outcome)
                completed.sort(key=lambda item: item.sent_at or 0.0)
                for outcome in completed:
                    outcomes.append(outcome)
                    if outcome.sqlite_lock_error:
                        consecutive_lock_errors += 1
                        max_consecutive_lock_errors = max(
                            max_consecutive_lock_errors,
                            consecutive_lock_errors,
                        )
                    else:
                        consecutive_lock_errors = 0
                    if consecutive_lock_errors >= 5:
                        stop_reason = "sqlite_lock_failure"
                        self.stop_event.set()

            if stop_reason != "sqlite_lock_failure" and len(outcomes) < self.max_messages:
                stop_reason = "duration_limit"
        finally:
            self.stop_event.set()
            for future in futures:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)

        report = build_report(
            outcomes,
            users=self.users,
            inbox_sizes=self.adapter.inbox_sizes(),
            seed=self.seed,
            max_messages=self.max_messages,
            max_duration_seconds=self.max_duration_seconds,
            max_workers=self.max_workers,
            rate_limit=self.rate_limiter.max_calls,
            speed=self.speed,
            stop_reason=stop_reason,
            max_consecutive_lock_errors=max_consecutive_lock_errors,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
            elapsed_seconds=time.monotonic() - started_monotonic,
        )
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return report

    def _seed_schedule(self) -> None:
        for user in self.behavior_policy.speaking_users():
            action = self.behavior_policy.first_action(user, sequence=self._claim_sequence())
            if action is not None:
                self._schedule.put(action)

    def _schedule_following(self, previous: SimulationAction) -> None:
        user = self.users_by_id[previous.sender_id]
        action = self.behavior_policy.following_action(
            user,
            previous,
            sequence=self._claim_sequence(),
        )
        self._schedule.put(action)

    def _claim_sequence(self) -> int:
        sequence = self._next_sequence
        self._next_sequence += 1
        return sequence

    def _send_with_rate_limit(self, action: SimulationAction, deadline: float) -> RequestOutcome:
        if not self.rate_limiter.acquire(deadline=deadline, stop_event=self.stop_event):
            return RequestOutcome(action=action, sent=False, error="stopped_before_rate_limit_slot")
        if self.stop_event.is_set() or time.monotonic() >= deadline:
            return RequestOutcome(action=action, sent=False, error="stopped_before_http_send")
        return self.adapter.send(action, deadline=deadline)

    def _next_wait_timeout(self, started: float, deadline: float, has_futures: bool) -> float:
        remaining = max(0.01, deadline - time.monotonic())
        timeout = min(0.05 if has_futures else 0.2, remaining)
        try:
            action = self._schedule.get_nowait()
        except Empty:
            return timeout
        self._schedule.put(action)
        due_at = started + action.due_simulated_seconds / self.speed
        return max(0.001, min(timeout, max(0.0, due_at - time.monotonic())))


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * ratio + 0.999999)))
    return round(ordered[index], 2)


def build_report(
    outcomes: list[RequestOutcome],
    *,
    users: list[VirtualUser],
    inbox_sizes: dict[str, int],
    seed: int,
    max_messages: int,
    max_duration_seconds: float,
    max_workers: int,
    rate_limit: int,
    speed: float,
    stop_reason: str,
    max_consecutive_lock_errors: int,
    started_at: datetime,
    finished_at: datetime,
    elapsed_seconds: float,
) -> dict[str, object]:
    latencies = [item.latency_ms for item in outcomes]
    successful_http = [item for item in outcomes if 200 <= item.status_code < 300]
    tool_results = [
        tool
        for outcome in successful_http
        for tool in outcome.response.get("tool_results") or []
        if isinstance(tool, dict)
    ]
    successful_tools = [
        item
        for item in tool_results
        if item.get("called") is True and item.get("allowed") is True and not item.get("error")
    ]
    empty_replies = [
        item for item in successful_http if not str(item.response.get("final_reply") or "").strip()
    ]
    group_messages = sum(item.action.channel == "group" for item in outcomes)
    persona_counts = {
        PERSONA_LURKER: sum(user.persona == PERSONA_LURKER for user in users),
        PERSONA_ACTIVE_GAMBLER: sum(user.persona == PERSONA_ACTIVE_GAMBLER for user in users),
        PERSONA_TROUBLEMAKER: sum(user.persona == PERSONA_TROUBLEMAKER for user in users),
    }
    return {
        "status": "failed" if stop_reason == "sqlite_lock_failure" else "completed",
        "stop_reason": stop_reason,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "configuration": {
            "seed": seed,
            "virtual_user_count": len(users),
            "group_count": 1,
            "message_limit": max_messages,
            "duration_limit_seconds": max_duration_seconds,
            "max_workers": max_workers,
            "rate_limit_per_second": rate_limit,
            "speed": speed,
        },
        "persona_counts": persona_counts,
        "total_messages": len(outcomes),
        "group_messages": group_messages,
        "private_messages": len(outcomes) - group_messages,
        "successful_http_responses": len(successful_http),
        "failed_http_requests": len(outcomes) - len(successful_http),
        "agent_response_latency_ms": {
            "average": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
        },
        "tool_calls": {
            "total": len(tool_results),
            "successful": len(successful_tools),
            "failed": len(tool_results) - len(successful_tools),
            "success_rate": round(len(successful_tools) / len(tool_results), 4) if tool_results else 1.0,
        },
        "sqlite_lock_wait_count": sum(item.sqlite_lock_error for item in outcomes),
        "max_consecutive_sqlite_lock_errors": max_consecutive_lock_errors,
        "has_empty_final_reply": bool(empty_replies),
        "empty_final_reply_count": len(empty_replies),
        "inbox_delivery_count": sum(item.inbox_deliveries for item in outcomes),
        "users_with_inbox_messages": sum(size > 0 for size in inbox_sizes.values()),
        "inbox_sizes": inbox_sizes,
        "errors": [
            {
                "sequence": item.action.sequence,
                "status_code": item.status_code,
                "error": item.error,
                "response": item.response,
            }
            for item in outcomes
            if item.error or not (200 <= item.status_code < 300)
        ][:50],
    }
