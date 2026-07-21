"""Layer 3: priority-queue scheduler, concurrency, rate limits, and reports."""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, PriorityQueue
from typing import Any, Callable

try:
    from .behavior_policy import BehaviorPolicy, SimulationAction, reply_requires_user
    from .sim_adapter import RequestOutcome, SimulationAdapter
    from .sim_factory import (
        DEFAULT_USER_COUNT,
        PERSONA_ACTIVE_GAMBLER,
        PERSONA_LURKER,
        PERSONA_TROUBLEMAKER,
        VirtualUser,
    )
except ImportError:  # pragma: no cover - direct script execution path
    from behavior_policy import BehaviorPolicy, SimulationAction, reply_requires_user  # type: ignore
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
MAX_DIALOG_TURNS = 5
LOCK_TIMEOUT = 10.0


@dataclass(slots=True)
class DialogState:
    """Conversation state owned by one virtual user within a simulation run."""

    turn_count: int = 0
    pending_response_to: str | None = None
    last_agent_reply: str = ""
    status: str = "active"
    last_conversation_id: str | None = None
    last_channel: str | None = None


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
        max_dialog_turns: int = MAX_DIALOG_TURNS,
        lock_timeout_seconds: float = LOCK_TIMEOUT,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        if len(users) != DEFAULT_USER_COUNT:
            raise ValueError("SimulationOrchestrator requires exactly 100 virtual users")
        if not 1 <= max_workers <= DEFAULT_WORKERS:
            raise ValueError("max_workers must be between 1 and 10")
        if speed <= 0:
            raise ValueError("speed must be positive")
        if max_dialog_turns < 1:
            raise ValueError("max_dialog_turns must be positive")
        if lock_timeout_seconds <= 0:
            raise ValueError("lock_timeout_seconds must be positive")
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
        self.max_dialog_turns = int(max_dialog_turns)
        self.lock_timeout_seconds = float(lock_timeout_seconds)
        self.event_sink = event_sink
        self.observer_errors: list[str] = []
        self.rate_limiter = RateLimiter(max_calls=rate_limit)
        self.stop_event = threading.Event()
        self._schedule: PriorityQueue[SimulationAction] = PriorityQueue()
        self._next_sequence = 1
        self.active_sessions: dict[str, DialogState] = {
            user.customer_id: DialogState() for user in users
        }
        self._inflight_conversations: set[str] = set()
        self._inflight_users: set[str] = set()
        self._queued_sequences_by_user: dict[str, set[int]] = {}
        self._cancelled_sequences: set[int] = set()
        self._timeout_actions_pending: set[str] = set()
        self._timeout_broken_user_ids: set[str] = set()
        self._timeout_broken_conversation_ids: set[str] = set()

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

                if not self.stop_event.is_set() and submitted < self.max_messages:
                    self._enqueue_expired_lock_actions(started_monotonic)

                while (
                    not self.stop_event.is_set()
                    and submitted < self.max_messages
                    and len(futures) < self.max_workers
                ):
                    action = self._take_dispatchable_action(started_monotonic)
                    if action is None:
                        break
                    future = executor.submit(self._send_with_rate_limit, action, deadline)
                    futures[future] = action
                    if action.event_type != "timeout_exit":
                        self._inflight_conversations.add(action.conversation_id)
                        self._inflight_users.add(action.sender_id)
                    state = self.active_sessions[action.sender_id]
                    state.pending_response_to = "agent"
                    state.last_conversation_id = action.conversation_id
                    state.last_channel = action.channel
                    submitted += 1

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
                    if action.event_type != "timeout_exit":
                        self._inflight_conversations.discard(action.conversation_id)
                        self._inflight_users.discard(action.sender_id)
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
                    self._emit_observation(outcome)
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
                    self._advance_dialog_after_outcome(
                        outcome,
                        allow_schedule=(
                            not self.stop_event.is_set() and submitted < self.max_messages
                        ),
                    )

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
            active_sessions=self.active_sessions,
            timeout_broken_user_ids=self._timeout_broken_user_ids,
            max_dialog_turns=self.max_dialog_turns,
            lock_timeout_seconds=self.lock_timeout_seconds,
            stop_reason=stop_reason,
            max_consecutive_lock_errors=max_consecutive_lock_errors,
            observer_errors=self.observer_errors,
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

    def _emit_observation(self, outcome: RequestOutcome) -> None:
        """Publish one completed chat turn without affecting simulation success."""

        if self.event_sink is None:
            return
        try:
            self.event_sink(outcome_to_transcript_entry(outcome))
        except Exception as exc:
            self.observer_errors.append(f"{type(exc).__name__}: {str(exc)[:200]}")

    def _seed_schedule(self) -> None:
        for user in self.behavior_policy.speaking_users():
            action = self.behavior_policy.first_action(
                user,
                sequence=self._claim_sequence(),
                dialog_state=self.active_sessions[user.customer_id],
            )
            if action is not None:
                self._enqueue_action(action)

    def _schedule_following(self, previous: SimulationAction) -> None:
        user = self.users_by_id[previous.sender_id]
        action = self.behavior_policy.following_action(
            user,
            previous,
            sequence=self._claim_sequence(),
            dialog_state=self.active_sessions[user.customer_id],
        )
        if action is not None:
            self._enqueue_action(action)

    def _advance_dialog_after_outcome(
        self,
        outcome: RequestOutcome,
        *,
        allow_schedule: bool,
    ) -> None:
        """Apply one HTTP result before selecting the user's next utterance."""

        action = outcome.action
        state = self.active_sessions[action.sender_id]
        state.turn_count += 1
        state.last_conversation_id = action.conversation_id
        state.last_channel = action.channel

        if action.event_type == "timeout_exit":
            state.pending_response_to = None
            state.status = "idle"
            self._cancel_queued_actions_for_user(action.sender_id)
            self._timeout_actions_pending.discard(action.conversation_id)
            self._release_speaker_lock(action.conversation_id)
            return

        if action.conversation_id in self._timeout_broken_conversation_ids:
            # A late network response must not revive a dialogue already closed
            # by the lock timeout safety valve.
            state.pending_response_to = None
            state.status = "idle"
            self._release_speaker_lock(action.conversation_id)
            return

        if not (200 <= outcome.status_code < 300):
            state.pending_response_to = None
            state.status = "idle"
            return

        reply = str(outcome.response.get("final_reply") or "")
        state.last_agent_reply = reply
        mentioned_user_id = outcome.next_speaker_only

        if mentioned_user_id and mentioned_user_id != action.sender_id:
            state.pending_response_to = None
            state.status = "idle"
            if allow_schedule:
                self._schedule_mentioned_user(mentioned_user_id, outcome)
            return

        if state.turn_count >= self.max_dialog_turns:
            state.pending_response_to = None
            state.status = "idle"
            self._release_speaker_lock(action.conversation_id, action.sender_id)
            return

        state.pending_response_to = "user" if reply_requires_user(reply) else None
        state.status = "active"
        if not allow_schedule:
            return
        before = len(self._queued_sequences_by_user.get(action.sender_id, set()))
        self._schedule_following(action)
        after = len(self._queued_sequences_by_user.get(action.sender_id, set()))
        if after == before:
            state.status = "idle"

    def _schedule_mentioned_user(
        self,
        mentioned_user_id: str,
        outcome: RequestOutcome,
    ) -> None:
        """Replace stale queued speech with a reply from the mentioned user."""

        target = self.users_by_id.get(mentioned_user_id)
        if target is None:
            return
        state = self.active_sessions[mentioned_user_id]
        if state.turn_count >= self.max_dialog_turns:
            state.status = "idle"
            self._release_speaker_lock(outcome.action.conversation_id, mentioned_user_id)
            return
        state.status = "active"
        state.pending_response_to = "user"
        state.last_agent_reply = str(outcome.response.get("final_reply") or "")
        state.last_conversation_id = outcome.action.conversation_id
        state.last_channel = outcome.action.channel
        self._cancel_queued_actions_for_user(mentioned_user_id)
        action = self.behavior_policy.following_action(
            target,
            outcome.action,
            sequence=self._claim_sequence(),
            dialog_state=state,
        )
        if action is not None:
            self._enqueue_action(action)

    def _enqueue_action(self, action: SimulationAction) -> None:
        self._schedule.put(action)
        self._queued_sequences_by_user.setdefault(action.sender_id, set()).add(action.sequence)

    def _cancel_queued_actions_for_user(self, user_id: str) -> None:
        sequences = self._queued_sequences_by_user.pop(user_id, set())
        self._cancelled_sequences.update(sequences)

    def _forget_queued_action(self, action: SimulationAction) -> None:
        sequences = self._queued_sequences_by_user.get(action.sender_id)
        if not sequences:
            return
        sequences.discard(action.sequence)
        if not sequences:
            self._queued_sequences_by_user.pop(action.sender_id, None)

    def _take_dispatchable_action(self, started_monotonic: float) -> SimulationAction | None:
        """Pick a due action without violating per-user or per-conversation order."""

        deferred: list[SimulationAction] = []
        selected: SimulationAction | None = None
        now = time.monotonic()
        scan_limit = self._schedule.qsize()
        for _ in range(scan_limit):
            try:
                action = self._schedule.get_nowait()
            except Empty:
                break
            if action.sequence in self._cancelled_sequences:
                self._cancelled_sequences.discard(action.sequence)
                self._forget_queued_action(action)
                continue
            due_at = started_monotonic + action.due_simulated_seconds / self.speed
            if now < due_at:
                deferred.append(action)
                break
            state = self.active_sessions[action.sender_id]
            if action.event_type != "timeout_exit" and state.status == "idle":
                self._forget_queued_action(action)
                continue
            next_speaker = self._next_speaker_only(action.conversation_id)
            blocked = action.event_type != "timeout_exit" and (
                action.conversation_id in self._inflight_conversations
                or action.sender_id in self._inflight_users
                or (next_speaker is not None and next_speaker != action.sender_id)
            )
            if blocked:
                deferred.append(action)
                continue
            selected = action
            self._forget_queued_action(action)
            break
        for action in deferred:
            self._schedule.put(action)
        return selected

    def _enqueue_expired_lock_actions(self, started_monotonic: float) -> None:
        expired_method = getattr(self.adapter, "expired_speaker_locks", None)
        if not callable(expired_method):
            return
        for conversation_id, user_id in expired_method(self.lock_timeout_seconds):
            if conversation_id in self._timeout_actions_pending:
                continue
            user = self.users_by_id.get(user_id)
            if user is None:
                self._release_speaker_lock(conversation_id, user_id)
                continue
            self._cancel_queued_actions_for_user(user_id)
            channel = "group" if ":group:" in conversation_id else "private"
            due_simulated_seconds = max(
                0.0,
                (time.monotonic() - started_monotonic) * self.speed,
            )
            self._enqueue_action(
                SimulationAction(
                    due_simulated_seconds=due_simulated_seconds,
                    sequence=self._claim_sequence(),
                    channel=channel,
                    conversation_id=conversation_id,
                    sender_id=user_id,
                    sender_name=user.display_name,
                    text="（沉默/退出）",
                    event_type="timeout_exit",
                )
            )
            self._timeout_actions_pending.add(conversation_id)
            self._timeout_broken_user_ids.add(user_id)
            self._timeout_broken_conversation_ids.add(conversation_id)

    def _next_speaker_only(self, conversation_id: str) -> str | None:
        method = getattr(self.adapter, "next_speaker_only", None)
        return method(conversation_id) if callable(method) else None

    def _release_speaker_lock(self, conversation_id: str, user_id: str | None = None) -> None:
        method = getattr(self.adapter, "release_speaker_lock", None)
        if callable(method):
            method(conversation_id, expected_user_id=user_id)

    def _claim_sequence(self) -> int:
        sequence = self._next_sequence
        self._next_sequence += 1
        return sequence

    def _send_with_rate_limit(self, action: SimulationAction, deadline: float) -> RequestOutcome:
        if not self.rate_limiter.acquire(deadline=deadline, stop_event=self.stop_event):
            return RequestOutcome(action=action, sent=False, error="stopped_before_rate_limit_slot")
        if self.stop_event.is_set() or time.monotonic() >= deadline:
            return RequestOutcome(action=action, sent=False, error="stopped_before_http_send")
        materialized = self.behavior_policy.materialize_action(
            action,
            user=self.users_by_id[action.sender_id],
            dialog_state=self.active_sessions[action.sender_id],
        )
        if self.stop_event.is_set() or time.monotonic() >= deadline:
            return RequestOutcome(
                action=materialized,
                sent=False,
                error="stopped_after_message_generation",
            )
        return self.adapter.send(materialized, deadline=deadline)

    def _next_wait_timeout(self, started: float, deadline: float, has_futures: bool) -> float:
        remaining = max(0.01, deadline - time.monotonic())
        timeout = min(0.05 if has_futures else 0.2, remaining)
        lock_wait_method = getattr(self.adapter, "seconds_until_lock_timeout", None)
        if callable(lock_wait_method):
            lock_wait = lock_wait_method(self.lock_timeout_seconds)
            if lock_wait is not None:
                timeout = min(timeout, max(0.001, lock_wait))
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
    active_sessions: dict[str, DialogState],
    timeout_broken_user_ids: set[str],
    max_dialog_turns: int,
    lock_timeout_seconds: float,
    stop_reason: str,
    max_consecutive_lock_errors: int,
    observer_errors: list[str],
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
    started_sessions = [state for state in active_sessions.values() if state.turn_count > 0]
    completed_multiturn_sessions = [state for state in started_sessions if state.turn_count >= 3]
    average_dialog_turns = (
        round(sum(state.turn_count for state in started_sessions) / len(started_sessions), 2)
        if started_sessions
        else 0.0
    )
    completion_rate = (
        round(len(completed_multiturn_sessions) / len(started_sessions), 4)
        if started_sessions
        else 0.0
    )
    # Workers finish out of order under concurrency. Persist the conversation in
    # production arrival order so a failed run can be replayed without guessing.
    transcript = [
        outcome_to_transcript_entry(item)
        for item in sorted(outcomes, key=lambda outcome: outcome.action.sequence)
    ]
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
            "max_dialog_turns": max_dialog_turns,
            "lock_timeout_seconds": lock_timeout_seconds,
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
        "observer_errors": list(observer_errors),
        "has_empty_final_reply": bool(empty_replies),
        "empty_final_reply_count": len(empty_replies),
        "inbox_delivery_count": sum(item.inbox_deliveries for item in outcomes),
        "users_with_inbox_messages": sum(size > 0 for size in inbox_sizes.values()),
        "inbox_sizes": inbox_sizes,
        "multi_turn_conversations": {
            "started_sessions": len(started_sessions),
            "sessions_with_at_least_3_turns": len(completed_multiturn_sessions),
            "completion_rate": completion_rate,
            "average_dialog_turns": average_dialog_turns,
            "timeout_broken_sessions": len(timeout_broken_user_ids),
        },
        "multi_turn_completion_rate": completion_rate,
        "sessions_with_at_least_3_turns": len(completed_multiturn_sessions),
        "average_dialog_turns": average_dialog_turns,
        "timeout_broken_sessions": len(timeout_broken_user_ids),
        "dialog_states": {
            user_id: {
                "turn_count": state.turn_count,
                "pending_response_to": state.pending_response_to,
                "status": state.status,
                "last_conversation_id": state.last_conversation_id,
            }
            for user_id, state in active_sessions.items()
            if state.turn_count > 0 or state.last_agent_reply
        },
        "transcript": transcript,
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


def outcome_to_transcript_entry(item: RequestOutcome) -> dict[str, Any]:
    """Normalize one HTTP outcome for both live streaming and final reports."""

    return {
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "sequence": item.action.sequence,
        "conversation_id": item.action.conversation_id,
        "channel": item.action.channel,
        "event_type": item.action.event_type,
        "user": {
            "customer_id": item.action.sender_id,
            "display_name": item.action.sender_name,
            "text": item.action.text,
            "generation": {
                "source": item.action.generation_source,
                "model": item.action.generator_model,
                "trace_id": item.action.generation_trace_id,
                "latency_ms": item.action.generation_latency_ms,
                "error": item.action.generation_error,
            },
        },
        "agent": {
            "reply": str(item.response.get("final_reply") or ""),
            "trace_id": str(item.response.get("trace_id") or ""),
            "objective_status": next(
                (
                    str(action.get("objective_status") or "")
                    for action in reversed(item.response.get("actions") or [])
                    if isinstance(action, dict)
                ),
                "",
            ),
        },
        "tool_calls": [
            str(tool.get("name") or "")
            for tool in item.response.get("tool_results") or []
            if isinstance(tool, dict)
        ],
        "latency_ms": round(item.latency_ms, 2),
        "status_code": item.status_code,
        "error": item.error,
    }
