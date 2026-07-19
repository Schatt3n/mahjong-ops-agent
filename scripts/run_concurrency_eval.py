#!/usr/bin/env python3
"""Run repeatable concurrency and live-model evaluations for Mahjong Ops Agent."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any, Callable


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mahjong_agent_runtime import (  # noqa: E402
    AgentRuntime,
    InMemoryTraceRecorder,
    OpenAICompatibleAgentClient,
    SQLiteAgentStore,
    ToolGateway,
    UserMessage,
)
from mahjong_agent_runtime.env import load_dotenv_defaults  # noqa: E402
from mahjong_agent_runtime.token_estimation import estimate_tokens  # noqa: E402


DEFAULT_WORK_DIR = ROOT / "runtime_data" / "concurrency_eval"
DEFAULT_REPORT_PATH = ROOT / "runtime_data" / "concurrency_eval_report.json"


@dataclass(slots=True)
class EvalResult:
    name: str
    passed: bool
    elapsed_ms: int
    operation_count: int
    checks: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": "passed" if self.passed else "failed",
            "elapsed_ms": self.elapsed_ms,
            "operation_count": self.operation_count,
            "checks": self.checks,
            "errors": self.errors,
            "metrics": self.metrics,
        }


class DelayedDeterministicClient:
    """Deterministic model substitute that also proves calls overlap in time."""

    def __init__(self, delay_seconds: float = 0.05) -> None:
        self.delay_seconds = max(0.0, delay_seconds)
        self._lock = threading.Lock()
        self.call_count = 0
        self.active_calls = 0
        self.max_active_calls = 0

    def complete(self, messages: list[dict[str, str]], *, trace_id: str, timeout_seconds: float) -> str:
        del messages, trace_id, timeout_seconds
        with self._lock:
            self.call_count += 1
            self.active_calls += 1
            self.max_active_calls = max(self.max_active_calls, self.active_calls)
        try:
            time.sleep(self.delay_seconds)
            return completed_action_json("好的。")
        finally:
            with self._lock:
                self.active_calls -= 1


class ObservedLLMClient:
    """Thread-safe metrics wrapper around the real OpenAI-compatible client."""

    def __init__(self, delegate: OpenAICompatibleAgentClient) -> None:
        self.delegate = delegate
        self._lock = threading.Lock()
        self.call_count = 0
        self.active_calls = 0
        self.max_active_calls = 0
        self.failed_calls = 0
        self.latencies_ms: list[float] = []
        self.estimated_prompt_tokens = 0

    def complete(self, messages: list[dict[str, str]], *, trace_id: str, timeout_seconds: float) -> str:
        prompt_tokens = sum(estimate_tokens(item.get("content", "")) for item in messages)
        with self._lock:
            self.call_count += 1
            self.active_calls += 1
            self.max_active_calls = max(self.max_active_calls, self.active_calls)
            self.estimated_prompt_tokens += prompt_tokens
        started = time.perf_counter()
        try:
            return self.delegate.complete(messages, trace_id=trace_id, timeout_seconds=timeout_seconds)
        except Exception:
            with self._lock:
                self.failed_calls += 1
            raise
        finally:
            elapsed = (time.perf_counter() - started) * 1000
            with self._lock:
                self.active_calls -= 1
                self.latencies_ms.append(elapsed)

    def metrics(self) -> dict[str, Any]:
        metrics = {
            "call_count": self.call_count,
            "failed_calls": self.failed_calls,
            "max_concurrent_client_calls": self.max_active_calls,
            "estimated_prompt_tokens": self.estimated_prompt_tokens,
            "model_call_latency_ms": latency_summary(self.latencies_ms),
        }
        provider_metrics = getattr(self.delegate, "concurrency_metrics", None)
        if callable(provider_metrics):
            metrics["provider_concurrency"] = provider_metrics()
        return metrics


def completed_action_json(reply: str) -> str:
    return json.dumps(
        {
            "goal": "完成本轮并发评测消息",
            "objective_status": "completed",
            "reasoning_summary": "本轮无需工具，可以直接回复。",
            "objective_state": {
                "current_phase": "reply",
                "known_facts": {},
                "missing_facts": [],
                "blockers": [],
            },
            "objective_plan": [
                {
                    "step_id": "reply",
                    "title": "回复用户",
                    "status": "done",
                    "tool": None,
                    "depends_on": [],
                    "decision_rule": "并发运行时评测固定回复。",
                }
            ],
            "plan_revision_reason": "无需调整。",
            "reply_to_user": reply,
            "tool_calls": [],
            "needs_human": False,
            "stop_reason": {
                "can_stop": True,
                "why": "本轮回复已生成。",
                "pending_work": [],
                "depends_on_tool_results": False,
            },
            "badcase": None,
        },
        ensure_ascii=False,
    )


def build_deterministic_runtime(path: pathlib.Path, client: DelayedDeterministicClient) -> AgentRuntime:
    store = SQLiteAgentStore(path)
    return AgentRuntime(
        llm_client=client,
        store=store,
        tool_gateway=ToolGateway(store),
        trace_recorder=InMemoryTraceRecorder(),
        customer_visible_text_generation_enabled=False,
        reply_self_review_enabled=False,
    )


def check(name: str, condition: bool, expected: Any, actual: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(condition), "expected": expected, "actual": actual}


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * ratio + 0.999999)))
    return round(ordered[index], 2)


def latency_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "average": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "min": round(min(values), 2),
        "average": round(sum(values) / len(values), 2),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": round(max(values), 2),
    }


def run_parallel(
    count: int,
    workers: int,
    operation: Callable[[int], Any],
) -> tuple[list[Any], list[str], list[float]]:
    outcomes: list[Any] = [None] * count
    errors: list[str] = []
    latencies: list[float] = [0.0] * count

    def measured(index: int) -> tuple[int, Any, float]:
        started = time.perf_counter()
        outcome = operation(index)
        return index, outcome, (time.perf_counter() - started) * 1000

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(measured, index) for index in range(count)]
        for future in as_completed(futures):
            try:
                index, outcome, elapsed = future.result()
                outcomes[index] = outcome
                latencies[index] = elapsed
            except Exception as exc:  # noqa: PERF203 - preserve all concurrent failures
                errors.append(f"{type(exc).__name__}: {exc}")
    return outcomes, errors, latencies


def scenario_duplicate_message(work_dir: pathlib.Path, operations: int, workers: int) -> EvalResult:
    path = work_dir / "duplicate_message.sqlite3"
    client = DelayedDeterministicClient()
    runtimes = [build_deterministic_runtime(path, client) for _ in range(max(2, min(workers, 8)))]
    message = UserMessage(
        conversation_id="concurrency_duplicate_message",
        sender_id="customer_duplicate",
        sender_name="",
        text="今晚有人吗",
        message_id="same-source-message-id",
    )
    started = time.perf_counter()
    outcomes, errors, latencies = run_parallel(
        operations,
        workers,
        lambda index: runtimes[index % len(runtimes)].handle_user_message(
            message,
            trace_id=f"trace_concurrency_duplicate_{index}",
        ),
    )
    store = SQLiteAgentStore(path)
    result_trace_ids = {item.trace_id for item in outcomes if item is not None}
    turns = store.recent_turns(message.conversation_id, 100)
    checks = [
        check("all_requests_returned", not errors and len([item for item in outcomes if item]) == operations, operations, len([item for item in outcomes if item])),
        check("model_called_once", client.call_count == 1, 1, client.call_count),
        check("one_persisted_result", len(result_trace_ids) == 1, 1, len(result_trace_ids)),
        check("conversation_version_once", store.conversation_version(message.conversation_id) == 1, 1, store.conversation_version(message.conversation_id)),
        check("one_user_and_one_assistant_turn", len(turns) == 2, 2, len(turns)),
    ]
    return EvalResult(
        name="duplicate_message_idempotency",
        passed=not errors and all(item["passed"] for item in checks),
        elapsed_ms=int((time.perf_counter() - started) * 1000),
        operation_count=operations,
        checks=checks,
        errors=errors,
        metrics={"latency_ms": latency_summary(latencies), "model_max_concurrency": client.max_active_calls},
    )


def scenario_parallel_conversations(work_dir: pathlib.Path, operations: int, workers: int) -> EvalResult:
    path = work_dir / "parallel_conversations.sqlite3"
    client = DelayedDeterministicClient(delay_seconds=0.08)
    runtime = build_deterministic_runtime(path, client)
    started = time.perf_counter()

    def invoke(index: int):
        conversation_id = f"parallel_conversation_{index}"
        return runtime.handle_user_message(
            UserMessage(
                conversation_id=conversation_id,
                sender_id=f"parallel_customer_{index}",
                sender_name="",
                text=f"并发消息 {index}",
                message_id=f"parallel_message_{index}",
            ),
            trace_id=f"trace_parallel_conversation_{index}",
        )

    outcomes, errors, latencies = run_parallel(operations, workers, invoke)
    store = SQLiteAgentStore(path)
    wrong_conversations = [
        index
        for index, item in enumerate(outcomes)
        if item is None or item.conversation_id != f"parallel_conversation_{index}"
    ]
    wrong_versions = [
        index
        for index in range(operations)
        if store.conversation_version(f"parallel_conversation_{index}") != 1
    ]
    checks = [
        check("all_conversations_completed", not errors and not wrong_conversations, [], wrong_conversations),
        check("conversation_versions_isolated", not wrong_versions, [], wrong_versions),
        check("model_calls_overlapped", client.max_active_calls >= min(2, workers), f">={min(2, workers)}", client.max_active_calls),
        check("one_model_call_per_conversation", client.call_count == operations, operations, client.call_count),
    ]
    return EvalResult(
        name="parallel_conversation_isolation",
        passed=not errors and all(item["passed"] for item in checks),
        elapsed_ms=int((time.perf_counter() - started) * 1000),
        operation_count=operations,
        checks=checks,
        errors=errors,
        metrics={"latency_ms": latency_summary(latencies), "model_max_concurrency": client.max_active_calls},
    )


def create_setup_game(path: pathlib.Path, *, seats: int = 1) -> str:
    store = SQLiteAgentStore(path)
    game, _ = store.create_game(
        conversation_id="shared_game_conversation",
        organizer_id="shared_organizer",
        organizer_name="",
        requirement={"game_type": "hangzhou_mahjong", "stake": "0.5", "known_player_count": seats},
        known_players=[
            {
                "customer_id": "shared_organizer",
                "display_name": "",
                "seat_count": seats,
            }
        ],
        trace_id="trace_concurrency_setup",
    )
    return game.game_id


def scenario_last_seat(work_dir: pathlib.Path, operations: int, workers: int) -> EvalResult:
    path = work_dir / "last_seat.sqlite3"
    game_id = create_setup_game(path, seats=3)
    stores = [SQLiteAgentStore(path) for _ in range(max(2, min(workers, 8)))]
    started = time.perf_counter()

    def confirm(index: int) -> str:
        try:
            stores[index % len(stores)].record_candidate_reply(
                game_id=game_id,
                customer_id=f"candidate_{index}",
                display_name="",
                status="confirmed",
                seat_count=1,
                trace_id=f"trace_last_seat_{index}",
            )
            return "confirmed"
        except ValueError as exc:
            if "seat capacity exceeded" not in str(exc):
                raise
            return "rejected"

    outcomes, errors, latencies = run_parallel(operations, workers, confirm)
    game = SQLiteAgentStore(path).require_game(game_id)
    checks = [
        check("exactly_one_confirmation", outcomes.count("confirmed") == 1, 1, outcomes.count("confirmed")),
        check("other_confirmations_rejected", outcomes.count("rejected") == operations - 1, operations - 1, outcomes.count("rejected")),
        check("table_never_overfilled", game.seat_summary()["claimed_seats"] == 4, 4, game.seat_summary()["claimed_seats"]),
        check("remaining_seats_zero", game.remaining_seats() == 0, 0, game.remaining_seats()),
        check("game_ready", game.status.value == "ready", "ready", game.status.value),
    ]
    return EvalResult(
        name="last_seat_race",
        passed=not errors and all(item["passed"] for item in checks),
        elapsed_ms=int((time.perf_counter() - started) * 1000),
        operation_count=operations,
        checks=checks,
        errors=errors,
        metrics={"latency_ms": latency_summary(latencies)},
    )


def scenario_shared_participant_first_ready_wins(
    work_dir: pathlib.Path,
    operations: int,
    workers: int,
) -> EvalResult:
    """Race many overlapping options that all provisionally contain one customer."""

    path = work_dir / "shared_participant_first_ready.sqlite3"
    setup = SQLiteAgentStore(path)
    start_at = (dt.datetime.now().astimezone() + dt.timedelta(days=1)).replace(
        hour=18,
        minute=0,
        second=0,
        microsecond=0,
    )
    game_ids: list[str] = []
    for index in range(operations):
        game, _ = setup.create_game(
            conversation_id=f"shared_option_conversation_{index}",
            organizer_id=f"organizer_{index}",
            organizer_name="",
            requirement={
                "game_type": "hangzhou_mahjong",
                "stake": "0.5",
                "smoke_preference": "no_smoking",
                "start_time_kind": "scheduled",
                "start_at": start_at.isoformat(),
                "duration_hours": 4,
                "known_player_count": 2,
                "needed_seats": 2,
            },
            known_players=[
                {"customer_id": f"organizer_{index}", "display_name": "", "seat_count": 1},
                {"customer_id": "shared_customer", "display_name": "", "seat_count": 1},
            ],
            trace_id=f"trace_shared_option_setup_{index}",
        )
        setup.record_candidate_reply(
            game_id=game.game_id,
            customer_id=f"prefilled_candidate_{index}",
            display_name="",
            status="confirmed",
            seat_count=1,
            trace_id=f"trace_shared_option_prefill_{index}",
        )
        game_ids.append(game.game_id)

    stores = [SQLiteAgentStore(path) for _ in range(max(2, min(workers, 8)))]
    started = time.perf_counter()

    def fill_last_seat(index: int) -> str:
        stores[index % len(stores)].record_candidate_reply(
            game_id=game_ids[index],
            customer_id=f"final_candidate_{index}",
            display_name="",
            status="confirmed",
            seat_count=1,
            trace_id=f"trace_shared_option_finish_{index}",
        )
        return game_ids[index]

    outcomes, errors, latencies = run_parallel(operations, workers, fill_last_seat)
    games = [SQLiteAgentStore(path).require_game(game_id) for game_id in game_ids]
    winners = [game for game in games if game.status.value == "ready"]
    losers = [game for game in games if game.status.value != "ready"]
    active_shared_games = [
        game.game_id
        for game in games
        if any(
            participant.customer_id == "shared_customer"
            and participant.status in {"joined", "confirmed"}
            for participant in game.participants
        )
    ]
    superseded_shared_count = sum(
        1
        for game in losers
        if any(
            participant.customer_id == "shared_customer" and participant.status == "superseded"
            for participant in game.participants
        )
    )
    checks = [
        check(
            "all_final_confirmations_returned",
            not errors and len([item for item in outcomes if item]) == operations,
            operations,
            len([item for item in outcomes if item]),
        ),
        check("exactly_one_overlapping_option_ready", len(winners) == 1, 1, len(winners)),
        check("shared_customer_committed_once", len(active_shared_games) == 1, 1, len(active_shared_games)),
        check(
            "shared_customer_kept_by_winner",
            bool(winners) and active_shared_games == [winners[0].game_id],
            [winners[0].game_id] if winners else [],
            active_shared_games,
        ),
        check(
            "losing_options_released_with_audit",
            superseded_shared_count == operations - 1,
            operations - 1,
            superseded_shared_count,
        ),
        check(
            "losing_options_recalculated",
            all(game.remaining_seats() == 1 for game in losers),
            True,
            [game.remaining_seats() for game in losers],
        ),
    ]
    return EvalResult(
        name="shared_participant_first_ready_wins_race",
        passed=not errors and all(item["passed"] for item in checks),
        elapsed_ms=int((time.perf_counter() - started) * 1000),
        operation_count=operations,
        checks=checks,
        errors=errors,
        metrics={"latency_ms": latency_summary(latencies)},
    )


def scenario_room_reservation(work_dir: pathlib.Path, operations: int, workers: int) -> EvalResult:
    path = work_dir / "room_reservation.sqlite3"
    setup = SQLiteAgentStore(path)
    setup.configure_rooms(["room_1"])
    stores = [SQLiteAgentStore(path) for _ in range(max(2, min(workers, 8)))]
    start_at = (dt.datetime.now().astimezone() + dt.timedelta(days=1)).replace(hour=18, minute=0, second=0, microsecond=0)
    end_at = start_at + dt.timedelta(hours=4)
    started = time.perf_counter()

    def reserve(index: int) -> str:
        try:
            stores[index % len(stores)].reserve_room(
                conversation_id=f"room_conversation_{index}",
                game_id=None,
                start_at=start_at,
                end_at=end_at,
                room_id="room_1",
                trace_id=f"trace_room_race_{index}",
            )
            return "reserved"
        except ValueError as exc:
            if "unavailable" not in str(exc) and "no room" not in str(exc):
                raise
            return "rejected"

    outcomes, errors, latencies = run_parallel(operations, workers, reserve)
    reservations = SQLiteAgentStore(path).room_reservations
    active = [item for item in reservations.values() if item.status in {"held", "confirmed"}]
    checks = [
        check("one_reservation_succeeds", outcomes.count("reserved") == 1, 1, outcomes.count("reserved")),
        check("one_active_reservation_persisted", len(active) == 1, 1, len(active)),
        check("others_rejected", outcomes.count("rejected") == operations - 1, operations - 1, outcomes.count("rejected")),
    ]
    return EvalResult(
        name="room_inventory_race",
        passed=not errors and all(item["passed"] for item in checks),
        elapsed_ms=int((time.perf_counter() - started) * 1000),
        operation_count=operations,
        checks=checks,
        errors=errors,
        metrics={"latency_ms": latency_summary(latencies)},
    )


def scenario_duplicate_game(work_dir: pathlib.Path, operations: int, workers: int) -> EvalResult:
    path = work_dir / "duplicate_game.sqlite3"
    stores = [SQLiteAgentStore(path) for _ in range(max(2, min(workers, 8)))]
    started = time.perf_counter()

    def create(index: int) -> str:
        try:
            stores[index % len(stores)].create_game(
                conversation_id="duplicate_game_conversation",
                organizer_id="duplicate_game_customer",
                organizer_name="",
                requirement={"game_type": "hangzhou_mahjong", "stake": "0.5"},
                known_players=[{"customer_id": "duplicate_game_customer", "display_name": ""}],
                trace_id=f"trace_duplicate_game_{index}",
            )
            return "created"
        except ValueError as exc:
            if "active game already exists" not in str(exc):
                raise
            return "rejected"

    outcomes, errors, latencies = run_parallel(operations, workers, create)
    games = SQLiteAgentStore(path).active_games("duplicate_game_conversation")
    checks = [
        check("one_game_created", outcomes.count("created") == 1, 1, outcomes.count("created")),
        check("one_active_game_persisted", len(games) == 1, 1, len(games)),
        check("duplicate_games_rejected", outcomes.count("rejected") == operations - 1, operations - 1, outcomes.count("rejected")),
    ]
    return EvalResult(
        name="duplicate_active_game_race",
        passed=not errors and all(item["passed"] for item in checks),
        elapsed_ms=int((time.perf_counter() - started) * 1000),
        operation_count=operations,
        checks=checks,
        errors=errors,
        metrics={"latency_ms": latency_summary(latencies)},
    )


def scenario_duplicate_invite(work_dir: pathlib.Path, operations: int, workers: int) -> EvalResult:
    path = work_dir / "duplicate_invite.sqlite3"
    game_id = create_setup_game(path)
    stores = [SQLiteAgentStore(path) for _ in range(max(2, min(workers, 8)))]
    started = time.perf_counter()

    def create(index: int) -> str:
        try:
            stores[index % len(stores)].create_invite_drafts(
                game_id=game_id,
                invitations=[
                    {
                        "customer_id": "same_candidate",
                        "display_name": "",
                        "message_text": "七点打吗？",
                    }
                ],
                trace_id=f"trace_duplicate_invite_{index}",
            )
            return "created"
        except ValueError as exc:
            if "already has an open invitation" not in str(exc):
                raise
            return "rejected"

    outcomes, errors, latencies = run_parallel(operations, workers, create)
    drafts = [item for item in SQLiteAgentStore(path).invite_drafts.values() if item.game_id == game_id]
    checks = [
        check("one_invite_created", outcomes.count("created") == 1, 1, outcomes.count("created")),
        check("one_open_invite_persisted", len(drafts) == 1, 1, len(drafts)),
        check("duplicate_invites_rejected", outcomes.count("rejected") == operations - 1, operations - 1, outcomes.count("rejected")),
    ]
    return EvalResult(
        name="duplicate_invite_race",
        passed=not errors and all(item["passed"] for item in checks),
        elapsed_ms=int((time.perf_counter() - started) * 1000),
        operation_count=operations,
        checks=checks,
        errors=errors,
        metrics={"latency_ms": latency_summary(latencies)},
    )


def scenario_atomic_versions(work_dir: pathlib.Path, operations: int, workers: int) -> EvalResult:
    path = work_dir / "atomic_versions.sqlite3"
    stores = [SQLiteAgentStore(path) for _ in range(max(2, min(workers, 8)))]
    started = time.perf_counter()

    def advance(index: int) -> int:
        version, _ = stores[index % len(stores)].advance_conversation_version(
            "same_version_conversation",
            trace_id=f"trace_version_race_{index}",
            reason="concurrency_eval",
        )
        return version

    outcomes, errors, latencies = run_parallel(operations, workers, advance)
    expected = list(range(1, operations + 1))
    actual = sorted(item for item in outcomes if isinstance(item, int))
    persisted = SQLiteAgentStore(path).conversation_version("same_version_conversation")
    checks = [
        check("versions_are_gapless", actual == expected, expected, actual),
        check("final_version_matches", persisted == operations, operations, persisted),
    ]
    return EvalResult(
        name="atomic_conversation_versions",
        passed=not errors and all(item["passed"] for item in checks),
        elapsed_ms=int((time.perf_counter() - started) * 1000),
        operation_count=operations,
        checks=checks,
        errors=errors,
        metrics={"latency_ms": latency_summary(latencies)},
    )


def run_deterministic_suite(work_dir: pathlib.Path, operations: int, workers: int) -> list[EvalResult]:
    deterministic_dir = work_dir / "deterministic"
    shutil.rmtree(deterministic_dir, ignore_errors=True)
    deterministic_dir.mkdir(parents=True, exist_ok=True)
    scenarios = [
        scenario_duplicate_message,
        scenario_parallel_conversations,
        scenario_last_seat,
        scenario_shared_participant_first_ready_wins,
        scenario_room_reservation,
        scenario_duplicate_game,
        scenario_duplicate_invite,
        scenario_atomic_versions,
    ]
    return [scenario(deterministic_dir, operations, workers) for scenario in scenarios]


def run_live_suite(
    work_dir: pathlib.Path,
    *,
    workers: int,
    repeats: int,
    timeout_seconds: float,
    max_steps: int,
    max_calls_per_turn: int,
    max_tokens_per_call: int,
    skip_review: bool,
    skip_text_generation: bool,
) -> dict[str, Any]:
    import run_real_owner_chat_live_eval as owner_eval

    base_client = OpenAICompatibleAgentClient.from_env()
    if base_client is None:
        return {
            "status": "skipped",
            "reason": "missing MAHJONG_LLM_API_KEY/DEEPSEEK_API_KEY or MAHJONG_LLM_MODEL",
        }
    observed = ObservedLLMClient(base_client)
    live_dir = work_dir / "live"
    shutil.rmtree(live_dir, ignore_errors=True)
    live_dir.mkdir(parents=True, exist_ok=True)
    args = SimpleNamespace(
        db_path=live_dir / "scenario.sqlite3",
        max_tokens_per_call=max_tokens_per_call,
        max_calls_per_turn=max_calls_per_turn,
        max_steps=max_steps,
        timeout_seconds=timeout_seconds,
        skip_text_generation=skip_text_generation,
        skip_review=skip_review,
    )
    tasks = [
        replace(scenario, scenario_id=f"{scenario.scenario_id}_run_{repeat_index + 1}")
        for repeat_index in range(max(1, repeats))
        for scenario in owner_eval.live_eval_scenarios()
    ]
    started = time.perf_counter()

    def evaluate(scenario):
        scenario_started = time.perf_counter()
        report = owner_eval.run_scenario(observed, args, scenario)
        report["elapsed_ms"] = int((time.perf_counter() - scenario_started) * 1000)
        return report

    reports: list[dict[str, Any]] = []
    execution_errors: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        future_to_id = {pool.submit(evaluate, scenario): scenario.scenario_id for scenario in tasks}
        for future in as_completed(future_to_id):
            try:
                reports.append(future.result())
            except Exception as exc:
                execution_errors.append(f"{future_to_id[future]}: {type(exc).__name__}: {exc}")
    reports.sort(key=lambda item: item.get("scenario_id", ""))
    passed_count = sum(1 for report in reports if report.get("status") == "passed")
    expected_count = len(tasks)
    scenario_latencies = [float(report.get("elapsed_ms", 0)) for report in reports]
    provider = base_client.config.provider
    model = base_client.config.model
    metrics = observed.metrics()
    metrics["scenario_latency_ms"] = latency_summary(scenario_latencies)
    checks = [
        check("all_scenarios_returned", len(reports) == expected_count and not execution_errors, expected_count, len(reports)),
        check("all_scenarios_passed", passed_count == expected_count, expected_count, passed_count),
        check("real_model_calls_overlapped", observed.max_active_calls >= min(2, workers), f">={min(2, workers)}", observed.max_active_calls),
        check("no_model_call_failed", observed.failed_calls == 0, 0, observed.failed_calls),
    ]
    return {
        "status": "passed" if not execution_errors and all(item["passed"] for item in checks) else "failed",
        "provider": provider,
        "model": model,
        "scenario_count": expected_count,
        "passed_count": passed_count,
        "failed_count": expected_count - passed_count,
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
        "checks": checks,
        "errors": execution_errors,
        "metrics": metrics,
        "reports": reports,
    }


def print_summary(payload: dict[str, Any]) -> None:
    print("\nConcurrency evaluation")
    print(f"status: {payload['status']}")
    for result in payload.get("deterministic") or []:
        print(
            f"- {result['name']}: {result['status']} "
            f"({result['operation_count']} operations, {result['elapsed_ms']} ms)"
        )
        for failed in [item for item in result.get("checks") or [] if not item.get("passed")]:
            print(f"  failed: {failed['name']} expected={failed['expected']} actual={failed['actual']}")
        for error in result.get("errors") or []:
            print(f"  error: {error}")
    live = payload.get("live")
    if live:
        print(
            f"- live_deepseek: {live.get('status')} "
            f"({live.get('passed_count', 0)}/{live.get('scenario_count', 0)} scenarios, "
            f"{live.get('elapsed_ms', 0)} ms)"
        )
        for failed in [item for item in live.get("checks") or [] if not item.get("passed")]:
            print(f"  failed: {failed['name']} expected={failed['expected']} actual={failed['actual']}")
        for report in live.get("reports") or []:
            print(
                f"  {report.get('scenario_id')}: {report.get('status')} "
                f"{report.get('elapsed_ms', 0)} ms -> {report.get('final_reply', '')}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run deterministic and real-DeepSeek concurrency evaluations.")
    parser.add_argument("--mode", choices=["all", "deterministic", "live"], default="all")
    parser.add_argument("--operations", type=int, default=40)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--live-workers", type=int, default=3)
    parser.add_argument("--live-repeats", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=float(os.getenv("MAHJONG_AGENT_LLM_TIMEOUT_SECONDS", "45")))
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-calls-per-turn", type=int, default=8)
    parser.add_argument("--max-tokens-per-call", type=int, default=int(os.getenv("MAHJONG_AGENT_MAX_TOKENS_PER_CALL", "24000")))
    parser.add_argument("--skip-review", action="store_true")
    parser.add_argument("--skip-text-generation", action="store_true")
    parser.add_argument("--work-dir", type=pathlib.Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--report-path", type=pathlib.Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--dotenv-path", type=pathlib.Path, default=ROOT / ".env")
    parser.add_argument("--no-dotenv", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    if not args.no_dotenv:
        load_dotenv_defaults(args.dotenv_path)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    deterministic_results: list[EvalResult] = []
    if args.mode in {"all", "deterministic"}:
        deterministic_results = run_deterministic_suite(
            args.work_dir,
            operations=max(2, args.operations),
            workers=max(1, args.workers),
        )
    live_result: dict[str, Any] | None = None
    if args.mode in {"all", "live"}:
        live_result = run_live_suite(
            args.work_dir,
            workers=max(1, args.live_workers),
            repeats=max(1, args.live_repeats),
            timeout_seconds=args.timeout_seconds,
            max_steps=max(1, args.max_steps),
            max_calls_per_turn=max(1, args.max_calls_per_turn),
            max_tokens_per_call=max(1, args.max_tokens_per_call),
            skip_review=args.skip_review,
            skip_text_generation=args.skip_text_generation,
        )
    deterministic_payload = [result.to_dict() for result in deterministic_results]
    deterministic_passed = all(result.passed for result in deterministic_results)
    live_passed = live_result is None or live_result.get("status") == "passed"
    payload = {
        "status": "passed" if deterministic_passed and live_passed else "failed",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "mode": args.mode,
        "configuration": {
            "operations": args.operations,
            "workers": args.workers,
            "live_workers": args.live_workers,
            "live_repeats": args.live_repeats,
        },
        "deterministic": deterministic_payload,
        "live": live_result,
    }
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print_summary(payload)
    print(f"report: {args.report_path}")
    return 1 if args.strict and payload["status"] != "passed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
