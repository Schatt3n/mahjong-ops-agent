#!/usr/bin/env python3
"""运行可重复的并发一致性与真实模型压力评测。"""

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
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mahjong_agent_runtime import (  # noqa: E402
    AgentLLMConfig,
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
    """单个并发评测场景的通过状态、耗时、操作数与错误详情。"""

    name: str
    passed: bool
    elapsed_ms: int
    operation_count: int
    checks: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换为只包含可序列化字段的字典表示。"""
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
    """确定性模型替代品，同时证明调用在时间上重叠。"""

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
    """测量共享运行时客户端；独立语义判定器只复用配置，不计入这些调用指标。"""

    def __init__(self, delegate: OpenAICompatibleAgentClient) -> None:
        self.delegate = delegate
        self._lock = threading.Lock()
        self.call_count = 0
        self.active_calls = 0
        self.max_active_calls = 0
        self.failed_calls = 0
        self.latencies_ms: list[float] = []
        self.estimated_prompt_tokens = 0

    @property
    def config(self) -> AgentLLMConfig:
        """只读透传真实配置，供独立判定器复制；包装层不生成凭证或更改调用配置。"""

        return self.delegate.config

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
        """返回经本包装器调用的指标；独立判定器的请求不属于共享客户端观测范围。"""

        metrics = {
            "measurement_scope": "shared_runtime_client_only",
            "independent_semantic_judge_included": False,
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


def game_evidence(game: Any) -> dict[str, Any]:
    """返回人类可读测试报告所使用的紧凑持久化牌局状态。"""

    return {
        "game_id": game.game_id,
        "conversation_id": game.conversation_id,
        "status": game.status.value,
        "remaining_seats": game.remaining_seats(),
        "participants": [
            {
                "customer_id": participant.customer_id,
                "status": participant.status,
                "seat_count": participant.seat_count,
                "source": participant.source,
            }
            for participant in game.participants
        ],
    }


def build_deterministic_runtime(path: pathlib.Path, client: DelayedDeterministicClient) -> AgentRuntime:
    """构建deterministic运行时，只组合调用方允许进入当前作用域的数据。"""
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
            except Exception as exc:  # noqa: PERF203 - 需要保留每个并发任务的失败详情
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
    """创建setup牌局，并返回持久化后的权威结果。"""
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
    """制造多个重叠选项的竞态，这些选项都临时包含同一个客户。"""

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

    initial_games = [setup.require_game(game_id) for game_id in game_ids]
    initial_evidence = [game_evidence(game) for game in initial_games]

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
    persisted = SQLiteAgentStore(path)
    winner_evidence = game_evidence(winners[0]) if winners else None
    loser_evidence = [game_evidence(game) for game in losers]
    relevant_transitions = [
        transition.to_dict()
        for transition in persisted.transitions
        if transition.entity_id in game_ids
        or transition.entity_id == "shared_customer"
        or transition.trace_id.startswith("trace_shared_option_finish_")
    ]
    return EvalResult(
        name="shared_participant_first_ready_wins_race",
        passed=not errors and all(item["passed"] for item in checks),
        elapsed_ms=int((time.perf_counter() - started) * 1000),
        operation_count=operations,
        checks=checks,
        errors=errors,
        metrics={
            "latency_ms": latency_summary(latencies),
            "evidence": {
                "test_kind": "deterministic_sqlite_concurrency",
                "production_entrypoint": "SQLiteAgentStore.record_candidate_reply",
                "database_path": str(path),
                "fixture": {
                    "option_count": operations,
                    "shared_customer_id": "shared_customer",
                    "initial_games": initial_evidence,
                },
                "simulated_candidate_replies": [
                    {
                        "game_id": game_ids[index],
                        "customer_id": f"final_candidate_{index}",
                        "status": "confirmed",
                        "seat_count": 1,
                        "trace_id": f"trace_shared_option_finish_{index}",
                    }
                    for index in range(operations)
                ],
                "outcome": {
                    "winner": winner_evidence,
                    "losers": loser_evidence,
                    "active_shared_game_ids": active_shared_games,
                    "released_shared_participation_count": superseded_shared_count,
                },
                "state_transitions": relevant_transitions,
            },
        },
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


def load_active_revision_fixture() -> dict[str, Any]:
    """读取已独立保存的改目标输入与人工语义标准，禁止临时按实际回复生成答案。"""
    from run_inquiry_grounding_eval import _semantic_criteria_ids

    path = ROOT / "eval" / "golden" / "active_input_revision.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) != 1 or records[0].get("kind") != "active_input_revision_golden":
        raise ValueError("active input revision requires its single declared golden fixture")
    fixture = records[0]
    messages = fixture.get("messages")
    if (not isinstance(messages, list) or len(messages) != 2
            or any(not isinstance(item, dict) or not item.get("text") or not item.get("message_id") for item in messages)
            or messages[0]["message_id"] == messages[1]["message_id"]):
        raise ValueError("active revision fixture requires two ordered, uniquely identified inputs")
    _semantic_criteria_ids(fixture["expected"])
    checks = fixture["expected"].get("hard_checks")
    if not isinstance(checks, list) or not checks or len(set(checks)) != len(checks):
        raise ValueError("active revision fixture requires unique hard check identifiers")
    return fixture


def run_live_active_input_revision(client: Any, args: Any, work_dir: pathlib.Path) -> dict[str, Any]:
    """在旧模型调用阻塞时接收明确改向，用同一真实 Runtime 核对失效与重新规划。

    Event 只控制测试时序，不伪造模型动作或工具结果；释放后两次主调用仍使用真实
    配置。所有原始输入、模型决策、工具与状态独立保存，旧结果不能靠隐藏报告通过。
    """
    import run_real_owner_chat_live_eval as owner_eval
    from run_inquiry_grounding_eval import assess_reply_semantics
    from mahjong_agent_runtime import CustomerProfile

    fixture = load_active_revision_fixture()
    work_dir.mkdir(parents=True, exist_ok=False)
    store = SQLiteAgentStore(work_dir / "active_revision.sqlite3")
    trace = InMemoryTraceRecorder()
    report: dict[str, Any] = {"case_id": fixture["id"], "status": "failed", "checks": [], "fixture": fixture}
    entered, release = threading.Event(), threading.Event()

    class FirstCallGate:
        """仅暂停第一份旧上下文的调用入口，确保后续输入在旧结果回灌前持久到达。"""
        config = client.config

        def __init__(self):
            self.used = False
            self.lock = threading.Lock()

        def complete(self, messages, *, trace_id, timeout_seconds):
            """控制时序后透传真实调用；不修改请求、响应或最终模型结论。"""
            with self.lock:
                first = not self.used
                self.used = True
            if first:
                entered.set()
                if not release.wait(timeout=max(10, timeout_seconds)):
                    raise TimeoutError("supplementary input was not accepted before the old call resumed")
            return client.complete(messages, trace_id=trace_id, timeout_seconds=timeout_seconds)

    try:
        setup = fixture["fixture"]
        customer_id = fixture["customer_id"]
        store.configure_rooms(["revision_eval_room"])
        store.upsert_customer(CustomerProfile(customer_id=customer_id, display_name="改向测试客户",
            preferred_games=setup["preferred_games"], preferred_stakes=setup["preferred_stakes"],
            smoke_preference=setup["smoke_preference"]))
        target = dict(setup["existing_game"])
        target.update(start_time_kind="scheduled", start_time=owner_eval.future_planned_start_at(target["start_time"]), smoke_preference="any")
        store.create_game(conversation_id="fixture_existing_game", organizer_id="fixture_host", organizer_name="牌友",
            requirement=target, known_players=[{"customer_id": name, "display_name": "牌友", "seat_count": 1}
                for name in ("fixture_host", "fixture_peer_a", "fixture_peer_b")], trace_id="revision_fixture")
        baseline = [game.to_dict() for game in store.games.values()]
        scheduled_count = len(store.scheduled_tasks)
        runtime = owner_eval.build_runtime(FirstCallGate(), store, trace, args)
        messages = [UserMessage(conversation_id=fixture["conversation_id"], sender_id=customer_id,
            sender_name="改向测试客户", text=item["text"], message_id=item["message_id"]) for item in fixture["messages"]]
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(runtime.handle_user_message, messages[0], trace_id="trace_active_revision_old")
            if not entered.wait(timeout=max(10, args.timeout_seconds)):
                raise TimeoutError("old main model call never started")
            latest = pool.submit(runtime.handle_user_message, messages[1], trace_id="trace_active_revision_latest")
            try:
                deadline = time.monotonic() + max(10, args.timeout_seconds)
                while store.conversation_version(fixture["conversation_id"]) < 2 and time.monotonic() < deadline:
                    time.sleep(0.01)
                accepted_before_old_finished = store.conversation_version(fixture["conversation_id"]) >= 2 and not first.done()
            finally:
                release.set()
            old_result = first.result(timeout=max(60, args.timeout_seconds * args.max_steps))
            result = latest.result(timeout=max(60, args.timeout_seconds * args.max_steps))
        events = trace.get_trace(result.trace_id)
        scenario = SimpleNamespace(message=messages[1])
        evidence = owner_eval.reply_semantic_evidence(result, scenario, events, store)
        assessment = assess_reply_semantics(client, evidence=evidence, expectation=fixture["expected"],
            trace_id="trace_active_revision_semantics", timeout_seconds=args.timeout_seconds)
        old_business = [item for item in old_result.tool_results if item.called and item.allowed
                        and item.name not in {"customer_visible_text_generation", "customer_visible_content_review"}]
        queries = [item for item in result.tool_results if item.name == "search_current_games" and item.called and item.allowed and not item.error]
        expected = fixture["expected"]["latest_requirement"]
        query_matches_latest = any(all(str(owner_eval.eval_requirement_value(item.result.get("requirement", {}), key)) == str(value)
            for key, value in expected.items()) and item.result.get("requirement", {}).get("start_time_kind") != "asap_when_full" for item in queries)
        raw_inputs = [turn.content for turn in store.recent_turns(fixture["conversation_id"], limit=100) if turn.role.value == "user"]
        context_inputs = [item.get("content") for item in evidence.get("recent_conversation", []) if item.get("role") == "user"]
        checks = [
            check("new_input_persisted_before_old_model_finished", accepted_before_old_finished, True, accepted_before_old_finished),
            check("old_run_yielded_without_reply_or_business_tool", old_result.status == "superseded" and not old_result.final_reply and not old_business,
                  "superseded with no reply or business effects", {"status": old_result.status, "reply": old_result.final_reply, "tools": [item.name for item in old_business]}),
            check("both_raw_inputs_and_latest_context_preserved", raw_inputs == [item.text for item in messages] and context_inputs[-2:] == raw_inputs,
                  [item.text for item in messages], {"raw": raw_inputs, "context": context_inputs}),
            check("latest_query_uses_revised_goal", query_matches_latest, expected, [item.result.get("requirement") for item in queries]),
            check("latest_waits_for_customer_choice", result.status == "waiting_user", "waiting_user", result.status),
            check("no_old_creation_or_implicit_join_or_follow_up", [game.to_dict() for game in store.games.values()] == baseline
                  and not store.invite_drafts and not store.outbound_message_drafts and len(store.scheduled_tasks) == scheduled_count,
                  "fixture games unchanged and no new outbound or scheduled work", {"games": len(store.games), "invites": len(store.invite_drafts), "scheduled": len(store.scheduled_tasks)}),
            check("latest_reply_matches_human_meaning", assessment.get("passed") is True and not assessment.get("error"), True, assessment),
        ]
        if {item["name"] for item in checks} != set(fixture["expected"]["hard_checks"]):
            raise ValueError("active revision runner did not execute every declared hard check")
        recovery = owner_eval.assess_action_contract_recovery(events, runtime_status=result.status,
            final_reply=result.final_reply, business_checks_passed=all(item["passed"] for item in checks),
            semantic_passed=assessment.get("passed") is True and not assessment.get("error"))
        # 改向回放同样保留完整边界，不能只给旧58例计量而漏掉最新目标的合同错误。
        raw_trace_path = work_dir / "latest_raw_trace.json"
        raw_trace_path.write_text(json.dumps({"trace_id": result.trace_id, "scenario_id": fixture["id"],
            "iteration": 1, "events": [event.to_dict() for event in events]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        recovery_evidence = {"trace_id": result.trace_id, "scenario_id": fixture["id"], "iteration": 1,
            "runtime_status": result.status, "final_reply": result.final_reply, "checks": checks,
            "trace_steps": [event.step for event in events], "raw_trace_path": str(raw_trace_path), "recovery_metrics": recovery}
        report.update(status="passed" if all(item["passed"] for item in checks) and recovery["quality_status"] == "passed" else "failed", checks=checks,
            old_result=old_result.to_dict(), latest_result=result.to_dict(), semantic_assessment=assessment,
            recovery_evidence=recovery_evidence,
            old_decision_trace=owner_eval.decision_trace_snapshots(trace.get_trace(old_result.trace_id)),
            latest_decision_trace=owner_eval.decision_trace_snapshots(events), db_path=str(store.path))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        report["error"] = error.replace(client.config.api_key, "[redacted]") if client.config.api_key else error
    finally:
        release.set()
        store._connection.close()
    (work_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def load_customer_visible_review_contract_cases(dataset_path: pathlib.Path | None = None) -> list[dict[str, Any]]:
    """读取固定正反审查资产；标签仅供调用后的后端验收，不属于模型上下文。"""
    path = dataset_path or ROOT / "eval" / "golden" / "customer_visible_review_contract.jsonl"
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    required = {"case_id", "text", "expected_approved", "requirement", "source"}
    if len(cases) != 8 or any(not isinstance(case, dict) or set(case) != required for case in cases):
        raise ValueError("customer visible review contract requires its eight complete fixed cases")
    ids = [case["case_id"] for case in cases]
    if (any(not isinstance(case_id, str) or not case_id.strip() for case_id in ids) or len(set(ids)) != len(ids)
            or any(not isinstance(case["text"], str) or not case["text"].strip() or type(case["expected_approved"]) is not bool
                   or not isinstance(case["requirement"], dict) or not case["requirement"] or not isinstance(case["source"], dict) for case in cases)
            or sum(case["expected_approved"] for case in cases) != 4):
        raise ValueError("customer visible review cases require unique ids, text, typed labels and four positive/four negative cases")
    return cases


def load_follow_up_commitment_review_cases(dataset_path: pathlib.Path | None = None) -> list[dict[str, Any]]:
    """读取独立跟进责任资产，准备方式只允许固定枚举，期望判决不参与准备或模型输入。"""
    path = dataset_path or ROOT / "eval" / "golden" / "follow_up_commitment_review.jsonl"
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    fields = {"case_id", "text", "expected_approved", "fixture", "requirement", "source"}
    fixtures = {"open_game_empty_candidates", "open_game_follow_up", "open_game_cancelled_follow_up"}
    if len(cases) != 6 or any(not isinstance(case, dict) or set(case) != fields for case in cases):
        raise ValueError("follow up commitment review requires its six complete fixed cases")
    ids = [case["case_id"] for case in cases]
    if (any(not isinstance(value, str) or not value.strip() for value in ids) or len(set(ids)) != len(ids)
            or any(not isinstance(case["text"], str) or not case["text"].strip()
                   or type(case["expected_approved"]) is not bool or not isinstance(case["fixture"], str)
                   or case["fixture"] not in fixtures
                   or not isinstance(case["requirement"], dict) or not case["requirement"]
                   or not isinstance(case["source"], dict) for case in cases)
            or sum(case["expected_approved"] for case in cases) != 3
            or [sum(case["fixture"] == value for case in cases) for value in (
                "open_game_empty_candidates", "open_game_follow_up", "open_game_cancelled_follow_up")] != [3, 2, 1]):
        raise ValueError("follow up review requires unique ids, typed labels and fixed fixture coverage")
    return cases


def _review_contract_business_state(store: SQLiteAgentStore) -> dict[str, Any]:
    """读取审查前后业务状态；保留任务状态与有限载荷，避免把取消任务的行数当成有效安排。"""
    tasks = []
    for task in sorted(store.scheduled_tasks.values(), key=lambda item: item.task_id):
        raw = task.to_dict()
        summary = {key: raw[key] for key in (
            "task_id", "task_type", "status", "conversation_id", "subject_id", "aggregate_type", "aggregate_id", "due_at")}
        summary["payload"] = {key: raw["payload"][key] for key in (
            "goal", "waiting_for", "task_context_id", "source_message_id", "expires_at", "review_budget", "review_count", "closed_reason")
            if key in raw["payload"]}
        tasks.append(summary)
    return {"game_count": len(store.games), "invite_count": len(store.invite_drafts),
        "outbound_count": len(store.outbound_message_drafts), "scheduled_count": len(store.scheduled_tasks),
        "waiting_count": len(store.waiting_demands), "scheduled_tasks": tasks}


def run_live_customer_visible_review_contract(
    client: Any, args: Any, work_dir: pathlib.Path, include_follow_up_cases: bool = False,
) -> dict[str, Any]:
    """用生产审查入口逐例验证固定文本，事实来自隔离SQLite上的真实工具执行。

    这是审查模型的受控真实回归，不替代完整主Agent规划回放。每例只调用一次审查；
    无效输出、网络错误和错误批准均保留证据并失败，不据期望标签重试或伪造工具结果。
    完整集合追加跟进责任资产；旧八例仍可独立执行，完整运行器始终明确启用追加集合。
    """
    from mahjong_agent_runtime import AgentAction, CustomerProfile, ToolCall
    from mahjong_agent_runtime.budget import TokenBudget
    from mahjong_agent_runtime.customer_visible_review import external_action_evidence_from_tool_results

    cases = load_customer_visible_review_contract_cases()
    dataset_paths = ["eval/golden/customer_visible_review_contract.jsonl"]
    if include_follow_up_cases:
        cases += load_follow_up_commitment_review_cases()
        dataset_paths.append("eval/golden/follow_up_commitment_review.jsonl")
    if len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("review assets must have globally unique case ids")
    work_dir.mkdir(parents=True, exist_ok=False)
    reports: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases, start=1):
        case_id = case["case_id"]
        case_dir = work_dir / case_id
        case_dir.mkdir(exist_ok=False)
        store = SQLiteAgentStore(case_dir / "review.sqlite3")
        trace = InMemoryTraceRecorder()
        trace_id = f"trace_customer_visible_review_{case_index:02d}"
        fixture = case.get("fixture", "empty_inventory")
        report: dict[str, Any] = {"case_id": case_id, "source_id": case_id, "trace_id": trace_id,
            "status": "failed", "expected_approved": case["expected_approved"], "text": case["text"],
            "source": case["source"], "checks": [], "raw_approved": None, "final_approved": None,
            "raw_response": None, "review_result": None, "tool_result": None, "db_path": str(store.path),
            "fixture": fixture, "asset_path": dataset_paths[0] if fixture == "empty_inventory" else dataset_paths[-1],
            "prerequisite_results": [], "business_state": {"before": None, "after": None}}
        try:
            # 来源标识只留在后端报告，模型可见标识使用中性序号，避免ID暗示期望判决。
            message = UserMessage(conversation_id=f"review_conversation_{case_index:02d}", sender_id="review_customer",
                sender_name="客户", text=("下午有红中1到3块的局吗？我一个人，烟都行。" if fixture == "empty_inventory"
                    else "帮我组一局本店玩法，1块底256封顶，烟都行，人齐开，我一个人。"), message_id=f"review_message_{case_index:02d}")
            store.upsert_customer(CustomerProfile(customer_id=message.sender_id, display_name=message.sender_name,
                preferred_games=[case["requirement"]["game_type"]],
                preferred_stakes=["1-3块都行"] if fixture == "empty_inventory" else ["1"], smoke_preference="any"))
            runtime = AgentRuntime(llm_client=client, store=store, tool_gateway=ToolGateway(store), trace_recorder=trace,
                reply_self_review_enabled=True, reply_self_review_client=client, llm_timeout_seconds=args.timeout_seconds)
            if fixture != "empty_inventory":
                prepared = runtime.task_context_manager.prepare(message, trace_id=trace_id)
                trace.record(trace_id, "task_context_prepared", prepared.to_dict())
            store.append_user_turn(message, trace_id)
            prerequisites = []

            def execute(name: str, arguments: dict[str, Any]):
                """固定夹具只通过真实网关准备；每个失败结果先保存再阻断审查。"""
                step = len(prerequisites)
                call = ToolCall(name=name, arguments=arguments, reason="准备本例真实业务事实")
                trace.record(trace_id, "tool_called", {"call": call.to_dict(), "step_index": step, "call_index": 1})
                result = runtime.tool_gateway.execute(call, trace_id=trace_id, conversation_id=message.conversation_id,
                    sender_id=message.sender_id, sender_name=message.sender_name, step_index=step, source_message_id=message.message_id)
                trace.record(trace_id, "tool_result", result.to_dict())
                prerequisites.append(result)
                report["prerequisite_results"].append(result.to_dict())
                if not result.called or not result.allowed or result.error:
                    raise ValueError("controlled review prerequisite tool did not execute successfully")
                return result

            if fixture == "empty_inventory":
                query = execute("search_current_games", {"requirement": dict(case["requirement"]), "limit": 8})
            else:
                created = execute("create_game", {"organizer_id": message.sender_id, "organizer_name": message.sender_name,
                    "requirement": dict(case["requirement"]),
                    "known_players": [{"customer_id": message.sender_id, "display_name": message.sender_name, "seat_count": 1}]})
                query = execute("search_customers", {"game_id": created.result["game"]["game_id"],
                    "requirement": dict(case["requirement"]), "exclude_customer_ids": [message.sender_id], "limit": 8})
                if fixture in {"open_game_follow_up", "open_game_cancelled_follow_up"}:
                    stamp = dt.datetime.now().astimezone()
                    execute("schedule_follow_up", {"goal": "有限复查本次组局是否有新进展", "waiting_for": "players",
                        "due_at": (stamp + dt.timedelta(minutes=5)).isoformat(),
                        "expires_at": (stamp + dt.timedelta(minutes=30)).isoformat(), "review_budget": 1})
                if fixture == "open_game_cancelled_follow_up":
                    execute("cancel_follow_up", {"reason": "本次不再跟进"})
            report["tool_result"] = query.to_dict()
            built = runtime.context_builder.build(message, trace_id=trace_id,
                previous_tool_results=[prerequisites[-1]], turn_tool_evidence=prerequisites)
            trace.record(trace_id, "context_built", built.payload)
            action = AgentAction(goal="审查查询后的客户可见文本", objective_status="waiting_user", reasoning_summary="仅审查固定文本。",
                reply_to_user=case["text"])
            items = [{"item_id": "reply_to_user", "source": "reply_to_user", "recipient_id": message.sender_id,
                "recipient_name": message.sender_name, "text": case["text"], "source_text": case["text"],
                "action_evidence": external_action_evidence_from_tool_results(report["prerequisite_results"])}]
            report["business_state"]["before"] = _review_contract_business_state(store)
            review = runtime.action_processor.review_customer_visible_content(message=message, trace_id=trace_id, action=action,
                review_items=items, context_payload=built.payload, review_scope="reply_to_user",
                turn_budget=TokenBudget(max_tokens_per_call=args.max_tokens_per_call, max_calls_per_turn=1))
            if review is not None:
                trace.record(trace_id, "tool_result", review.to_dict())
                report["review_result"] = review.to_dict()
                report["raw_approved"] = review.result.get("raw_approved")
                report["final_approved"] = review.result.get("approved")
            events = trace.get_trace(trace_id)
            responses = [event.content.get("content") for event in events if event.step == "customer_visible_content_review_response"]
            report["raw_response"] = responses[0] if len(responses) == 1 else None
            report["business_state"]["after"] = _review_contract_business_state(store)
            state = {key: report["business_state"]["after"][key]
                for key in ("game_count", "invite_count", "outbound_count", "scheduled_count")}
            review_completed = bool(review is not None and review.called and review.allowed and not review.error
                and type(report["raw_approved"]) is bool and type(report["final_approved"]) is bool and len(responses) == 1)
            checks = [
                check("review_completed_once", review_completed, True, report["review_result"]),
                check("review_verdict_matches_expected", review_completed and report["raw_approved"] is case["expected_approved"]
                    and report["final_approved"] is case["expected_approved"],
                    case["expected_approved"], report["final_approved"]),
            ]
            if fixture == "empty_inventory":
                checks.insert(0, check("actual_empty_sqlite_read", not store.games and query.result.get("matches") == [] and query.result.get("alternatives") == [],
                    "actual successful read from empty game inventory", query.to_dict()))
                checks.append(check("no_business_side_effects", all(value == 0 for value in state.values())
                    and report["business_state"]["before"] == report["business_state"]["after"], "empty business state", state))
            else:
                names = [result.name for result in prerequisites]
                expected_names = ["create_game", "search_customers"]
                if fixture != "open_game_empty_candidates":
                    expected_names.append("schedule_follow_up")
                if fixture == "open_game_cancelled_follow_up":
                    expected_names.append("cancel_follow_up")
                active = [task for task in store.scheduled_tasks.values() if task.status.value in {"pending", "processing"}]
                fixture_matches = (len(active) == 1 and active[0].task_type == "conversation_follow_up"
                    if fixture == "open_game_follow_up" else not active)
                checks[:0] = [check("actual_open_game_empty_candidate_search", len(store.active_games()) == 1
                    and store.active_games()[0].recruitment_status.value == "open" and query.result.get("candidates") == [],
                    "actual open game with exhausted candidate search", query.to_dict()),
                    check("prerequisites_executed_in_declared_order", names == expected_names, expected_names, names),
                    check("follow_up_state_matches_fixture", fixture_matches, fixture, report["business_state"]["after"])]
                checks.append(check("no_review_business_side_effects", report["business_state"]["before"] == report["business_state"]["after"],
                    report["business_state"]["before"], report["business_state"]["after"]))
            report.update(status="passed" if all(item["passed"] for item in checks) else "failed", checks=checks, persisted_state=state)
        except Exception as exc:
            key = getattr(getattr(client, "config", None), "api_key", None)
            error = f"{type(exc).__name__}: {exc}"
            report["error"] = error.replace(key, "[redacted]") if isinstance(key, str) and key else error
        finally:
            events = trace.get_trace(trace_id)
            raw_trace_path = case_dir / "raw_trace.json"
            raw_trace_path.write_text(json.dumps({"trace_id": trace_id, "case_id": case_id, "source_id": case_id,
                "events": [event.to_dict() for event in events]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            report.update(raw_trace_path=str(raw_trace_path), trace_steps=[event.step for event in events])
            store._connection.close()
        reports.append(report)
    passed_count = sum(report["status"] == "passed" for report in reports)
    checks = [check("all_review_contract_sources_executed", [report["source_id"] for report in reports] == [case["case_id"] for case in cases],
        [case["case_id"] for case in cases], [report["source_id"] for report in reports]),
        check("all_review_contract_cases_passed", passed_count == len(cases), len(cases), passed_count)]
    payload = {"status": "passed" if all(item["passed"] for item in checks) else "failed",
        "execution_scope": ("controlled_customer_visible_review_with_real_sqlite_tool_evidence" if include_follow_up_cases
            else "controlled_customer_visible_review_with_real_sqlite_read_evidence"),
        "dataset_path": dataset_paths[0], "dataset_paths": dataset_paths, "source_ids": [case["case_id"] for case in cases],
        "case_count": len(cases), "passed_count": passed_count, "failed_count": len(cases)-passed_count,
        "checks": checks, "reports": reports}
    (work_dir / "report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


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
    """并发重放完整老板场景，保留原始语义案例标识并按轮次隔离持久化证据。

    独立回复判定器在每个工作线程的场景执行后同步调用，使用独立客户端，
    不共享运行时客户端的限流器或计量；报告分别声明判定尝试数和工作线程上界。
    """

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
    base_args = SimpleNamespace(
        db_path=None,
        max_tokens_per_call=max_tokens_per_call,
        max_calls_per_turn=max_calls_per_turn,
        max_steps=max_steps,
        timeout_seconds=timeout_seconds,
        skip_text_generation=skip_text_generation,
        skip_review=skip_review,
    )
    tasks = [
        (scenario, repeat_index + 1)
        for repeat_index in range(max(1, repeats))
        for scenario in owner_eval.live_eval_scenarios()
    ]
    started = time.perf_counter()

    def evaluate(scenario, iteration):
        """只为执行轮次生成隔离路径，不改决定黄金期望映射的 scenario_id。"""

        scenario_started = time.perf_counter()
        # 实时语义用例刻意复用同一会话标识，使断言贴近真实老板会话。
        # 但各用例不能共享持久化状态，否则无关夹具会被视为同一客户的并发消息，
        # 还可能让用例 A 的引用错误解析到用例 B 的历史。模型客户端仍然共享，
        # 因而这里依旧能够测量真实模型服务调用的重叠情况。
        scenario_args = SimpleNamespace(
            **{
                **vars(base_args),
                "db_path": live_dir / f"run_{iteration:02d}" / "owner_live.sqlite3",
            }
        )
        report = owner_eval.run_scenario(observed, scenario_args, scenario, iteration=iteration)
        report["execution_id"] = f"{scenario.scenario_id}_run_{iteration}"
        report["elapsed_ms"] = int((time.perf_counter() - scenario_started) * 1000)
        return report

    reports: list[dict[str, Any]] = []
    execution_errors: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        future_to_id = {
            pool.submit(evaluate, scenario, iteration): f"{scenario.scenario_id}_run_{iteration}"
            for scenario, iteration in tasks
        }
        for future in as_completed(future_to_id):
            try:
                reports.append(future.result())
            except Exception as exc:
                execution_errors.append(f"{future_to_id[future]}: {type(exc).__name__}: {exc}")
    reports.sort(key=lambda item: (item.get("scenario_id", ""), item.get("iteration", 0)))
    passed_count = sum(1 for report in reports if report.get("status") == "passed")
    expected_count = len(tasks)
    scenario_latencies = [float(report.get("elapsed_ms", 0)) for report in reports]
    provider = base_client.config.provider
    model = base_client.config.model
    active_revision = run_live_active_input_revision(observed, base_args, work_dir / "active_input_revision")
    review_contract = run_live_customer_visible_review_contract(observed, base_args, work_dir / "customer_visible_review_contract", True)
    metrics = observed.metrics()
    metrics["scenario_latency_ms"] = latency_summary(scenario_latencies)
    metrics["independent_semantic_judge"] = {
        "included_in_shared_runtime_metrics": False,
        "shares_runtime_provider_semaphore": False,
        "client_scope": "separate_client_per_scenario",
        "attempt_count": sum(len((report.get("semantic_assessment") or {}).get("attempts", [])) for report in reports),
        "max_concurrent_case_bound": min(len(tasks), max(1, workers)),
        "concurrency_bound_source": "synchronous_judging_in_scenario_worker_threads_not_measured_provider_peak",
    }
    checks = [
        check("all_scenarios_returned", len(reports) == expected_count and not execution_errors, expected_count, len(reports)),
        check("all_scenarios_passed", passed_count == expected_count, expected_count, passed_count),
        check("real_model_calls_overlapped", observed.max_active_calls >= min(2, workers), f">={min(2, workers)}", observed.max_active_calls),
        check("no_model_call_failed", observed.failed_calls == 0, 0, observed.failed_calls),
        check("active_goal_revision_passed", active_revision.get("status") == "passed", "passed", active_revision.get("status")),
        check("customer_visible_review_contract_passed", review_contract.get("status") == "passed", "passed", review_contract.get("status")),
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
        "recovery_summary": owner_eval.summarize_action_contract_recovery(reports),
        "active_input_revision": active_revision,
        "customer_visible_review_contract": review_contract,
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
    parser.add_argument("--max-tokens-per-call", type=int, default=int(os.getenv("MAHJONG_AGENT_MAX_TOKENS_PER_CALL", "64000")))
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
