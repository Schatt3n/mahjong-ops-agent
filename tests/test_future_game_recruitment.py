from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import json

import pytest

from mahjong_agent_runtime import (
    AgentRuntime,
    CustomerProfile,
    InMemoryAgentStore,
    InMemoryTraceRecorder,
    SQLiteAgentStore,
    ScheduledAgentTaskScheduler,
    StaticAgentClient,
    UserMessage,
)
from mahjong_agent_runtime.models import RecruitmentStatus, ScheduledTaskStatus, now


def _store(kind: str, tmp_path, name: str = "future_recruitment"):
    if kind == "memory":
        return InMemoryAgentStore()
    return SQLiteAgentStore(tmp_path / f"{name}.sqlite3")


def _future_requirement(start_at, *, stake: str = "0.5") -> dict:
    return {
        "game_type": "hangzhou_mahjong",
        "variant": "caiqiao",
        "stake": stake,
        "smoke_preference": "no_smoking",
        "start_time_kind": "scheduled",
        "planned_start_at": start_at.isoformat(),
        "duration_hours": 4,
        "known_player_count": 1,
        "needed_seats": 3,
    }


def test_iso_start_time_is_used_by_lifecycle_and_recruitment_policy():
    store = InMemoryAgentStore()
    start_at = now() + timedelta(hours=8)
    game = store.create_game(
        conversation_id="future_iso_start_time",
        organizer_id="customer_a",
        organizer_name="客户A",
        requirement={
            **_future_requirement(start_at),
            "start_time": start_at.isoformat(),
            "planned_start_at": None,
        },
        known_players=[],
        trace_id="trace_future_iso_start_time",
    )[0]

    assert game.planned_start_at == start_at
    assert game.planned_end_at == start_at + timedelta(hours=4)
    assert game.expires_at == start_at + timedelta(hours=4)
    assert game.recruitment_opens_at == start_at - timedelta(hours=2)
    assert game.recruitment_status == RecruitmentStatus.SCHEDULED
    assert store.scheduled_task_for_game(game.game_id) is not None


def test_create_game_normalizes_role_label_without_losing_requester_seats():
    store = InMemoryAgentStore()
    start_at = now() + timedelta(hours=8)
    game = store.create_game(
        conversation_id="future_requester_seats",
        organizer_id="customer_a",
        organizer_name="客户A",
        requirement=_future_requirement(start_at),
        known_players=[
            {
                "customer_id": "customer_a",
                "display_name": "客户A",
                "status": "organizer",
                "seat_count": 2,
                "known_member_ids": ["customer_a"],
                "anonymous_seat_count": 1,
            }
        ],
        trace_id="trace_future_requester_seats",
    )[0]

    assert game.participants[0].status == "joined"
    assert game.seat_summary()["claimed_seats"] == 2
    assert game.seat_summary()["remaining_seats"] == 2


def _create_future_game(store, start_at, *, conversation_id: str = "future_owner", stake: str = "0.5"):
    return store.create_game(
        conversation_id=conversation_id,
        organizer_id="customer_a",
        organizer_name="客户A",
        requirement=_future_requirement(start_at, stake=stake),
        known_players=[
            {
                "customer_id": "customer_a",
                "display_name": "客户A",
                "status": "confirmed",
                "seat_count": 1,
            }
        ],
        trace_id=f"trace_create_{conversation_id}",
    )[0]


def _action_json(*, status: str, tool_calls: list[dict] | None = None, reply: str = "") -> str:
    calls = tool_calls or []
    needs_tool = status == "needs_tool"
    return json.dumps(
        {
            "goal": "在预约局招募窗口开放后继续找人",
            "objective_status": status,
            "reasoning_summary": "根据预约局当前状态继续执行。",
            "objective_state": {
                "current_phase": "recruit_candidates" if needs_tool else "done",
                "known_facts": {},
                "missing_facts": [],
                "blockers": [],
            },
            "objective_plan": [
                {
                    "step_id": "recruit",
                    "title": "查询候选人并生成邀约",
                    "status": "in_progress" if needs_tool else "done",
                    "tool": calls[0]["name"] if calls else None,
                    "depends_on": [],
                    "decision_rule": "以最新工具结果为准。",
                }
            ],
            "plan_revision_reason": "预约局已进入招募窗口。",
            "reply_to_user": reply,
            "tool_calls": calls,
            "needs_human": False,
            "stop_reason": {
                "can_stop": not needs_tool,
                "why": "已完成内部调度" if not needs_tool else "还需要执行工具",
                "pending_work": [] if not needs_tool else [calls[0]["name"]],
                "depends_on_tool_results": False,
            },
            "badcase": None,
        },
        ensure_ascii=False,
    )


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
def test_future_game_is_visible_but_private_recruitment_waits_until_two_hours_before_start(
    kind: str,
    tmp_path,
    monkeypatch,
) -> None:
    store = _store(kind, tmp_path)
    start_at = now() + timedelta(hours=8)
    game = _create_future_game(store, start_at)

    assert game.recruitment_status == RecruitmentStatus.SCHEDULED
    assert game.recruitment_opens_at == start_at - timedelta(hours=2)
    assert store.active_games(game.conversation_id)[0].game_id == game.game_id
    task = store.scheduled_task_for_game(game.game_id)
    assert task is not None
    assert task.status == ScheduledTaskStatus.PENDING
    assert task.due_at == game.recruitment_opens_at

    with pytest.raises(ValueError, match="private candidate outreach is not open yet"):
        store.create_invite_drafts(
            game_id=game.game_id,
            invitations=[
                {
                    "customer_id": "candidate_a",
                    "display_name": "候选人A",
                    "message_text": "明天下午一点，0.5无烟，打吗？",
                }
            ],
            trace_id="trace_early_invite",
        )

    opened, transition = store.open_game_recruitment(
        game.game_id,
        trace_id="trace_open_window",
        at=game.recruitment_opens_at,
    )
    assert opened.recruitment_status == RecruitmentStatus.OPEN
    assert transition is not None
    monkeypatch.setattr(
        "mahjong_agent_runtime.domains.game_domain.now",
        lambda: game.recruitment_opens_at + timedelta(seconds=1),
    )
    drafts, _ = store.create_invite_drafts(
        game_id=game.game_id,
        invitations=[
            {
                "customer_id": "candidate_a",
                "display_name": "候选人A",
                "message_text": "明天下午一点，0.5无烟，打吗？",
            }
        ],
        trace_id="trace_window_invite",
    )
    assert len(drafts) == 1
    assert store.require_game(game.game_id).recruitment_status == RecruitmentStatus.ACTIVE


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
def test_same_customer_can_hold_non_overlapping_today_and_tomorrow_games(kind: str, tmp_path) -> None:
    store = _store(kind, tmp_path, "multiple_days")
    today_start = now() + timedelta(hours=1)
    tomorrow_start = now() + timedelta(days=1, hours=1)
    first = _create_future_game(store, today_start, conversation_id="same_customer")
    second = _create_future_game(store, tomorrow_start, conversation_id="same_customer", stake="1")

    assert [item.game_id for item in store.active_games("same_customer")] == [first.game_id, second.game_id]
    assert store.scheduled_task_for_game(first.game_id) is None
    assert store.scheduled_task_for_game(second.game_id) is not None


def test_sqlite_future_task_survives_restart_and_only_one_node_claims_it(tmp_path) -> None:
    path = tmp_path / "future_restart.sqlite3"
    initial_store = SQLiteAgentStore(path)
    start_at = now() + timedelta(hours=8)
    game = _create_future_game(initial_store, start_at)
    due_at = game.recruitment_opens_at
    assert due_at is not None

    restarted = SQLiteAgentStore(path)
    persisted = restarted.scheduled_task_for_game(game.game_id)
    assert persisted is not None
    assert persisted.due_at == due_at
    stores = [SQLiteAgentStore(path), SQLiteAgentStore(path)]

    def claim(index: int):
        return stores[index].claim_scheduled_task(persisted.task_id, at=due_at)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(claim, range(2)))

    assert sum(item is not None for item in results) == 1
    final_task = SQLiteAgentStore(path).scheduled_task_for_game(game.game_id)
    assert final_task is not None
    assert final_task.status == ScheduledTaskStatus.PROCESSING
    assert final_task.attempts == 1


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
def test_scheduler_executes_due_task_once_and_persists_completion(kind: str, tmp_path) -> None:
    store = _store(kind, tmp_path, "scheduler_once")
    start_at = now() + timedelta(hours=8)
    game = _create_future_game(store, start_at)
    task = store.scheduled_task_for_game(game.game_id)
    assert task is not None
    handled: list[str] = []
    trace = InMemoryTraceRecorder()
    scheduler = ScheduledAgentTaskScheduler(
        store=store,
        handler=lambda claimed, _trace_id: handled.append(claimed.task_id),
        trace_recorder=trace,
    )

    assert scheduler.run_due_once(at=task.due_at) == 1
    assert scheduler.run_due_once(at=task.due_at + timedelta(minutes=1)) == 0
    assert handled == [task.task_id]
    persisted = store.scheduled_task_for_game(game.game_id)
    assert persisted is not None
    assert persisted.status == ScheduledTaskStatus.COMPLETED
    assert persisted.attempts == 1


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
def test_scheduler_retries_failed_work_without_duplicate_parallel_execution(kind: str, tmp_path) -> None:
    store = _store(kind, tmp_path, "scheduler_retry")
    game = _create_future_game(store, now() + timedelta(hours=8))
    task = store.scheduled_task_for_game(game.game_id)
    assert task is not None
    attempts: list[str] = []

    def fail_once(claimed, _trace_id):
        attempts.append(claimed.task_id)
        if len(attempts) == 1:
            raise RuntimeError("temporary provider timeout")

    scheduler = ScheduledAgentTaskScheduler(
        store=store,
        handler=fail_once,
        trace_recorder=InMemoryTraceRecorder(),
        retry_delay_seconds=30,
        max_attempts=3,
    )

    assert scheduler.run_due_once(at=task.due_at) == 1
    retrying = store.scheduled_task_for_game(game.game_id)
    assert retrying is not None
    assert retrying.status == ScheduledTaskStatus.PENDING
    assert retrying.attempts == 1
    assert retrying.last_error == "RuntimeError: temporary provider timeout"
    assert scheduler.run_due_once(at=retrying.due_at) == 1
    completed = store.scheduled_task_for_game(game.game_id)
    assert completed is not None
    assert completed.status == ScheduledTaskStatus.COMPLETED
    assert completed.attempts == 2


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
def test_cancelling_future_game_cancels_pending_recruitment_task(kind: str, tmp_path) -> None:
    store = _store(kind, tmp_path, "cancel_future")
    game = _create_future_game(store, now() + timedelta(hours=8))

    store.update_game_status(
        game_id=game.game_id,
        status="cancelled",
        reason="requester_cancelled_future_reservation",
        trace_id="trace_cancel_future",
    )

    task = store.scheduled_task_for_game(game.game_id)
    assert task is not None
    assert task.status == ScheduledTaskStatus.CANCELLED


def test_due_internal_event_reenters_main_agent_and_creates_candidate_draft(monkeypatch) -> None:
    store = InMemoryAgentStore()
    store.upsert_customer(
        CustomerProfile(
            customer_id="candidate_a",
            display_name="候选人A",
            preferred_games=["hangzhou_mahjong"],
            preferred_stakes=["0.5"],
            smoke_preference="no_smoking",
        )
    )
    game = _create_future_game(store, now() + timedelta(hours=8), conversation_id="future_agent_loop")
    task = store.scheduled_task_for_game(game.game_id)
    assert task is not None
    event_at = task.due_at + timedelta(seconds=1)
    monkeypatch.setattr("mahjong_agent_runtime.domains.game_domain.now", lambda: event_at)
    opened, _ = store.open_game_recruitment(game.game_id, trace_id="trace_open", at=event_at)
    client = StaticAgentClient(
        outputs=[
            _action_json(
                status="needs_tool",
                tool_calls=[
                    {
                        "name": "search_customers",
                        "arguments": {
                            "requirement": opened.requirement,
                            "exclude_customer_ids": ["customer_a"],
                            "limit": 8,
                        },
                        "reason": "招募窗口已开放，查询当前合适候选人。",
                    }
                ],
            ),
            _action_json(
                status="needs_tool",
                tool_calls=[
                    {
                        "name": "create_invite_drafts",
                        "arguments": {
                            "game_id": game.game_id,
                            "invitations": [
                                {
                                    "customer_id": "candidate_a",
                                    "display_name": "候选人A",
                                    "message_text": "明天下午一点，0.5无烟，打吗？",
                                }
                            ],
                        },
                        "reason": "为匹配候选人生成待发送邀约。",
                    }
                ],
            ),
            _action_json(status="completed", reply="内部招募任务已执行。"),
        ]
    )
    runtime = AgentRuntime(
        llm_client=client,
        store=store,
        trace_recorder=InMemoryTraceRecorder(),
    )

    result = runtime.handle_system_event(
        UserMessage(
            conversation_id=game.conversation_id,
            sender_id=game.organizer_id,
            sender_name=game.organizer_name,
            text="预约局招募窗口已开放，请根据当前事实继续找人。",
            message_id=task.idempotency_key,
            metadata={
                "internal_event": True,
                "event_type": "game_recruitment_window_opened",
                "game_id": game.game_id,
            },
        ),
        trace_id="trace_future_agent_loop",
    )

    assert [call.name for action in result.actions for call in action.tool_calls] == [
        "search_customers",
        "create_invite_drafts",
    ]
    assert result.final_reply == "内部招募任务已执行。"
    assert len(store.invite_drafts) == 1
    draft = next(iter(store.invite_drafts.values()))
    assert draft.customer_id == "candidate_a"
    assert draft.message_text == "明天下午一点，0.5无烟，打吗？"
    assistant_turn = store.recent_turns(game.conversation_id, 1)[0]
    assert assistant_turn.metadata["internal_event"] is True
    assert assistant_turn.metadata["delivery_mode"] == "internal_only"
