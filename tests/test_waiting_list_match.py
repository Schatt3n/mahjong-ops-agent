from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import json
import threading
from zoneinfo import ZoneInfo

import pytest

from mahjong_agent_runtime import (
    AgentAction,
    AgentRuntime,
    InMemoryAgentStore,
    InMemoryTraceRecorder,
    ScheduledAgentTaskScheduler,
    SQLiteAgentStore,
    ToolCall,
    UserMessage,
    WaitingDemand,
    handle_waiting_expiration_task,
)
from mahjong_agent_runtime.matching import MatchTrigger, waiting_demand_mismatch_reason
from mahjong_agent_runtime.models import WaitingDemandStatus, now


def _store(kind: str, tmp_path, name: str):
    if kind == "memory":
        return InMemoryAgentStore()
    return SQLiteAgentStore(tmp_path / f"{name}.sqlite3")


def _terminal_action(reply: str) -> str:
    return json.dumps(
        {
            "goal": "把匹配到的新局告知等待中的客户并征求确认",
            "objective_status": "waiting_user",
            "reasoning_summary": "只告知公开局信息，不替客户加入。",
            "objective_state": {
                "current_phase": "wait_user",
                "known_facts": {},
                "missing_facts": [],
                "blockers": [],
            },
            "objective_plan": [
                {
                    "step_id": "notify",
                    "title": "通知客户并等待确认",
                    "status": "done",
                    "depends_on": [],
                }
            ],
            "plan_revision_reason": "系统发现了与等待需求匹配的新局。",
            "reply_to_user": reply,
            "tool_calls": [],
            "needs_human": False,
            "stop_reason": {
                "can_stop": True,
                "why": "已经征求客户确认，等待客户回复。",
                "pending_work": [],
                "depends_on_tool_results": False,
            },
            "badcase": None,
        },
        ensure_ascii=False,
    )


class TriggerAwareClient:
    """Deterministic model double that only accepts a system match trigger."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def complete(self, messages, *, trace_id: str, timeout_seconds: float) -> str:
        payload = json.loads(messages[-1]["content"])
        self.calls.append(payload)
        trigger = payload.get("system_trigger")
        assert trigger is not None
        game = trigger["game"]
        reply = (
            f"有个{game['time_label']}{game['stake']}"
            f"{game['smoke_label']}的{game['shortage_label']}，你要打吗？"
        )
        return _terminal_action(reply)


class MatchThenJoinClient:
    """Drive notification first, then a normal customer-confirmation tool turn."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def complete(self, messages, *, trace_id: str, timeout_seconds: float) -> str:
        payload = json.loads(messages[-1]["content"])
        self.calls.append(payload)
        trigger = payload.get("system_trigger")
        if trigger is not None:
            game = trigger["game"]
            return _terminal_action(
                f"有个{game['time_label']}{game['stake']}{game['smoke_label']}的"
                f"{game['shortage_label']}，你要打吗？"
            )
        previous = list(payload.get("previous_tool_results") or [])
        if previous:
            assert previous[-1]["name"] == "join_game"
            assert previous[-1]["allowed"] is True
            return _terminal_action("好")
        games = list(payload.get("active_games") or [])
        assert len(games) == 1
        current = payload["current_message"]
        return json.dumps(
            {
                "goal": "记录客户对等待匹配局的明确参加确认",
                "objective_status": "needs_tool",
                "reasoning_summary": "客户明确回复打，需要先写入参与状态。",
                "objective_state": {
                    "current_phase": "record_feedback",
                    "known_facts": {"active_game_id": games[0]["game_id"]},
                    "missing_facts": [],
                    "blockers": [],
                },
                "objective_plan": [
                    {
                        "step_id": "join",
                        "title": "记录客户加入",
                        "status": "in_progress",
                        "tool": "join_game",
                        "depends_on": [],
                    }
                ],
                "plan_revision_reason": "客户已对系统通知的关联局明确确认。",
                "reply_to_user": "",
                "tool_calls": [
                    {
                        "name": "join_game",
                        "arguments": {
                            "game_id": games[0]["game_id"],
                            "customer_id": current["sender_id"],
                            "display_name": current["sender_name"],
                            "seat_count": 1,
                        },
                        "reason": "客户明确确认参加关联局",
                    }
                ],
                "needs_human": False,
                "stop_reason": {
                    "can_stop": False,
                    "why": "必须先记录参加状态",
                    "pending_work": ["join_game"],
                    "depends_on_tool_results": False,
                },
                "badcase": None,
            },
            ensure_ascii=False,
        )


def _register_waiting_demand(
    runtime: AgentRuntime,
    *,
    conversation_id: str = "conversation-a",
    sender_id: str = "customer-a",
    stake: str = "0.5",
    smoke: str = "无烟",
    time_preference: str = "不限",
    extra_constraints: list[str] | None = None,
    expires_at=None,
):
    arguments = {
        "stake": stake,
        "smoke_preference": smoke,
        "time_preference": time_preference,
        "extra_constraints": list(extra_constraints or []),
    }
    if expires_at is not None:
        arguments["expires_at"] = expires_at.isoformat()
    result = runtime.tool_gateway.execute(
        ToolCall(name="register_waiting_demand", arguments=arguments, reason="当前无匹配局，客户愿意等待"),
        trace_id="trace-register-demand",
        conversation_id=conversation_id,
        sender_id=sender_id,
        sender_name="客户A",
        step_index=1,
        source_message_id=f"register-{sender_id}-{stake}-{smoke}-{expires_at}",
    )
    assert result.called and result.allowed, result.error
    return result


def _create_game_through_runtime(
    runtime: AgentRuntime,
    *,
    conversation_id: str,
    organizer_id: str,
    stake: str = "0.5",
    smoke: str = "no_smoking",
    known_players: list[dict] | None = None,
):
    start_at = now() + timedelta(hours=1)
    call = ToolCall(
        name="create_game",
        reason="客户明确要求创建新局",
        arguments={
            "organizer_id": organizer_id,
            "organizer_name": organizer_id,
            "requirement": {
                "game_type": "hangzhou_mahjong",
                "stake": stake,
                "smoke_preference": smoke,
                "start_time_kind": "scheduled",
                "planned_start_at": start_at.isoformat(),
                "known_player_count": 3,
                "needed_seats": 1,
            },
            "known_players": list(known_players or []),
        },
    )
    action = AgentAction(
        goal="创建新局",
        objective_status="needs_tool",
        reasoning_summary="条件已明确。",
        tool_calls=[call],
        stop_reason={
            "can_stop": False,
            "why": "需要创建局",
            "pending_work": ["create_game"],
            "depends_on_tool_results": False,
        },
    )
    message = UserMessage(
        conversation_id=conversation_id,
        sender_id=organizer_id,
        sender_name=organizer_id,
        text="三缺一，今晚七点，0.5无烟",
        message_id=f"create-{conversation_id}",
    )
    return runtime.tool_execution_service.execute_tool_calls(
        action,
        message=message,
        trace_id=f"trace-create-{conversation_id}",
        previous_step_tool_results=[],
        step_index=1,
        run_id=f"run-{conversation_id}",
        run_version=runtime.store.conversation_version(conversation_id),
        context_payload={},
    )


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
def test_waiting_demand_crud_and_expiration(kind: str, tmp_path) -> None:
    store = _store(kind, tmp_path, "waiting-crud")
    demand = WaitingDemand(
        demand_id="demand-crud",
        conversation_id="conversation-a",
        sender_id="customer-a",
        sender_name="客户A",
        demand={
            "stake": "0.5",
            "smoke_preference": "no_smoking",
            "time_preference": "今晚",
            "extra_constraints": [],
        },
        expires_at=now() + timedelta(hours=2),
    )

    assert store.insert_waiting_demand(demand) == demand.demand_id
    assert [item.demand_id for item in store.list_active_demands()] == [demand.demand_id]

    store.update_demand_status(demand.demand_id, WaitingDemandStatus.CANCELLED)
    assert store.list_active_demands() == []

    stale = WaitingDemand(
        demand_id="demand-stale",
        conversation_id="conversation-a",
        sender_id="customer-a",
        sender_name="客户A",
        demand={"stake": "0.5"},
        expires_at=now() - timedelta(seconds=1),
    )
    store.insert_waiting_demand(stale)
    expired = store.expire_stale_demands(at=now(), trace_id="trace-expire")
    assert [item.demand_id for item in expired] == [stale.demand_id]
    assert store.waiting_demand(stale.demand_id).status == WaitingDemandStatus.EXPIRED

    matched_stale = WaitingDemand(
        demand_id="demand-matched-stale",
        conversation_id="conversation-a",
        sender_id="customer-a",
        sender_name="客户A",
        demand={"stake": "0.5"},
        status=WaitingDemandStatus.MATCHED,
        expires_at=now() - timedelta(seconds=1),
        matched_game_id="game-old",
    )
    store.insert_waiting_demand(matched_stale)
    expired = store.expire_stale_demands(at=now(), trace_id="trace-expire-matched")
    assert [item.demand_id for item in expired] == [matched_stale.demand_id]
    assert store.waiting_demand(matched_stale.demand_id).status == WaitingDemandStatus.EXPIRED


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
def test_normal_match_reenters_agent_and_creates_one_outbound_notification(kind: str, tmp_path) -> None:
    client = TriggerAwareClient()
    store = _store(kind, tmp_path, "waiting-normal")
    runtime = AgentRuntime(llm_client=client, store=store)
    _register_waiting_demand(runtime)

    execution = _create_game_through_runtime(
        runtime,
        conversation_id="conversation-b",
        organizer_id="customer-b",
        known_players=[
            {"customer_id": "customer-b", "display_name": "客户B", "status": "confirmed"},
            {"customer_id": "customer-c", "display_name": "客户C", "status": "confirmed"},
            {"customer_id": "customer-d", "display_name": "客户D", "status": "confirmed"},
        ],
    )

    assert execution.tool_results[0].called is True
    demand = next(iter(store.waiting_demands.values()))
    assert demand.status == WaitingDemandStatus.MATCHED
    assert demand.matched_game_id
    drafts = list(store.outbound_message_drafts.values())
    assert len(drafts) == 1
    assert drafts[0].conversation_id == "conversation-a"
    assert drafts[0].recipient_id == "customer-a"
    assert "0.5" in drafts[0].message_text
    assert "你要打吗" in drafts[0].message_text
    assert drafts[0].metadata["waiting_demand_id"] == demand.demand_id
    assert client.calls[0]["system_trigger"]["trigger_type"] == "waiting_demand_match_found"
    assert "participants" not in client.calls[0]["system_trigger"]["game"]
    game = store.require_game(demand.matched_game_id)
    assert all(item.customer_id != "customer-a" for item in game.participants)


def test_stake_mismatch_does_not_trigger_notification(tmp_path) -> None:
    client = TriggerAwareClient()
    runtime = AgentRuntime(llm_client=client, store=SQLiteAgentStore(tmp_path / "stake-mismatch.sqlite3"))
    _register_waiting_demand(runtime, stake="0.5")

    _create_game_through_runtime(
        runtime,
        conversation_id="conversation-b",
        organizer_id="customer-b",
        stake="1",
    )

    assert runtime.store.list_active_demands()
    assert runtime.store.outbound_message_drafts == {}
    assert client.calls == []


def test_tonight_demand_keeps_its_meaning_across_midnight() -> None:
    timezone = ZoneInfo("Asia/Shanghai")
    demand = WaitingDemand(
        demand_id="late-night-demand",
        conversation_id="conversation-a",
        sender_id="customer-a",
        sender_name="客户A",
        demand={
            "stake": "0.5",
            "smoke_preference": "no_smoking",
            "time_preference": "今晚",
            "extra_constraints": [],
        },
        created_at=datetime(2026, 7, 21, 23, 30, tzinfo=timezone),
        expires_at=datetime(2026, 7, 22, 2, 0, tzinfo=timezone),
    )
    game = InMemoryAgentStore().create_game(
        conversation_id="conversation-b",
        organizer_id="customer-b",
        organizer_name="客户B",
        requirement={
            "game_type": "hangzhou_mahjong",
            "stake": "0.5",
            "smoke_preference": "no_smoking",
            "start_time_kind": "scheduled",
            "planned_start_at": datetime(2026, 7, 22, 0, 30, tzinfo=timezone).isoformat(),
        },
        known_players=[],
        trace_id="trace-midnight-game",
    )[0]

    assert waiting_demand_mismatch_reason(demand, game) == ""


def test_avoid_player_constraint_blocks_match(tmp_path) -> None:
    client = TriggerAwareClient()
    runtime = AgentRuntime(llm_client=client, store=SQLiteAgentStore(tmp_path / "constraint.sqlite3"))
    _register_waiting_demand(runtime, extra_constraints=["不和老王打"])

    _create_game_through_runtime(
        runtime,
        conversation_id="conversation-b",
        organizer_id="customer-b",
        known_players=[
            {"customer_id": "old-wang", "display_name": "老王", "status": "confirmed"},
        ],
    )

    assert runtime.store.list_active_demands()
    assert runtime.store.outbound_message_drafts == {}
    assert client.calls == []


def test_expired_demand_does_not_trigger_notification(tmp_path) -> None:
    client = TriggerAwareClient()
    runtime = AgentRuntime(llm_client=client, store=SQLiteAgentStore(tmp_path / "expired.sqlite3"))
    _register_waiting_demand(runtime, expires_at=now() - timedelta(seconds=1))

    _create_game_through_runtime(
        runtime,
        conversation_id="conversation-b",
        organizer_id="customer-b",
    )

    demand = next(iter(runtime.store.waiting_demands.values()))
    assert demand.status == WaitingDemandStatus.EXPIRED
    assert runtime.store.outbound_message_drafts == {}
    assert client.calls == []


def test_same_demand_and_game_notification_is_idempotent(tmp_path) -> None:
    client = TriggerAwareClient()
    runtime = AgentRuntime(llm_client=client, store=SQLiteAgentStore(tmp_path / "idempotent.sqlite3"))
    _register_waiting_demand(runtime)
    execution = _create_game_through_runtime(
        runtime,
        conversation_id="conversation-b",
        organizer_id="customer-b",
    )
    game_id = execution.tool_results[0].result["game"]["game_id"]

    runtime.match_trigger.match_game(game_id, trace_id="trace-repeat", source_tool="create_game")

    assert len(runtime.store.outbound_message_drafts) == 1
    assert len(client.calls) == 1


class _RecordingDispatcher:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.calls: list[tuple[str, str]] = []

    def dispatch_match(self, demand: WaitingDemand, game, *, trace_id: str):
        with self._lock:
            self.calls.append((demand.demand_id, game.game_id))
        return None


def test_two_games_created_concurrently_cannot_claim_the_same_demand_twice(tmp_path) -> None:
    store = SQLiteAgentStore(tmp_path / "concurrent-match.sqlite3")
    store.insert_waiting_demand(
        WaitingDemand(
            demand_id="concurrent-demand",
            conversation_id="conversation-a",
            sender_id="customer-a",
            sender_name="客户A",
            demand={
                "stake": "0.5",
                "smoke_preference": "no_smoking",
                "time_preference": "不限",
                "extra_constraints": [],
            },
            expires_at=now() + timedelta(hours=2),
        )
    )
    games = []
    for suffix in ("b", "c"):
        games.append(
            store.create_game(
                conversation_id=f"conversation-{suffix}",
                organizer_id=f"customer-{suffix}",
                organizer_name=f"客户{suffix.upper()}",
                requirement={
                    "game_type": "hangzhou_mahjong",
                    "stake": "0.5",
                    "smoke_preference": "no_smoking",
                    "start_time_kind": "scheduled",
                    "planned_start_at": (now() + timedelta(hours=1)).isoformat(),
                },
                known_players=[],
                trace_id=f"trace-create-{suffix}",
            )[0]
        )
    dispatcher = _RecordingDispatcher()
    trigger = MatchTrigger(store=store, dispatcher=dispatcher, trace_recorder=InMemoryTraceRecorder())

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(
            executor.map(
                lambda game: trigger.match_game(
                    game.game_id,
                    trace_id=f"trace-match-{game.game_id}",
                    source_tool="create_game",
                ),
                games,
            )
        )

    assert len(dispatcher.calls) == 1
    demand = store.waiting_demand("concurrent-demand")
    assert demand.status == WaitingDemandStatus.MATCHED
    assert demand.matched_game_id in {game.game_id for game in games}


def test_cancelled_demand_is_not_matched(tmp_path) -> None:
    client = TriggerAwareClient()
    runtime = AgentRuntime(llm_client=client, store=SQLiteAgentStore(tmp_path / "cancelled.sqlite3"))
    registered = _register_waiting_demand(runtime)
    demand_id = registered.result["waiting_demand"]["demand_id"]
    cancelled = runtime.tool_gateway.execute(
        ToolCall(
            name="cancel_waiting_demand",
            arguments={"demand_id": demand_id, "reason": "用户说算了不打了"},
            reason="用户明确取消等待需求",
        ),
        trace_id="trace-cancel-demand",
        conversation_id="conversation-a",
        sender_id="customer-a",
        sender_name="客户A",
        step_index=1,
        source_message_id="cancel-demand-message",
    )
    assert cancelled.called and cancelled.allowed, cancelled.error

    _create_game_through_runtime(
        runtime,
        conversation_id="conversation-b",
        organizer_id="customer-b",
    )

    assert runtime.store.waiting_demand(demand_id).status == WaitingDemandStatus.CANCELLED
    assert runtime.store.outbound_message_drafts == {}
    assert client.calls == []


def test_customer_confirmation_after_match_joins_cross_conversation_game(tmp_path) -> None:
    client = MatchThenJoinClient()
    runtime = AgentRuntime(llm_client=client, store=SQLiteAgentStore(tmp_path / "match-then-join.sqlite3"))
    _register_waiting_demand(runtime)
    execution = _create_game_through_runtime(
        runtime,
        conversation_id="conversation-b",
        organizer_id="customer-b",
        known_players=[
            {"customer_id": "customer-b", "display_name": "客户B", "status": "confirmed"},
            {"customer_id": "customer-c", "display_name": "客户C", "status": "confirmed"},
            {"customer_id": "customer-d", "display_name": "客户D", "status": "confirmed"},
        ],
    )
    game_id = execution.tool_results[0].result["game"]["game_id"]

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="conversation-a",
            sender_id="customer-a",
            sender_name="客户A",
            text="打",
            message_id="customer-a-confirms-match",
        ),
        trace_id="trace-customer-a-confirms-match",
    )

    assert result.final_reply == "好"
    game = runtime.store.require_game(game_id)
    joined = [item for item in game.participants if item.customer_id == "customer-a"]
    assert len(joined) == 1
    assert joined[0].seat_count == 1
    assert len(client.calls) == 3


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
def test_scheduled_waiting_expiration_marks_stale_and_enqueues_next_sweep(kind: str, tmp_path) -> None:
    store = _store(kind, tmp_path, "scheduled-expiration")
    stamp = now()
    store.insert_waiting_demand(
        WaitingDemand(
            demand_id="scheduled-stale-demand",
            conversation_id="conversation-a",
            sender_id="customer-a",
            sender_name="客户A",
            demand={"stake": "0.5"},
            expires_at=stamp + timedelta(seconds=1),
        )
    )
    task, _ = store.ensure_waiting_demand_expiration_task(
        due_at=stamp + timedelta(seconds=2),
        trace_id="trace-schedule-expiration",
    )
    trace = InMemoryTraceRecorder()
    scheduler = ScheduledAgentTaskScheduler(
        store=store,
        handler=lambda claimed, trace_id: handle_waiting_expiration_task(
            store,
            claimed,
            trace_id=trace_id,
            trace_recorder=trace,
        ),
        trace_recorder=trace,
    )

    assert scheduler.run_due_once(at=task.due_at) == 1
    assert store.waiting_demand("scheduled-stale-demand").status == WaitingDemandStatus.EXPIRED
    future_tasks = store.due_scheduled_tasks(at=task.due_at + timedelta(minutes=2), limit=10)
    assert len(future_tasks) == 1
    assert future_tasks[0].task_id != task.task_id
