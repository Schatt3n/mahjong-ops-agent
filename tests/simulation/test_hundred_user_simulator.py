from __future__ import annotations

import json
import sqlite3
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest


SIMULATION_DIR = Path(__file__).resolve().parent
if str(SIMULATION_DIR) not in sys.path:
    sys.path.insert(0, str(SIMULATION_DIR))

from behavior_policy import (  # noqa: E402
    FOLLOW_UP_REPLY_POOL,
    NEW_TOPIC_POOL,
    BehaviorPolicy,
    QUESTION_POOL,
    SimulationAction,
)
from hundred_user_simulator import HundredUserSimulator, parse_speed  # noqa: E402
from sim_adapter import (  # noqa: E402
    RequestOutcome,
    SimulationAdapter,
    StaticAgentLLMClient,
    build_runtime,
    required_llm_mode,
    running_http_backend,
)
from sim_factory import (  # noqa: E402
    GROUP_ID,
    PERSONA_ACTIVE_GAMBLER,
    PERSONA_LURKER,
    PERSONA_TROUBLEMAKER,
    VirtualUser,
    build_population,
    ensure_isolated_database,
)
from sim_orchestrator import DialogState, RateLimiter, SimulationOrchestrator  # noqa: E402


def _users() -> list[VirtualUser]:
    personas = [PERSONA_LURKER] * 80 + [PERSONA_ACTIVE_GAMBLER] * 15 + [PERSONA_TROUBLEMAKER] * 5
    return [
        VirtualUser(
            customer_id=f"sim_user_{index:03d}",
            display_name=f"用户{index}",
            balance=float(index * 10),
            preferred_game="sichuan_mahjong" if index % 2 else "national_standard_mahjong",
            persona=persona,
        )
        for index, persona in enumerate(personas, start=1)
    ]


class _LockedAdapter:
    def __init__(self, users: list[VirtualUser]) -> None:
        self.users = users

    def send(self, action: SimulationAction, *, deadline: float) -> RequestOutcome:
        del deadline
        return RequestOutcome(
            action=action,
            sent=True,
            status_code=500,
            response={"detail": "database is locked"},
            error="HTTP 500",
            sent_at=time.monotonic(),
        )

    def inbox_sizes(self) -> dict[str, int]:
        return {user.customer_id: 0 for user in self.users}


class _UnusedAdapter(_LockedAdapter):
    def send(self, action: SimulationAction, *, deadline: float) -> RequestOutcome:  # pragma: no cover
        raise AssertionError("duration limit should stop before the first scheduled action")


class _ExpiredLockAdapter(_LockedAdapter):
    def __init__(self, users: list[VirtualUser], locked_user_id: str) -> None:
        super().__init__(users)
        self.conversation_id = f"sim:group:{GROUP_ID}"
        self.locked_user_id = locked_user_id
        self.sent_actions: list[SimulationAction] = []

    def next_speaker_only(self, conversation_id: str) -> str | None:
        return self.locked_user_id if conversation_id == self.conversation_id else None

    def expired_speaker_locks(self, timeout_seconds: float):
        del timeout_seconds
        return [(self.conversation_id, self.locked_user_id)] if self.locked_user_id else []

    def seconds_until_lock_timeout(self, timeout_seconds: float) -> float | None:
        del timeout_seconds
        return 0.0 if self.locked_user_id else None

    def release_speaker_lock(
        self,
        conversation_id: str,
        *,
        expected_user_id: str | None = None,
    ) -> bool:
        if conversation_id != self.conversation_id:
            return False
        if expected_user_id is not None and expected_user_id != self.locked_user_id:
            return False
        self.locked_user_id = ""
        return True

    def send(self, action: SimulationAction, *, deadline: float) -> RequestOutcome:
        del deadline
        self.sent_actions.append(action)
        return RequestOutcome(
            action=action,
            sent=True,
            status_code=200,
            response={"trace_id": "trace_timeout", "final_reply": "收到"},
            sent_at=time.monotonic(),
        )


def test_population_factory_is_deterministic_and_isolated(tmp_path: Path) -> None:
    first_store, first_users = build_population(tmp_path / "first" / "test_sim.db", seed=42)
    second_store, second_users = build_population(tmp_path / "second" / "test_sim.db", seed=42)

    assert first_users == second_users
    assert len(first_users) == 100
    assert {user.preferred_game for user in first_users} == {
        "sichuan_mahjong",
        "national_standard_mahjong",
    }
    assert [user.persona for user in first_users].count(PERSONA_LURKER) == 80
    assert [user.persona for user in first_users].count(PERSONA_ACTIVE_GAMBLER) == 15
    assert [user.persona for user in first_users].count(PERSONA_TROUBLEMAKER) == 5

    with first_store._lock:
        connection = first_store._connection
        assert connection.execute("SELECT COUNT(*) FROM runtime_customers").fetchone()[0] == 100
        assert connection.execute("SELECT COUNT(*) FROM simulation_user_profiles").fetchone()[0] == 100
        assert connection.execute("SELECT COUNT(*) FROM simulation_group_chats").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM simulation_group_members").fetchone()[0] == 100
        assert connection.execute(
            "SELECT COUNT(*) FROM simulation_group_members WHERE group_id = ?",
            (GROUP_ID,),
        ).fetchone()[0] == 100
        balance = connection.execute(
            "SELECT balance FROM simulation_user_profiles WHERE customer_id = ?",
            (first_users[0].customer_id,),
        ).fetchone()[0]
    assert balance == first_users[0].balance
    assert first_store.customers[first_users[0].customer_id].profile_facts == [
        f"simulation_balance={first_users[0].balance:.2f}"
    ]

    # Keep the second connection live long enough to prove both independent DBs open.
    assert second_store._connection.execute("SELECT COUNT(*) FROM runtime_customers").fetchone()[0] == 100


def test_database_guard_refuses_development_or_wrong_filename(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="filename must be test_sim.db"):
        ensure_isolated_database(tmp_path / "anything.sqlite3")
    assert ensure_isolated_database(tmp_path / "test_sim.db").name == "test_sim.db"


def test_behavior_policy_uses_personas_typos_recalls_and_eighty_twenty_channels() -> None:
    users = _users()
    policy = BehaviorPolicy(users, seed=42)
    assert len(QUESTION_POOL) == 20
    assert len(policy.speaking_users()) == 20
    assert policy.first_action(users[0], sequence=1) is None

    active = next(user for user in users if user.persona == PERSONA_ACTIVE_GAMBLER)
    first = policy.first_action(active, sequence=2)
    assert first is not None
    assert first.text == "还有位置吗"
    assert first.due_simulated_seconds == 10.0
    following = policy.following_action(active, first, sequence=3)
    assert following.due_simulated_seconds == 20.0

    trouble = next(user for user in users if user.persona == PERSONA_TROUBLEMAKER)
    trouble_actions = [
        policy.get_next_action(trouble, sequence=10 + index, due_simulated_seconds=float(index))
        for index in range(1, 4)
    ]
    assert all(action is not None for action in trouble_actions)
    assert trouble_actions[0].event_type == "text"  # type: ignore[union-attr]
    assert trouble_actions[2].event_type == "recall"  # type: ignore[union-attr]
    assert trouble_actions[2].recalled_message_id == trouble_actions[1].message_id  # type: ignore[union-attr]
    assert trouble_actions[2].conversation_id == trouble_actions[1].conversation_id  # type: ignore[union-attr]

    channels: list[str] = []
    speakers = policy.speaking_users()
    for index in range(5000):
        action = policy.get_next_action(
            speakers[index % len(speakers)],
            sequence=1000 + index,
            due_simulated_seconds=float(index),
        )
        assert action is not None
        channels.append(action.channel)
    group_ratio = channels.count("group") / len(channels)
    assert 0.78 <= group_ratio <= 0.82


def test_behavior_policy_uses_dialog_state_for_follow_up_or_silence() -> None:
    users = _users()
    active = next(user for user in users if user.persona == PERSONA_ACTIVE_GAMBLER)
    policy = BehaviorPolicy(users, seed=42)
    state = DialogState(
        turn_count=1,
        pending_response_to="user",
        last_agent_reply=f"@{active.display_name} 你几点方便？",
        last_conversation_id=f"sim:group:{GROUP_ID}",
        last_channel="group",
    )

    action = policy.get_next_action(
        active,
        sequence=101,
        due_simulated_seconds=20.0,
        dialog_state=state,
    )
    assert action is not None
    assert action.text in FOLLOW_UP_REPLY_POOL
    assert action.channel == "group"
    assert action.conversation_id == f"sim:group:{GROUP_ID}"

    state.last_agent_reply = "好的，我看一下。"
    next_action = policy.get_next_action(
        active,
        sequence=102,
        due_simulated_seconds=30.0,
        dialog_state=state,
    )
    assert next_action is None or next_action.text in NEW_TOPIC_POOL


def test_wechat_payload_contains_raw_channel_fields() -> None:
    user = _users()[80]
    action = BehaviorPolicy(_users(), seed=42).first_action(user, sequence=1)
    assert action is not None
    payload = action.to_wechat_payload()
    raw = payload["metadata"]["raw_wechat_payload"]
    assert raw["platform_name"] == "wechaty"
    assert raw["talker"]["id"] == user.customer_id
    assert raw["source_message_id"] == action.message_id
    assert raw["is_room"] is (action.channel == "group")


def test_rate_limiter_never_grants_more_than_five_per_second() -> None:
    clock = [0.0]

    def advance(delay: float, stop_event) -> bool:
        del stop_event
        clock[0] += delay
        return False

    limiter = RateLimiter(
        max_calls=5,
        monotonic_fn=lambda: clock[0],
        wait_fn=advance,
    )
    assert all(limiter.acquire() for _ in range(6))
    assert limiter.grant_history() == [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    with pytest.raises(ValueError, match="between 1 and 5"):
        RateLimiter(max_calls=6)


def test_llm_mode_is_mandatory_and_mock_never_loads_real_client(tmp_path: Path, monkeypatch) -> None:
    assert required_llm_mode({"SIM_LLM_MODE": "mock"}) == "mock"
    assert required_llm_mode({"SIM_LLM_MODE": "real"}) == "real"
    with pytest.raises(RuntimeError, match="SIM_LLM_MODE is required"):
        required_llm_mode({})
    with pytest.raises(RuntimeError, match="SIM_LLM_MODE is required"):
        required_llm_mode({"SIM_LLM_MODE": "auto"})

    import sim_adapter

    monkeypatch.setattr(
        sim_adapter,
        "load_dotenv_defaults",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("real env loaded in mock mode")),
    )
    store, _ = build_population(tmp_path / "test_sim.db")
    _, client = build_runtime(store, "mock")
    assert isinstance(client, StaticAgentLLMClient)


def test_adapter_broadcasts_group_replies_and_private_replies() -> None:
    users = _users()
    adapter = SimulationAdapter(base_url="http://127.0.0.1:1", users=users)
    group_action = SimulationAction(
        due_simulated_seconds=1.0,
        sequence=1,
        channel="group",
        conversation_id=f"sim:group:{GROUP_ID}",
        sender_id=users[0].customer_id,
        sender_name=users[0].display_name,
        text="还有位置吗",
    )
    group_outcome = RequestOutcome(
        action=group_action,
        sent=True,
        status_code=200,
        response={"trace_id": "trace_group", "final_reply": "有的"},
    )
    assert adapter._deliver_reply(group_outcome) == 100
    assert all(adapter.inbox_for(user.customer_id)[0].text == "有的" for user in users)

    private_action = SimulationAction(
        due_simulated_seconds=2.0,
        sequence=2,
        channel="private",
        conversation_id=f"sim:private:{users[0].customer_id}",
        sender_id=users[0].customer_id,
        sender_name=users[0].display_name,
        text="包间多少钱",
    )
    private_outcome = RequestOutcome(
        action=private_action,
        sent=True,
        status_code=200,
        response={"trace_id": "trace_private", "final_reply": "我看一下"},
    )
    assert adapter._deliver_reply(private_outcome) == 1
    assert len(adapter.inbox_for(users[0].customer_id)) == 2
    assert len(adapter.inbox_for(users[1].customer_id)) == 1


def test_adapter_extracts_mentions_and_broadcast_reply_releases_turn_lock() -> None:
    users = _users()
    adapter = SimulationAdapter(base_url="http://127.0.0.1:1", users=users)
    conversation_id = f"sim:group:{GROUP_ID}"
    action = SimulationAction(
        due_simulated_seconds=1.0,
        sequence=1,
        channel="group",
        conversation_id=conversation_id,
        sender_id=users[80].customer_id,
        sender_name=users[80].display_name,
        text="还有位置吗",
    )
    mentioned = users[81]
    mentioned_outcome = RequestOutcome(
        action=action,
        sent=True,
        status_code=200,
        response={"trace_id": "trace_mention", "final_reply": f"@{mentioned.display_name} 几点来？"},
    )
    adapter._deliver_reply(mentioned_outcome)
    assert mentioned_outcome.next_speaker_only == mentioned.customer_id
    assert adapter.next_speaker_only(conversation_id) == mentioned.customer_id

    broadcast_outcome = RequestOutcome(
        action=action,
        sent=True,
        status_code=200,
        response={"trace_id": "trace_broadcast", "final_reply": "大家都可以说"},
    )
    adapter._deliver_reply(broadcast_outcome)
    assert broadcast_outcome.next_speaker_only is None
    assert adapter.next_speaker_only(conversation_id) is None


def test_orchestrator_only_dispatches_the_mentioned_group_user(tmp_path: Path) -> None:
    users = _users()
    adapter = SimulationAdapter(base_url="http://127.0.0.1:1", users=users)
    policy = BehaviorPolicy(users, seed=42)
    orchestrator = SimulationOrchestrator(
        users=users,
        behavior_policy=policy,
        adapter=adapter,
        max_messages=2,
        max_duration_seconds=1,
        speed=1000,
        report_path=tmp_path / "unused.json",
    )
    conversation_id = f"sim:group:{GROUP_ID}"
    target = users[81]
    other = users[80]
    lock_action = SimulationAction(
        due_simulated_seconds=0.0,
        sequence=1,
        channel="group",
        conversation_id=conversation_id,
        sender_id=other.customer_id,
        sender_name=other.display_name,
        text="测试",
    )
    adapter._deliver_reply(
        RequestOutcome(
            action=lock_action,
            sent=True,
            status_code=200,
            response={"final_reply": f"@{target.display_name} 确认一下？"},
        )
    )
    orchestrator._enqueue_action(replace(lock_action, sequence=2))
    orchestrator._enqueue_action(
        replace(
            lock_action,
            sequence=3,
            sender_id=target.customer_id,
            sender_name=target.display_name,
        )
    )

    selected = orchestrator._take_dispatchable_action(time.monotonic() - 1)
    assert selected is not None
    assert selected.sender_id == target.customer_id


def test_mock_http_simulation_generates_complete_report_and_inboxes(tmp_path: Path) -> None:
    store, users = build_population(tmp_path / "test_sim.db", seed=42)
    runtime, client = build_runtime(store, "mock")
    report_path = tmp_path / "sim_report.json"
    with running_http_backend(runtime) as base_url:
        simulator = HundredUserSimulator(
            users=users,
            base_url=base_url,
            max_messages=10,
            max_duration_seconds=5,
            speed=100.0,
            report_path=report_path,
        )
        report = simulator.run()

    persisted_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report == persisted_report
    assert report["stop_reason"] == "message_limit"
    assert report["total_messages"] == 10
    assert report["tool_calls"] == {"total": 10, "successful": 10, "failed": 0, "success_rate": 1.0}
    assert report["sqlite_lock_wait_count"] == 0
    assert report["has_empty_final_reply"] is False
    assert report["users_with_inbox_messages"] == 100
    assert client.call_count == 20
    assert len(report["transcript"]) == 10
    assert [turn["sequence"] for turn in report["transcript"]] == sorted(
        turn["sequence"] for turn in report["transcript"]
    )
    first_turn = report["transcript"][0]
    assert first_turn["user"]["text"]
    assert first_turn["agent"]["reply"]
    assert first_turn["agent"]["trace_id"]
    assert first_turn["tool_calls"] == ["search_current_games"]


def test_seeded_mock_replies_are_reproducible_but_vary_across_seeds(tmp_path: Path) -> None:
    first_store, _ = build_population(tmp_path / "first" / "test_sim.db", seed=42)
    second_store, _ = build_population(tmp_path / "second" / "test_sim.db", seed=42)
    third_store, _ = build_population(tmp_path / "third" / "test_sim.db", seed=42)
    _, first = build_runtime(first_store, "mock", seed=42)
    _, second = build_runtime(second_store, "mock", seed=42)
    _, third = build_runtime(third_store, "mock", seed=7)
    payload = [{"role": "user", "content": json.dumps({"current_message": {"sender_name": "张哥"}, "previous_tool_results": [{}]}, ensure_ascii=False)}]

    first_reply = json.loads(first.complete(payload, trace_id="t1", timeout_seconds=1))["reply_to_user"]
    second_reply = json.loads(second.complete(payload, trace_id="t2", timeout_seconds=1))["reply_to_user"]
    third_reply = json.loads(third.complete(payload, trace_id="t3", timeout_seconds=1))["reply_to_user"]

    assert first_reply == second_reply
    assert first_reply != third_reply


def test_mock_http_conversation_reaches_turn_limit_and_reports_completion(tmp_path: Path) -> None:
    store, generated_users = build_population(tmp_path / "test_sim.db", seed=42)
    active_id = generated_users[80].customer_id
    users = [
        replace(
            user,
            persona=(PERSONA_ACTIVE_GAMBLER if user.customer_id == active_id else PERSONA_LURKER),
        )
        for user in generated_users
    ]
    runtime, _ = build_runtime(store, "mock")
    report_path = tmp_path / "multi_turn_report.json"
    with running_http_backend(runtime) as base_url:
        simulator = HundredUserSimulator(
            users=users,
            base_url=base_url,
            max_messages=10,
            max_duration_seconds=1.5,
            speed=1000.0,
            report_path=report_path,
        )
        report = simulator.run()

    state = simulator.orchestrator.active_sessions[active_id]
    assert report["total_messages"] == 5
    assert state.turn_count == 5
    assert state.status == "idle"
    assert state.pending_response_to is None
    assert report["sessions_with_at_least_3_turns"] == 1
    assert report["multi_turn_completion_rate"] == 1.0
    assert report["average_dialog_turns"] == 5.0
    assert report["timeout_broken_sessions"] == 0


def test_expired_turn_lock_forces_silence_and_counts_broken_session(tmp_path: Path) -> None:
    users = _users()
    locked_user = users[0]
    adapter = _ExpiredLockAdapter(users, locked_user.customer_id)
    orchestrator = SimulationOrchestrator(
        users=users,
        behavior_policy=BehaviorPolicy(users, seed=42),
        adapter=adapter,  # type: ignore[arg-type]
        max_messages=1,
        max_duration_seconds=1,
        speed=1000,
        lock_timeout_seconds=0.01,
        report_path=tmp_path / "timeout_report.json",
    )
    report = orchestrator.run()

    assert len(adapter.sent_actions) == 1
    assert adapter.sent_actions[0].text == "（沉默/退出）"
    assert adapter.sent_actions[0].event_type == "timeout_exit"
    assert orchestrator.active_sessions[locked_user.customer_id].status == "idle"
    assert report["timeout_broken_sessions"] == 1


def test_timeout_exit_can_bypass_a_stuck_inflight_request(tmp_path: Path) -> None:
    users = _users()
    locked_user = users[0]
    adapter = _ExpiredLockAdapter(users, locked_user.customer_id)
    orchestrator = SimulationOrchestrator(
        users=users,
        behavior_policy=BehaviorPolicy(users, seed=42),
        adapter=adapter,  # type: ignore[arg-type]
        max_messages=2,
        max_duration_seconds=1,
        speed=1,
        lock_timeout_seconds=0.01,
        report_path=tmp_path / "unused_timeout_report.json",
    )
    orchestrator._inflight_conversations.add(adapter.conversation_id)
    orchestrator._inflight_users.add(locked_user.customer_id)
    started = time.monotonic() - 1
    orchestrator._enqueue_expired_lock_actions(started)

    selected = orchestrator._take_dispatchable_action(started)
    assert selected is not None
    assert selected.event_type == "timeout_exit"
    assert selected.text == "（沉默/退出）"


def test_orchestrator_fails_after_five_consecutive_sqlite_locks(tmp_path: Path) -> None:
    users = _users()
    adapter = _LockedAdapter(users)
    orchestrator = SimulationOrchestrator(
        users=users,
        behavior_policy=BehaviorPolicy(users, seed=42),
        adapter=adapter,  # type: ignore[arg-type]
        max_messages=100,
        max_duration_seconds=5,
        speed=1000,
        report_path=tmp_path / "lock_report.json",
    )
    report = orchestrator.run()
    assert report["status"] == "failed"
    assert report["stop_reason"] == "sqlite_lock_failure"
    assert report["sqlite_lock_wait_count"] >= 5
    assert report["max_consecutive_sqlite_lock_errors"] >= 5


def test_orchestrator_stops_at_duration_before_first_scheduled_event(tmp_path: Path) -> None:
    users = _users()
    adapter = _UnusedAdapter(users)
    orchestrator = SimulationOrchestrator(
        users=users,
        behavior_policy=BehaviorPolicy(users, seed=42),
        adapter=adapter,  # type: ignore[arg-type]
        max_messages=500,
        max_duration_seconds=0.1,
        speed=1.0,
        report_path=tmp_path / "duration_report.json",
    )
    report = orchestrator.run()
    assert report["stop_reason"] == "duration_limit"
    assert report["total_messages"] == 0


@pytest.mark.parametrize(("raw", "expected"), [("10x", 10.0), ("0.5x", 0.5), ("2", 2.0)])
def test_speed_parser(raw: str, expected: float) -> None:
    assert parse_speed(raw) == expected
