from __future__ import annotations

import datetime as dt
import importlib.util
from pathlib import Path

from mahjong_agent_runtime import (
    AgentRuntimeResult,
    InMemoryAgentStore,
    InMemoryTraceRecorder,
    PendingInputBatchStatus,
    PendingInputScheduler,
    SQLiteAgentStore,
    UserMessage,
    aggregate_pending_input_batch,
)
from mahjong_agent_runtime.processing import input_batch_run_is_stale


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "scripts" / "agent_runtime_app.py"
spec = importlib.util.spec_from_file_location("agent_runtime_app_input_aggregation_test", APP_PATH)
assert spec is not None and spec.loader is not None
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)


def message(text: str, message_id: str, *, sender_id: str = "zhang") -> UserMessage:
    return UserMessage(
        conversation_id="group_001",
        sender_id=sender_id,
        sender_name=sender_id,
        text=text,
        message_id=message_id,
        metadata={"channel": "wechaty"},
    )


def test_fragments_are_ordered_deduplicated_and_scoped_by_sender() -> None:
    store = InMemoryAgentStore()
    base = dt.datetime.now().astimezone()

    first, _, added = store.upsert_pending_input_fragment(
        message("老板", "m1"), trace_id="trace_1", quiet_deadline=base + dt.timedelta(seconds=30)
    )
    second, _, added_second = store.upsert_pending_input_fragment(
        message("帮我组个局", "m2"), trace_id="trace_2", quiet_deadline=base + dt.timedelta(seconds=35)
    )
    duplicate, transition, duplicate_added = store.upsert_pending_input_fragment(
        message("帮我组个局", "m2"), trace_id="trace_2_retry", quiet_deadline=base + dt.timedelta(seconds=40)
    )
    other_sender, _, _ = store.upsert_pending_input_fragment(
        message("我也想打", "m3", sender_id="wang"),
        trace_id="trace_3",
        quiet_deadline=base + dt.timedelta(seconds=30),
    )

    assert added is True
    assert added_second is True
    assert second.batch_id == first.batch_id
    assert second.version == 2
    assert [item["text"] for item in second.fragments] == ["老板", "帮我组个局"]
    assert duplicate_added is False
    assert transition is None
    assert duplicate.version == 2
    assert other_sender.batch_id != second.batch_id
    assert len(store.pending_input_batches) == 2


def test_aggregate_preserves_fragments_and_marks_quiet_trigger() -> None:
    store = InMemoryAgentStore()
    deadline = dt.datetime.now().astimezone() + dt.timedelta(seconds=30)
    batch, _, _ = store.upsert_pending_input_fragment(
        message("老板", "m1"), trace_id="trace_1", quiet_deadline=deadline
    )
    batch, _, _ = store.upsert_pending_input_fragment(
        message("0.5，无烟，人齐开", "m2"), trace_id="trace_2", quiet_deadline=deadline
    )

    aggregate = aggregate_pending_input_batch(batch, quiet_period_elapsed=True, trigger="quiet_period_elapsed")

    assert aggregate.text == "老板\n0.5，无烟，人齐开"
    assert aggregate.metadata["input_window"]["quiet_period_elapsed"] is True
    assert aggregate.metadata["input_window"]["source_message_ids"] == ["m1", "m2"]
    assert aggregate.metadata["input_window"]["batch_version"] == 2


def test_new_fragment_invalidates_claimed_batch_version() -> None:
    store = InMemoryAgentStore()
    deadline = dt.datetime.now().astimezone() + dt.timedelta(seconds=30)
    batch, _, _ = store.upsert_pending_input_fragment(
        message("帮我组个局", "m1"), trace_id="trace_1", quiet_deadline=deadline
    )
    claimed, _ = store.claim_pending_input_batch(
        batch_id=batch.batch_id,
        expected_version=batch.version,
        trace_id="trace_claim",
    )
    assert claimed is not None
    aggregate = aggregate_pending_input_batch(claimed, quiet_period_elapsed=False, trigger="message_arrived")

    current, transition, added = store.upsert_pending_input_fragment(
        message("0.5，无烟", "m2"), trace_id="trace_2", quiet_deadline=deadline
    )
    finished, _ = store.finish_pending_input_batch(
        batch_id=batch.batch_id,
        expected_version=claimed.version,
        status=PendingInputBatchStatus.COMPLETED,
        trace_id="trace_old_finish",
    )

    assert added is True
    assert transition is not None and transition.from_status == PendingInputBatchStatus.PROCESSING.value
    assert current.version == 2
    assert current.status == PendingInputBatchStatus.PENDING
    assert finished is None
    assert input_batch_run_is_stale(store, aggregate) is True


def test_sqlite_pending_batch_survives_restart_and_claim_is_compare_and_set(tmp_path: Path) -> None:
    path = tmp_path / "input_batches.sqlite3"
    deadline = dt.datetime.now().astimezone() - dt.timedelta(seconds=1)
    first_store = SQLiteAgentStore(path)
    batch, _, _ = first_store.upsert_pending_input_fragment(
        message("老板", "m1"), trace_id="trace_1", quiet_deadline=deadline
    )

    second_store = SQLiteAgentStore(path)
    recovered = second_store.pending_input_batch("group_001", "zhang")
    assert recovered is not None
    assert recovered.batch_id == batch.batch_id
    assert second_store.due_pending_input_batches(at=dt.datetime.now().astimezone())[0].batch_id == batch.batch_id

    claimed_by_first, _ = first_store.claim_pending_input_batch(
        batch_id=batch.batch_id,
        expected_version=1,
        trace_id="trace_claim_1",
    )
    claimed_by_second, _ = second_store.claim_pending_input_batch(
        batch_id=batch.batch_id,
        expected_version=1,
        trace_id="trace_claim_2",
    )

    assert claimed_by_first is not None
    assert claimed_by_second is None


def test_scheduler_replays_due_batch_once() -> None:
    store = InMemoryAgentStore()
    trace = InMemoryTraceRecorder()
    deadline = dt.datetime.now().astimezone() - dt.timedelta(seconds=1)
    batch, _, _ = store.upsert_pending_input_fragment(
        message("帮我组个局", "m1"), trace_id="trace_1", quiet_deadline=deadline
    )
    handled: list[str] = []

    def handler(due_batch, trace_id: str) -> None:
        claimed, _ = store.claim_pending_input_batch(
            batch_id=due_batch.batch_id,
            expected_version=due_batch.version,
            trace_id=trace_id,
        )
        assert claimed is not None
        store.finish_pending_input_batch(
            batch_id=claimed.batch_id,
            expected_version=claimed.version,
            status=PendingInputBatchStatus.COMPLETED,
            trace_id=trace_id,
        )
        handled.append(due_batch.batch_id)

    scheduler = PendingInputScheduler(store=store, handler=handler, trace_recorder=trace)

    assert scheduler.run_due_once() == 1
    assert scheduler.run_due_once() == 0
    assert handled == [batch.batch_id]


class FakeRuntime:
    def __init__(self) -> None:
        self.store = InMemoryAgentStore()
        self.trace_recorder = InMemoryTraceRecorder()
        self.received: list[UserMessage] = []

    def handle_user_message(self, incoming: UserMessage, *, trace_id: str) -> AgentRuntimeResult:
        self.received.append(incoming)
        return AgentRuntimeResult(
            trace_id=trace_id,
            conversation_id=incoming.conversation_id,
            final_reply="好的，我帮你看看。",
        )


def test_model_can_wait_then_release_one_merged_message(monkeypatch) -> None:
    runtime = FakeRuntime()
    actions = iter(["wait_for_more_input", "wait_for_more_input", "process_business"])

    def gate(*args, **kwargs):
        action = next(actions)
        return {
            "action": action,
            "should_route": action == "process_business",
            "should_wait": action == "wait_for_more_input",
            "category": "operational",
            "confidence": 0.95,
            "reasoning_summary": "test",
            "evidence": [],
            "errors": [],
        }

    monkeypatch.setattr(app, "run_wechaty_input_gate", gate)
    first = app.route_user_message_with_aggregation(
        runtime, message("老板", "m1"), trace_id="trace_1", channel="wechaty"
    )
    second = app.route_user_message_with_aggregation(
        runtime, message("帮我组个局", "m2"), trace_id="trace_2", channel="wechaty"
    )
    third = app.route_user_message_with_aggregation(
        runtime, message("0.5，无烟，人齐开", "m3"), trace_id="trace_3", channel="wechaty"
    )

    assert first["waiting_for_more_input"] is True
    assert second["waiting_for_more_input"] is True
    assert third["waiting_for_more_input"] is False
    assert third["background_processing"] is True
    assert third["agent_result"]["final_reply"] == "好的，我看看"
    assert len(runtime.received) == 0

    queued = runtime.store.pending_input_batch("group_001", "zhang")
    assert queued is not None
    assert queued.decision["dispatch_mode"] == "background_after_immediate_ack"
    background = app.dispatch_pending_input_batch(
        runtime,
        queued,
        trace_id="trace_background",
        quiet_period_elapsed=False,
        trigger="background_after_immediate_ack",
        gate_decision_override=queued.decision,
        allow_immediate_ack=False,
    )

    assert background["input_status"] == PendingInputBatchStatus.COMPLETED.value
    assert len(runtime.received) == 1
    assert runtime.received[0].text == "老板\n帮我组个局\n0.5，无烟，人齐开"
    assert runtime.received[0].metadata["delivery_context"]["mode"] == "follow_up_after_immediate_ack"
    assert runtime.received[0].metadata["delivery_context"]["immediate_ack_text"] == "好的，我看看"


def test_immediate_ack_queue_survives_sqlite_restart(tmp_path: Path) -> None:
    path = tmp_path / "immediate_ack.sqlite3"
    store = SQLiteAgentStore(path)
    deadline = dt.datetime.now().astimezone() + dt.timedelta(seconds=30)
    batch, _, _ = store.upsert_pending_input_fragment(
        message("帮我组个局", "m1"),
        trace_id="trace_input",
        quiet_deadline=deadline,
    )
    due_at = dt.datetime.now().astimezone() + dt.timedelta(seconds=1)
    queued = store.queue_pending_input_batch(
        batch_id=batch.batch_id,
        expected_version=batch.version,
        decision={
            "action": "process_business",
            "dispatch_mode": "background_after_immediate_ack",
            "queued_batch_version": batch.version,
            "immediate_ack_text": "好的，我看看",
        },
        due_at=due_at,
    )

    assert queued is not None
    reopened = SQLiteAgentStore(path)
    recovered = reopened.pending_input_batch("group_001", "zhang")
    assert recovered is not None
    assert recovered.decision["dispatch_mode"] == "background_after_immediate_ack"
    assert reopened.due_pending_input_batches(at=due_at + dt.timedelta(seconds=1))[0].batch_id == batch.batch_id


def test_duplicate_platform_message_does_not_repeat_immediate_ack(monkeypatch) -> None:
    runtime = FakeRuntime()

    monkeypatch.setattr(
        app,
        "run_wechaty_input_gate",
        lambda *args, **kwargs: {
            "action": "process_business",
            "should_route": True,
            "should_wait": False,
            "category": "operational",
            "confidence": 0.99,
            "reasoning_summary": "ready",
            "evidence": [],
            "errors": [],
        },
    )
    first = app.route_user_message_with_aggregation(
        runtime,
        message("0.5无烟人齐开", "same-message"),
        trace_id="trace_first",
        channel="wechaty",
    )
    duplicate = app.route_user_message_with_aggregation(
        runtime,
        message("0.5无烟人齐开", "same-message"),
        trace_id="trace_duplicate",
        channel="wechaty",
    )

    assert first["agent_result"]["final_reply"] == "好的，我看看"
    assert duplicate["input_status"] == "duplicate"
    assert duplicate["agent_result"] is None


def test_queued_business_input_is_observed_as_background_agent_work() -> None:
    observation = app.build_wechaty_channel_observation(
        payload={
            "source_message_id": "m1",
            "conversation_id": "wechaty:contact:a",
            "sender_id": "a",
            "raw_text": "帮我组个局",
        },
        record={
            "source_message_id": "m1",
            "trace_id": "trace_queued",
            "received_at": dt.datetime.now().astimezone().isoformat(),
        },
        route_result={
            "routed_to_agent": True,
            "background_processing": True,
            "audit": {
                "reason": "business_input_queued_after_immediate_ack",
                "input_gate": {"action": "process_business", "category": "operational", "confidence": 0.99},
            },
        },
    )

    assert observation["route_status"] == "queued"
    assert observation["route_mode"] == "background_agent"
    assert observation["business_message_detected"] is True


def test_delayed_reply_does_not_repeat_exact_immediate_ack() -> None:
    runtime = FakeRuntime()
    deadline = dt.datetime.now().astimezone()
    batch, _, _ = runtime.store.upsert_pending_input_fragment(
        message("帮我组个局", "m1"),
        trace_id="trace_input",
        quiet_deadline=deadline,
    )
    batch = runtime.store.queue_pending_input_batch(
        batch_id=batch.batch_id,
        expected_version=batch.version,
        decision={
            "action": "process_business",
            "dispatch_mode": "background_after_immediate_ack",
            "immediate_ack_text": "好的，我看看",
        },
        due_at=deadline,
    )
    assert batch is not None

    delivery = app.deliver_delayed_wechaty_reply(
        runtime,
        batch,
        {"agent_result": {"final_reply": "好的，我看看"}},
        trace_id="trace_background",
    )

    assert delivery == {
        "sent": False,
        "reason": "duplicate_of_immediate_ack",
        "reply": "好的，我看看",
    }


def test_quiet_elapsed_cannot_wait_forever(monkeypatch) -> None:
    runtime = FakeRuntime()
    deadline = dt.datetime.now().astimezone() - dt.timedelta(seconds=1)
    batch, _, _ = runtime.store.upsert_pending_input_fragment(
        message("帮我组个局", "m1"), trace_id="trace_1", quiet_deadline=deadline
    )

    def gate(*args, **kwargs):
        assert kwargs["quiet_period_elapsed"] is True
        return {
            "action": "process_business",
            "should_route": True,
            "should_wait": False,
            "category": "operational",
            "confidence": 0.8,
            "reasoning_summary": "静默期已结束，进入主流程补充信息。",
            "evidence": [],
            "errors": [],
        }

    monkeypatch.setattr(app, "run_wechaty_input_gate", gate)
    result = app.dispatch_pending_input_batch(
        runtime,
        batch,
        trace_id="trace_due",
        quiet_period_elapsed=True,
        trigger="quiet_period_elapsed",
    )

    assert result["input_status"] == PendingInputBatchStatus.COMPLETED.value
    assert len(runtime.received) == 1
    assert runtime.received[0].metadata["input_window"]["quiet_period_elapsed"] is True
