"""批次耗时必须推进各任务的领取与结算时钟，不依赖真实等待或模型调用。"""

from datetime import datetime, timedelta

import pytest

from mahjong_agent_runtime import InMemoryAgentStore, InMemoryTraceRecorder, SQLiteAgentStore
from mahjong_agent_runtime.group_chat import BoardPublicationError
from mahjong_agent_runtime.models import ScheduledTaskStatus
from mahjong_agent_runtime.scheduled_tasks import ScheduledAgentTaskScheduler, ScheduledTaskDeferred


STAMP = datetime.fromisoformat("2026-11-02T19:00:00+08:00")


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    value = InMemoryAgentStore() if request.param == "memory" else SQLiteAgentStore(tmp_path / "batch-clock.sqlite3")
    yield value
    if isinstance(value, SQLiteAgentStore):
        value._connection.close()


def test_each_production_batch_claim_gets_full_lease_after_earlier_review_time(store):
    clock = [STAMP]
    for room in ("a", "b", "c"):
        store.ensure_group_board_publish_task(room_id=room, due_at=STAMP, trace_id="seed")
    observed = []

    def handler(task, _trace_id):
        started = clock[0]
        assert task.lease_until == started + timedelta(seconds=180)
        clock[0] += timedelta(seconds=70)
        # 与生产外发guard相同的过期边界；每次审查70秒均应处于自己的180秒租约内。
        if task.lease_until <= clock[0]:
            raise BoardPublicationError("publication claim expired before send")
        observed.append((task.aggregate_id, started, task.lease_until))

    scheduler = ScheduledAgentTaskScheduler(store=store, handler=handler, trace_recorder=InMemoryTraceRecorder(),
        clock=lambda: clock[0])
    assert scheduler.run_due_once() == 3
    assert [item[1] for item in observed] == [STAMP + timedelta(seconds=value) for value in (0, 70, 140)]
    for index, task in enumerate(sorted(store.scheduled_tasks.values(), key=lambda item: item.aggregate_id), start=1):
        assert task.status == ScheduledTaskStatus.COMPLETED and task.attempts == 1
        assert task.completed_at == STAMP + timedelta(seconds=index * 70)


@pytest.mark.parametrize("outcome", ["failure", "deferred"])
def test_production_retry_or_pause_due_time_starts_after_handler_finishes(store, outcome):
    clock = [STAMP]
    task, _ = store.ensure_group_board_publish_task(room_id="a", due_at=STAMP, trace_id="seed")

    def handler(_task, _trace_id):
        clock[0] += timedelta(seconds=80)
        if outcome == "deferred":
            raise ScheduledTaskDeferred("permission_paused", delay_seconds=7)
        raise RuntimeError("transport_failed")

    scheduler = ScheduledAgentTaskScheduler(store=store, handler=handler, trace_recorder=InMemoryTraceRecorder(),
        clock=lambda: clock[0], retry_delay_seconds=7)
    assert scheduler.run_task_once(task.task_id) == 1
    persisted = store.scheduled_tasks[task.task_id]
    assert persisted.status == ScheduledTaskStatus.PENDING
    assert persisted.updated_at == STAMP + timedelta(seconds=80)
    assert persisted.due_at == STAMP + timedelta(seconds=87)
    assert persisted.attempts == (0 if outcome == "deferred" else 1)


@pytest.mark.parametrize("entry", ["batch", "single"])
def test_explicit_at_remains_fixed_simulation_time_without_reading_wall_clock(store, entry):
    task, _ = store.ensure_group_board_publish_task(room_id="a", due_at=STAMP, trace_id="seed")

    def forbidden_clock():
        raise AssertionError("explicit-at simulation must not read the wall clock")

    def handler(claimed, _trace_id):
        assert claimed.lease_until == STAMP + timedelta(seconds=180)

    scheduler = ScheduledAgentTaskScheduler(store=store, handler=handler, trace_recorder=InMemoryTraceRecorder(),
        clock=forbidden_clock)
    assert (scheduler.run_due_once(at=STAMP) if entry == "batch"
            else scheduler.run_task_once(task.task_id, at=STAMP)) == 1
    assert store.scheduled_tasks[task.task_id].completed_at == STAMP
