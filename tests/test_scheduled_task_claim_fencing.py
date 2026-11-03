"""定时任务领取凭证与延期预算的确定性回归，不使用模型或真实外发。"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from threading import Event

import pytest

from mahjong_agent_runtime import InMemoryAgentStore, InMemoryTraceRecorder, SQLiteAgentStore
from mahjong_agent_runtime.models import ScheduledTaskStatus
from mahjong_agent_runtime.scheduled_tasks import ScheduledAgentTaskScheduler, ScheduledTaskDeferred


STAMP = datetime.fromisoformat("2026-11-02T19:00:00+08:00")


@pytest.fixture(params=["memory", "sqlite", "sqlite_reopened"])
def task_store(request, tmp_path):
    """重开模式验证凭证与暂停意图确实持久化；所有库均为独立临时夹具。"""
    path = tmp_path / "task-fencing.sqlite3"
    store = InMemoryAgentStore() if request.param == "memory" else SQLiteAgentStore(path)
    opened = [store]

    def reopen():
        """只关闭并重开可再生的测试库，不接触既有运行库或原始证据。"""
        if request.param == "sqlite_reopened":
            opened[0]._connection.close()
            opened[0] = SQLiteAgentStore(path)
        return opened[0]

    task, _ = store.ensure_group_board_publish_task(room_id="synthetic-room", due_at=STAMP, trace_id="seed")
    yield task.task_id, reopen
    if isinstance(opened[0], SQLiteAgentStore):
        opened[0]._connection.close()


def credentials(claimed):
    """捕获领取快照的不可变双凭证，不从当前Store反取新代凭证。"""
    return {"expected_attempts": claimed.attempts, "expected_lease_until": claimed.lease_until}


def finish(store, task_id, claimed, operation):
    """让过期worker分别尝试成功、失败、延期，三条状态写入路径必须同样受保护。"""
    arguments = dict(trace_id="old-worker", at=STAMP + timedelta(seconds=3), **credentials(claimed))
    if operation == "complete":
        return store.complete_scheduled_task(task_id, **arguments)
    if operation == "fail":
        return store.fail_scheduled_task(task_id, error="stale transport error", **arguments)
    return store.defer_scheduled_task(task_id, reason="stale permission pause", **arguments)


@pytest.mark.parametrize("operation", ["complete", "fail", "defer"])
def test_expired_claim_cannot_update_recovered_claim(task_store, operation):
    """租约过期后新worker已领取，旧结果不得覆盖新状态、预算、due或审计。"""
    task_id, reopen = task_store
    store = reopen()
    first = store.claim_scheduled_task(task_id, at=STAMP, lease_seconds=1)
    store = reopen()
    second = store.claim_scheduled_task(task_id, at=STAMP + timedelta(seconds=2), lease_seconds=10)
    assert first.attempts == 1 and second.attempts == 2
    before = store.scheduled_tasks[task_id].to_dict()
    transitions = [item.to_dict() for item in store.transitions]
    _, transition = finish(store, task_id, first, operation)
    assert transition is None
    assert store.scheduled_tasks[task_id].to_dict() == before
    assert [item.to_dict() for item in store.transitions] == transitions
    _, accepted = store.complete_scheduled_task(task_id, trace_id="new-worker", **credentials(second))
    assert accepted is not None
    assert reopen().scheduled_tasks[task_id].status == ScheduledTaskStatus.COMPLETED


@pytest.mark.parametrize("operation", ["complete", "fail", "defer"])
def test_defer_reuses_retry_budget_but_never_reuses_claim_credential(task_store, operation):
    """变化的租约长度也不能碰撞：暂停前后attempt相同，时间水位仍区分旧新领取。"""
    task_id, reopen = task_store
    store = reopen()
    first = store.claim_scheduled_task(task_id, at=STAMP, lease_seconds=120)
    deferred, transition = store.defer_scheduled_task(task_id, trace_id="pause", reason="channel_disabled",
        retry_delay_seconds=1, at=STAMP, **credentials(first))
    assert transition is not None and deferred.attempts == 0
    assert deferred.status == ScheduledTaskStatus.PENDING and deferred.completed_at is None
    store = reopen()
    # 没有时间水位时 1秒后+119秒租约会恰好复用旧lease_until，掩盖attempt回退造成的碰撞。
    second = store.claim_scheduled_task(task_id, at=STAMP + timedelta(seconds=1), lease_seconds=119)
    assert first.attempts == second.attempts == 1
    assert second.lease_until > first.lease_until
    before = store.scheduled_tasks[task_id].to_dict()
    _, stale = finish(store, task_id, first, operation)
    assert stale is None and store.scheduled_tasks[task_id].to_dict() == before
    _, accepted = store.complete_scheduled_task(task_id, trace_id="current", **credentials(second))
    assert accepted is not None


def test_task_id_without_claim_credential_cannot_complete_fail_or_defer(task_store):
    """新合同没有无凭证旧调用回退；遗漏凭证必须在函数入口拒绝。"""
    task_id, reopen = task_store
    store = reopen()
    store.claim_scheduled_task(task_id, at=STAMP)
    before = store.scheduled_tasks[task_id].to_dict()
    for operation in (store.complete_scheduled_task, store.fail_scheduled_task, store.defer_scheduled_task):
        with pytest.raises(TypeError):
            operation(task_id, trace_id="missing-credential")
    assert store.scheduled_tasks[task_id].to_dict() == before


@pytest.mark.parametrize("outcome", ["success", "failure", "deferred"])
def test_scheduler_captures_credential_before_handler_and_ignores_stale_outcome(task_store, outcome):
    """handler内重领模拟阻塞旧worker恢复；Scheduler不能借可变对象拿到新worker凭证。"""
    task_id, reopen = task_store
    store = reopen()
    trace = InMemoryTraceRecorder()
    second_claim = []

    def handler(claimed, _trace_id):
        """只重领测试任务，不执行牌局操作、模型调用或通道发送。"""
        second_claim.append(store.claim_scheduled_task(task_id, at=claimed.lease_until + timedelta(seconds=1)))
        assert claimed.attempts == 1 and second_claim[0].attempts == 2
        if outcome == "failure":
            raise RuntimeError("old transport failure")
        if outcome == "deferred":
            raise ScheduledTaskDeferred("old permission pause")

    scheduler = ScheduledAgentTaskScheduler(store=store, handler=handler, trace_recorder=trace)
    assert scheduler.run_due_once(at=STAMP) == 1
    persisted = reopen().scheduled_tasks[task_id]
    assert persisted.status == ScheduledTaskStatus.PROCESSING and persisted.attempts == 2
    steps = [event.step for event in trace.events]
    expected = {"success": "completion", "failure": "failure", "deferred": "deferral"}[outcome]
    assert f"scheduled_agent_task_{expected}_ignored" in steps
    assert "scheduled_agent_task_completed" not in steps


def test_scheduler_transport_failure_retries_without_false_completion(task_store):
    """发送失败使用现有有限重试，下一次真实成功才落completed。"""
    task_id, reopen = task_store
    store = reopen()
    trace = InMemoryTraceRecorder()
    calls = []

    def handler(claimed, _trace_id):
        """受控失败一次，不联系任何真实渠道。"""
        calls.append(claimed.task_id)
        if len(calls) == 1:
            raise RuntimeError("retryable transport failure")

    scheduler = ScheduledAgentTaskScheduler(store=store, handler=handler, trace_recorder=trace, retry_delay_seconds=3)
    assert scheduler.run_due_once(at=STAMP) == 1
    pending = reopen().scheduled_tasks[task_id]
    assert pending.status == ScheduledTaskStatus.PENDING and pending.attempts == 1
    assert pending.completed_at is None and pending.due_at == STAMP + timedelta(seconds=3)
    assert "scheduled_agent_task_completed" not in [event.step for event in trace.events]
    scheduler.store = reopen()
    assert scheduler.run_due_once(at=pending.due_at) == 1
    assert reopen().scheduled_tasks[task_id].status == ScheduledTaskStatus.COMPLETED


def test_permission_pauses_preserve_budget_then_transport_failures_reach_cap(task_store):
    """多次权限暂停保留待办且不耗失败次数，恢复后的真实失败仍受上限约束。"""
    task_id, reopen = task_store
    store = reopen()
    trace = InMemoryTraceRecorder()
    mode = ["failure"]

    def handler(_claimed, _trace_id):
        """由测试切换暂停和错误类型，实际权限及外发开关始终不变。"""
        if mode[0] == "pause":
            raise ScheduledTaskDeferred("outbound_disabled", delay_seconds=2)
        raise RuntimeError("transport rejected")

    scheduler = ScheduledAgentTaskScheduler(store=store, handler=handler, trace_recorder=trace, max_attempts=2)
    scheduler.run_due_once(at=STAMP)
    mode[0] = "pause"
    for _ in range(5):
        scheduler.store = reopen()
        pending = scheduler.store.scheduled_tasks[task_id]
        assert pending.status == ScheduledTaskStatus.PENDING
        scheduler.run_due_once(at=pending.due_at)
        persisted = reopen().scheduled_tasks[task_id]
        assert persisted.status == ScheduledTaskStatus.PENDING and persisted.attempts == 1
        assert persisted.completed_at is None and persisted.last_error == "outbound_disabled"
    mode[0] = "failure"
    scheduler.store = reopen()
    scheduler.run_due_once(at=scheduler.store.scheduled_tasks[task_id].due_at)
    final = reopen().scheduled_tasks[task_id]
    assert final.status == ScheduledTaskStatus.FAILED and final.attempts == 2
    assert "scheduled_agent_task_completed" not in [event.step for event in trace.events]


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_concurrent_recovery_wins_before_old_worker_completion(backend, tmp_path):
    """用事件屏障固定交错，不用睡眠碰运气；SQLite不同连接共享同一事务事实源。"""
    path = tmp_path / "concurrent-fence.sqlite3"
    first_store = InMemoryAgentStore() if backend == "memory" else SQLiteAgentStore(path)
    second_store = first_store if backend == "memory" else SQLiteAgentStore(path)
    task, _ = first_store.ensure_group_board_publish_task(room_id="synthetic-room", due_at=STAMP, trace_id="seed")
    first = first_store.claim_scheduled_task(task.task_id, at=STAMP, lease_seconds=1)
    recovered = Event()

    def old_worker():
        """确保新领取先提交，再提交旧完成结果。"""
        assert recovered.wait(5)
        return first_store.complete_scheduled_task(task.task_id, trace_id="old", **credentials(first))

    def new_worker():
        """第二连接原子重领，提交后才放行旧worker。"""
        claim = second_store.claim_scheduled_task(task.task_id, at=STAMP + timedelta(seconds=2))
        assert claim is not None
        recovered.set()
        return claim

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            old_future = executor.submit(old_worker)
            new_future = executor.submit(new_worker)
            current = new_future.result(timeout=10)
            assert old_future.result(timeout=10)[1] is None
        assert first_store.scheduled_tasks[task.task_id].status == ScheduledTaskStatus.PROCESSING
        assert first_store.scheduled_tasks[task.task_id].attempts == current.attempts == 2
    finally:
        if backend == "sqlite":
            first_store._connection.close()
            second_store._connection.close()


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_room_successor_cannot_publish_ahead_of_current_unexpired_claim(backend, tmp_path):
    """固定两个worker交错：同群后继必须等旧领取结束，其他群仍可正常并行。"""
    path = tmp_path / "room-serial.sqlite3"
    first_store = InMemoryAgentStore() if backend == "memory" else SQLiteAgentStore(path)
    second_store = first_store if backend == "memory" else SQLiteAgentStore(path)
    task, _ = first_store.ensure_group_board_publish_task(room_id="synthetic-room", due_at=STAMP, trace_id="seed")
    current = first_store.claim_scheduled_task(task.task_id, at=STAMP, lease_seconds=10)
    successor, _ = first_store.ensure_group_board_publish_task(room_id="synthetic-room", due_at=STAMP, trace_id="change")
    other_room, _ = first_store.ensure_group_board_publish_task(room_id="other-room", due_at=STAMP, trace_id="other")
    attempt_finished = Event()

    def successor_worker():
        """先尝试后继领取并确认被阻止，不能提前发布。"""
        blocked = second_store.claim_scheduled_task(successor.task_id, at=STAMP + timedelta(seconds=1))
        independent = second_store.claim_scheduled_task(other_room.task_id, at=STAMP + timedelta(seconds=1))
        attempt_finished.set()
        return blocked, independent

    def current_worker():
        """等待后继尝试结束后再完成旧领取，避免测试靠线程概率通过。"""
        assert attempt_finished.wait(5)
        return first_store.complete_scheduled_task(task.task_id, trace_id="current", **credentials(current))

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            later = executor.submit(successor_worker)
            earlier = executor.submit(current_worker)
            blocked, independent = later.result(timeout=10)
            assert blocked is None and independent is not None
            assert earlier.result(timeout=10)[1] is not None
        assert second_store.claim_scheduled_task(successor.task_id, at=STAMP + timedelta(seconds=2)) is not None
    finally:
        if backend == "sqlite":
            first_store._connection.close()
            second_store._connection.close()


def test_expired_room_claim_does_not_block_successor_after_restart(task_store):
    """过期旧租约不永久堵住整个群，重启读取的后继仍可接管刷新责任。"""
    task_id, reopen = task_store
    store = reopen()
    store.claim_scheduled_task(task_id, at=STAMP, lease_seconds=1)
    successor, _ = store.ensure_group_board_publish_task(room_id="synthetic-room", due_at=STAMP, trace_id="change")
    store = reopen()
    assert store.claim_scheduled_task(successor.task_id, at=STAMP + timedelta(seconds=2)) is not None


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.0, float("inf"), float("nan"), "60", None])
def test_scheduler_lease_budget_requires_a_positive_integer(value):
    """租约配置不能静默截断、接受布尔值或产生不可到期的非有限数。"""
    with pytest.raises(ValueError, match="lease_seconds"):
        ScheduledAgentTaskScheduler(store=InMemoryAgentStore(), handler=lambda *_: None,
            trace_recorder=InMemoryTraceRecorder(), lease_seconds=value)


@pytest.mark.parametrize("lease_seconds", [1, 180, 267])
def test_scheduler_passes_its_configured_lease_budget_to_the_store(task_store, lease_seconds):
    """处理器收到真实配置的租约；默认审查预算较长时无需依赖短Store缺省值。"""
    task_id, reopen = task_store
    observed = []

    def handler(claimed, _trace_id):
        """只观察领取预算，不调用模型、不修改业务对象。"""
        observed.append(claimed.lease_until)

    scheduler = ScheduledAgentTaskScheduler(store=reopen(), handler=handler,
        trace_recorder=InMemoryTraceRecorder(), lease_seconds=lease_seconds)
    assert scheduler.run_task_once(task_id, at=STAMP) == 1
    assert observed == [STAMP + timedelta(seconds=lease_seconds)]
    assert reopen().scheduled_tasks[task_id].status == ScheduledTaskStatus.COMPLETED
