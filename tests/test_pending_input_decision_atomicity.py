"""批次分类与完成必须保持原始片段、精确版本和跨连接提交的原子边界。"""

from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from threading import Event

import pytest

from mahjong_agent_runtime import InMemoryAgentStore, SQLiteAgentStore, UserMessage
from mahjong_agent_runtime.models import PendingInputBatchStatus, now


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    """两种事实源都验证过期结果不能回写新输入；每例使用独立运行数据。"""
    value = InMemoryAgentStore() if request.param == "memory" else SQLiteAgentStore(tmp_path / "pending.sqlite3")
    yield value
    if request.param == "sqlite":
        value._connection.close()


def _receive(store, message_id):
    """通过原 Store 接收接口持久化一条可区分的用户原文。"""
    message = UserMessage(
        conversation_id="local-api-chat", sender_id="customer", sender_name="客户",
        text={"m1": "先组晚上", "m2": "改成下午，只查局"}[message_id],
        message_id=message_id, metadata={"channel": "api"},
    )
    batch, _, accepted = store.upsert_pending_input_fragment(
        message, trace_id=f"receive-{message_id}", quiet_deadline=now() + timedelta(seconds=20),
    )
    assert accepted
    return batch


def _write_result(store, batch, operation):
    """两种读改写入口共用相同的旧批次身份与版本，不能从调用方提升版本。"""
    if operation == "decision":
        return store.record_pending_input_decision(
            batch_id=batch.batch_id, expected_version=batch.version, decision={"action": "process_business"},
        )
    result, _ = store.finish_pending_input_batch(
        batch_id=batch.batch_id, expected_version=batch.version, status=PendingInputBatchStatus.COMPLETED,
        trace_id="old-result", decision={"agent_result": {"final_reply": "旧回复"}},
    )
    return result


@pytest.mark.parametrize("operation", ["decision", "finish"])
def test_exact_pending_version_can_be_updated_without_changing_fragments(store, operation):
    batch = _receive(store, "m1")
    result = _write_result(store, batch, operation)
    assert result is not None and result.version == batch.version
    current = store.pending_input_batch("local-api-chat", "customer")
    assert current.fragments == batch.fragments
    assert current.status == (PendingInputBatchStatus.COMPLETED if operation == "finish" else PendingInputBatchStatus.PENDING)
    assert store.conversation_version("local-api-chat") == 1


@pytest.mark.parametrize("operation", ["decision", "finish"])
def test_stale_pending_result_cannot_overwrite_already_received_new_message(store, operation):
    old = _receive(store, "m1")
    latest = _receive(store, "m2")
    before = latest.to_dict()
    assert _write_result(store, old, operation) is None
    assert store.pending_input_batch("local-api-chat", "customer").to_dict() == before
    assert [fragment["message_id"] for fragment in latest.fragments] == ["m1", "m2"]
    assert store.conversation_version("local-api-chat") == 2


@pytest.mark.parametrize("operation", ["decision", "finish"])
def test_sqlite_pending_snapshot_cannot_overwrite_another_connections_new_input(tmp_path, monkeypatch, operation):
    first = SQLiteAgentStore(tmp_path / "two-connections.sqlite3")
    second = SQLiteAgentStore(first.path)
    old = _receive(first, "m1")
    snapshot_read = Event()
    release_snapshot = Event()
    incoming_begin = Event()
    incoming_finished = Event()
    original_save = SQLiteAgentStore._save_pending_input_batch

    def pause_after_read(self, batch):
        # 调用本保存助手时，目标方法已经读取并修改了旧快照，但尚未执行第一次写入。
        if self is first:
            snapshot_read.set()
            if not release_snapshot.wait(5):
                raise RuntimeError("pending result commit barrier was not released")
        return original_save(self, batch)

    def observe_incoming(statement):
        if statement.strip().upper() == "BEGIN IMMEDIATE":
            incoming_begin.set()

    def receive_new_input():
        result = _receive(second, "m2")
        incoming_finished.set()
        return result

    monkeypatch.setattr(SQLiteAgentStore, "_save_pending_input_batch", pause_after_read)
    second._connection.set_trace_callback(observe_incoming)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            old_future = pool.submit(_write_result, first, old, operation)
            try:
                assert snapshot_read.wait(5)
                # 原实现只用连接上下文，SELECT 后仍无事务；本断言使旧实现稳定失败。
                assert first._connection.in_transaction
                new_future = pool.submit(receive_new_input)
                assert incoming_begin.wait(5)
                assert not incoming_finished.is_set()
            finally:
                release_snapshot.set()
            assert old_future.result(timeout=5) is not None
            latest = new_future.result(timeout=5)
        current = second.pending_input_batch("local-api-chat", "customer")
        assert current.to_dict() == latest.to_dict()
        assert current.status == PendingInputBatchStatus.PENDING
        if operation == "decision":
            assert current.batch_id == old.batch_id and current.version == 2
            assert [item["message_id"] for item in current.fragments] == ["m1", "m2"]
        else:
            # 旧批次已原子完成后，后来输入应创建新待处理批次，不能被旧完成状态吞掉。
            assert current.batch_id != old.batch_id
            assert [item["message_id"] for item in current.fragments] == ["m2"]
        assert second.conversation_version("local-api-chat") == 2
    finally:
        release_snapshot.set()
        second._connection.set_trace_callback(None)
        first._connection.close()
        second._connection.close()


@pytest.mark.parametrize("operation", ["decision", "finish"])
def test_sqlite_pending_write_failure_rolls_back_batch_and_audit(tmp_path, monkeypatch, operation):
    store = SQLiteAgentStore(tmp_path / "rollback.sqlite3")
    old = _receive(store, "m1")
    before = store.pending_input_batch("local-api-chat", "customer").to_dict()
    transitions = [item.to_dict() for item in store.transitions]
    original_save = SQLiteAgentStore._save_pending_input_batch

    def fail_after_save(self, batch):
        original_save(self, batch)
        raise RuntimeError("injected pending write failure")

    monkeypatch.setattr(SQLiteAgentStore, "_save_pending_input_batch", fail_after_save)
    try:
        with pytest.raises(RuntimeError, match="injected pending write failure"):
            _write_result(store, old, operation)
        assert store.pending_input_batch("local-api-chat", "customer").to_dict() == before
        assert [item.to_dict() for item in store.transitions] == transitions
        assert not store._connection.in_transaction
    finally:
        store._connection.close()
