"""用事件屏障和双连接验证复查版本核验到草稿创建的短接收锁边界。"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import timedelta
import json
from queue import Queue
from threading import Event, Lock, Thread, get_ident

import pytest

from mahjong_agent_runtime import (
    AgentAction, AgentRuntime, InMemoryAgentStore, InMemoryTraceRecorder, SQLiteAgentStore,
    StaticAgentClient, ToolCall, UserMessage,
)
from mahjong_agent_runtime.coordination import InProcessCoordinationManager
from mahjong_agent_runtime.follow_up import handle_follow_up_task
from mahjong_agent_runtime.models import now


@pytest.fixture(params=["memory", "sqlite"])
def runtime_pair(request, tmp_path):
    """SQLite 使用独立连接/协调器，Memory 使用同一事实源和共享协调器。"""
    first = InMemoryAgentStore() if request.param == "memory" else SQLiteAgentStore(tmp_path / "followup-fence.sqlite3")
    second = first if request.param == "memory" else SQLiteAgentStore(first.path)
    manager = InProcessCoordinationManager() if request.param == "memory" else None
    action = AgentAction(goal="复查本次进展", objective_status="completed", reasoning_summary="机制测试",
        reply_to_user="有新进展了。", stop_reason={"can_stop": True, "why": "复查完成", "pending_work": [], "depends_on_tool_results": False})
    writer = AgentRuntime(llm_client=StaticAgentClient([json.dumps(action.to_dict(), ensure_ascii=False)]),
        store=first, trace_recorder=InMemoryTraceRecorder(), coordination_manager=manager)
    receiver = AgentRuntime(llm_client=StaticAgentClient([]), store=second, coordination_manager=manager)
    yield writer, receiver
    if request.param == "sqlite":
        first._connection.close()
        second._connection.close()


def test_new_input_between_follow_up_check_and_insert_cannot_leave_live_old_draft(runtime_pair, monkeypatch):
    """A核验后卡在插入前，B真实接收；短锁保证B清理能看见A先落的草稿，不让旧输出复活。"""
    writer, receiver = runtime_pair
    stamp = now()
    message = UserMessage(conversation_id="follow-up-race", sender_id="customer", sender_name="客户",
        text="三分钟后再看看，有进展告诉我", message_id="initial-input", sent_at=stamp)
    assert writer.receive_user_message(message, trace_id="initial-input")["conversation_version"] == 1
    due = stamp + timedelta(minutes=3)
    scheduled = writer.tool_gateway.execute(ToolCall("schedule_follow_up", {
        "goal": "复查当前进展并按需要反馈", "waiting_for": "players", "due_at": due.isoformat(),
        "expires_at": (stamp + timedelta(hours=1)).isoformat(), "review_budget": 1}),
        conversation_id=message.conversation_id, sender_id=message.sender_id, sender_name=message.sender_name,
        trace_id="schedule", step_index=1, source_message_id=message.message_id)
    assert scheduled.called and scheduled.allowed and not scheduled.error
    task = writer.store.claim_scheduled_task(scheduled.result["follow_up"]["task_id"], at=due)
    assert task is not None

    before_insert, release_insert, receive_attempted, received = Event(), Event(), Event(), Event()
    ownership_guard = Lock()
    receipt_owners = set()
    observed = {}
    results = {}
    errors = Queue()
    original_receipt = AgentRuntime.input_receipt_lock
    original_create = type(writer.store).create_outbound_message_drafts

    @contextmanager
    def tracked_receipt(self, conversation_id):
        """只观测真实协调锁持有者，仍使用生产锁保证跨连接顺序。"""
        if self is receiver:
            receive_attempted.set()
        with original_receipt(self, conversation_id):
            with ownership_guard:
                receipt_owners.add(get_ident())
            try:
                yield
            finally:
                with ownership_guard:
                    receipt_owners.remove(get_ident())

    def pause_before_insert(self, *, conversation_id, drafts, trace_id):
        """屏障位于版本比较之后、Store事务之前，覆盖先前返回前插入测试未覆盖的窗口。"""
        if self is writer.store and any(item.get("purpose") == "follow_up" for item in drafts):
            with ownership_guard:
                observed["receipt_owned_at_insert"] = get_ident() in receipt_owners
            before_insert.set()
            if not release_insert.wait(5):
                raise AssertionError("follow-up insert barrier was not released")
        return original_create(self, conversation_id=conversation_id, drafts=drafts, trace_id=trace_id)

    monkeypatch.setattr(AgentRuntime, "input_receipt_lock", tracked_receipt)
    monkeypatch.setattr(type(writer.store), "create_outbound_message_drafts", pause_before_insert)

    def follow_up_worker():
        try:
            results["follow_up"] = handle_follow_up_task(writer, task, "follow-up-race", at=due)
        except BaseException as exc:
            errors.put(exc)

    def input_worker():
        try:
            newer = UserMessage(conversation_id=message.conversation_id, sender_id=message.sender_id, sender_name="客户",
                text="不用继续了", message_id="new-input", sent_at=due + timedelta(seconds=1))
            results["receipt"] = receiver.receive_user_message(newer, trace_id="new-input")
            received.set()
        except BaseException as exc:
            errors.put(exc)

    first = Thread(target=follow_up_worker, name="follow-up-writer")
    second = Thread(target=input_worker, name="new-input-receiver")
    first.start()
    try:
        assert before_insert.wait(5), "follow-up did not reach the post-check insert boundary"
        second.start()
        assert receive_attempted.wait(5)
        # 旧实现无fence时确定让B先接收完成，再放A写入以复现旧草稿复活；
        # 修复后B需等同一短锁，先放A提交，B随后必须能清理已落库的旧草稿。
        if not observed["receipt_owned_at_insert"]:
            assert received.wait(5), "unfenced receiver did not finish before old draft creation"
    finally:
        release_insert.set()
        first.join(5)
        if second.ident is not None:
            second.join(5)
    assert not first.is_alive() and not second.is_alive(), "race workers did not finish"
    if not errors.empty():
        raise AssertionError("worker failed") from errors.get_nowait()
    assert results["receipt"]["accepted"] is True
    assert results["receipt"]["conversation_version"] == receiver.store.conversation_version(message.conversation_id) == 2
    drafts = list(receiver.store.outbound_message_drafts.values())
    observed["draft_statuses"] = [draft.status.value for draft in drafts]
    assert len(drafts) == 1 and all(draft.status.value == "superseded" for draft in drafts), observed
    assert observed["receipt_owned_at_insert"] is True
    assert len(writer.llm_client.calls) == 1 and receiver.llm_client.calls == []
    if isinstance(receiver.store, SQLiteAgentStore):
        reopened = SQLiteAgentStore(receiver.store.path)
        try:
            assert [draft.status.value for draft in reopened.outbound_message_drafts.values()] == ["superseded"]
        finally:
            reopened._connection.close()
