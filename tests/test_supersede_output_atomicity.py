"""替换待发输出必须先取得事实源写锁，不能覆盖另一连接已确认的交付事实。"""
from __future__ import annotations

from queue import Queue
import sqlite3
from threading import Event, Thread

import pytest

from mahjong_agent_runtime import AgentRuntime, CustomerProfile, InMemoryAgentStore, SQLiteAgentStore, StaticAgentClient, UserMessage
from mahjong_agent_runtime.models import ConversationRole, ScheduledAgentTask, now


def seed_invites(store):
    """合法创建同会话下的待审批邀约，供输入接收与发送状态回调竞争。"""
    for customer_id in ("organizer", "candidate", "already_sent"):
        store.upsert_customer(CustomerProfile(customer_id=customer_id, display_name=customer_id))
    game, _ = store.create_game(conversation_id="chat", organizer_id="organizer", organizer_name="发起方",
        requirement={"game_type": "national_standard_mahjong", "stake": "1", "start_time_kind": "asap_when_full"},
        known_players=[{"customer_id": "organizer", "display_name": "发起方"}], trace_id="initial-owner-request")
    drafts, _ = store.create_invite_drafts(game_id=game.game_id, invitations=[
        {"customer_id": "candidate", "display_name": "候选", "message_text": "来吗？"},
        {"customer_id": "already_sent", "display_name": "另一位", "message_text": "来吗？"},
    ], trace_id="initial-owner-request")
    return drafts


def new_input():
    """同一发起方补充条件，不冒用候选人身份以避开草稿替换。"""
    return UserMessage(conversation_id="chat", sender_id="organizer", sender_name="发起方",
        text="这次改成两块的", message_id="new-owner-input")


def start_worker(target, errors):
    """把线程中的异常传回测试线程，避免后台断言失败被当作测试通过。"""
    def run():
        try:
            target()
        except BaseException as exc:
            errors.put(exc)
    thread = Thread(target=run)
    thread.start()
    return thread


def assert_workers_finished(threads, errors):
    """有界等待已经解除屏障的线程，并在主线程重抛后台异常。"""
    for thread in threads:
        thread.join(5)
    assert all(not thread.is_alive() for thread in threads), "worker did not finish after release"
    if not errors.empty():
        raise AssertionError("worker failed") from errors.get_nowait()


@pytest.fixture
def sqlite_pair(tmp_path):
    """同一数据库使用独立连接和独立进程内锁，检验SQLite事务而非共享RLock。"""
    path = tmp_path / "supersede.sqlite3"
    stores = [SQLiteAgentStore(path), SQLiteAgentStore(path)]
    yield stores
    for store in stores:
        store._connection.close()


def test_sqlite_receive_supersede_owns_write_lock_before_reading_pending_invite(sqlite_pair, monkeypatch):
    """A先替换时，B不能把已经被A读取的草稿先提交为sent再遭旧快照覆盖。"""
    first, second = sqlite_pair
    drafts = seed_invites(first)
    target_id = drafts[0].draft_id
    read, release = Event(), Event()
    errors = Queue()
    observed = {}
    client = StaticAgentClient(outputs=[])
    runtime = AgentRuntime(llm_client=client, store=first)
    original = type(first)._save_invite

    def pause_before_save(self, draft):
        if self is first and draft.draft_id == target_id and draft.status.value == "superseded":
            observed["in_transaction"] = self._connection.in_transaction
            read.set()
            if not release.wait(5):
                raise RuntimeError("supersede barrier was not released")
        return original(self, draft)

    monkeypatch.setattr(type(first), "_save_invite", pause_before_save)
    received = {}
    worker = start_worker(lambda: received.update(runtime.receive_user_message(new_input(), trace_id="new-input")), errors)
    try:
        assert read.wait(5), "normal receive did not reach pending invite supersession"
        assert observed["in_transaction"] is True
        # 接收和废弃旧输出现在共用一个事务；其他连接不能提前看到半完成的 receipt。
        assert second.conversation_version("chat") == 0
        assert second.find_conversation_turn("chat", message_id="new-owner-input") is None
        second._connection.execute("PRAGMA busy_timeout=0")
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            second.update_invite_delivery_status(draft_id=target_id, status="sent", trace_id="competing-delivery",
                reason="deterministic temporary database delivery callback")
    finally:
        release.set()
        assert_workers_finished([worker], errors)
    assert received["accepted"] is True
    assert second.conversation_version("chat") == 1
    assert second.find_conversation_turn("chat", message_id="new-owner-input").content == "这次改成两块的"
    assert second.invite_drafts[target_id].status.value == "superseded"
    with pytest.raises(ValueError, match="superseded->sent"):
        second.update_invite_delivery_status(draft_id=target_id, status="sent", trace_id="late-delivery",
            reason="retry cannot resurrect an obsolete invitation")
    assert not any(t.to_status == "sent" for t in second.transitions if t.entity_id == target_id)
    assert client.calls == []


def test_sqlite_delivery_commit_wins_and_receive_rechecks_sent_invite(sqlite_pair, monkeypatch):
    """B先写成功状态时，A等写锁后必须重读sent，不能沿用pending快照。"""
    first, second = sqlite_pair
    drafts = seed_invites(first)
    target_id = drafts[0].draft_id
    saved, release_delivery, receive_begin = Event(), Event(), Event()
    errors = Queue()
    observed = {}
    original = type(first)._save_invite
    client = StaticAgentClient(outputs=[])
    runtime = AgentRuntime(llm_client=client, store=first)

    def pause_after_save(self, draft):
        result = original(self, draft)
        if self is second and draft.draft_id == target_id and draft.status.value == "sent":
            observed["in_transaction"] = self._connection.in_transaction
            saved.set()
            if not release_delivery.wait(5):
                raise RuntimeError("delivery commit barrier was not released")
        return result

    def observe_receive_transaction(statement):
        if statement.strip().upper() == "BEGIN IMMEDIATE":
            receive_begin.set()

    monkeypatch.setattr(type(first), "_save_invite", pause_after_save)
    first._connection.set_trace_callback(observe_receive_transaction)
    delivered = {}
    received = {}
    threads = []
    try:
        def finish_delivery():
            draft, _ = second.update_invite_delivery_status(draft_id=target_id, status="sent", trace_id="delivery-first",
                reason="deterministic temporary database delivery callback")
            delivered["status"] = draft.status.value
        threads.append(start_worker(finish_delivery, errors))
        assert saved.wait(5), "delivery did not write before its commit barrier"
        assert observed["in_transaction"] is True
        threads.append(start_worker(lambda: received.update(runtime.receive_user_message(new_input(), trace_id="new-input")), errors))
        assert receive_begin.wait(5), "normal receive did not begin its write transaction"
    finally:
        release_delivery.set()
        assert_workers_finished(threads, errors)
        first._connection.set_trace_callback(None)
    assert delivered["status"] == "sent" and received["accepted"] is True
    assert first.invite_drafts[target_id].status.value == "sent"
    pairs = [(t.from_status, t.to_status) for t in first.transitions if t.entity_id == target_id]
    assert pairs == [(None, "pending_approval"), ("pending_approval", "sent")]
    assert client.calls == []


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_new_input_only_supersedes_pending_outputs_and_preserves_started_or_completed_delivery(backend, tmp_path):
    """两事实源只替换尚未交付的输出，已发送、已开始和已展示历史保持不变。"""
    store = InMemoryAgentStore() if backend == "memory" else SQLiteAgentStore(tmp_path / "states.sqlite3")
    try:
        pending, sent = seed_invites(store)
        store.update_invite_delivery_status(draft_id=sent.draft_id, status="sent", trace_id="already-sent",
            reason="deterministic temporary database delivery callback")
        for trace_id in ("pending-reply", "dispatching-reply", "delivered-reply"):
            store.append_assistant_turn("chat", trace_id, trace_id,
                metadata={"delivery_status": "pending_operator_send", "conversation_version": 0})
        assert store.claim_assistant_reply_delivery("chat", trace_id="dispatching-reply")
        assert store.claim_assistant_reply_delivery("chat", trace_id="delivered-reply")
        assert store.mark_assistant_reply_delivered("chat", trace_id="delivered-reply")
        task = ScheduledAgentTask(task_id="finished-follow-up", task_type="conversation_follow_up",
            aggregate_type="conversation", aggregate_id="finished-goal", conversation_id="chat",
            subject_id="organizer", subject_name="发起方", due_at=now(), idempotency_key="finished-follow-up")
        store.save_follow_up_task(task, trace_id="follow-up-setup")
        assert store.claim_scheduled_task(task.task_id, at=task.due_at, lease_seconds=30)
        store.complete_scheduled_task(task.task_id, trace_id="follow-up-completed")
        completed_task = store.scheduled_tasks[task.task_id].to_dict()
        assert completed_task["status"] == "completed"
        client = StaticAgentClient(outputs=[])
        runtime = AgentRuntime(llm_client=client, store=store)
        message = new_input()
        receipt = runtime.receive_user_message(message, trace_id="new-input")
        assert receipt["accepted"] is True
        assert store.invite_drafts[pending.draft_id].status.value == "superseded"
        assert store.invite_drafts[sent.draft_id].status.value == "sent"
        replies = {turn.trace_id: turn for turn in store.recent_turns("chat") if turn.role == ConversationRole.ASSISTANT}
        assert {key: turn.metadata["delivery_status"] for key, turn in replies.items()} == {
            "pending-reply": "superseded", "dispatching-reply": "dispatching", "delivered-reply": "delivered"}
        assert all(turn.content == trace_id for trace_id, turn in replies.items())
        assert store.scheduled_tasks[task.task_id].to_dict() == completed_task
        count = len(store.transitions)
        duplicate = runtime.receive_user_message(message, trace_id="duplicate-input")
        assert duplicate["accepted"] is False and len(store.transitions) == count
        assert client.calls == []
    finally:
        if isinstance(store, SQLiteAgentStore):
            store._connection.close()
