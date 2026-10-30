"""验证原始输入接收、版本更新和本地交付确认的原子与幂等边界。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import pytest

from mahjong_agent_runtime import InMemoryAgentStore, SQLiteAgentStore, UserMessage
from mahjong_agent_runtime.models import ConversationRole


EMPTY_SUPERSESSION_COUNTS = {"invite_drafts": 0, "outbound_message_drafts": 0, "assistant_replies": 0}


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path: Path):
    """让相同接收与交付合同在两种事实源上执行。"""
    current = InMemoryAgentStore() if request.param == "memory" else SQLiteAgentStore(tmp_path / "receipts.sqlite3")
    yield current
    if isinstance(current, SQLiteAgentStore):
        current._connection.close()


def user_message(message_id: str = "original-message", **overrides) -> UserMessage:
    """生成可核对原文和时间的输入，不经过模型或真实通道。"""
    values = {
        "conversation_id": "conversation", "sender_id": "customer", "sender_name": "客户",
        "text": "下午一块的，时间都行", "message_id": message_id,
        "sent_at": datetime.fromisoformat("2026-10-29T15:01:02.123456+08:00"),
        "metadata": {"channel": "manual"},
    }
    values.update(overrides)
    return UserMessage(**values)


def test_accept_user_input_is_idempotent_and_preserves_raw_turn(store) -> None:
    message = user_message()
    version, transition, accepted, counts, superseded = store.accept_user_input(message, trace_id="first-receipt", message_key="stable-key")
    assert (version, accepted) == (1, True)
    assert counts == EMPTY_SUPERSESSION_COUNTS and superseded == []
    assert transition is not None
    assert transition.entity_type == "conversation_version"
    assert transition.from_status == "0" and transition.to_status == "1"
    assert transition.reason == "user_message_received"
    duplicate = user_message(text="重试的内容不能覆盖原话")
    assert store.accept_user_input(duplicate, trace_id="duplicate", message_key="stable-key") == (1, None, False, EMPTY_SUPERSESSION_COUNTS, [])
    turns = store.recent_turns(message.conversation_id)
    assert len(turns) == 1
    turn = turns[0]
    assert turn.role == ConversationRole.USER
    assert turn.content == message.text and turn.occurred_at == message.sent_at
    assert turn.trace_id == "first-receipt"
    assert turn.sender_id == message.sender_id and turn.sender_name == message.sender_name
    assert turn.metadata == {
        "channel": "manual", "source_message_id": message.message_id,
        "input_receipt_key": "stable-key", "conversation_version": 1,
        "input_receipt_preparation": {}, "input_receipt_hooks_dispatched": False,
    }
    assert store.conversation_version(message.conversation_id) == 1


def test_distinct_receipts_advance_versions_without_rewriting_existing_records(store) -> None:
    store.append_user_turn(user_message("legacy", text="旧原始记录"), "legacy-trace")
    first = store.accept_user_input(user_message("first"), trace_id="first-trace", message_key="key-1")
    second = store.accept_user_input(user_message("second", text="再补一句"), trace_id="second-trace", message_key="key-2")
    assert first[0] == 1 and second[0] == 2
    assert store.accept_user_input(user_message("first"), trace_id="first-retry", message_key="key-1") == (1, None, False, EMPTY_SUPERSESSION_COUNTS, [])
    turns = store.recent_turns("conversation")
    assert [turn.trace_id for turn in turns] == ["legacy-trace", "first-trace", "second-trace"]
    assert turns[0].content == "旧原始记录" and "input_receipt_key" not in turns[0].metadata
    assert store.conversation_version("conversation") == 2


def test_receipt_key_is_scoped_by_conversation_and_sender(store) -> None:
    assert store.accept_user_input(user_message(), trace_id="a", message_key="same-key")[0] == 1
    assert store.accept_user_input(user_message(sender_id="another"), trace_id="b", message_key="same-key")[0] == 2
    assert store.accept_user_input(user_message(conversation_id="another"), trace_id="c", message_key="same-key")[0] == 1
    assert len(store.recent_turns("conversation")) == 2
    assert len(store.recent_turns("another")) == 1


def test_input_metadata_cannot_forge_receipt_version_or_task_context(store) -> None:
    context, _ = store.activate_task_context(
        conversation_id="conversation", customer_id="customer", trace_id="prepare",
        activity_at=user_message().sent_at, started_at=user_message().sent_at, reason="new_task",
        force_new=False, archive_previous=False,
    )
    store.accept_user_input(user_message("first"), trace_id="first", message_key="first-key")
    supplied = {
        "input_receipt_key": "first-key", "conversation_version": 9000,
        "task_context_id": "forged-task", "_trusted_source_task_context_id": "forged-trusted-task",
        "source_message_id": "forged-source", "custom": "保留",
    }
    message = user_message("second", metadata=supplied)
    version, _, accepted, counts, superseded = store.accept_user_input(message, trace_id="second", message_key="second-key")
    assert (version, accepted) == (2, True)
    assert counts == EMPTY_SUPERSESSION_COUNTS and superseded == []
    turn = store.recent_turns("conversation")[-1]
    assert turn.metadata["input_receipt_key"] == "second-key"
    assert turn.metadata["conversation_version"] == 2
    assert turn.metadata["task_context_id"] == context.task_context_id
    assert "_trusted_source_task_context_id" not in turn.metadata
    assert turn.metadata["source_message_id"] == "second"
    assert turn.metadata["custom"] == "保留"
    assert message.metadata == supplied
    no_task = user_message("no-task", conversation_id="no-task", metadata=supplied)
    store.accept_user_input(no_task, trace_id="no-task", message_key="no-task-key")
    assert "task_context_id" not in store.recent_turns("no-task")[0].metadata


def test_receipt_without_message_id_keeps_stable_source_reference(store) -> None:
    store.accept_user_input(user_message(""), trace_id="fallback", message_key="stable-fallback")
    assert store.recent_turns("conversation")[0].metadata["source_message_id"] == "stable-fallback"


def test_empty_receipt_key_is_rejected_without_side_effects(store) -> None:
    with pytest.raises(ValueError, match="message_key"):
        store.accept_user_input(user_message(), trace_id="empty-key", message_key=" ")
    assert store.recent_turns("conversation") == []
    assert store.conversation_version("conversation") == 0


def test_sqlite_receipt_is_atomic_and_idempotent_across_connections(tmp_path: Path) -> None:
    path = tmp_path / "parallel-receipts.sqlite3"
    stores = [SQLiteAgentStore(path) for _ in range(4)]
    try:
        def accept(index: int):
            item = index % 12
            version, _, added, counts, superseded = stores[index % len(stores)].accept_user_input(
                user_message(f"message-{item}", text=f"原话{item}"),
                trace_id=f"receipt-{index}", message_key=f"key-{item}",
            )
            assert counts == EMPTY_SUPERSESSION_COUNTS and superseded == []
            return item, version, added

        with ThreadPoolExecutor(max_workers=8) as pool:
            receipts = list(pool.map(accept, range(48)))
        assert sum(added for _, _, added in receipts) == 12
        assert {version for _, version, _ in receipts} == set(range(1, 13))
        for item in range(12):
            assert len({version for key, version, _ in receipts if key == item}) == 1
        assert stores[0].conversation_version("conversation") == 12
        assert len(stores[0].recent_turns("conversation")) == 12
        restarted = SQLiteAgentStore(path)
        try:
            expected = next(version for item, version, _ in receipts if item == 0)
            assert restarted.accept_user_input(user_message("message-0"), trace_id="restart-retry", message_key="key-0") == (expected, None, False, EMPTY_SUPERSESSION_COUNTS, [])
        finally:
            restarted._connection.close()
    finally:
        for current in stores:
            current._connection.close()


def test_sqlite_receipt_rolls_back_raw_turn_and_version_when_transition_fails(tmp_path: Path, monkeypatch) -> None:
    store = SQLiteAgentStore(tmp_path / "rollback.sqlite3")
    try:
        def fail_transition(self, transition):
            raise RuntimeError("receipt transition failure")

        monkeypatch.setattr(type(store), "_append_transition", fail_transition)
        with pytest.raises(RuntimeError, match="receipt transition failure"):
            store.accept_user_input(user_message(), trace_id="rollback", message_key="rollback-key")
        assert store.recent_turns("conversation") == []
        assert store.conversation_version("conversation") == 0
    finally:
        store._connection.close()


def test_mark_delivered_is_precise_idempotent_and_preserves_shown_history(store) -> None:
    store.append_assistant_turn("conversation", "已经展示", "shown", metadata={
        "delivery_status": "pending_operator_send", "custom": "保留", "conversation_version": 0,
    })
    store.append_assistant_turn("conversation", "还没展示", "pending", metadata={"delivery_status": "pending_operator_send"})
    assert not store.mark_assistant_reply_delivered("another", trace_id="shown")
    assert not store.mark_assistant_reply_delivered("conversation", trace_id="unknown")
    assert store.mark_assistant_reply_delivered("conversation", trace_id="shown")
    assert store.mark_assistant_reply_delivered("conversation", trace_id="shown")
    store.accept_user_input(user_message(), trace_id="new-input", message_key="new-input")
    store.supersede_pending_outputs("conversation", sender_id="customer", trace_id="new-input", reason="new_user_input")
    replies = {turn.trace_id: turn for turn in store.recent_turns("conversation") if turn.role == ConversationRole.ASSISTANT}
    assert replies["shown"].content == "已经展示"
    assert replies["shown"].metadata == {"delivery_status": "delivered", "custom": "保留", "conversation_version": 0}
    assert replies["pending"].metadata["delivery_status"] == "superseded"
    assert not store.mark_assistant_reply_delivered("conversation", trace_id="pending")
    assert store.mark_assistant_reply_delivered("conversation", trace_id="shown")


def test_mark_delivered_does_not_touch_user_turn_or_other_trace(store) -> None:
    store.append_user_turn(user_message(metadata={"delivery_status": "pending_operator_send"}), "same-trace")
    store.append_assistant_turn("conversation", "其他回复", "other-trace", metadata={"delivery_status": "pending_operator_send"})
    assert not store.mark_assistant_reply_delivered("conversation", trace_id="same-trace")
    assert all(turn.metadata["delivery_status"] == "pending_operator_send" for turn in store.recent_turns("conversation"))


def test_delivery_claim_wins_before_new_input_without_retroactive_supersession(store) -> None:
    store.append_assistant_turn("conversation", "开始展示", "claimed", metadata={"delivery_status": "pending_operator_send", "conversation_version": 0})
    assert store.claim_assistant_reply_delivery("conversation", trace_id="claimed")
    store.accept_user_input(user_message(), trace_id="next", message_key="next")
    store.supersede_pending_outputs("conversation", sender_id="customer", trace_id="next", reason="new_input")
    # 已开始且后来实际成功可确认；一次新的重试不能继续使用过期版本。
    assert not store.claim_assistant_reply_delivery("conversation", trace_id="claimed")
    assert store.mark_assistant_reply_delivered("conversation", trace_id="claimed")
    reply = next(turn for turn in store.recent_turns("conversation") if turn.trace_id == "claimed")
    assert reply.metadata["delivery_status"] == "delivered" and reply.content == "开始展示"


def test_new_input_wins_before_delivery_claim_and_old_reply_cannot_revive(store) -> None:
    store.append_assistant_turn("conversation", "未展示的旧回复", "old", metadata={"delivery_status": "pending_operator_send", "conversation_version": 0})
    store.accept_user_input(user_message(), trace_id="next", message_key="next")
    assert not store.claim_assistant_reply_delivery("conversation", trace_id="old")
    store.supersede_pending_outputs("conversation", sender_id="customer", trace_id="next", reason="new_input")
    assert not store.claim_assistant_reply_delivery("conversation", trace_id="old")
    assert not store.mark_assistant_reply_delivered("conversation", trace_id="old")
    assert not store.claim_assistant_reply_delivery("conversation", trace_id="unknown")


def test_source_batch_version_is_checked_even_when_conversation_version_is_current(store) -> None:
    """较晚捕获的会话版本不能洗白已过期 batch 的待交付来源。"""
    from datetime import timedelta
    from mahjong_agent_runtime.models import now
    deadline = now() + timedelta(seconds=30)
    first, _, _ = store.upsert_pending_input_fragment(user_message("raw-1"), trace_id="raw-1", quiet_deadline=deadline)
    source = {"batch_id": first.batch_id, "batch_version": first.version, "sender_id": "customer"}
    store.append_assistant_turn("conversation", "第一次交付", "claimed-batch", metadata={
        "delivery_status": "pending_operator_send", "conversation_version": store.conversation_version("conversation"), "input_window": source,
    })
    assert store.claim_assistant_reply_delivery("conversation", trace_id="claimed-batch")
    store.upsert_pending_input_fragment(user_message("raw-2"), trace_id="raw-2", quiet_deadline=deadline)
    assert not store.claim_assistant_reply_delivery("conversation", trace_id="claimed-batch")
    store.append_assistant_turn("conversation", "旧来源晚写入", "stale-batch", metadata={
        "delivery_status": "pending_operator_send", "conversation_version": store.conversation_version("conversation"), "input_window": source,
    })
    assert not store.claim_assistant_reply_delivery("conversation", trace_id="stale-batch")
    # 已开始的交付实际成功后仍能确认；这不授予新的重发权限。
    assert store.mark_assistant_reply_delivered("conversation", trace_id="claimed-batch")
    assert store.claim_assistant_reply_delivery("conversation", trace_id="claimed-batch")


def test_original_receipt_nested_metadata_is_not_a_mutable_caller_ticket(store) -> None:
    """保留原始来源快照，调用者后改字典不能绕过接收事务篡改入库证据。"""
    source = user_message(metadata={"channel": "manual", "input_window": {"batch_id": "first", "batch_version": 1}})
    store.accept_user_input(source, trace_id="first", message_key="first")
    source.metadata["input_window"]["batch_version"] = 999
    stored = store.find_conversation_turn("conversation", message_id=source.message_id)
    assert stored.metadata["input_window"] == {"batch_id": "first", "batch_version": 1}
