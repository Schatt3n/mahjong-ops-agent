"""接收原话、推进版本和废弃待发状态必须同成同败，重试不得伤及新输出。"""

from __future__ import annotations

from pathlib import Path

import pytest

from mahjong_agent_runtime import (
    AgentRuntime, CustomerProfile, InMemoryAgentStore, SQLiteAgentStore, StaticAgentClient, UserMessage,
)
from mahjong_agent_runtime.models import ConversationRole
import mahjong_agent_runtime.stores.memory.conversation as memory_conversation


ZERO_COUNTS = {"invite_drafts": 0, "outbound_message_drafts": 0, "assistant_replies": 0}
ALL_COUNTS = {key: 1 for key in ZERO_COUNTS}


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path: Path):
    """两种事实源共用失败注入合同；SQLite 每个测试使用独立可重开的数据库。"""
    current = InMemoryAgentStore() if request.param == "memory" else SQLiteAgentStore(tmp_path / "atomic-receipt.sqlite3")
    yield current
    if isinstance(current, SQLiteAgentStore):
        current._connection.close()


def message(message_id="new-input", conversation_id="chat", sender_id="owner"):
    """只提供原始消息，接收版本与废弃决策必须由后端产生。"""
    return UserMessage(conversation_id=conversation_id, sender_id=sender_id, sender_name="发起人",
                       text="这次改成两块的", message_id=message_id)


def seed_pending_outputs(store, *, suffix="old", conversation_id="chat"):
    """真实创建邀约、外发草稿和未展示回复，覆盖全部待废弃持久状态。"""
    for customer_id in ("owner", "candidate"):
        store.upsert_customer(CustomerProfile(customer_id=customer_id, display_name=customer_id))
    game, _ = store.create_game(
        conversation_id=conversation_id, organizer_id="owner", organizer_name="发起人",
        requirement={"game_type": "national_standard_mahjong", "stake": "1", "start_time_kind": "asap_when_full"},
        known_players=[{"customer_id": "owner", "display_name": "发起人"}], trace_id=f"seed-{suffix}",
    )
    invites, _ = store.create_invite_drafts(game_id=game.game_id,
        invitations=[{"customer_id": "candidate", "display_name": "候选人", "message_text": "来打吗？"}],
        trace_id=f"seed-{suffix}")
    outbound, _ = store.create_outbound_message_drafts(conversation_id=conversation_id,
        drafts=[{"recipient_id": "candidate", "channel": "manual", "message_text": "还有空吗？", "purpose": "test"}],
        trace_id=f"seed-{suffix}")
    store.append_assistant_turn(conversation_id, "旧回复还未展示", f"reply-{suffix}", metadata={
        "delivery_status": "pending_operator_send", "conversation_version": store.conversation_version(conversation_id),
    })
    return invites[0].draft_id, outbound[0].draft_id


def receipt_state(store):
    """只比较本次接收事务负责的全部事实，任务上下文准备属于单独的既有生命周期。"""
    return {
        "version": store.conversation_version("chat"),
        "turns": [turn.to_dict() for turn in store.recent_turns("chat", limit=100)],
        "invites": {key: value.to_dict() for key, value in store.invite_drafts.items()},
        "outbound": {key: value.to_dict() for key, value in store.outbound_message_drafts.items()},
        "transitions": [item.to_dict() for item in store.transitions if item.entity_type in {
            "conversation_version", "invite_draft", "outbound_message_draft", "assistant_reply"}],
    }


def inject_failure(store, monkeypatch, entity_type):
    """Memory 在复制后构造迁移时失败；SQLite 在所有数据更新后追加审计时失败。"""
    if isinstance(store, InMemoryAgentStore):
        original = memory_conversation.StateTransition

        def fail_transition(*args, **kwargs):
            transition = original(*args, **kwargs)
            if transition.entity_type == entity_type and transition.to_status == "superseded":
                raise RuntimeError("injected supersession failure")
            return transition

        monkeypatch.setattr(memory_conversation, "StateTransition", fail_transition)
    else:
        original = type(store)._append_transition

        def fail_transition(self, transition):
            original(self, transition)
            if transition.entity_type == entity_type and transition.to_status == "superseded":
                raise RuntimeError("injected supersession failure")

        monkeypatch.setattr(type(store), "_append_transition", fail_transition)


@pytest.mark.parametrize("entity_type", ["invite_draft", "outbound_message_draft", "assistant_reply"])
def test_failed_supersession_rolls_back_whole_receipt_and_retry_succeeds(store, monkeypatch, entity_type):
    """任一废弃步骤失败不能留下原话/版本/部分草稿；同 ID 重试必须仍是首次接收。"""
    seed_pending_outputs(store)
    before = receipt_state(store)
    with monkeypatch.context() as patch:
        inject_failure(store, patch, entity_type)
        with pytest.raises(RuntimeError, match="injected supersession failure"):
            store.accept_user_input(message(), trace_id="failed", message_key="same-key")
    assert receipt_state(store) == before
    assert store.find_conversation_turn("chat", message_id="new-input") is None
    if isinstance(store, SQLiteAgentStore):
        reopened = SQLiteAgentStore(store.path)
        try:
            assert receipt_state(reopened) == before
        finally:
            reopened._connection.close()
    version, transition, added, counts, superseded = store.accept_user_input(
        message(), trace_id="retry", message_key="same-key")
    assert version == 1 and added and transition.trace_id == "retry"
    assert counts == ALL_COUNTS
    assert {item.entity_type for item in superseded} == {"invite_draft", "outbound_message_draft", "assistant_reply"}
    assert all(item.trace_id == "retry" and item.to_status == "superseded" for item in superseded)
    assert store.find_conversation_turn("chat", message_id="new-input").trace_id == "retry"
    assert not any(item.trace_id == "failed" for item in store.transitions)
    after = receipt_state(store)
    assert store.accept_user_input(message(), trace_id="duplicate", message_key="same-key") == (1, None, False, ZERO_COUNTS, [])
    assert receipt_state(store) == after


def test_runtime_failed_receipt_can_retry_without_skipping_supersession(store, monkeypatch):
    """真实 receive 入口的失败不会记成已收，从而避免重试跳过旧草稿废弃。"""
    seed_pending_outputs(store)
    client = StaticAgentClient(outputs=[])
    runtime = AgentRuntime(llm_client=client, store=store)
    before = receipt_state(store)
    with monkeypatch.context() as patch:
        inject_failure(store, patch, "assistant_reply")
        with pytest.raises(RuntimeError, match="injected supersession failure"):
            runtime.receive_user_message(message(), trace_id="failed-runtime")
    assert receipt_state(store) == before
    received = runtime.receive_user_message(message(), trace_id="retry-runtime")
    assert received["accepted"] and received["conversation_version"] == 1
    superseded = [event.content for event in runtime.trace_recorder.get_trace("retry-runtime")
                  if event.step == "pending_outputs_superseded"]
    assert superseded and superseded[0]["counts"] == ALL_COUNTS
    assert len([turn for turn in store.recent_turns("chat") if turn.role == ConversationRole.USER]) == 1
    assert all(draft.status.value == "superseded" for draft in store.invite_drafts.values())
    assert all(draft.status.value == "superseded" for draft in store.outbound_message_drafts.values())
    assert client.calls == []


def test_duplicate_receipt_keeps_original_version_and_does_not_supersede_new_outputs(store):
    """旧消息重传只能返回原版本，不能清理后来输入下新建的待发结果。"""
    first = message("first")
    store.accept_user_input(first, trace_id="first", message_key="first-key")
    store.accept_user_input(message("second"), trace_id="second", message_key="second-key")
    seed_pending_outputs(store, suffix="new")
    before = receipt_state(store)
    assert store.accept_user_input(first, trace_id="old-retry", message_key="first-key") == (1, None, False, ZERO_COUNTS, [])
    assert receipt_state(store) == before and store.conversation_version("chat") == 2


def test_public_supersession_keeps_its_own_atomic_boundary(store, monkeypatch):
    """独立废弃接口继续同成同败，不能依赖外部一定先调用输入接收。"""
    seed_pending_outputs(store)
    before = receipt_state(store)
    with monkeypatch.context() as patch:
        inject_failure(store, patch, "assistant_reply")
        with pytest.raises(RuntimeError, match="injected supersession failure"):
            store.supersede_pending_outputs("chat", sender_id="owner", trace_id="failed", reason="new-input")
    assert receipt_state(store) == before
    counts, transitions = store.supersede_pending_outputs("chat", sender_id="owner", trace_id="retry", reason="new-input")
    assert counts == ALL_COUNTS and len(transitions) == 3
    assert store.conversation_version("chat") == 0
    assert not any(item.trace_id == "failed" for item in store.transitions)


def test_supersession_keeps_other_conversation_and_candidate_invite_scope(store):
    """相同客户在另一会话的输出不受影响；候选人来信不替发起方撤掉邀约。"""
    seed_pending_outputs(store)
    other_ids = seed_pending_outputs(store, suffix="other", conversation_id="other-chat")
    other = (store.invite_drafts[other_ids[0]].to_dict(), store.outbound_message_drafts[other_ids[1]].to_dict())
    _, _, added, counts, _ = store.accept_user_input(message(sender_id="candidate"), trace_id="candidate", message_key="candidate-key")
    assert added and counts == {**ZERO_COUNTS, "assistant_replies": 1}
    assert all(draft.status.value == "pending_approval" for draft in store.invite_drafts.values())
    assert all(draft.status.value == "pending_approval" for draft in store.outbound_message_drafts.values())
    assert (store.invite_drafts[other_ids[0]].to_dict(), store.outbound_message_drafts[other_ids[1]].to_dict()) == other
    assert store.recent_turns("other-chat")[-1].metadata["delivery_status"] == "pending_operator_send"


def test_memory_supersession_does_not_mutate_references_to_uncommitted_originals():
    """内存事实源发布新副本，调用方旧引用不能看到失败准备中的中间状态。"""
    store = InMemoryAgentStore()
    invite_id, outbound_id = seed_pending_outputs(store)
    invite = store.invite_drafts[invite_id]
    outbound = store.outbound_message_drafts[outbound_id]
    reply = store.recent_turns("chat")[-1]
    store.accept_user_input(message(), trace_id="accepted", message_key="accepted")
    assert invite.status.value == outbound.status.value == "pending_approval"
    assert reply.metadata["delivery_status"] == "pending_operator_send"
    assert store.invite_drafts[invite_id].status.value == store.outbound_message_drafts[outbound_id].status.value == "superseded"
    assert store.recent_turns("chat")[-2].metadata["delivery_status"] == "superseded"


def test_default_version_key_does_not_rekey_raw_input_evidence(store):
    """既有空会话版本键可归一化，原话与幂等查找仍保留收到的会话标识。"""
    source = message(conversation_id="")
    assert store.accept_user_input(source, trace_id="first", message_key="stable")[0] == 1
    assert store.find_conversation_turn("", message_id=source.message_id).content == source.text
    assert store.accept_user_input(source, trace_id="duplicate", message_key="stable") == (1, None, False, ZERO_COUNTS, [])
    assert store.conversation_version("") == 1
