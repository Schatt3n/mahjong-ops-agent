"""核对店家故障接手待办的持久性和接手时的会话栅栏。"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from threading import Barrier

import pytest

from mahjong_agent_runtime import AgentRunState, AgentRunStatus, InMemoryAgentStore, SQLiteAgentStore, UserMessage
from mahjong_agent_runtime.models import PendingInputBatchStatus, using_clock


STAMP = datetime.fromisoformat("2026-11-26T16:00:00+08:00")


def _run(run_id, status, *, handoff="reply_review_timeout", version=0, delivered=False):
    return AgentRunState(
        run_id=run_id,
        trace_id=f"trace-{run_id}",
        conversation_id="private-1",
        run_version=version,
        message={"sender_id": "guest-1", "message_id": f"message-{run_id}"},
        status=status,
        handoff_reason=handoff,
        handoff_wait_delivered=delivered,
        lease_owner="automation" if status == AgentRunStatus.RUNNING else "",
        lease_until=STAMP + timedelta(minutes=1) if status == AgentRunStatus.RUNNING else None,
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_handoff_queue_persists_terminal_failures_and_does_not_claim_auto_recovery(tmp_path, backend):
    """待办独立于自动租约；退出重试后仍可接手，清除原因则关闭待办。"""
    path = tmp_path / "handoff.sqlite3"
    store = InMemoryAgentStore() if backend == "memory" else SQLiteAgentStore(path)
    with using_clock(lambda: STAMP):
        for run in [
            _run("retryable", AgentRunStatus.RECOVERABLE, delivered=True),
            _run("failed", AgentRunStatus.FAILED),
            _run("needs-help", AgentRunStatus.COMPLETED),
            _run("superseded", AgentRunStatus.SUPERSEDED),
            _run("completed", AgentRunStatus.COMPLETED, handoff=""),
        ]:
            store.create_agent_run(run)

    if backend == "sqlite":
        store = SQLiteAgentStore(path)
    pending = store.pending_agent_run_handoffs()
    assert [run.run_id for run in pending] == ["failed", "needs-help", "retryable"]
    assert [run.run_id for run in store.pending_agent_run_handoffs(limit=1)] == ["failed"]
    assert pending[-1].handoff_wait_delivered is True
    pending[-1].handoff_reason = ""
    assert store.agent_run("retryable").handoff_reason == "reply_review_timeout"

    retry = store.claim_agent_run(
        "retryable", lease_owner="recovery", at=STAMP, lease_until=STAMP + timedelta(minutes=1),
    )
    assert retry.handoff_wait_delivered is True
    assert "retryable" in {run.run_id for run in store.pending_agent_run_handoffs()}
    retry.handoff_reason = ""
    assert store.save_agent_run(retry, expected_lease_owner="recovery")
    assert "retryable" not in {run.run_id for run in store.pending_agent_run_handoffs()}


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_operator_claim_fences_concurrent_retry_and_preserves_delivered_notice(tmp_path, backend):
    """一个店家领取获胜，旧自动写入与待发回复失效，已送达回执保留。"""
    path = tmp_path / "claim.sqlite3"
    writer = InMemoryAgentStore() if backend == "memory" else SQLiteAgentStore(path)
    other = writer if backend == "memory" else SQLiteAgentStore(path)
    writer.create_agent_run(_run("currently-running", AgentRunStatus.RUNNING, delivered=True))
    writer.append_assistant_turn(
        "private-1", "稍等一下", "wait-trace", metadata={"delivery_status": "delivered", "conversation_version": 0},
    )
    writer.append_assistant_turn(
        "private-1", "位给你留上了", "trace-currently-running",
        metadata={"delivery_status": "pending_operator_send", "conversation_version": 0},
    )
    worker = writer.agent_run("currently-running")
    barrier = Barrier(2)

    def claim(store):
        barrier.wait(timeout=5)
        return store.claim_agent_run_handoff("currently-running", expected_conversation_version=0)

    assert not writer.claim_agent_run_handoff("currently-running", expected_conversation_version=1)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = [executor.submit(claim, store) for store in (writer, other)]
        assert sorted(f.result(timeout=8) for f in results) == [False, True]
    claimed = writer.agent_run("currently-running")
    assert claimed.status == AgentRunStatus.SUPERSEDED
    assert claimed.handoff_wait_delivered is True
    assert claimed.handoff_reason == "reply_review_timeout"
    assert writer.conversation_version("private-1") == 1
    assert writer.pending_agent_run_handoffs() == []
    assert writer.save_agent_run(worker, expected_lease_owner="automation") is False
    assert writer.recoverable_agent_runs(at=STAMP + timedelta(hours=1)) == []
    turns = writer.recent_turns("private-1")
    assert [(turn.content, turn.metadata["delivery_status"]) for turn in turns] == [
        ("稍等一下", "delivered"),
        ("位给你留上了", "superseded"),
    ]


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_input_gate_handoff_persists_and_fences_classifier_and_business_claim(tmp_path, backend):
    """入口故障批次只允许一个领取者成功，旧分类凭证和旧结算随即失效。"""
    path = tmp_path / "gate-handoff.sqlite3"
    writer = InMemoryAgentStore() if backend == "memory" else SQLiteAgentStore(path)
    other = writer if backend == "memory" else SQLiteAgentStore(path)

    def add(conversation, status=PendingInputBatchStatus.PENDING):
        batch, _, _ = writer.upsert_pending_input_fragment(
            UserMessage(conversation_id=conversation, sender_id="guest", sender_name="顾客",
                        text="这桌还有位吗", message_id=f"source-{conversation}", sent_at=STAMP),
            trace_id=conversation, quiet_deadline=STAMP,
        )
        recovery = {"attempt": 1, "conversation_version": writer.conversation_version(conversation),
                    "origin_trace_id": conversation, "route": "complete_message",
                    "open": True, "wait_delivered": True}
        decision = {"gate_recovery": recovery}
        assert writer.queue_pending_input_batch(
            batch_id=batch.batch_id, expected_version=batch.version, decision=decision, due_at=STAMP,
        )
        if status != PendingInputBatchStatus.PENDING:
            assert writer.finish_pending_input_batch(
                batch_id=batch.batch_id, expected_version=batch.version, status=status,
                trace_id=conversation, decision=decision,
            )[0]
        return batch

    with using_clock(lambda: STAMP):
        pending = add("pending")
        exhausted = add("exhausted", PendingInputBatchStatus.FAILED)
        stale = add("stale")
        writer.advance_conversation_version(stale.conversation_id, trace_id="new", reason="input")
        processing = add("processing")
        assert writer.claim_pending_input_batch(
            batch_id=processing.batch_id, expected_version=processing.version, trace_id="processing",
        )[0]

    assert {batch.conversation_id for batch in other.pending_input_gate_handoffs()} == {"pending", "exhausted"}
    for batch in (stale, processing):
        assert not writer.claim_input_gate_handoff(batch.batch_id, expected_version=batch.version, trace_id="invalid")
    assert not writer.claim_input_gate_handoff(pending.batch_id, expected_version=pending.version + 1, trace_id="invalid")
    barrier = Barrier(2)

    def claim(store):
        barrier.wait(timeout=5)
        return store.claim_input_gate_handoff(pending.batch_id, expected_version=pending.version, trace_id="owner")

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(claim, store) for store in (writer, other)]
        assert sorted(future.result(timeout=8) for future in futures) == [False, True]
    saved = writer.pending_input_batch(pending.conversation_id, pending.sender_id)
    assert saved.status == PendingInputBatchStatus.IGNORED and saved.version == pending.version + 1
    assert saved.decision["gate_recovery"]["open"] is False and saved.decision["gate_recovery"]["wait_delivered"] is True
    assert other.claim_pending_input_batch(
        batch_id=pending.batch_id, expected_version=pending.version, trace_id="old-classifier",
    ) == (None, None)
    assert other.finish_pending_input_batch(
        batch_id=pending.batch_id, expected_version=pending.version, status=PendingInputBatchStatus.FAILED,
        trace_id="old-classifier", decision={"gate_recovery": {"open": True}},
    ) == (None, None)
    assert writer.claim_input_gate_handoff(exhausted.batch_id, expected_version=exhausted.version, trace_id="owner")
    assert other.pending_input_gate_handoffs() == []
    assert not writer.pending_agent_run_handoffs()
