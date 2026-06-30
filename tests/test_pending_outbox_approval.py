from __future__ import annotations

from datetime import datetime

from mahjong_agent.approval import PendingOutboxApprovalConfig, PendingOutboxApprovalService
from mahjong_agent.models import DEFAULT_TZ
from mahjong_agent.tool_orchestrator import InMemoryToolExecutionLedger, SQLiteToolExecutionLedger
from mahjong_agent.tools import (
    OUTBOX_APPROVED,
    OUTBOX_PENDING_APPROVAL,
    OUTBOX_REJECTED,
    SQLitePendingOutboxStore,
)
from mahjong_agent.workflow_models import ToolName


def draft() -> dict:
    return {
        "id": "outbox_test_001",
        "trace_id": "trace_create_outbox",
        "conversation_id": "boss_trial",
        "target_customer_id": "ran",
        "target_display_name": "冉姐",
        "message_text": "冉姐，14:00，0.5无烟，打吗？",
        "status": OUTBOX_PENDING_APPROVAL,
        "source": "test",
        "metadata": {"approval_status": OUTBOX_PENDING_APPROVAL},
    }


def test_pending_outbox_approval_service_approves_without_sending(tmp_path) -> None:
    store = SQLitePendingOutboxStore(tmp_path / "outbox.sqlite3")
    store.create_many([draft()])
    ledger = InMemoryToolExecutionLedger()
    service = PendingOutboxApprovalService(store, execution_ledger=ledger)
    decided_at = datetime(2026, 7, 1, 16, 0, tzinfo=DEFAULT_TZ)

    result = service.decide(
        outbox_id="outbox_test_001",
        decision="approved",
        reviewer_id="boss",
        reviewer_name="老板",
        reason="话术可以",
        final_message_text="冉姐，14:00，0.5无烟，打吗",
        trace_id="trace_approval_001",
        now=decided_at,
        idempotency_key="approval_once",
    )

    persisted = store.get("outbox_test_001")
    history = ledger.history(tool_name=ToolName.RECORD_APPROVAL_DECISION)

    assert result["ok"] is True
    assert result["deduplicated"] is False
    assert result["approval"]["status"] == OUTBOX_APPROVED
    assert result["approval"]["decision_reason"] == "话术可以"
    assert result["approval"]["final_message_text"] == "冉姐，14:00，0.5无烟，打吗"
    assert persisted["status"] == OUTBOX_APPROVED
    assert persisted["message_text"] == "冉姐，14:00，0.5无烟，打吗"
    assert persisted["metadata"]["original_message_text"] == "冉姐，14:00，0.5无烟，打吗？"
    assert persisted["metadata"]["decision_trace_id"] == "trace_approval_001"
    assert "sent" not in persisted["status"]
    assert len(history) == 1
    assert history[0].called is True
    assert history[0].request.tool_name == ToolName.RECORD_APPROVAL_DECISION


def test_pending_outbox_approval_service_deduplicates_same_decision(tmp_path) -> None:
    store = SQLitePendingOutboxStore(tmp_path / "outbox.sqlite3")
    store.create_many([draft()])
    service = PendingOutboxApprovalService(store)

    first = service.decide(
        outbox_id="outbox_test_001",
        decision="通过",
        trace_id="trace_approval_001",
        idempotency_key="same_approval",
    )
    second = service.decide(
        outbox_id="outbox_test_001",
        decision="通过",
        trace_id="trace_approval_retry",
        idempotency_key="same_approval",
    )

    assert first["ok"] is True
    assert second["ok"] is True
    assert second["deduplicated"] is True
    assert second["approval"]["decision_trace_id"] == "trace_approval_001"


def test_pending_outbox_approval_service_blocks_when_approval_disabled(tmp_path) -> None:
    store = SQLitePendingOutboxStore(tmp_path / "outbox.sqlite3")
    store.create_many([draft()])
    service = PendingOutboxApprovalService(
        store,
        config=PendingOutboxApprovalConfig(approval_enabled=False),
    )

    result = service.decide(outbox_id="outbox_test_001", decision="approved")

    assert result["ok"] is False
    assert result["code"] == "runtime_policy_approval_disabled"
    assert store.get("outbox_test_001")["status"] == OUTBOX_PENDING_APPROVAL
    assert result["tool_result"].allowed is False


def test_pending_outbox_approval_service_blocks_terminal_conflict(tmp_path) -> None:
    store = SQLitePendingOutboxStore(tmp_path / "outbox.sqlite3")
    store.create_many([draft()])
    service = PendingOutboxApprovalService(store)

    approved = service.decide(outbox_id="outbox_test_001", decision="approved")
    rejected = service.decide(outbox_id="outbox_test_001", decision="rejected")

    assert approved["ok"] is True
    assert rejected["ok"] is False
    assert rejected["code"] == "terminal_approval_conflict"
    assert store.get("outbox_test_001")["status"] == OUTBOX_APPROVED


def test_pending_outbox_approval_service_persists_sqlite_and_ledger(tmp_path) -> None:
    outbox_path = tmp_path / "outbox.sqlite3"
    ledger_path = tmp_path / "tool_ledger.sqlite3"
    store = SQLitePendingOutboxStore(outbox_path)
    store.create_many([draft()])
    service = PendingOutboxApprovalService(
        store,
        execution_ledger=SQLiteToolExecutionLedger(ledger_path),
    )

    result = service.decide(
        outbox_id="outbox_test_001",
        decision="拒绝",
        reviewer_id="boss",
        reason="今天先不打扰",
        trace_id="trace_reject",
    )
    reloaded_store = SQLitePendingOutboxStore(outbox_path)
    reloaded_ledger = SQLiteToolExecutionLedger(ledger_path)

    assert result["ok"] is True
    assert reloaded_store.get("outbox_test_001")["status"] == OUTBOX_REJECTED
    assert reloaded_store.get("outbox_test_001")["metadata"]["decision_reason"] == "今天先不打扰"
    assert reloaded_ledger.history(tool_name=ToolName.RECORD_APPROVAL_DECISION)[0].request.tool_name == (
        ToolName.RECORD_APPROVAL_DECISION
    )
