from __future__ import annotations

from datetime import datetime

import pytest

from mahjong_agent.models import DEFAULT_TZ
from mahjong_agent.tools import (
    OUTBOX_APPROVED,
    OUTBOX_PENDING_APPROVAL,
    OUTBOX_REJECTED,
    InMemoryPendingOutboxStore,
    PendingOutboxTool,
    SQLitePendingOutboxStore,
)
from mahjong_agent.workflow_models import GameRequirement, SlotSource, SlotValue


def confirmed_slot(name: str, value) -> SlotValue:
    return SlotValue(
        name=name,
        value=value,
        source=SlotSource.EXPLICIT,
        confidence=0.9,
        confirmed=True,
        needs_confirmation=False,
    )


def requirement() -> GameRequirement:
    game = GameRequirement()
    game.set_slot(confirmed_slot("stake", "0.5"))
    game.set_slot(confirmed_slot("start_time_mode", "people_ready"))
    game.set_slot(confirmed_slot("smoke", "no_smoke"))
    game.set_slot(confirmed_slot("duration_hours", 4))
    return game


def unconfirmed_slot(name: str, value) -> SlotValue:
    return SlotValue(
        name=name,
        value=value,
        source=SlotSource.INFERRED,
        confidence=0.5,
        confirmed=False,
        needs_confirmation=False,
    )


def candidates() -> list[dict]:
    return [
        {
            "customer_id": "ran",
            "display_name": "冉姐",
            "score": 98,
            "reasons": ["常打0.5"],
            "warnings": [],
        },
        {
            "customer_id": "liu",
            "display_name": "刘姐",
            "score": 92,
            "reasons": ["无烟匹配"],
            "warnings": ["最近邀约过"],
        },
    ]


def test_pending_outbox_tool_can_store_drafts_in_memory() -> None:
    store = InMemoryPendingOutboxStore()
    result = PendingOutboxTool(store=store).create_pending_invites(
        requirement(),
        candidates(),
        conversation_id="boss_trial",
        trace_id="trace_outbox",
    )

    assert result["result_count"] == 2
    assert result["stored_count"] == 2
    pending = store.list_pending(conversation_id="boss_trial")
    assert len(pending) == 2
    assert pending[0]["status"] == OUTBOX_PENDING_APPROVAL
    assert pending[0]["metadata"]["approval_status"] == OUTBOX_PENDING_APPROVAL
    assert pending[0]["metadata"]["candidate_reasons"] == ["常打0.5"]


def test_pending_outbox_tool_humanizes_internal_modes_and_omits_unconfirmed_smoke() -> None:
    game = GameRequirement()
    game.set_slot(confirmed_slot("stake", "1"))
    game.set_slot(confirmed_slot("start_time_mode", "asap_when_full"))
    game.set_slot(confirmed_slot("duration_mode", "overnight"))
    game.set_slot(unconfirmed_slot("smoke", "any"))

    result = PendingOutboxTool().create_pending_invites(
        game,
        candidates()[:1],
        conversation_id="boss_trial",
        trace_id="trace_outbox_humanized",
    )

    text = result["drafts"][0]["message_text"]
    assert text == "冉姐，人齐开，1，通宵，打吗？"
    assert "asap_when_full" not in text
    assert "烟都可" not in text


def test_pending_outbox_tool_uses_stable_ids_from_idempotency_key() -> None:
    store = InMemoryPendingOutboxStore()
    tool = PendingOutboxTool(store=store)

    first = tool.create_pending_invites(
        requirement(),
        candidates(),
        conversation_id="boss_trial",
        trace_id="trace_outbox",
        base_idempotency_key="action_test:create_pending_outbox",
    )
    second = tool.create_pending_invites(
        requirement(),
        candidates(),
        conversation_id="boss_trial",
        trace_id="trace_outbox_retry",
        base_idempotency_key="action_test:create_pending_outbox",
    )

    assert [item["id"] for item in second["drafts"]] == [item["id"] for item in first["drafts"]]
    assert first["drafts"][0]["id"].startswith("outbox_")
    assert first["drafts"][0]["metadata"]["draft_idempotency_key"] == "action_test:create_pending_outbox"
    pending = store.list_pending(conversation_id="boss_trial")
    assert len(pending) == 2
    assert [item["id"] for item in pending] == [item["id"] for item in first["drafts"]]


def test_sqlite_pending_outbox_store_persists_pending_drafts(tmp_path) -> None:
    path = tmp_path / "outbox" / "pending_outbox.sqlite3"
    store = SQLitePendingOutboxStore(path)
    result = PendingOutboxTool(store=store).create_pending_invites(
        requirement(),
        candidates(),
        conversation_id="boss_trial",
        trace_id="trace_outbox_sqlite",
    )

    reloaded = SQLitePendingOutboxStore(path)
    pending = reloaded.list_pending(conversation_id="boss_trial")

    assert result["stored_count"] == 2
    assert len(pending) == 2
    assert pending[0]["id"] == result["drafts"][0]["id"]
    assert pending[0]["target_customer_id"] == "ran"
    assert pending[0]["message_text"].endswith("打吗？")
    assert reloaded.get(result["drafts"][1]["id"])["metadata"]["candidate_warnings"] == ["最近邀约过"]


def test_in_memory_pending_outbox_store_records_approval_decision() -> None:
    store = InMemoryPendingOutboxStore()
    result = PendingOutboxTool(store=store).create_pending_invites(
        requirement(),
        candidates(),
        conversation_id="boss_trial",
        trace_id="trace_outbox",
    )
    outbox_id = result["drafts"][0]["id"]
    decided_at = datetime(2026, 7, 1, 15, 30, tzinfo=DEFAULT_TZ)

    approved = store.update_status(
        outbox_id,
        OUTBOX_APPROVED,
        reviewer_id="boss",
        decision_reason="话术确认",
        trace_id="trace_approval_001",
        now=decided_at,
    )

    assert approved is not None
    assert approved["status"] == OUTBOX_APPROVED
    assert approved["metadata"]["approval_status"] == OUTBOX_APPROVED
    assert approved["metadata"]["reviewer_id"] == "boss"
    assert approved["metadata"]["decision_reason"] == "话术确认"
    assert approved["metadata"]["decision_trace_id"] == "trace_approval_001"
    assert approved["metadata"]["decided_at"] == decided_at.isoformat()
    assert len(store.list_pending(conversation_id="boss_trial")) == 1
    assert store.get(outbox_id)["status"] == OUTBOX_APPROVED


def test_sqlite_pending_outbox_store_persists_rejection_decision(tmp_path) -> None:
    path = tmp_path / "outbox" / "pending_outbox.sqlite3"
    store = SQLitePendingOutboxStore(path)
    result = PendingOutboxTool(store=store).create_pending_invites(
        requirement(),
        candidates(),
        conversation_id="boss_trial",
        trace_id="trace_outbox_sqlite",
    )
    outbox_id = result["drafts"][1]["id"]

    rejected = store.update_status(
        outbox_id,
        OUTBOX_REJECTED,
        reviewer_id="boss",
        decision_reason="今天不打扰",
        trace_id="trace_reject_001",
    )
    reloaded = SQLitePendingOutboxStore(path)
    persisted = reloaded.get(outbox_id)

    assert rejected["status"] == OUTBOX_REJECTED
    assert persisted["status"] == OUTBOX_REJECTED
    assert persisted["metadata"]["approval_status"] == OUTBOX_REJECTED
    assert persisted["metadata"]["reviewer_id"] == "boss"
    assert persisted["metadata"]["decision_reason"] == "今天不打扰"
    assert persisted["metadata"]["decision_trace_id"] == "trace_reject_001"
    assert len(reloaded.list_pending(conversation_id="boss_trial")) == 1


def test_pending_outbox_store_rejects_unknown_status() -> None:
    store = InMemoryPendingOutboxStore()
    result = PendingOutboxTool(store=store).create_pending_invites(
        requirement(),
        candidates(),
        conversation_id="boss_trial",
        trace_id="trace_outbox",
    )

    with pytest.raises(ValueError, match="Unsupported pending outbox status"):
        store.update_status(result["drafts"][0]["id"], "sent")
