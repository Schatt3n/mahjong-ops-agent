from __future__ import annotations

import json
import sqlite3
import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from ..models import DEFAULT_TZ
from ..workflow_models import GameRequirement, new_workflow_id


OUTBOX_PENDING_APPROVAL = "pending_approval"
OUTBOX_APPROVED = "approved"
OUTBOX_REJECTED = "rejected"

OUTBOX_DECISION_STATUSES = frozenset({OUTBOX_APPROVED, OUTBOX_REJECTED})
OUTBOX_STATUSES = frozenset({OUTBOX_PENDING_APPROVAL, OUTBOX_APPROVED, OUTBOX_REJECTED})


class PendingOutboxStore(Protocol):
    def create_many(self, drafts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ...

    def list_pending(self, *, conversation_id: str | None = None) -> list[dict[str, Any]]:
        ...

    def get(self, outbox_id: str) -> dict[str, Any] | None:
        ...

    def update_status(
        self,
        outbox_id: str,
        status: str,
        *,
        final_message_text: str | None = None,
        reviewer_id: str | None = None,
        decision_reason: str | None = None,
        trace_id: str | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any] | None:
        ...


class InMemoryPendingOutboxStore:
    def __init__(self) -> None:
        self._items: dict[str, dict[str, Any]] = {}

    def create_many(self, drafts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        stored: list[dict[str, Any]] = []
        for draft in drafts:
            item = _stored_outbox_item(draft)
            self._items.setdefault(str(item["id"]), item)
            stored.append(dict(self._items[str(item["id"])]))
        return stored

    def list_pending(self, *, conversation_id: str | None = None) -> list[dict[str, Any]]:
        items = [dict(item) for item in self._items.values() if item.get("status") == OUTBOX_PENDING_APPROVAL]
        if conversation_id is not None:
            items = [item for item in items if item.get("conversation_id") == str(conversation_id)]
        return sorted(items, key=lambda item: str(item.get("created_at") or ""))

    def get(self, outbox_id: str) -> dict[str, Any] | None:
        item = self._items.get(str(outbox_id))
        return dict(item) if item else None

    def update_status(
        self,
        outbox_id: str,
        status: str,
        *,
        final_message_text: str | None = None,
        reviewer_id: str | None = None,
        decision_reason: str | None = None,
        trace_id: str | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any] | None:
        item = self._items.get(str(outbox_id))
        if item is None:
            return None
        updated = _outbox_item_with_status(
            item,
            status,
            final_message_text=final_message_text,
            reviewer_id=reviewer_id,
            decision_reason=decision_reason,
            trace_id=trace_id,
            now=now,
        )
        self._items[str(outbox_id)] = updated
        return dict(updated)


class SQLitePendingOutboxStore:
    """SQLite-backed pending outbox store for local production trials."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def create_many(self, drafts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        items = [_stored_outbox_item(draft) for draft in drafts]
        with self._connect() as conn:
            for item in items:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO controlled_pending_outbox (
                        id, trace_id, conversation_id, target_customer_id,
                        target_display_name, message_text, status, source,
                        metadata_json, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item["id"],
                        item["trace_id"],
                        item["conversation_id"],
                        item["target_customer_id"],
                        item["target_display_name"],
                        item["message_text"],
                        item["status"],
                        item["source"],
                        json.dumps(item["metadata"], ensure_ascii=False, sort_keys=True),
                        item["created_at"],
                        item["updated_at"],
                    ),
                )
        return [self.get(str(item["id"])) or item for item in items]

    def list_pending(self, *, conversation_id: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM controlled_pending_outbox WHERE status = ?"
        params: list[str] = [OUTBOX_PENDING_APPROVAL]
        if conversation_id is not None:
            sql += " AND conversation_id = ?"
            params.append(str(conversation_id))
        sql += " ORDER BY created_at ASC, id ASC"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_item(row) for row in rows]

    def get(self, outbox_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM controlled_pending_outbox WHERE id = ?",
                (str(outbox_id),),
            ).fetchone()
        return self._row_to_item(row) if row else None

    def update_status(
        self,
        outbox_id: str,
        status: str,
        *,
        final_message_text: str | None = None,
        reviewer_id: str | None = None,
        decision_reason: str | None = None,
        trace_id: str | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM controlled_pending_outbox WHERE id = ?",
                (str(outbox_id),),
            ).fetchone()
            if row is None:
                return None
            updated = _outbox_item_with_status(
                self._row_to_item(row),
                status,
                final_message_text=final_message_text,
                reviewer_id=reviewer_id,
                decision_reason=decision_reason,
                trace_id=trace_id,
                now=now,
            )
            conn.execute(
                """
                UPDATE controlled_pending_outbox
                SET status = ?, message_text = ?, metadata_json = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    updated["status"],
                    updated["message_text"],
                    json.dumps(updated["metadata"], ensure_ascii=False, sort_keys=True),
                    updated["updated_at"],
                    updated["id"],
                ),
            )
        return updated

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS controlled_pending_outbox (
                    id TEXT PRIMARY KEY,
                    trace_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    target_customer_id TEXT NOT NULL,
                    target_display_name TEXT NOT NULL,
                    message_text TEXT NOT NULL,
                    status TEXT NOT NULL,
                    source TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_controlled_pending_outbox_status
                    ON controlled_pending_outbox(status, conversation_id, created_at);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _row_to_item(self, row: sqlite3.Row) -> dict[str, Any]:
        try:
            metadata = json.loads(str(row["metadata_json"] or "{}"))
        except json.JSONDecodeError:
            metadata = {"raw_metadata_json": row["metadata_json"]}
        return {
            "id": str(row["id"]),
            "trace_id": str(row["trace_id"]),
            "conversation_id": str(row["conversation_id"]),
            "target_customer_id": str(row["target_customer_id"]),
            "target_display_name": str(row["target_display_name"]),
            "message_text": str(row["message_text"]),
            "status": str(row["status"]),
            "source": str(row["source"]),
            "metadata": metadata if isinstance(metadata, dict) else {"metadata": metadata},
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
        }


@dataclass(slots=True)
class PendingOutboxTool:
    max_drafts: int = 8
    store: PendingOutboxStore | None = None

    def create_pending_invites(
        self,
        requirement: GameRequirement,
        candidates: list[dict[str, Any]],
        *,
        conversation_id: str,
        trace_id: str,
        base_idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        drafts: list[dict[str, Any]] = []
        for index, candidate in enumerate(candidates[: self.max_drafts]):
            customer_id = str(candidate.get("customer_id") or "")
            display_name = str(candidate.get("display_name") or customer_id or "牌友")
            now_text = datetime.now(DEFAULT_TZ).isoformat()
            outbox_id = _draft_id(
                base_idempotency_key=base_idempotency_key,
                candidate_id=customer_id,
                display_name=display_name,
                index=index,
            )
            drafts.append(
                {
                    "id": outbox_id,
                    "trace_id": trace_id,
                    "conversation_id": conversation_id,
                    "target_customer_id": customer_id,
                    "target_display_name": display_name,
                    "message_text": self._invite_text(display_name, requirement),
                    "status": OUTBOX_PENDING_APPROVAL,
                    "source": "tool_orchestrator",
                    "created_at": now_text,
                    "updated_at": now_text,
                    "metadata": {
                        "approval_status": OUTBOX_PENDING_APPROVAL,
                        "candidate_score": candidate.get("score"),
                        "candidate_reasons": list(candidate.get("reasons") or []),
                        "candidate_warnings": list(candidate.get("warnings") or []),
                        "draft_idempotency_key": base_idempotency_key,
                    },
                }
            )
        stored = self.store.create_many(drafts) if self.store else []
        return {
            "drafts": drafts,
            "result_count": len(drafts),
            "stored_count": len(stored),
            "policy": "只创建待审批草稿，不自动发送。",
        }

    def _invite_text(self, display_name: str, requirement: GameRequirement) -> str:
        slots = requirement.slots
        time_text = _confirmed_slot_value(slots, "start_at") or _start_mode_text(
            _confirmed_slot_value(slots, "start_time_mode")
        )
        stake = _confirmed_slot_value(slots, "stake")
        smoke = _smoke_text(_confirmed_slot_value(slots, "smoke"))
        duration = _duration_text(slots)
        parts = [str(time_text or "").strip(), str(stake or "").strip() + ("无烟" if smoke == "无烟" and stake else "")]
        if smoke and smoke != "无烟":
            parts.append(smoke)
        if duration:
            parts.append(duration)
        body = "，".join(part for part in parts if part)
        if not body:
            body = "有一桌"
        return f"{display_name}，{body}，打吗？"


def _draft_id(
    *,
    base_idempotency_key: str | None,
    candidate_id: str,
    display_name: str,
    index: int,
) -> str:
    if not base_idempotency_key:
        return new_workflow_id("outbox")
    identity = candidate_id or display_name or str(index)
    digest = hashlib.sha256(f"{base_idempotency_key}:{identity}:{index}".encode("utf-8")).hexdigest()[:24]
    return f"outbox_{digest}"


def _slot_value(slots: dict[str, Any], name: str) -> Any:
    slot = slots.get(name)
    return slot.value if slot else None


def _confirmed_slot_value(slots: dict[str, Any], name: str) -> Any:
    slot = slots.get(name)
    if not slot or not getattr(slot, "usable", False):
        return None
    return slot.value


def _start_mode_text(value: Any) -> str | None:
    if value in {"people_ready", "asap_when_full", "when_full", "ready_when_full"}:
        return "人齐开"
    if value == "fixed":
        return None
    return None


def _smoke_text(value: Any) -> str | None:
    if value == "no_smoke":
        return "无烟"
    if value == "smoke_ok":
        return "有烟"
    if value == "any":
        return "烟都可"
    return str(value) if value else None


def _duration_text(slots: dict[str, Any]) -> str | None:
    duration = _confirmed_slot_value(slots, "duration_hours")
    if duration:
        return f"约{duration}小时"
    mode = _confirmed_slot_value(slots, "duration_mode")
    if mode == "overnight":
        return "通宵"
    return None


def _stored_outbox_item(draft: dict[str, Any]) -> dict[str, Any]:
    now = datetime.now(DEFAULT_TZ).isoformat()
    metadata = draft.get("metadata") if isinstance(draft.get("metadata"), dict) else {}
    status = str(draft.get("status") or OUTBOX_PENDING_APPROVAL)
    metadata = dict(metadata)
    metadata.setdefault("approval_status", status)
    return {
        "id": str(draft.get("id") or new_workflow_id("outbox")),
        "trace_id": str(draft.get("trace_id") or ""),
        "conversation_id": str(draft.get("conversation_id") or ""),
        "target_customer_id": str(draft.get("target_customer_id") or ""),
        "target_display_name": str(draft.get("target_display_name") or ""),
        "message_text": str(draft.get("message_text") or ""),
        "status": status,
        "source": str(draft.get("source") or "tool_orchestrator"),
        "metadata": metadata,
        "created_at": str(draft.get("created_at") or now),
        "updated_at": str(draft.get("updated_at") or now),
    }


def _outbox_item_with_status(
    item: dict[str, Any],
    status: str,
    *,
    final_message_text: str | None,
    reviewer_id: str | None,
    decision_reason: str | None,
    trace_id: str | None,
    now: datetime | None,
) -> dict[str, Any]:
    status = _validate_outbox_status(status)
    updated_at = (now or datetime.now(DEFAULT_TZ)).isoformat()
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    metadata = dict(metadata)
    metadata["approval_status"] = status
    updated_message_text = str(item.get("message_text") or "")
    if status in OUTBOX_DECISION_STATUSES:
        if final_message_text is not None:
            updated_message_text = str(final_message_text)
        metadata.setdefault("original_message_text", str(item.get("message_text") or ""))
        metadata["final_message_text"] = updated_message_text
        metadata["reviewer_id"] = str(reviewer_id or "")
        metadata["decision_reason"] = str(decision_reason or "")
        metadata["decision_trace_id"] = str(trace_id or item.get("trace_id") or "")
        metadata["decided_at"] = updated_at
    updated = dict(item)
    updated["status"] = status
    updated["message_text"] = updated_message_text
    updated["metadata"] = metadata
    updated["updated_at"] = updated_at
    return updated


def _validate_outbox_status(status: str) -> str:
    normalized = str(status or "").strip()
    if normalized not in OUTBOX_STATUSES:
        allowed = ", ".join(sorted(OUTBOX_STATUSES))
        raise ValueError(f"Unsupported pending outbox status: {status!r}. Allowed: {allowed}")
    return normalized
