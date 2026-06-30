from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .tool_orchestrator import InMemoryToolExecutionLedger, ToolExecutionLedger
from .tools import (
    OUTBOX_APPROVED,
    OUTBOX_PENDING_APPROVAL,
    OUTBOX_REJECTED,
    PendingOutboxStore,
)
from .workflow_models import RiskLevel, ToolCallRequest, ToolExecutionMode, ToolName, ToolResult


APPROVAL_DECISION_ALIASES = {
    "approve": OUTBOX_APPROVED,
    "approved": OUTBOX_APPROVED,
    "同意": OUTBOX_APPROVED,
    "通过": OUTBOX_APPROVED,
    "审批通过": OUTBOX_APPROVED,
    "reject": OUTBOX_REJECTED,
    "rejected": OUTBOX_REJECTED,
    "拒绝": OUTBOX_REJECTED,
    "审批拒绝": OUTBOX_REJECTED,
}


@dataclass(slots=True)
class PendingOutboxApprovalConfig:
    approval_enabled: bool = True


class PendingOutboxApprovalService:
    """Controlled approval boundary for pending outbound drafts.

    Approval is a state decision made by the owner. It never sends the message;
    delivery must use a separate send gateway after approval.
    """

    def __init__(
        self,
        store: PendingOutboxStore,
        *,
        execution_ledger: ToolExecutionLedger | None = None,
        config: PendingOutboxApprovalConfig | None = None,
    ) -> None:
        self.store = store
        self.execution_ledger = execution_ledger or InMemoryToolExecutionLedger()
        self.config = config or PendingOutboxApprovalConfig()

    def decide(
        self,
        *,
        outbox_id: str,
        decision: str,
        reviewer_id: str = "boss_manual",
        reviewer_name: str = "老板",
        reason: str = "",
        final_message_text: str | None = None,
        trace_id: str | None = None,
        now: datetime | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        status = normalize_approval_decision(decision)
        request = self._request(
            outbox_id=outbox_id,
            raw_decision=decision,
            status=status,
            reviewer_id=reviewer_id,
            reviewer_name=reviewer_name,
            reason=reason,
            final_message_text=final_message_text,
            trace_id=trace_id,
            idempotency_key=idempotency_key,
        )

        if not self.config.approval_enabled:
            return self._record_error(request, "runtime_policy_approval_disabled", "审批功能已被运行时策略关闭。")
        if status not in {OUTBOX_APPROVED, OUTBOX_REJECTED}:
            return self._record_error(request, "invalid_approval_decision", "审批结果只能是 approved/rejected。")

        existing = self.execution_ledger.lookup(str(request.idempotency_key or ""))
        if existing is not None:
            deduped = ToolResult(
                request=request,
                called=existing.called,
                allowed=existing.allowed,
                result=dict(existing.result or {}),
                error=existing.error,
                deduplicated=True,
            )
            self.execution_ledger.record(deduped)
            return self._payload_from_result(deduped)

        item = self.store.get(str(outbox_id))
        if item is None:
            return self._record_error(request, "outbox_not_found", "找不到待审批 outbox。")

        current_status = str(item.get("status") or OUTBOX_PENDING_APPROVAL)
        if current_status in {OUTBOX_APPROVED, OUTBOX_REJECTED}:
            if current_status != status:
                return self._record_error(
                    request,
                    "terminal_approval_conflict",
                    "outbox 已经是终态，不能改成另一个审批结果。",
                )
            payload = self._approval_payload(
                outbox_item=item,
                status=status,
                reviewer_id=reviewer_id,
                reviewer_name=reviewer_name,
                reason=reason,
                trace_id=trace_id,
                already_decided=True,
            )
            recorded = self.execution_ledger.record(
                ToolResult(request=request, called=False, allowed=True, result=payload)
            )
            return self._payload_from_result(recorded)

        updated = self.store.update_status(
            str(outbox_id),
            status,
            final_message_text=final_message_text,
            reviewer_id=reviewer_id,
            decision_reason=reason,
            trace_id=trace_id,
            now=now,
        )
        if updated is None:
            return self._record_error(request, "outbox_not_found", "找不到待审批 outbox。")

        payload = self._approval_payload(
            outbox_item=updated,
            status=status,
            reviewer_id=reviewer_id,
            reviewer_name=reviewer_name,
            reason=reason,
            trace_id=trace_id,
            already_decided=False,
        )
        recorded = self.execution_ledger.record(
            ToolResult(request=request, called=True, allowed=True, result=payload)
        )
        return self._payload_from_result(recorded)

    def _request(
        self,
        *,
        outbox_id: str,
        raw_decision: str,
        status: str | None,
        reviewer_id: str,
        reviewer_name: str,
        reason: str,
        final_message_text: str | None,
        trace_id: str | None,
        idempotency_key: str | None,
    ) -> ToolCallRequest:
        key = idempotency_key or _approval_idempotency_key(
            outbox_id=outbox_id,
            status=status or str(raw_decision or ""),
            final_message_text=final_message_text,
        )
        return ToolCallRequest(
            tool_name=ToolName.RECORD_APPROVAL_DECISION,
            arguments={
                "outbox_id": str(outbox_id or ""),
                "decision": str(raw_decision or ""),
                "normalized_status": status,
                "reviewer_id": reviewer_id,
                "reviewer_name": reviewer_name,
                "reason": reason,
                "trace_id": trace_id,
                "has_final_message_text": final_message_text is not None,
            },
            risk_level=RiskLevel.MEDIUM,
            execution_mode=ToolExecutionMode.STATE_WRITE,
            idempotency_key=key,
            reason="老板审批待发送草稿；只更新审批状态，不直接发送。",
        )

    def _approval_payload(
        self,
        *,
        outbox_item: dict[str, Any],
        status: str,
        reviewer_id: str,
        reviewer_name: str,
        reason: str,
        trace_id: str | None,
        already_decided: bool,
    ) -> dict[str, Any]:
        metadata = outbox_item.get("metadata") if isinstance(outbox_item.get("metadata"), dict) else {}
        approval = {
            "target_type": "outbox",
            "target_id": str(outbox_item.get("id") or ""),
            "status": status,
            "reviewer_id": str(metadata.get("reviewer_id") or reviewer_id),
            "reviewer_name": reviewer_name,
            "decision_reason": str(metadata.get("decision_reason") or reason),
            "decision_trace_id": str(metadata.get("decision_trace_id") or trace_id or ""),
            "decided_at": str(metadata.get("decided_at") or ""),
            "original_message_text": str(metadata.get("original_message_text") or outbox_item.get("message_text") or ""),
            "final_message_text": str(metadata.get("final_message_text") or outbox_item.get("message_text") or ""),
        }
        return {
            "ok": True,
            "approval": approval,
            "outbox_item": outbox_item,
            "already_decided": already_decided,
            "policy": "审批只更新待审批草稿状态，不触发真实发送。",
        }

    def _record_error(self, request: ToolCallRequest, code: str, message: str) -> dict[str, Any]:
        result = self.execution_ledger.record(
            ToolResult(
                request=request,
                called=False,
                allowed=False,
                result={"ok": False, "code": code, "message": message},
                error=message,
            )
        )
        return self._payload_from_result(result)

    def _payload_from_result(self, result: ToolResult) -> dict[str, Any]:
        payload = dict(result.result or {})
        payload.setdefault("ok", result.allowed and not result.error)
        payload["deduplicated"] = result.deduplicated
        payload["tool_result"] = result
        return payload


def normalize_approval_decision(decision: str) -> str | None:
    raw = str(decision or "").strip().lower()
    return APPROVAL_DECISION_ALIASES.get(raw)


def _approval_idempotency_key(
    *,
    outbox_id: str,
    status: str,
    final_message_text: str | None,
) -> str:
    digest = hashlib.sha256(str(final_message_text or "").encode("utf-8")).hexdigest()[:16]
    return f"approval_decision:{outbox_id}:{status}:{digest}"
