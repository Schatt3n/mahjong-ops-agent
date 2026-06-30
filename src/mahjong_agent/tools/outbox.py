from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..workflow_models import GameRequirement, new_workflow_id


@dataclass(slots=True)
class PendingOutboxTool:
    max_drafts: int = 8

    def create_pending_invites(
        self,
        requirement: GameRequirement,
        candidates: list[dict[str, Any]],
        *,
        conversation_id: str,
        trace_id: str,
    ) -> dict[str, Any]:
        drafts: list[dict[str, Any]] = []
        for candidate in candidates[: self.max_drafts]:
            customer_id = str(candidate.get("customer_id") or "")
            display_name = str(candidate.get("display_name") or customer_id or "牌友")
            drafts.append(
                {
                    "id": new_workflow_id("outbox"),
                    "trace_id": trace_id,
                    "conversation_id": conversation_id,
                    "target_customer_id": customer_id,
                    "target_display_name": display_name,
                    "message_text": self._invite_text(display_name, requirement),
                    "status": "pending_approval",
                    "source": "tool_orchestrator",
                    "metadata": {
                        "candidate_score": candidate.get("score"),
                        "candidate_reasons": list(candidate.get("reasons") or []),
                    },
                }
            )
        return {
            "drafts": drafts,
            "result_count": len(drafts),
            "policy": "只创建待审批草稿，不自动发送。",
        }

    def _invite_text(self, display_name: str, requirement: GameRequirement) -> str:
        slots = requirement.slots
        time_text = _slot_value(slots, "start_at") or _start_mode_text(_slot_value(slots, "start_time_mode"))
        stake = _slot_value(slots, "stake")
        smoke = _smoke_text(_slot_value(slots, "smoke"))
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


def _slot_value(slots: dict[str, Any], name: str) -> Any:
    slot = slots.get(name)
    return slot.value if slot else None


def _start_mode_text(value: Any) -> str | None:
    if value == "people_ready":
        return "人齐开"
    if value == "fixed":
        return None
    return str(value) if value else None


def _smoke_text(value: Any) -> str | None:
    if value == "no_smoke":
        return "无烟"
    if value == "smoke_ok":
        return "有烟"
    if value == "any":
        return "烟都可"
    return str(value) if value else None


def _duration_text(slots: dict[str, Any]) -> str | None:
    duration = _slot_value(slots, "duration_hours")
    if duration:
        return f"约{duration}小时"
    mode = _slot_value(slots, "duration_mode")
    if mode == "overnight":
        return "通宵"
    return str(mode) if mode else None
