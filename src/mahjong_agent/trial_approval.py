from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

from .trial_persistence import ActionExecutor, ActionPlanProjector, ActionRecordFactory


TraceIdFactory = Callable[[], str]
NowFactory = Callable[[], datetime]
DateTimeParser = Callable[[Any], datetime | None]
ApprovalDecisionExecutor = Callable[[dict[str, Any]], dict[str, Any]]
StateLoader = Callable[[datetime], dict[str, Any]]
GameCacheUpdater = Callable[[str], None]


@dataclass
class TrialApprovalDecisionAdapter:
    """Controlled adapter for boss-trial approval decisions.

    This keeps HTTP/API payload handling and action-ledger projection outside
    the trial script. It delegates the actual state write to the supplied
    executor, so the current legacy trial tables keep working during migration.
    """

    approval_executor: ApprovalDecisionExecutor
    action_record_factory: ActionRecordFactory
    action_executor: ActionExecutor
    action_plan_projector: ActionPlanProjector
    state_loader: StateLoader
    trace_id_factory: TraceIdFactory
    now_factory: NowFactory
    parse_datetime: DateTimeParser
    game_cache_updater: GameCacheUpdater | None = None

    def decide(self, payload: dict[str, Any]) -> dict[str, Any]:
        trace_id = str(payload.get("trace_id") or self.trace_id_factory())
        now = self.parse_datetime(payload.get("now")) or self.now_factory()
        approval_id = str(payload.get("approval_id") or "").strip()
        target_type = str(payload.get("target_type") or "").strip()
        target_id = str(payload.get("target_id") or "").strip()
        decision = str(payload.get("decision") or payload.get("status") or "").strip().lower()
        action = self.action_record_factory(
            trace_id=trace_id,
            stage="approval_decision",
            action_name="record_approval_decision",
            arguments={
                "approval_id": approval_id,
                "target_type": target_type,
                "target_id": target_id,
                "decision": decision,
            },
            proposed_by="boss_manual",
            source="boss_manual",
            risk_level="high",
            approval_required=True,
            reason="老板审批待发送草稿，当前只更新审批状态，不直接发送真实消息。",
            now=now,
            validation={
                "allowed": True,
                "code": "manual_approved",
                "reason": "老板手动审批草稿，视为已审批动作。",
                "notes": ["审批通过不等于真实发送；真实发送仍需发送适配器执行。"],
            },
        )
        execution_payload = {**payload, "trace_id": trace_id, "now": now.isoformat()}
        result = self.action_executor(action, lambda: self.approval_executor(execution_payload))
        game_id = _game_id_from_approval_result(result)
        if game_id and self.game_cache_updater:
            self.game_cache_updater(game_id)
        result["agent_actions"] = [
            self.action_plan_projector(stage="approval_decision", source="boss_manual", action=action)
        ]
        result["state"] = self.state_loader(now)
        return result


def _game_id_from_approval_result(result: dict[str, Any]) -> str | None:
    approval = result.get("approval") if isinstance(result.get("approval"), dict) else {}
    metadata = approval.get("metadata") if isinstance(approval.get("metadata"), dict) else {}
    game_id = metadata.get("game_id")
    return str(game_id) if game_id else None
