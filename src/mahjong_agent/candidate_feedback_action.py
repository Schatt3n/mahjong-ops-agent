from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from .candidate_semantics import candidate_action_for_feedback_type


RuntimePolicyValidator = Callable[..., dict[str, Any] | None]
StateWritePolicyValidator = Callable[..., dict[str, Any] | None]
ActionCompactor = Callable[[dict[str, Any]], dict[str, Any]]
ToolAuditLogger = Callable[[str, str, dict[str, Any]], None]


@dataclass(slots=True)
class CandidateFeedbackActionService:
    """Build controlled action records for candidate feedback.

    This service only constructs and audits the backend action contract. It does
    not execute the action, update outbox state, or send messages.
    """

    protocol_version: str
    runtime_policy_validator: RuntimePolicyValidator | None = None
    state_write_policy_validator: StateWritePolicyValidator | None = None
    action_compactor: ActionCompactor | None = None
    tool_audit_logger: ToolAuditLogger | None = None
    final_game_statuses: set[str] = field(default_factory=set)

    def build(
        self,
        *,
        trace_id: str,
        proposal: dict[str, Any],
        validation: dict[str, Any],
        classification: dict[str, Any],
        outbox_item: dict[str, Any],
        game: dict[str, Any] | None,
        now: datetime,
    ) -> dict[str, Any]:
        feedback_type = str(classification.get("feedback_type") or "candidate_question")
        validated_action = str(validation.get("validated_action") or candidate_action_for_feedback_type(feedback_type))
        proposed_action = str(proposal.get("proposed_action") or "")
        game_id = str(outbox_item.get("game_id") or (game or {}).get("id") or "")
        args = {
            "game_id": game_id,
            "outbox_id": outbox_item.get("id"),
            "customer_id": outbox_item.get("customer_id"),
            "feedback_type": feedback_type,
            "validated_action": validated_action,
            "proposed_action": proposed_action,
        }
        action_hash = self._action_hash(trace_id=trace_id, arguments=args)
        backend_validation = validation.get("validation") if isinstance(validation.get("validation"), dict) else {}
        validation_notes = list(backend_validation.get("notes") or [])
        allowed = True
        code = "allowed"
        reason = "候选人反馈动作通过后端校验，可以写入状态。"
        if proposed_action and proposed_action != validated_action:
            code = "normalized_or_downgraded"
            reason = "模型动作已由后端归一化或降级为安全状态写入。"
        if backend_validation.get("accepted") is False:
            code = "downgraded_to_safe_feedback"
            reason = "模型提案未达到直接状态提交条件，后端已降级为安全反馈状态。"
        if str((game or {}).get("status") or "") in self.final_game_statuses and feedback_type in {"accepted", "arrived"}:
            allowed = False
            code = "final_game_reject"
            reason = "当前局已归档，拒绝确认候选人。"
        risk_level = "medium" if feedback_type in {"accepted", "arrived", "candidate_negotiation", "do_not_disturb"} else "low"
        action = {
            "action_id": f"act_{action_hash}",
            "idempotency_key": f"{trace_id}:candidate_feedback:record_candidate_feedback:{action_hash}",
            "protocol": self.protocol_version,
            "stage": "candidate_feedback",
            "tool_name": "record_candidate_feedback",
            "arguments": args,
            "proposed_by": str(proposal.get("source") or "unknown"),
            "source": str(proposal.get("source") or "unknown"),
            "risk_level": risk_level,
            "side_effect": True,
            "approval_required": False,
            "reason": str(proposal.get("reasoning_summary") or "候选人回复触发状态写入。")[:240],
            "created_at": now.isoformat(),
            "validation": {
                "allowed": allowed,
                "code": code,
                "reason": reason,
                "notes": validation_notes,
                "effective_feedback_type": feedback_type,
                "effective_action": validated_action,
                "backend_validation": backend_validation,
            },
        }
        allowed = self._apply_runtime_policy(action)
        allowed = self._apply_state_write_policy(action) and allowed
        self._audit(trace_id=trace_id, action=action, allowed=allowed)
        return action

    def _action_hash(self, *, trace_id: str, arguments: dict[str, Any]) -> str:
        stable_payload = json.dumps(
            {
                "trace_id": trace_id,
                "stage": "candidate_feedback",
                "tool_name": "record_candidate_feedback",
                "arguments": arguments,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(stable_payload.encode("utf-8")).hexdigest()[:16]

    def _apply_runtime_policy(self, action: dict[str, Any]) -> bool:
        if not self.runtime_policy_validator:
            return bool(action.get("validation", {}).get("allowed"))
        policy_verdict = self.runtime_policy_validator(
            stage="candidate_feedback",
            action_name="record_candidate_feedback",
            side_effect=True,
        )
        if not policy_verdict:
            return bool(action.get("validation", {}).get("allowed"))
        self._merge_policy_verdict(action, policy_verdict)
        return False

    def _apply_state_write_policy(self, action: dict[str, Any]) -> bool:
        if not self.state_write_policy_validator:
            return bool(action.get("validation", {}).get("allowed"))
        state_write_policy_verdict = self.state_write_policy_validator(
            stage="candidate_feedback",
            action_name="record_candidate_feedback",
            proposed_by=str(action.get("proposed_by") or ""),
            source=str(action.get("source") or ""),
        )
        if not state_write_policy_verdict:
            return bool(action.get("validation", {}).get("allowed"))
        self._merge_policy_verdict(action, state_write_policy_verdict)
        return False

    def _merge_policy_verdict(self, action: dict[str, Any], verdict: dict[str, Any]) -> None:
        original_notes = [str(item) for item in action["validation"].get("notes") or []]
        action["validation"] = {
            **action["validation"],
            **verdict,
            "notes": original_notes + [str(item) for item in verdict.get("notes") or []],
        }

    def _audit(self, *, trace_id: str, action: dict[str, Any], allowed: bool) -> None:
        if not self.tool_audit_logger:
            return
        compact = self.action_compactor(action) if self.action_compactor else action
        self.tool_audit_logger(
            trace_id,
            "action_validation",
            {
                "protocol": self.protocol_version,
                "stage": "candidate_feedback",
                "source": action["source"],
                "proposed_count": 1,
                "allowed_count": 1 if allowed else 0,
                "rejected_count": 0 if allowed else 1,
                "validated_actions": [compact] if allowed else [],
                "rejected_actions": [compact] if not allowed else [],
            },
        )
