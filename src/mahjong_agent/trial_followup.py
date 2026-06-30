from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

from .trial_persistence import ActionExecutor


DraftFactory = Callable[..., dict[str, Any]]
FallbackFactory = Callable[..., str]
TextGuard = Callable[..., str]
ToolPlanValidator = Callable[..., dict[str, Any]]
ValidatedActionLookup = Callable[[dict[str, Any] | None, str], dict[str, Any] | None]
PlanProjector = Callable[[dict[str, Any]], dict[str, Any]]
FollowupStateWriter = Callable[..., dict[str, Any]]
ToolAuditLogger = Callable[[str, str, dict[str, Any]], None]


@dataclass(slots=True)
class TrialOrganizerFollowupAdapter:
    """Controlled adapter for candidate-negotiation organizer followups.

    Candidate negotiation may require asking the game organizer whether a new
    time or duration is acceptable. This adapter owns the controlled tool flow:
    validate send_message, create only a pending followup, add approval data,
    and emit audit events. It does not call LLMs directly and never sends a
    real message.
    """

    fallback_factory: FallbackFactory
    draft_factory: DraftFactory
    text_guard: TextGuard
    tool_plan_validator: ToolPlanValidator
    validated_action_lookup: ValidatedActionLookup
    action_executor: ActionExecutor
    followup_state_writer: FollowupStateWriter
    plan_projector: PlanProjector
    tool_audit_logger: ToolAuditLogger

    def create(
        self,
        *,
        trace_id: str,
        classification: dict[str, Any],
        candidate_text: str,
        suggested_candidate_reply: str,
        outbox_item: dict[str, Any],
        game: dict[str, Any] | None,
        now: datetime,
    ) -> dict[str, Any] | None:
        if classification.get("feedback_type") != "candidate_negotiation" or not game:
            return None
        organizer_id = str(game.get("organizer_id") or "").strip()
        organizer_name = str(game.get("organizer_name") or "发起人").strip() or "发起人"
        game_id = str(game.get("id") or outbox_item.get("game_id") or "").strip()
        if not organizer_id or not game_id:
            return None

        fallback = self.fallback_factory(
            classification=classification,
            candidate_name=str(outbox_item.get("customer_name") or "候选人"),
            organizer_name=organizer_name,
        )
        draft = self.draft_factory(
            trace_id=trace_id,
            classification=classification,
            candidate_text=candidate_text,
            suggested_candidate_reply=suggested_candidate_reply,
            outbox_item=outbox_item,
            game=game,
            fallback=fallback,
            now=now,
        )
        message_text = self.text_guard(
            str(draft.get("text") or fallback),
            fallback=fallback,
            classification=classification,
            organizer_name=organizer_name,
        )
        requested_by = "llm" if draft.get("source") == "llm" and draft.get("should_create_message") is not False else "backend_fallback"
        followup_tool_plan = self.tool_plan_validator(
            trace_id=trace_id,
            plan={
                "source": requested_by,
                "stage": "organizer_followup_draft",
                "fallback_used": requested_by != "llm",
                "tool_calls": [
                    {
                        "tool_name": "send_message",
                        "arguments": {
                            "execution_mode": "create_pending_followup",
                            "game_id": game_id,
                            "related_outbox_id": outbox_item.get("id"),
                            "recipient_id": organizer_id,
                            "recipient_role": "organizer",
                        },
                        "reason": str(draft.get("reasoning_summary") or "候选人提出新条件，需发起人确认。"),
                        "requested_by": requested_by,
                    }
                ],
                "reasoning_summary": str(draft.get("reasoning_summary") or "候选人提出新条件，需发起人确认。"),
            },
            stage="organizer_followup_draft",
            game=game,
            missing_fields=[],
            tool_results={},
            now=now,
        )
        send_action = self._send_action(followup_tool_plan)
        send_action_record = self.validated_action_lookup(followup_tool_plan, "send_message")
        if send_action is None or send_action_record is None:
            return {
                "skipped": True,
                "status": "已拦截",
                "reason": "后端状态校验拒绝创建发起人 followup。",
                "agent_actions": [self.plan_projector(followup_tool_plan)],
                "direct_send_executed": False,
            }

        request_payload = self._request_payload(
            followup_tool_plan=followup_tool_plan,
            send_action=send_action,
            requested_by=requested_by,
            game_id=game_id,
            outbox_item=outbox_item,
            organizer_id=organizer_id,
            organizer_name=organizer_name,
        )
        self.tool_audit_logger(trace_id, "tool_request", request_payload)
        followup = self.action_executor(
            send_action_record,
            lambda: self.followup_state_writer(
                action=send_action_record,
                game_id=game_id,
                related_outbox_id=str(outbox_item.get("id") or ""),
                recipient_id=organizer_id,
                recipient_name=organizer_name,
                message_text=message_text,
                reason=str(draft.get("reasoning_summary") or "候选人提出新条件，需发起人确认。"),
                draft_source=str(draft.get("source") or "rules"),
            ),
        )
        agent_actions = [self.plan_projector(followup_tool_plan)]
        result = {
            **followup,
            "source": draft.get("source") or "rules",
            "model": draft.get("model"),
            "reasoning_summary": draft.get("reasoning_summary"),
            "needs_approval": True,
            "direct_send_executed": False,
            "agent_actions": agent_actions,
        }
        self.tool_audit_logger(
            trace_id,
            "tool_response",
            {
                **request_payload,
                "result_count": 1,
                "direct_send_executed": False,
                "deduplicated": bool(result.get("deduplicated")),
                "followup": {
                    "id": result.get("id"),
                    "recipient_name": result.get("recipient_name"),
                    "status": result.get("status"),
                    "message_text": result.get("message_text"),
                },
            },
        )
        return result

    def _send_action(self, followup_tool_plan: dict[str, Any]) -> dict[str, Any] | None:
        return next(
            (
                action
                for action in followup_tool_plan.get("tool_calls") or []
                if isinstance(action, dict) and action.get("tool_name") == "send_message"
            ),
            None,
        )

    def _request_payload(
        self,
        *,
        followup_tool_plan: dict[str, Any],
        send_action: dict[str, Any],
        requested_by: str,
        game_id: str,
        outbox_item: dict[str, Any],
        organizer_id: str,
        organizer_name: str,
    ) -> dict[str, Any]:
        return {
            "tool_name": "send_message",
            "called": True,
            "requested_by": send_action.get("requested_by") or requested_by,
            "tool_plan_source": followup_tool_plan.get("source"),
            "action_id": send_action.get("action_id"),
            "idempotency_key": send_action.get("idempotency_key"),
            "risk_level": "high",
            "approval_required": True,
            "direct_send_allowed": False,
            "execution_mode": (send_action.get("arguments") or {}).get("execution_mode") or "create_pending_followup",
            "call_reason": "候选人提出新条件，需先向发起人确认，当前只创建待审批消息。",
            "query": {
                "game_id": game_id,
                "related_outbox_id": outbox_item.get("id"),
                "recipient_id": organizer_id,
                "recipient_name": organizer_name,
                "recipient_role": "organizer",
            },
            "hard_filters": [
                "只能发给本局发起人或已确认玩家",
                "禁止直接发送真实消息",
                "必须写入待审批 followup_messages",
                "不能直接修改局时间或确认候选人入局",
            ],
        }
