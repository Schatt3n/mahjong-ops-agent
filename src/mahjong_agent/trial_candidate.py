from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

from .candidate_semantics import CandidateSemanticProposalResult
from .trial_persistence import ActionExecutor, ActionPlanProjector


TraceIdFactory = Callable[[], str]
NowFactory = Callable[[], datetime]
DateTimeParser = Callable[[Any], datetime | None]
OutboxLookup = Callable[[str], dict[str, Any] | None]
GameLookup = Callable[[str], dict[str, Any] | None]
FallbackProposalFactory = Callable[[str, dict[str, Any], dict[str, Any] | None], dict[str, Any]]
LLMProposalFactory = Callable[..., dict[str, Any]]
SemanticProposalFactory = Callable[..., CandidateSemanticProposalResult]
ProposalValidator = Callable[..., dict[str, Any]]
CandidateReplyFactory = Callable[[dict[str, Any], str, dict[str, Any], dict[str, Any] | None], str]
CandidateReplyGuard = Callable[..., str]
CandidateActionFactory = Callable[..., dict[str, Any]]
OrganizerFollowupFactory = Callable[..., dict[str, Any] | None]
FeedbackRecorder = Callable[[dict[str, Any]], dict[str, Any]]
StateLoader = Callable[[datetime], dict[str, Any]]
CustomerReloader = Callable[[], None]
GameCacheUpdater = Callable[[str], None]
JsonDumper = Callable[[Any], str]


@dataclass
class TrialCandidateMessageAdapter:
    """Controlled adapter for candidate replies in the trial UI.

    The current trial page still owns the concrete semantic helpers and legacy
    SQLite writes. This adapter moves request handling and response assembly out
    of the script so those pieces can be replaced incrementally.
    """

    outbox_lookup: OutboxLookup
    game_lookup: GameLookup
    proposal_validator: ProposalValidator
    candidate_reply_factory: CandidateReplyFactory
    candidate_reply_guard: CandidateReplyGuard
    candidate_action_factory: CandidateActionFactory
    organizer_followup_factory: OrganizerFollowupFactory
    action_executor: ActionExecutor
    action_plan_projector: ActionPlanProjector
    feedback_recorder: FeedbackRecorder
    state_loader: StateLoader
    trace_id_factory: TraceIdFactory
    now_factory: NowFactory
    parse_datetime: DateTimeParser
    fallback_proposal_factory: FallbackProposalFactory | None = None
    llm_proposal_factory: LLMProposalFactory | None = None
    semantic_proposal_factory: SemanticProposalFactory | None = None
    customer_reloader: CustomerReloader | None = None
    game_cache_updater: GameCacheUpdater | None = None
    json_dumper: JsonDumper | None = None

    def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        trace_id = str(payload.get("trace_id") or self.trace_id_factory())
        outbox_id = str(payload.get("outbox_id") or "").strip()
        text = str(payload.get("text") or "").strip()
        now = self.parse_datetime(payload.get("now")) or self.now_factory()
        if not outbox_id:
            raise ValueError("缺少 outbox_id")
        if not text:
            raise ValueError("候选人回复不能为空")
        item = self.outbox_lookup(outbox_id)
        if not item:
            raise ValueError("找不到这条候选人邀约")
        game = self.game_lookup(str(item["game_id"]))
        semantic_result = self._semantic_proposal(
            trace_id=trace_id,
            candidate_text=text,
            outbox_item=item,
            game=game,
            now=now,
        )
        fallback_proposal = semantic_result.fallback
        proposal = semantic_result.proposal
        validation = self.proposal_validator(
            proposal,
            candidate_text=text,
            outbox_item=item,
            game=game,
            fallback=fallback_proposal,
        )
        classification = validation["classification"]
        fallback_reply = self.candidate_reply_factory(classification, text, item, game)
        suggested_boss_reply = self.candidate_reply_guard(
            str(proposal.get("reply_text") or fallback_reply),
            fallback=fallback_reply,
            classification=classification,
        )
        candidate_state_action = self.candidate_action_factory(
            trace_id=trace_id,
            proposal=proposal,
            validation=validation,
            classification=classification,
            outbox_item=item,
            game=game,
            now=now,
        )
        organizer_followup = self.organizer_followup_factory(
            trace_id=trace_id,
            classification=classification,
            candidate_text=text,
            suggested_candidate_reply=suggested_boss_reply,
            outbox_item=item,
            game=game,
            now=now,
        )
        feedback_payload = self._feedback_payload(
            payload=payload,
            item=item,
            outbox_id=outbox_id,
            text=text,
            classification=classification,
            proposal=proposal,
            validation=validation,
            candidate_state_action=candidate_state_action,
            suggested_boss_reply=suggested_boss_reply,
        )
        feedback_result = self.action_executor(
            candidate_state_action,
            lambda: self.feedback_recorder(feedback_payload),
        )
        if feedback_result.get("ok") and not feedback_result.get("deduplicated") and self.customer_reloader:
            self.customer_reloader()
        agent_actions = [
            self.action_plan_projector(
                stage="candidate_feedback",
                source=str(proposal.get("source") or "unknown"),
                action=candidate_state_action,
            )
        ]
        if isinstance(organizer_followup, dict) and organizer_followup.get("agent_actions"):
            agent_actions.extend(
                entry for entry in organizer_followup.get("agent_actions") or [] if isinstance(entry, dict)
            )
        updated_item = self.outbox_lookup(outbox_id) or item
        state = self.state_loader(now)
        if self.game_cache_updater:
            self.game_cache_updater(str(item["game_id"]))
        return {
            "ok": bool(feedback_result.get("ok", True)),
            "rejected": bool(feedback_result.get("rejected")),
            "reason": feedback_result.get("reason"),
            "candidate_message": {
                **classification,
                "text": text,
                "outbox_id": outbox_id,
                "customer_id": item["customer_id"],
                "customer_name": item["customer_name"],
                "suggested_boss_reply": suggested_boss_reply,
                "reply_source": proposal.get("source"),
                "model": proposal.get("model"),
                "semantic_type": proposal.get("semantic_type"),
                "proposed_action": proposal.get("proposed_action"),
                "validated_action": validation.get("validated_action"),
                "semantic_confidence": proposal.get("confidence"),
                "reasoning_summary": proposal.get("reasoning_summary"),
                "validation": validation.get("validation"),
            },
            "agent_actions": agent_actions,
            "organizer_followup": organizer_followup,
            "outbox_item": updated_item,
            "auto_success": feedback_result.get("auto_success"),
            "state": state,
        }

    def _semantic_proposal(
        self,
        *,
        trace_id: str,
        candidate_text: str,
        outbox_item: dict[str, Any],
        game: dict[str, Any] | None,
        now: datetime,
    ) -> CandidateSemanticProposalResult:
        if self.semantic_proposal_factory:
            return self.semantic_proposal_factory(
                trace_id=trace_id,
                candidate_text=candidate_text,
                outbox_item=outbox_item,
                game=game,
                now=now,
            )
        if not self.fallback_proposal_factory or not self.llm_proposal_factory:
            raise RuntimeError("candidate semantic proposal factory is not configured")
        fallback = self.fallback_proposal_factory(candidate_text, outbox_item, game)
        proposal = self.llm_proposal_factory(
            trace_id=trace_id,
            candidate_text=candidate_text,
            outbox_item=outbox_item,
            game=game,
            fallback=fallback,
            now=now,
        )
        return CandidateSemanticProposalResult(proposal=proposal, fallback=fallback)

    def _feedback_payload(
        self,
        *,
        payload: dict[str, Any],
        item: dict[str, Any],
        outbox_id: str,
        text: str,
        classification: dict[str, Any],
        proposal: dict[str, Any],
        validation: dict[str, Any],
        candidate_state_action: dict[str, Any],
        suggested_boss_reply: str,
    ) -> dict[str, Any]:
        return {
            "game_id": item["game_id"],
            "outbox_id": outbox_id,
            "customer_id": item["customer_id"],
            "feedback_type": classification["feedback_type"],
            "notes": self._json_dumps(
                {
                    "kind": "candidate_message",
                    "candidate_text": text,
                    "boss_reply": suggested_boss_reply,
                    "classification": classification,
                    "semantic_proposal": proposal,
                    "validation": validation,
                    "controlled_action": candidate_state_action,
                }
            ),
            "profile_note": f"邀约回复：{text}",
            "now": payload.get("now"),
        }

    def _json_dumps(self, value: Any) -> str:
        if self.json_dumper:
            return self.json_dumper(value)
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
