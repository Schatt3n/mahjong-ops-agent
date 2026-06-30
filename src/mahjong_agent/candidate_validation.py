from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .candidate_semantics import (
    candidate_action_for_feedback_type,
    feedback_type_for_candidate_action,
    normalize_candidate_proposed_action,
)


FallbackClassifier = Callable[[str, dict[str, Any] | None], dict[str, Any]]
NegotiationClassifier = Callable[[str, dict[str, Any] | None], dict[str, Any] | None]
GameFullChecker = Callable[[dict[str, Any] | None], bool]
ExtractedFactApplier = Callable[[dict[str, Any], dict[str, Any], dict[str, Any] | None], None]


STATE_CHANGING_FEEDBACK_TYPES = {"accepted", "arrived", "declined", "do_not_disturb"}


@dataclass(slots=True)
class CandidateActionProposalValidator:
    """Backend validator for candidate-reply action proposals.

    The LLM proposes semantic intent and actions. This validator owns the
    backend contract checks: action whitelist, confidence thresholds, terminal
    game states, full-game conflicts, and negotiation overrides. It does not
    call tools, mutate state, or generate outbound messages.
    """

    fallback_classifier: FallbackClassifier
    negotiation_classifier: NegotiationClassifier
    game_full_checker: GameFullChecker
    extracted_fact_applier: ExtractedFactApplier | None = None
    min_state_change_confidence: float = 0.68
    final_game_statuses: set[str] = field(default_factory=set)

    def validate(
        self,
        proposal: dict[str, Any],
        *,
        candidate_text: str,
        outbox_item: dict[str, Any],
        game: dict[str, Any] | None,
        fallback: dict[str, Any],
    ) -> dict[str, Any]:
        if proposal.get("source") != "llm":
            classification = dict(
                fallback.get("backend_fallback_classification")
                or self.fallback_classifier(candidate_text, game)
            )
            return {
                "classification": classification,
                "validated_action": candidate_action_for_feedback_type(str(classification.get("feedback_type") or "")),
                "validation": {
                    "accepted": True,
                    "mode": "fallback_rules",
                    "reason": "LLM 不可用，使用安全降级分类器。",
                },
            }

        action = normalize_candidate_proposed_action(
            str(proposal.get("proposed_action") or ""),
            semantic_type=str(proposal.get("semantic_type") or ""),
        )
        feedback_type = feedback_type_for_candidate_action(action)
        confidence = _safe_float(proposal.get("confidence")) or 0.0
        validation_notes: list[str] = []
        accepted = True

        if not feedback_type:
            accepted = False
            feedback_type = "candidate_question"
            validation_notes.append("LLM proposed_action 不在白名单内。")
        if confidence < self.min_state_change_confidence and feedback_type in STATE_CHANGING_FEEDBACK_TYPES:
            accepted = False
            feedback_type = "candidate_question"
            validation_notes.append(f"置信度 {confidence:.2f} 低于状态提交阈值。")

        contract_negotiation = self.negotiation_classifier(candidate_text, game)
        if feedback_type in {"accepted", "arrived"} and contract_negotiation:
            feedback_type = "candidate_negotiation"
            validation_notes.append("候选人回复包含与原局不同的时间/时长，转为待协商。")

        already_confirmed = str(outbox_item.get("status") or "") in {"已确认", "已到店"}
        game_status = str((game or {}).get("status") or "")
        if feedback_type in {"accepted", "arrived"} and game_status in self.final_game_statuses:
            accepted = False
            feedback_type = "candidate_question"
            validation_notes.append("当前局已归档，不能继续确认候选人。")
        if feedback_type in {"accepted", "arrived"} and not already_confirmed and self.game_full_checker(game):
            accepted = False
            feedback_type = "candidate_question"
            validation_notes.append("当前局缺口已满，不能继续确认新候选人。")

        classification = self._classification_from_validated_action(
            feedback_type=feedback_type,
            proposal=proposal,
            candidate_text=candidate_text,
            game=game,
        )
        if validation_notes:
            classification["validation_notes"] = validation_notes
        return {
            "classification": classification,
            "validated_action": candidate_action_for_feedback_type(str(classification.get("feedback_type") or "")),
            "validation": {
                "accepted": accepted and not validation_notes,
                "mode": "llm_proposal_backend_validated",
                "confidence": confidence,
                "notes": validation_notes,
            },
        }

    def _classification_from_validated_action(
        self,
        *,
        feedback_type: str,
        proposal: dict[str, Any],
        candidate_text: str,
        game: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if feedback_type == "candidate_negotiation":
            classification = self.negotiation_classifier(candidate_text, game)
            if not classification:
                classification = {
                    "intent": "candidate_negotiation",
                    "feedback_type": "candidate_negotiation",
                    "status": "待协商",
                }
            else:
                classification = dict(classification)
            if self.extracted_fact_applier:
                self.extracted_fact_applier(classification, proposal, game)
            return classification
        mapping = {
            "accepted": ("accepted", "已确认"),
            "arrived": ("arrived", "已到店"),
            "declined": ("declined", "拒绝"),
            "ask_later": ("ask_later", "下次再问"),
            "candidate_question": ("candidate_question", "待确认"),
            "do_not_disturb": ("do_not_disturb", "别再打扰"),
        }
        intent, status = mapping.get(feedback_type, ("candidate_question", "待确认"))
        return {
            "intent": intent,
            "feedback_type": feedback_type if feedback_type in mapping else "candidate_question",
            "status": status,
        }


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
