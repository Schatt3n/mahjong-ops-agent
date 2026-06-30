from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re
from typing import Any, Callable


VALID_CANDIDATE_SEMANTIC_TYPES = {
    "accepted",
    "arrived",
    "declined",
    "ask_later",
    "candidate_question",
    "candidate_negotiation",
    "do_not_disturb",
    "uncertain",
}

VALID_CANDIDATE_ACTIONS = {
    "mark_candidate_confirmed",
    "mark_candidate_arrived",
    "mark_candidate_declined",
    "mark_candidate_ask_later",
    "answer_candidate_question",
    "start_negotiation",
    "set_do_not_disturb",
    "request_human_review",
    "no_state_change",
}

SEMANTIC_TO_ACTION = {
    "accepted": "mark_candidate_confirmed",
    "arrived": "mark_candidate_arrived",
    "declined": "mark_candidate_declined",
    "ask_later": "mark_candidate_ask_later",
    "candidate_question": "answer_candidate_question",
    "candidate_negotiation": "start_negotiation",
    "do_not_disturb": "set_do_not_disturb",
    "uncertain": "request_human_review",
}

ACTION_TO_FEEDBACK_TYPE = {
    "mark_candidate_confirmed": "accepted",
    "mark_candidate_arrived": "arrived",
    "mark_candidate_declined": "declined",
    "mark_candidate_ask_later": "ask_later",
    "answer_candidate_question": "candidate_question",
    "start_negotiation": "candidate_negotiation",
    "set_do_not_disturb": "do_not_disturb",
    "request_human_review": "candidate_question",
    "no_state_change": "candidate_question",
}

FEEDBACK_TYPE_TO_ACTION = {
    "accepted": "mark_candidate_confirmed",
    "arrived": "mark_candidate_arrived",
    "declined": "mark_candidate_declined",
    "ask_later": "mark_candidate_ask_later",
    "candidate_question": "answer_candidate_question",
    "candidate_negotiation": "start_negotiation",
    "do_not_disturb": "set_do_not_disturb",
}


FallbackProposalFactory = Callable[[str, dict[str, Any], dict[str, Any] | None], dict[str, Any]]
LLMProposalFactory = Callable[..., dict[str, Any]]


@dataclass(frozen=True, slots=True)
class CandidateSemanticProposalResult:
    """Candidate reply semantic contract result.

    The proposal is what the model or fallback semantic resolver suggests.
    The fallback is always preserved so the backend validator can compare or
    safely degrade without re-running semantic heuristics.
    """

    proposal: dict[str, Any]
    fallback: dict[str, Any]


@dataclass(slots=True)
class CandidateSemanticProposalAdapter:
    """Build a candidate-reply semantic proposal without side effects.

    This adapter is intentionally limited to the LLM/fallback proposal boundary.
    It does not validate actions, execute tools, update state, or send messages.
    """

    fallback_proposal_factory: FallbackProposalFactory
    llm_proposal_factory: LLMProposalFactory

    def propose(
        self,
        *,
        trace_id: str,
        candidate_text: str,
        outbox_item: dict[str, Any],
        game: dict[str, Any] | None,
        now: datetime,
    ) -> CandidateSemanticProposalResult:
        fallback = self._safe_fallback(candidate_text, outbox_item, game)
        try:
            proposal = self.llm_proposal_factory(
                trace_id=trace_id,
                candidate_text=candidate_text,
                outbox_item=outbox_item,
                game=game,
                fallback=fallback,
                now=now,
            )
        except Exception as exc:
            proposal = self._fallback_with_note(
                fallback,
                f"LLM candidate semantic proposal raised {type(exc).__name__}: {exc}",
            )
        if not isinstance(proposal, dict):
            proposal = self._fallback_with_note(
                fallback,
                f"LLM candidate semantic proposal returned {type(proposal).__name__}, expected dict.",
            )
        return CandidateSemanticProposalResult(proposal=dict(proposal), fallback=fallback)

    def _safe_fallback(
        self,
        candidate_text: str,
        outbox_item: dict[str, Any],
        game: dict[str, Any] | None,
    ) -> dict[str, Any]:
        fallback = self.fallback_proposal_factory(candidate_text, outbox_item, game)
        if isinstance(fallback, dict):
            return dict(fallback)
        return {
            "source": "rules",
            "model": None,
            "semantic_type": "uncertain",
            "proposed_action": "request_human_review",
            "confidence": 0.0,
            "reply_text": "",
            "risk_level": "medium",
            "reasoning_summary": "Fallback semantic resolver returned invalid contract.",
            "notes": [f"fallback_return_type={type(fallback).__name__}"],
            "extracted_facts": {},
            "backend_fallback_classification": {
                "intent": "candidate_question",
                "feedback_type": "candidate_question",
                "status": "待确认",
            },
            "outbox_id": outbox_item.get("id"),
        }

    def _fallback_with_note(self, fallback: dict[str, Any], note: str) -> dict[str, Any]:
        proposal = dict(fallback)
        notes = proposal.get("notes") if isinstance(proposal.get("notes"), list) else []
        proposal["notes"] = [*notes, note]
        proposal["reasoning_summary"] = note
        return proposal


def normalize_candidate_semantic_type(value: str) -> str:
    normalized = re.sub(r"[\s_-]+", "", str(value or "").lower())
    aliases = {
        "accept": "accepted",
        "accepted": "accepted",
        "confirm": "accepted",
        "confirmed": "accepted",
        "candidateaccept": "accepted",
        "arrive": "arrived",
        "arrived": "arrived",
        "decline": "declined",
        "declined": "declined",
        "reject": "declined",
        "asklater": "ask_later",
        "later": "ask_later",
        "question": "candidate_question",
        "candidatequestion": "candidate_question",
        "negotiation": "candidate_negotiation",
        "candidatenegotiation": "candidate_negotiation",
        "donotdisturb": "do_not_disturb",
        "dnd": "do_not_disturb",
        "uncertain": "uncertain",
    }
    if normalized in aliases:
        return aliases[normalized]
    value = str(value or "")
    return value if value in VALID_CANDIDATE_SEMANTIC_TYPES else "uncertain"


def normalize_candidate_proposed_action(value: str, *, semantic_type: str) -> str:
    normalized = re.sub(r"[\s_-]+", "", str(value or "").lower())
    aliases = {
        "markcandidateconfirmed": "mark_candidate_confirmed",
        "confirmcandidate": "mark_candidate_confirmed",
        "markconfirmed": "mark_candidate_confirmed",
        "markcandidatearrived": "mark_candidate_arrived",
        "markarrived": "mark_candidate_arrived",
        "markcandidatedeclined": "mark_candidate_declined",
        "declinecandidate": "mark_candidate_declined",
        "markcandidateasklater": "mark_candidate_ask_later",
        "asklater": "mark_candidate_ask_later",
        "answercandidatequestion": "answer_candidate_question",
        "answerquestion": "answer_candidate_question",
        "startnegotiation": "start_negotiation",
        "negotiation": "start_negotiation",
        "setdonotdisturb": "set_do_not_disturb",
        "requesthumanreview": "request_human_review",
        "nostatechange": "no_state_change",
    }
    action = aliases.get(normalized, str(value or ""))
    if action in VALID_CANDIDATE_ACTIONS:
        return action
    return SEMANTIC_TO_ACTION.get(semantic_type, "request_human_review")


def feedback_type_for_candidate_action(action: str) -> str:
    return ACTION_TO_FEEDBACK_TYPE.get(str(action or ""), "")


def candidate_action_for_feedback_type(feedback_type: str) -> str:
    return FEEDBACK_TYPE_TO_ACTION.get(str(feedback_type or ""), "request_human_review")
