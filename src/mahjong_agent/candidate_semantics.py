from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable


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
