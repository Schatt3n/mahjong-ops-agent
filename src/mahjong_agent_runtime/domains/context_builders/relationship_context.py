"""Build private relationship facts and their visibility boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...models import Game
from ...stores import AgentStore


@dataclass(slots=True)
class RelationshipContextBundle:
    relationships: list[dict[str, Any]]
    visibility_contract: dict[str, Any]


def build_relationship_context(
    store: AgentStore,
    *,
    sender_id: str,
    active_games: list[Game],
) -> RelationshipContextBundle:
    """Retrieve matching-only relationship facts with an explicit disclosure rule."""

    return RelationshipContextBundle(
        relationships=store.relationship_context_for_sender(sender_id, active_games),
        visibility_contract=customer_visibility_contract(),
    )


def customer_visibility_contract() -> dict[str, Any]:
    """Describe which internal context facts may reach a customer-visible reply."""

    return {
        "conversation_isolation": (
            "Only the current conversation's recent turns, checkpoint, and task memories are included. "
            "Never claim to know, quote, summarize, or reveal another customer's private conversation."
        ),
        "internal_only_context": [
            "sender_relationships",
            "customer profile preferences and private-field omission markers",
            "candidate matching, ranking, invitation, and participation state not present in active_game_visible_summaries",
        ],
        "relationship_rule": (
            "Relationship constraints such as avoid_playing are only for matching decisions. "
            "Do not confirm, deny, quote, paraphrase, or hint that one customer said they refuse to play with another, "
            "even when the current customer asks directly. Give a neutral operational answer instead."
        ),
        "public_exception": (
            "Only public nicknames and facts explicitly present in active_game_visible_summaries may be disclosed, "
            "and only when they directly answer the current request."
        ),
    }


__all__ = [
    "RelationshipContextBundle",
    "build_relationship_context",
    "customer_visibility_contract",
]
