"""Compatibility in-memory store imports for historical ``mahjong_agent_v3`` users."""

from __future__ import annotations

from mahjong_agent_runtime.store import (
    ALLOWED_GAME_TRANSITIONS,
    InMemoryAgentStore as InMemoryAgentStoreV3,
    invite_status_from_candidate_status,
    score_customer,
    score_requirement,
    smoke_matches,
    value_matches,
)

__all__ = [
    "ALLOWED_GAME_TRANSITIONS",
    "InMemoryAgentStoreV3",
    "invite_status_from_candidate_status",
    "score_customer",
    "score_requirement",
    "smoke_matches",
    "value_matches",
]
