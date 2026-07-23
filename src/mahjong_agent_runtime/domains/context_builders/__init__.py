"""Focused builders for bounded, privacy-aware Agent context."""

from .builder import AgentContextBuilder, BuiltContext, ContextPackingPolicy, DEFAULT_PROMPT_PATH
from .contracts import output_contract, planning_contract
from .customer_context import (
    CustomerContextBundle,
    build_customer_context,
    compact_candidate,
    compact_draft,
)
from .game_context import (
    GameContextBundle,
    active_game_visible_summary,
    build_game_context,
    compact_context_value,
    compact_game,
    compact_party,
    compact_requirement,
    sender_active_game_memberships,
)
from .message_context import build_message_reference_contract, resolve_quoted_message_context
from .relationship_context import (
    RelationshipContextBundle,
    build_relationship_context,
    customer_visibility_contract,
)
from .sanitization import (
    SAFE_CONTEXT_MESSAGE_METADATA_KEYS,
    SAFE_CONTEXT_QUOTED_METADATA_KEYS,
    context_text_preview,
    message_reference_for_context,
    sanitize_context_media_candidates,
    sanitize_context_observation_summary,
    sanitize_current_message_for_context,
    sanitize_message_metadata_for_context,
    sanitize_quoted_message_metadata_for_context,
    sanitize_resolved_message_reference_for_context,
)
from .tool_results import (
    compact_match,
    compact_tool_payload,
    compact_tool_result_dict,
    reference_duplicate_latest_tool_results,
    tool_result_for_context,
    turn_payload_for_context,
)

__all__ = [
    "AgentContextBuilder",
    "BuiltContext",
    "ContextPackingPolicy",
    "CustomerContextBundle",
    "DEFAULT_PROMPT_PATH",
    "GameContextBundle",
    "RelationshipContextBundle",
    "SAFE_CONTEXT_MESSAGE_METADATA_KEYS",
    "SAFE_CONTEXT_QUOTED_METADATA_KEYS",
    "active_game_visible_summary",
    "build_customer_context",
    "build_game_context",
    "build_message_reference_contract",
    "build_relationship_context",
    "compact_candidate",
    "compact_context_value",
    "compact_draft",
    "compact_game",
    "compact_match",
    "compact_party",
    "compact_requirement",
    "compact_tool_payload",
    "compact_tool_result_dict",
    "context_text_preview",
    "customer_visibility_contract",
    "message_reference_for_context",
    "output_contract",
    "planning_contract",
    "resolve_quoted_message_context",
    "reference_duplicate_latest_tool_results",
    "sanitize_context_media_candidates",
    "sanitize_context_observation_summary",
    "sanitize_current_message_for_context",
    "sanitize_message_metadata_for_context",
    "sanitize_quoted_message_metadata_for_context",
    "sanitize_resolved_message_reference_for_context",
    "sender_active_game_memberships",
    "tool_result_for_context",
    "turn_payload_for_context",
]
