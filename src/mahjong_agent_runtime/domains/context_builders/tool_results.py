"""Compact tool history before it is fed back to the main model."""

from __future__ import annotations

import json
from typing import Any

from ...models import ConversationTurn, ToolResult
from .customer_context import compact_candidate, compact_draft
from .game_context import compact_game
from .sanitization import sanitize_message_metadata_for_context


PASSTHROUGH_TOOL_RESULT_KEYS = (
    "requirement",
    "reference_requirement",
    "customer_reply_contract",
    "configured",
    "start_at",
    "end_at",
    "room_count",
    "available_room_ids",
    "occupied_room_ids",
    "available_count",
    "recorded_status",
    "next_step_policy",
    "approved",
    "needs_human",
    "raw_approved",
    "reasoning_summary",
    "violations",
    "item_reviews",
    "instruction",
    "review_scope",
    "items",
    "exclude_customer_ids",
    "continuation",
    "stale_run",
    "current_version",
    "run_version",
)


def turn_payload_for_context(turn: ConversationTurn) -> dict[str, Any]:
    """Serialize one turn and compact persisted tool results exactly once."""

    payload = turn.to_dict()
    if payload.get("role") == "user":
        payload["metadata"] = sanitize_message_metadata_for_context(payload.get("metadata"))
    if payload.get("role") != "tool":
        return payload
    try:
        raw_results = json.loads(str(payload.get("content") or "[]"))
    except json.JSONDecodeError:
        return payload
    if not isinstance(raw_results, list):
        return payload
    payload["content"] = json.dumps(
        [compact_tool_result_dict(item) for item in raw_results],
        ensure_ascii=False,
    )
    payload["metadata"] = {**dict(payload.get("metadata") or {}), "compacted_for_context": True}
    return payload


def tool_result_for_context(result: ToolResult) -> dict[str, Any]:
    return compact_tool_result_dict(result.to_dict())


def reference_duplicate_latest_tool_results(
    latest_results: list[dict[str, Any]],
    turn_evidence: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Replace duplicated latest feedback with pointers to full turn evidence.

    ``previous_tool_results`` is the recent-feedback view and
    ``turn_tool_evidence`` is the complete ordered record. Near the token
    budget, serializing the same result twice adds cost without adding facts.
    The pointer preserves latest-result ordering and execution status while
    the referenced evidence item retains the complete payload.
    """

    referenced: list[dict[str, Any]] = []
    compacted_count = 0
    claimed_indexes: set[int] = set()
    for latest in latest_results:
        evidence_index = _matching_evidence_index(
            latest,
            turn_evidence,
            claimed_indexes=claimed_indexes,
        )
        if evidence_index is None:
            referenced.append(latest)
            continue
        claimed_indexes.add(evidence_index)
        referenced.append(
            {
                "name": latest.get("name"),
                "called": latest.get("called"),
                "allowed": latest.get("allowed"),
                "call_id": latest.get("call_id"),
                "error": latest.get("error"),
                "deduplicated": latest.get("deduplicated", False),
                "result_reference": {
                    "source": "turn_tool_evidence",
                    "index": evidence_index,
                    "call_id": latest.get("call_id"),
                },
            }
        )
        compacted_count += 1
    return referenced, compacted_count


def _matching_evidence_index(
    latest: dict[str, Any],
    turn_evidence: list[dict[str, Any]],
    *,
    claimed_indexes: set[int],
) -> int | None:
    """Find the latest equivalent evidence item without collapsing duplicates."""

    call_id = latest.get("call_id")
    for index in range(len(turn_evidence) - 1, -1, -1):
        if index in claimed_indexes:
            continue
        candidate = turn_evidence[index]
        if call_id and candidate.get("call_id") == call_id:
            return index
        if not call_id and candidate == latest:
            return index
    return None


def compact_tool_result_dict(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {
            "name": "unknown",
            "called": False,
            "allowed": False,
            "result": {},
            "error": "invalid tool result payload",
        }
    compact: dict[str, Any] = {
        "name": raw.get("name"),
        "called": raw.get("called"),
        "allowed": raw.get("allowed"),
        "call_id": raw.get("call_id"),
        "error": raw.get("error"),
        "deduplicated": raw.get("deduplicated", False),
        "result": compact_tool_payload(raw.get("result") or {}),
    }
    if raw.get("state_transitions"):
        compact["state_transitions"] = [
            {
                "entity_type": item.get("entity_type"),
                "entity_id": item.get("entity_id"),
                "from_status": item.get("from_status"),
                "to_status": item.get("to_status"),
                "reason": item.get("reason"),
            }
            for item in raw.get("state_transitions") or []
            if isinstance(item, dict)
        ][:8]
    return compact


def compact_tool_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    compact: dict[str, Any] = {
        key: payload[key]
        for key in PASSTHROUGH_TOOL_RESULT_KEYS
        if key in payload
    }
    if "matches" in payload:
        compact["matches"] = [compact_match(item) for item in list(payload.get("matches") or [])[:5]]
        compact["match_count"] = len(payload.get("matches") or [])
    if "candidates" in payload:
        compact["candidates"] = [
            compact_candidate(item)
            for item in list(payload.get("candidates") or [])[:12]
        ]
        compact["candidate_count"] = len(payload.get("candidates") or [])
    if "game" in payload:
        compact["game"] = compact_game(payload.get("game"))
    if "drafts" in payload:
        compact["drafts"] = [compact_draft(item) for item in list(payload.get("drafts") or [])[:20]]
        compact["draft_count"] = len(payload.get("drafts") or [])
    if "checkpoint" in payload and isinstance(payload.get("checkpoint"), dict):
        checkpoint = payload["checkpoint"]
        compact["checkpoint"] = {
            "summary": checkpoint.get("summary"),
            "facts": checkpoint.get("facts"),
            "open_questions": checkpoint.get("open_questions"),
        }
    if "badcase" in payload:
        compact["badcase"] = payload["badcase"]
    return compact


def compact_match(match: Any) -> dict[str, Any]:
    if not isinstance(match, dict):
        return {}
    return {
        "score": match.get("score"),
        "reasons": match.get("reasons"),
        "join_projection": match.get("join_projection"),
        "game": compact_game(match.get("game")),
    }


__all__ = [
    "PASSTHROUGH_TOOL_RESULT_KEYS",
    "compact_match",
    "compact_tool_payload",
    "compact_tool_result_dict",
    "reference_duplicate_latest_tool_results",
    "tool_result_for_context",
    "turn_payload_for_context",
]
