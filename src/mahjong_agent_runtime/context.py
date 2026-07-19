from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import ConversationTurn, MessageReference, ToolResult, UserMessage
from .store import (
    InMemoryAgentStore,
    customer_visible_name,
    game_for_model_context,
    outbound_message_draft_for_model_context,
)
from .tools import ToolGateway
from .token_estimation import estimate_tokens as shared_estimate_tokens


DEFAULT_PROMPT_PATH = Path(__file__).with_name("prompts").joinpath("agent_runtime_system.md")


SAFE_CONTEXT_MESSAGE_METADATA_KEYS = {
    "channel",
    "platform_name",
    "source",
    "message_type",
    "source_message_id",
    "is_room",
    "self_message",
    "has_text",
    "text_source",
    "modalities",
    "media_candidates",
    "raw_observation_summary",
    "media_requires_transcription",
    "media_requires_ocr",
    "transcript_confidence",
    "ocr_confidence",
    "language",
}


SAFE_CONTEXT_QUOTED_METADATA_KEYS = {
    "source",
    "raw_chatusr",
    "platform_message_id",
    "platformMessageId",
    "source_message_id",
    "sourceMessageId",
    "message_type",
    "text_source",
    "channel",
    "resolved_message_reference",
}


@dataclass(slots=True)
class BuiltContext:
    messages: list[dict[str, str]]
    payload: dict[str, Any]
    audit: dict[str, Any]


@dataclass(slots=True)
class ContextPackingPolicy:
    max_turns_considered: int = 60
    max_recent_conversation_tokens: int = 4_000

    def pack_turns(self, turns: list[ConversationTurn]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Pack raw turns into the bounded recent_conversation window.

        This method only handles the local recent window budget. Higher-level checkpoint
        pruning happens in AgentContextBuilder because it needs access to checkpoint time.
        """

        considered = list(turns)[-self.max_turns_considered :]
        included_reversed: list[dict[str, Any]] = []
        estimated_tokens = 0
        omitted_for_budget = 0
        for turn in reversed(considered):
            payload = turn_payload_for_context(turn)
            turn_tokens = estimate_tokens(payload)
            if included_reversed and estimated_tokens + turn_tokens > self.max_recent_conversation_tokens:
                omitted_for_budget += 1
                continue
            included_reversed.append(payload)
            estimated_tokens += turn_tokens
        included = list(reversed(included_reversed))
        omitted_before_window = max(0, len(turns) - len(considered))
        audit = {
            "total_turns_available": len(turns),
            "included_turn_count": len(included),
            "omitted_turn_count": omitted_before_window + omitted_for_budget,
            "omitted_before_window": omitted_before_window,
            "omitted_for_budget": omitted_for_budget,
            "estimated_recent_conversation_tokens": estimated_tokens,
        }
        return included, audit


@dataclass(slots=True)
class AgentContextBuilder:
    store: InMemoryAgentStore
    tool_gateway: ToolGateway
    prompt_path: Path = DEFAULT_PROMPT_PATH
    packing_policy: ContextPackingPolicy = field(default_factory=ContextPackingPolicy)

    def build(
        self,
        message: UserMessage,
        *,
        trace_id: str,
        previous_tool_results: list[ToolResult] | None = None,
        run_id: str | None = None,
        run_version: int | None = None,
    ) -> BuiltContext:
        prompt = self.prompt_path.read_text(encoding="utf-8")
        task_context = self.store.current_task_context(message.conversation_id, message.sender_id)
        checkpoint = self.store.get_conversation_checkpoint(message.conversation_id)
        raw_turns = self.store.recent_turns(message.conversation_id, self.packing_policy.max_turns_considered)
        omitted_before_task_context = 0
        checkpoint_excluded_by_task_context = False
        if task_context is not None:
            task_turns: list[ConversationTurn] = []
            for turn in raw_turns:
                turn_task_context_id = str(turn.metadata.get("task_context_id") or "")
                belongs_to_current = turn_task_context_id == task_context.task_context_id
                legacy_inside_window = not turn_task_context_id and turn.occurred_at >= task_context.started_at
                if belongs_to_current or legacy_inside_window:
                    task_turns.append(turn)
                else:
                    omitted_before_task_context += 1
            raw_turns = task_turns
            checkpoint_matches = checkpoint is not None and (
                checkpoint.task_context_id == task_context.task_context_id
                or (
                    checkpoint.task_context_id is None
                    and checkpoint.updated_at >= task_context.started_at
                )
            )
            if checkpoint is not None and not checkpoint_matches:
                checkpoint = None
                checkpoint_excluded_by_task_context = True
        # The loop persists tool turns for audit/replay and also passes the same
        # results explicitly to the next model step. Keep one authoritative copy
        # in the prompt so a large tool response is not charged twice.
        deduplicated_current_trace_tool_turns = 0
        if previous_tool_results:
            retained_turns: list[ConversationTurn] = []
            for turn in raw_turns:
                if turn.role.value == "tool" and turn.trace_id == trace_id:
                    deduplicated_current_trace_tool_turns += 1
                    continue
                retained_turns.append(turn)
            raw_turns = retained_turns
        checkpoint_covered_turn_count = 0
        if checkpoint is not None:
            turns_after_checkpoint = [turn for turn in raw_turns if turn.occurred_at > checkpoint.updated_at]
            checkpoint_covered_turn_count = max(0, len(raw_turns) - len(turns_after_checkpoint))
            raw_turns = turns_after_checkpoint
        recent_conversation, audit = self.packing_policy.pack_turns(
            raw_turns
        )
        audit = {
            **audit,
            "total_turns_available": (
                audit["total_turns_available"]
                + checkpoint_covered_turn_count
                + omitted_before_task_context
            ),
            "omitted_turn_count": (
                audit["omitted_turn_count"]
                + checkpoint_covered_turn_count
                + omitted_before_task_context
            ),
            "omitted_covered_by_checkpoint": checkpoint_covered_turn_count,
            "omitted_before_task_context": omitted_before_task_context,
            "deduplicated_current_trace_tool_turn_count": deduplicated_current_trace_tool_turns,
        }
        profile = self.store.customers.get(message.sender_id)
        current_version = self.store.conversation_version(message.conversation_id)
        all_active_games = self.store.active_games()
        related_game_ids = {
            game.game_id
            for game in all_active_games
            if game.conversation_id == message.conversation_id
            or game.organizer_id == message.sender_id
            or any(participant.customer_id == message.sender_id for participant in game.participants)
        }
        related_game_ids.update(
            draft.game_id
            for draft in self.store.invite_drafts.values()
            if draft.customer_id == message.sender_id
        )
        active_games = [game for game in all_active_games if game.game_id in related_game_ids]
        active_game_contexts = [
            compact_game(game_for_model_context(item, self.store.customers))
            for item in active_games
        ]
        active_game_visible_summaries = [active_game_visible_summary(item) for item in active_games]
        sender_relationships = self.store.relationship_context_for_sender(message.sender_id, active_games)
        task_memories = self.store.task_memory_context(message.conversation_id, message.sender_id)
        pending_memory_candidates = self.store.pending_memory_candidates_for_context(message.conversation_id, message.sender_id)
        current_message = sanitize_current_message_for_context(message.to_dict())
        quoted_message_context = self._resolve_quoted_message_context(message, current_message)
        quoted_message = message.quoted_message
        quoted_message_present = quoted_message is not None
        quoted_message_has_provided_business_ref = bool(
            quoted_message
            and (
                quoted_message.business_ref_type
                or quoted_message.business_ref_id
            )
        )
        quoted_message_reference_status = "absent"
        if quoted_message_present:
            quoted_message_reference_status = "resolved" if quoted_message_context is not None else "unresolved"
            if quoted_message_context is None and quoted_message_has_provided_business_ref:
                quoted_message_reference_status = "provided_business_ref"
        message_reference_contract = {
            "primary_binding": "quoted_message" if quoted_message_present else "current_message",
            "quoted_message_present": quoted_message_present,
            "business_reference_status": quoted_message_reference_status,
            "business_reference_resolved": bool(
                quoted_message_context is not None or quoted_message_has_provided_business_ref
            ),
            "interpretation_instruction": (
                "Interpret the current reply against current_message.quoted_message before recent_conversation or active_games."
                if quoted_message_present
                else "Interpret the current reply from current_message, then use recent context only to resolve omissions."
            ),
            "state_write_instruction": (
                "The quote has no authoritative business reference. Do not infer a state-changing acceptance, rejection, "
                "arrival, cancellation, or participant update solely from this short reply plus a nearby active game. "
                "Resolve the business object with a read tool or ask the user before a write."
                if quoted_message_present
                and quoted_message_context is None
                and not quoted_message_has_provided_business_ref
                else "Any state write must still be supported by the current message and authoritative business state."
            ),
        }
        sender_active_game_memberships = []
        for game in active_games:
            for participant in game.participants:
                if participant.customer_id != message.sender_id:
                    continue
                sender_active_game_memberships.append(
                    {
                        "game_id": game.game_id,
                        "participant_status": participant.status,
                        "seat_count": participant.seat_count,
                        "participation_already_recorded": participant.status
                        in {"joined", "confirmed", "accepted", "arrived"},
                        "write_instruction": (
                            "Do not call record_candidate_reply with the same participation meaning unless the current "
                            "message explicitly changes status or seat_count."
                        ),
                    }
                )
        audit = {
            **audit,
            "conversation_checkpoint_present": checkpoint is not None,
            "checkpoint_excluded_by_task_context": checkpoint_excluded_by_task_context,
            "conversation_checkpoint_source_trace_id": checkpoint.source_trace_id if checkpoint else None,
            "checkpoint_covered_turn_count": checkpoint_covered_turn_count,
            "sender_relationship_count": len(sender_relationships),
            "task_memory_count": len(task_memories),
            "pending_memory_candidate_count": len(pending_memory_candidates),
            "active_game_visible_summary_count": len(active_game_visible_summaries),
            "quoted_message_present": quoted_message_present,
            "quoted_message_id": quoted_message.message_id if quoted_message else None,
            "quoted_message_reference_resolved": quoted_message_context is not None,
            "quoted_message_reference_status": quoted_message_reference_status,
            "quoted_message_business_ref_type": quoted_message_context.get("business_ref_type") if quoted_message_context else None,
            "conversation_version": current_version,
            "run_version": run_version,
            "run_current": run_version is None or int(run_version) == current_version,
            "task_context_id": task_context.task_context_id if task_context else None,
            "task_context_started_at": task_context.started_at.isoformat() if task_context else None,
        }
        payload = {
            "runtime": "mahjong_agent_runtime",
            "trace_id": trace_id,
            "customer_visibility_contract": {
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
            },
            "conversation_state": {
                "conversation_id": message.conversation_id,
                "task_context_id": task_context.task_context_id if task_context else None,
                "current_version": current_version,
                "run_id": run_id,
                "run_version": run_version,
                "run_current": run_version is None or int(run_version) == current_version,
                "version_contract": (
                    "每条新用户消息都会推进 conversation version；旧版本未发送的回复、邀约草稿和外发草稿会被标记为 superseded。"
                    "如果工具结果提示 stale_run，必须停止旧动作并基于当前消息重新判断。"
                ),
            },
            "task_context_window": {
                "task_context_id": task_context.task_context_id if task_context else None,
                "started_at": task_context.started_at.isoformat() if task_context else None,
                "reset_reason": task_context.reset_reason if task_context else None,
                "scope_contract": (
                    "recent_conversation, conversation_checkpoint, task_memories and pending task facts only belong "
                    "to this business episode. Stable sender_profile and approved customer relationships may cross episodes."
                ),
            },
            "current_message": current_message,
            "message_reference_contract": message_reference_contract,
            "quoted_message_context": quoted_message_context,
            "recent_conversation": recent_conversation,
            "conversation_checkpoint": checkpoint.to_dict() if checkpoint else None,
            "context_budget": audit,
            "sender_profile": profile.to_model_context() if profile else None,
            "sender_relationships": sender_relationships,
            "task_memories": task_memories,
            "pending_memory_candidates": pending_memory_candidates,
            "active_games": active_game_contexts,
            "active_game_visible_summaries": active_game_visible_summaries,
            "sender_active_game_memberships": sender_active_game_memberships,
            "active_parties": [
                {
                    "game_id": game_context["game_id"],
                    "seat_summary": dict(game_context.get("seat_summary") or {}),
                }
                for game_context in active_game_contexts
            ],
            "outbound_message_drafts": [
                outbound_message_draft_for_model_context(item, self.store.customers)
                for item in self.store.outbound_message_drafts.values()
                if item.conversation_id == message.conversation_id or item.recipient_id == message.sender_id
                if task_context is None
                or item.metadata.get("task_context_id") == task_context.task_context_id
                or (
                    not item.metadata.get("task_context_id")
                    and item.created_at >= task_context.started_at
                )
            ],
            "available_tools": self.tool_gateway.tool_specs_for_prompt(),
            "previous_tool_results": [tool_result_for_context(item) for item in previous_tool_results or []],
            "planning_contract": planning_contract(),
            "output_contract": output_contract(),
        }
        return BuiltContext(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
            ],
            payload=payload,
            audit=audit,
        )

    def _resolve_quoted_message_context(
        self,
        message: UserMessage,
        current_message: dict[str, Any],
    ) -> dict[str, Any] | None:
        quoted = message.quoted_message
        if quoted is None or not quoted.message_id:
            return None
        resolver = getattr(self.store, "resolve_message_reference", None)
        if not callable(resolver):
            return None
        reference = resolver(
            conversation_id=quoted.conversation_id or message.conversation_id,
            message_id=quoted.message_id,
        )
        if reference is None:
            return None
        reference_payload = message_reference_for_context(reference, self.store.customers)
        quoted_payload = dict(current_message.get("quoted_message") or quoted.to_dict())
        quoted_payload["business_ref_type"] = quoted_payload.get("business_ref_type") or reference.business_ref_type
        quoted_payload["business_ref_id"] = quoted_payload.get("business_ref_id") or reference.business_ref_id
        quoted_payload["conversation_id"] = quoted_payload.get("conversation_id") or reference.conversation_id
        quoted_payload["text"] = quoted_payload.get("text") or reference.text
        if reference.sender_id:
            quoted_payload["sender_name"] = customer_visible_name(
                self.store.customers,
                reference.sender_id,
                quoted_payload.get("sender_name") or reference.sender_name,
            )
        quoted_payload["metadata"] = {
            **dict(quoted_payload.get("metadata") or {}),
            "resolved_message_reference": {
                "business_ref_type": reference.business_ref_type,
                "business_ref_id": reference.business_ref_id,
                "channel": reference.channel,
                "recipient_id": reference.recipient_id,
                "recipient_name": customer_visible_name(
                    self.store.customers,
                    reference.recipient_id or "",
                    reference.recipient_name,
                ),
                "source": reference.metadata.get("source"),
            },
        }
        quoted_payload["metadata"] = sanitize_quoted_message_metadata_for_context(quoted_payload.get("metadata"))
        current_message["quoted_message"] = quoted_payload
        return reference_payload


def message_reference_for_context(
    reference: MessageReference,
    customers: dict[str, Any],
) -> dict[str, Any]:
    payload = reference.to_dict()
    payload["sender_name"] = customer_visible_name(customers, reference.sender_id or "", reference.sender_name)
    payload["recipient_name"] = customer_visible_name(customers, reference.recipient_id or "", reference.recipient_name)
    payload["metadata"] = sanitize_quoted_message_metadata_for_context(payload.get("metadata"))
    return payload


def context_text_preview(value: Any, limit: int = 160) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def sanitize_context_media_candidates(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    sanitized: list[dict[str, Any]] = []
    for item in value[:12]:
        if not isinstance(item, dict):
            continue
        safe_item = {
            "path": context_text_preview(item.get("path"), 160),
            "kind": context_text_preview(item.get("kind"), 40),
            "value_type": context_text_preview(item.get("value_type"), 40),
        }
        text_preview = context_text_preview(item.get("text_preview"), 120)
        if text_preview:
            safe_item["text_preview"] = text_preview
        sanitized.append({key: val for key, val in safe_item.items() if val not in {"", None}})
    return sanitized


def sanitize_context_observation_summary(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    sanitized: dict[str, int] = {}
    for key in ("quote_candidate_count", "media_candidate_count"):
        try:
            sanitized[key] = max(int(value.get(key) or 0), 0)
        except (TypeError, ValueError):
            continue
    return sanitized


def sanitize_message_metadata_for_context(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}
    sanitized: dict[str, Any] = {}
    for key, value in metadata.items():
        if key not in SAFE_CONTEXT_MESSAGE_METADATA_KEYS:
            continue
        if key == "modalities":
            if isinstance(value, list):
                sanitized[key] = [context_text_preview(item, 40) for item in value[:12] if str(item or "").strip()]
            continue
        if key == "media_candidates":
            sanitized[key] = sanitize_context_media_candidates(value)
            continue
        if key == "raw_observation_summary":
            sanitized[key] = sanitize_context_observation_summary(value)
            continue
        if isinstance(value, bool) or value is None:
            sanitized[key] = value
        elif isinstance(value, (int, float)):
            sanitized[key] = value
        elif isinstance(value, str):
            sanitized[key] = context_text_preview(value, 160)
    return sanitized


def sanitize_resolved_message_reference_for_context(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    sanitized: dict[str, Any] = {}
    for key in ("business_ref_type", "business_ref_id", "channel", "recipient_id", "recipient_name", "source"):
        raw_value = value.get(key)
        if raw_value is None:
            sanitized[key] = None
        elif isinstance(raw_value, (int, float, bool)):
            sanitized[key] = raw_value
        else:
            sanitized[key] = context_text_preview(raw_value, 160)
    return sanitized


def sanitize_quoted_message_metadata_for_context(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}
    sanitized: dict[str, Any] = {}
    for key, value in metadata.items():
        if key not in SAFE_CONTEXT_QUOTED_METADATA_KEYS:
            continue
        if key == "resolved_message_reference":
            sanitized_reference = sanitize_resolved_message_reference_for_context(value)
            if sanitized_reference:
                sanitized[key] = sanitized_reference
            continue
        if isinstance(value, bool) or value is None:
            sanitized[key] = value
        elif isinstance(value, (int, float)):
            sanitized[key] = value
        elif isinstance(value, str):
            sanitized[key] = context_text_preview(value, 160)
    return sanitized


def sanitize_current_message_for_context(current_message: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(current_message)
    sanitized["metadata"] = sanitize_message_metadata_for_context(sanitized.get("metadata"))
    quoted_message = sanitized.get("quoted_message")
    if isinstance(quoted_message, dict):
        quoted_payload = dict(quoted_message)
        quoted_payload["metadata"] = sanitize_quoted_message_metadata_for_context(quoted_payload.get("metadata"))
        sanitized["quoted_message"] = quoted_payload
    return sanitized


def turn_payload_for_context(turn: ConversationTurn) -> dict[str, Any]:
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
    payload["content"] = json.dumps([compact_tool_result_dict(item) for item in raw_results], ensure_ascii=False)
    payload["metadata"] = {**dict(payload.get("metadata") or {}), "compacted_for_context": True}
    return payload


def tool_result_for_context(result: ToolResult) -> dict[str, Any]:
    return compact_tool_result_dict(result.to_dict())


def compact_tool_result_dict(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {"name": "unknown", "called": False, "allowed": False, "result": {}, "error": "invalid tool result payload"}
    compact: dict[str, Any] = {
        "name": raw.get("name"),
        "called": raw.get("called"),
        "allowed": raw.get("allowed"),
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
    compact: dict[str, Any] = {}
    for key in [
        "requirement",
        "reference_requirement",
        "customer_reply_contract",
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
        "stale_run",
        "current_version",
        "run_version",
    ]:
        if key in payload:
            compact[key] = payload[key]
    if "matches" in payload:
        matches = [compact_match(item) for item in list(payload.get("matches") or [])[:5]]
        compact["matches"] = matches
        compact["match_count"] = len(payload.get("matches") or [])
    if "candidates" in payload:
        candidates = [compact_candidate(item) for item in list(payload.get("candidates") or [])[:12]]
        compact["candidates"] = candidates
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


def compact_game(game: Any) -> dict[str, Any]:
    if not isinstance(game, dict):
        return {}
    return {
        "game_id": game.get("game_id"),
        "conversation_id": game.get("conversation_id"),
        "organizer_id": game.get("organizer_id"),
        "organizer_name": game.get("organizer_name"),
        "status": game.get("status"),
        "requirement": compact_requirement(game.get("requirement")),
        "seat_summary": game.get("seat_summary"),
        "remaining_seats": game.get("remaining_seats"),
        "planned_start_at": game.get("planned_start_at"),
        "planned_end_at": game.get("planned_end_at"),
        "expires_at": game.get("expires_at"),
        "participants": [
            {
                "customer_id": item.get("customer_id"),
                "display_name": item.get("display_name"),
                "status": item.get("status"),
                "seat_count": item.get("seat_count"),
                "source": item.get("source"),
            }
            for item in list(game.get("participants") or [])[:8]
            if isinstance(item, dict)
        ],
        "parties": [compact_party(item) for item in list(game.get("parties") or [])[:8]],
    }


def compact_party(party: Any) -> dict[str, Any]:
    if not isinstance(party, dict):
        return {}
    return {
        "party_id": party.get("party_id"),
        "contact_id": party.get("contact_id"),
        "contact_name": party.get("contact_name"),
        "seat_count": party.get("seat_count"),
        "anonymous_seat_count": party.get("anonymous_seat_count"),
        "status": party.get("status"),
        "source": party.get("source"),
    }


def compact_requirement(requirement: Any) -> dict[str, Any]:
    """Project a requirement into the facts needed for the next decision.

    Party and seat structures already live on the compact game object. Keeping
    their second copy inside requirement caused prompt growth without adding a
    new fact, especially after participant updates.
    """

    if not isinstance(requirement, dict):
        return {}
    structural_duplicates = {
        "requesting_party",
        "seat_claims",
        "parties",
        "participants",
        "known_players",
    }
    return {
        key: compact_context_value(value)
        for key, value in requirement.items()
        if key not in structural_duplicates
    }


def compact_context_value(value: Any, *, max_list_items: int = 12, max_string_chars: int = 1000) -> Any:
    """Bound nested tool payloads while retaining deterministic JSON shapes."""

    if isinstance(value, dict):
        return {
            str(key): compact_context_value(item, max_list_items=max_list_items, max_string_chars=max_string_chars)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            compact_context_value(item, max_list_items=max_list_items, max_string_chars=max_string_chars)
            for item in value[:max_list_items]
        ]
    if isinstance(value, str) and len(value) > max_string_chars:
        return value[:max_string_chars] + "...[truncated]"
    return value


def compact_candidate(candidate: Any) -> dict[str, Any]:
    if not isinstance(candidate, dict):
        return {}
    customer = candidate.get("customer") if isinstance(candidate.get("customer"), dict) else {}
    return {
        "score": candidate.get("score"),
        "reasons": candidate.get("reasons"),
        "warnings": candidate.get("warnings"),
        "relationship": candidate.get("relationship"),
        "customer": {
            "customer_id": customer.get("customer_id"),
            "display_name": customer.get("display_name"),
            "gender": customer.get("gender"),
            "preferred_games": customer.get("preferred_games"),
            "preferred_stakes": customer.get("preferred_stakes"),
            "preferred_time_tags": customer.get("preferred_time_tags"),
            "smoke_preference": customer.get("smoke_preference"),
            "response_score": customer.get("response_score"),
            "fatigue_score": customer.get("fatigue_score"),
            "no_contact": customer.get("no_contact"),
            "notes": customer.get("notes"),
        },
    }


def compact_draft(draft: Any) -> dict[str, Any]:
    if not isinstance(draft, dict):
        return {}
    return {
        "draft_id": draft.get("draft_id"),
        "game_id": draft.get("game_id"),
        "customer_id": draft.get("customer_id") or draft.get("recipient_id"),
        "display_name": draft.get("display_name") or draft.get("recipient_name"),
        "message_text": draft.get("message_text"),
        "status": draft.get("status"),
        "purpose": draft.get("purpose"),
        "channel": draft.get("channel"),
    }


def output_contract() -> dict[str, Any]:
    return {
        "format": "json_object",
        "required_keys": [
            "goal",
            "objective_status",
            "reasoning_summary",
            "objective_state",
            "objective_plan",
            "plan_revision_reason",
            "reply_to_user",
            "tool_calls",
            "needs_human",
            "stop_reason",
        ],
        "objective_status_values": ["needs_tool", "waiting_user", "completed", "needs_human", "unknown"],
        "field_types": {
            "goal": "string",
            "objective_status": "string",
            "reasoning_summary": "string",
            "objective_state": "object; structured current task state, including known facts, missing facts, current phase, active IDs, blockers",
            "objective_plan": "array; ordered plan steps. each step should include step_id, title, status, tool, depends_on, decision_rule",
            "plan_revision_reason": "string; why this plan is created or changed after reading current message/tool results",
            "reply_to_user": "string",
            "tool_calls": "array",
            "needs_human": "boolean",
            "stop_reason": "object",
            "badcase": "null; deprecated side-channel, call record_badcase tool instead",
        },
        "objective_state_contract": {
            "current_phase": "recommended string: understand_intent | query_existing_games | collect_missing_info | create_game | search_customers | draft_invites | record_feedback | answer_user | wait_user | human_review",
            "known_facts": "recommended object; facts already safe to use for this objective",
            "missing_facts": "recommended array of strings; facts still needed before state writes or drafts",
            "active_game_id": "optional string|null",
            "blockers": "recommended array of strings",
            "reply_scope": (
                "recommended object for terminal replies: requested_information, allowed_response_facts, "
                "background_facts_to_withhold. Context facts may support reasoning without becoming customer-visible."
            ),
        },
        "objective_plan_contract": {
            "step_status_values": ["pending", "in_progress", "done", "blocked", "skipped"],
            "required_step_keys": ["step_id", "title", "status"],
            "recommended_step_keys": ["tool", "depends_on", "decision_rule"],
            "tool_step_rule": "Any step that needs system state should map to one available tool. Use objective_status=needs_tool while such steps are still in_progress.",
            "revision_rule": "After previous_tool_results are present, mark completed tool steps done, update known facts/blockers, and choose the next step instead of restarting from scratch.",
        },
        "stop_reason_contract": {
            "can_stop": "required boolean; false when objective_status=needs_tool, true for terminal statuses",
            "why": "required non-empty string explaining why the agent can stop now or why it must continue with tools",
            "pending_work": "required array of strings; non-empty when can_stop=false",
            "depends_on_tool_results": "required boolean; true if the decision depends on previous_tool_results or system state",
        },
        "tool_call_contract": {
            "name": "required non-empty string",
            "arguments": "required object, validated again by ToolGateway schema",
            "reason": "required non-empty string explaining why this tool is needed now",
            "idempotency_key": "optional string|null; backend still derives authoritative idempotency key",
        },
        "invariants": [
            "objective_status=needs_tool requires at least one tool_call",
            "objective_status=needs_tool requires empty reply_to_user",
            "objective_status=waiting_user|completed|needs_human|unknown must not include tool_calls",
            "objective_status=waiting_user|completed|needs_human|unknown requires non-empty reply_to_user",
            "objective_status=needs_human requires needs_human=true",
            "needs_human=true requires objective_status=needs_human",
            "objective_status=needs_tool requires stop_reason.can_stop=false and non-empty pending_work",
            "objective_status=waiting_user|completed|needs_human|unknown requires stop_reason.can_stop=true",
            "invalid contract means backend will not execute any tool",
            "badcase must be null; badcase/eval writes must use record_badcase tool_call",
            "terminal reply must answer only current_message or an explicitly unresolved confirmation; do not append adjacent active-game facts, shortage, time, or calls to action that the user did not ask for",
            "casual-chat replies may use business state for continuity but must not surface that state unless current_message explicitly refers to it",
        ],
    }


def planning_contract() -> dict[str, Any]:
    return {
        "purpose": "把每轮用户输入转成一个可执行目标，然后用工具结果持续修订计划。",
        "loop_rule": (
            "每一轮先更新 objective_state，再给出 objective_plan。"
            "如果计划中的下一步依赖系统事实，必须通过 tool_calls 调用工具；工具返回后基于 previous_tool_results 修订计划。"
        ),
        "state_progression": [
            "理解意图和上下文",
            "确认已知槽位、画像默认值和缺失槽位",
            "需要事实时查询当前局、房态或候选人",
            "需要写入时创建/更新局、记录候选人反馈或生成待审批草稿",
            "根据工具结果决定继续调用工具、追问用户、短句回复或转人工",
        ],
        "do_not": [
            "不要只用一句自然语言承诺代替应执行的工具步骤",
            "不要在工具结果回来后丢掉上一轮已确认的计划和槽位",
            "不要把计划、工具名或后台细节暴露给客户",
        ],
    }


def active_game_visible_summary(game: Any) -> dict[str, Any]:
    requirement = dict(getattr(game, "requirement", {}) or {})
    public_requirement_keys = (
        "user_visible_summary",
        "game_type",
        "stake",
        "base_stake",
        "cap_score",
        "stake_label",
        "smoke_preference",
        "start_time_kind",
        "start_time",
        "duration_kind",
        "duration_hours",
        "known_player_count",
        "needed_seats",
    )
    return {
        "game_id": game.game_id,
        "status": game.status.value,
        "user_visible_summary": str(requirement.get("user_visible_summary") or ""),
        "status_query_reply_contract": {
            "when_to_use": "用户问当前局况、现在几个人、还差几人、有没有进展时使用。",
            "preferred_reply_source": "user_visible_summary",
            "preferred_reply_text": str(requirement.get("user_visible_summary") or ""),
            "preservation_mode": "all_decision_anchors",
            "required_semantic_source": "preferred_reply_text",
            "invalid_rewrite": "只保留人数或缺口，丢失 preferred_reply_text 中的时间、公开昵称、局名或缺口短码。",
            "rule": "如果 user_visible_summary 非空，优先原样使用或轻微口语化；不要只根据 seat_summary 重新概括而丢掉时间、公开昵称、局名或缺口短码。",
        },
        "seat_summary": game.seat_summary(),
        "public_requirement": {
            key: requirement.get(key)
            for key in public_requirement_keys
            if requirement.get(key) is not None
        },
    }


def estimate_tokens(value: Any) -> int:
    """Compatibility wrapper around the shared conservative estimator."""

    return shared_estimate_tokens(value)
