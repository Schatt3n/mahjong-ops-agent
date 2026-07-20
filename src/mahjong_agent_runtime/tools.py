from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from .models import ToolCall, ToolResult
from .store import (
    InMemoryAgentStore,
    game_for_model_context,
    invite_draft_for_model_context,
    normalize_requirement,
    outbound_message_draft_for_model_context,
)


ToolHandler = Callable[[ToolCall, str, str, str, str], ToolResult]
_IDEMPOTENCY_LOCKS: dict[str, threading.RLock] = {}
_IDEMPOTENCY_LOCKS_GUARD = threading.RLock()
CANDIDATE_REPLY_STATUSES = ["accepted", "confirmed", "arrived", "declined", "negotiating", "no_reply"]
GAME_STATUSES = ["forming", "inviting", "ready", "cancelled", "finished"]

CANDIDATE_REPLY_NEXT_STEP_POLICIES: dict[str, dict[str, Any]] = {
    "declined": {
        "terminal_for_current_offer": True,
        "requires_explicit_user_request_to_search_alternatives": True,
        "instruction": (
            "This tool has recorded that the current user declined or rejected the current offer. "
            "Unless the same user message explicitly asks to continue looking for another game, stop this turn "
            "with a short acknowledgement. Do not call search_current_games, search_customers, create_game, or "
            "create_invite_drafts just because the user explained a preference while declining."
        ),
    },
    "negotiating": {
        "terminal_for_current_offer": False,
        "requires_coordination_before_confirmation": True,
        "instruction": (
            "This tool has recorded a negotiation on the current offer. Continue by coordinating the current game's "
            "open question or by replying that you will ask; do not switch to a new search unless the user explicitly "
            "asks for another game."
        ),
    },
    "no_reply": {
        "terminal_for_current_offer": True,
        "instruction": "This tool has recorded no reply. Avoid customer-visible claims that the user confirmed.",
    },
    "accepted": {
        "terminal_for_current_offer": True,
        "instruction": (
            "This tool has recorded acceptance of the current offer. Reply with a minimal acknowledgement like ok/好/okk. "
            "Do not restate time, stake, smoke, ready/full status, or arrival instructions unless the user explicitly asked for status."
        ),
    },
    "confirmed": {
        "terminal_for_current_offer": True,
        "instruction": (
            "This tool has recorded confirmation of the current offer. Reply with a minimal acknowledgement like ok/好/okk. "
            "Do not restate time, stake, smoke, ready/full status, or arrival instructions unless the user explicitly asked for status."
        ),
    },
    "arrived": {
        "terminal_for_current_offer": True,
        "instruction": "This tool has recorded arrival. Reply briefly; no further search is needed from this fact alone.",
    },
}


def cross_game_commitment_summary(transitions: list[Any]) -> dict[str, Any]:
    winner_game_ids = sorted(
        {
            transition.entity_id
            for transition in transitions
            if transition.entity_type == "game"
            and transition.to_status == "ready"
            and transition.reason == "seats_full"
        }
    )
    released = []
    for transition in transitions:
        if transition.entity_type != "game_participant" or transition.to_status != "superseded":
            continue
        game_id, _, customer_id = transition.entity_id.partition(":")
        committed_game_id = transition.reason.partition("participant_committed_to_game:")[2]
        released.append(
            {
                "customer_id": customer_id,
                "released_from_game_id": game_id,
                "committed_to_game_id": committed_game_id,
            }
        )
    return {
        "winner_game_ids": winner_game_ids,
        "released_participations": released,
        "affected_game_ids": sorted(
            {
                item["released_from_game_id"]
                for item in released
            }
            | set(winner_game_ids)
        ),
        "instruction": (
            "A participant may be provisionally present in many options. When the first overlapping game becomes "
            "ready, the backend atomically commits that participant there and releases every conflicting option. "
            "Use released_participations to coordinate follow-up messages; never re-add the participant to a losing "
            "overlapping game unless the winning commitment is cancelled first."
        ),
    }


def current_game_search_reply_contract(requirement: dict[str, Any], matches: list[dict[str, Any]]) -> dict[str, Any]:
    match_summaries = [
        str(item.get("game", {}).get("requirement", {}).get("user_visible_summary") or "").strip()
        for item in matches
    ]
    match_summaries = [item for item in match_summaries if item]
    has_matches = bool(matches)
    return {
        "source_tool": "search_current_games",
        "matched_query_requirement": requirement,
        "matched_result_summaries": match_summaries,
        "search_result_semantics": {
            "status": "actionable_matches" if has_matches else "no_actionable_match",
            "backend_retrieval_policy_applied": True,
            "actionable_match_count": len(matches),
            "instruction": (
                "The non-empty matches list is the backend-selected actionable candidate set. It may contain a nearby "
                "or otherwise decision-worthy alternative under the domain retrieval policy, not only raw-field exact "
                "matches. Do not recompute eligibility from raw fields, reject the returned candidates, claim that no "
                "game exists, or repeat the same semantic search merely because a returned time or other field differs. "
                "Offer the matched_result_summaries and disclose only differences the customer must decide."
                if has_matches
                else
                "The backend found no actionable current game under the executed requirement. Do not repeat the same "
                "semantic search unless the user changes a constraint, system state becomes stale, or the tool reports an error."
            ),
        },
        "reply_shape": "Use one matched_result_summary plus a short confirmation question.",
        "customer_visible_rule": (
            "When a matched current game satisfies the user's request, the customer-visible reply should prioritize "
            "the game's user_visible_summary and a short requester confirmation such as 可以不/可以吗; use 打吗/来吗 "
            "mainly for candidate invitations. Do not expand matched query "
            "slots or profile-default slots such as game_type, stake, smoke_preference, requester seat count, or backend "
            "search reasons into the reply unless the field is already in matched_result_summaries or the result differs "
            "from what the user requested and must be disclosed for decision-making."
        ),
        "good_reply_examples": ["七点三缺一，可以不", "七点三缺一，可以吗"],
        "bad_reply_examples": [
            "七点三缺一，0.5无烟杭麻，打吗",
            "已按你的画像找到七点三缺一",
            "六点半没有，七点有个三缺一，0.5无烟，可以不",
        ],
    }


@dataclass(slots=True)
class ToolDefinition:
    name: str
    description: str
    risk_level: str
    execution_mode: str
    schema: dict[str, Any]
    handler: ToolHandler | None = None
    parallel_safe: bool = False

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "risk_level": self.risk_level,
            "execution_mode": self.execution_mode,
            "parallel_safe": self.parallel_safe,
            "schema": self.schema,
        }


@dataclass(slots=True)
class ToolGateway:
    store: InMemoryAgentStore
    tools: dict[str, ToolDefinition] = field(default_factory=dict)
    trace_recorder: Any | None = None
    allowed_execution_modes: set[str] = field(default_factory=lambda: {"read_only", "state_write", "draft_write", "audit_write"})
    allowed_risk_levels: set[str] = field(default_factory=lambda: {"low", "medium"})

    def __post_init__(self) -> None:
        if not self.tools:
            self.tools.update(default_tool_definitions(self.store))

    def tool_specs_for_prompt(self) -> list[dict[str, Any]]:
        return [definition.to_prompt_dict() for definition in self.tools.values()]

    def execute(
        self,
        call: ToolCall,
        *,
        trace_id: str,
        conversation_id: str,
        sender_id: str,
        sender_name: str,
        step_index: int,
        source_message_id: str | None = None,
        message_reference_contract: dict[str, Any] | None = None,
    ) -> ToolResult:
        definition = self.tools.get(call.name)
        idempotency_key = (
            backend_tool_idempotency_key(
                call,
                conversation_id=conversation_id,
                sender_id=sender_id,
                source_message_id=source_message_id,
            )
            or call.idempotency_key
            or f"{trace_id}:tool:{step_index}:{call.name}"
        )
        self._record(
            trace_id,
            "tool_gateway_received",
            {"tool_name": call.name, "call": call.to_dict(), "step_index": step_index, "idempotency_key": idempotency_key},
        )
        with idempotency_lock_for_key(idempotency_key):
            existing = self.store.idempotent_result(idempotency_key)
            self._record(
                trace_id,
                "tool_idempotency_checked",
                {"tool_name": call.name, "step_index": step_index, "idempotency_key": idempotency_key, "hit": existing is not None},
            )
            if existing is not None:
                result = ToolResult(
                    name=existing.name,
                    called=existing.called,
                    allowed=existing.allowed,
                    result=dict(existing.result),
                    error=existing.error,
                    idempotency_key=idempotency_key,
                    deduplicated=True,
                    state_transitions=list(existing.state_transitions),
                )
                return self._complete(trace_id, step_index, result, outcome="deduplicated")
            if definition is None:
                result = ToolResult(name=call.name, called=False, allowed=False, error=f"unknown tool: {call.name}", idempotency_key=idempotency_key)
                self._record(trace_id, "tool_definition_checked", {"tool_name": call.name, "allowed": False}, level="WARN")
                return self._complete(trace_id, step_index, result, outcome="blocked", remember_key=idempotency_key)
            self._record(trace_id, "tool_definition_checked", {"tool_name": call.name, "allowed": True})
            schema_error = validate_schema(call.arguments, definition.schema)
            if schema_error:
                result = ToolResult(name=call.name, called=False, allowed=False, error=schema_error, idempotency_key=idempotency_key)
                self._record(trace_id, "tool_schema_checked", {"tool_name": call.name, "allowed": False, "error": schema_error}, level="WARN")
                return self._complete(trace_id, step_index, result, outcome="blocked", remember_key=idempotency_key)
            self._record(trace_id, "tool_schema_checked", {"tool_name": call.name, "allowed": True})
            permission_error = self._permission_error(definition)
            if permission_error:
                result = ToolResult(name=call.name, called=False, allowed=False, error=permission_error, idempotency_key=idempotency_key)
                self._record(trace_id, "tool_permission_checked", {"tool_name": call.name, "allowed": False, "error": permission_error}, level="WARN")
                return self._complete(trace_id, step_index, result, outcome="blocked", remember_key=idempotency_key)
            self._record(trace_id, "tool_permission_checked", {"tool_name": call.name, "allowed": True})
            authorization_error = self._authorization_error(
                call,
                conversation_id=conversation_id,
                sender_id=sender_id,
                message_reference_contract=message_reference_contract,
            )
            if authorization_error:
                result = ToolResult(
                    name=call.name,
                    called=False,
                    allowed=False,
                    error=authorization_error,
                    idempotency_key=idempotency_key,
                )
                self._record(
                    trace_id,
                    "tool_authorization_checked",
                    {"tool_name": call.name, "allowed": False, "error": authorization_error},
                    level="WARN",
                )
                return self._complete(trace_id, step_index, result, outcome="blocked", remember_key=idempotency_key)
            self._record(trace_id, "tool_authorization_checked", {"tool_name": call.name, "allowed": True})
            claimed_result = ToolResult(
                name=call.name,
                called=False,
                allowed=True,
                result={"idempotency_status": "claimed", "claimed_by_trace_id": trace_id},
                error="tool execution is already in progress for this idempotency key",
                idempotency_key=idempotency_key,
            )
            claimed, claimed_existing = self.store.claim_idempotent_result(idempotency_key, claimed_result)
            self._record(
                trace_id,
                "tool_idempotency_claimed",
                {
                    "tool_name": call.name,
                    "step_index": step_index,
                    "idempotency_key": idempotency_key,
                    "claimed": claimed,
                    "existing": claimed_existing.to_dict() if claimed_existing else None,
                },
                level="WARN" if not claimed else "INFO",
            )
            if not claimed:
                if claimed_existing is None:
                    claimed_existing = ToolResult(
                        name=call.name,
                        called=False,
                        allowed=False,
                        error="idempotency key already claimed but result is unavailable",
                        idempotency_key=idempotency_key,
                    )
                result = ToolResult(
                    name=claimed_existing.name,
                    called=claimed_existing.called,
                    allowed=claimed_existing.allowed,
                    result=dict(claimed_existing.result),
                    error=claimed_existing.error,
                    idempotency_key=idempotency_key,
                    deduplicated=True,
                    state_transitions=list(claimed_existing.state_transitions),
                )
                return self._complete(trace_id, step_index, result, outcome="deduplicated")
            try:
                if definition.handler is None:
                    raise RuntimeError(f"tool has no handler: {call.name}")
                result = definition.handler(call, trace_id, conversation_id, sender_id, sender_name)
            except Exception as exc:
                self._record(
                    trace_id,
                    "tool_exception",
                    {
                        "tool_name": call.name,
                        "step_index": step_index,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "idempotency_key": idempotency_key,
                    },
                    level="ERROR",
                )
                result = ToolResult(name=call.name, called=False, allowed=False, error=f"{type(exc).__name__}: {exc}")
            result.idempotency_key = idempotency_key
            return self._complete(
                trace_id,
                step_index,
                result,
                outcome="executed" if result.called and result.allowed else "failed",
                remember_key=idempotency_key,
            )

    def _complete(
        self,
        trace_id: str,
        step_index: int,
        result: ToolResult,
        *,
        outcome: str,
        remember_key: str | None = None,
    ) -> ToolResult:
        self._record(
            trace_id,
            "tool_gateway_completed",
            {
                "tool_name": result.name,
                "step_index": step_index,
                "outcome": outcome,
                "called": result.called,
                "allowed": result.allowed,
                "error": result.error,
                "idempotency_key": result.idempotency_key,
                "deduplicated": result.deduplicated,
            },
            level="WARN" if result.error else "INFO",
        )
        if remember_key:
            self.store.remember_result(remember_key, result)
        return result

    def _permission_error(self, definition: ToolDefinition) -> str | None:
        if definition.execution_mode not in self.allowed_execution_modes:
            return f"tool execution_mode not allowed: {definition.execution_mode}"
        if definition.risk_level not in self.allowed_risk_levels:
            return f"tool risk_level not allowed: {definition.risk_level}"
        return None

    def _authorization_error(
        self,
        call: ToolCall,
        *,
        conversation_id: str,
        sender_id: str,
        message_reference_contract: dict[str, Any] | None = None,
    ) -> str | None:
        """Bind write tools to the authenticated message subject and conversation.

        The model proposes business arguments, but it is not an identity provider.
        These checks prevent a malformed or adversarial proposal from writing as a
        different customer or mutating a game owned by another conversation.
        """

        reference_contract = message_reference_contract or {}
        if (
            call.name in {"join_game", "record_candidate_reply"}
            and reference_contract.get("quoted_message_present") is True
            and reference_contract.get("business_reference_resolved") is not True
        ):
            return (
                "authoritative quoted-message business reference required: "
                f"{call.name} cannot infer a participation state write from an unresolved quote; "
                "resolve the referenced invitation/game with a read tool or ask the user"
            )

        subject_argument = {
            "create_game": "organizer_id",
            "join_game": "customer_id",
            "record_candidate_reply": "customer_id",
        }.get(call.name)
        if subject_argument:
            proposed_subject = str(call.arguments.get(subject_argument) or "")
            if proposed_subject != sender_id:
                return (
                    f"tool subject mismatch: {subject_argument} must equal authenticated sender_id; "
                    f"expected={sender_id!r}, got={proposed_subject!r}"
                )

        if call.name == "record_user_memory":
            memory_items = list(call.arguments.get("task_memories") or []) + list(
                call.arguments.get("pending_long_term_memories") or []
            )
            for item in memory_items:
                if not isinstance(item, dict):
                    continue
                customer_id = str(item.get("customer_id") or sender_id)
                if customer_id != sender_id:
                    return (
                        "tool subject mismatch: record_user_memory may only write memory for "
                        f"authenticated sender_id={sender_id!r}"
                    )

        game_argument = {
            "create_invite_drafts": "game_id",
            "join_game": "game_id",
            "record_candidate_reply": "game_id",
            "update_game_requirement": "game_id",
            "update_game_status": "game_id",
            "reserve_room": "game_id",
        }.get(call.name)
        if game_argument:
            game_id = str(call.arguments.get(game_argument) or "")
            if game_id:
                try:
                    game = self.store.require_game(game_id)
                except ValueError as exc:
                    return str(exc)
                invited_candidate = call.name in {"join_game", "record_candidate_reply"} and any(
                    draft.game_id == game_id and draft.customer_id == sender_id
                    for draft in self.store.invite_drafts.values()
                )
                if game.conversation_id != conversation_id and not invited_candidate:
                    return (
                        "tool resource mismatch: game belongs to another conversation; "
                        f"expected={conversation_id!r}, got={game.conversation_id!r}"
                    )
        return None

    def _record(self, trace_id: str, step: str, content: dict[str, Any], *, level: str = "INFO") -> None:
        if self.trace_recorder is not None:
            self.trace_recorder.record(trace_id, step, content, level=level)


def default_tool_definitions(store: InMemoryAgentStore) -> dict[str, ToolDefinition]:
    requirement_schema = {"type": "object", "additionalProperties": True}
    non_empty_string = {"type": "string", "minLength": 1}
    known_player_schema = {
        "type": "object",
        "required": ["customer_id", "display_name"],
        "additionalProperties": True,
        "properties": {
            "customer_id": non_empty_string,
            "display_name": non_empty_string,
            "status": {"type": "string", "enum": ["joined", "confirmed"]},
            "source": {"type": "string"},
            "seat_count": {"type": "integer", "minimum": 1, "maximum": 4},
            "known_member_ids": {"type": "array", "items": {"type": "string"}},
            "anonymous_seat_count": {"type": "integer", "minimum": 0, "maximum": 4},
        },
    }
    requesting_party_schema = {
        "type": "object",
        "required": ["contact_id", "contact_name", "seat_count"],
        "additionalProperties": True,
        "properties": {
            "contact_id": non_empty_string,
            "contact_name": non_empty_string,
            "seat_count": {"type": "integer", "minimum": 1, "maximum": 4},
            "known_member_ids": {"type": "array", "items": {"type": "string"}},
            "anonymous_seat_count": {"type": "integer", "minimum": 0, "maximum": 4},
            "source": {"type": "string"},
        },
    }
    invitation_schema = {
        "type": "object",
        "required": ["customer_id", "display_name", "message_text"],
        "additionalProperties": True,
        "properties": {
            "customer_id": non_empty_string,
            "display_name": non_empty_string,
            "message_text": non_empty_string,
            "metadata": {"type": "object", "additionalProperties": True},
        },
    }
    outbound_message_draft_schema = {
        "type": "object",
        "required": ["recipient_id", "recipient_name", "channel", "message_text", "purpose"],
        "additionalProperties": False,
        "properties": {
            "recipient_id": non_empty_string,
            "recipient_name": non_empty_string,
            "channel": non_empty_string,
            "message_text": non_empty_string,
            "purpose": non_empty_string,
            "metadata": {"type": "object", "additionalProperties": True},
        },
    }
    checkpoint_schema = {
        "type": "object",
        "required": ["summary"],
        "additionalProperties": False,
        "properties": {
            "summary": non_empty_string,
            "facts": {"type": "object", "additionalProperties": True},
            "open_questions": {"type": "array", "items": non_empty_string},
        },
    }
    badcase_schema = {
        "type": "object",
        "required": ["reason", "input", "actual", "expected"],
        "additionalProperties": True,
        "properties": {
            "reason": non_empty_string,
            "input": {"type": "object", "additionalProperties": True},
            "actual": {"type": "object", "additionalProperties": True},
            "expected": {"type": "object", "additionalProperties": True},
            "tags": {"type": "array", "items": non_empty_string},
            "metadata": {"type": "object", "additionalProperties": True},
        },
    }
    memory_item_schema = {
        "type": "object",
        "required": ["customer_id", "memory_type", "field", "value", "evidence", "confidence"],
        "additionalProperties": True,
        "properties": {
            "customer_id": non_empty_string,
            "memory_type": non_empty_string,
            "field": {
                **non_empty_string,
                "description": (
                    "稳定的结构化字段名。时长上限使用 max_duration_hours，明确约定时长使用 duration_hours；"
                    "避免同一语义在不同轮次使用不同字段名。"
                ),
            },
            "value": {},
            "target_customer_id": {"type": "string"},
            "target_customer_name": {"type": "string"},
            "operation": {"type": "string"},
            "evidence": non_empty_string,
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "risk_level": {"type": "string", "enum": ["low", "medium", "high"]},
            "scope": {"type": "string", "enum": ["current_task", "session", "today", "long_term"]},
            "metadata": {"type": "object", "additionalProperties": True},
        },
    }
    memory_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "task_memories": {"type": "array", "items": memory_item_schema},
            "pending_long_term_memories": {"type": "array", "items": memory_item_schema},
        },
    }

    def search_current_games(call: ToolCall, trace_id: str, conversation_id: str, sender_id: str, sender_name: str) -> ToolResult:
        requirement = normalize_requirement(dict(call.arguments.get("requirement") or {}))
        matches = store.search_current_games(
            requirement,
            limit=int(call.arguments.get("limit") or 8),
            sender_id=sender_id,
            conversation_id=conversation_id,
        )
        return ToolResult(
            name=call.name,
            called=True,
            allowed=True,
            result={
                "requirement": requirement,
                "matches": matches,
                "customer_reply_contract": current_game_search_reply_contract(requirement, matches),
            },
        )

    def check_room_availability(call: ToolCall, trace_id: str, conversation_id: str, sender_id: str, sender_name: str) -> ToolResult:
        availability = store.search_room_availability(
            start_at=call.arguments.get("start_at"),
            end_at=call.arguments.get("end_at"),
        )
        return ToolResult(
            name=call.name,
            called=True,
            allowed=True,
            result={
                **availability,
                "instruction": (
                    "Only state that a room is available when configured=true and available_count>0. "
                    "This read does not reserve or promise a room. When configured=false, availability is unknown: "
                    "a forming game may still be created with room confirmation pending, but room availability must not be promised."
                ),
                "next_step_policy": {
                    "query_completed": True,
                    "repeat_same_query": False,
                    "may_create_forming_game_with_room_pending": not availability["configured"],
                    "may_state_room_available": bool(
                        availability["configured"] and availability["available_count"] > 0
                    ),
                    "must_report_unavailable": bool(
                        availability["configured"] and availability["available_count"] <= 0
                    ),
                    "instruction": (
                        "Do not repeat check_room_availability with the same interval. Mark the room-check plan step done. "
                        "If inventory is unconfigured, continue the requested business flow with room confirmation "
                        "pending and do not claim that a room exists. If configured and available_count is zero, do not "
                        "create or promise a fixed-time game for this interval; offer another time."
                    ),
                },
            },
        )

    def reserve_room(call: ToolCall, trace_id: str, conversation_id: str, sender_id: str, sender_name: str) -> ToolResult:
        reservation, transition = store.reserve_room(
            conversation_id=conversation_id,
            game_id=str(call.arguments.get("game_id") or "") or None,
            start_at=call.arguments.get("start_at"),
            end_at=call.arguments.get("end_at"),
            room_id=str(call.arguments.get("room_id") or "") or None,
            trace_id=trace_id,
        )
        return ToolResult(
            name=call.name,
            called=True,
            allowed=True,
            result={"reservation": reservation.to_dict()},
            state_transitions=[transition],
        )
    def search_customers(call: ToolCall, trace_id: str, conversation_id: str, sender_id: str, sender_name: str) -> ToolResult:
        requirement = normalize_requirement(dict(call.arguments.get("requirement") or {}))
        exclude_customer_ids = [str(item) for item in call.arguments.get("exclude_customer_ids") or []]
        candidates = store.search_customers(
            requirement,
            exclude_customer_ids=exclude_customer_ids,
            limit=int(call.arguments.get("limit") or 8),
            sender_id=sender_id,
            conversation_id=conversation_id,
        )
        return ToolResult(
            name=call.name,
            called=True,
            allowed=True,
            result={"requirement": requirement, "exclude_customer_ids": exclude_customer_ids, "candidates": candidates},
        )

    def create_game(call: ToolCall, trace_id: str, conversation_id: str, sender_id: str, sender_name: str) -> ToolResult:
        known_players = known_players_with_requesting_party(
            known_players=list(call.arguments.get("known_players") or []),
            requesting_party=call.arguments.get("requesting_party"),
        )
        game, transition = store.create_game(
            conversation_id=conversation_id,
            organizer_id=str(call.arguments["organizer_id"]),
            organizer_name=str(call.arguments["organizer_name"]),
            requirement=normalize_requirement(dict(call.arguments.get("requirement") or {})),
            known_players=known_players,
            trace_id=trace_id,
        )
        scheduled_task = store.scheduled_task_for_game(game.game_id)
        return ToolResult(
            name=call.name,
            called=True,
            allowed=True,
            result={
                "game": game_for_model_context(game, store.customers),
                "recruitment_policy": {
                    "status": game.recruitment_status.value,
                    "opens_at": game.recruitment_opens_at.isoformat() if game.recruitment_opens_at else None,
                    "scheduled_task": scheduled_task.to_dict() if scheduled_task else None,
                    "instruction": (
                        "When status=scheduled, keep the game visible but do not search private candidates or create "
                        "invite drafts. A durable system task will re-enter the main Agent when the window opens."
                    ),
                },
            },
            state_transitions=[transition],
        )

    def join_game(call: ToolCall, trace_id: str, conversation_id: str, sender_id: str, sender_name: str) -> ToolResult:
        game, transitions = store.join_game(
            game_id=str(call.arguments["game_id"]),
            customer_id=str(call.arguments["customer_id"]),
            display_name=str(call.arguments["display_name"]),
            seat_count=int(call.arguments.get("seat_count") or 1),
            trace_id=trace_id,
        )
        return ToolResult(
            name=call.name,
            called=True,
            allowed=True,
            result={
                "game": game_for_model_context(game, store.customers),
                "recorded_status": "confirmed",
                "next_step_policy": CANDIDATE_REPLY_NEXT_STEP_POLICIES["confirmed"],
                "cross_game_commitment": cross_game_commitment_summary(transitions),
            },
            state_transitions=transitions,
        )

    def create_invite_drafts(call: ToolCall, trace_id: str, conversation_id: str, sender_id: str, sender_name: str) -> ToolResult:
        drafts, transitions = store.create_invite_drafts(
            game_id=str(call.arguments.get("game_id") or ""),
            invitations=list(call.arguments.get("invitations") or []),
            trace_id=trace_id,
        )
        return ToolResult(
            name=call.name,
            called=True,
            allowed=True,
            result={"drafts": [invite_draft_for_model_context(item, store.customers) for item in drafts]},
            state_transitions=transitions,
        )

    def update_game_requirement(call: ToolCall, trace_id: str, conversation_id: str, sender_id: str, sender_name: str) -> ToolResult:
        game, transition = store.update_game_requirement(
            game_id=str(call.arguments.get("game_id") or ""),
            requirement_patch=normalize_requirement(dict(call.arguments.get("requirement_patch") or {})),
            reason=str(call.arguments.get("reason") or ""),
            trace_id=trace_id,
        )
        return ToolResult(
            name=call.name,
            called=True,
            allowed=True,
            result={"game": game_for_model_context(game, store.customers)},
            state_transitions=[transition],
        )

    def create_outbound_message_drafts(call: ToolCall, trace_id: str, conversation_id: str, sender_id: str, sender_name: str) -> ToolResult:
        drafts, transitions = store.create_outbound_message_drafts(
            conversation_id=conversation_id,
            drafts=list(call.arguments.get("drafts") or []),
            trace_id=trace_id,
        )
        return ToolResult(
            name=call.name,
            called=True,
            allowed=True,
            result={"drafts": [outbound_message_draft_for_model_context(item, store.customers) for item in drafts]},
            state_transitions=transitions,
        )

    def record_candidate_reply(call: ToolCall, trace_id: str, conversation_id: str, sender_id: str, sender_name: str) -> ToolResult:
        status = str(call.arguments["status"])
        game, transitions = store.record_candidate_reply(
            game_id=str(call.arguments["game_id"]),
            customer_id=str(call.arguments["customer_id"]),
            display_name=str(call.arguments["display_name"]),
            status=status,
            seat_count=int(call.arguments.get("seat_count") or 1),
            trace_id=trace_id,
        )
        return ToolResult(
            name=call.name,
            called=True,
            allowed=True,
            result={
                "game": game_for_model_context(game, store.customers),
                "recorded_status": status,
                "next_step_policy": CANDIDATE_REPLY_NEXT_STEP_POLICIES.get(status, {}),
                "cross_game_commitment": cross_game_commitment_summary(transitions),
            },
            state_transitions=transitions,
        )

    def update_game_status(call: ToolCall, trace_id: str, conversation_id: str, sender_id: str, sender_name: str) -> ToolResult:
        game, transition = store.update_game_status(
            game_id=str(call.arguments["game_id"]),
            status=str(call.arguments["status"]),
            reason=str(call.arguments["reason"]),
            trace_id=trace_id,
        )
        return ToolResult(
            name=call.name,
            called=True,
            allowed=True,
            result={"game": game_for_model_context(game, store.customers)},
            state_transitions=[transition],
        )

    def record_badcase(call: ToolCall, trace_id: str, conversation_id: str, sender_id: str, sender_name: str) -> ToolResult:
        record = store.record_badcase(dict(call.arguments), trace_id=trace_id, conversation_id=conversation_id)
        return ToolResult(name=call.name, called=True, allowed=True, result={"recorded": True, "badcase": record})

    def update_context_checkpoint(call: ToolCall, trace_id: str, conversation_id: str, sender_id: str, sender_name: str) -> ToolResult:
        checkpoint, transition = store.upsert_conversation_checkpoint(
            conversation_id=conversation_id,
            summary=str(call.arguments["summary"]),
            facts=dict(call.arguments.get("facts") or {}),
            open_questions=[str(item) for item in call.arguments.get("open_questions") or []],
            trace_id=trace_id,
        )
        return ToolResult(
            name=call.name,
            called=True,
            allowed=True,
            result={"checkpoint": checkpoint.to_dict()},
            state_transitions=[transition],
        )

    def record_user_memory(call: ToolCall, trace_id: str, conversation_id: str, sender_id: str, sender_name: str) -> ToolResult:
        task_memories = []
        pending_candidates = []
        transitions = []
        for raw in call.arguments.get("task_memories") or []:
            if not isinstance(raw, dict):
                continue
            metadata = dict(raw.get("metadata") or {}) if isinstance(raw.get("metadata"), dict) else {}
            if raw.get("target_customer_name") and "target_customer_name" not in metadata:
                metadata["target_customer_name"] = str(raw.get("target_customer_name") or "")
            memory, transition = store.record_task_memory(
                conversation_id=conversation_id,
                customer_id=str(raw.get("customer_id") or sender_id),
                memory_type=str(raw.get("memory_type") or ""),
                field=str(raw.get("field") or ""),
                value=raw.get("value"),
                target_customer_id=str(raw.get("target_customer_id") or "") or None,
                evidence=str(raw.get("evidence") or ""),
                confidence=float(raw.get("confidence") or 0.0),
                risk_level=str(raw.get("risk_level") or "medium"),
                scope=str(raw.get("scope") or "current_task"),
                metadata=metadata,
                trace_id=trace_id,
            )
            task_memories.append(memory.to_dict())
            transitions.append(transition)
        for raw in call.arguments.get("pending_long_term_memories") or []:
            if not isinstance(raw, dict):
                continue
            metadata = dict(raw.get("metadata") or {}) if isinstance(raw.get("metadata"), dict) else {}
            if raw.get("target_customer_name") and "target_customer_name" not in metadata:
                metadata["target_customer_name"] = str(raw.get("target_customer_name") or "")
            candidate, transition = store.record_pending_memory_candidate(
                conversation_id=conversation_id,
                customer_id=str(raw.get("customer_id") or sender_id),
                memory_type=str(raw.get("memory_type") or ""),
                field=str(raw.get("field") or ""),
                value=raw.get("value"),
                operation=str(raw.get("operation") or "set"),
                target_customer_id=str(raw.get("target_customer_id") or "") or None,
                evidence=str(raw.get("evidence") or ""),
                confidence=float(raw.get("confidence") or 0.0),
                risk_level=str(raw.get("risk_level") or "medium"),
                scope=str(raw.get("scope") or "long_term"),
                metadata=metadata,
                trace_id=trace_id,
            )
            pending_candidates.append(candidate.to_dict())
            transitions.append(transition)
        return ToolResult(
            name=call.name,
            called=True,
            allowed=True,
            result={
                "task_memories": task_memories,
                "pending_long_term_memories": pending_candidates,
                "next_step_policy": {
                    "memory_write_does_not_authorize_downstream_actions": True,
                    "requires_explicit_user_request_to_expand_goal": True,
                    "allows_resume_when_previous_plan_was_blocked_by_this_fact": True,
                    "default_next_action": "reply_with_short_confirmation",
                    "instruction": (
                        "The memory is now active, but this write does not authorize new downstream work. "
                        "Only continue search, matching, or draft creation when the current user message explicitly "
                        "requests it, or when the prior plan was already blocked waiting for exactly this fact. "
                        "Otherwise stop with a short confirmation. Pending long-term candidates are not yet profiles."
                    )
                },
            },
            state_transitions=transitions,
        )

    return {
        "check_room_availability": ToolDefinition(
            "check_room_availability",
            "只读查询指定起止时间内的真实房间库存。只要用户询问明确时段的局，推荐或创建前先查询；未配置库存时不能声称有房。",
            "low",
            "read_only",
            {
                "type": "object",
                "required": ["start_at", "end_at"],
                "additionalProperties": False,
                "properties": {
                    "start_at": non_empty_string,
                    "end_at": non_empty_string,
                },
            },
            check_room_availability,
            parallel_safe=True,
        ),
        "reserve_room": ToolDefinition(
            "reserve_room",
            "在已确认时间区间内原子占用一个可用房间。必须先查询库存；成功才表示已暂占，不能凭模型文字承诺。",
            "medium",
            "state_write",
            {
                "type": "object",
                "required": ["start_at", "end_at"],
                "additionalProperties": False,
                "properties": {
                    "game_id": {"type": "string"},
                    "room_id": {"type": "string"},
                    "start_at": non_empty_string,
                    "end_at": non_empty_string,
                },
            },
            reserve_room,
        ),
        "search_current_games": ToolDefinition(
            "search_current_games",
            "只读查询当前局池。模型提供结构化 requirement；工具只按字段匹配，不理解自然语言。",
            "low",
            "read_only",
            {"type": "object", "required": ["requirement"], "properties": {"requirement": requirement_schema, "limit": {"type": "integer", "minimum": 1, "maximum": 20}}},
            search_current_games,
            parallel_safe=True,
        ),
        "search_customers": ToolDefinition(
            "search_customers",
            "只读查询候选客户。模型负责给出筛选条件；工具只做确定性排序，并会参考关系画像避开不愿同桌的人。若已知当前局内人员，应在 requirement 里提供 existing_player_ids 或 organizer_id。",
            "low",
            "read_only",
            {"type": "object", "required": ["requirement"], "properties": {"requirement": requirement_schema, "exclude_customer_ids": {"type": "array", "items": {"type": "string"}}, "limit": {"type": "integer", "minimum": 1, "maximum": 20}}},
            search_customers,
            parallel_safe=True,
        ),
        "create_game": ToolDefinition(
            "create_game",
            "创建待组局记录。只落库，不发消息、不确认房间。固定时间且距离开局超过招募提前量时，后端会持久化定时任务；局立即进入列表，但暂不私聊候选人。模型必须显式提供 organizer_id 和 organizer_name，后端不从当前消息脑补组织者。",
            "medium",
            "state_write",
            {
                "type": "object",
                "required": ["requirement", "organizer_id", "organizer_name"],
                "additionalProperties": False,
                "properties": {
                    "requirement": requirement_schema,
                    "organizer_id": non_empty_string,
                    "organizer_name": non_empty_string,
                    "known_players": {"type": "array", "items": known_player_schema},
                    "requesting_party": requesting_party_schema,
                },
            },
            create_game,
        ),
        "join_game": ToolDefinition(
            "join_game",
            "把当前已鉴权客户加入指定局。仅用于客户明确接受/确认参加；后端原子校验容量、跨局冲突和状态机，并写入独立参与者表。拒绝、协商、未回复仍使用 record_candidate_reply。",
            "medium",
            "state_write",
            {
                "type": "object",
                "required": ["game_id", "customer_id", "display_name"],
                "additionalProperties": False,
                "properties": {
                    "game_id": non_empty_string,
                    "customer_id": non_empty_string,
                    "display_name": non_empty_string,
                    "seat_count": {"type": "integer", "minimum": 1, "maximum": 4},
                },
            },
            join_game,
        ),
        "create_invite_drafts": ToolDefinition(
            "create_invite_drafts",
            "创建待审批邀约草稿。只生成草稿，不代表已发送。未来局在 recruitment_opens_at 之前会被统一时间策略拒绝；不要通过改写话术绕过。",
            "medium",
            "draft_write",
            {
                "type": "object",
                "required": ["game_id", "invitations"],
                "additionalProperties": False,
                "properties": {
                    "game_id": non_empty_string,
                    "invitations": {"type": "array", "items": invitation_schema, "minItems": 1},
                },
            },
            create_invite_drafts,
        ),
        "update_game_requirement": ToolDefinition(
            "update_game_requirement",
            "更新尚未成局的组局条件。仅用于客户明确补充或协商确认后的时间、时长、玩法、档位、烟况等条件；不能修改参与者、座位快照或生命周期计算字段。",
            "medium",
            "state_write",
            {
                "type": "object",
                "required": ["game_id", "requirement_patch", "reason"],
                "additionalProperties": False,
                "properties": {
                    "game_id": non_empty_string,
                    "requirement_patch": requirement_schema,
                    "reason": non_empty_string,
                },
            },
            update_game_requirement,
        ),
        "create_outbound_message_drafts": ToolDefinition(
            "create_outbound_message_drafts",
            "创建通道无关的待审批外发消息草稿。只落库，不代表已发送，可用于当前用户回复、群消息或其他渠道输出。",
            "medium",
            "draft_write",
            {
                "type": "object",
                "required": ["drafts"],
                "additionalProperties": False,
                "properties": {
                    "drafts": {"type": "array", "items": outbound_message_draft_schema, "minItems": 1},
                },
            },
            create_outbound_message_drafts,
        ),
        "record_candidate_reply": ToolDefinition(
            "record_candidate_reply",
            "记录某个局里客户/候选人本轮发生的参与状态或代表座位数变化，并推进受控状态。适用于已邀约候选人，也适用于当前已在局内的客户。status 可表示 accepted/confirmed/arrived/declined/negotiating/no_reply；客户拒绝、退出、不打了或条件不接受时也要调用，通常用 declined。若 active_games 中该客户已经是相同状态且座位数没有变化，不要重复调用。若客户表示“我这边两个人/我们3个”，模型必须把代表座位数写入 seat_count。",
            "medium",
            "state_write",
            {
                "type": "object",
                "required": ["game_id", "customer_id", "display_name", "status"],
                "additionalProperties": False,
                "properties": {
                    "game_id": non_empty_string,
                    "customer_id": non_empty_string,
                    "display_name": non_empty_string,
                    "status": {"type": "string", "enum": CANDIDATE_REPLY_STATUSES},
                    "seat_count": {"type": "integer", "minimum": 1, "maximum": 4},
                },
            },
            record_candidate_reply,
        ),
        "update_game_status": ToolDefinition(
            "update_game_status",
            "只按状态机更新局的生命周期状态。非法状态迁移由后端拒绝；本工具不能修改时长、烟况、档位、时间或人数等 requirement，不能为了记录用户约束而调用。",
            "medium",
            "state_write",
            {
                "type": "object",
                "required": ["game_id", "status", "reason"],
                "additionalProperties": False,
                "properties": {
                    "game_id": non_empty_string,
                    "status": {"type": "string", "enum": GAME_STATUSES},
                    "reason": non_empty_string,
                },
            },
            update_game_status,
        ),
        "record_badcase": ToolDefinition(
            "record_badcase",
            "记录 badcase/eval 候选样本，不改变业务状态。",
            "low",
            "audit_write",
            badcase_schema,
            record_badcase,
        ),
        "record_user_memory": ToolDefinition(
            "record_user_memory",
            "记录用户表达的当前任务约束和待确认长期画像候选。当前任务约束会立即影响查现有局和找候选人；长期画像候选只进入待审核队列，不直接改客户画像。",
            "medium",
            "state_write",
            memory_schema,
            record_user_memory,
        ),
        "update_context_checkpoint": ToolDefinition(
            "update_context_checkpoint",
            "更新当前会话的长期上下文 checkpoint。模型负责总结需要跨窗口保留的事实、待确认问题和当前任务状态；工具只校验并存储。",
            "medium",
            "state_write",
            checkpoint_schema,
            update_context_checkpoint,
        ),
    }


def known_players_with_requesting_party(
    *,
    known_players: list[dict[str, Any]],
    requesting_party: Any,
) -> list[dict[str, Any]]:
    players = [dict(item) for item in known_players if isinstance(item, dict)]
    if not isinstance(requesting_party, dict):
        return players
    contact_id = str(requesting_party.get("contact_id") or requesting_party.get("customer_id") or "").strip()
    if not contact_id:
        return players
    payload = {
        "customer_id": contact_id,
        "display_name": str(requesting_party.get("contact_name") or requesting_party.get("display_name") or contact_id),
        "source": str(requesting_party.get("source") or "requesting_party"),
        "seat_count": requesting_party.get("seat_count") or requesting_party.get("party_size") or 1,
        "known_member_ids": list(requesting_party.get("known_member_ids") or [contact_id]),
        "anonymous_seat_count": requesting_party.get("anonymous_seat_count"),
    }
    for index, item in enumerate(players):
        if str(item.get("customer_id") or "").strip() != contact_id:
            continue
        merged = {**item}
        for key, value in payload.items():
            if value is None:
                continue
            if key not in merged or merged.get(key) in (None, "", [], {}):
                merged[key] = value
        players[index] = merged
        break
    else:
        players.insert(0, payload)
    return players


def validate_schema(arguments: dict[str, Any], schema: dict[str, Any]) -> str | None:
    return validate_value("arguments", arguments, schema)


def validate_object(key: str, value: dict[str, Any], schema: dict[str, Any]) -> str | None:
    for required_key in schema.get("required") or []:
        if required_key not in value:
            return f"missing required argument: {required_key}"
    properties = schema.get("properties") or {}
    if schema.get("additionalProperties") is False:
        for item_key in value:
            if item_key not in properties:
                return f"unexpected argument: {item_key}"
    for item_key, item_value in value.items():
        prop = properties.get(item_key)
        if not prop:
            continue
        error = validate_value(item_key, item_value, prop)
        if error:
            return error
    return None


def validate_value(key: str, value: Any, schema: dict[str, Any]) -> str | None:
    expected = schema.get("type")
    if expected == "object" and not isinstance(value, dict):
        return f"{key} must be object"
    if expected == "object":
        return validate_object(key, value, schema)
    if expected == "array":
        if not isinstance(value, list):
            return f"{key} must be array"
        if "minItems" in schema and len(value) < int(schema["minItems"]):
            return f"{key} must contain at least {schema['minItems']} item(s)"
        if "maxItems" in schema and len(value) > int(schema["maxItems"]):
            return f"{key} must contain at most {schema['maxItems']} item(s)"
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                error = validate_value(f"{key}[{index}]", item, item_schema)
                if error:
                    return error
        return None
    if expected == "string":
        if not isinstance(value, str):
            return f"{key} must be string"
        if "minLength" in schema and len(value.strip()) < int(schema["minLength"]):
            return f"{key} must have length >= {schema['minLength']}"
        if "enum" in schema and value not in set(str(item) for item in schema["enum"]):
            return f"{key} must be one of: {', '.join(str(item) for item in schema['enum'])}"
        return None
    if expected == "boolean" and not isinstance(value, bool):
        return f"{key} must be boolean"
    if expected == "integer":
        if not isinstance(value, int):
            return f"{key} must be integer"
        if "minimum" in schema and value < int(schema["minimum"]):
            return f"{key} must be >= {schema['minimum']}"
        if "maximum" in schema and value > int(schema["maximum"]):
            return f"{key} must be <= {schema['maximum']}"
    if expected == "number":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return f"{key} must be number"
        if "minimum" in schema and float(value) < float(schema["minimum"]):
            return f"{key} must be >= {schema['minimum']}"
        if "maximum" in schema and float(value) > float(schema["maximum"]):
            return f"{key} must be <= {schema['maximum']}"
    return None


def backend_tool_idempotency_key(
    call: ToolCall,
    *,
    conversation_id: str,
    sender_id: str,
    source_message_id: str | None,
) -> str | None:
    if not source_message_id:
        return None
    canonical_args = json.dumps(call.arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical_args.encode("utf-8")).hexdigest()[:24]
    return (
        f"conversation:{conversation_id}:sender:{sender_id}:"
        f"message:{source_message_id}:tool:{call.name}:args:{digest}"
    )


def idempotency_lock_for_key(key: str) -> threading.RLock:
    with _IDEMPOTENCY_LOCKS_GUARD:
        lock = _IDEMPOTENCY_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _IDEMPOTENCY_LOCKS[key] = lock
        return lock
