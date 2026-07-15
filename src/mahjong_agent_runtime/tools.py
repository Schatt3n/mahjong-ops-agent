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


def current_game_search_reply_contract(requirement: dict[str, Any], matches: list[dict[str, Any]]) -> dict[str, Any]:
    match_summaries = [
        str(item.get("game", {}).get("requirement", {}).get("user_visible_summary") or "").strip()
        for item in matches
    ]
    match_summaries = [item for item in match_summaries if item]
    return {
        "source_tool": "search_current_games",
        "matched_query_requirement": requirement,
        "matched_result_summaries": match_summaries,
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
        "bad_reply_examples": ["七点三缺一，0.5无烟杭麻，打吗", "已按你的画像找到七点三缺一"],
    }


@dataclass(slots=True)
class ToolDefinition:
    name: str
    description: str
    risk_level: str
    execution_mode: str
    schema: dict[str, Any]
    handler: ToolHandler | None = None

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "risk_level": self.risk_level,
            "execution_mode": self.execution_mode,
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
            "status": {"type": "string"},
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
        return ToolResult(
            name=call.name,
            called=True,
            allowed=True,
            result={"game": game_for_model_context(game, store.customers)},
            state_transitions=[transition],
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
        "search_current_games": ToolDefinition(
            "search_current_games",
            "只读查询当前局池。模型提供结构化 requirement；工具只按字段匹配，不理解自然语言。",
            "low",
            "read_only",
            {"type": "object", "required": ["requirement"], "properties": {"requirement": requirement_schema, "limit": {"type": "integer", "minimum": 1, "maximum": 20}}},
            search_current_games,
        ),
        "search_customers": ToolDefinition(
            "search_customers",
            "只读查询候选客户。模型负责给出筛选条件；工具只做确定性排序，并会参考关系画像避开不愿同桌的人。若已知当前局内人员，应在 requirement 里提供 existing_player_ids 或 organizer_id。",
            "low",
            "read_only",
            {"type": "object", "required": ["requirement"], "properties": {"requirement": requirement_schema, "exclude_customer_ids": {"type": "array", "items": {"type": "string"}}, "limit": {"type": "integer", "minimum": 1, "maximum": 20}}},
            search_customers,
        ),
        "create_game": ToolDefinition(
            "create_game",
            "创建待组局记录。只落库，不发消息、不确认房间。模型必须显式提供 organizer_id 和 organizer_name，后端不从当前消息脑补组织者。",
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
        "create_invite_drafts": ToolDefinition(
            "create_invite_drafts",
            "创建待审批邀约草稿。只生成草稿，不代表已发送。",
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
