"""Permissioned, idempotent gateway for all model-proposed tool calls."""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field
from typing import Any

from ...models import ToolCall, ToolResult
from ...stores import AgentStore
from .registry import ToolDefinition, default_tool_definitions
from .validation import validate_schema


_IDEMPOTENCY_LOCKS: dict[str, threading.RLock] = {}
_IDEMPOTENCY_LOCKS_GUARD = threading.RLock()


@dataclass(slots=True)
class ToolGateway:
    store: AgentStore
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
                waiting_match_candidate = call.name in {"join_game", "record_candidate_reply"} and any(
                    draft.recipient_id == sender_id
                    and draft.purpose == "waiting_match_notification"
                    and str(draft.metadata.get("game_id") or "") == game_id
                    for draft in self.store.outbound_message_drafts.values()
                )
                if game.conversation_id != conversation_id and not (invited_candidate or waiting_match_candidate):
                    return (
                        "tool resource mismatch: game belongs to another conversation; "
                        f"expected={conversation_id!r}, got={game.conversation_id!r}"
                    )
        return None

    def _record(self, trace_id: str, step: str, content: dict[str, Any], *, level: str = "INFO") -> None:
        if self.trace_recorder is not None:
            self.trace_recorder.record(trace_id, step, content, level=level)


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
