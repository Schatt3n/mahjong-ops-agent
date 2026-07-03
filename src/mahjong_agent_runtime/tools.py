from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from .models import ToolCallV3, ToolResultV3
from .store import InMemoryAgentStoreV3


ToolHandlerV3 = Callable[[ToolCallV3, str, str, str, str], ToolResultV3]
_IDEMPOTENCY_LOCKS: dict[str, threading.RLock] = {}
_IDEMPOTENCY_LOCKS_GUARD = threading.RLock()
CANDIDATE_REPLY_STATUSES = ["accepted", "confirmed", "arrived", "declined", "negotiating", "no_reply"]
GAME_STATUSES = ["forming", "inviting", "ready", "cancelled", "finished"]


@dataclass(slots=True)
class ToolDefinitionV3:
    name: str
    description: str
    risk_level: str
    execution_mode: str
    schema: dict[str, Any]
    handler: ToolHandlerV3 | None = None

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "risk_level": self.risk_level,
            "execution_mode": self.execution_mode,
            "schema": self.schema,
        }


@dataclass(slots=True)
class ToolGatewayV3:
    store: InMemoryAgentStoreV3
    tools: dict[str, ToolDefinitionV3] = field(default_factory=dict)
    trace_recorder: Any | None = None
    allowed_execution_modes: set[str] = field(default_factory=lambda: {"read_only", "state_write", "draft_write", "audit_write"})
    allowed_risk_levels: set[str] = field(default_factory=lambda: {"low", "medium"})

    def __post_init__(self) -> None:
        if not self.tools:
            self.tools.update(default_tool_definitions_v3(self.store))

    def tool_specs_for_prompt(self) -> list[dict[str, Any]]:
        return [definition.to_prompt_dict() for definition in self.tools.values()]

    def execute(
        self,
        call: ToolCallV3,
        *,
        trace_id: str,
        conversation_id: str,
        sender_id: str,
        sender_name: str,
        step_index: int,
        source_message_id: str | None = None,
    ) -> ToolResultV3:
        definition = self.tools.get(call.name)
        idempotency_key = (
            backend_tool_idempotency_key(call, source_message_id=source_message_id)
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
                result = ToolResultV3(
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
                result = ToolResultV3(name=call.name, called=False, allowed=False, error=f"unknown tool: {call.name}", idempotency_key=idempotency_key)
                self._record(trace_id, "tool_definition_checked", {"tool_name": call.name, "allowed": False}, level="WARN")
                return self._complete(trace_id, step_index, result, outcome="blocked", remember_key=idempotency_key)
            self._record(trace_id, "tool_definition_checked", {"tool_name": call.name, "allowed": True})
            schema_error = validate_schema(call.arguments, definition.schema)
            if schema_error:
                result = ToolResultV3(name=call.name, called=False, allowed=False, error=schema_error, idempotency_key=idempotency_key)
                self._record(trace_id, "tool_schema_checked", {"tool_name": call.name, "allowed": False, "error": schema_error}, level="WARN")
                return self._complete(trace_id, step_index, result, outcome="blocked", remember_key=idempotency_key)
            self._record(trace_id, "tool_schema_checked", {"tool_name": call.name, "allowed": True})
            permission_error = self._permission_error(definition)
            if permission_error:
                result = ToolResultV3(name=call.name, called=False, allowed=False, error=permission_error, idempotency_key=idempotency_key)
                self._record(trace_id, "tool_permission_checked", {"tool_name": call.name, "allowed": False, "error": permission_error}, level="WARN")
                return self._complete(trace_id, step_index, result, outcome="blocked", remember_key=idempotency_key)
            self._record(trace_id, "tool_permission_checked", {"tool_name": call.name, "allowed": True})
            claimed_result = ToolResultV3(
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
                    claimed_existing = ToolResultV3(
                        name=call.name,
                        called=False,
                        allowed=False,
                        error="idempotency key already claimed but result is unavailable",
                        idempotency_key=idempotency_key,
                    )
                result = ToolResultV3(
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
                result = ToolResultV3(name=call.name, called=False, allowed=False, error=f"{type(exc).__name__}: {exc}")
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
        result: ToolResultV3,
        *,
        outcome: str,
        remember_key: str | None = None,
    ) -> ToolResultV3:
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

    def _permission_error(self, definition: ToolDefinitionV3) -> str | None:
        if definition.execution_mode not in self.allowed_execution_modes:
            return f"tool execution_mode not allowed: {definition.execution_mode}"
        if definition.risk_level not in self.allowed_risk_levels:
            return f"tool risk_level not allowed: {definition.risk_level}"
        return None

    def _record(self, trace_id: str, step: str, content: dict[str, Any], *, level: str = "INFO") -> None:
        if self.trace_recorder is not None:
            self.trace_recorder.record(trace_id, step, content, level=level)


def default_tool_definitions_v3(store: InMemoryAgentStoreV3) -> dict[str, ToolDefinitionV3]:
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

    def search_current_games(call: ToolCallV3, trace_id: str, conversation_id: str, sender_id: str, sender_name: str) -> ToolResultV3:
        matches = store.search_current_games(dict(call.arguments.get("requirement") or {}), limit=int(call.arguments.get("limit") or 8))
        return ToolResultV3(name=call.name, called=True, allowed=True, result={"matches": matches})

    def search_customers(call: ToolCallV3, trace_id: str, conversation_id: str, sender_id: str, sender_name: str) -> ToolResultV3:
        candidates = store.search_customers(
            dict(call.arguments.get("requirement") or {}),
            exclude_customer_ids=[str(item) for item in call.arguments.get("exclude_customer_ids") or []],
            limit=int(call.arguments.get("limit") or 8),
        )
        return ToolResultV3(name=call.name, called=True, allowed=True, result={"candidates": candidates})

    def create_game(call: ToolCallV3, trace_id: str, conversation_id: str, sender_id: str, sender_name: str) -> ToolResultV3:
        game, transition = store.create_game(
            conversation_id=conversation_id,
            organizer_id=str(call.arguments["organizer_id"]),
            organizer_name=str(call.arguments["organizer_name"]),
            requirement=dict(call.arguments.get("requirement") or {}),
            known_players=list(call.arguments.get("known_players") or []),
            trace_id=trace_id,
        )
        return ToolResultV3(name=call.name, called=True, allowed=True, result={"game": game.to_dict()}, state_transitions=[transition])

    def create_invite_drafts(call: ToolCallV3, trace_id: str, conversation_id: str, sender_id: str, sender_name: str) -> ToolResultV3:
        drafts, transitions = store.create_invite_drafts(
            game_id=str(call.arguments.get("game_id") or ""),
            invitations=list(call.arguments.get("invitations") or []),
            trace_id=trace_id,
        )
        return ToolResultV3(name=call.name, called=True, allowed=True, result={"drafts": [item.to_dict() for item in drafts]}, state_transitions=transitions)

    def create_outbound_message_drafts(call: ToolCallV3, trace_id: str, conversation_id: str, sender_id: str, sender_name: str) -> ToolResultV3:
        drafts, transitions = store.create_outbound_message_drafts(
            conversation_id=conversation_id,
            drafts=list(call.arguments.get("drafts") or []),
            trace_id=trace_id,
        )
        return ToolResultV3(name=call.name, called=True, allowed=True, result={"drafts": [item.to_dict() for item in drafts]}, state_transitions=transitions)

    def record_candidate_reply(call: ToolCallV3, trace_id: str, conversation_id: str, sender_id: str, sender_name: str) -> ToolResultV3:
        game, transitions = store.record_candidate_reply(
            game_id=str(call.arguments["game_id"]),
            customer_id=str(call.arguments["customer_id"]),
            display_name=str(call.arguments["display_name"]),
            status=str(call.arguments["status"]),
            trace_id=trace_id,
        )
        return ToolResultV3(name=call.name, called=True, allowed=True, result={"game": game.to_dict()}, state_transitions=transitions)

    def update_game_status(call: ToolCallV3, trace_id: str, conversation_id: str, sender_id: str, sender_name: str) -> ToolResultV3:
        game, transition = store.update_game_status(
            game_id=str(call.arguments["game_id"]),
            status=str(call.arguments["status"]),
            reason=str(call.arguments["reason"]),
            trace_id=trace_id,
        )
        return ToolResultV3(name=call.name, called=True, allowed=True, result={"game": game.to_dict()}, state_transitions=[transition])

    def record_badcase(call: ToolCallV3, trace_id: str, conversation_id: str, sender_id: str, sender_name: str) -> ToolResultV3:
        record = store.record_badcase(dict(call.arguments), trace_id=trace_id, conversation_id=conversation_id)
        return ToolResultV3(name=call.name, called=True, allowed=True, result={"recorded": True, "badcase": record})

    def update_context_checkpoint(call: ToolCallV3, trace_id: str, conversation_id: str, sender_id: str, sender_name: str) -> ToolResultV3:
        checkpoint, transition = store.upsert_conversation_checkpoint(
            conversation_id=conversation_id,
            summary=str(call.arguments["summary"]),
            facts=dict(call.arguments.get("facts") or {}),
            open_questions=[str(item) for item in call.arguments.get("open_questions") or []],
            trace_id=trace_id,
        )
        return ToolResultV3(
            name=call.name,
            called=True,
            allowed=True,
            result={"checkpoint": checkpoint.to_dict()},
            state_transitions=[transition],
        )

    return {
        "search_current_games": ToolDefinitionV3(
            "search_current_games",
            "只读查询当前局池。模型提供结构化 requirement；工具只按字段匹配，不理解自然语言。",
            "low",
            "read_only",
            {"type": "object", "required": ["requirement"], "properties": {"requirement": requirement_schema, "limit": {"type": "integer", "minimum": 1, "maximum": 20}}},
            search_current_games,
        ),
        "search_customers": ToolDefinitionV3(
            "search_customers",
            "只读查询候选客户。模型负责给出筛选条件；工具只做确定性排序。",
            "low",
            "read_only",
            {"type": "object", "required": ["requirement"], "properties": {"requirement": requirement_schema, "exclude_customer_ids": {"type": "array", "items": {"type": "string"}}, "limit": {"type": "integer", "minimum": 1, "maximum": 20}}},
            search_customers,
        ),
        "create_game": ToolDefinitionV3(
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
                },
            },
            create_game,
        ),
        "create_invite_drafts": ToolDefinitionV3(
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
        "create_outbound_message_drafts": ToolDefinitionV3(
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
        "record_candidate_reply": ToolDefinitionV3(
            "record_candidate_reply",
            "记录候选人反馈并推进受控状态。",
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
                },
            },
            record_candidate_reply,
        ),
        "update_game_status": ToolDefinitionV3(
            "update_game_status",
            "按状态机更新局状态。非法状态迁移由后端拒绝。",
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
        "record_badcase": ToolDefinitionV3(
            "record_badcase",
            "记录 badcase/eval 候选样本，不改变业务状态。",
            "low",
            "audit_write",
            {"type": "object", "additionalProperties": True},
            record_badcase,
        ),
        "update_context_checkpoint": ToolDefinitionV3(
            "update_context_checkpoint",
            "更新当前会话的长期上下文 checkpoint。模型负责总结需要跨窗口保留的事实、待确认问题和当前任务状态；工具只校验并存储。",
            "medium",
            "state_write",
            checkpoint_schema,
            update_context_checkpoint,
        ),
    }


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
    return None


def backend_tool_idempotency_key(call: ToolCallV3, *, source_message_id: str | None) -> str | None:
    if not source_message_id:
        return None
    canonical_args = json.dumps(call.arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical_args.encode("utf-8")).hexdigest()[:24]
    return f"message:{source_message_id}:tool:{call.name}:args:{digest}"


def idempotency_lock_for_key(key: str) -> threading.RLock:
    with _IDEMPOTENCY_LOCKS_GUARD:
        lock = _IDEMPOTENCY_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _IDEMPOTENCY_LOCKS[key] = lock
        return lock
