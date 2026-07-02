from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .eval import EvalRecorderV2, InMemoryEvalRecorderV2
from .models import ToolCallV2, ToolResultV2
from .store import InMemoryAgentStoreV2


@dataclass(slots=True)
class ToolDefinitionV2:
    name: str
    description: str
    risk_level: str
    execution_mode: str
    schema: dict[str, Any]

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "risk_level": self.risk_level,
            "execution_mode": self.execution_mode,
            "schema": self.schema,
        }


@dataclass(slots=True)
class ToolGatewayV2:
    store: InMemoryAgentStoreV2
    tools: dict[str, ToolDefinitionV2] = field(default_factory=dict)
    eval_recorder: EvalRecorderV2 = field(default_factory=InMemoryEvalRecorderV2)

    def __post_init__(self) -> None:
        if not self.tools:
            self.tools.update(default_tool_definitions_v2())

    def tool_specs_for_prompt(self) -> list[dict[str, Any]]:
        return [definition.to_prompt_dict() for definition in self.tools.values()]

    def execute(
        self,
        call: ToolCallV2,
        *,
        trace_id: str,
        conversation_id: str,
        sender_id: str,
        sender_name: str,
        step_index: int,
    ) -> ToolResultV2:
        definition = self.tools.get(call.name)
        idempotency_key = call.idempotency_key or f"{trace_id}:tool:{step_index}:{call.name}"
        existing = self.store.idempotent_result(idempotency_key)
        if existing is not None:
            return ToolResultV2(
                name=existing.name,
                called=existing.called,
                allowed=existing.allowed,
                result=dict(existing.result),
                error=existing.error,
                idempotency_key=idempotency_key,
                deduplicated=True,
                state_transitions=list(existing.state_transitions),
            )
        if definition is None:
            return self._remember(
                idempotency_key,
                ToolResultV2(
                    name=call.name,
                    called=False,
                    allowed=False,
                    error=f"unknown tool: {call.name}",
                    idempotency_key=idempotency_key,
                ),
            )
        schema_error = validate_schema(call.arguments, definition.schema)
        if schema_error:
            return self._remember(
                idempotency_key,
                ToolResultV2(
                    name=call.name,
                    called=False,
                    allowed=False,
                    error=schema_error,
                    idempotency_key=idempotency_key,
                ),
            )
        try:
            result = self._execute_allowed(
                call,
                trace_id=trace_id,
                conversation_id=conversation_id,
                sender_id=sender_id,
                sender_name=sender_name,
            )
        except Exception as exc:
            result = ToolResultV2(
                name=call.name,
                called=False,
                allowed=False,
                error=f"{type(exc).__name__}: {exc}",
                idempotency_key=idempotency_key,
            )
        result.idempotency_key = idempotency_key
        return self._remember(idempotency_key, result)

    def _execute_allowed(
        self,
        call: ToolCallV2,
        *,
        trace_id: str,
        conversation_id: str,
        sender_id: str,
        sender_name: str,
    ) -> ToolResultV2:
        args = call.arguments
        if call.name == "search_current_games":
            matches = self.store.search_current_games(
                dict(args.get("requirement") or {}),
                limit=int(args.get("limit") or 8),
            )
            return ToolResultV2(name=call.name, called=True, allowed=True, result={"matches": matches})
        if call.name == "search_customers":
            candidates = self.store.search_customers(
                dict(args.get("requirement") or {}),
                exclude_customer_ids=[str(item) for item in args.get("exclude_customer_ids") or []],
                limit=int(args.get("limit") or 8),
            )
            return ToolResultV2(name=call.name, called=True, allowed=True, result={"candidates": candidates})
        if call.name == "create_game":
            game, transition = self.store.create_game(
                conversation_id=conversation_id,
                organizer_id=str(args.get("organizer_id") or sender_id),
                organizer_name=str(args.get("organizer_name") or sender_name),
                requirement=dict(args.get("requirement") or {}),
                known_players=list(args.get("known_players") or []),
                trace_id=trace_id,
            )
            return ToolResultV2(
                name=call.name,
                called=True,
                allowed=True,
                result={"game": game.to_dict()},
                state_transitions=[transition],
            )
        if call.name == "create_invite_drafts":
            drafts, transitions = self.store.create_invite_drafts(
                game_id=str(args.get("game_id") or ""),
                invitations=list(args.get("invitations") or []),
                trace_id=trace_id,
            )
            return ToolResultV2(
                name=call.name,
                called=True,
                allowed=True,
                result={"drafts": [draft.to_dict() for draft in drafts]},
                state_transitions=transitions,
            )
        if call.name == "record_candidate_reply":
            game, transitions = self.store.record_candidate_reply(
                game_id=str(args.get("game_id") or ""),
                customer_id=str(args.get("customer_id") or ""),
                status=str(args.get("status") or ""),
                trace_id=trace_id,
            )
            return ToolResultV2(
                name=call.name,
                called=True,
                allowed=True,
                result={"game": game.to_dict()},
                state_transitions=transitions,
            )
        if call.name == "record_badcase":
            record = self.eval_recorder.record_badcase(
                dict(args),
                trace_id=trace_id,
                conversation_id=conversation_id,
            )
            return ToolResultV2(
                name=call.name,
                called=True,
                allowed=True,
                result={"recorded": True, "badcase": record},
            )
        return ToolResultV2(name=call.name, called=False, allowed=False, error=f"tool not implemented: {call.name}")

    def _remember(self, key: str | None, result: ToolResultV2) -> ToolResultV2:
        self.store.remember_result(key, result)
        return result


def default_tool_definitions_v2() -> dict[str, ToolDefinitionV2]:
    return {
        "search_current_games": ToolDefinitionV2(
            name="search_current_games",
            description="查询当前仍有效、可加入或可协商的局。只读工具，模型决定查询条件。",
            risk_level="low",
            execution_mode="read_only",
            schema={
                "type": "object",
                "properties": {
                    "requirement": {"type": "object"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                },
                "required": ["requirement"],
            },
        ),
        "search_customers": ToolDefinitionV2(
            name="search_customers",
            description="按模型给出的组局条件搜索候选客户。只返回候选，不发消息。",
            risk_level="low",
            execution_mode="read_only",
            schema={
                "type": "object",
                "properties": {
                    "requirement": {"type": "object"},
                    "exclude_customer_ids": {"type": "array", "items": {"type": "string"}},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                },
                "required": ["requirement"],
            },
        ),
        "create_game": ToolDefinitionV2(
            name="create_game",
            description="创建一个新的待组局状态。模型必须提供结构化 requirement 和已知玩家。",
            risk_level="medium",
            execution_mode="state_write",
            schema={
                "type": "object",
                "properties": {
                    "requirement": {"type": "object"},
                    "organizer_id": {"type": "string"},
                    "organizer_name": {"type": "string"},
                    "known_players": {"type": "array", "items": {"type": "object"}},
                },
                "required": ["requirement"],
            },
        ),
        "create_invite_drafts": ToolDefinitionV2(
            name="create_invite_drafts",
            description="创建待审批候选人邀约草稿。模型负责为每个候选人生成可见文案；后端只落 pending draft。",
            risk_level="high",
            execution_mode="create_pending",
            schema={
                "type": "object",
                "properties": {
                    "game_id": {"type": "string"},
                    "invitations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "customer_id": {"type": "string"},
                                "display_name": {"type": "string"},
                                "message_text": {"type": "string"},
                            },
                            "required": ["customer_id", "message_text"],
                        },
                    },
                },
                "required": ["game_id", "invitations"],
            },
        ),
        "record_candidate_reply": ToolDefinitionV2(
            name="record_candidate_reply",
            description="记录候选人回复并推进局状态。语义由模型判断，状态合法性由后端校验。",
            risk_level="medium",
            execution_mode="state_write",
            schema={
                "type": "object",
                "properties": {
                    "game_id": {"type": "string"},
                    "customer_id": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": ["confirmed", "declined", "negotiating", "no_reply"],
                    },
                },
                "required": ["game_id", "customer_id", "status"],
            },
        ),
        "record_badcase": ToolDefinitionV2(
            name="record_badcase",
            description="当模型或老板发现回复不对时，记录 badcase/eval 候选。不会改变业务状态。",
            risk_level="low",
            execution_mode="audit_write",
            schema={
                "type": "object",
                "properties": {
                    "reason": {"type": "string"},
                    "input": {"type": "object"},
                    "expected": {"type": "object"},
                },
                "required": ["reason"],
            },
        ),
    }


def validate_schema(value: dict[str, Any], schema: dict[str, Any]) -> str | None:
    if not isinstance(value, dict):
        return "arguments must be object"
    return _validate_value(value, schema, path="arguments")


def _validate_value(value: Any, schema: dict[str, Any], *, path: str) -> str | None:
    schema_type = schema.get("type")
    if schema_type == "object":
        if not isinstance(value, dict):
            return f"{path} must be object"
        for required in schema.get("required") or []:
            if required not in value:
                return f"{path}.{required} is required"
        properties = schema.get("properties") or {}
        if isinstance(properties, dict):
            for name, child_schema in properties.items():
                if name in value and isinstance(child_schema, dict):
                    error = _validate_value(value[name], child_schema, path=f"{path}.{name}")
                    if error:
                        return error
        return None
    if schema_type == "array":
        if not isinstance(value, list):
            return f"{path} must be array"
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                error = _validate_value(item, item_schema, path=f"{path}[{index}]")
                if error:
                    return error
        return None
    if schema_type == "string":
        if not isinstance(value, str):
            return f"{path} must be string"
        allowed = schema.get("enum")
        if allowed and value not in allowed:
            return f"{path} must be one of {allowed}"
        return None
    if schema_type == "integer":
        if not isinstance(value, int):
            return f"{path} must be integer"
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if minimum is not None and value < minimum:
            return f"{path} must be >= {minimum}"
        if maximum is not None and value > maximum:
            return f"{path} must be <= {maximum}"
        return None
    if schema_type == "number":
        if not isinstance(value, (int, float)):
            return f"{path} must be number"
        return None
    if schema_type == "boolean":
        if not isinstance(value, bool):
            return f"{path} must be boolean"
        return None
    return None
