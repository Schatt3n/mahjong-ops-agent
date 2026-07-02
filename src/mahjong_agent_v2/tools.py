from __future__ import annotations

import re
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
    trace_recorder: Any | None = None
    allowed_execution_modes: set[str] = field(
        default_factory=lambda: {"read_only", "state_write", "create_pending", "audit_write"}
    )
    allowed_risk_levels: set[str] = field(default_factory=lambda: {"low", "medium", "high"})

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
        self._record(
            trace_id,
            "tool_gateway_received",
            {
                "tool_name": call.name,
                "call": call.to_dict(),
                "step_index": step_index,
                "idempotency_key": idempotency_key,
            },
        )
        existing = self.store.idempotent_result(idempotency_key)
        self._record(
            trace_id,
            "tool_idempotency_checked",
            {
                "tool_name": call.name,
                "step_index": step_index,
                "idempotency_key": idempotency_key,
                "hit": existing is not None,
            },
        )
        if existing is not None:
            return self._complete(
                trace_id,
                step_index,
                ToolResultV2(
                    name=existing.name,
                    called=existing.called,
                    allowed=existing.allowed,
                    result=dict(existing.result),
                    error=existing.error,
                    idempotency_key=idempotency_key,
                    deduplicated=True,
                    state_transitions=list(existing.state_transitions),
                ),
                outcome="deduplicated",
            )
        if definition is None:
            self._record(
                trace_id,
                "tool_definition_checked",
                {"tool_name": call.name, "step_index": step_index, "allowed": False, "error": "unknown_tool"},
                level="WARN",
            )
            return self._complete(
                trace_id,
                step_index,
                ToolResultV2(
                    name=call.name,
                    called=False,
                    allowed=False,
                    error=f"unknown tool: {call.name}",
                    idempotency_key=idempotency_key,
                ),
                outcome="blocked",
                remember_key=idempotency_key,
            )
        self._record(
            trace_id,
            "tool_definition_checked",
            {
                "tool_name": call.name,
                "step_index": step_index,
                "allowed": True,
                "risk_level": definition.risk_level,
                "execution_mode": definition.execution_mode,
            },
        )
        schema_error = validate_schema(call.arguments, definition.schema)
        if schema_error:
            self._record(
                trace_id,
                "tool_schema_checked",
                {"tool_name": call.name, "step_index": step_index, "allowed": False, "error": schema_error},
                level="WARN",
            )
            return self._complete(
                trace_id,
                step_index,
                ToolResultV2(
                    name=call.name,
                    called=False,
                    allowed=False,
                    error=schema_error,
                    idempotency_key=idempotency_key,
                ),
                outcome="blocked",
                remember_key=idempotency_key,
            )
        self._record(
            trace_id,
            "tool_schema_checked",
            {"tool_name": call.name, "step_index": step_index, "allowed": True},
        )
        permission_error = self._permission_error(definition)
        if permission_error:
            self._record(
                trace_id,
                "tool_permission_checked",
                {
                    "tool_name": call.name,
                    "step_index": step_index,
                    "allowed": False,
                    "risk_level": definition.risk_level,
                    "execution_mode": definition.execution_mode,
                    "error": permission_error,
                },
                level="WARN",
            )
            return self._complete(
                trace_id,
                step_index,
                ToolResultV2(
                    name=call.name,
                    called=False,
                    allowed=False,
                    error=permission_error,
                    idempotency_key=idempotency_key,
                ),
                outcome="blocked",
                remember_key=idempotency_key,
            )
        self._record(
            trace_id,
            "tool_permission_checked",
            {
                "tool_name": call.name,
                "step_index": step_index,
                "allowed": True,
                "risk_level": definition.risk_level,
                "execution_mode": definition.execution_mode,
            },
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
        return self._complete(
            trace_id,
            step_index,
            result,
            outcome="executed" if result.called and result.allowed else "failed",
            remember_key=idempotency_key,
        )

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
            matches = [_with_public_game_summary(item) for item in matches]
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
                result={"game": public_game_payload(game)},
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
        if call.name == "update_game_status":
            game, transition = self.store.update_game_status(
                game_id=str(args.get("game_id") or ""),
                status=str(args.get("status") or ""),
                reason=str(args.get("reason") or "update_game_status"),
                trace_id=trace_id,
            )
            return ToolResultV2(
                name=call.name,
                called=True,
                allowed=True,
                result={"game": public_game_payload(game)},
                state_transitions=[transition],
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

    def _complete(
        self,
        trace_id: str,
        step_index: int,
        result: ToolResultV2,
        *,
        outcome: str,
        remember_key: str | None = None,
    ) -> ToolResultV2:
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
            return self._remember(remember_key, result)
        return result

    def _record(self, trace_id: str, step: str, content: dict[str, Any], *, level: str = "INFO") -> None:
        if self.trace_recorder is None:
            return
        record = getattr(self.trace_recorder, "record", None)
        if callable(record):
            record(trace_id, step, content, level=level)

    def _permission_error(self, definition: ToolDefinitionV2) -> str | None:
        if definition.execution_mode not in self.allowed_execution_modes:
            return f"tool execution_mode not allowed: {definition.execution_mode}"
        if definition.risk_level not in self.allowed_risk_levels:
            return f"tool risk_level not allowed: {definition.risk_level}"
        return None


def default_tool_definitions_v2() -> dict[str, ToolDefinitionV2]:
    requirement_schema = requirement_schema_v2()
    return {
        "search_current_games": ToolDefinitionV2(
            name="search_current_games",
            description="查询当前仍有效、可加入或可协商的局。只读工具，模型决定查询条件；不确定的人数/缺口字段请留空，不要写范围对象。",
            risk_level="low",
            execution_mode="read_only",
            schema={
                "type": "object",
                "properties": {
                    "requirement": requirement_schema,
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
                    "requirement": requirement_schema,
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
                    "requirement": requirement_schema,
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
                                "message_text": {
                                    "type": "string",
                                    "minLength": 2,
                                    "maxLength": 80,
                                    "x-public-text": True,
                                },
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
        "update_game_status": ToolDefinitionV2(
            name="update_game_status",
            description="更新局生命周期状态，例如取消局或标记完成。模型提出目标状态；后端状态机校验是否合法。",
            risk_level="medium",
            execution_mode="state_write",
            schema={
                "type": "object",
                "properties": {
                    "game_id": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": ["forming", "inviting", "ready", "cancelled", "finished"],
                    },
                    "reason": {"type": "string"},
                },
                "required": ["game_id", "status", "reason"],
            },
        ),
    }


def requirement_schema_v2() -> dict[str, Any]:
    """Shared model-facing contract for mahjong game requirements.

    The backend validates shape only. It does not infer mahjong meaning from
    free text; the model must fill these fields from context.
    """

    return {
        "type": "object",
        "properties": {
            "game_type": {
                "type": "string",
                "description": "内部结构化玩法编码，例如 hangzhou_mahjong；只能用于工具参数，不能写进客户可见文案。",
            },
            "game_type_label": {
                "type": "string",
                "description": "客户可见玩法中文名，例如杭麻、川麻、红中。",
            },
            "stake": {"type": "string", "description": "结构化档位，例如 0.5、1、2-16。"},
            "stake_options": {"type": "array", "items": {"type": "string"}},
            "start_time": {"type": "string", "description": "明确时间，如 14:00；不确定时留空。"},
            "start_time_text": {
                "type": "string",
                "description": "客户可见时间表达，如人齐开、通宵、今晚、下午4点。",
                "x-public-text": True,
            },
            "start_time_kind": {
                "type": "string",
                "enum": ["exact", "asap_when_full", "overnight", "flexible", "unknown"],
                "description": "内部结构化时间类型。写给客户/候选人时必须转成自然中文：asap_when_full=人齐开，overnight=通宵，flexible=时间可商量。",
            },
            "duration_hours": {"type": "number"},
            "duration_kind": {
                "type": "string",
                "enum": ["fixed_hours", "overnight", "flexible", "unknown"],
                "description": "内部结构化时长类型。写给客户/候选人时必须转成自然中文。",
            },
            "duration_text": {
                "type": "string",
                "description": "客户可见时长表达，如约4小时、通宵、时间可商量。",
                "x-public-text": True,
            },
            "smoke_preference": {
                "type": "string",
                "enum": ["any", "non_smoking", "smoke_ok", "unknown"],
                "description": "内部结构化烟况。写给客户/候选人时必须转成自然中文：any=烟都可，non_smoking=无烟，smoke_ok=有烟。",
            },
            "smoke_label": {
                "type": "string",
                "description": "客户可见烟况中文，如无烟、有烟、烟都可。",
                "x-public-text": True,
            },
            "current_players": {"type": "integer", "minimum": 0, "maximum": 4},
            "missing_players": {
                "type": "integer",
                "minimum": 0,
                "maximum": 4,
                "description": "明确缺几人时填写整数，例如三缺一填 1；只知道想找局/有人吗但缺口不明确时留空，不要填写对象、范围或 minimum/maximum。",
            },
            "seats_total": {"type": "integer", "minimum": 2, "maximum": 8},
            "candidate_preferences": {"type": "object"},
            "user_visible_summary": {
                "type": "string",
                "description": "给模型写客户/候选人文案时参考的自然中文摘要，不允许包含内部枚举、snake_case 或 JSON。",
                "x-public-text": True,
            },
        },
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
        minimum = schema.get("minItems")
        maximum = schema.get("maxItems")
        if minimum is not None and len(value) < minimum:
            return f"{path} must contain at least {minimum} items"
        if maximum is not None and len(value) > maximum:
            return f"{path} must contain at most {maximum} items"
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
        minimum = schema.get("minLength")
        maximum = schema.get("maxLength")
        if minimum is not None and len(value.strip()) < minimum:
            return f"{path} must contain at least {minimum} chars"
        if maximum is not None and len(value) > maximum:
            return f"{path} must contain at most {maximum} chars"
        allowed = schema.get("enum")
        if allowed and value not in allowed:
            return f"{path} must be one of {allowed}"
        if schema.get("x-public-text"):
            public_error = validate_public_text(value, path=path)
            if public_error:
                return public_error
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


def public_game_payload(game) -> dict[str, Any]:
    payload = game.to_dict()
    payload["requirement_public_summary"] = public_requirement_summary(payload.get("requirement") or {})
    return payload


def public_requirement_summary(requirement: dict[str, Any]) -> str:
    summary = str(requirement.get("user_visible_summary") or requirement.get("public_summary") or "").strip()
    if summary:
        return summary
    parts: list[str] = []
    game_label = str(requirement.get("game_type_label") or "").strip()
    if game_label:
        parts.append(game_label)
    stake = str(requirement.get("stake") or "").strip()
    options = requirement.get("stake_options")
    if not stake and isinstance(options, list):
        stake = "/".join(str(item) for item in options if str(item).strip())
    if stake:
        parts.append(f"{stake}档")
    start_time = str(requirement.get("start_time_text") or requirement.get("start_time") or "").strip()
    if start_time:
        parts.append(start_time)
    smoke = str(requirement.get("smoke_label") or "").strip()
    if smoke:
        parts.append(smoke)
    duration = str(requirement.get("duration_text") or "").strip()
    if not duration and requirement.get("duration_hours") is not None:
        duration = f"约{requirement.get('duration_hours')}小时"
    if duration:
        parts.append(duration)
    missing = requirement.get("missing_players")
    if isinstance(missing, int) and missing > 0:
        parts.append(f"缺{missing}")
    return " ".join(parts)


def validate_public_text(value: str, *, path: str) -> str | None:
    if re.search(r"\b[a-zA-Z]+_[a-zA-Z0-9_]+\b", value):
        return f"{path} must be customer-visible text and cannot contain internal snake_case codes"
    if "{" in value or "}" in value:
        return f"{path} must be customer-visible text and cannot contain serialized objects"
    return None


def _with_public_game_summary(match: dict[str, Any]) -> dict[str, Any]:
    copied = dict(match)
    game = copied.get("game")
    if isinstance(game, dict):
        game = dict(game)
        game["requirement_public_summary"] = public_requirement_summary(game.get("requirement") or {})
        copied["game"] = game
    return copied
