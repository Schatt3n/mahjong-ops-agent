"""提供工具执行网关，统一完成参数校验、授权、幂等、并发锁和审计。

"""

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
from .participation_access import participation_access_source


_IDEMPOTENCY_LOCKS: dict[str, threading.RLock] = {}
_IDEMPOTENCY_LOCKS_GUARD = threading.RLock()


@dataclass(slots=True)
class ToolGateway:
    """工具可信执行边界；模型只能提出调用，网关负责校验后执行并记录证据。

    主 Agent 链路上游是 ``ToolExecutionService._execute_one``；应用层记录人工 badcase
    时也会直接调用。动态下游按 ``ToolCall.name`` 从注册表查找 ``ToolDefinition``，
    最后调用其 handler；模型从不直接调用具体工具函数。
    """
    store: AgentStore
    tools: dict[str, ToolDefinition] = field(default_factory=dict)
    trace_recorder: Any | None = None
    allowed_execution_modes: set[str] = field(default_factory=lambda: {"read_only", "state_write", "draft_write", "audit_write"})
    allowed_risk_levels: set[str] = field(default_factory=lambda: {"low", "medium"})

    def __post_init__(self) -> None:
        """在未注入自定义注册表时装配默认工具，保留测试和部署时替换工具集的能力。"""
        if not self.tools:
            self.tools.update(default_tool_definitions(self.store))

    def tool_specs_for_prompt(self) -> list[dict[str, Any]]:
        """向模型暴露声明式工具契约；工具是否真的可执行仍由网关在运行时裁决。"""
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
        """先为幂等计算规范参数，再校验原始调用并动态执行 handler。

        canonicalizer 不会改写 ``call.arguments``；schema 校验和 handler 收到的仍是模型
        提交的原始参数。随后还会检查部署权限、资源授权和幂等占领。
        """
        # 模型参数先经工具自己的规范化逻辑处理，再由后端生成权威幂等键。
        # 模型提供的 idempotency_key 只能作为建议，不能成为副作用操作的身份依据。
        definition = self.tools.get(call.name)
        canonical_arguments = (
            definition.canonical_arguments(call.arguments)
            if definition is not None
            else dict(call.arguments)
        )
        idempotency_key = (
            backend_tool_idempotency_key(
                call,
                conversation_id=conversation_id,
                sender_id=sender_id,
                source_message_id=source_message_id,
                arguments=canonical_arguments,
            )
            or trace_tool_idempotency_key(
                call,
                trace_id=trace_id,
                conversation_id=conversation_id,
                sender_id=sender_id,
                arguments=canonical_arguments,
            )
        )
        self._record(
            trace_id,
            "tool_gateway_received",
            {"tool_name": call.name, "call": call.to_dict(), "step_index": step_index, "idempotency_key": idempotency_key},
        )
        # 锁粒度是单个幂等键：相同操作串行，不同用户、不同局的工具调用仍可并行。
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
            # 四层可信执行校验：注册存在、参数契约、部署权限、业务资源归属。
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
                # 授权拒绝没有副作用；同一轮补做查询后权限依据可能改变，不能把拒绝永久去重。
                # 原始判定仍完整写入 trace，成功写入的幂等语义保持不变。
                return self._complete(trace_id, step_index, result, outcome="blocked")
            self._record(trace_id, "tool_authorization_checked", {"tool_name": call.name, "allowed": True})
            # 先持久化占领幂等键，再执行有副作用的 handler。即使多进程同时到达，
            # 也只有成功占领的一方可以落库或外发消息，其余调用复用已有结果。
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
                # 导航：handler 在 registry 中以 partial(具体工具函数, store) 绑定，
                # 因此 VSCode 可能看不到从这里到 domains.tools.* 的静态调用关系。
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
        """统一记录工具终态，并把结果写入幂等存储供重试、恢复和审计复用。"""
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
        """按部署允许列表拦截执行模式或风险级别超出当前环境的工具。"""
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
        """把写操作绑定到已认证发送者、当前会话及其可访问的业务资源。

        模型负责提出业务参数，但模型不是身份提供方。这里防止错误或对抗性输出
        冒充其他客户写入数据。会话是隐私边界，不是 Game 的唯一所有者：本人参与可凭
        当前任务的查询证据、邀请或已有参与关系跨会话进行，管理操作仍限定创建会话。对于
        引用了历史消息的入局操作，还要求先解析出权威业务引用，禁止凭语义猜测落库。
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
                if call.name in {"join_game", "record_candidate_reply"}:
                    if participation_access_source(self.store, game, conversation_id=conversation_id, sender_id=sender_id):
                        return None
                    return (
                        "tool participation access required: first search current games in this customer's "
                        "current task or resolve their invitation; knowing a game_id alone grants no write access"
                    )
                if game.conversation_id != conversation_id:
                    return (
                        "tool resource mismatch: game management requires its originating conversation; "
                        "participation access does not grant management permission"
                    )
        return None

    def _record(self, trace_id: str, step: str, content: dict[str, Any], *, level: str = "INFO") -> None:
        """记录网关判定证据，使一次工具调用为何执行或被拒绝可以完整回溯。"""
        if self.trace_recorder is not None:
            self.trace_recorder.record(trace_id, step, content, level=level)


def backend_tool_idempotency_key(
    call: ToolCall,
    *,
    conversation_id: str,
    sender_id: str,
    source_message_id: str | None,
    arguments: dict[str, Any] | None = None,
) -> str | None:
    """基于上游消息、调用主体、工具名和规范化参数生成跨重试稳定的幂等键。"""
    if not source_message_id:
        return None
    canonical_args = json.dumps(
        call.arguments if arguments is None else arguments,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical_args.encode("utf-8")).hexdigest()[:24]
    return (
        f"conversation:{conversation_id}:sender:{sender_id}:"
        f"message:{source_message_id}:tool:{call.name}:args:{digest}"
    )


def trace_tool_idempotency_key(
    call: ToolCall,
    *,
    trace_id: str,
    conversation_id: str,
    sender_id: str,
    arguments: dict[str, Any] | None = None,
) -> str:
    """上游没有消息 ID 时，基于本次 trace 和规范化参数生成后端幂等键。

    模型生成的键和循环步号都不是可信执行元数据。同一 trace 内使用规范化参数
    标识实际操作，可以避免模型在中间调用了其他工具后再次提出同一动作而重复执行。
    """

    canonical_args = json.dumps(
        call.arguments if arguments is None else arguments,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical_args.encode("utf-8")).hexdigest()[:24]
    return (
        f"trace:{trace_id}:conversation:{conversation_id}:sender:{sender_id}:"
        f"tool:{call.name}:args:{digest}"
    )


def idempotency_lock_for_key(key: str) -> threading.RLock:
    """获取进程内按幂等键分片的可重入锁，缩小并发互斥范围。"""
    with _IDEMPOTENCY_LOCKS_GUARD:
        lock = _IDEMPOTENCY_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _IDEMPOTENCY_LOCKS[key] = lock
        return lock
