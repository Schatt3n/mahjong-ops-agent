"""提供工具执行网关，统一完成参数校验、授权、幂等、并发锁和审计。

"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field
from typing import Any

from ...models import ToolCall, ToolResult, now
from ...stores import AgentStore
from ...stores.idempotency_common import tool_result_is_retryable_no_effect
from .registry import ToolDefinition, default_tool_definitions
from .validation import validate_schema
from .participation_access import participation_access_source
from ..game_domain import OPENING_NOT_BEFORE, planned_game_start_at
from ..game_participants import (
    LateParticipationRequiresConfirmation, ParticipationDeadlineRequiresConfirmation, ParticipationTermsChangedError,
    validate_current_participation_terms,
    validate_game_accepts_confirmation, validate_late_participation,
)
from ..invitation_policy import tool_confirms_seat
from ..recruitment_options import validate_reported_withdrawal


_IDEMPOTENCY_LOCKS: dict[str, threading.RLock] = {}
_IDEMPOTENCY_LOCKS_GUARD = threading.RLock()
_PARTICIPATION_TOOLS_REQUIRING_QUOTE_REFERENCE = frozenset({
    "join_game", "record_candidate_reply", "record_registered_member_withdrawal",
})


def _quote_reference_is_unresolved(contract: dict[str, Any] | None) -> bool:
    """向模型展示和实际执行参与工具时使用同一引用边界。"""
    return bool(
        contract and contract.get("quoted_message_present") is True
        and contract.get("business_reference_resolved") is not True
    )


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

    def tool_specs_for_prompt(
        self, *, message_reference_contract: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """展示本轮可提出的工具，避免模型把未解析引用误当成参与操作的入口。"""
        hidden = _PARTICIPATION_TOOLS_REQUIRING_QUOTE_REFERENCE if _quote_reference_is_unresolved(
            message_reference_contract
        ) else ()
        return [definition.to_prompt_dict() for name, definition in self.tools.items() if name not in hidden]

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
        execution_scope: str | None = None,
    ) -> ToolResult:
        """原始参数先过合同，再允许规范化与复用幂等结果；合法调用动态执行 handler。

        canonicalizer 不会改写 ``call.arguments``；schema 校验和 handler 收到的仍是模型
        提交的原始参数。非法调用只记录审计，不污染合法参数的幂等缓存。
        """
        # 模型参数先经工具自己的规范化逻辑处理，再由后端生成权威幂等键。
        # 模型提供的 idempotency_key 只能作为建议，不能成为副作用操作的身份依据。
        definition = self.tools.get(call.name)
        schema_error = validate_schema(call.arguments, definition.schema) if definition is not None else None
        requirement = call.arguments.get("requirement")
        if (call.name == "create_game" and isinstance(requirement, dict) and OPENING_NOT_BEFORE in requirement
                and (not isinstance(requirement[OPENING_NOT_BEFORE], str)
                     or requirement[OPENING_NOT_BEFORE] != (message_reference_contract or {}).get(OPENING_NOT_BEFORE))):
            schema_error = "opening_not_before_at is a verified backend condition, not a model argument"
        canonical_arguments = (
            definition.canonical_arguments(call.arguments)
            if definition is not None and schema_error is None
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
            # 规范参数可能省略非法未知字段；缓存命中不能替代原始参数校验，拒绝结果也
            # 不能占用随后合法调用的缓存键，否则纠正参数后仍会复用旧拒绝。
            if definition is None:
                result = ToolResult(name=call.name, called=False, allowed=False, error=f"unknown tool: {call.name}", idempotency_key=idempotency_key)
                self._record(trace_id, "tool_definition_checked", {"tool_name": call.name, "allowed": False}, level="WARN")
                return self._complete(trace_id, step_index, result, outcome="blocked")
            self._record(trace_id, "tool_definition_checked", {"tool_name": call.name, "allowed": True})
            # 四层可信执行校验：注册存在、参数契约、部署权限、业务资源归属。
            if schema_error:
                result = ToolResult(name=call.name, called=False, allowed=False, error=schema_error, idempotency_key=idempotency_key)
                self._record(trace_id, "tool_schema_checked", {"tool_name": call.name, "allowed": False, "error": schema_error}, level="WARN")
                return self._complete(trace_id, step_index, result, outcome="blocked")
            self._record(trace_id, "tool_schema_checked", {"tool_name": call.name, "allowed": True})
            board_claim_error = self._board_claim_error(call, message_reference_contract)
            if board_claim_error:
                # 精确目标约束先于幂等缓存，防止同条消息复用错误牌局的历史结果。
                result = ToolResult(name=call.name, called=False, allowed=False, error=board_claim_error,
                                    idempotency_key=idempotency_key)
                self._record(trace_id, "tool_authorization_checked", {"tool_name": call.name, "allowed": False,
                    "error": board_claim_error, "rule_id": "GROUP_BOARD_CLAIM_TARGET"}, level="WARN")
                return self._complete(trace_id, step_index, result, outcome="blocked")
            if call.name == "create_game" and "game_creation_authorization" in (message_reference_contract or {}):
                creation_scope = message_reference_contract["game_creation_authorization"]
                if creation_scope != "create":
                    # 分类服务只提供可信绑定的范围。拒绝先于幂等缓存及所有业务副作用。
                    inquire_only = creation_scope == "inquire"
                    error = ("顾客只委托了解有没有人愿意打，未授权新建牌局。" if inquire_only
                             else "拟建牌局的委托范围或具体条件尚未核验。")
                    instruction = (
                        "此建局未执行。按顾客实际授权处理，必要时只读查局或查候选意向，不要重复提交同一写入。"
                        if inquire_only else
                        "此建局未执行。核对本人原话；已有组桌委托要保留。检查拟定的开场、结束、到店界限和牌局有效期。"
                        "若只缺开场时刻等必要条件，直接询问该缺项；不要重问已获得的授权，也不要用只读候选查询代替澄清。"
                        "不要原样重提这次写入。"
                    )
                    rule_id = "CREATE_GAME_CUSTOMER_AUTHORIZATION"
                    result = ToolResult(name=call.name, called=False, allowed=False, error=error, idempotency_key=idempotency_key,
                        result={"validation_error": {"code": "create_game_authorization_required", "message": error},
                                "instruction": instruction})
                    self._record(trace_id, "tool_authorization_checked", {"tool_name": call.name, "allowed": False,
                                 "error": error, "rule_id": rule_id}, level="WARN")
                    return self._complete(trace_id, step_index, result, outcome="blocked")
            # 每轮作用域由已核验的输入来源生成，不能修改共享 gateway 的全局权限。
            # 先检查再复用幂等结果，避免旧动作缓存绕过当前的询局边界。
            visual_read_only = execution_scope == "visual_read_only"
            full_table_notice = execution_scope == "full_table_notice_read_only"
            read_or_audit = definition.execution_mode in {"read_only", "audit_write"}
            if execution_scope is not None and not (
                (visual_read_only and read_or_audit)
                or (full_table_notice and definition.execution_mode == "read_only")
                or (execution_scope == "query_only" and (
                    read_or_audit or call.name in {"record_user_memory", "update_context_checkpoint"}
                ))
            ):
                error = (
                    "本轮只有图片或表情，可以查询或澄清。请让客户用文字确认后再执行业务变更。"
                    if visual_read_only else
                    "成局回告只读，同一回合的通知不授权任何业务写入。"
                    if full_table_notice else
                    "本轮是只读询局或目标未确定：可以查询、记录本人条件或澄清，不得建局、占座、邀约或改变业务状态。"
                )
                rule_id = {
                    "visual_read_only": "WECHAT_MEDIA_NO_BUSINESS_WRITE",
                    "full_table_notice_read_only": "FULL_TABLE_NOTICE_NO_BUSINESS_WRITE",
                }.get(execution_scope, "GROUP_QUERY_NO_BUSINESS_WRITE")
                result = ToolResult(name=call.name, called=False, allowed=False, error=error, idempotency_key=idempotency_key)
                self._record(trace_id, "tool_scope_checked", {"tool_name":call.name, "scope":execution_scope,
                    "allowed":False, "rule_id":rule_id}, level="WARN")
                # 作用域也是部署权限的一部分，沿用标准权限拒绝事件，保证完整trace能证明未领取写幂等键。
                self._record(trace_id, "tool_permission_checked", {"tool_name":call.name,
                    "allowed":False, "error":error, "rule_id":rule_id}, level="WARN")
                return self._complete(trace_id, step_index, result, outcome="blocked")
            retry_no_effect = bool(existing is not None and definition.retry_no_effect_failures
                                   and tool_result_is_retryable_no_effect(existing))
            if existing is not None and not retry_no_effect:
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
            permission_error = self._permission_error(definition)
            if permission_error:
                result = ToolResult(name=call.name, called=False, allowed=False, error=permission_error, idempotency_key=idempotency_key)
                self._record(trace_id, "tool_permission_checked", {"tool_name": call.name, "allowed": False, "error": permission_error}, level="WARN")
                return self._complete(trace_id, step_index, result, outcome="blocked", remember_key=idempotency_key)
            self._record(trace_id, "tool_permission_checked", {"tool_name": call.name, "allowed": True})
            validation_details = {}
            try:
                authorization_error = self._authorization_error(
                    call,
                    conversation_id=conversation_id,
                    sender_id=sender_id,
                    message_reference_contract=message_reference_contract,
                )
            except LateParticipationRequiresConfirmation as exc:
                authorization_error = str(exc)
                validation_details = {"validation_error": exc.validation_error()}
            except ParticipationDeadlineRequiresConfirmation as exc:
                authorization_error = str(exc)
                validation_details = _participation_deadline_feedback(exc)
            if authorization_error is None and call.name == "create_game":
                # 只核对尚未执行的新建。已经真实建过的同一操作在前面的幂等分支复用旧收据。
                checked_at = now()
                planned_start_at = planned_game_start_at(
                    call.arguments.get("requirement") or {}, created_at=checked_at,
                )
                if planned_start_at is not None and planned_start_at < checked_at:
                    authorization_error = "拟建的新局开场时间已经过了。"
                    validation_details = {
                        "validation_error": {
                            "code": "create_game_planned_start_passed", "message": authorization_error,
                            "planned_start_at": planned_start_at.isoformat(), "checked_at": checked_at.isoformat(),
                        },
                        "instruction": "这次没有建局。保留顾客的组桌委托和其他已确认条件；若还要新组，请确认接下来能开的时间。",
                    }
            if authorization_error:
                result = ToolResult(
                    name=call.name,
                    called=False,
                    allowed=False,
                    result=validation_details,
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
            # 仅显式支持且Store已证明回滚的工具允许重新领取失败键；成功或未知结果永不重放。
            claim_options = {"retry_failed_no_effect":True} if definition.retry_no_effect_failures else {}
            claimed, claimed_existing = self.store.claim_idempotent_result(idempotency_key, claimed_result, **claim_options)
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
                seat_proof = _expected_participation_terms(message_reference_contract or {})
                if seat_proof is not None and tool_confirms_seat(call):
                    # 审查依据来自后端；跨会话改期可发生在幂等键领取和业务写入之间。
                    result = definition.handler(call, trace_id, conversation_id, sender_id, sender_name,
                                                expected_participation_terms=seat_proof)
                else:
                    result = definition.handler(call, trace_id, conversation_id, sender_id, sender_name)
            except LateParticipationRequiresConfirmation as exc:
                # 预检后可能刚过开场或协商已变更，Store 在写锁内拒绝仍有同样可恢复的反馈。
                self._record(trace_id, "late_participation_blocked_before_write", {
                    "tool_name": call.name, "step_index": step_index, "status": exc.status,
                }, level="WARN")
                result = ToolResult(name=call.name, called=False, allowed=False, error=str(exc),
                                    result={"validation_error": exc.validation_error()})
            except ParticipationTermsChangedError as exc:
                self._record(trace_id, "participation_terms_changed_before_write", {
                    "tool_name": call.name, "step_index": step_index, "idempotency_key": idempotency_key,
                }, level="WARN")
                result = ToolResult(name=call.name, called=False, allowed=False, error=str(exc),
                                    result={"validation_error": {"code": "participation_terms_changed", "message": str(exc)}})
            except ParticipationDeadlineRequiresConfirmation as exc:
                self._record(trace_id, "participation_deadline_blocked_before_write", {
                    "tool_name": call.name, "step_index": step_index, **exc.validation_error(),
                }, level="WARN")
                result = ToolResult(name=call.name, called=False, allowed=False, error=str(exc),
                                    result=_participation_deadline_feedback(exc))
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

    @staticmethod
    def _board_claim_error(call: ToolCall, reference_contract: dict[str, Any] | None) -> str | None:
        """群看板报名只可确认所指牌局；拒绝或协商不受此报名目标约束，仍按现有权限校验。"""
        if "group_board_claim" not in (reference_contract or {}):
            return None
        if not tool_confirms_seat(call):
            return None
        claim = reference_contract["group_board_claim"]
        expected_game_id = claim.get("game_id") if isinstance(claim, dict) else None
        if not expected_game_id:
            return "group board claim requires the original verified board item; clarify the target before reserving a seat"
        if call.arguments.get("game_id") != expected_game_id:
            return f"group board claim can only reserve the referenced game_id={expected_game_id!r}; use the customer's original choice"
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

        if (
            call.name in _PARTICIPATION_TOOLS_REQUIRING_QUOTE_REFERENCE
            and _quote_reference_is_unresolved(message_reference_contract)
        ):
            return (
                "authoritative quoted-message business reference required: "
                f"{call.name} cannot infer a participation state write from an unresolved quote; "
                "resolve the referenced invitation/game with a read tool or ask the user"
            )
        if call.name == "record_registered_member_withdrawal":
            try:
                game = self.store.require_game(str(call.arguments["game_id"]))
                validate_reported_withdrawal(
                    game, customer_id=str(call.arguments["customer_id"]),
                    reported_by=(sender_id, conversation_id), status="declined",
                )
            except ValueError as exc:
                return str(exc)
            return None

        if call.name == "create_game" and call.arguments.get("alternative_to_game_id"):
            try:
                source = self.store.require_game(str(call.arguments["alternative_to_game_id"]))
            except ValueError as exc:
                return str(exc)
            if source.organizer_id != sender_id or source.conversation_id != conversation_id:
                return "alternative source requires its authenticated organizer and originating conversation"
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
                if (call.name == "update_game_requirement" and sender_id != game.organizer_id
                        and {"preferred_gender", "required_gender"} & call.arguments["requirement_patch"].keys()):
                    return "seat gender preferences require the authenticated organizer"
                if game.requirement.get("recruitment_mode") == "board_only" and (
                    call.name == "create_invite_drafts"
                    or call.name in {"join_game", "record_candidate_reply"} and sender_id != game.organizer_id
                ):
                    return "group self recruitment is board-only; the organizer must confirm before inviting or assigning a new seat"
                if call.name in {"join_game", "record_candidate_reply"}:
                    if participation_access_source(self.store, game, conversation_id=conversation_id, sender_id=sender_id):
                        if tool_confirms_seat(call):
                            try:
                                validate_game_accepts_confirmation(game)
                            except ValueError:
                                return None  # 保持过期或终态由原来的事务内状态机处理。
                            validate_late_participation(self.store, game, customer_id=sender_id,
                                                        seat_count=call.arguments.get("seat_count", 1))
                            try:
                                validate_current_participation_terms(
                                    game, _expected_participation_terms(message_reference_contract or {}),
                                )
                            except ParticipationTermsChangedError as exc:
                                return (f"{exc} 按顾客原话说明待确认的条件，"
                                        "或向本人确认是否现在留位。")
                        return None
                    return (
                        "tool participation access required: first search current games in this customer's "
                        "current task or resolve their invitation; knowing a game_id alone grants no write access"
                    )
                if game.conversation_id != conversation_id and call.name == "update_game_requirement":
                    from ...group_chat.stake_confirmation import permits_board_stake_answer

                    if permits_board_stake_answer(self.store, game, conversation_id=conversation_id,
                                                 sender_id=sender_id, patch=call.arguments.get("requirement_patch")):
                        return None
                if game.conversation_id != conversation_id:
                    return (
                        "tool resource mismatch: game management requires its originating conversation; "
                        "participation access does not grant management permission"
                    )
                if call.name == "update_game_status" and game.organizer_id != sender_id:
                    return "whole-game lifecycle management requires the authenticated organizer"
        return None

    def _record(self, trace_id: str, step: str, content: dict[str, Any], *, level: str = "INFO") -> None:
        """记录网关判定证据，使一次工具调用为何执行或被拒绝可以完整回溯。"""
        if self.trace_recorder is not None:
            self.trace_recorder.record(trace_id, step, content, level=level)



def _expected_participation_terms(contract: dict[str, Any]) -> dict[str, Any] | None:
    """携带独立许可及本人时限，网关和 Store 共用同一份写入证据。"""
    terms = None if "unknown_duration_participation" not in contract else {}
    decision = contract.get("unknown_duration_participation")
    if (isinstance(decision, dict) and decision.get("scope") in ("direct_now", "accept_delivered_offer")
            and isinstance(decision.get("table"), dict)):
        terms = decision["table"]
    if deadline := contract.get("customer_departure_deadline"):
        return {"reviewed_table_terms": terms, "customer_departure_deadline": deadline}
    return terms


def _participation_deadline_feedback(exc: ParticipationDeadlineRequiresConfirmation) -> dict[str, Any]:
    """告知本次没有留位，后续核实条件或找合适的桌，避免重复写入。"""
    return {
        "validation_error": exc.validation_error(),
        "instruction": "未占席。按顾客原话保留离店界限，先说明实际约定的散场时间或散场未定。"
                       "需要先离席时确认同桌能否协调，或按该界限找其它桌。不要重复提交同一报名。",
    }


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
