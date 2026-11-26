"""在可信边界内执行模型提出的工具调用，并整理下一轮需要的证据。

"""

from __future__ import annotations

import json
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass, field, replace
from time import monotonic
from typing import Any

from ..group_chat.parsing import explicit_numeric_cap_candidates, explicit_own_party_candidates
from ..group_chat.tool_scope import tool_execution_scope
from ..group_chat.topic_context import verified_board_claim_reference
from ..hooks import HookManager
from ..llm import AgentLLMClient
from ..models import AgentAction, ToolCall, ToolResult, UserMessage
from ..runtime_components import ActionProcessingResult
from ..stores import AgentStore
from ..tool_consistency import (
    EXPLICIT_PARTY_FACT_FIELDS,
    explicit_fact_sources_for_call,
    latest_read_requirement,
    superseded_read_duration_evidence,
    validate_explicit_task_fact_consistency,
    validate_model_create_game_stake,
    validate_table_availability,
    validate_tool_call_consistency,
)
from ..domains.tools import ToolGateway
from ..domains.creation_authorization import check_creation_authorization, creation_clock_conflict
from ..domains.game_participants import validate_game_accepts_confirmation
from ..domains.model_context import has_agreed_game_duration
from ..domains.invitation_policy import tool_confirms_seat
from ..domains.participation_authorization import check_unknown_duration_participation, customer_participation_deadline
from ..domains.table_availability_authorization import check_table_availability, requires_table_clock_review
from ..domains.tools.participation_access import participation_access_source
from ..domains.tools.validation import validate_schema
from .contracts import SingleToolExecution
from .tool_scheduler import ToolCallScheduler
from .future_responsibility import resolve_replaced_follow_up


def input_batch_run_is_stale(store: Any, message: UserMessage) -> bool:
    """返回更新的分片批次是否取代了本次运行。"""

    metadata = message.metadata if isinstance(message.metadata, dict) else {}
    window = metadata.get("input_window") if isinstance(metadata.get("input_window"), dict) else {}
    batch_id = str(window.get("batch_id") or "")
    try:
        batch_version = int(window.get("batch_version"))
    except (TypeError, ValueError):
        return False
    if not batch_id:
        return False
    current = store.pending_input_batch(message.conversation_id, message.sender_id)
    return current is None or current.batch_id != batch_id or current.version != batch_version


@dataclass(slots=True)
class ToolExecutionService:
    """在动作层与可信工具网关之间执行一致性、版本和调度守卫。

    上游：``CustomerVisibleActionService.process_tool_action``。下游：
    ``ToolCallScheduler`` 与 ``ToolGateway``；Scheduler 持有的 ``execute_one`` 回调实际
    指向本类的 :meth:`_execute_one`，这是静态“查找引用”容易漏掉的调用关系。
    """

    store: AgentStore
    tool_gateway: ToolGateway
    trace_recorder: Any
    hook_manager: HookManager | None = None
    max_parallel_read_tools: int = 4
    input_receipt_lock: Any = None
    creation_authorization_client: AgentLLMClient | None = None
    creation_authorization_timeout_seconds: float = 45.0
    table_availability_client: AgentLLMClient | None = None
    table_availability_timeout_seconds: float = 20.0
    participation_authorization_client: AgentLLMClient | None = None
    participation_authorization_timeout_seconds: float = 20.0
    _scheduler: ToolCallScheduler = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """完成数据类初始化后的依赖组装和不变量校验。"""
        self.max_parallel_read_tools = max(1, int(self.max_parallel_read_tools))
        self._scheduler = ToolCallScheduler(
            tool_gateway=self.tool_gateway,
            trace_recorder=self.trace_recorder,
            max_parallel_read_tools=self.max_parallel_read_tools,
        )

    def execute_tool_calls(
        self,
        action: AgentAction,
        *,
        message: UserMessage,
        trace_id: str,
        previous_step_tool_results: list[ToolResult],
        step_index: int,
        run_id: str,
        run_version: int,
        context_payload: dict[str, Any] | None = None,
    ) -> ActionProcessingResult:
        """调度本步工具，并把结果写成 conversation tool turn 供下一模型步读取。"""

        # 语义角色由主模型声明；只绑定同一客户有效原话支持的当前断言，不绑定参加条件。
        context_payload = dict(context_payload or {})
        assertions = action.objective_state.get("fact_assertions") or []
        bound_assertions = []
        for assertion in assertions:
            if not isinstance(assertion, dict) or assertion.get("role") != "current":
                continue
            source_id = str(assertion.get("source_message_id") or "")
            source = self.store.find_conversation_turn(message.conversation_id, message_id=source_id) if source_id else None
            if source is not None and source.sender_id == message.sender_id and not source.metadata.get("context_invalidated"):
                # 保留每条断言及其对象范围；相同字段的多个值不能由dict后写覆盖。
                # scope是模型声明的语义归属，来源鉴权仍由这里和Gateway分别保障。
                bound = deepcopy(assertion)
                if context_payload.get("group_topic_context") and bound.get("field") in EXPLICIT_PARTY_FACT_FIELDS:
                    bound["source_own_party_candidates"] = explicit_own_party_candidates(source.content)
                if bound.get("field") == "cap_score":
                    bound["source_cap_candidates"] = explicit_numeric_cap_candidates(source.content)
                bound_assertions.append(bound)
        context_payload["explicit_task_facts"] = {
            "facts": {}, "bound_assertions": bound_assertions,
            "binding_call_ids": [call.call_id for call in action.tool_calls if call.call_id],
            "binding_fields": list(dict.fromkeys(str(item.get("field")) for item in bound_assertions)),
        }
        context_payload.pop("verified_table_availability", None)
        context_payload.pop("verified_personal_table_availability", None)
        if (self.table_availability_client is not None
                and tool_execution_scope(self.store, message) is None
                and requires_table_clock_review(message)
                and not self.stale_tool_result(call_name="table_availability", message=message,
                                               run_id=run_id, run_version=run_version)):
            self._review_table_availability(action, message, trace_id, step_index, context_payload)
        results_by_index, consistency_blocked, stale_blocked = self._scheduler.execute(
            action,
            execute_one=self._execute_one,
            message=message,
            trace_id=trace_id,
            previous_step_tool_results=previous_step_tool_results,
            step_index=step_index,
            run_id=run_id,
            run_version=run_version,
            context_payload=context_payload,
        )
        tool_results = [results_by_index[index] for index in sorted(results_by_index)]
        pending_tool_results = list(tool_results)
        self.store.append_tool_turn(
            message.conversation_id,
            json.dumps([item.to_dict() for item in pending_tool_results], ensure_ascii=False),
            trace_id,
        )
        if stale_blocked:
            self.trace_recorder.record(
                trace_id,
                "final_output",
                {
                    "reply": "",
                    "reason": "conversation_run_stale",
                    "run_id": run_id,
                    "run_version": run_version,
                    "current_version": self.store.conversation_version(message.conversation_id),
                },
                level="WARN",
            )
            return ActionProcessingResult(
                action=action,
                tool_results=tool_results,
                pending_tool_results=pending_tool_results,
                final_reply="",
                stop_loop=True,
            )
        if consistency_blocked:
            self.trace_recorder.record(
                trace_id,
                "tool_argument_consistency_feedback",
                {"results": [item.to_dict() for item in pending_tool_results]},
                level="WARN",
            )
        return ActionProcessingResult(
            action=action,
            tool_results=tool_results,
            pending_tool_results=pending_tool_results,
        )

    def _review_table_availability(
        self, action: AgentAction, message: UserMessage, trace_id: str, step_index: int, context_payload: dict[str, Any],
    ) -> None:
        """在锁外核验来话，另查复核只给冲突的个人只读查询使用。"""
        relevant_calls = [call for call in action.tool_calls if call.name in {"search_current_games", "create_game"}
                          and (definition := self.tool_gateway.tools.get(call.name)) is not None
                          and validate_schema(call.arguments, definition.schema) is None]
        if not relevant_calls:
            return
        deadline = monotonic() + self.table_availability_timeout_seconds
        verified = check_table_availability(
            self.store, self.table_availability_client, message, trace_id=trace_id, step_index=step_index,
            timeout_seconds=self.table_availability_timeout_seconds, trace_recorder=self.trace_recorder,
        )
        context_payload["verified_table_availability"] = verified
        remaining = deadline - monotonic()
        if (not verified or verified.get("lookup_scope") != "specific_table"
                or verified.get("also_requests_personal_options") is True or remaining <= 0
                or not any(call.name == "search_current_games" and call.arguments.get("lookup_scope") == "personal"
                           for call in relevant_calls)):
            return
        # 复核只读取原话，与首次核验共用时限。原桌和所有写工具继续使用首次证据。
        rechecked = check_table_availability(
            self.store, self.table_availability_client, message, trace_id=trace_id, step_index=step_index,
            timeout_seconds=remaining, trace_recorder=self.trace_recorder, recheck_personal_read=True,
        )
        if rechecked and rechecked.get("also_requests_personal_options") is True:
            context_payload["verified_personal_table_availability"] = rechecked

    def _execute_one(
        self,
        call: ToolCall,
        *,
        call_index: int,
        call_id: str | None,
        observed_results: list[ToolResult],
        message: UserMessage,
        trace_id: str,
        step_index: int,
        run_id: str,
        run_version: int,
        context_payload: dict[str, Any] | None,
    ) -> SingleToolExecution:
        """在一次网关调用前校验明确事实、前序证据和运行版本。

        校验通过后才进入 ``ToolGateway.execute``；执行完成时发出的
        ``after_tool_execute`` Hook 会间接唤起 Runtime 注册的匹配和群看板触发器。
        """

        explicit_fact_error, explicit_fact_reference = validate_explicit_task_fact_consistency(
            call,
            context_payload,
        )
        if explicit_fact_error:
            result = ToolResult(
                name=call.name,
                called=False,
                allowed=False,
                call_id=call_id,
                result={
                    "instruction": (
                        "This tool was NOT executed. Recheck the proposed requirement and the complete current-fact "
                        "reference against their source messages and assertion roles. If the reference itself is invalid, "
                        "reconcile which assertions remain current using the actual user context; do not silently drop "
                        "facts, copy a conflicting requirement, or repeat it unchanged. Submit a valid corrected action "
                        "only after resolving the source/role or argument conflict."
                    ),
                    "call": call.to_dict(),
                    "reference_tool_name": "explicit_task_facts",
                    "reference_requirement": deepcopy(explicit_fact_reference),
                    "reference_fact_sources": explicit_fact_sources_for_call(call, context_payload),
                    "validation_error": {"code": "explicit_task_fact_consistency_rejected", "message": explicit_fact_error},
                },
                error=explicit_fact_error,
            )
            self.trace_recorder.record(
                trace_id,
                "tool_explicit_fact_consistency_error",
                {"call": call.to_dict(), "error": explicit_fact_error, "step_index": step_index},
                level="WARN",
            )
            self.trace_recorder.record(trace_id, "tool_result", result.to_dict(), level="WARN")
            return SingleToolExecution(result=result, blocked_by_consistency=True)

        superseded_duration = superseded_read_duration_evidence(
            call, observed_results, (context_payload or {}).get("verified_table_availability"),
        )
        consistency_error = validate_tool_call_consistency(
            call, observed_results, read_duration_superseded=superseded_duration is not None,
        ) or validate_model_create_game_stake(call)
        if consistency_error:
            result = ToolResult(
                name=call.name,
                called=False,
                allowed=False,
                call_id=call_id,
                result={
                    "instruction": (
                        "Preserve explicit requirement fields from previous read-only tool results unless the user "
                        "has clearly changed them. When a required customer fact is still unknown, keep the current "
                        "goal and ask for that fact before proposing another write. Otherwise correct the arguments "
                        "and call the tool again."
                    ),
                    "call": call.to_dict(),
                    "reference_tool_name": "search_current_games",
                    "reference_requirement": latest_read_requirement(
                        observed_results, tool_name="search_current_games"
                    ) or {},
                    "validation_error": {"code": "tool_requirement_consistency_rejected", "message": consistency_error},
                },
                error=consistency_error,
            )
            self.trace_recorder.record(
                trace_id,
                "tool_argument_consistency_error",
                {"call": call.to_dict(), "error": consistency_error, "step_index": step_index},
                level="WARN",
            )
            self.trace_recorder.record(trace_id, "tool_result", result.to_dict(), level="WARN")
            return SingleToolExecution(result=result, blocked_by_consistency=True)

        if call.name == "create_game" and (clock_conflict := creation_clock_conflict(self.store, call, message)):
            error = "拟开场把顾客唯一的未明确时段钟点翻到了相反半天，缺少明确的开场依据。"
            result = ToolResult(name=call.name, called=False, allowed=False, call_id=call_id, error=error, result={
                "validation_error": {"code": "create_game_half_day_time_conflict", "message": error},
                "reference_clock_observation": clock_conflict,
                "instruction": "此建局未执行。保留原来明确的组桌委托；区分到店和开场。若开场未定，请只问顾客想几点开，不要默认任一半天或重提同一时间。",
            })
            self.trace_recorder.record(trace_id, "tool_argument_consistency_error",
                {"call": call.to_dict(), "error": error, "step_index": step_index, **clock_conflict}, level="WARN")
            self.trace_recorder.record(trace_id, "tool_result", result.to_dict(), level="WARN")
            return SingleToolExecution(result=result, blocked_by_consistency=True)

        stale_result = self.stale_tool_result(
            call_name=call.name,
            message=message,
            run_id=run_id,
            run_version=run_version,
        )
        if stale_result is not None:
            stale_result.call_id = call_id
            self.trace_recorder.record(trace_id, "conversation_run_stale", stale_result.to_dict(), level="WARN")
            self.trace_recorder.record(trace_id, "tool_result", stale_result.to_dict(), level="WARN")
            return SingleToolExecution(result=stale_result, blocked_by_stale_run=True)

        proposed_requirement = call.arguments.get("requirement")
        has_query_end_deadline = isinstance(proposed_requirement, dict) and "must_end_by" in proposed_requirement
        if "verified_table_availability" in (context_payload or {}) or has_query_end_deadline:
            # 校验本轮来话支持的时间用途。拒绝条件只回灌，不擅自改写模型查询或替它选桌。
            lookup_scope = call.arguments.get("lookup_scope")
            rechecked = ((context_payload or {}).get("verified_personal_table_availability")
                         if call.name == "search_current_games" and lookup_scope == "personal" else None)
            verified = rechecked or (context_payload or {}).get("verified_table_availability")
            error = validate_table_availability(call, verified)
            if (not error and rechecked and isinstance(proposed_requirement, dict)
                    and any(key in proposed_requirement for key in ("start_time_range", "planned_start_at", "start_time"))
                    and not (rechecked.get("opening_window") or rechecked.get("arrival_lookup_floor_at"))):
                error = "原话支持另查，但拟查的开场时间没有核实依据。保留另查目的，按已核实条件查询或澄清开场范围。"
            if error:
                specific = (lookup_scope == "specific_table" or (lookup_scope is None and verified
                            and verified.get("lookup_scope") == "specific_table"
                            and not verified.get("also_requests_personal_options")))
                instruction = (
                    "本次工具未执行。先按原桌已有条件查当前状态与余位，再说顾客何时到是否需要另行确认。"
                    "人齐开不代表已经约定开场或能等，查到空位也不代表已有报名授权。"
                    if call.name == "search_current_games" and specific
                    else "本次工具未执行。用核实过的档位、时段和最长可打多久查现成桌，保留顾客原先的委托。"
                         "到店估计之前的临近备选只供顾客确认，不能答应一定赶得上或同桌愿意等。"
                         "有可选桌就先反馈。需要建新桌但尚未确认固定开场时只问开场时间。"
                )
                result = ToolResult(name=call.name, called=False, allowed=False, call_id=call_id, error=error, result={
                    "validation_error": {"code": "verified_table_availability_conflict", "message": error},
                    "table_availability_evidence": verified,
                    **({"lookup_scope": lookup_scope} if lookup_scope in ("specific_table", "personal") else {}),
                    "instruction": instruction,
                })
                self.trace_recorder.record(trace_id, "tool_table_availability_rejected",
                    {"call": call.to_dict(), "error": error, "step_index": step_index}, level="WARN")
                self.trace_recorder.record(trace_id, "tool_result", result.to_dict(), level="WARN")
                return SingleToolExecution(result=result, blocked_by_consistency=True)

        definition = self.tool_gateway.tools.get(call.name)
        creation_scope = None
        if (call.name == "create_game" and self.creation_authorization_client is not None
                and tool_execution_scope(self.tool_gateway.store, message) is None and definition is not None
                and validate_schema(call.arguments, definition.schema) is None):
            # 审核可能等待模型，必须在输入栅栏之外执行。写入前重新核对运行版本。
            creation_scope = check_creation_authorization(
                self.store, self.creation_authorization_client, call, message, trace_id=trace_id,
                step_index=step_index * 100 + call_index, timeout_seconds=self.creation_authorization_timeout_seconds,
                trace_recorder=self.trace_recorder, observed_tool_results=observed_results,
            )
        participation_scope = departure_deadline = None
        if (tool_confirms_seat(call)
                and tool_execution_scope(self.tool_gateway.store, message) is None and definition is not None
                and validate_schema(call.arguments, definition.schema) is None):
            participation_scope, departure_deadline = self._review_participation(
                call, message, trace_id=trace_id, step_index=step_index * 100 + call_index,
            )
        self._emit(
            "before_tool_execute",
            trace_id=trace_id,
            payload={"call": call.to_dict(), "step_index": step_index, "call_index": call_index},
        )
        writes_state = definition is not None and definition.execution_mode in {"state_write", "draft_write"}
        # 本地写入与输入接收共享短提交边界；模型、审查和只读网络调用均不持此锁。
        fence = self.input_receipt_lock(message.conversation_id) if writes_state and self.input_receipt_lock else nullcontext()
        with fence:
            stale_result = self.stale_tool_result(call_name=call.name, message=message, run_id=run_id, run_version=run_version)
            if stale_result is not None:
                stale_result.call_id = call_id
                return SingleToolExecution(result=stale_result, blocked_by_stale_run=True)
            reference_contract = dict((context_payload or {}).get("message_reference_contract") or {})
            reference_contract.pop("game_creation_authorization", None)
            if creation_scope is not None:
                reference_contract["game_creation_authorization"] = creation_scope
            reference_contract.pop("unknown_duration_participation", None)
            if participation_scope is not None:
                reference_contract["unknown_duration_participation"] = participation_scope
            reference_contract.pop("customer_departure_deadline", None)
            if departure_deadline is not None:
                reference_contract["customer_departure_deadline"] = departure_deadline
            # 报名目标只从本条输入的持久群路由事实恢复，不接受模型上下文或消息 metadata 提议的目标。
            reference_contract.pop("group_board_claim", None)
            if call.name in {"join_game", "record_candidate_reply"}:
                board_claim = verified_board_claim_reference(self.tool_gateway.store, message)
                if board_claim is not None:
                    reference_contract["group_board_claim"] = board_claim
            self.trace_recorder.record(trace_id, "tool_called", {"call": call.to_dict(), "step_index": step_index, "call_index": call_index})
            result = self.tool_gateway.execute(
                call, trace_id=trace_id, conversation_id=message.conversation_id,
                sender_id=message.sender_id, sender_name=message.sender_name,
                step_index=step_index * 100 + call_index, source_message_id=message.message_id,
                message_reference_contract=reference_contract,
                execution_scope=tool_execution_scope(self.tool_gateway.store, message),
            )
        result.call_id = call_id
        verified_schedule = (context_payload or {}).get("verified_table_availability") or {}
        if (call.name == "create_game" and result.called and result.allowed and not result.error
                and verified_schedule.get("prior_play_until")):
            # 回执附同一目标的本人柔性时点。脱离网关缓存副本，避免变成以后其它来话的桌况。
            result = replace(result, result={**result.result, "table_availability_evidence": {
                "play_until_at": verified_schedule["play_until_at"],
                "prior_play_until": verified_schedule["prior_play_until"],
            }})
        if superseded_duration is not None and result.called and result.allowed and not result.error:
            self.trace_recorder.record(trace_id, "tool_read_duration_superseded", {
                **superseded_duration,
                "source_message_id": message.message_id,
                "run_version": run_version,
                "step_index": step_index,
                "create_call_id": call_id,
            })
        resolve_replaced_follow_up(self.store, result, observed_results, message, trace_id)
        self.trace_recorder.record(trace_id, "tool_result", result.to_dict())
        self._emit(
            "after_tool_execute",
            trace_id=trace_id,
            payload={
                "call": call.to_dict(),
                "result": result.to_dict(),
                "conversation_id": message.conversation_id,
                "sender_id": message.sender_id,
                "source_message_id": message.message_id,
                "step_index": step_index,
                "call_index": call_index,
            },
        )
        for transition in result.state_transitions:
            event = "state_transition_replayed" if result.deduplicated else "state_transition"
            self.trace_recorder.record(trace_id, event, transition.to_dict())
        return SingleToolExecution(result=result)

    def _review_participation(self, call: ToolCall, message: UserMessage, *, trace_id: str,
                              step_index: int) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """在锁外绑定本人时限及留位授权，写入事务按当前桌况重查。"""
        unavailable = {"scope": "unconfirmed"} if self.participation_authorization_client is not None else None
        if call.arguments.get("customer_id") != message.sender_id:
            return unavailable, None
        try:
            game = self.store.require_game(call.arguments["game_id"])
            validate_game_accepts_confirmation(game)
        except ValueError:
            return unavailable, None
        if not participation_access_source(
            self.store, game, conversation_id=message.conversation_id, sender_id=message.sender_id,
        ):
            return unavailable, None
        deadline = customer_participation_deadline(self.store, message, game)
        if has_agreed_game_duration(game.requirement) or self.participation_authorization_client is None:
            return unavailable, deadline
        scope = check_unknown_duration_participation(
            self.store, self.participation_authorization_client, message, game, trace_id=trace_id,
            step_index=step_index, timeout_seconds=self.participation_authorization_timeout_seconds,
            trace_recorder=self.trace_recorder,
        )
        return scope, deadline

    def stale_tool_result(
        self,
        *,
        call_name: str,
        message: UserMessage,
        run_id: str,
        run_version: int,
    ) -> ToolResult | None:
        """拒绝旧版本尚未执行的工具提案；已提交的工具事实不回滚。"""
        current_version = self.store.conversation_version(message.conversation_id)
        stale_input_batch = input_batch_run_is_stale(self.store, message)
        if current_version == int(run_version) and not stale_input_batch:
            return None
        return ToolResult(
            name=call_name,
            called=False,
            allowed=False,
            result={
                "run_id": run_id,
                "run_version": run_version,
                "current_version": current_version,
                "stale_input_batch": stale_input_batch,
                "instruction": (
                    "This run is stale because a newer user fragment or conversation turn arrived. "
                    "Do not write state or create drafts from the old version; rebuild context from the latest user input."
                ),
            },
            error="stale run: input batch or conversation version changed before the tool could execute",
        )

    def _emit(self, event_name: str, *, trace_id: str, payload: dict[str, Any]) -> None:
        if self.hook_manager is not None:
            self.hook_manager.emit(event_name, trace_id=trace_id, payload=payload)


__all__ = ["ToolExecutionService", "input_batch_run_is_stale"]
