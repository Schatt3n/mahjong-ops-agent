"""统一处理客户可见文本的生成、审查、持久化和最终回复。

"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from ..customer_visible_contract import (
    CUSTOMER_EXECUTION_FAILURE_REPLY, customer_visible_duration_fit_violations, customer_visible_seat_claim_violations,
    customer_visible_table_start_violations,
)

from ..copywriting import action_with_customer_visible_rewrites
from ..domains.delivered_game_offer import (
    DELIVERED_GAME_OFFER, game_offer_for_review, seat_facts_for_private_reply, table_duration_facts_for_private_reply,
)
from ..hooks import HookManager
from ..models import AgentAction, ToolResult, UserMessage
from ..runtime_components import ActionProcessingResult, TurnBudgets
from ..stores import AgentStore
from ..visibility import (
    CustomerVisibleProcessor,
    customer_visible_content_review_approved,
    customer_visible_items_for_action,
)
from ..customer_visible_review import CUSTOMER_VISIBLE_CONTENT_REVIEW_TOOL_NAME, external_action_evidence_from_tool_results
from .tool_service import ToolExecutionService, input_batch_run_is_stale


_DRAFT_REVIEW_FIELDS = {"content_review_approved", "content_review_trace_id"}


def _with_draft_review_proof(action: AgentAction, *, approved_trace_id: str | None = None) -> AgentAction:
    """清理草稿中的保留审核字段，只接纳当前后端独立审核的成功凭证。"""

    payload = action.to_dict()
    changed = False
    for call in payload.get("tool_calls") or []:
        if not isinstance(call, dict) or not isinstance(call.get("arguments"), dict):
            continue
        item_key = {
            "create_invite_drafts": "invitations",
            "create_outbound_message_drafts": "drafts",
        }.get(str(call.get("name") or ""))
        items = call["arguments"].get(item_key) if item_key else None
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            raw_metadata = item.get("metadata")
            metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
            if approved_trace_id is None and _DRAFT_REVIEW_FIELDS.isdisjoint(metadata):
                continue
            metadata = {key: value for key, value in metadata.items() if key not in _DRAFT_REVIEW_FIELDS}
            if approved_trace_id is not None:
                metadata.update({"content_review_approved": True, "content_review_trace_id": approved_trace_id})
            item["metadata"] = metadata
            changed = True
    return AgentAction.from_payload(payload) if changed else action


def attach_content_review_proof(action: AgentAction, *, trace_id: str) -> AgentAction:
    """为已批准的外发草稿加盖后端持有的审核证据。"""

    return _with_draft_review_proof(action, approved_trace_id=trace_id)


def customer_visible_rewrites(result: ToolResult) -> dict[str, str]:
    """从生成工具结果中读取条目标识到最终文本的重写映射。"""

    return {
        str(item.get("item_id") or ""): str(item.get("final_text") or "")
        for item in result.result.get("item_rewrites", [])
        if isinstance(item, dict)
    }


@dataclass(slots=True)
class CustomerVisibleActionService:
    """统一收口可能让客户看到的工具草稿和最终回复。

    上游：``ActionProcessor.process_action``。这里负责文本生成/审查和持久化边界；
    真正执行工具委托 ``ToolExecutionService``，真正通道发送仍由应用层完成。
    """

    store: AgentStore
    trace_recorder: Any
    tool_execution_service: ToolExecutionService
    hook_manager: HookManager | None = None
    attention_handler: Callable[[UserMessage, str, str, int, str, bool], None] | None = None

    def process_tool_action(
        self,
        action: AgentAction,
        *,
        processor: CustomerVisibleProcessor,
        message: UserMessage,
        trace_id: str,
        context_payload: dict[str, Any],
        previous_pending_tool_results: list[ToolResult],
        step_index: int,
        budgets: TurnBudgets,
        run_id: str,
        run_version: int,
    ) -> ActionProcessingResult:
        """仅对邀请/外发草稿保留独立生成与审查，再为获批动作加凭证并执行工具。

        审查失败时只把结构化反馈回灌给下一模型步，不会执行包含外发草稿的工具。
        """

        # 即使当前关闭独立审核，模型提出的草稿也不能自行取得人工外发所依赖的审核凭证。
        action = _with_draft_review_proof(action)
        collected_results: list[ToolResult] = []
        stale = self._stop_superseded(action, message=message, trace_id=trace_id, run_id=run_id, run_version=run_version)
        if stale is not None:
            return stale
        review_items = customer_visible_items_for_action(action)
        original_text = {
            str(item.get("item_id") or ""): str(item.get("text") or "") for item in review_items
        }
        generation_result = processor.run_text_generation(
            message=message,
            trace_id=trace_id,
            action=action,
            items=review_items,
            context_payload=context_payload,
            turn_budget=budgets.text_generation,
            generation_scope="tool_calls",
        )
        if generation_result is not None:
            collected_results.append(generation_result)
            action = self.apply_customer_visible_rewrites(action, generation_result, trace_id=trace_id)
            review_items = customer_visible_items_for_action(action)
        review_items = [
            {
                **item,
                "source_text": original_text.get(
                    str(item.get("item_id") or ""),
                    str(item.get("text") or ""),
                ),
            }
            for item in review_items
        ]

        stale = self._stop_superseded(action, message=message, trace_id=trace_id, run_id=run_id, run_version=run_version,
                                      tool_results=collected_results)
        if stale is not None:
            return stale
        review_result = processor.run_content_review(
            message=message,
            trace_id=trace_id,
            action=action,
            review_items=review_items,
            context_payload=context_payload,
            turn_budget=budgets.review,
            review_scope="tool_calls",
            on_failure=lambda reason, retrying: self._notify_review_failure(
                message, trace_id, run_id, run_version, reason, retrying),
            may_retry=lambda: not self.run_is_stale(message, run_version),
        )
        if review_result is not None:
            collected_results.append(review_result)
            self.trace_recorder.record(trace_id, "tool_result", review_result.to_dict())
            self.store.append_tool_turn(
                message.conversation_id,
                json.dumps([review_result.to_dict()], ensure_ascii=False),
                trace_id,
            )
            stale = self._stop_superseded(action, message=message, trace_id=trace_id, run_id=run_id, run_version=run_version,
                                          tool_results=collected_results)
            if stale is not None:
                return stale
            if review_result.result.get("transport_timeout_exhausted"):
                return self._stop_after_review_timeout(action, message=message, trace_id=trace_id,
                    run_id=run_id, run_version=run_version, tool_results=collected_results)
            if not customer_visible_content_review_approved(review_result):
                return ActionProcessingResult(
                    action=action,
                    tool_results=collected_results,
                    pending_tool_results=[review_result],
                    continue_loop=True,
                )
            action = attach_content_review_proof(action, trace_id=trace_id)

        execution = self.tool_execution_service.execute_tool_calls(
            action,
            message=message,
            trace_id=trace_id,
            previous_step_tool_results=list(previous_pending_tool_results),
            step_index=step_index,
            run_id=run_id,
            run_version=run_version,
            context_payload=context_payload,
        )
        collected_results.extend(execution.tool_results)
        if execution.stop_loop:
            return ActionProcessingResult(
                action=action,
                tool_results=collected_results,
                pending_tool_results=execution.pending_tool_results,
                final_reply=execution.final_reply,
                stop_loop=True,
            )
        return ActionProcessingResult(
            action=action,
            tool_results=collected_results,
            pending_tool_results=execution.pending_tool_results,
            continue_loop=True,
        )

    def process_reply_action(
        self,
        action: AgentAction,
        *,
        processor: CustomerVisibleProcessor,
        message: UserMessage,
        trace_id: str,
        context_payload: dict[str, Any],
        budgets: TurnBudgets,
        run_id: str,
        run_version: int,
    ) -> ActionProcessingResult:
        """把主Agent最终原稿直接交独立审查并持久化，不再由第二模型改写业务沟通行为。

        普通回复写为待运营发送的 assistant turn；``internal_event`` 只写内部记录，
        新版本已到达时则拦截当前陈旧运行的输出。审查拒绝仅回灌主Agent重新生成，
        不采纳审查器的建议文本作为客户回复；通道发送仍由应用层负责。
        """

        collected_results: list[ToolResult] = []
        stale = self._stop_superseded(action, message=message, trace_id=trace_id, run_id=run_id, run_version=run_version)
        if stale is not None:
            return stale
        proposed_reply = action.reply_to_user.strip()
        if action.needs_human and not proposed_reply:
            proposed_reply = CUSTOMER_EXECUTION_FAILURE_REPLY

        if bool(message.metadata.get("internal_event")):
            return self._complete_internal_event(
                action,
                message=message,
                trace_id=trace_id,
                proposed_reply=proposed_reply,
                run_id=run_id,
                run_version=run_version,
            )

        # 静默是明确的终止决策，不生成空的发送草稿，也不消耗话术/审查调用。
        if action.response_mode == "silent":
            self.trace_recorder.record(trace_id, "final_output", {
                "reply": "", "response_mode": "silent", "silence_reason": action.silence_reason,
                "objective_status": action.objective_status,
            })
            self.store.append_assistant_turn(message.conversation_id, "", trace_id,
                                             metadata={"delivery_status": "intentionally_silent",
                                                       "silence_reason": action.silence_reason})
            return ActionProcessingResult(action=action, final_reply="", stop_loop=True)

        tool_evidence = list(context_payload.get("turn_tool_evidence") or context_payload.get("previous_tool_results") or [])
        offer = game_offer_for_review(self.store, message, action, tool_evidence)
        seat_facts = seat_facts_for_private_reply(self.store, message, tool_evidence)
        table_duration = table_duration_facts_for_private_reply(self.store, seat_facts)
        review_item = {
            "item_id": "reply_to_user",
            "source": "reply_to_user",
            "recipient_id": message.sender_id,
            "recipient_name": message.sender_name,
            "text": proposed_reply,
            "source_text": proposed_reply,
            "action_evidence": external_action_evidence_from_tool_results(tool_evidence),
            **({"game_offer_candidate": offer} if offer else {}),
            **({"seat_claim_evidence": seat_facts} if seat_facts else {}),
            **({"table_duration_fit_evidence": table_duration} if table_duration else {}),
        }
        stale = self._stop_superseded(action, message=message, trace_id=trace_id, run_id=run_id, run_version=run_version,
                                      tool_results=collected_results)
        if stale is not None:
            return stale
        review_result = processor.run_content_review(
            message=message,
            trace_id=trace_id,
            action=action,
            review_items=[review_item],
            context_payload=context_payload,
            turn_budget=budgets.review,
            review_scope="reply_to_user",
            on_failure=lambda reason, retrying: self._notify_review_failure(
                message, trace_id, run_id, run_version, reason, retrying),
            may_retry=lambda: not self.run_is_stale(message, run_version),
        )
        if review_result is not None:
            collected_results.append(review_result)
            self.trace_recorder.record(trace_id, "tool_result", review_result.to_dict())
            self.store.append_tool_turn(
                message.conversation_id,
                json.dumps([review_result.to_dict()], ensure_ascii=False),
                trace_id,
            )
            stale = self._stop_superseded(action, message=message, trace_id=trace_id, run_id=run_id, run_version=run_version,
                                          tool_results=collected_results)
            if stale is not None:
                return stale
            if review_result.result.get("transport_timeout_exhausted"):
                return self._stop_after_review_timeout(action, message=message, trace_id=trace_id,
                    run_id=run_id, run_version=run_version, tool_results=collected_results)
            if not customer_visible_content_review_approved(review_result):
                return ActionProcessingResult(
                    action=action,
                    tool_results=collected_results,
                    pending_tool_results=[review_result],
                    continue_loop=True,
                )

        if self.run_is_stale(message, run_version):
            self._trace_stale_reply(message, trace_id=trace_id, run_id=run_id, run_version=run_version)
            return ActionProcessingResult(
                action=action,
                tool_results=collected_results,
                final_reply="",
                stop_loop=True,
            )

        # 审查可能关闭，也可能在另一个会话补位或开打后才返回。发送前复核与本轮查询绑定的局况。
        fresh_seat_facts = seat_facts_for_private_reply(self.store, message, tool_evidence)
        fresh_duration = table_duration_facts_for_private_reply(self.store, fresh_seat_facts)
        seat_violations = customer_visible_seat_claim_violations(proposed_reply, fresh_seat_facts)
        start_violations = customer_visible_table_start_violations(proposed_reply, fresh_seat_facts)
        duration_violations = customer_visible_duration_fit_violations(proposed_reply, fresh_duration)
        if violations := seat_violations + start_violations + duration_violations:
            rejection = ToolResult(name=CUSTOMER_VISIBLE_CONTENT_REVIEW_TOOL_NAME, called=True, allowed=True, result={
                "approved": False, "review_scope": "reply_to_user",
                "violations": [f"customer_visible_contract:{rule}" for rule in violations],
                "seat_claim_evidence": fresh_seat_facts,
                **({"table_duration_fit_evidence": fresh_duration} if duration_violations else {}),
                "instruction": (
                    ("按本轮真实核对的同桌余位重写回复，不能重复报名或用旧话术推断余位。" if seat_violations else "")
                    + ("这桌仍在凑人，还未实际开打。按当前局况重写，不把原定开场当成已开打。" if start_violations else "")
                    + ("这桌没约好打多久。可以承接顾客说的到店和可打时间，但不能保证这些时间与本桌相容。"
                       if duration_violations else "")
                ),
            })
            self.trace_recorder.record(trace_id, "tool_result", rejection.to_dict())
            self.store.append_tool_turn(message.conversation_id, json.dumps([rejection.to_dict()], ensure_ascii=False), trace_id)
            return ActionProcessingResult(action=action, tool_results=[*collected_results, rejection],
                                          pending_tool_results=[rejection], continue_loop=True)

        # 独立审查必须绑定整条原样放行的可见回复。这里只记录待发送牌局引用，渠道回执前不可读作已送达。
        offer_receipt = None
        if offer and review_result is not None:
            reviews = review_result.result.get("item_reviews") or []
            if len(reviews) == 1:
                if action.objective_status == "waiting_user" and reviews[0].get("reviewed_game_offer_id") == offer["game_id"]:
                    offer_receipt = {**offer, "recipient_id": message.sender_id}
                elif reviews[0].get("reviewed_game_reference_id") == offer["game_id"]:
                    offer_receipt = {**offer, "recipient_id": message.sender_id, "game_reference_only": True}
        self.append_pending_assistant_turn(
            message.conversation_id,
            proposed_reply,
            trace_id,
            run_id=run_id,
            run_version=run_version,
            source_message=message,
            game_offer=offer_receipt,
        )
        self.trace_recorder.record(
            trace_id,
            "final_output",
            {"reply": proposed_reply, "objective_status": action.objective_status},
        )
        self._emit(
            "before_reply_send",
            trace_id=trace_id,
            payload={"reply": proposed_reply, "action": action.to_dict()},
        )
        return ActionProcessingResult(
            action=action,
            tool_results=collected_results,
            final_reply=proposed_reply,
            stop_loop=True,
        )

    def apply_customer_visible_rewrites(
        self, action: AgentAction, result: ToolResult, *, trace_id: str
    ) -> AgentAction:
        """应用客户客户可见rewrites规则，并保持领域状态合法。"""
        rewrites = customer_visible_rewrites(result)
        if not rewrites:
            return action
        rewritten = action_with_customer_visible_rewrites(action, rewrites)
        self.trace_recorder.record(
            trace_id, "action_after_customer_visible_text_generation", rewritten.to_dict()
        )
        return rewritten

    def _stop_after_review_timeout(self, action: AgentAction, *, message: UserMessage, trace_id: str,
                                   run_id: str, run_version: int, tool_results: list[ToolResult]) -> ActionProcessingResult:
        """一次原样复审仍拿不到结果时保留店家待办，终止且不执行原草稿。"""
        self.append_pending_assistant_turn(message.conversation_id, CUSTOMER_EXECUTION_FAILURE_REPLY,
            trace_id, run_id=run_id, run_version=run_version, source_message=message)
        self.trace_recorder.record(trace_id, "final_output", {
            "reply": CUSTOMER_EXECUTION_FAILURE_REPLY, "reason": "reply_review_timeout_exhausted",
        }, level="WARN")
        return ActionProcessingResult(action=action, tool_results=tool_results,
            final_reply=CUSTOMER_EXECUTION_FAILURE_REPLY, stop_loop=True, runtime_status="needs_human")

    def _notify_review_failure(self, message: UserMessage, trace_id: str, run_id: str,
                               run_version: int, reason: str, retrying: bool) -> None:
        """审查故障和店家待办使用同一运行身份，重试仍须经过完整审查。"""
        if self.attention_handler is not None:
            self.attention_handler(message, trace_id, run_id, run_version, reason, retrying)

    def _stop_superseded(self, action: AgentAction, *, message: UserMessage, trace_id: str,
                         run_id: str, run_version: int, tool_results: list[ToolResult] | None = None) -> ActionProcessingResult | None:
        """生成/审查边界让旧输入静默停止；不会把旧审查反馈交回模型重试。"""
        if not self.run_is_stale(message, run_version):
            return None
        self._trace_stale_reply(message, trace_id=trace_id, run_id=run_id, run_version=run_version)
        return ActionProcessingResult(action=action, tool_results=list(tool_results or []), final_reply="", stop_loop=True)

    def run_is_stale(self, message: UserMessage, run_version: int) -> bool:
        return (
            self.store.conversation_version(message.conversation_id) != int(run_version)
            or input_batch_run_is_stale(self.store, message)
        )

    def append_pending_assistant_turn(
        self,
        conversation_id: str,
        text: str,
        trace_id: str,
        *,
        run_id: str,
        run_version: int,
        source_message: UserMessage,
        game_offer: dict[str, Any] | None = None,
    ) -> None:
        """在当前输入版本下记录待发回复及可选的已审牌局引用；实际投递仍由渠道确认。"""
        from contextlib import nullcontext
        fence = self.tool_execution_service.input_receipt_lock
        with fence(conversation_id) if fence else nullcontext():
            if self.run_is_stale(source_message, run_version):
                return
            if text == CUSTOMER_EXECUTION_FAILURE_REPLY:
                state = self.store.agent_run(run_id)
                if state is not None and state.handoff_wait_delivered:
                    return
            metadata = {"delivery_status": "pending_operator_send", "run_id": run_id, "conversation_version": run_version}
            if game_offer is not None:
                metadata[DELIVERED_GAME_OFFER] = game_offer
            window = source_message.metadata.get("input_window")
            if isinstance(window, dict) and window.get("batch_id"):
                if type(window.get("batch_version")) is not int:
                    return
                metadata["input_window"] = {"batch_id": window["batch_id"], "batch_version": window["batch_version"],
                                            "sender_id": source_message.sender_id}
            self.store.append_assistant_turn(conversation_id, text, trace_id, metadata=metadata)

    def _complete_internal_event(
        self,
        action: AgentAction,
        *,
        message: UserMessage,
        trace_id: str,
        proposed_reply: str,
        run_id: str,
        run_version: int,
    ) -> ActionProcessingResult:
        summary = proposed_reply or "内部定时任务已完成。"
        self.store.append_assistant_turn(
            message.conversation_id,
            summary,
            trace_id,
            metadata={
                "internal_event": True,
                "delivery_mode": "internal_only",
                "run_id": run_id,
                "run_version": run_version,
            },
        )
        self.trace_recorder.record(
            trace_id,
            "final_output",
            {
                "reply": summary,
                "delivery_mode": "internal_only",
                "run_id": run_id,
                "run_version": run_version,
            },
        )
        return ActionProcessingResult(action=action, final_reply=summary, stop_loop=True)

    def _trace_stale_reply(
        self, message: UserMessage, *, trace_id: str, run_id: str, run_version: int
    ) -> None:
        payload = {
            "run_id": run_id,
            "run_version": run_version,
            "current_version": self.store.conversation_version(message.conversation_id),
        }
        self.trace_recorder.record(
            trace_id,
            "conversation_run_stale",
            {**payload, "blocked": "final_reply"},
            level="WARN",
        )
        self.trace_recorder.record(
            trace_id,
            "final_output",
            {**payload, "reply": "", "reason": "conversation_run_stale"},
            level="WARN",
        )

    def _emit(self, event_name: str, *, trace_id: str, payload: dict[str, Any]) -> None:
        if self.hook_manager is not None:
            self.hook_manager.emit(event_name, trace_id=trace_id, payload=payload)


__all__ = [
    "CustomerVisibleActionService",
    "attach_content_review_proof",
    "customer_visible_rewrites",
]
