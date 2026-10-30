"""持久化复查到期时读取最新目标，再交回主 Agent 决策；不预存催促话术。"""

from __future__ import annotations

from datetime import datetime
from .models import SystemTriggerMessage, now


def handle_follow_up_task(runtime, task, trace_id: str, *, at=None):
    """验证目标、取消状态和期限后唤醒；复用系统触发、审查与幂等机制。"""
    stamp = at or now()
    trigger = SystemTriggerMessage(trigger_id=task.idempotency_key, trigger_type="conversation_follow_up_due",
                                   conversation_id=task.conversation_id, sender_id=task.subject_id, sender_name=task.subject_name,
                                   payload={**task.payload, "task_id": task.task_id, "evaluated_at": stamp.isoformat(),
                                            "remaining_seconds": max(0, int((datetime.fromisoformat(task.payload["expires_at"])-stamp).total_seconds())),
                                            "response_contract": "先核对用户目标和最新事实，再决定回复、静默或有界复查。原始截止时间保持不变；不要把原来的相对期限从本轮重新计时，不必重复用户已经知道的条件。"}, created_at=stamp)
    result = runtime.handle_system_trigger(trigger, trace_id=trace_id)
    # 新输入仅取短接收锁，不再等待长执行锁；必须把版本核验、去重与草稿写入
    # 放在同一接收fence内，避免新接收已清理旧输出后又冒出待发的旧复查草稿。
    # 锁顺序与普通可见输出一致：会话执行锁 -> 短接收锁 -> Store事务，无模型或网络调用。
    with runtime.conversation_lock(task.conversation_id), runtime.input_receipt_lock(task.conversation_id):
        reply_turn = next((turn for turn in reversed(runtime.store.recent_turns(task.conversation_id))
                           if turn.trace_id == result.trace_id and turn.role.value == "assistant"), None)
        if result.final_reply and (reply_turn is None or reply_turn.metadata.get("conversation_version") != runtime.store.conversation_version(task.conversation_id)):
            runtime.trace_recorder.record(trace_id, "follow_up_skipped", {"task_id": task.task_id, "reason": "new_user_message_superseded_reply"})
            return result
        if result.final_reply and not any(draft.metadata.get("system_trigger_id") == task.idempotency_key for draft in runtime.store.outbound_message_drafts.values()):
            drafts, transitions = runtime.store.create_outbound_message_drafts(
                conversation_id=task.conversation_id, drafts=[{
                    "recipient_id": task.subject_id, "recipient_name": task.subject_name,
                    "channel": "console", "message_text": result.final_reply, "purpose": "follow_up",
                    "metadata": {"system_trigger_id": trigger.trigger_id, "customer_visible_processing": "normal_agent_loop"},
                }], trace_id=trace_id)
            result.state_transitions.extend(transitions)
    runtime.trace_recorder.record(trace_id, "follow_up_decision", {"task_id": task.task_id, "result": result.to_dict()})
    return result


def follow_up_trigger_is_current(runtime, trigger) -> bool:
    """在 Runtime 已取得会话锁后检查租约和目标；避免重复获取非重入会话锁。"""
    task = runtime.store.scheduled_tasks.get(trigger.payload["task_id"])
    context = runtime.store.current_task_context(trigger.conversation_id, trigger.sender_id)
    expected = trigger.payload.get("task_context_id")
    return bool(task and task.status.value == "processing"
                and (not expected or (context and context.task_context_id == expected))
                and int(task.payload.get("review_count", 0)) <= int(task.payload.get("review_budget", 0))
                and trigger.created_at <= datetime.fromisoformat(task.payload["expires_at"]))
