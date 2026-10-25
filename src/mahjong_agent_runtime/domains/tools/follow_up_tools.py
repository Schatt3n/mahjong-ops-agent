"""将模型提出的等待策略保存为有界复查任务；到期动作仍由主 Agent 判断。"""

from __future__ import annotations

from datetime import datetime, timedelta
from ...models import ScheduledAgentTask, ScheduledTaskStatus, ToolResult, new_id, now


def follow_up_tasks(store, conversation_id: str, sender_id: str):
    """只返回当前客户当前业务目标的复查历史，隔离其他会话和目标。"""
    context = store.current_task_context(conversation_id, sender_id)
    key = context.task_context_id if context else f"{conversation_id}:{sender_id}"
    return key, sorted((task for task in store.scheduled_tasks.values() if task.task_type == "conversation_follow_up" and task.aggregate_id == key and task.conversation_id == conversation_id and task.subject_id == sender_id), key=lambda task: task.created_at)


def schedule_follow_up(store, call, trace_id: str, conversation_id: str, sender_id: str, sender_name: str):
    """保存复查的责任、时间和期限；新计划继承总次数与最初期限，不能无限延期。"""
    stamp = now()
    due = datetime.fromisoformat(call.arguments["due_at"])
    expires = datetime.fromisoformat(call.arguments["expires_at"])
    if due.tzinfo is None or expires.tzinfo is None or not stamp + timedelta(seconds=60) <= due <= expires <= stamp + timedelta(days=1):
        raise ValueError("复查须至少间隔60秒，且在24小时内的有效期内")
    key, history = follow_up_tasks(store, conversation_id, sender_id)
    latest = history[-1] if history else None
    count = int(latest.payload.get("review_count", 0)) if latest else 0
    limit = min(int(call.arguments.get("review_budget", 2)), int(latest.payload["review_budget"]) if latest else 3)
    if latest:
        expires = min(expires, datetime.fromisoformat(latest.payload["expires_at"]))
    if count >= limit or due > expires:
        raise ValueError("该目标复查次数或有效期已耗尽，不能继续重设")
    context = store.current_task_context(conversation_id, sender_id)
    source_turn = next((turn for turn in reversed(store.recent_turns(conversation_id)) if turn.role.value == "user" and turn.sender_id == sender_id and not turn.metadata.get("context_invalidated") and turn.metadata.get("input_source") != "system_trigger"), None)
    source_id = latest.payload.get("source_message_id") if latest else None
    if not source_id and source_turn:
        source_id = source_turn.metadata.get("source_message_id")
    task = ScheduledAgentTask(task_id=new_id("follow_up"), task_type="conversation_follow_up", aggregate_type="conversation_goal", aggregate_id=key,
                              conversation_id=conversation_id, subject_id=sender_id, subject_name=sender_name,
                              due_at=due, idempotency_key=new_id("follow_up_event"), payload={
                                  "goal": call.arguments["goal"], "waiting_for": call.arguments["waiting_for"],
                                  "expires_at": expires.isoformat(), "review_budget": limit, "review_count": count,
                                  "task_context_id": context.task_context_id if context else None,
                                  "source_message_id": source_id,
                              })
    transitions = store.save_follow_up_task(task, trace_id=trace_id)
    return ToolResult(name=call.name, called=True, allowed=True, result={"follow_up": task.to_dict(), "instruction": "复查已持久化。到期先读当前事实，可静默、有限重排或结束；不是自动发送催促。"}, state_transitions=transitions)


def cancel_follow_up(store, call, trace_id: str, conversation_id: str, sender_id: str, sender_name: str):
    """结束当前客户的复查计划，不能取消其他客户或业务目标的计划。"""
    key, history = follow_up_tasks(store, conversation_id, sender_id)
    if not history or history[-1].status not in {ScheduledTaskStatus.PENDING, ScheduledTaskStatus.PROCESSING}:
        return ToolResult(name=call.name, called=True, allowed=True, result={"closed": True, "already_closed": True})
    task = ScheduledAgentTask(task_id=new_id("follow_up_closed"), task_type="conversation_follow_up", aggregate_type="conversation_goal", aggregate_id=key,
                              conversation_id=conversation_id, subject_id=sender_id, subject_name=sender_name, due_at=now(), idempotency_key=new_id("follow_up_closed"),
                              status=ScheduledTaskStatus.CANCELLED, payload={**(history[-1].payload if history else {}), "closed_reason": call.arguments["reason"]})
    transitions = store.save_follow_up_task(task, trace_id=trace_id)
    return ToolResult(name=call.name, called=True, allowed=True, result={"closed": True}, state_transitions=transitions)
