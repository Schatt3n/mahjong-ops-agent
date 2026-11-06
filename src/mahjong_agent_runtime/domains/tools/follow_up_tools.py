"""将模型提出的等待策略保存为有界复查任务；到期动作仍由主 Agent 判断。"""

from __future__ import annotations

from datetime import datetime, timedelta
from ...models import ScheduledAgentTask, ScheduledTaskStatus, ToolResult, new_id, now
from ..participation_deadlines import ParticipationDeadlineError, ParticipationDeadlineRolledBackError


class FollowUpTimeWindowError(ValueError):
    """在任何落库前拒绝普通复查时间，用于精确识别可被个人截止替代的失败尝试。"""


def follow_up_tasks(store, conversation_id: str, sender_id: str):
    """只返回当前客户当前业务目标的复查历史，隔离其他会话和目标。"""
    context = store.current_task_context(conversation_id, sender_id)
    key = context.task_context_id if context else f"{conversation_id}:{sender_id}"
    return key, sorted((task for task in store.scheduled_tasks.values() if task.task_type == "conversation_follow_up" and task.aggregate_id == key and task.conversation_id == conversation_id and task.subject_id == sender_id), key=lambda task: task.created_at)


def _schedule_follow_up(store, call, trace_id: str, conversation_id: str, sender_id: str, sender_name: str):
    """保存复查的责任、时间和期限；新计划继承总次数与最初期限，不能无限延期。"""
    stamp = now()
    due = datetime.fromisoformat(call.arguments["due_at"])
    expires = datetime.fromisoformat(call.arguments["expires_at"])
    if due.tzinfo is None or expires.tzinfo is None or not stamp + timedelta(seconds=60) <= due <= expires <= stamp + timedelta(days=1):
        raise FollowUpTimeWindowError("复查须至少间隔60秒，且在24小时内的有效期内")
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


def responsibility_continuation(key: str, *, completed: bool, capability: str):
    """具名责任只有同一责任的成功才可解除；失败可以澄清或转人工，不能假完成。"""
    return {"obligation_key":key, "can_stop":completed,
            "allowed_terminal_statuses":["waiting_user", "needs_human", "unknown"],
            "pending_capabilities":[] if completed else [capability]}


def schedule_follow_up(store, call, trace_id: str, conversation_id: str, sender_id: str, sender_name: str):
    """将明确提出但未持久化的复查作为失败责任回灌，不能被其他工具成功掩盖。"""
    key, history = follow_up_tasks(store, conversation_id, sender_id)
    try:
        result = _schedule_follow_up(store, call, trace_id, conversation_id, sender_id, sender_name)
    except Exception as exc:
        result = ToolResult(name=call.name, called=False, allowed=False, error=f"{type(exc).__name__}: {exc}")
        stamp = now()
        upper = min(stamp+timedelta(days=1), datetime.fromisoformat(history[-1].payload["expires_at"])) if history else stamp+timedelta(days=1)
        result.result["validation_error"] = {"code":"follow_up_not_persisted", "message":str(exc),
            "valid_time_window":{"due_at_min":(stamp+timedelta(seconds=60)).isoformat(),
                "expires_at_max":upper.isoformat(), "due_at_not_after_expires_at":True},
            "instruction":"普通复查仍受原目标最初期限和剩余次数限制；个人未成局退出条件应使用set_participation_deadline。"}
        if isinstance(exc, FollowUpTimeWindowError):
            # 这是后端在落库前生成的失败证据；不是模型自称已撤销/替代了某个责任。
            try:
                context_id, source_id = _deadline_source(store, conversation_id, sender_id, trace_id)
                result.result["failed_follow_up"] = {"task_context_id":context_id,
                    "source_message_id":source_id, "conversation_id":conversation_id, "customer_id":sender_id,
                    "due_at":call.arguments["due_at"], "expires_at":call.arguments["expires_at"],
                    "waiting_for":call.arguments["waiting_for"], "not_persisted":True,
                    "failure_kind":"time_window_parameter_rejected"}
            except ParticipationDeadlineError:
                pass
    result.result["continuation"] = responsibility_continuation(f"follow_up:{key}",
        completed=bool(result.allowed and result.called and not result.error), capability="persist_follow_up_or_clarify")
    return result


def _deadline_source(store, conversation_id, sender_id, trace_id):
    """读取本轮真实本人消息；来源ID由后端持久会话提供，不接受模型参数。"""
    context = store.current_task_context(conversation_id, sender_id)
    source = next((turn for turn in reversed(store.recent_turns(conversation_id))
                   if turn.role.value == "user" and turn.sender_id == sender_id and turn.trace_id == trace_id
                   and not turn.metadata.get("context_invalidated")
                   and turn.metadata.get("input_source") != "system_trigger"), None)
    if context is None or source is None or not source.metadata.get("source_message_id"):
        raise ParticipationDeadlineError("deadline_source_required", "需要本人当前轮的真实用户消息才能变更截止责任。")
    return context.task_context_id, source.metadata["source_message_id"]


def _change_participation_deadline(store, call, trace_id, conversation_id, sender_id, sender_name, *, clear):
    """只以鉴权本人身份提交事务；多席、非法时间、持久化失败均返回可恢复责任。"""
    game_id = call.arguments["game_id"]
    key = f"deadline:{game_id}:{sender_id}"
    try:
        task_context_id, source_message_id = _deadline_source(store, conversation_id, sender_id, trace_id)
        common = dict(game_id=game_id, customer_id=sender_id, conversation_id=conversation_id,
            task_context_id=task_context_id, source_message_id=source_message_id, trace_id=trace_id)
        if clear:
            transitions = store.clear_participation_deadline(**common)
            payload = {"cleared":True, "game_id":game_id}
        else:
            task, transitions = store.save_participation_deadline(**common, customer_name=sender_name,
                deadline_at=datetime.fromisoformat(call.arguments["deadline_at"]))
            payload = {"participation_deadline":task.to_dict()}
        payload["continuation"] = responsibility_continuation(key, completed=True, capability="persist_participation_deadline")
        return ToolResult(name=call.name, called=True, allowed=True, result=payload, state_transitions=transitions)
    except Exception as exc:
        return ToolResult(name=call.name, called=False, allowed=False, error=f"{type(exc).__name__}: {exc}", result={
            "retryable_no_effect":isinstance(exc, ParticipationDeadlineRolledBackError),
            "validation_error":{"code":getattr(exc,"code","deadline_persistence_failed"), "message":str(exc)},
            "continuation":responsibility_continuation(key, completed=False, capability="persist_participation_deadline")})


def set_participation_deadline(store, call, trace_id: str, conversation_id: str, sender_id: str, sender_name: str):
    """为本人独立席位登记未成局退出条件；不设置整局取消或其他客户的截止。"""
    return _change_participation_deadline(store, call, trace_id, conversation_id, sender_id, sender_name, clear=False)


def clear_participation_deadline(store, call, trace_id: str, conversation_id: str, sender_id: str, sender_name: str):
    """本人明确撤销条件时撤销待执行责任，不退出席位。"""
    return _change_participation_deadline(store, call, trace_id, conversation_id, sender_id, sender_name, clear=True)


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
