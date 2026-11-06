"""个人截止的既有Scheduler回调；执行权威条件事务，不拼接或绕过审查外发文字。"""

from .models import AgentRuntimeResult, now


def handle_participation_deadline_task(store, task, trace_id: str, *, at=None, trace_recorder=None):
    """到期条件与人数写入同事务结算；任务重试复用已执行结果，群看板沿现有责任刷新。"""
    outcome, transitions = store.execute_participation_deadline(task, trace_id=trace_id, at=at or now())
    if trace_recorder is not None:
        trace_recorder.record(trace_id, "participation_deadline_evaluated", {
            **outcome, "source_message_id":task.payload.get("source_message_id"),
            "deadline_at":task.payload.get("deadline_at"),
            "state_transitions":[item.to_dict() for item in transitions],
        })
    # 此处只有已授权的领域条件执行，没有模型回复；不伪装为客户发送的消息或创建外发草稿。
    return AgentRuntimeResult(trace_id=trace_id, conversation_id=task.conversation_id,
        final_reply="", status="completed", state_transitions=transitions)
