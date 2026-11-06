"""用本轮真实工具证据证明复查失败被同一截止责任替代，拒绝模型自报恢复。"""

from datetime import datetime
import json


def resolve_replaced_follow_up(store, result, observed_results, message, trace_id):
    """仅承接同输入/目标/时刻且可唯一关联本人牌局的未落库时间参数失败。"""
    if result.name != "set_participation_deadline" or not result.called or not result.allowed or result.error:
        return
    task = result.result.get("participation_deadline") or {}
    detail = task.get("payload") or {}
    if task.get("status") not in {"pending", "processing"} or detail.get("execution_outcome"):
        return
    current = store.current_task_context(message.conversation_id, message.sender_id)
    if (current is None or detail.get("task_context_id") != current.task_context_id
            or detail.get("source_message_id") != message.message_id
            or task.get("subject_id") != message.sender_id or task.get("conversation_id") != message.conversation_id):
        return
    # 前一步反馈可能已被其他只读步骤替换；补读本轮持久工具证据，绝不读取其他人的会话。
    evidence = []
    for turn in store.recent_turns(message.conversation_id, limit=60):
        if turn.role.value != "tool" or turn.trace_id != trace_id or turn.metadata.get("context_invalidated"):
            continue
        try:
            values = json.loads(turn.content)
        except (ValueError, TypeError):
            continue
        if isinstance(values, list):
            evidence.extend(item for item in values if isinstance(item,dict))
    evidence.extend(item.to_dict() for item in observed_results)
    resolved = []
    for previous in evidence:
        data = previous.get("result") or {}
        failed = data.get("failed_follow_up") or {}
        if (previous.get("name") != "schedule_follow_up" or previous.get("called") or previous.get("allowed")
                or not previous.get("error") or failed.get("not_persisted") is not True
                or failed.get("failure_kind") != "time_window_parameter_rejected" or failed.get("waiting_for") != "players"
                or failed.get("customer_id") != message.sender_id or failed.get("conversation_id") != message.conversation_id
                or failed.get("source_message_id") != message.message_id
                or failed.get("task_context_id") != current.task_context_id):
            continue
        try:
            due = datetime.fromisoformat(failed["due_at"])
            expires = datetime.fromisoformat(failed["expires_at"])
            if due != datetime.fromisoformat(detail["deadline_at"]):
                continue
        except (ValueError, TypeError, KeyError):
            continue
        candidates = [game for game in store.games.values() if game.status.value in {"forming","inviting"}
            and game.expires_at == expires and any(member.customer_id == message.sender_id
                and member.status in {"joined","confirmed"} for member in game.participants)]
        if len(candidates) != 1 or candidates[0].game_id != detail.get("game_id"):
            continue
        key = str((data.get("continuation") or {}).get("obligation_key") or "")
        if key == f"follow_up:{current.task_context_id}":
            resolved.append(key)
    if resolved:
        result.result["continuation"]["resolved_obligation_keys"] = sorted(set(resolved))
