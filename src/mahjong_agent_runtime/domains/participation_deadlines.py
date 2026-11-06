"""本人等待截止的领域合同；定时器只执行明确授权的独立席位条件。"""

from __future__ import annotations

from datetime import datetime

from ..models import ScheduledAgentTask, ScheduledTaskStatus, new_id

DEADLINE_TASK_TYPE = "participation_deadline"
LIVE_DEADLINE_STATUSES = {ScheduledTaskStatus.PENDING, ScheduledTaskStatus.PROCESSING}


class ParticipationDeadlineError(ValueError):
    """携带稳定错误码，供工具反馈和审查区分越权、时间与席位歧义。"""

    def __init__(self, code: str, message: str):
        """保存可枚举领域原因；不根据自然语言推断用户意图。"""
        super().__init__(message)
        self.code = code


class ParticipationDeadlineRolledBackError(RuntimeError):
    """仅由Store确认事务完整回滚后抛出；未知结果不能标记为安全重试。"""


def validate_deadline_source(store, *, conversation_id, customer_id, source_message_id, task_context_id):
    """只接受本人当前目标里的真实用户消息，拒绝系统唤醒和模型伪造来源。"""
    turn = store.find_conversation_turn(conversation_id, message_id=source_message_id)
    context = store.current_task_context(conversation_id, customer_id)
    if (turn is None or turn.role.value != "user" or turn.sender_id != customer_id
            or turn.metadata.get("context_invalidated") or turn.metadata.get("input_source") == "system_trigger"
            or context is None or context.task_context_id != task_context_id
            or turn.metadata.get("task_context_id") != task_context_id):
        raise ParticipationDeadlineError("deadline_source_required", "截止变更需要本人当前目标的原始用户消息依据。")


def validate_participation_deadline(game, customer_id: str, deadline_at: datetime, *, at: datetime):
    """验证独立本人席位和已知牌局生命周期；不放宽普通复查的24小时预算。"""
    member = next((item for item in game.participants if item.customer_id == customer_id
                   and item.status in {"joined", "confirmed"}), None)
    if member is None:
        raise ParticipationDeadlineError("own_participation_required", "只能为本人已登记的参与设置截止，不能替别人退出。")
    if member.seat_count != 1 or member.anonymous_seat_count or set(member.known_member_ids)-{customer_id}:
        raise ParticipationDeadlineError("participation_scope_ambiguous", "当前报名包含多席同行，需先明确本人可独立退出的席位。")
    if game.status.value not in {"forming", "inviting"} or game.remaining_seats() == 0:
        raise ParticipationDeadlineError("game_not_waiting", "牌局已成局或已结束，不能新建未成局退出责任。")
    if deadline_at.tzinfo is None or not at < deadline_at or game.expires_at is None or deadline_at > game.expires_at:
        raise ParticipationDeadlineError("deadline_outside_game_window", "截止须为带时区的未来时间，且不晚于该牌局已知有效期。")
    return member


def build_participation_deadline(*, game_id, customer_id, customer_name, conversation_id,
                                 task_context_id, source_message_id, deadline_at, at):
    """每次改期产生新任务ID，旧领取凭证不能覆盖新责任；沿用现有调度表。"""
    return ScheduledAgentTask(task_id=new_id("participation_deadline"), task_type=DEADLINE_TASK_TYPE,
        aggregate_type="game_participant", aggregate_id=f"{game_id}:{customer_id}",
        conversation_id=conversation_id, subject_id=customer_id, subject_name=customer_name,
        due_at=deadline_at, idempotency_key=new_id("participation_deadline_event"),
        created_at=at, updated_at=at, payload={"game_id":game_id, "customer_id":customer_id,
            "deadline_at":deadline_at.isoformat(), "task_context_id":task_context_id,
            "source_message_id":source_message_id, "condition":"game_not_ready", "effect":"withdraw_self"})


def matching_live_deadlines(tasks, game_id, customer_id):
    """责任按牌局与本人隔离，同一人不同目标/通道的同一席位只有一个有效截止。"""
    return [task for task in tasks if task.task_type == DEADLINE_TASK_TYPE and task.subject_id == customer_id
            and task.payload.get("game_id") == game_id and task.status in LIVE_DEADLINE_STATUSES]


def deadline_execution_decision(task, claim, game, *, at):
    """在写锁内重读租约与人数；已经成局时不能因旧快照再撤人。"""
    if (task is None or task.task_type != DEADLINE_TASK_TYPE or claim.lease_until is None
            or not task.matches_claim(expected_attempts=claim.attempts, expected_lease_until=claim.lease_until)
            or task.due_at > at):
        return "stale_claim"
    # 当前代次仍持有任务但租约已过期，必须让Scheduler失败重排，不能正常返回后被标完成。
    # 被新代/取消替换的旧worker已在上面返回，不能消耗新任务的重试预算。
    if task.lease_until <= at:
        raise RuntimeError("participation deadline lease expired before execution; retry required")
    if task.payload.get("execution_outcome"):
        return "already_applied"
    if game.status.value not in {"forming", "inviting"} or game.remaining_seats() == 0:
        return "game_ready_or_closed"
    member = next((item for item in game.participants if item.customer_id == task.subject_id
                   and item.status in {"joined", "confirmed"}), None)
    if member is None:
        return "already_not_participating"
    if member.seat_count != 1 or member.anonymous_seat_count or set(member.known_member_ids)-{task.subject_id}:
        raise ParticipationDeadlineError("participation_scope_ambiguous", "截止到期时席位范围已改变，不能自动退出同行。")
    return "withdraw_self"
