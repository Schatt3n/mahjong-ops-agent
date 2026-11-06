"""内存版个人截止事务；与人数写入共用同一锁，并模拟失败回滚。"""

from copy import deepcopy
from contextlib import contextmanager

from ...domains.participation_deadlines import (
    build_participation_deadline, deadline_execution_decision, matching_live_deadlines,
    validate_deadline_source, validate_participation_deadline,
    ParticipationDeadlineError, ParticipationDeadlineRolledBackError,
)
from ...models import ScheduledTaskStatus, StateTransition, now


class InMemoryParticipationDeadlineStoreMixin:
    """保存个人条件责任并原子执行；不读取模型摘要作为执行依据。"""

    __slots__ = ()

    @contextmanager
    def _participation_deadline_write(self):
        """仅在完整恢复已触及的内存事实后标记可重试，领域拒绝保持原错误码。"""
        with self._lock:
            tasks, transitions = deepcopy(self.scheduled_tasks), deepcopy(self.transitions)
            try:
                yield
            except Exception as exc:
                self.scheduled_tasks.clear(); self.scheduled_tasks.update(tasks)
                self.transitions.clear(); self.transitions.extend(transitions)
                if isinstance(exc, ParticipationDeadlineError):
                    raise
                raise ParticipationDeadlineRolledBackError(str(exc)) from exc

    def save_participation_deadline(self, *, game_id, customer_id, customer_name, conversation_id,
                                    task_context_id, source_message_id, deadline_at, trace_id, at=None):
        """先验证再替换；同来源同时间重放复用原任务，不复活已经撤销的旧截止。"""
        stamp = at or now()
        with self._participation_deadline_write():
            validate_deadline_source(self, conversation_id=conversation_id, customer_id=customer_id,
                source_message_id=source_message_id, task_context_id=task_context_id)
            for old in self.scheduled_tasks.values():
                if (old.task_type == "participation_deadline" and old.subject_id == customer_id
                        and old.payload.get("game_id") == game_id and old.conversation_id == conversation_id
                        and old.payload.get("source_message_id") == source_message_id and old.due_at == deadline_at):
                    return deepcopy(old), []
            validate_participation_deadline(self.require_game(game_id), customer_id, deadline_at, at=stamp)
            task = build_participation_deadline(game_id=game_id, customer_id=customer_id,
                customer_name=customer_name, conversation_id=conversation_id, task_context_id=task_context_id,
                source_message_id=source_message_id, deadline_at=deadline_at, at=stamp)
            transitions = self._clear_participation_deadlines_locked(game_id, customer_id, trace_id, stamp, "deadline_replanned")
            self.scheduled_tasks[task.task_id] = task
            transition = StateTransition("scheduled_agent_task", task.task_id, None, "pending", "deadline_registered", trace_id)
            self.transitions.append(transition); transitions.append(transition)
            return deepcopy(task), transitions

    def clear_participation_deadline(self, *, game_id, customer_id, conversation_id, task_context_id,
                                     source_message_id, trace_id, at=None):
        """只撤销当前客户的条件责任，保留原参与和所有历史任务证据。"""
        with self._participation_deadline_write():
            validate_deadline_source(self, conversation_id=conversation_id, customer_id=customer_id,
                source_message_id=source_message_id, task_context_id=task_context_id)
            self.require_game(game_id)
            return self._clear_participation_deadlines_locked(game_id, customer_id, trace_id, at or now(), "deadline_revoked")

    def _clear_participation_deadlines_locked(self, game_id, customer_id, trace_id, at, reason):
        """调用方持锁；取消旧代及其租约，防止正在处理的旧worker撤人。"""
        transitions = []
        for task in matching_live_deadlines(self.scheduled_tasks.values(), game_id, customer_id):
            transition = StateTransition("scheduled_agent_task", task.task_id, task.status.value, "cancelled", reason, trace_id)
            task.status = ScheduledTaskStatus.CANCELLED
            task.completed_at = task.updated_at = at
            task.lease_until = None
            transitions.append(transition)
        self.transitions.extend(transitions)
        return transitions

    def execute_participation_deadline(self, claim, *, trace_id, at=None):
        """在人数写锁内核验租约、检查是否成局、释放本人席位并登记看板刷新。"""
        stamp = at or now()
        with self._lock:
            current = self.scheduled_tasks.get(claim.task_id)
            if current is None:
                return {"outcome":"stale_claim", "task_id":claim.task_id}, []
            # 领取对象只提供租约凭证；作用对象和执行范围始终重读数据库中的任务。
            game = self.require_game(current.payload["game_id"])
            decision = deadline_execution_decision(current, claim, game, at=stamp)
            if decision in {"stale_claim", "already_applied"}:
                return {"outcome":decision, "task_id":claim.task_id}, []
            saved = {name:deepcopy(getattr(self,name)) for name in (
                "games", "invite_drafts", "scheduled_tasks", "game_conversation_link_records", "transitions")}
            try:
                transitions = []
                if decision == "withdraw_self":
                    game, transitions = self.record_candidate_reply(game_id=game.game_id, customer_id=current.subject_id,
                        display_name=current.subject_name, status="declined", seat_count=1, trace_id=trace_id)
                    self._ensure_game_board_arrangement_locked(game.game_id, trace_id=trace_id, at=stamp, urgent=True)
                current.payload.update(execution_outcome=decision, executed_at=stamp.isoformat())
                current.updated_at = stamp
                return {"outcome":decision, "task_id":claim.task_id, "game_id":game.game_id,
                        "customer_id":current.subject_id, "remaining_seats":game.remaining_seats()}, transitions
            except Exception:
                for name, value in saved.items():
                    target = getattr(self,name)
                    target.clear(); target.update(value) if isinstance(target,dict) else target.extend(value)
                raise
