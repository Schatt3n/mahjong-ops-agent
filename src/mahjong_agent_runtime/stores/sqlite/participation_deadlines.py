"""SQLite个人截止事务；调度记录、本人席位和看板责任共享权威数据库。"""

from contextlib import contextmanager

from ...domains.participation_deadlines import (
    build_participation_deadline, deadline_execution_decision, matching_live_deadlines,
    validate_deadline_source, validate_participation_deadline,
    ParticipationDeadlineError, ParticipationDeadlineRolledBackError,
)
from ...models import ScheduledTaskStatus, StateTransition, now


class SQLiteParticipationDeadlineStoreMixin:
    """沿用现有调度表，无第二调度器、额外事实源或运行时schema兼容。"""

    __slots__ = ()

    @contextmanager
    def _participation_deadline_write(self):
        """只标记事务体抛错且标准事务已成功回滚的失败；提交/回滚异常不推断为无副作用。"""
        body_error = None
        try:
            with self._write_transaction():
                try:
                    yield
                except Exception as exc:
                    body_error = exc
                    raise
        except ParticipationDeadlineError:
            raise
        except Exception as exc:
            if exc is body_error:
                raise ParticipationDeadlineRolledBackError(str(exc)) from exc
            raise

    def save_participation_deadline(self, *, game_id, customer_id, customer_name, conversation_id,
                                    task_context_id, source_message_id, deadline_at, trace_id, at=None):
        """事务内先验证再替换；同来源同时间的重放保持原任务身份与终态。"""
        stamp = at or now()
        with self._participation_deadline_write():
            validate_deadline_source(self, conversation_id=conversation_id, customer_id=customer_id,
                source_message_id=source_message_id, task_context_id=task_context_id)
            for old in self.scheduled_tasks.values():
                if (old.task_type == "participation_deadline" and old.subject_id == customer_id
                        and old.payload.get("game_id") == game_id and old.conversation_id == conversation_id
                        and old.payload.get("source_message_id") == source_message_id and old.due_at == deadline_at):
                    return old, []
            validate_participation_deadline(self.require_game(game_id), customer_id, deadline_at, at=stamp)
            task = build_participation_deadline(game_id=game_id, customer_id=customer_id,
                customer_name=customer_name, conversation_id=conversation_id, task_context_id=task_context_id,
                source_message_id=source_message_id, deadline_at=deadline_at, at=stamp)
            transitions = self._clear_participation_deadlines_in_transaction(game_id, customer_id, trace_id, stamp, "deadline_replanned")
            self._save_scheduled_agent_task(task)
            transition = StateTransition("scheduled_agent_task", task.task_id, None, "pending", "deadline_registered", trace_id)
            self._append_transition(transition); transitions.append(transition)
            return task, transitions

    def clear_participation_deadline(self, *, game_id, customer_id, conversation_id, task_context_id,
                                     source_message_id, trace_id, at=None):
        """只取消本人截止；和重新报名、改期并发时以写事务提交顺序为准。"""
        with self._participation_deadline_write():
            validate_deadline_source(self, conversation_id=conversation_id, customer_id=customer_id,
                source_message_id=source_message_id, task_context_id=task_context_id)
            self.require_game(game_id)
            return self._clear_participation_deadlines_in_transaction(game_id, customer_id, trace_id, at or now(), "deadline_revoked")

    def _clear_participation_deadlines_in_transaction(self, game_id, customer_id, trace_id, at, reason):
        """取消任务与释放旧租约同事务提交，旧worker只能观察到失效凭证。"""
        transitions = []
        for task in matching_live_deadlines(self.scheduled_tasks.values(), game_id, customer_id):
            transition = StateTransition("scheduled_agent_task", task.task_id, task.status.value, "cancelled", reason, trace_id)
            task.status = ScheduledTaskStatus.CANCELLED
            task.completed_at = task.updated_at = at
            task.lease_until = None
            self._save_scheduled_agent_task(task); self._append_transition(transition)
            transitions.append(transition)
        return transitions

    def execute_participation_deadline(self, claim, *, trace_id, at=None):
        """同事务核验租约、读取最新人数、执行本人退出；已满局不会被旧快照拆散。"""
        stamp = at or now()
        with self._write_transaction():
            current = self.scheduled_tasks.get(claim.task_id)
            if current is None:
                return {"outcome":"stale_claim", "task_id":claim.task_id}, []
            # 领取快照只作凭证，不能改变数据库保存的受托人和牌局范围。
            game = self.require_game(current.payload["game_id"])
            decision = deadline_execution_decision(current, claim, game, at=stamp)
            if decision in {"stale_claim", "already_applied"}:
                return {"outcome":decision, "task_id":claim.task_id}, []
            transitions = []
            if decision == "withdraw_self":
                game, transitions = self._record_candidate_reply_in_transaction(game_id=game.game_id,
                    customer_id=current.subject_id, display_name=current.subject_name, status="declined", seat_count=1, trace_id=trace_id)
                self._ensure_game_board_arrangement_in_transaction(game.game_id, trace_id=trace_id, at=stamp, urgent=True)
            current.payload.update(execution_outcome=decision, executed_at=stamp.isoformat())
            current.updated_at = stamp
            self._save_scheduled_agent_task(current)
            return {"outcome":decision, "task_id":claim.task_id, "game_id":game.game_id,
                    "customer_id":current.subject_id, "remaining_seats":game.remaining_seats()}, transitions
