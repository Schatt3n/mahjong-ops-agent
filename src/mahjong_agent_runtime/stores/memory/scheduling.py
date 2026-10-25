"""实现内存存储层的scheduling数据访问能力，主要用于本地运行和测试。

"""

from __future__ import annotations

from datetime import datetime, timedelta

from ...models import (
    Game,
    GameStatus,
    RecruitmentStatus,
    ScheduledAgentTask,
    ScheduledTaskStatus,
    StateTransition,
    now,
)
from ...domains import (
    GAME_RECRUITMENT_TASK_TYPE,
    SCHEDULED_TASK_PROCESSING_LEASE_SECONDS,
    apply_game_recruitment_policy,
    game_recruitment_task_id,
    game_schedule_sort_key,
    normalize_datetime,
)
from ...domains.waiting_domain import WAITING_DEMAND_EXPIRY_TASK_TYPE, waiting_expiry_task_id


class InMemorySchedulingStoreMixin:
    """从兼容性存储中提取的后端特定操作。"""

    __slots__ = ()

    def active_games(self, conversation_id: str | None = None) -> list[Game]:
        with self._lock:
            self._expire_stale_games_locked(trace_id="system_lifecycle")
            games = [
                item
                for item in self.games.values()
                if item.status.value in {GameStatus.FORMING.value, GameStatus.INVITING.value, GameStatus.READY.value}
            ]
            for game in games:
                apply_game_recruitment_policy(game)
            if conversation_id:
                games = [item for item in games if item.conversation_id == conversation_id]
            return sorted(games, key=game_schedule_sort_key)

    def scheduled_task_for_game(self, game_id: str) -> ScheduledAgentTask | None:
        """返回与一次牌局关联的持久化招募触发器。"""

        task_id = game_recruitment_task_id(game_id)
        with self._lock:
            return self.scheduled_tasks.get(task_id)

    def ensure_game_recruitment_task(
        self,
        game_id: str,
        *,
        trace_id: str,
    ) -> tuple[ScheduledAgentTask | None, StateTransition | None]:
        """确保牌局recruitment任务满足后续步骤依赖的不变量。

"""

        with self._lock:
            game = self.require_game(game_id)
            return self._sync_game_recruitment_task_locked(game, trace_id=trace_id)

    def ensure_waiting_demand_expiration_task(
        self,
        *,
        due_at: datetime,
        trace_id: str,
    ) -> tuple[ScheduledAgentTask, StateTransition | None]:
        """确保候补需求expiration任务满足后续步骤依赖的不变量。

"""

        task_id = waiting_expiry_task_id(due_at)
        with self._lock:
            existing = self.scheduled_tasks.get(task_id)
            if existing is not None:
                return existing, None
            task = ScheduledAgentTask(
                task_id=task_id,
                task_type=WAITING_DEMAND_EXPIRY_TASK_TYPE,
                aggregate_type="waiting_list",
                aggregate_id="global",
                conversation_id="system:waiting-list",
                subject_id="system",
                subject_name="system",
                due_at=due_at,
                idempotency_key=task_id,
                payload={"event_type": WAITING_DEMAND_EXPIRY_TASK_TYPE},
            )
            self.scheduled_tasks[task_id] = task
            transition = StateTransition(
                "scheduled_agent_task",
                task_id,
                None,
                task.status.value,
                "waiting_demand_expiration_scheduled",
                trace_id,
            )
            self.transitions.append(transition)
            return task, transition

    def _sync_game_recruitment_task_locked(
        self,
        game: Game,
        *,
        trace_id: str,
    ) -> tuple[ScheduledAgentTask | None, StateTransition | None]:
        apply_game_recruitment_policy(game)
        task_id = game_recruitment_task_id(game.game_id)
        existing = self.scheduled_tasks.get(task_id)
        if game.recruitment_status != RecruitmentStatus.SCHEDULED or game.recruitment_opens_at is None:
            if existing is None or existing.status in {
                ScheduledTaskStatus.COMPLETED,
                ScheduledTaskStatus.CANCELLED,
                ScheduledTaskStatus.FAILED,
            }:
                return existing, None
            old = existing.status.value
            existing.status = ScheduledTaskStatus.CANCELLED
            existing.completed_at = now()
            existing.lease_until = None
            existing.updated_at = now()
            transition = StateTransition(
                "scheduled_agent_task",
                existing.task_id,
                old,
                existing.status.value,
                "recruitment_no_longer_scheduled",
                trace_id,
            )
            self.transitions.append(transition)
            return existing, transition

        task_context = self.current_task_context(game.conversation_id, game.organizer_id)
        payload = {
            "event_type": "game_recruitment_window_opened",
            "game_id": game.game_id,
            "planned_start_at": game.planned_start_at.isoformat() if game.planned_start_at else None,
            "recruitment_opens_at": game.recruitment_opens_at.isoformat(),
            "task_context_id": task_context.task_context_id if task_context else None,
        }
        if existing is None:
            task = ScheduledAgentTask(
                task_id=task_id,
                task_type=GAME_RECRUITMENT_TASK_TYPE,
                aggregate_type="game",
                aggregate_id=game.game_id,
                conversation_id=game.conversation_id,
                subject_id=game.organizer_id,
                subject_name=game.organizer_name,
                due_at=game.recruitment_opens_at,
                idempotency_key=f"{GAME_RECRUITMENT_TASK_TYPE}:{game.game_id}:{game.recruitment_opens_at.isoformat()}",
                payload=payload,
            )
            self.scheduled_tasks[task.task_id] = task
            transition = StateTransition(
                "scheduled_agent_task",
                task.task_id,
                None,
                task.status.value,
                "future_game_created",
                trace_id,
            )
            self.transitions.append(transition)
            return task, transition

        old = existing.status.value
        existing.due_at = game.recruitment_opens_at
        existing.idempotency_key = (
            f"{GAME_RECRUITMENT_TASK_TYPE}:{game.game_id}:{game.recruitment_opens_at.isoformat()}"
        )
        existing.payload = payload
        existing.status = ScheduledTaskStatus.PENDING
        existing.lease_until = None
        existing.completed_at = None
        existing.last_error = ""
        existing.updated_at = now()
        transition = StateTransition(
            "scheduled_agent_task",
            existing.task_id,
            old,
            existing.status.value,
            "future_game_schedule_updated",
            trace_id,
        )
        self.transitions.append(transition)
        return existing, transition

    def due_scheduled_tasks(self, *, at: datetime, limit: int = 100) -> list[ScheduledAgentTask]:
        """列出到期的工作；已过期的处理租约可供恢复。"""

        at = normalize_datetime(at)
        with self._lock:
            due = [
                item
                for item in self.scheduled_tasks.values()
                if (
                    item.status == ScheduledTaskStatus.PENDING and item.due_at <= at
                )
                or (
                    item.status == ScheduledTaskStatus.PROCESSING
                    and item.lease_until is not None
                    and item.lease_until <= at
                )
            ]
            return sorted(due, key=lambda item: (item.due_at, item.task_id))[: int(limit)]

    def open_game_recruitment(
        self,
        game_id: str,
        *,
        trace_id: str,
        at: datetime | None = None,
    ) -> tuple[Game, StateTransition | None]:
        """持久化从计划可见性到活跃招募的状态转换。"""

        stamp = normalize_datetime(at or now())
        with self._lock:
            game = self.require_game(game_id)
            old = game.recruitment_status.value
            apply_game_recruitment_policy(game, at=stamp)
            if game.recruitment_status == RecruitmentStatus.SCHEDULED:
                raise ValueError(
                    "recruitment window is not open: "
                    f"recruitment_opens_at={game.recruitment_opens_at.isoformat() if game.recruitment_opens_at else None}"
                )
            if old == game.recruitment_status.value:
                return game, None
            game.updated_at = stamp
            transition = StateTransition(
                "game_recruitment",
                game.game_id,
                old,
                game.recruitment_status.value,
                "scheduled_recruitment_window_opened",
                trace_id,
            )
            self.transitions.append(transition)
            return game, transition

    def claim_scheduled_task(
        self,
        task_id: str,
        *,
        at: datetime,
        lease_seconds: int = SCHEDULED_TASK_PROCESSING_LEASE_SECONDS,
    ) -> ScheduledAgentTask | None:
        """原子领取定时任务，防止并发消费者重复处理。

"""

        at = normalize_datetime(at)
        with self._lock:
            task = self.scheduled_tasks.get(task_id)
            if task is None or task.due_at > at:
                return None
            recoverable = (
                task.status == ScheduledTaskStatus.PROCESSING
                and task.lease_until is not None
                and task.lease_until <= at
            )
            if task.status != ScheduledTaskStatus.PENDING and not recoverable:
                return None
            task.status = ScheduledTaskStatus.PROCESSING
            task.attempts += 1
            if task.task_type == "conversation_follow_up" and not recoverable:
                task.payload["review_count"] = int(task.payload.get("review_count", 0)) + 1
            task.lease_until = at + timedelta(seconds=max(1, int(lease_seconds)))
            task.updated_at = at
            return task

    def complete_scheduled_task(
        self,
        task_id: str,
        *,
        trace_id: str,
        at: datetime | None = None,
    ) -> tuple[ScheduledAgentTask | None, StateTransition | None]:
        stamp = normalize_datetime(at or now())
        with self._lock:
            task = self.scheduled_tasks.get(task_id)
            if task is None or task.status in {ScheduledTaskStatus.COMPLETED, ScheduledTaskStatus.CANCELLED}:
                return task, None
            old = task.status.value
            task.status = ScheduledTaskStatus.COMPLETED
            task.completed_at = stamp
            task.lease_until = None
            task.updated_at = stamp
            transition = StateTransition(
                "scheduled_agent_task",
                task.task_id,
                old,
                task.status.value,
                "scheduled_agent_task_completed",
                trace_id,
            )
            self.transitions.append(transition)
            return task, transition

    def fail_scheduled_task(
        self,
        task_id: str,
        *,
        trace_id: str,
        error: str,
        max_attempts: int = 3,
        retry_delay_seconds: int = 60,
        at: datetime | None = None,
    ) -> tuple[ScheduledAgentTask | None, StateTransition | None]:
        stamp = normalize_datetime(at or now())
        with self._lock:
            task = self.scheduled_tasks.get(task_id)
            if task is None:
                return None, None
            old = task.status.value
            task.last_error = str(error or "scheduled agent task failed")
            task.lease_until = None
            if task.attempts >= max(1, int(max_attempts)):
                task.status = ScheduledTaskStatus.FAILED
                task.completed_at = stamp
            else:
                task.status = ScheduledTaskStatus.PENDING
                task.due_at = stamp + timedelta(seconds=max(1, int(retry_delay_seconds)))
            task.updated_at = stamp
            transition = StateTransition(
                "scheduled_agent_task",
                task.task_id,
                old,
                task.status.value,
                "scheduled_agent_task_failed",
                trace_id,
            )
            self.transitions.append(transition)
            return task, transition

    def save_follow_up_task(self, task: ScheduledAgentTask, *, trace_id: str) -> list[StateTransition]:
        """在锁内替换同一目标的未完成复查，任务 ID 不复用，避免旧租约覆盖新计划。"""
        with self._lock:
            transitions = []
            for existing in self.scheduled_tasks.values():
                if existing.task_type == "conversation_follow_up" and existing.aggregate_id == task.aggregate_id and existing.conversation_id == task.conversation_id and existing.subject_id == task.subject_id and existing.status in {ScheduledTaskStatus.PENDING, ScheduledTaskStatus.PROCESSING}:
                    old = existing.status.value
                    existing.status = ScheduledTaskStatus.CANCELLED
                    existing.completed_at = now()
                    transitions.append(StateTransition("scheduled_agent_task", existing.task_id, old, "cancelled", "follow_up_replanned", trace_id))
            self.scheduled_tasks[task.task_id] = task
            transitions.append(StateTransition("scheduled_agent_task", task.task_id, None, task.status.value, "follow_up_planned", trace_id))
            self.transitions.extend(transitions)
            return transitions
