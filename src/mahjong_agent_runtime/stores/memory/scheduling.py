"""InMemory scheduling store operations."""

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
    """Backend-specific operations extracted from the compatibility store."""

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
        """Return the durable recruitment trigger associated with one game."""

        task_id = game_recruitment_task_id(game_id)
        with self._lock:
            return self.scheduled_tasks.get(task_id)

    def ensure_game_recruitment_task(
        self,
        game_id: str,
        *,
        trace_id: str,
    ) -> tuple[ScheduledAgentTask | None, StateTransition | None]:
        """Synchronize one future recruitment task with the game's current start time."""

        with self._lock:
            game = self.require_game(game_id)
            return self._sync_game_recruitment_task_locked(game, trace_id=trace_id)

    def ensure_waiting_demand_expiration_task(
        self,
        *,
        due_at: datetime,
        trace_id: str,
    ) -> tuple[ScheduledAgentTask, StateTransition | None]:
        """Create the next idempotent minute-bucket maintenance task."""

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

        payload = {
            "event_type": "game_recruitment_window_opened",
            "game_id": game.game_id,
            "planned_start_at": game.planned_start_at.isoformat() if game.planned_start_at else None,
            "recruitment_opens_at": game.recruitment_opens_at.isoformat(),
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
        """List due work; an expired processing lease is eligible for recovery."""

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
        """Persist the transition from scheduled visibility to active recruitment."""

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
        """Atomically claim a due task so only one local worker executes it."""

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
            if task is None or task.status == ScheduledTaskStatus.COMPLETED:
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
