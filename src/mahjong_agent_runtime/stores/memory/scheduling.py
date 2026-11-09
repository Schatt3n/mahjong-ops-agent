"""实现内存存储层的scheduling数据访问能力，主要用于本地运行和测试。

"""

from __future__ import annotations

from copy import deepcopy
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
from ...domains.stake_values import stake_options
from ..task_store import (
    BOARD_CANDIDATE_DECISION_TASK_TYPE,
    _board_candidate_fit_expiry,
    _board_candidate_resolution,
    _board_stake_confirmation_same_purpose,
    _matching_board_candidate_task,
    _new_board_candidate_decision,
    _validate_new_board_candidate_fit_question,
)


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

    def ensure_board_stake_confirmation_task(
        self, task: ScheduledAgentTask, *, trace_id: str,
        update_task: ScheduledAgentTask | None = None,
    ) -> tuple[ScheduledAgentTask, list[StateTransition]]:
        """锁内一起登记桌主询问及询局者回告；已有到期时间、租约和终态均不重置。

        核价按局只问一次，补位征询按同局同候选人、报价回告按同局同收件人去重。锁内重读档位，
        覆盖“先确认价格、后登记订阅”的竞争，避免回告永远等不到再次变更。
        """
        if (task.task_type != "group_board_stake_confirmation" or task.aggregate_type != "game"
                or not task.task_id or not task.aggregate_id):
            raise ValueError("board stake confirmation requires its game-scoped task type")
        fit_expiry = (_board_candidate_fit_expiry(task, update_task)
                      if task.payload.get("question_kind") == "candidate_fit_only" else None)
        expiry = None
        if update_task is not None:
            if (update_task.task_type != "group_board_stake_update" or update_task.aggregate_type != "game"
                    or update_task.aggregate_id != task.aggregate_id or not update_task.task_id
                    or not update_task.subject_id or not update_task.payload.get("room_id")):
                raise ValueError("board stake update requires the same game and a recipient")
            try:
                expiry = datetime.fromisoformat(update_task.payload["expires_at"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("board stake update requires a timezone-aware expiry") from exc
            if expiry.tzinfo is None:
                raise ValueError("board stake update requires a timezone-aware expiry")
        with self._lock:
            stamp = now()
            game = self.games.get(task.aggregate_id)
            known = bool(game is not None and stake_options(game.requirement))
            selected, insertions, transitions = [], [], []
            for proposed in [task] + ([update_task] if update_task is not None else []):
                is_update = proposed.task_type == "group_board_stake_update"
                matching = []
                for existing in [*self.scheduled_tasks.values(), *insertions]:
                    same_purpose = _board_stake_confirmation_same_purpose(existing, proposed)
                    if (existing.task_id == proposed.task_id or existing.idempotency_key == proposed.idempotency_key) and not same_purpose:
                        raise ValueError("board stake confirmation task identity conflict")
                    if same_purpose:
                        matching.append(existing)
                if matching:
                    selected.append(matching[0])
                    continue
                if proposed.status != ScheduledTaskStatus.PENDING:
                    raise ValueError("new board stake confirmation task must be pending")
                if fit_expiry is not None:
                    _validate_new_board_candidate_fit_question(proposed, self.require_game(task.aggregate_id),
                        known_stake=known, expiry=fit_expiry, at=stamp)
                saved = deepcopy(proposed)
                reason = "board_stake_update_registered" if is_update else "board_stake_confirmation_scheduled"
                if is_update:
                    saved.due_at = stamp if known and expiry > stamp else expiry
                # 已知报价只消除问价责任，不能替桌主回答是否接受与软偏好不同的候选人。
                elif known and saved.payload.get("question_kind") != "candidate_fit_only":
                    saved.status = ScheduledTaskStatus.COMPLETED
                    saved.completed_at = saved.updated_at = stamp
                    reason = "board_stake_confirmation_already_known"
                selected.append(saved)
                insertions.append(saved)
                transitions.append(StateTransition("scheduled_agent_task", saved.task_id, None,
                    saved.status.value, reason, trace_id))
            for saved in insertions:
                self.scheduled_tasks[saved.task_id] = saved
            self.transitions.extend(transitions)
            return deepcopy(selected[0]), transitions

    def _activate_board_stake_updates_locked(self, game: Game, *, trace_id: str, at: datetime) -> None:
        """在牌局首次获得档位的同一锁内唤醒有效回告，已领取或过期责任不被改写。"""
        for task in self.scheduled_tasks.values():
            if (task.task_type != "group_board_stake_update" or task.aggregate_type != "game"
                    or task.aggregate_id != game.game_id or task.status != ScheduledTaskStatus.PENDING):
                continue
            try:
                expiry = datetime.fromisoformat(task.payload["expires_at"])
            except (KeyError, TypeError, ValueError):
                continue
            if expiry.tzinfo is None or expiry <= at or task.due_at <= at:
                continue
            task.due_at = task.updated_at = at
            self.transitions.append(StateTransition("scheduled_agent_task", task.task_id,
                task.status.value, task.status.value, "board_stake_update_ready", trace_id))

    def ensure_board_candidate_decision_task(
        self, task: ScheduledAgentTask, *, trace_id: str,
    ) -> tuple[ScheduledAgentTask, list[StateTransition]]:
        """在锁内按桌与候选人保存待答责任，同一责任重试保留原有证据和时效。"""
        if task.task_type != BOARD_CANDIDATE_DECISION_TASK_TYPE or task.aggregate_type != "game":
            raise ValueError("board candidate decision requires its game-scoped task type")
        with self._lock:
            existing = _matching_board_candidate_task(self.scheduled_tasks.values(), task,
                                                       candidate_id=task.payload.get("candidate_id"))
            if existing is not None:
                return deepcopy(existing), []
            saved = _new_board_candidate_decision(deepcopy(task), self.require_game(task.aggregate_id), at=now())
            transition = StateTransition("scheduled_agent_task", saved.task_id, None,
                                         saved.status.value, "board_candidate_waiting_for_host", trace_id)
            self.scheduled_tasks[saved.task_id] = saved
            self.transitions.append(transition)
            return deepcopy(saved), [transition]

    def resolve_board_candidate_decision(
        self, task_id: str, *, host_id: str, conversation_id: str, decision: str,
        evidence_message_id: str, trace_id: str, update_task: ScheduledAgentTask | None = None,
        at: datetime | None = None,
    ) -> tuple[ScheduledAgentTask, list[StateTransition]]:
        """锁内先验证完整证据与两条责任，再一起保存桌主答复和候选人回告。"""
        with self._lock:
            task = self.scheduled_tasks.get(task_id)
            if task is None:
                raise ValueError("board candidate decision task not found")
            game = self.require_game(task.aggregate_id)
            receipt = self.resolve_message_reference(conversation_id=conversation_id,
                                                     message_id=task.payload.get("question_message_id", ""))
            evidence = self.find_conversation_turn(conversation_id, message_id=evidence_message_id)
            saved, update, changes = _board_candidate_resolution(task, game, decision=decision, host_id=host_id,
                conversation_id=conversation_id, evidence=evidence, receipt=receipt, update_task=deepcopy(update_task),
                at=normalize_datetime(at or now()), trace_id=trace_id)
            if update is not None:
                if _matching_board_candidate_task(self.scheduled_tasks.values(), update, candidate_id=update.subject_id):
                    raise ValueError("board candidate update already exists before resolution")
                self.scheduled_tasks[update.task_id] = update
            if changes:
                self.scheduled_tasks[saved.task_id] = saved
                self.transitions.extend(changes)
            return deepcopy(saved), changes

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
        # 群友自招只登记看板；时间到达不能代替客户授予店方主动邀约权限。
        if (game.requirement.get("recruitment_mode") == "board_only"
                or game.recruitment_status != RecruitmentStatus.SCHEDULED or game.recruitment_opens_at is None):
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
        """原子领取任务；同群看板只允许一个未过期发布租约，避免新板先发后被旧板覆盖。"""

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
            if task.task_type == "publish_group_board" and task.aggregate_type == "group_room":
                if any(other.task_id != task_id and other.task_type == "publish_group_board"
                       and other.aggregate_type == "group_room" and other.aggregate_id == task.aggregate_id
                       and other.status == ScheduledTaskStatus.PROCESSING
                       and other.lease_until is not None and other.lease_until > at
                       for other in self.scheduled_tasks.values()):
                    return None
            task.status = ScheduledTaskStatus.PROCESSING
            task.attempts += 1
            if task.task_type == "conversation_follow_up" and not recoverable:
                task.payload["review_count"] = int(task.payload.get("review_count", 0)) + 1
            next_lease = at + timedelta(seconds=max(1, int(lease_seconds)))
            # 暂停不消耗失败预算，attempts可能回退；保留的时间水位仍保证领取凭证不重用。
            task.lease_until = max(next_lease, task.lease_until + timedelta(microseconds=1)) if task.lease_until else next_lease
            task.updated_at = at
            # 调用方持有领取快照；后续领取不能原地改写旧worker手中的凭证。
            return deepcopy(task)

    def complete_scheduled_task(
        self,
        task_id: str,
        *,
        trace_id: str,
        expected_attempts: int,
        expected_lease_until: datetime,
        at: datetime | None = None,
    ) -> tuple[ScheduledAgentTask | None, StateTransition | None]:
        """原子完成当前领取；租约已被重领、取消或延期时保持新状态不变。"""
        stamp = normalize_datetime(at or now())
        with self._lock:
            task = self.scheduled_tasks.get(task_id)
            if task is None or not task.matches_claim(expected_attempts=expected_attempts, expected_lease_until=expected_lease_until):
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
        expected_attempts: int,
        expected_lease_until: datetime,
        max_attempts: int = 3,
        retry_delay_seconds: int = 60,
        at: datetime | None = None,
    ) -> tuple[ScheduledAgentTask | None, StateTransition | None]:
        """仅把当前领取记为失败或重试，旧异常不能取消新的pending或processing任务。"""
        stamp = normalize_datetime(at or now())
        with self._lock:
            task = self.scheduled_tasks.get(task_id)
            if task is None or not task.matches_claim(expected_attempts=expected_attempts, expected_lease_until=expected_lease_until):
                return task, None
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

    def defer_scheduled_task(
        self, task_id: str, *, trace_id: str, reason: str,
        expected_attempts: int, expected_lease_until: datetime,
        retry_delay_seconds: int = 60, at: datetime | None = None,
    ) -> tuple[ScheduledAgentTask | None, StateTransition | None]:
        """暂停当前领取并保留可恢复意图；权限关闭不计传输失败，也不伪造已发布。"""
        stamp = normalize_datetime(at or now())
        with self._lock:
            task = self.scheduled_tasks.get(task_id)
            if task is None or not task.matches_claim(expected_attempts=expected_attempts, expected_lease_until=expected_lease_until):
                return task, None
            task.status = ScheduledTaskStatus.PENDING
            task.attempts = max(0, task.attempts - 1)
            task.due_at = stamp + timedelta(seconds=max(1, int(retry_delay_seconds)))
            # lease_until仅保留为唯一性水位；pending状态不授予任何worker执行权。
            task.completed_at = None
            task.last_error = str(reason)
            task.updated_at = stamp
            transition = StateTransition("scheduled_agent_task", task.task_id, ScheduledTaskStatus.PROCESSING.value,
                task.status.value, "scheduled_agent_task_deferred", trace_id)
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
