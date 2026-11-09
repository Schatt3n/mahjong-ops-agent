"""实现 SQLite 存储层的scheduling数据访问与事务操作。

"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta

from ...models import (
    DEFAULT_TZ,
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
from .serialization import (
    _loads,
    _scheduled_agent_task_from_payload,
)


class SQLiteSchedulingStoreMixin:
    """从兼容存储中提取的后端特定操作。"""

    __slots__ = ()

    def active_games(self, conversation_id: str | None = None) -> list[Game]:
        self._expire_stale_games(trace_id="system_lifecycle")
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
        task_id = game_recruitment_task_id(game_id)
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM runtime_scheduled_agent_tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            return _scheduled_agent_task_from_payload(_loads(row["payload"])) if row else None

    def ensure_board_stake_confirmation_task(
        self, task: ScheduledAgentTask, *, trace_id: str,
        update_task: ScheduledAgentTask | None = None,
    ) -> tuple[ScheduledAgentTask, list[StateTransition]]:
        """原子登记一次桌主询问及每位询局者回告，跨连接重试不改既有任务。

        核价按局去重，补位征询按同局同候选人去重。登记前在写事务内重读档位；若桌主已抢先确认，
        报价回告立即到期且不新排问价。补位征询独立排队，收到桌主有效答复后才会登记该候选人回告。
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
        with self._write_transaction():
            stamp = now()
            row = self._connection.execute("SELECT payload FROM runtime_games WHERE game_id = ?",
                                           (task.aggregate_id,)).fetchone()
            known = bool(row is not None and stake_options(_loads(row["payload"]).get("requirement", {})))
            selected, insertions, transitions = [], [], []
            for proposed in [task] + ([update_task] if update_task is not None else []):
                rows = self._connection.execute(
                    "SELECT payload FROM runtime_scheduled_agent_tasks "
                    "WHERE task_id = ? OR idempotency_key = ? OR (task_type = ? AND aggregate_id = ?) "
                    "ORDER BY task_id",
                    (proposed.task_id, proposed.idempotency_key, proposed.task_type, proposed.aggregate_id),
                ).fetchall()
                existing_tasks = [_scheduled_agent_task_from_payload(_loads(row["payload"])) for row in rows]
                is_update = proposed.task_type == "group_board_stake_update"
                matching = []
                for existing in [*existing_tasks, *insertions]:
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
                # 档位与候选人接受性是不同问题，不能以报价已知替代桌主同意。
                elif known and saved.payload.get("question_kind") != "candidate_fit_only":
                    saved.status = ScheduledTaskStatus.COMPLETED
                    saved.completed_at = saved.updated_at = stamp
                    reason = "board_stake_confirmation_already_known"
                selected.append(saved)
                insertions.append(saved)
                transitions.append(StateTransition("scheduled_agent_task", saved.task_id, None,
                    saved.status.value, reason, trace_id))
            for saved, transition in zip(insertions, transitions):
                self._save_scheduled_agent_task(saved)
                self._append_transition(transition)
            return selected[0], transitions

    def _activate_board_stake_updates_in_transaction(self, game: Game, *, trace_id: str, at: datetime) -> None:
        """随牌局档位首次确认原子唤醒回告；不另开事务，不改已领取或终态任务。"""
        rows = self._connection.execute(
            "SELECT payload FROM runtime_scheduled_agent_tasks WHERE task_type = ? AND aggregate_id = ? AND status = ?",
            ("group_board_stake_update", game.game_id, ScheduledTaskStatus.PENDING.value),
        ).fetchall()
        for row in rows:
            task = _scheduled_agent_task_from_payload(_loads(row["payload"]))
            if task.aggregate_type != "game":
                continue
            try:
                expiry = datetime.fromisoformat(task.payload["expires_at"])
            except (KeyError, TypeError, ValueError):
                continue
            if expiry.tzinfo is None or expiry <= at or task.due_at <= at:
                continue
            task.due_at = task.updated_at = at
            self._save_scheduled_agent_task(task)
            self._append_transition(StateTransition("scheduled_agent_task", task.task_id,
                task.status.value, task.status.value, "board_stake_update_ready", trace_id))

    def _board_candidate_task_candidates(self, proposed: ScheduledAgentTask) -> list[ScheduledAgentTask]:
        """按任务身份与聚合读取可能重复或冲突的持久责任，不扫描其他牌局。"""
        rows = self._connection.execute(
            "SELECT payload FROM runtime_scheduled_agent_tasks "
            "WHERE task_id = ? OR idempotency_key = ? OR (task_type = ? AND aggregate_id = ?)",
            (proposed.task_id, proposed.idempotency_key, proposed.task_type, proposed.aggregate_id),
        ).fetchall()
        return [_scheduled_agent_task_from_payload(_loads(row["payload"])) for row in rows]

    def ensure_board_candidate_decision_task(
        self, task: ScheduledAgentTask, *, trace_id: str,
    ) -> tuple[ScheduledAgentTask, list[StateTransition]]:
        """以写事务登记桌主待答责任，跨连接重复登记不会刷新有效期或另建候选。"""
        if task.task_type != BOARD_CANDIDATE_DECISION_TASK_TYPE or task.aggregate_type != "game":
            raise ValueError("board candidate decision requires its game-scoped task type")
        with self._write_transaction():
            existing = _matching_board_candidate_task(self._board_candidate_task_candidates(task), task,
                                                       candidate_id=task.payload.get("candidate_id"))
            if existing is not None:
                return existing, []
            saved = _new_board_candidate_decision(task, self.require_game(task.aggregate_id), at=now())
            transition = StateTransition("scheduled_agent_task", saved.task_id, None,
                                         saved.status.value, "board_candidate_waiting_for_host", trace_id)
            self._save_scheduled_agent_task(saved)
            self._append_transition(transition)
            return saved, [transition]

    def resolve_board_candidate_decision(
        self, task_id: str, *, host_id: str, conversation_id: str, decision: str,
        evidence_message_id: str, trace_id: str, update_task: ScheduledAgentTask | None = None,
        at: datetime | None = None,
    ) -> tuple[ScheduledAgentTask, list[StateTransition]]:
        """在同一写事务核验答复和收据、完成待答并登记候选人回告及迁移。"""
        with self._write_transaction():
            row = self._connection.execute("SELECT payload FROM runtime_scheduled_agent_tasks WHERE task_id = ?",
                                           (task_id,)).fetchone()
            if row is None:
                raise ValueError("board candidate decision task not found")
            task = _scheduled_agent_task_from_payload(_loads(row["payload"]))
            game = self.require_game(task.aggregate_id)
            receipt = self.resolve_message_reference(conversation_id=conversation_id,
                                                     message_id=task.payload.get("question_message_id", ""))
            evidence = self.find_conversation_turn(conversation_id, message_id=evidence_message_id)
            stamp = at or now()
            if stamp.utcoffset() is None:
                stamp = stamp.replace(tzinfo=DEFAULT_TZ)
            saved, update, changes = _board_candidate_resolution(task, game, decision=decision, host_id=host_id,
                conversation_id=conversation_id, evidence=evidence, receipt=receipt, update_task=update_task,
                at=stamp, trace_id=trace_id)
            if update is not None:
                if _matching_board_candidate_task(self._board_candidate_task_candidates(update), update,
                                                  candidate_id=update.subject_id):
                    raise ValueError("board candidate update already exists before resolution")
                self._save_scheduled_agent_task(update)
            if changes:
                self._save_scheduled_agent_task(saved)
                for transition in changes:
                    self._append_transition(transition)
            return saved, changes

    def ensure_game_recruitment_task(
        self,
        game_id: str,
        *,
        trace_id: str,
    ) -> tuple[ScheduledAgentTask | None, StateTransition | None]:
        """确保牌局recruitment任务满足后续步骤依赖的不变量。"""
        with self._write_transaction():
            row = self._connection.execute(
                "SELECT payload FROM runtime_games WHERE game_id = ?",
                (game_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"game not found: {game_id}")
            game = self._hydrate_game(_loads(row["payload"]))
            task, transition = self._sync_game_recruitment_task_in_transaction(game, trace_id=trace_id)
            self._save_game(game)
            if transition is not None:
                self._append_transition(transition)
            return task, transition

    def ensure_waiting_demand_expiration_task(
        self,
        *,
        due_at: datetime,
        trace_id: str,
    ) -> tuple[ScheduledAgentTask, StateTransition | None]:
        """确保候补需求expiration任务满足后续步骤依赖的不变量。

"""

        task_id = waiting_expiry_task_id(due_at)
        with self._write_transaction():
            row = self._connection.execute(
                "SELECT payload FROM runtime_scheduled_agent_tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if row is not None:
                return _scheduled_agent_task_from_payload(_loads(row["payload"])), None
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
            self._save_scheduled_agent_task(task)
            transition = StateTransition(
                "scheduled_agent_task",
                task_id,
                None,
                task.status.value,
                "waiting_demand_expiration_scheduled",
                trace_id,
            )
            self._append_transition(transition)
            return task, transition

    def _sync_game_recruitment_task_in_transaction(
        self,
        game: Game,
        *,
        trace_id: str,
    ) -> tuple[ScheduledAgentTask | None, StateTransition | None]:
        apply_game_recruitment_policy(game)
        task_id = game_recruitment_task_id(game.game_id)
        row = self._connection.execute(
            "SELECT payload FROM runtime_scheduled_agent_tasks WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        existing = _scheduled_agent_task_from_payload(_loads(row["payload"])) if row else None
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
            self._save_scheduled_agent_task(existing)
            return existing, StateTransition(
                "scheduled_agent_task",
                existing.task_id,
                old,
                existing.status.value,
                "recruitment_no_longer_scheduled",
                trace_id,
            )

        task_context = self.current_task_context(game.conversation_id, game.organizer_id)
        payload = {
            "event_type": "game_recruitment_window_opened",
            "game_id": game.game_id,
            "planned_start_at": game.planned_start_at.isoformat() if game.planned_start_at else None,
            "recruitment_opens_at": game.recruitment_opens_at.isoformat(),
            "task_context_id": task_context.task_context_id if task_context else None,
        }
        idempotency_key = f"{GAME_RECRUITMENT_TASK_TYPE}:{game.game_id}:{game.recruitment_opens_at.isoformat()}"
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
                idempotency_key=idempotency_key,
                payload=payload,
            )
            self._save_scheduled_agent_task(task)
            return task, StateTransition(
                "scheduled_agent_task",
                task.task_id,
                None,
                task.status.value,
                "future_game_created",
                trace_id,
            )

        old = existing.status.value
        existing.due_at = game.recruitment_opens_at
        existing.idempotency_key = idempotency_key
        existing.payload = payload
        existing.status = ScheduledTaskStatus.PENDING
        existing.lease_until = None
        existing.completed_at = None
        existing.last_error = ""
        existing.updated_at = now()
        self._save_scheduled_agent_task(existing)
        return existing, StateTransition(
            "scheduled_agent_task",
            existing.task_id,
            old,
            existing.status.value,
            "future_game_schedule_updated",
            trace_id,
        )

    def due_scheduled_tasks(self, *, at: datetime, limit: int = 100) -> list[ScheduledAgentTask]:
        at = at if at.tzinfo is not None else at.replace(tzinfo=DEFAULT_TZ)
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT payload FROM runtime_scheduled_agent_tasks
                WHERE (status = ? AND due_at <= ?)
                   OR (status = ? AND lease_until IS NOT NULL AND lease_until <= ?)
                ORDER BY due_at, task_id
                LIMIT ?
                """,
                (
                    ScheduledTaskStatus.PENDING.value,
                    at.isoformat(),
                    ScheduledTaskStatus.PROCESSING.value,
                    at.isoformat(),
                    int(limit),
                ),
            ).fetchall()
            return [_scheduled_agent_task_from_payload(_loads(row["payload"])) for row in rows]

    def open_game_recruitment(
        self,
        game_id: str,
        *,
        trace_id: str,
        at: datetime | None = None,
    ) -> tuple[Game, StateTransition | None]:
        stamp = at or now()
        with self._write_transaction():
            row = self._connection.execute(
                "SELECT payload FROM runtime_games WHERE game_id = ?",
                (game_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"game not found: {game_id}")
            game = self._hydrate_game(_loads(row["payload"]))
            old = game.recruitment_status.value
            apply_game_recruitment_policy(game, at=stamp)
            if game.recruitment_status == RecruitmentStatus.SCHEDULED:
                raise ValueError(
                    "recruitment window is not open: "
                    f"recruitment_opens_at={game.recruitment_opens_at.isoformat() if game.recruitment_opens_at else None}"
                )
            if old == game.recruitment_status.value:
                self._save_game(game)
                return game, None
            game.updated_at = stamp
            self._save_game(game)
            transition = StateTransition(
                "game_recruitment",
                game.game_id,
                old,
                game.recruitment_status.value,
                "scheduled_recruitment_window_opened",
                trace_id,
            )
            self._append_transition(transition)
            return game, transition

    def claim_scheduled_task(
        self,
        task_id: str,
        *,
        at: datetime,
        lease_seconds: int = SCHEDULED_TASK_PROCESSING_LEASE_SECONDS,
    ) -> ScheduledAgentTask | None:
        """同事务领取并串行化同群看板租约，防止多连接先发新板再补发旧板。"""
        at = at if at.tzinfo is not None else at.replace(tzinfo=DEFAULT_TZ)
        with self._write_transaction():
            row = self._connection.execute(
                "SELECT payload FROM runtime_scheduled_agent_tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if row is None:
                return None
            task = _scheduled_agent_task_from_payload(_loads(row["payload"]))
            if task.due_at > at:
                return None
            recoverable = (
                task.status == ScheduledTaskStatus.PROCESSING
                and task.lease_until is not None
                and task.lease_until <= at
            )
            if task.status != ScheduledTaskStatus.PENDING and not recoverable:
                return None
            if task.task_type == "publish_group_board" and task.aggregate_type == "group_room":
                rows = self._connection.execute(
                    "SELECT payload FROM runtime_scheduled_agent_tasks "
                    "WHERE task_type = ? AND aggregate_id = ? AND status = ? AND task_id != ?",
                    ("publish_group_board", task.aggregate_id, ScheduledTaskStatus.PROCESSING.value, task_id),
                ).fetchall()
                for other_row in rows:
                    other = _scheduled_agent_task_from_payload(_loads(other_row["payload"]))
                    if other.aggregate_type == "group_room" and other.lease_until is not None and other.lease_until > at:
                        return None
            task.status = ScheduledTaskStatus.PROCESSING
            task.attempts += 1
            if task.task_type == "conversation_follow_up" and not recoverable:
                task.payload["review_count"] = int(task.payload.get("review_count", 0)) + 1
            next_lease = at + timedelta(seconds=max(1, int(lease_seconds)))
            # 暂停会归还失败预算，故领取时间水位也必须前进，不能复用旧worker的双凭证。
            task.lease_until = max(next_lease, task.lease_until + timedelta(microseconds=1)) if task.lease_until else next_lease
            task.updated_at = at
            self._save_scheduled_agent_task(task)
            return task

    def complete_scheduled_task(
        self,
        task_id: str,
        *,
        trace_id: str,
        expected_attempts: int,
        expected_lease_until: datetime,
        at: datetime | None = None,
    ) -> tuple[ScheduledAgentTask | None, StateTransition | None]:
        """在写事务内比较领取凭证，拒绝旧worker完成后来重领或重新排队的任务。"""
        stamp = at or now()
        with self._write_transaction():
            row = self._connection.execute(
                "SELECT payload FROM runtime_scheduled_agent_tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if row is None:
                return None, None
            task = _scheduled_agent_task_from_payload(_loads(row["payload"]))
            if not task.matches_claim(expected_attempts=expected_attempts, expected_lease_until=expected_lease_until):
                return task, None
            old = task.status.value
            task.status = ScheduledTaskStatus.COMPLETED
            task.completed_at = stamp
            task.lease_until = None
            task.updated_at = stamp
            self._save_scheduled_agent_task(task)
            transition = StateTransition(
                "scheduled_agent_task",
                task.task_id,
                old,
                task.status.value,
                "scheduled_agent_task_completed",
                trace_id,
            )
            self._append_transition(transition)
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
        """只有当前领取可耗用重试预算，旧worker的异常不能覆盖新代任务状态。"""
        stamp = at or now()
        with self._write_transaction():
            row = self._connection.execute(
                "SELECT payload FROM runtime_scheduled_agent_tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if row is None:
                return None, None
            task = _scheduled_agent_task_from_payload(_loads(row["payload"]))
            if not task.matches_claim(expected_attempts=expected_attempts, expected_lease_until=expected_lease_until):
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
            self._save_scheduled_agent_task(task)
            transition = StateTransition(
                "scheduled_agent_task",
                task.task_id,
                old,
                task.status.value,
                "scheduled_agent_task_failed",
                trace_id,
            )
            self._append_transition(transition)
            return task, transition

    def defer_scheduled_task(
        self, task_id: str, *, trace_id: str, reason: str,
        expected_attempts: int, expected_lease_until: datetime,
        retry_delay_seconds: int = 60, at: datetime | None = None,
    ) -> tuple[ScheduledAgentTask | None, StateTransition | None]:
        """原子保留暂停后的发布意图，归还本次失败预算并保留领取唯一性时间水位。"""
        stamp = at or now()
        with self._write_transaction():
            row = self._connection.execute(
                "SELECT payload FROM runtime_scheduled_agent_tasks WHERE task_id = ?", (task_id,),
            ).fetchone()
            if row is None:
                return None, None
            task = _scheduled_agent_task_from_payload(_loads(row["payload"]))
            if not task.matches_claim(expected_attempts=expected_attempts, expected_lease_until=expected_lease_until):
                return task, None
            task.status = ScheduledTaskStatus.PENDING
            task.attempts = max(0, task.attempts - 1)
            task.due_at = stamp + timedelta(seconds=max(1, int(retry_delay_seconds)))
            # 保留lease_until作下一次领取的水位；只有processing状态代表活租约。
            task.completed_at = None
            task.last_error = str(reason)
            task.updated_at = stamp
            self._save_scheduled_agent_task(task)
            transition = StateTransition("scheduled_agent_task", task.task_id, ScheduledTaskStatus.PROCESSING.value,
                task.status.value, "scheduled_agent_task_deferred", trace_id)
            self._append_transition(transition)
            return task, transition

    def save_follow_up_task(self, task: ScheduledAgentTask, *, trace_id: str) -> list[StateTransition]:
        """事务内替换同一目标的复查；取消与新建一并提交，防止重复或丢失唤醒。"""
        with self._write_transaction():
            transitions = []
            for existing in self.scheduled_tasks.values():
                if existing.task_type == "conversation_follow_up" and existing.aggregate_id == task.aggregate_id and existing.conversation_id == task.conversation_id and existing.subject_id == task.subject_id and existing.status in {ScheduledTaskStatus.PENDING, ScheduledTaskStatus.PROCESSING}:
                    old = existing.status.value
                    existing.status = ScheduledTaskStatus.CANCELLED
                    existing.completed_at = now()
                    self._save_scheduled_agent_task(existing)
                    transitions.append(StateTransition("scheduled_agent_task", existing.task_id, old, "cancelled", "follow_up_replanned", trace_id))
            self._save_scheduled_agent_task(task)
            transitions.append(StateTransition("scheduled_agent_task", task.task_id, None, task.status.value, "follow_up_planned", trace_id))
            for transition in transitions:
                self._append_transition(transition)
            return transitions
