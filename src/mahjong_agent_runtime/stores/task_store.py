"""定义任务存储相关的存储接口，隔离领域层与具体持久化实现。

"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from typing import Any, Iterable, Protocol

from ..models import (
    ConversationRole,
    ConversationTurn,
    Game,
    GameStatus,
    MessageReference,
    PendingInputBatch,
    PendingMemoryCandidate,
    ScheduledAgentTask,
    ScheduledTaskStatus,
    StateTransition,
    TaskMemory,
)

BOARD_CANDIDATE_DECISION_TASK_TYPE = "group_board_candidate_decision"
BOARD_CANDIDATE_UPDATE_TASK_TYPE = "group_board_candidate_update"


def _board_stake_confirmation_same_purpose(existing: ScheduledAgentTask, proposed: ScheduledAgentTask) -> bool:
    """核价按局、报价回告及单独的补位征询按局和相关客户区分。"""
    if (existing.task_type != proposed.task_type or existing.aggregate_type != "game"
            or existing.aggregate_id != proposed.aggregate_id):
        return False
    if proposed.task_type == "group_board_stake_update":
        return existing.subject_id == proposed.subject_id
    is_fit = proposed.payload.get("question_kind") == "candidate_fit_only"
    if (existing.payload.get("question_kind") == "candidate_fit_only") != is_fit:
        return False
    if not is_fit:
        return True
    if existing.payload.get("candidate_id") != proposed.payload.get("candidate_id"):
        return False
    if (existing.subject_id != proposed.subject_id or existing.conversation_id != proposed.conversation_id
            or any(existing.payload.get(key) != proposed.payload.get(key)
                   for key in ("room_id", "candidate_conversation_id", "candidate_gender"))):
        raise ValueError("board candidate question identity conflict")
    return True


def _board_candidate_fit_expiry(task: ScheduledAgentTask, update_task: ScheduledAgentTask | None) -> datetime:
    """验证两方私聊绑定和询局凭证，已知档位的补位问题不能顺带订阅报价。"""
    payload_keys = (
        "candidate_id", "candidate_conversation_id", "inquiry_trace_id", "inquiry_source_message_id", "room_id",
    )
    identifiers = (task.idempotency_key, task.subject_id, task.conversation_id)
    identifiers += tuple(task.payload.get(key) for key in payload_keys)
    if (update_task is not None or not all(isinstance(value, str) and value.strip() for value in identifiers)
            or task.payload.get("candidate_gender") not in ("男", "女")
            or task.payload["candidate_id"] == task.subject_id
            or task.payload["candidate_conversation_id"] == task.conversation_id):
        raise ValueError("board candidate question requires distinct private sessions and inquiry evidence, without a stake update")
    return _board_candidate_expiry(task)


def _validate_new_board_candidate_fit_question(
    task: ScheduledAgentTask, game: Game, *, known_stake: bool, expiry: datetime, at: datetime,
) -> None:
    """在写事务内复核桌况，避免登记已失效或需改问档位的征询。"""
    _check_board_candidate_game(game, host_id=task.subject_id, at=at)
    if not known_stake or at >= expiry or task.due_at.utcoffset() is None or task.due_at >= expiry:
        raise ValueError("board candidate question requires a known stake and a future expiry")


def _matching_board_candidate_task(
    tasks: Iterable[ScheduledAgentTask], proposed: ScheduledAgentTask, *, candidate_id: str | None,
) -> ScheduledAgentTask | None:
    """在事务内按桌与候选人去重，拒绝任务标识或会话归属冲突。"""
    matching = None
    for task in tasks:
        recorded_candidate = (task.payload.get("candidate_id")
                              if task.task_type == BOARD_CANDIDATE_DECISION_TASK_TYPE else task.subject_id)
        same_purpose = (task.task_type == proposed.task_type and task.aggregate_type == "game"
                        and task.aggregate_id == proposed.aggregate_id and recorded_candidate == candidate_id)
        if same_purpose:
            if (task.subject_id != proposed.subject_id or task.conversation_id != proposed.conversation_id
                    or task.payload.get("room_id") != proposed.payload.get("room_id")
                    or task.payload.get("candidate_conversation_id") != proposed.payload.get("candidate_conversation_id")):
                raise ValueError("board candidate task has another conversation or room")
            matching = task
        elif task.task_id == proposed.task_id or task.idempotency_key == proposed.idempotency_key:
            raise ValueError("board candidate task identity conflict")
    return matching


def _board_candidate_expiry(task: ScheduledAgentTask) -> datetime:
    """从持久责任读取带时区的有效期，拒绝无法比较的时间。"""
    try:
        expiry = datetime.fromisoformat(task.payload["expires_at"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("board candidate task requires a timezone-aware expiry") from exc
    if expiry.utcoffset() is None:
        raise ValueError("board candidate task requires a timezone-aware expiry")
    return expiry


def _check_board_candidate_game(game: Game, *, host_id: str, at: datetime) -> None:
    """在写事务内重新确认桌主、公开自招局状态及真实空位。"""
    if (game.organizer_id != host_id or game.requirement.get("recruitment_mode") != "board_only"
            or game.status not in {GameStatus.FORMING, GameStatus.INVITING} or game.remaining_seats() < 1
            or (game.expires_at is not None and game.expires_at <= at)):
        raise ValueError("board candidate decision requires a live board game and its organizer")


def _new_board_candidate_decision(task: ScheduledAgentTask, game: Game, *, at: datetime) -> ScheduledAgentTask:
    """待答责任只在截止时唤醒，登记或重试都不会发出消息。"""
    candidate_id = task.payload.get("candidate_id")
    candidate_conversation = task.payload.get("candidate_conversation_id")
    if (task.task_type != BOARD_CANDIDATE_DECISION_TASK_TYPE or task.aggregate_type != "game"
            or not all((task.task_id, task.idempotency_key, task.subject_id, task.conversation_id,
                        candidate_id, candidate_conversation, task.payload.get("room_id"),
                        task.payload.get("inquiry_trace_id"), task.payload.get("question_message_id")))
            or candidate_id == task.subject_id or candidate_conversation == task.conversation_id
            or task.status != ScheduledTaskStatus.PENDING or task.payload.get("decision")):
        raise ValueError("board candidate decision requires a pending game-scoped task with two private sessions")
    _check_board_candidate_game(game, host_id=task.subject_id, at=at)
    expiry = _board_candidate_expiry(task)
    if game.expires_at is not None:
        expiry = min(expiry, game.expires_at)
    if at >= expiry:
        raise ValueError("board candidate decision has expired")
    return replace(task, due_at=expiry, payload={**task.payload, "expires_at": expiry.isoformat()})


def _board_candidate_resolution(
    task: ScheduledAgentTask, game: Game, *, decision: str, host_id: str, conversation_id: str,
    evidence: ConversationTurn | None, receipt: MessageReference | None, update_task: ScheduledAgentTask | None,
    at: datetime, trace_id: str,
) -> tuple[ScheduledAgentTask, ScheduledAgentTask | None, list[StateTransition]]:
    """校验交付与本人答复并规划原子写入，历史有效答复可由后续一轮提交。"""
    if (task.task_type != BOARD_CANDIDATE_DECISION_TASK_TYPE or task.aggregate_type != "game"
            or task.subject_id != host_id or task.conversation_id != conversation_id
            or decision not in {"accepted", "declined"}):
        raise ValueError("board candidate decision must belong to this organizer session")
    if task.status == ScheduledTaskStatus.COMPLETED and task.payload.get("decision") == decision:
        return task, None, []
    _check_board_candidate_game(game, host_id=host_id, at=at)
    if task.status != ScheduledTaskStatus.PENDING or _board_candidate_expiry(task) <= at:
        raise ValueError("board candidate decision is no longer pending")
    if (receipt is None or receipt.conversation_id != conversation_id or receipt.recipient_id != host_id
            or receipt.business_ref_type != "group_board_stake_confirmation" or receipt.business_ref_id != task.aggregate_id
            or not receipt.metadata.get("delivery_id")
            or receipt.metadata.get("candidate_decision_task_id") != task.task_id):
        raise ValueError("board candidate question was not delivered to this organizer")
    if (evidence is None or evidence.role != ConversationRole.USER or evidence.sender_id != host_id
            or not evidence.metadata.get("source_message_id") or evidence.metadata.get("context_invalidated")
            or evidence.metadata.get("system_trigger") or evidence.metadata.get("internal_event")
            or evidence.metadata.get("self_message") or evidence.metadata.get("source") == "group"
            or evidence.metadata.get("is_room") or evidence.metadata.get("room_id")
            or evidence.metadata.get("input_source") == "system_trigger"
            or evidence.occurred_at.utcoffset() is None or not receipt.created_at <= evidence.occurred_at <= at):
        raise ValueError("board candidate decision requires an unrecalled organizer reply after delivery")
    if (decision == "accepted") != (update_task is not None):
        raise ValueError("only an accepted candidate decision creates a candidate update")
    saved = replace(task, status=ScheduledTaskStatus.COMPLETED, completed_at=at, updated_at=at,
        payload={**task.payload, "decision": decision, "evidence_message_id": evidence.metadata["source_message_id"],
                 "evidence_trace_id": evidence.trace_id, "question_delivery_id": receipt.metadata["delivery_id"]})
    changes = [StateTransition("scheduled_agent_task", saved.task_id, task.status.value,
                               saved.status.value, f"board_candidate_{decision}", trace_id)]
    update = None
    if update_task is not None:
        if (update_task.task_type != BOARD_CANDIDATE_UPDATE_TASK_TYPE or update_task.aggregate_type != "game"
                or update_task.aggregate_id != task.aggregate_id or not update_task.task_id or not update_task.idempotency_key
                or update_task.subject_id != task.payload["candidate_id"]
                or update_task.conversation_id != task.payload["candidate_conversation_id"]
                or update_task.status != ScheduledTaskStatus.PENDING
                or any(update_task.payload.get(key) != task.payload[key]
                       for key in ("room_id", "inquiry_trace_id", "expires_at"))
                or update_task.payload.get("decision_task_id") != task.task_id):
            raise ValueError("candidate update must belong to the original candidate and inquiry")
        update = replace(update_task, due_at=at, updated_at=at, payload={**update_task.payload, "decision": decision})
        changes.append(StateTransition("scheduled_agent_task", update.task_id, None,
                                       update.status.value, "board_candidate_update_ready", trace_id))
    return saved, update, changes


class TaskStore(Protocol):
    """定义任务存储需要遵守的抽象接口契约。

"""

    @property
    def task_memories(self) -> dict[str, TaskMemory]:
        ...

    @property
    def pending_memory_candidates(self) -> dict[str, PendingMemoryCandidate]:
        ...

    @property
    def pending_input_batches(self) -> dict[str, PendingInputBatch]:
        ...

    @property
    def scheduled_tasks(self) -> dict[str, ScheduledAgentTask]:
        ...

    @property
    def badcases(self) -> list[dict[str, Any]]:
        ...

    def record_task_memory(self, **kwargs: Any) -> tuple[TaskMemory, StateTransition]:
        """记录任务记忆，形成后续决策和审计所需的持久证据。"""
        ...

    def record_pending_memory_candidate(
        self,
        **kwargs: Any,
    ) -> tuple[PendingMemoryCandidate, StateTransition]:
        """记录pending记忆候选人，形成后续决策和审计所需的持久证据。"""
        ...

    def task_memory_context(
        self,
        conversation_id: str,
        customer_id: str | None = None,
    ) -> list[dict[str, Any]]:
        ...

    def pending_memory_candidates_for_context(
        self,
        conversation_id: str,
        customer_id: str | None = None,
    ) -> list[dict[str, Any]]:
        ...

    def task_memory_excluded_customer_ids(
        self,
        conversation_id: str,
        customer_id: str | None = None,
    ) -> list[str]:
        ...

    def save_follow_up_task(self, task: ScheduledAgentTask, *, trace_id: str) -> list[StateTransition]:
        """原子替换同一目标的复查计划，并记录状态迁移。"""
        ...

    def ensure_board_stake_confirmation_task(
        self, task: ScheduledAgentTask, *, trace_id: str,
        update_task: ScheduledAgentTask | None = None,
    ) -> tuple[ScheduledAgentTask, list[StateTransition]]:
        """按桌登记核价及原子回告，或按桌和候选人登记单独补位征询；重试保留既有责任。"""
        ...

    def ensure_board_candidate_decision_task(
        self, task: ScheduledAgentTask, *, trace_id: str,
    ) -> tuple[ScheduledAgentTask, list[StateTransition]]:
        """登记同一桌和候选人的待答责任，保留初始询局与桌主沟通引用。"""
        ...

    def resolve_board_candidate_decision(
        self, task_id: str, *, host_id: str, conversation_id: str, decision: str,
        evidence_message_id: str, trace_id: str, update_task: ScheduledAgentTask | None = None,
        at: datetime | None = None,
    ) -> tuple[ScheduledAgentTask, list[StateTransition]]:
        """凭真实桌主答复完成责任并原子唤醒候选人回告；不改变任何席位。"""
        ...

    def save_participation_deadline(self, **kwargs: Any) -> tuple[ScheduledAgentTask, list[StateTransition]]:
        """验证真实来源和本人席位，原子替换同一牌局的个人截止责任。"""
        ...

    def clear_participation_deadline(self, **kwargs: Any) -> list[StateTransition]:
        """撤销本人的条件责任，保留原参与状态及历史证据。"""
        ...

    def execute_participation_deadline(self, claim: ScheduledAgentTask, *, trace_id: str, at: datetime | None = None) -> tuple[dict, list[StateTransition]]:
        """在人数事务内检查租约和成局状态，未成局仅释放本人独立席位。"""
        ...

    def scheduled_task_for_game(self, game_id: str) -> ScheduledAgentTask | None:
        ...

    def ensure_game_recruitment_task(
        self,
        game_id: str,
        *,
        trace_id: str,
    ) -> tuple[ScheduledAgentTask | None, StateTransition | None]:
        """确保牌局recruitment任务满足后续步骤依赖的不变量。"""
        ...

    def ensure_waiting_demand_expiration_task(
        self,
        *,
        due_at: datetime,
        trace_id: str,
    ) -> tuple[ScheduledAgentTask, StateTransition | None]:
        """确保候补需求expiration任务满足后续步骤依赖的不变量。"""
        ...

    def due_scheduled_tasks(self, *, at: datetime, limit: int = 100) -> list[ScheduledAgentTask]:
        ...

    def open_game_recruitment(
        self,
        game_id: str,
        *,
        trace_id: str,
        at: datetime | None = None,
    ) -> tuple[Any, list[StateTransition]]:
        ...

    def claim_scheduled_task(
        self,
        task_id: str,
        *,
        at: datetime,
        lease_seconds: int = 120,
    ) -> ScheduledAgentTask | None:
        """原子领取定时任务，防止并发消费者重复处理。"""
        ...

    def complete_scheduled_task(
        self,
        task_id: str,
        *,
        trace_id: str,
        expected_attempts: int,
        expected_lease_until: datetime,
        at: datetime | None = None,
    ) -> tuple[ScheduledAgentTask | None, StateTransition | None]:
        """按当前领取双凭证完成任务，失效领取保持后续任务状态不变。"""
        ...

    def fail_scheduled_task(
        self,
        task_id: str,
        *,
        error: str,
        trace_id: str,
        expected_attempts: int,
        expected_lease_until: datetime,
        max_attempts: int = 3,
        retry_delay_seconds: int = 60,
        at: datetime | None = None,
    ) -> tuple[ScheduledAgentTask | None, StateTransition | None]:
        """按领取凭证耗用有限失败预算；旧worker不能使新代失败或重新排队。"""
        ...

    def defer_scheduled_task(
        self, task_id: str, *, trace_id: str, reason: str,
        expected_attempts: int, expected_lease_until: datetime,
        retry_delay_seconds: int = 60, at: datetime | None = None,
    ) -> tuple[ScheduledAgentTask | None, StateTransition | None]:
        """不耗用失败预算地暂停当前领取，保留持久任务意图以待权限或通道恢复。"""
        ...

    def upsert_pending_input_fragment(self, **kwargs: Any) -> PendingInputBatch:
        """新增或更新pending输入消息片段，使重复写入收敛到同一条记录。"""
        ...

    def pending_input_batch(
        self,
        conversation_id: str,
        sender_id: str,
    ) -> PendingInputBatch | None:
        ...

    def due_pending_input_batches(self, *, at: datetime, limit: int = 100) -> list[PendingInputBatch]:
        ...

    def claim_pending_input_batch(self, **kwargs: Any) -> PendingInputBatch | None:
        """原子领取pending输入批次，防止并发消费者重复处理。"""
        ...

    def record_pending_input_decision(self, **kwargs: Any) -> PendingInputBatch:
        """记录pending输入决策，形成后续决策和审计所需的持久证据。"""
        ...

    def queue_pending_input_batch(self, **kwargs: Any) -> PendingInputBatch:
        ...

    def finish_pending_input_batch(self, **kwargs: Any) -> PendingInputBatch:
        ...

    def record_badcase(
        self,
        payload: dict[str, Any],
        *,
        trace_id: str,
        conversation_id: str,
    ) -> dict[str, Any]:
        """记录badcase，形成后续决策和审计所需的持久证据。"""
        ...

    def clear_runtime_state(
        self,
        *,
        conversation_id: str | None = None,
        include_customers: bool = False,
        include_badcases: bool = False,
    ) -> dict[str, int]:
        """清理运行时状态，用于显式重置受控状态。"""
        ...
