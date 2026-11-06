"""划分同一渠道中的独立业务任务，并为新消息选择或恢复正确的任务上下文。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from .models import ConversationTaskContext, Game, GameStatus, StateTransition, UserMessage


ACTIVE_GAME_STATUSES = {GameStatus.FORMING, GameStatus.INVITING, GameStatus.READY}
TERMINAL_GAME_STATUSES = {GameStatus.CANCELLED, GameStatus.FINISHED}


@dataclass(slots=True)
class TaskContextPreparation:
    """任务上下文选择结果，说明本轮是新建、恢复还是引用历史任务。"""

    context: ConversationTaskContext
    reset_applied: bool
    reason: str
    transitions: list[StateTransition] = field(default_factory=list)
    related_active_game_ids: list[str] = field(default_factory=list)
    related_terminal_game_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """转换为只包含可序列化字段的字典表示。"""
        return {
            "task_context": self.context.to_dict(),
            "reset_applied": self.reset_applied,
            "reason": self.reason,
            "related_active_game_ids": list(self.related_active_game_ids),
            "related_terminal_game_ids": list(self.related_terminal_game_ids),
            "transitions": [item.to_dict() for item in self.transitions],
        }


@dataclass(slots=True)
class TaskContextManager:
    """在首次模型步骤前选择或新建当前任务边界。

    上游：``loop_support.prepare_turn``。它依据活跃/终态牌局、空闲时长和当前任务记录
    做生命周期选择，不负责理解完整用户目标；旧任务的显式引用恢复由 ContextBuilder
    中的 ``recover_referenced_task_contexts`` 单独处理。
    """

    store: Any
    idle_reset_seconds: int = 4 * 60 * 60

    def prepare(self, message: UserMessage, *, trace_id: str) -> TaskContextPreparation:
        """先核验显式话题来源，再处理普通会话空闲周期，不由关键词选择业务目标。"""
        topic = self._prepare_topic(message, trace_id=trace_id)
        if topic is not None:
            return topic
        # active_games 会先执行生命周期过期处理，再由此处检查终态历史。
        active_games = [game for game in self.store.active_games() if self._related(game, message)]
        all_related_games = [game for game in self.store.games.values() if self._related(game, message)]
        terminal_games = [game for game in all_related_games if game.status in TERMINAL_GAME_STATUSES]
        terminal_games.sort(key=lambda game: game.updated_at)
        current = self.store.current_task_context(message.conversation_id, message.sender_id)
        previous_turns = self.store.recent_turns(message.conversation_id, 60)

        should_reset = False
        archive_previous = False
        reason = "continue_current_task"
        started_at = current.started_at if current is not None else message.sent_at

        if current is None:
            latest_turn = previous_turns[-1] if previous_turns else None
            idle_gap = message.sent_at - latest_turn.occurred_at if latest_turn else None
            if terminal_games and not active_games:
                should_reset = True
                archive_previous = True
                reason = "previous_related_game_terminal"
            elif idle_gap is not None and idle_gap >= timedelta(seconds=self.idle_reset_seconds) and not active_games:
                should_reset = True
                archive_previous = True
                reason = "idle_task_timeout"
            elif active_games:
                timestamps = [game.created_at for game in active_games]
                timestamps.extend(turn.occurred_at for turn in previous_turns)
                started_at = min(timestamps) if timestamps else message.sent_at
                reason = "recover_active_task"
            elif previous_turns:
                started_at = previous_turns[0].occurred_at
                reason = "recover_recent_task"
            else:
                reason = "first_message"
        else:
            terminal_after_start = [game for game in terminal_games if game.updated_at >= current.started_at]
            idle_gap = message.sent_at - current.updated_at
            if terminal_after_start and not active_games:
                should_reset = True
                archive_previous = True
                started_at = message.sent_at
                reason = "previous_related_game_terminal"
            elif idle_gap >= timedelta(seconds=self.idle_reset_seconds) and not active_games:
                should_reset = True
                archive_previous = True
                started_at = message.sent_at
                reason = "idle_task_timeout"

        context, transitions = self.store.activate_task_context(
            conversation_id=message.conversation_id,
            customer_id=message.sender_id,
            trace_id=trace_id,
            activity_at=message.sent_at,
            started_at=started_at,
            reason=reason,
            force_new=current is None or should_reset,
            archive_previous=archive_previous,
        )
        return TaskContextPreparation(
            context=context,
            reset_applied=should_reset or (current is None and archive_previous),
            reason=reason,
            transitions=transitions,
            related_active_game_ids=sorted(game.game_id for game in active_games),
            related_terminal_game_ids=sorted(game.game_id for game in terminal_games),
        )

    def _prepare_topic(self, message: UserMessage, *, trace_id: str):
        """可信话题来源恢复本人任务；新目标隔离旧上下文，但不取消旧业务或归档证据。"""
        from .group_chat.topic_context import verified_topic_message, selected_topic_binding
        from .group_chat.tool_scope import unresolved_topic_target
        record = verified_topic_message(self.store, message)
        if record is None:
            return None
        if not record.get("topic_id"):
            if not unresolved_topic_target(self.store,record):
                return None
            # 没有可验证目标的修改/取消不能接续恰好活跃的旧需求。隔离澄清不取消旧局，
            # 工具层同时保持只读；消息接收回执确保重投不会重复创建此隔离任务。
            current = self.store.current_task_context(message.conversation_id,message.sender_id)
            context,transitions = self.store.activate_task_context(
                conversation_id=message.conversation_id,customer_id=message.sender_id,
                trace_id=trace_id,activity_at=message.sent_at,started_at=message.sent_at,
                reason="unresolved_group_topic_target",force_new=True,archive_previous=False,
            )
            return TaskContextPreparation(context=context,reset_applied=current is not None,
                reason="unresolved_group_topic_target",transitions=transitions)
        binding = selected_topic_binding(self.store, record, message.conversation_id)
        current = self.store.current_task_context(message.conversation_id, message.sender_id)
        if binding:
            context, transitions = self.store.resume_task_context(
                task_context_id=binding["task_context_id"], conversation_id=message.conversation_id,
                customer_id=message.sender_id, trace_id=trace_id, activity_at=message.sent_at,
            )
            reason = "resume_group_topic_task"
        else:
            context, transitions = self.store.activate_task_context(
                conversation_id=message.conversation_id, customer_id=message.sender_id,
                trace_id=trace_id, activity_at=message.sent_at, started_at=message.sent_at,
                reason="new_group_topic_task", force_new=True, archive_previous=False,
            )
            reason = "new_group_topic_task"
        self.store.bind_group_topic_customer(topic_id=record["topic_id"], customer_id=message.sender_id,
            conversation_id=message.conversation_id, task_context_id=context.task_context_id,
            game_ids=binding["game_ids"] if binding else (), source_message_id=message.message_id,
            allow_conversation_change=True, at=message.sent_at)
        return TaskContextPreparation(context=context, reset_applied=bool(current and current.task_context_id != context.task_context_id),
            reason=reason, transitions=transitions, related_active_game_ids=binding["game_ids"] if binding else [])

    @staticmethod
    def _related(game: Game, message: UserMessage) -> bool:
        return bool(
            game.conversation_id == message.conversation_id
            or game.organizer_id == message.sender_id
            or any(participant.customer_id == message.sender_id for participant in game.participants)
        )
