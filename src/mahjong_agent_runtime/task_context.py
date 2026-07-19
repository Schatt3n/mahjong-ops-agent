from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from .models import ConversationTaskContext, Game, GameStatus, StateTransition, UserMessage


ACTIVE_GAME_STATUSES = {GameStatus.FORMING, GameStatus.INVITING, GameStatus.READY}
TERMINAL_GAME_STATUSES = {GameStatus.CANCELLED, GameStatus.FINISHED}


@dataclass(slots=True)
class TaskContextPreparation:
    context: ConversationTaskContext
    reset_applied: bool
    reason: str
    transitions: list[StateTransition] = field(default_factory=list)
    related_active_game_ids: list[str] = field(default_factory=list)
    related_terminal_game_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
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
    """Resolve the current business episode before the model context is built."""

    store: Any
    idle_reset_seconds: int = 4 * 60 * 60

    def prepare(self, message: UserMessage, *, trace_id: str) -> TaskContextPreparation:
        # active_games also applies lifecycle expiry before we inspect terminal history.
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

    @staticmethod
    def _related(game: Game, message: UserMessage) -> bool:
        return bool(
            game.conversation_id == message.conversation_id
            or game.organizer_id == message.sender_id
            or any(participant.customer_id == message.sender_id for participant in game.participants)
        )
