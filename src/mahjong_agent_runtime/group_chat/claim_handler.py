"""实现群聊场景中的claim处理器能力。

"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from ..domains import game_commitment_windows_overlap
from ..models import GameStatus, now
from ..stores import AgentStore
from .board_engine import BoardEngine
from .models import ClaimResult, GroupMessage
from .notify_dispatcher import NotifyDispatcher


class ClaimHandler:
    """原子处理看板席位认领，并在成功后触发通知与看板更新。"""

    def __init__(
        self,
        *,
        board_engine: BoardEngine,
        store: AgentStore,
        notify_dispatcher: NotifyDispatcher,
        clock: Callable[[], datetime] = now,
    ) -> None:
        self.board_engine = board_engine
        self.store = store
        self.notify_dispatcher = notify_dispatcher
        self.clock = clock

    def process_claim(
        self,
        msg: GroupMessage,
        item_no: int,
        *,
        trace_id: str,
    ) -> ClaimResult:
        """处理claim，将输入转换为受控的下一步结果。"""
        existing = self.store.get_game_claim_by_source(msg.conversation_id, msg.message_id)
        if existing is not None:
            return ClaimResult(status=existing.status, game_id=existing.game_id, deduplicated=True)
        game = self.board_engine.resolve_item_no(msg.room_id, item_no, msg.quoted_message_id)
        identity = self.store.get_channel_identity(msg.channel, msg.sender_external_id)
        if game is None:
            if identity is not None:
                self.notify_dispatcher.notify_rejection(
                    identity,
                    None,
                    "game_not_found",
                    trace_id=trace_id,
                    room_id=msg.room_id,
                )
            return ClaimResult(status="rejected", reason="game_not_found")
        if identity is None:
            return ClaimResult(status="rejected", game_id=game.game_id, reason="identity_unknown")
        rejection = self._eligibility_reason(identity.customer_id, game)
        if rejection:
            self.notify_dispatcher.notify_rejection(
                identity,
                game,
                rejection,
                trace_id=trace_id,
                room_id=msg.room_id,
            )
            return ClaimResult(status="rejected", game_id=game.game_id, reason=rejection)
        try:
            _, updated, _, deduplicated = self.store.atomic_claim_seat(
                room_id=msg.room_id,
                game_id=game.game_id,
                customer_id=identity.customer_id,
                display_name=identity.public_name or msg.sender_name,
                source_conversation_id=msg.conversation_id,
                source_message_id=msg.message_id,
                trace_id=trace_id,
            )
        except ValueError as exc:
            reason = self._mutation_error_reason(str(exc))
            self.notify_dispatcher.notify_rejection(
                identity,
                game,
                reason,
                trace_id=trace_id,
                room_id=msg.room_id,
            )
            return ClaimResult(status="rejected", game_id=game.game_id, reason=reason)
        if not deduplicated:
            self.notify_dispatcher.notify_claim_success(identity, updated, trace_id=trace_id)
            self.board_engine.on_game_event(msg.room_id, "seat_claimed", trace_id=trace_id)
            if updated.remaining_seats() == 0:
                self.notify_dispatcher.notify_game_full(updated, trace_id=trace_id)
        return ClaimResult(status="claimed", game_id=updated.game_id, deduplicated=deduplicated)

    def _eligibility_reason(self, customer_id: str, game) -> str:
        if game.status not in {GameStatus.FORMING, GameStatus.INVITING} or game.remaining_seats() <= 0:
            return "game_full"
        if any(
            item.customer_id == customer_id and item.status in {"joined", "confirmed"}
            for item in game.participants
        ):
            return "already_joined"
        for other in self.store.games.values():
            if other.game_id == game.game_id or other.status not in {
                GameStatus.FORMING,
                GameStatus.INVITING,
                GameStatus.READY,
            }:
                continue
            if not any(
                item.customer_id == customer_id and item.status in {"joined", "confirmed"}
                for item in other.participants
            ):
                continue
            if game_commitment_windows_overlap(game, other):
                return "time_conflict"
        return ""

    @staticmethod
    def _mutation_error_reason(error: str) -> str:
        lowered = error.lower()
        if "already" in lowered or "duplicate" in lowered:
            return "already_joined"
        if "seat" in lowered or "full" in lowered or "capacity" in lowered:
            return "game_full"
        return "claim_conflict"


__all__ = ["ClaimHandler"]
