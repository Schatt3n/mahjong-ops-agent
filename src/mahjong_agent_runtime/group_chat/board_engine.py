"""Materialized public-room board built from authoritative game state."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

from ..models import Game, GameStatus, MessageReference, new_id, now
from ..stores import AgentStore
from .messenger import GroupMessenger
from .models import BoardSnapshot, BoardSnapshotItem, GameConversationLink, GroupMessage
from .parsing import parse_game_post
from .projections import public_game_start_display


BOARD_TASK_TYPE = "publish_group_board"
DEFAULT_MERGE_WINDOW_SECONDS = 30


class BoardEngine:
    """Project active games into versioned, quote-resolvable room boards.

    The game aggregate remains the source of truth. A board snapshot only
    freezes the public numbering used by one outbound message, so a quoted
    claim can be resolved exactly even after the current board is reordered.
    """

    CRITICAL_EVENTS = frozenset({"seat_claimed", "seat_released"})

    def __init__(
        self,
        *,
        store: AgentStore,
        messenger: GroupMessenger,
        clock: Callable[[], datetime] = now,
    ) -> None:
        self.store = store
        self.messenger = messenger
        self.clock = clock

    def on_game_event(
        self,
        room_id: str,
        event_type: str,
        *,
        trace_id: str,
    ):
        """Coalesce normal changes and move seat changes to the front of the queue."""

        policy = self.store.get_group_room_policy(room_id)
        merge_seconds = (
            max(0, int(policy.merge_window_seconds))
            if policy is not None
            else DEFAULT_MERGE_WINDOW_SECONDS
        )
        urgent = event_type in self.CRITICAL_EVENTS
        due_at = self.clock() if urgent else self.clock() + timedelta(seconds=merge_seconds)
        return self.store.ensure_group_board_publish_task(
            room_id=room_id,
            due_at=due_at,
            trace_id=trace_id,
            urgent=urgent,
        )

    def import_game_from_post(self, msg: GroupMessage, *, trace_id: str) -> Game:
        """Import an explicit room post without manufacturing a room conversation."""

        parsed = parse_game_post(msg.text, anchor=msg.sent_at)
        if parsed is None:
            raise ValueError("message is not an explicit game post")
        identity = self.store.get_channel_identity(msg.channel, msg.sender_external_id)
        customer_id = identity.customer_id if identity is not None else f"{msg.channel}:{msg.sender_external_id}"
        public_name = (
            identity.public_name
            if identity is not None and identity.public_name
            else msg.sender_name or msg.sender_external_id
        )
        game_conversation_id = f"{msg.conversation_id}:post:{msg.message_id}"
        game, _ = self.store.create_game(
            conversation_id=game_conversation_id,
            organizer_id=customer_id,
            organizer_name=public_name,
            requirement=parsed,
            known_players=[],
            trace_id=trace_id,
        )
        self.store.link_game_conversation(
            GameConversationLink(
                link_id=new_id("game_conversation_link"),
                game_id=game.game_id,
                conversation_id=msg.conversation_id,
                room_id=msg.room_id,
                customer_id=customer_id,
                link_type="origin",
            )
        )
        self.on_game_event(msg.room_id, "game_created", trace_id=trace_id)
        return game

    def publish(self, room_id: str, *, trace_id: str) -> BoardSnapshot | None:
        """Publish one new board version, or stay silent when no open games exist."""

        games = self.store.get_board_eligible_games(room_id)
        if not games:
            return None
        items: list[BoardSnapshotItem] = []
        lines = ["当前缺人局："]
        for item_no, game in enumerate(games, start=1):
            rendered = self._render_game(item_no, game)
            items.append(BoardSnapshotItem(item_no=item_no, game_id=game.game_id, rendered_text=rendered))
            lines.append(rendered)
        lines.extend(("", "回复编号即可认领，如\"2来\""))
        board_text = "\n".join(lines)
        snapshot_id = new_id("board_snapshot")
        conversation_id = f"wechaty:room:{room_id}"
        source_message_id = f"group_board:{snapshot_id}"
        # Register the business anchor before transport. Some WeChat puppets only
        # reveal the platform message ID through the later self-message echo.
        self.store.register_message_reference(
            MessageReference(
                message_id=source_message_id,
                conversation_id=conversation_id,
                business_ref_type="group_board_snapshot",
                business_ref_id=snapshot_id,
                text=board_text,
                channel="wechaty",
                recipient_id=room_id,
                metadata={
                    "room_id": room_id,
                    "item_game_ids": [item.game_id for item in items],
                    "trace_id": trace_id,
                    "reference_role": "transport_source_anchor",
                },
            )
        )
        external_message_id = self.messenger.send_group_message(
            room_id,
            board_text,
            metadata={
                "trace_id": trace_id,
                "message_type": "game_board",
                "source_message_id": source_message_id,
                "business_ref_type": "group_board_snapshot",
                "business_ref_id": snapshot_id,
            },
        )
        if not external_message_id or external_message_id.startswith("suppressed:"):
            return None
        snapshot = BoardSnapshot(
            snapshot_id=snapshot_id,
            room_id=room_id,
            conversation_id=conversation_id,
            external_message_id=external_message_id,
            rendered_text=board_text,
            items=items,
            created_at=self.clock(),
        )
        self.store.save_board_snapshot(snapshot)
        if external_message_id != source_message_id:
            self.store.link_message_reference(
                conversation_id=conversation_id,
                message_id=external_message_id,
                source_message_id=source_message_id,
                channel="wechaty",
                text=board_text,
                metadata={"reference_role": "transport_response"},
            )
        return snapshot

    def resolve_item_no(
        self,
        room_id: str,
        item_no: int,
        quoted_message_id: str | None = None,
    ) -> Game | None:
        """Resolve a board number without guessing an obsolete unquoted version."""

        snapshot = None
        if quoted_message_id:
            snapshot = self.store.get_board_snapshot_by_message_id(room_id, quoted_message_id)
            if snapshot is None:
                reference = self.store.resolve_message_reference(
                    conversation_id=f"wechaty:room:{room_id}",
                    message_id=quoted_message_id,
                )
                if reference is not None and reference.business_ref_type == "group_board_snapshot":
                    candidate = self.store.get_board_snapshot(reference.business_ref_id)
                    if candidate is not None and candidate.room_id == room_id:
                        snapshot = candidate
        else:
            snapshot = self.store.get_latest_board_snapshot(room_id)
        if snapshot is None:
            return None
        board_item = next((item for item in snapshot.items if item.item_no == int(item_no)), None)
        if board_item is None:
            return None
        try:
            game = self.store.require_game(board_item.game_id)
        except (KeyError, ValueError):
            return None
        if game.status not in {GameStatus.FORMING, GameStatus.INVITING} or game.remaining_seats() <= 0:
            return None
        return game

    @staticmethod
    def _render_game(item_no: int, game: Game) -> str:
        requirement = game.requirement
        start = public_game_start_display(game)
        stake = str(
            requirement.get("stake_label")
            or requirement.get("stake")
            or requirement.get("base_stake")
            or "档位待定"
        )
        smoke = {
            "no_smoking": "无烟",
            "smoking": "有烟",
            "any": "烟都可",
        }.get(str(requirement.get("smoke_preference") or ""), "烟况待定")
        remaining = game.remaining_seats()
        claimed = max(0, game.seats_total - remaining)
        seat_code = f"{claimed}7{remaining}"
        game_name = BoardEngine._display_game_name(requirement)
        fields = [f"{item_no}、{start}"]
        if game_name:
            fields.append(game_name)
        fields.extend((stake, smoke, seat_code))
        return " ".join(fields)

    @staticmethod
    def _display_game_name(requirement: dict[str, Any]) -> str:
        game_type = str(requirement.get("requested_game") or "")
        variant = str(requirement.get("game_variant") or "")
        if variant == "caiqiao":
            return "cq"
        return {
            "hangzhou_mahjong": "杭麻",
            "sichuan_mahjong": "川麻",
            "red_center_mahjong": "红中",
        }.get(game_type, "")


__all__ = ["BOARD_TASK_TYPE", "BoardEngine"]
