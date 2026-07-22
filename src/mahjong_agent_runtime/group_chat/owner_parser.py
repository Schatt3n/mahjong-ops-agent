"""Deterministic parsing for owner-authored public board messages."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from ..models import new_id
from .models import BoardItem, BoardState, GroupMessage


_BOARD_LINE = re.compile(
    r"^\s*(?P<game_type>cq|财敲|红中|川麻换三|川麻|杭麻)\s*"
    r"(?P<table_id>173|272|371)(?P<rest>.*)$",
    re.IGNORECASE,
)
_TABLE_ID = re.compile(r"(?<!\d)(173|272|371)(?!\d)")
_TIME = re.compile(r"(?<!\d)([01]?\d|2[0-3])\s*[.:：]\s*([0-5]\d)(?!\d)")
_EXPLICIT_STAKE = re.compile(r"(?<!\d)(0\.5|[1-9]\d*(?:\.\d+)?)\s*(?:块|元)")
_THREE_DIGIT_STAKE = re.compile(r"(?<!\d)(\d{3})(?!\d)")
_SPECIAL_RULE = re.compile(r"(?<!\d)(\d+)\s*爆")


@dataclass(slots=True)
class OwnerParseResult:
    action: str
    board_state: BoardState | None = None
    changed_item_ids: tuple[str, ...] = ()


class OwnerMessageParser:
    """Parse only stable owner board syntax; uncertain owner prose is ignored."""

    def __init__(self, *, owner_external_ids: set[str] | None = None) -> None:
        self.owner_external_ids = set(owner_external_ids or set())

    def is_owner(self, message: GroupMessage) -> bool:
        return message.sender_external_id in self.owner_external_ids or bool(
            message.metadata.get("is_owner") or message.metadata.get("sender_role") == "owner"
        )

    def process(self, message: GroupMessage, current: BoardState | None) -> OwnerParseResult:
        if not self.is_owner(message):
            return OwnerParseResult(action="not_owner")

        parsed_lines = [self._parse_board_line(line) for line in message.text.splitlines() if line.strip()]
        board_items = [item for item in parsed_lines if item is not None]
        if len(board_items) >= 2:
            for display_no, item in enumerate(board_items, start=1):
                item.display_no = display_no
            board = BoardState(
                room_id=message.room_id,
                items=board_items,
                source_message_id=message.message_id,
                last_published_at=message.sent_at,
            )
            return OwnerParseResult(
                action="board_replaced",
                board_state=board,
                changed_item_ids=tuple(item.id for item in board_items),
            )

        if len(board_items) == 1:
            if current is None:
                item = board_items[0]
                item.display_no = 1
                board = BoardState(
                    room_id=message.room_id,
                    items=[item],
                    source_message_id=message.message_id,
                    last_published_at=message.sent_at,
                )
                return OwnerParseResult(
                    action="board_replaced",
                    board_state=board,
                    changed_item_ids=(item.id,),
                )
            updated = self._upsert_single_board_item(message, current, board_items[0])
            if updated is not None:
                board, changed_id = updated
                return OwnerParseResult(
                    action="board_updated",
                    board_state=board,
                    changed_item_ids=(changed_id,),
                )

        if current is None:
            return OwnerParseResult(action="owner_ignored")
        updated = self._apply_single_line_update(message, current)
        if updated is None:
            return OwnerParseResult(action="owner_ignored")
        board, changed_id = updated
        return OwnerParseResult(action="board_updated", board_state=board, changed_item_ids=(changed_id,))

    def _upsert_single_board_item(
        self,
        message: GroupMessage,
        current: BoardState,
        parsed: BoardItem,
    ) -> tuple[BoardState, str] | None:
        """Update one unambiguous item, or append a genuinely new board item."""

        candidates = [
            item
            for item in current.items
            if item.game_type.lower() == parsed.game_type.lower() and item.table_id == parsed.table_id
        ]
        if parsed.time not in (None, "人齐开"):
            timed = [item for item in candidates if item.time == parsed.time]
            if timed:
                candidates = timed
        if len(candidates) > 1:
            return None
        if len(candidates) == 1:
            target = candidates[0]
            replacement = replace(
                parsed,
                id=target.id,
                display_no=target.display_no,
                participants=list(target.participants),
            )
            board = BoardState(
                room_id=current.room_id,
                items=[replacement if item.id == target.id else item for item in current.items],
                source_message_id=message.message_id,
                last_published_at=message.sent_at,
            )
            return board, target.id

        parsed.display_no = max((item.display_no for item in current.items), default=0) + 1
        board = BoardState(
            room_id=current.room_id,
            items=[*current.items, parsed],
            source_message_id=message.message_id,
            last_published_at=message.sent_at,
        )
        return board, parsed.id

    def _parse_board_line(self, line: str) -> BoardItem | None:
        match = _BOARD_LINE.match(line)
        if match is None:
            return None
        rest = match.group("rest")
        table_id = match.group("table_id")
        time_value = self._extract_time(rest)
        smoking = "无烟" if "无烟" in rest else "有烟" if "有烟" in rest else None
        special = _SPECIAL_RULE.search(rest)
        explicit_stake = _EXPLICIT_STAKE.search(rest)
        stakes = f"{explicit_stake.group(1)}块" if explicit_stake else ""
        if not stakes:
            candidates = [token for token in _THREE_DIGIT_STAKE.findall(rest) if token != table_id]
            if candidates:
                stakes = candidates[-1]
        status = "full" if "满了" in rest or "已满" in rest else "playing" if "开打" in rest else "waiting"
        return BoardItem(
            id=new_id("group_board_item"),
            display_no=0,
            game_type=match.group("game_type").lower() if match.group("game_type").lower() == "cq" else match.group("game_type"),
            table_id=table_id,
            time=time_value,
            smoking=smoking,
            stakes=stakes,
            special_rules=f"{special.group(1)}爆" if special else None,
            status=status,
            slots_filled=int(table_id[0]),
        )

    def _apply_single_line_update(
        self,
        message: GroupMessage,
        current: BoardState,
    ) -> tuple[BoardState, str] | None:
        table_match = _TABLE_ID.search(message.text)
        if table_match is None:
            return None
        table_id = table_match.group(1)
        candidates = [item for item in current.items if item.table_id == table_id]
        for game_type in ("红中", "川麻换三", "川麻", "杭麻", "cq", "财敲"):
            if game_type.lower() in message.text.lower():
                candidates = [item for item in candidates if item.game_type.lower() == game_type.lower()]
                break
        if len(candidates) != 1:
            return None
        target = candidates[0]
        time_value = self._extract_time(message.text) or target.time
        status = target.status
        if "满了" in message.text or "已满" in message.text:
            status = "full"
        elif "开打" in message.text:
            status = "playing"
        elif "人齐开" in message.text:
            status = "waiting"
        replacement = replace(target, time=time_value, status=status)
        board = BoardState(
            room_id=current.room_id,
            items=[replacement if item.id == target.id else item for item in current.items],
            source_message_id=message.message_id,
            last_published_at=message.sent_at,
        )
        return board, target.id

    @staticmethod
    def _extract_time(text: str) -> str | None:
        if "人齐开" in text:
            return "人齐开"
        match = _TIME.search(text)
        if match is None:
            return None
        return f"{int(match.group(1)):02d}:{int(match.group(2)):02d}"


__all__ = ["OwnerMessageParser", "OwnerParseResult"]
