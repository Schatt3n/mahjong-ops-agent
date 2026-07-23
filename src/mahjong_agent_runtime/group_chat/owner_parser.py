"""Deterministic parsing for owner-authored public board messages.

The owner board is a small domain language, not one fixed text template.  This
parser extracts fields independently of their order and intentionally refuses
to invent absent facts.  Natural-language member intent remains a model task.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from ..knowledge import default_terminology_repository
from ..models import new_id
from .models import BoardItem, BoardState, GroupMessage


_SEAT_CODE = re.compile(r"(?<!\d)(173|272|371)(?!\d)")
_NUMBERED_PREFIX = re.compile(r"^\s*(?P<number>\d{1,2})(?:\ufe0f?\u20e3|[、.．:：])\s*")
_FIXED_TIME = re.compile(r"(?<!\d)([01]?\d|2[0-3])\s*[.:：]\s*([0-5]\d)(?!\d)")
_TIME_RANGE = re.compile(r"(?<!\d)([01]?\d|2[0-3])\s*[-—~至]\s*([01]?\d|2[0-4])(?!\d)")
_CHINESE_TIME = re.compile(r"(?<![\d一二两三四五六七八九十])([0-2]?\d|[一二两三四五六七八九十]{1,3})\s*点(?:(半)|([0-5]?\d)分?)?")
_LABELED_STAKE = re.compile(r"大小\s*[:：]?\s*(0\.5|[1-9]\d*(?:\.\d+)?)")
_EXPLICIT_STAKE = re.compile(r"(?<!\d)(0\.5|[1-9]\d*(?:\.\d+)?)\s*(?:块|元)")
_CHINESE_STAKE = re.compile(r"(?<![一二两三四五六七八九十])(一|二|两|三|四|五|六|七|八|九|十)\s*(?:块|元)")
# Unit-less stakes are common in board shorthand (``1无烟``), but local
# three-digit codes such as 368/568 are a separate opaque field. Larger stakes
# remain unambiguous when written with a unit (for example ``100块``).
_SMOKE_ADJACENT_STAKE = re.compile(r"(?<!\d)(0\.5|[1-9]\d?)\s*(?=无烟|有烟|少烟|烟都可)")
_THREE_DIGIT = re.compile(r"(?<!\d)(\d{3})(?!\d)")
_SPECIAL_RULE = re.compile(r"(?<!\d)(\d+)\s*爆")
_DURATION = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)\s*(?:h|小时)(?!\w)", re.IGNORECASE)
_VERBOSE_LABEL = re.compile(r"(?:人数|时间|大小)\s*[:：]")
_TEMPORARY_CONSTRAINTS = ("无情侣", "不要情侣", "禁情侣")
_CHINESE_DIGITS = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}


@dataclass(slots=True)
class OwnerParseResult:
    action: str
    board_state: BoardState | None = None
    changed_item_ids: tuple[str, ...] = ()


class OwnerMessageParser:
    """Parse stable owner board syntax while preserving unknown fields."""

    def __init__(self, *, owner_external_ids: set[str] | None = None) -> None:
        self.owner_external_ids = set(owner_external_ids or set())

    def is_owner(self, message: GroupMessage) -> bool:
        return message.sender_external_id in self.owner_external_ids or bool(
            message.metadata.get("is_owner") or message.metadata.get("sender_role") == "owner"
        )

    def process(self, message: GroupMessage, current: BoardState | None) -> OwnerParseResult:
        if not self.is_owner(message):
            return OwnerParseResult(action="not_owner")

        quoted_update = self._apply_quoted_update(message, current)
        if quoted_update is not None:
            board, changed_id = quoted_update
            return OwnerParseResult(action="board_updated", board_state=board, changed_item_ids=(changed_id,))

        if _VERBOSE_LABEL.search(message.text):
            parsed = self._parse_board_line(
                message.text,
                source_message_id=message.message_id,
                anchor=message.sent_at,
            )
            board_items = [parsed] if parsed is not None else []
        else:
            board_items = [
                item
                for line in message.text.splitlines()
                if line.strip()
                for item in [
                    self._parse_board_line(
                        line,
                        source_message_id=message.message_id,
                        anchor=message.sent_at,
                    )
                ]
                if item is not None
            ]

        if len(board_items) >= 2:
            self._assign_snapshot_numbers(board_items)
            board = self._board(message, board_items)
            return OwnerParseResult(
                action="board_replaced",
                board_state=board,
                changed_item_ids=tuple(item.id for item in board_items),
            )

        if len(board_items) == 1:
            item = board_items[0]
            if current is None:
                item.display_no = item.display_no or 1
                board = self._board(message, [item])
                return OwnerParseResult(action="board_replaced", board_state=board, changed_item_ids=(item.id,))
            updated = self._upsert_single_board_item(message, current, item)
            if updated is not None:
                board, changed_id = updated
                return OwnerParseResult(action="board_updated", board_state=board, changed_item_ids=(changed_id,))

        if current is None:
            return OwnerParseResult(action="owner_ignored")
        updated = self._apply_single_line_update(message, current)
        if updated is None:
            return OwnerParseResult(action="owner_ignored")
        board, changed_id = updated
        return OwnerParseResult(action="board_updated", board_state=board, changed_item_ids=(changed_id,))

    @staticmethod
    def _board(message: GroupMessage, items: list[BoardItem]) -> BoardState:
        return BoardState(
            room_id=message.room_id,
            items=items,
            source_message_id=message.message_id,
            last_published_at=message.sent_at,
        )

    @staticmethod
    def _assign_snapshot_numbers(items: list[BoardItem]) -> None:
        explicit = [item.display_no for item in items if item.display_no > 0]
        if len(explicit) != len(items) or len(set(explicit)) != len(explicit):
            for number, item in enumerate(items, start=1):
                item.display_no = number

    def _upsert_single_board_item(
        self,
        message: GroupMessage,
        current: BoardState,
        parsed: BoardItem,
    ) -> tuple[BoardState, str] | None:
        """Update one logical board item even when its 272/371 progress changes."""

        candidates: list[BoardItem] = []
        if parsed.display_no > 0:
            candidates = [item for item in current.items if item.display_no == parsed.display_no]
            # A public board number is an explicit identity supplied by the
            # owner. If that number is not present in the current snapshot,
            # this is an append, even when another item has the same 272/371
            # participant progress code.
            if not candidates:
                board = self._board(message, [*current.items, parsed])
                return board, parsed.id
        if not candidates:
            candidates = [
                item
                for item in current.items
                if item.participant_code == parsed.participant_code
                and (not parsed.game_type or item.game_type == parsed.game_type)
            ]
            if parsed.time not in (None, "人齐开"):
                timed = [item for item in candidates if item.time == parsed.time]
                if timed:
                    candidates = timed
        if not candidates:
            candidates = [item for item in current.items if self._same_logical_game(item, parsed)]
        if len(candidates) > 1:
            return None
        if len(candidates) == 1:
            target = candidates[0]
            replacement = self._merge_item(target, parsed)
            board = self._board(
                message,
                [replacement if item.id == target.id else item for item in current.items],
            )
            return board, target.id

        parsed.display_no = parsed.display_no or max((item.display_no for item in current.items), default=0) + 1
        board = self._board(message, [*current.items, parsed])
        return board, parsed.id

    @staticmethod
    def _same_logical_game(left: BoardItem, right: BoardItem) -> bool:
        fields = (
            "game_type",
            "ruleset",
            "time",
            "end_time",
            "duration_hours",
            "rule_code",
            "smoking",
            "stakes",
            "special_rules",
        )
        compared = 0
        for field_name in fields:
            right_value = getattr(right, field_name)
            if right_value in (None, ""):
                continue
            compared += 1
            if getattr(left, field_name) != right_value:
                return False
        return compared >= 3

    @staticmethod
    def _merge_item(target: BoardItem, parsed: BoardItem) -> BoardItem:
        return replace(
            target,
            participant_code=parsed.participant_code or target.participant_code,
            time=parsed.time if parsed.time is not None else target.time,
            end_time=parsed.end_time if parsed.end_time is not None else target.end_time,
            duration_hours=(
                parsed.duration_hours if parsed.duration_hours is not None else target.duration_hours
            ),
            rule_code=parsed.rule_code or target.rule_code,
            game_type=parsed.game_type or target.game_type,
            ruleset=parsed.ruleset or target.ruleset,
            smoking=parsed.smoking or target.smoking,
            stakes=parsed.stakes or target.stakes,
            special_rules=parsed.special_rules or target.special_rules,
            temporary_constraints=(parsed.temporary_constraints or target.temporary_constraints),
            source_message_id=parsed.source_message_id or target.source_message_id,
            status=parsed.status if parsed.status != "waiting" or target.status == "waiting" else target.status,
            slots_filled=parsed.slots_filled,
            participants=list(target.participants),
        )

    def _parse_board_line(self, line: str, *, source_message_id: str, anchor) -> BoardItem | None:
        original = str(line or "").strip()
        prefix = _NUMBERED_PREFIX.match(original)
        display_no = int(prefix.group("number")) if prefix else 0
        text = original[prefix.end() :] if prefix else original
        seat_match = _SEAT_CODE.search(text)
        if seat_match is None:
            return None
        participant_code = seat_match.group(1)
        game_type, ruleset = self._extract_game_type(text)
        start_time, end_time = self._extract_times(text, anchor=anchor)
        smoking = self._extract_smoking(text)
        stakes = self._extract_stake(text, participant_code=participant_code)
        rule_code = self._extract_rule_code(
            text,
            participant_code=participant_code,
            verbose_template=bool(_VERBOSE_LABEL.search(text)),
        )
        special = _SPECIAL_RULE.search(text)
        duration = _DURATION.search(text)
        has_board_fact = bool(
            game_type or start_time or end_time or smoking or stakes or rule_code or special or duration
        )
        if not has_board_fact:
            return None
        standalone_full = bool(re.search(r"(?:^|\s|[,，])人齐(?:了)?(?:$|\s|[,，])", text)) and "人齐开" not in text
        status = (
            "full"
            if standalone_full or "满了" in text or "已满" in text
            else "playing"
            if "开打" in text
            else "waiting"
        )
        current_players = 4 if status == "full" else int(participant_code[0])
        return BoardItem(
            id=new_id("group_board_item"),
            display_no=display_no,
            game_type=game_type,
            participant_code=participant_code,
            time=start_time,
            smoking=smoking,
            stakes=stakes,
            special_rules=f"{special.group(1)}爆" if special else None,
            ruleset=ruleset,
            end_time=end_time,
            duration_hours=float(duration.group(1)) if duration else None,
            rule_code=rule_code,
            temporary_constraints=[item for item in _TEMPORARY_CONSTRAINTS if item in text],
            source_message_id=source_message_id,
            status=status,
            slots_filled=current_players,
        )

    @staticmethod
    def _extract_game_type(text: str) -> tuple[str, str | None]:
        matched = default_terminology_repository().first_match(
            text,
            categories={"game_type", "game_variant"},
        )
        if matched is not None:
            canonical = matched.term.canonical
            return (
                str(canonical.get("game_type") or ""),
                str(canonical.get("ruleset") or "") or None,
            )
        return "", None

    @staticmethod
    def _extract_smoking(text: str) -> str | None:
        if "无烟" in text or "禁烟" in text:
            return "无烟"
        if "少烟" in text:
            return "少烟"
        if "烟都可" in text or "烟不限" in text or "有烟无烟都" in text:
            return "不限"
        if "有烟" in text or "可烟" in text:
            return "有烟"
        return None

    @classmethod
    def _extract_times(cls, text: str, *, anchor) -> tuple[str | None, str | None]:
        if "人齐开" in text or "人齐就开" in text or "齐了开" in text:
            return "人齐开", None
        time_range = _TIME_RANGE.search(text)
        if time_range:
            return f"{int(time_range.group(1)):02d}:00", f"{int(time_range.group(2)):02d}:00"
        fixed = _FIXED_TIME.search(text)
        if fixed:
            hour = cls._resolve_short_hour(int(fixed.group(1)), anchor=anchor)
            return f"{hour:02d}:{int(fixed.group(2)):02d}", None
        chinese = _CHINESE_TIME.search(text)
        if chinese:
            hour = cls._chinese_number(chinese.group(1))
            if hour is None or hour > 23:
                return None, None
            hour = cls._resolve_short_hour(hour, anchor=anchor)
            minute = 30 if chinese.group(2) else int(chinese.group(3) or 0)
            return f"{hour:02d}:{minute:02d}", None
        return None, None

    @staticmethod
    def _resolve_short_hour(hour: int, *, anchor) -> int:
        """Interpret short clock forms in the message's local-day context.

        An owner posting a board during the afternoon normally means 16:00 when
        writing ``4点`` and 20:30 when writing ``8点半``. Explicit 24-hour forms
        remain unchanged. The original message is still retained as audit data.
        """

        if 1 <= hour <= 11 and getattr(anchor, "hour", 0) >= 12:
            return hour + 12
        return hour

    @classmethod
    def _extract_stake(cls, text: str, *, participant_code: str) -> str:
        labeled = _LABELED_STAKE.search(text)
        if labeled:
            return cls._clean_number(labeled.group(1))
        explicit = _EXPLICIT_STAKE.search(text)
        if explicit:
            unit = "块" if "块" in explicit.group(0) else "元"
            return f"{cls._clean_number(explicit.group(1))}{unit}"
        chinese = _CHINESE_STAKE.search(text)
        if chinese:
            value = cls._chinese_number(chinese.group(1))
            unit = "块" if "块" in chinese.group(0) else "元"
            return f"{value}{unit}" if value is not None else ""
        text_without_times = _FIXED_TIME.sub(" ", _TIME_RANGE.sub(" ", text))
        adjacent = _SMOKE_ADJACENT_STAKE.search(text_without_times)
        if adjacent:
            return cls._clean_number(adjacent.group(1))
        return ""

    @staticmethod
    def _extract_rule_code(
        text: str,
        *,
        participant_code: str,
        verbose_template: bool,
    ) -> str | None:
        """Preserve venue-local three-digit codes without guessing their semantics."""

        if verbose_template:
            return None
        text_without_times = _FIXED_TIME.sub(" ", _TIME_RANGE.sub(" ", text))
        candidates = [value for value in _THREE_DIGIT.findall(text_without_times) if value != participant_code]
        return candidates[-1] if candidates else None

    @staticmethod
    def _clean_number(value: str) -> str:
        number = float(value)
        return str(int(number)) if number.is_integer() else str(number)

    @classmethod
    def _chinese_number(cls, value: str) -> int | None:
        if value.isdigit():
            return int(value)
        if value == "十":
            return 10
        if "十" in value:
            left, _, right = value.partition("十")
            tens = cls._chinese_number(left) if left else 1
            ones = cls._chinese_number(right) if right else 0
            return None if tens is None or ones is None else tens * 10 + ones
        return _CHINESE_DIGITS.get(value)

    def _apply_quoted_update(
        self,
        message: GroupMessage,
        current: BoardState | None,
    ) -> tuple[BoardState, str] | None:
        quoted_text = str(message.metadata.get("quoted_text") or "")
        if current is None or (not message.quoted_message_id and not quoted_text):
            return None
        target: BoardItem | None = None
        if message.quoted_message_id:
            matches = [item for item in current.items if item.source_message_id == message.quoted_message_id]
            target = matches[0] if len(matches) == 1 else None
        if target is None and quoted_text:
            parsed_quote = self._parse_board_line(
                quoted_text,
                source_message_id=message.quoted_message_id or "",
                anchor=message.sent_at,
            )
            if parsed_quote is not None:
                matches = [item for item in current.items if self._same_logical_game(item, parsed_quote)]
                target = matches[0] if len(matches) == 1 else None
        if target is None:
            return None

        parsed_update = self._parse_board_line(
            message.text,
            source_message_id=message.message_id,
            anchor=message.sent_at,
        )
        if parsed_update is not None:
            parsed_update.display_no = target.display_no
            replacement = self._merge_item(target, parsed_update)
        elif "人齐开" not in message.text and re.search(r"人齐|满了|已满", message.text):
            replacement = replace(
                target,
                source_message_id=message.message_id,
                status="full",
                slots_filled=target.slots_total,
            )
        else:
            return None
        return self._board(
            message,
            [replacement if item.id == target.id else item for item in current.items],
        ), target.id

    def _apply_single_line_update(
        self,
        message: GroupMessage,
        current: BoardState,
    ) -> tuple[BoardState, str] | None:
        table_match = _SEAT_CODE.search(message.text)
        if table_match is None:
            return None
        candidates = [item for item in current.items if item.participant_code == table_match.group(1)]
        parsed_game, _ = self._extract_game_type(message.text)
        if parsed_game:
            candidates = [item for item in candidates if item.game_type == parsed_game]
        if len(candidates) != 1:
            return None
        target = candidates[0]
        time_value, end_time = self._extract_times(message.text, anchor=message.sent_at)
        status = target.status
        slots_filled = target.slots_filled
        if "满了" in message.text or "已满" in message.text or ("人齐" in message.text and "人齐开" not in message.text):
            status = "full"
            slots_filled = target.slots_total
        elif "开打" in message.text:
            status = "playing"
        replacement = replace(
            target,
            time=time_value or target.time,
            end_time=end_time or target.end_time,
            status=status,
            slots_filled=slots_filled,
            source_message_id=message.message_id,
        )
        return self._board(
            message,
            [replacement if item.id == target.id else item for item in current.items],
        ), target.id


__all__ = ["OwnerMessageParser", "OwnerParseResult"]
