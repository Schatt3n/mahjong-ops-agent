from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable


DateTimeParser = Callable[[Any], datetime | None]


@dataclass(slots=True)
class CandidateReplyFactService:
    """Extract candidate reply facts without side effects.

    This service owns deterministic fallback classification and fact merging
    for candidate replies. It does not call LLMs, update state, or send
    messages; higher layers decide whether a validated action may be persisted.
    """

    parse_datetime: DateTimeParser

    def classify_reply(self, text: str, game: dict[str, Any] | None = None) -> dict[str, Any]:
        normalized = re.sub(r"\s+", "", str(text or "").lower())
        contract_negotiation = self.classify_negotiation(text, game)
        if contract_negotiation:
            return contract_negotiation
        if re.search(r"(别再问|不要再问|以后别|别打扰|不用问我)", normalized):
            return {"intent": "do_not_disturb", "feedback_type": "do_not_disturb", "status": "别再打扰"}
        if re.search(r"(到了|到店|已到|我到|在门口|楼下)", normalized):
            return {"intent": "arrived", "feedback_type": "arrived", "status": "已到店"}
        if re.search(r"(不来|来不了|去不了|没空|有事|不打|打不了|不去了|今天不行|算了|不方便)", normalized):
            return {"intent": "declined", "feedback_type": "declined", "status": "拒绝"}
        if re.search(r"(几点|哪里|地址|几楼|有烟|无烟|抽烟|几个人|几缺|打多久|几个小时|多大|多少钱|什么局|玩法)", normalized):
            return {"intent": "candidate_question", "feedback_type": "candidate_question", "status": "待确认"}
        if re.search(r"(但是|但|不过|能不能|可不可以|行不行|要不|改|想打|希望|只能)", normalized):
            return {"intent": "candidate_negotiation", "feedback_type": "candidate_negotiation", "status": "待协商"}
        if re.search(r"(等会|待会|一会|晚点|稍等|看下|看看|问下|不确定|可能|再说|考虑)", normalized):
            return {"intent": "ask_later", "feedback_type": "ask_later", "status": "下次再问"}
        if re.search(r"(^|[，。,.!！])?(来|打|打的|打呀|打啊|打吧|可以|行|好|ok|有空|能来|我来|算我|能到|冲|可以的|行的)($|[，。,.!！])?", normalized):
            return {"intent": "accepted", "feedback_type": "accepted", "status": "已确认"}
        return {"intent": "candidate_question", "feedback_type": "candidate_question", "status": "待确认"}

    def classify_negotiation(self, text: str, game: dict[str, Any] | None) -> dict[str, Any] | None:
        start_request = self.requested_start_time(text, game)
        game_start = self.game_start_at(game)
        if start_request is not None and game_start is not None and abs((start_request - game_start).total_seconds()) >= 15 * 60:
            return {
                "intent": "candidate_negotiation",
                "feedback_type": "candidate_negotiation",
                "status": "待协商",
                "requested_start_at": start_request.isoformat(),
                "requested_start_time": start_request.strftime("%H:%M"),
                "requested_start_time_label": self.natural_time_label(start_request),
                "current_start_at": game_start.isoformat(),
                "current_start_time": game_start.strftime("%H:%M"),
            }
        duration_request = self.requested_duration(text)
        game_duration = self.game_duration_hours(game)
        if duration_request is not None and game_duration is not None and abs(duration_request - game_duration) >= 0.25:
            return {
                "intent": "candidate_negotiation",
                "feedback_type": "candidate_negotiation",
                "status": "待协商",
                "requested_duration_hours": duration_request,
                "current_duration_hours": game_duration,
            }
        return None

    def apply_extracted_negotiation_facts(
        self,
        classification: dict[str, Any],
        proposal: dict[str, Any],
        game: dict[str, Any] | None,
    ) -> None:
        facts = proposal.get("extracted_facts") if isinstance(proposal.get("extracted_facts"), dict) else {}
        duration = _safe_float(facts.get("requested_duration_hours"))
        if duration is not None and "requested_duration_hours" not in classification:
            classification["requested_duration_hours"] = duration
            game_duration = self.game_duration_hours(game)
            if game_duration is not None:
                classification["current_duration_hours"] = game_duration
        requested_start = self.parse_datetime(facts.get("requested_start_at"))
        requested_time = str(facts.get("requested_start_time") or "").strip()
        game_start = self.game_start_at(game)
        if requested_start is None and requested_time and game_start:
            match = re.search(r"^([01]?\d|2[0-3]):([0-5]\d)$", requested_time)
            if match:
                requested_start = game_start.replace(
                    hour=int(match.group(1)),
                    minute=int(match.group(2)),
                    second=0,
                    microsecond=0,
                )
        if requested_start is not None and "requested_start_time" not in classification:
            classification["requested_start_at"] = requested_start.isoformat()
            classification["requested_start_time"] = requested_start.strftime("%H:%M")
            classification["requested_start_time_label"] = self.natural_time_label(requested_start)
            if game_start is not None:
                classification["current_start_at"] = game_start.isoformat()
                classification["current_start_time"] = game_start.strftime("%H:%M")

    def game_duration_hours(self, game: dict[str, Any] | None) -> float | None:
        parsed = game.get("parsed") if isinstance(game, dict) and isinstance(game.get("parsed"), dict) else {}
        return _safe_float(parsed.get("duration_hours"))

    def game_start_at(self, game: dict[str, Any] | None) -> datetime | None:
        parsed = game.get("parsed") if isinstance(game, dict) and isinstance(game.get("parsed"), dict) else {}
        return self.parse_datetime(parsed.get("start_at"))

    def requested_start_time(self, text: str, game: dict[str, Any] | None) -> datetime | None:
        game_start = self.game_start_at(game)
        if game_start is None:
            return None
        normalized = re.sub(r"\s+", "", str(text or "").lower())
        hour: int | None = None
        minute = 0
        match = re.search(r"(?<!\d)([01]?\d|2[0-3])[:：.]([0-5]\d)(?!\d)", normalized)
        if match:
            hour = int(match.group(1))
            minute = int(match.group(2))
        if hour is None:
            match = re.search(r"(?<!\d)([01]?\d|2[0-3])点(半|[0-5]?\d分?)?", normalized)
            if match:
                hour = int(match.group(1))
                suffix = match.group(2) or ""
                minute = 30 if "半" in suffix else int(re.sub(r"\D", "", suffix) or 0)
        if hour is None:
            match = re.search(r"([一二两三四五六七八九十])点(半|[一二两三四五六七八九十]十分?|[0-5]?\d分?)?", normalized)
            if match:
                hour = _chinese_hour(match.group(1))
                suffix = match.group(2) or ""
                minute = 30 if "半" in suffix else _minute_from_chinese_or_digits(suffix)
        if hour is None:
            return None
        if re.search(r"凌晨|早上|上午|早晨", normalized):
            resolved_hour = 0 if hour == 12 else hour
        elif re.search(r"下午|晚上|今晚|傍晚|下班", normalized):
            resolved_hour = hour + 12 if 1 <= hour <= 11 else hour
        elif hour <= 11:
            candidates = [
                game_start.replace(hour=hour, minute=minute, second=0, microsecond=0),
                game_start.replace(hour=hour + 12, minute=minute, second=0, microsecond=0),
            ]
            return min(candidates, key=lambda item: abs((item - game_start).total_seconds()))
        else:
            resolved_hour = hour
        return game_start.replace(hour=min(resolved_hour, 23), minute=minute, second=0, microsecond=0)

    def requested_duration(self, text: str) -> float | None:
        normalized = re.sub(r"\s+", "", str(text or "").lower())
        match = re.search(r"(\d+(?:\.\d+)?)\s*(?:h|小时|个小时)", normalized)
        if match:
            return _safe_float(match.group(1))
        chinese_digits = {
            "一": 1,
            "二": 2,
            "两": 2,
            "三": 3,
            "四": 4,
            "五": 5,
            "六": 6,
            "七": 7,
            "八": 8,
            "九": 9,
            "十": 10,
        }
        match = re.search(r"([一二两三四五六七八九十])(?:个)?小时", normalized)
        if match:
            return float(chinese_digits.get(match.group(1), 0) or 0) or None
        return None

    def natural_time_label(self, value: datetime) -> str:
        hour = value.hour
        display_hour = hour - 12 if hour > 12 else hour
        display_hour = 12 if display_hour == 0 else display_hour
        hour_map = {
            1: "一",
            2: "两",
            3: "三",
            4: "四",
            5: "五",
            6: "六",
            7: "七",
            8: "八",
            9: "九",
            10: "十",
            11: "十一",
            12: "十二",
        }
        label = hour_map.get(display_hour, str(display_hour))
        if value.minute == 0:
            return f"{label}点"
        if value.minute == 30:
            return f"{label}点半"
        return value.strftime("%H:%M")


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _chinese_hour(value: str) -> int | None:
    digits = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
    return digits.get(value)


def _minute_from_chinese_or_digits(value: str) -> int:
    if not value:
        return 0
    digits = re.sub(r"\D", "", value)
    if digits:
        return min(59, int(digits))
    chinese_digits = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5}
    match = re.search(r"([一二两三四五])十", value)
    if match:
        return int(chinese_digits.get(match.group(1), 0) or 0) * 10
    return 0
