from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta

from .models import DEFAULT_TZ, ExtractionResult, GameRequest, GameStatus, Intent, Message
from .normalization import normalize_mahjong_text


CN_NUMBERS = {
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "俩": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
    "十一": 11,
    "十二": 12,
}


@dataclass(slots=True)
class MissingInfo:
    current_player_count: int
    missing_count: int
    raw: str


@dataclass(slots=True)
class ParsedTime:
    start_at: datetime | None
    confidence: float
    ambiguities: list[str]
    raw: str | None = None


@dataclass(slots=True)
class StakeInfo:
    level: str | None
    base_score: float | None = None
    cap_score: float | None = None
    raw: str | None = None


class MahjongMessageParser:
    """Rule-first parser for mahjong room messages.

    The parser deliberately keeps ambiguity visible. For example, "5点开"
    becomes 17:00 with low confidence and a follow-up question.
    """

    code_missing_map = {
        "371": (3, 1),
        "3缺1": (3, 1),
        "3差1": (3, 1),
        "31": (3, 1),
        "173": (1, 3),
        "1缺3": (1, 3),
        "1差3": (1, 3),
        "13": (1, 3),
        "272": (2, 2),
        "2缺2": (2, 2),
        "2差2": (2, 2),
        "22": (2, 2),
    }

    def __init__(self, default_region: str | None = None) -> None:
        self.default_region = self._normalize_region(
            default_region or os.getenv("MAHJONG_DEFAULT_REGION") or "hangzhou"
        )

    def parse(self, message: Message, now: datetime | None = None) -> ExtractionResult:
        now = self._ensure_tz(now or datetime.now(DEFAULT_TZ))
        text = self._normalize(message.text)
        intent = self._detect_intent(text)
        missing = self._parse_missing(text)
        stake = self._parse_stake(text)
        level = stake.level
        parsed_time = self._parse_start_time(text, now)
        duration_hours = self._parse_duration(text)
        rules = self._parse_rules(text)
        explicit_game_type = self._parse_explicit_game_type(text)
        used_region_default = explicit_game_type is None and self._should_apply_regional_default(
            text=text,
            intent=intent,
            missing=missing,
            stake=stake,
            parsed_time=parsed_time,
            rules=rules,
            duration_hours=duration_hours,
        )
        game_type = explicit_game_type or (
            self._regional_default_game_type() if used_region_default else "mahjong"
        )
        default_game_label = self._regional_default_label() if used_region_default else None
        if default_game_label and default_game_label not in rules:
            rules.insert(0, default_game_label)
        ruleset = self._parse_ruleset(text, game_type)
        variant = self._parse_variant(text, game_type, ruleset)
        play_options = self._parse_play_options(text, variant)

        signal_count = sum(
            [
                intent != Intent.UNKNOWN,
                missing is not None,
                level is not None,
                parsed_time.start_at is not None,
                bool(rules),
                duration_hours is not None,
                game_type != "mahjong",
            ]
        )

        if intent == Intent.UNKNOWN and signal_count >= 2:
            intent = Intent.FIND_PLAYERS

        if intent not in {Intent.FIND_PLAYERS, Intent.UPDATE_GAME} and signal_count < 3:
            return ExtractionResult(
                message_id=message.id,
                intent=intent,
                confidence=0.15 if intent == Intent.UNKNOWN else 0.35,
                raw={"text": message.text},
            )

        follow_up_questions: list[str] = []
        ambiguities = list(parsed_time.ambiguities)
        if missing is None:
            follow_up_questions.append("现在是几缺几？比如三缺一、二缺二。")
        if parsed_time.start_at is None and "人齐开" not in rules:
            follow_up_questions.append("希望几点开局？")
        for ambiguity in parsed_time.ambiguities:
            follow_up_questions.append(f"请确认：{ambiguity}。")
        if level is None:
            follow_up_questions.append("这桌打多大？比如 0.5、1、2。")

        status = GameStatus.OPEN if not follow_up_questions else GameStatus.NEED_CLARIFICATION
        confidence = min(0.95, 0.25 + signal_count * 0.12)
        if parsed_time.ambiguities:
            confidence -= 0.08
        if missing and level and parsed_time.start_at:
            confidence += 0.18
        confidence = round(max(0.1, min(confidence, 0.98)), 2)

        game = GameRequest(
            organizer_id=message.sender_id,
            organizer_name=message.sender_name,
            channel_id=message.channel_id,
            source_message_id=message.id,
            status=status,
            game_type=game_type,
            ruleset=ruleset,
            variant=variant,
            current_player_count=missing.current_player_count if missing else None,
            missing_count=missing.missing_count if missing else None,
            level=level,
            base_score=stake.base_score,
            cap_score=stake.cap_score,
            start_at=parsed_time.start_at,
            start_time_confidence=parsed_time.confidence,
            duration_hours=duration_hours,
            play_options=play_options,
            rules=rules,
            ambiguities=ambiguities,
            notes=self._notes(parsed_time, stake, default_game_label=default_game_label),
        )

        return ExtractionResult(
            message_id=message.id,
            intent=intent,
            confidence=confidence,
            game=game,
            follow_up_questions=follow_up_questions,
            raw={
                "text": message.text,
                "normalized": text,
                "missing_raw": missing.raw if missing else None,
                "time_raw": parsed_time.raw,
                "stake_raw": stake.raw,
            },
        )

    def _normalize(self, text: str) -> str:
        return normalize_mahjong_text(text).text

    def _detect_intent(self, text: str) -> Intent:
        if re.search(r"(满了|组好了|凑齐了|不用找了|不打了|取消)", text):
            return Intent.CANCEL_OR_FULL
        if re.search(r"(我来|算我|报名|可以来|加我一个|我能来)", text):
            return Intent.JOIN_GAME
        if re.search(r"(组局|开局|开桌|找人|摇人|缺|差|来一位|来一个|还少|有没有人|还有.*选手|帮.*找)", text):
            return Intent.FIND_PLAYERS
        if re.search(r"(川麻|四川麻将|成都麻将|杭麻|杭州麻将|cq|财敲|红中|捉鸡|湖南麻将|幺鸡|妖鸡|重庆麻将|重庆麻)", text):
            return Intent.FIND_PLAYERS
        if re.search(r"(有人.*(?:打|玩|搓|约).*(?:麻将|牌|麻)|(?:打|玩|搓|约).*(?:麻将|牌|麻|一把|一局).*吗|麻将.*(?:有人|有局|约吗|来吗)|(?:搓麻|搓一把|来一把|约一局))", text):
            return Intent.FIND_PLAYERS
        if re.search(r"(改到|提前|推迟|换成|变成)", text):
            return Intent.UPDATE_GAME
        return Intent.UNKNOWN

    def _parse_missing(self, text: str) -> MissingInfo | None:
        cn_or_digit = r"([一二两俩三123])"
        match = re.search(rf"{cn_or_digit}\s*(?:缺|差|等)\s*{cn_or_digit}", text)
        if match:
            current = self._to_int(match.group(1))
            missing = self._to_int(match.group(2))
            if current and missing and current + missing <= 4:
                return MissingInfo(current, missing, match.group(0))

        for code, (current, missing) in self.code_missing_map.items():
            if re.search(rf"(?<!\d){re.escape(code)}(?!\d)", text):
                return MissingInfo(current, missing, code)

        current_party = self._parse_current_party_size(text)
        if current_party is not None:
            return current_party

        one_more_patterns = [
            r"(?:缺|差|少|找|来)\s*(?:1|一|壹)\s*(?:个|位|人)?",
            r"(?:还差|还缺|再来)\s*(?:1|一|壹)\s*(?:个|位|人)?",
            r"(?:等)\s*(?:1|一|壹)\s*(?:个|位|人)?",
        ]
        if any(re.search(pattern, text) for pattern in one_more_patterns):
            return MissingInfo(3, 1, "缺一")

        two_more_patterns = [
            r"(?:缺|差|少|找|来)\s*(?:2|二|两|俩)\s*(?:个|位|人)?",
            r"(?:还差|还缺|再来)\s*(?:2|二|两|俩)\s*(?:个|位|人)?",
            r"(?:等)\s*(?:2|二|两|俩)\s*(?:个|位|人)?",
        ]
        if any(re.search(pattern, text) for pattern in two_more_patterns):
            return MissingInfo(2, 2, "缺二")

        return None

    def _parse_current_party_size(self, text: str) -> MissingInfo | None:
        count_token = r"(?P<count>1|2|3|4|一|二|两|俩|三|四)"
        patterns = [
            rf"(?:我(?:们)?这边|我们这边|我这|这边|我们|我|现在|目前|已有|已经有|一共|总共|这桌|这局)\s*"
            rf"(?:是|有|就|共|一共|已经)?\s*{count_token}\s*(?:个)?\s*(?:人|位)",
            rf"{count_token}\s*(?:个)?\s*(?:人|位)\s*(?:一起|这边|我们这边|我这边|要打|想打|组局)?",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if not match:
                continue
            current = self._to_int(match.group("count"))
            if current is None or current < 1 or current > 4:
                continue
            return MissingInfo(current, max(0, 4 - current), match.group(0))
        return None

    def _parse_stake(self, text: str) -> StakeInfo:
        explicit = re.search(
            r"(?:底注?|底|起步|底分)\s*(?P<base>\d+(?:\.\d+)?)\s*(?:封顶|封|顶)\s*(?P<cap>\d+(?:\.\d+)?)",
            text,
        )
        if explicit:
            base = self._to_number(explicit.group("base"))
            cap = self._to_number(explicit.group("cap"))
            return StakeInfo(self._format_level(base, cap), base, cap, explicit.group(0))

        slash_cap = re.search(
            r"(?<!\d)(?P<base>0\.[1-9]|[1-9]\d?)\s*(?:元|块)?\s*/\s*(?P<cap>\d{2,3})(?!\d)",
            text,
        )
        if slash_cap:
            base = self._to_number(slash_cap.group("base"))
            cap = self._to_number(slash_cap.group("cap"))
            return StakeInfo(self._format_level(base, cap), base, cap, slash_cap.group(0))

        cap_only = self._parse_cap_score(text)

        size_field = re.search(
            r"(?:大小|档位|底注|底分)\s*:?\s*(?:可\s*)?(?P<base>0\.[1-9]|[1-9]\d?)",
            text,
        )
        if size_field:
            base = self._to_number(size_field.group("base"))
            return self._stake_with_cap(base, size_field.group(0), cap_only)

        if "红中" in text:
            red_level = re.search(r"(?<!\d)(368\s*/\s*568|568\s*/\s*3|368)(?:\s*块)?(?!\d)", text)
            if red_level:
                return StakeInfo(re.sub(r"\s+", "", red_level.group(1)), None, None, red_level.group(0))

        for range_match in re.finditer(
            r"(?<!\d)(?P<base>\d+(?:\.\d+)?)\s*[-/到至]\s*(?P<cap>\d+(?:\.\d+)?)(?!\d)",
            text,
        ):
            start, end = range_match.span()
            before = text[max(0, start - 6) : start]
            after = text[end : end + 6]
            base = self._to_number(range_match.group("base"))
            cap = self._to_number(range_match.group("cap"))
            if re.search(r"(?:时间|开始)\s*:?\s*$", before) or before.endswith(":"):
                continue
            if re.match(r"\s*(?:点|时|通宵|h|开|开始)", after):
                continue
            if base is not None and cap is not None and base >= 10 and cap >= 10:
                continue
            if base is not None and cap is not None and (base >= cap or base > 10 or cap > 512):
                continue
            if base is not None and cap is not None and "." in range_match.group("base") and base >= 3 and cap <= 24:
                continue
            return StakeInfo(self._format_level(base, cap), base, cap, range_match.group(0))

        compact = re.search(r"(?<!\d)(216|132|164|232|264)(?!\d)", text)
        if compact:
            raw = compact.group(1)
            base = self._to_number(raw[0])
            cap = self._to_number(raw[1:])
            return StakeInfo(self._format_level(base, cap), base, cap, raw)

        if re.search(r"(?:五|5)\s*毛", text):
            return self._stake_with_cap(0.5, "五毛", cap_only)
        if re.search(r"(?:一|1)\s*(?:块|元|的)", text):
            return self._stake_with_cap(1, "一块", cap_only)
        if re.search(r"(?:二|两|2)\s*(?:块|元|的)", text):
            return self._stake_with_cap(2, "二块", cap_only)

        decimal = re.search(r"(?<![\d.])0\.[1-9](?!\d)", text)
        if decimal:
            value = self._to_number(decimal.group(0))
            return self._stake_with_cap(value, decimal.group(0), cap_only)

        for match in re.finditer(r"(?<![\d.])([1-9]\d?)(?:\.\d+)?(?![\d.])", text):
            raw = match.group(0)
            start, end = match.span()
            before = text[max(0, start - 2) : start]
            after = text[end : end + 2]
            if raw in self.code_missing_map:
                continue
            if after.startswith(("点", "时", "分", "个", "位", "人", "小时")):
                continue
            if before.endswith(("缺", "差", "少", "找", "来")):
                continue
            if re.search(r"(打|玩|档|块|元|的)$", before) or re.match(r"(的|档|块|元)", after):
                value = self._to_number(raw)
                return self._stake_with_cap(value, raw, cap_only)
        if cap_only:
            cap, cap_raw = cap_only
            return StakeInfo(None, None, cap, cap_raw)
        return StakeInfo(None)

    def _parse_level(self, text: str) -> str | None:
        return self._parse_stake(text).level

    def _parse_start_time(self, text: str, now: datetime) -> ParsedTime:
        time_match = self._find_time(text)
        if not time_match:
            return ParsedTime(None, 0.0, [])

        hour, minute, raw = time_match
        day_offset = 0
        if "后天" in text:
            day_offset = 2
        elif any(word in text for word in ["明天", "明晚", "明儿"]):
            day_offset = 1

        period = self._find_period(text)
        ambiguities: list[str] = []
        confidence = 0.82

        if period in {"afternoon", "evening"}:
            if 1 <= hour <= 11:
                hour += 12
            confidence = 0.92
        elif period == "morning":
            if hour == 12:
                hour = 0
            confidence = 0.9
        elif period == "noon":
            if hour < 11:
                hour += 12
            confidence = 0.86
        elif 1 <= hour <= 11:
            ambiguities.append(f"{raw} 是上午还是下午")
            hour += 12
            confidence = 0.55

        base_date = now.date() + timedelta(days=day_offset)
        start_at = datetime(
            base_date.year,
            base_date.month,
            base_date.day,
            min(hour, 23),
            minute,
            tzinfo=now.tzinfo or DEFAULT_TZ,
        )

        if day_offset == 0 and start_at < now - timedelta(minutes=30):
            start_at += timedelta(days=1)
            confidence = min(confidence, 0.68)
            ambiguities.append(f"{raw} 已经过了，是否指明天")

        return ParsedTime(start_at, round(confidence, 2), ambiguities, raw)

    def _find_time(self, text: str) -> tuple[int, int, str] | None:
        numeric_patterns = [
            r"(?<![\d.])(?P<hour>\d{1,2})\s*[:]\s*(?P<minute>\d{1,2})(?!\d)",
            r"(?<![\d.])(?P<hour>[01]?\d|2[0-3])\s*[.]\s*(?P<minute>[0-5]\d)(?!\d)",
            r"(?<![\d.])(?P<hour>\d{1,2})\s*(?:点|时)\s*(?P<half>半)?(?P<minute>\d{1,2})?\s*(?:分)?",
        ]
        for pattern in numeric_patterns:
            match = re.search(pattern, text)
            if match:
                hour = int(match.group("hour"))
                minute = 30 if match.groupdict().get("half") else int(match.groupdict().get("minute") or 0)
                if 0 <= hour <= 23 and 0 <= minute <= 59:
                    return hour, minute, match.group(0).strip()

        match = re.search(
            r"(?P<hour>十[一二]?|[一二两三四五六七八九十]{1,2})\s*(?:点|时)\s*(?P<half>半)?",
            text,
        )
        if match:
            hour = self._to_int(match.group("hour"))
            if hour is not None:
                minute = 30 if match.group("half") else 0
                return hour, minute, match.group(0).strip()
        return None

    def _find_period(self, text: str) -> str | None:
        if any(word in text for word in ["下午", "晚上", "今晚", "傍晚", "下班", "晚"]):
            return "evening"
        if any(word in text for word in ["上午", "早上", "早晨", "明早"]):
            return "morning"
        if "中午" in text:
            return "noon"
        if "凌晨" in text:
            return "morning"
        return None

    def _parse_duration(self, text: str) -> float | None:
        match = re.search(r"(?<!\d)(\d+(?:\.\d+)?)\s*(?:个)?\s*(?:小时|钟头|h)(?!\w)", text)
        if match:
            return float(match.group(1))

        match = re.search(r"([一二两三四五六七八九十]{1,3})\s*(?:个)?\s*(?:小时|钟头)", text)
        if match:
            value = self._to_int(match.group(1))
            return float(value) if value else None
        return None

    def _parse_rules(self, text: str) -> list[str]:
        rules: list[str] = []
        def add(rule: str) -> None:
            if rule not in rules:
                rules.append(rule)

        if "川麻" in text or "四川麻将" in text or "成都麻将" in text:
            add("川麻")
        if "杭麻" in text or "杭州麻将" in text or re.search(r"(?<![a-z])cq(?![a-z])|财敲", text):
            add("杭麻")
        if "红中" in text:
            add("红中")
        if "捉鸡" in text:
            add("捉鸡")
        if "湖南麻将" in text or "湖南麻" in text:
            add("湖南麻将")
        if "重庆麻将" in text or "重庆麻" in text:
            add("重庆麻将")
        if "无烟" in text or "不抽烟" in text or "禁烟" in text or re.search(r"(?:🚬|烟)\s*:?\s*(?:无|不抽)", text):
            add("无烟")
        if "少烟" in text or re.search(r"(?:🚬|烟)\s*:?\s*少", text):
            add("少烟")
        if (
            "烟都可" in text
            or "烟都行" in text
            or re.search(r"(?:🚬|烟)\s*(?:也|都)?\s*:?\s*(?:都可|都行|都可以|可以|可)", text)
            or re.search(r"烟[^，。,\n]{0,8}(?:可接受|没问题|都可|都行|都可以)", text)
            or "可烟" in text
        ):
            add("烟况都可")
        if "可抽烟" in text or "能抽烟" in text or "有烟" in text or re.search(r"(?:🚬|烟)\s*:?\s*有", text):
            add("可吸烟")
        if self._has_people_ready_start_signal(text):
            add("人齐开")
        if "包间" in text:
            add("包间")
        if "熟人" in text:
            add("熟人局")
        if "新手" in text:
            add("新手友好")
        if "预约" in text:
            add("预约")
        if "通宵" in text:
            add("通宵")
        if "可增" in text:
            add("可增")
        if re.search(r"(来个女孩子|女孩子|女生|妹子|女玩家)", text):
            add("女玩家相关")
        if "无情侣" in text or "非情侣" in text:
            add("非情侣")
        elif "情侣" in text:
            add("情侣相关")
        return rules

    def _has_people_ready_start_signal(self, text: str) -> bool:
        """Detect flexible start intent: start as soon as people are ready."""
        if "人齐开" in text or "齐人开" in text or "齐开" in text:
            return True
        patterns = [
            r"人齐(?!\s*(?:了|啦|没|吗))",
            r"(?:人够|够人|凑齐|凑够)[^，。,\n]{0,8}(?:开|开始|打|搞)",
            r"(?:尽快|越快越好)[^，。,\n]{0,8}(?:开|开始|打|搞|组)?",
            r"能早(?:点|些)?开就早(?:点|些)?开",
            r"(?:早点|早些|尽量早(?:点|些)?)[^，。,\n]{0,8}(?:开|开始|打|搞)?",
            r"时间[^，。,\n]{0,10}(?:可以|可|好)?(?:再)?商量",
            r"时间[^，。,\n]{0,10}(?:都可以|都行|好说|灵活|不固定|可商量)",
        ]
        return any(re.search(pattern, text) for pattern in patterns)

    def _parse_game_type(self, text: str) -> str:
        return self._parse_explicit_game_type(text) or "mahjong"

    def _parse_explicit_game_type(self, text: str) -> str | None:
        if "杭麻" in text or "杭州麻将" in text or re.search(r"(?<![a-z])cq(?![a-z])|财敲", text):
            return "hangzhou_mahjong"
        if "川麻" in text or "四川麻将" in text or "成都麻将" in text:
            return "sichuan_mahjong"
        if "幺鸡" in text or "妖鸡" in text:
            return "sichuan_mahjong"
        if "红中" in text:
            return "hongzhong_mahjong"
        if "捉鸡" in text:
            return "zhuoji_mahjong"
        if "湖南麻将" in text or "湖南麻" in text:
            return "hunan_mahjong"
        if "重庆麻将" in text or "重庆麻" in text:
            return "chongqing_mahjong"
        return None

    def _should_apply_regional_default(
        self,
        *,
        text: str,
        intent: Intent,
        missing: MissingInfo | None,
        stake: StakeInfo,
        parsed_time: ParsedTime,
        rules: list[str],
        duration_hours: float | None,
    ) -> bool:
        if self._regional_default_game_type() == "mahjong":
            return False
        if intent in {Intent.FIND_PLAYERS, Intent.UPDATE_GAME, Intent.JOIN_GAME}:
            return True
        if missing is not None:
            return True
        if stake.level and (parsed_time.start_at or rules or duration_hours):
            return True
        if re.search(r"(麻将|搓麻|打麻|玩麻|牌局|有局|组局|开局)", text):
            return True
        return False

    def _regional_default_game_type(self) -> str:
        return {
            "hangzhou": "hangzhou_mahjong",
            "hz": "hangzhou_mahjong",
            "杭州": "hangzhou_mahjong",
            "sichuan": "sichuan_mahjong",
            "sc": "sichuan_mahjong",
            "四川": "sichuan_mahjong",
        }.get(self.default_region, "mahjong")

    def _regional_default_label(self) -> str | None:
        return {
            "hangzhou_mahjong": "杭麻",
            "sichuan_mahjong": "川麻",
        }.get(self._regional_default_game_type())

    def _normalize_region(self, value: str) -> str:
        value = value.strip().lower()
        return {
            "杭州": "hangzhou",
            "hangzhou": "hangzhou",
            "hz": "hangzhou",
            "四川": "sichuan",
            "成都": "sichuan",
            "sichuan": "sichuan",
            "sc": "sichuan",
            "chengdu": "sichuan",
        }.get(value, value)

    def _parse_ruleset(self, text: str, game_type: str) -> str | None:
        if game_type == "hangzhou_mahjong":
            return "hangzhou_mahjong"
        if "幺鸡" in text or "妖鸡" in text:
            return "yaoji_mahjong"
        if game_type != "mahjong":
            return game_type
        return None

    def _parse_variant(self, text: str, game_type: str, ruleset: str | None) -> str | None:
        if game_type == "hangzhou_mahjong" and re.search(r"(?<![a-z])cq(?![a-z])|财敲", text):
            return "caiqiao"
        if ruleset == "yaoji_mahjong":
            if "素鸡" in text:
                return "suji"
            if re.search(r"幺鸡\s*47|妖鸡\s*47", text):
                return "yaoji_47"
            return "yaoji"
        if game_type == "hongzhong_mahjong" and "鲨鱼" in text:
            return "shayu"
        return None

    def _parse_play_options(self, text: str, variant: str | None) -> list[str]:
        options: list[str] = []
        def add(option: str) -> None:
            if option not in options:
                options.append(option)

        if variant == "caiqiao":
            add("财敲")
        if variant == "suji":
            add("素鸡")
        elif variant == "yaoji_47":
            add("幺鸡47")
        elif variant == "yaoji":
            add("幺鸡")
        elif variant == "shayu":
            add("鲨鱼")
        if "爆炸码" in text:
            add("爆炸码")
        elif "爆炸" in text:
            add("爆炸")
        if re.search(r"换\s*(?:三|3)\s*张", text):
            add("换三张")
        if "定缺" in text:
            add("定缺")
        if re.search(r"(?:三|3)\s*财翻", text):
            add("3财翻")
        if re.search(r"(?:四|4)\s*财翻", text):
            add("4财翻")
        elif re.search(r"(?<!\d)(?:四|4)\s*翻", text):
            add("4翻")
        for option in ["软跟", "硬跟", "不跟", "吃两摊", "碰无限", "十风", "跳碰亮白"]:
            if option in text:
                add(option)
        return options

    def _notes(self, parsed_time: ParsedTime, stake: StakeInfo, *, default_game_label: str | None = None) -> list[str]:
        notes: list[str] = []
        if default_game_label:
            notes.append(f"按当前地区默认玩法：{default_game_label}")
        if parsed_time.raw is not None:
            notes.append(f"时间原文：{parsed_time.raw}")
        if stake.raw is not None and stake.cap_score is not None:
            if stake.base_score is not None:
                notes.append(f"档位原文：{stake.raw}，底注{stake.base_score:g}，封顶{stake.cap_score:g}")
            else:
                notes.append(f"封顶原文：{stake.raw}，封顶{stake.cap_score:g}")
        return notes

    def _parse_cap_score(self, text: str) -> tuple[float, str] | None:
        patterns = [
            r"(?<!\d)(?P<cap>\d+(?:\.\d+)?)\s*(?:封顶|上限|封)(?!\d)",
            r"(?:封顶|封|上限)\s*(?P<cap>\d+(?:\.\d+)?)(?!\d)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if not match:
                continue
            cap = self._to_number(match.group("cap"))
            if cap is not None:
                return cap, match.group(0)
        return None

    def _stake_with_cap(
        self,
        base: float | None,
        raw: str,
        cap_only: tuple[float, str] | None,
    ) -> StakeInfo:
        if cap_only is None:
            return StakeInfo(self._format_level(base, None), base, None, raw)
        cap, cap_raw = cap_only
        return StakeInfo(self._format_level(base, None), base, cap, f"{raw}; {cap_raw}")

    def _to_int(self, value: str | None) -> int | None:
        if value is None:
            return None
        value = value.strip()
        if value.isdigit():
            return int(value)
        if value in CN_NUMBERS:
            return CN_NUMBERS[value]
        if value.startswith("十") and len(value) == 2:
            return 10 + CN_NUMBERS.get(value[1], 0)
        if "十" in value:
            left, _, right = value.partition("十")
            tens = CN_NUMBERS.get(left, 1)
            ones = CN_NUMBERS.get(right, 0) if right else 0
            return tens * 10 + ones
        return None

    def _to_number(self, value: str | None) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except ValueError:
            return None

    def _format_level(self, base: float | None, cap: float | None) -> str | None:
        if base is None:
            return None
        if cap is None:
            return f"{base:g}"
        return f"{base:g}-{cap:g}"

    def _ensure_tz(self, dt: datetime) -> datetime:
        if dt.tzinfo is None:
            return dt.replace(tzinfo=DEFAULT_TZ)
        return dt
