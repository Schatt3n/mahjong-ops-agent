"""把匿名真人群聊切成角色可续演的已发生前缀。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time
from pathlib import Path
import re
from typing import Any

from scripts.validate_real_group_chat_dataset import validate_datasets


DEFAULT_HISTORY_DATASET = Path(__file__).resolve().parents[2] / "eval/adversarial/real_group_chat_20261118.jsonl"
DEFAULT_HISTORY_CASE_ID = "real_group_smoking_and_arrival_coordination_001"
_ALIAS = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,79}")
_CONTENT_TYPES = {"text", "quote_reply", "emoji"}


@dataclass(frozen=True, slots=True)
class HistoricalGroupMessage:
    """一条历史消息的公开内容、时间及已出现的引用。"""

    message_alias: str
    actor_alias: str
    offset_seconds: int
    content_type: str
    text: str
    quoted_message_alias: str | None
    mentioned_actor_aliases: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HistoricalGroupSeed:
    """可追溯的历史前缀，后续真人发言只保留条数用于报告。"""

    case_id: str
    capture_date: date
    timeline_start_local: time
    cutoff: str
    prefix: tuple[HistoricalGroupMessage, ...]
    held_out_count: int

    @property
    def actor_aliases(self) -> tuple[str, ...]:
        """按首次在前缀公开出现的顺序获取说话人和被提及的人。"""
        return tuple(dict.fromkeys(alias for message in self.prefix
                                   for alias in (message.actor_alias, *message.mentioned_actor_aliases)))


def load_historical_group_seed(
    path: Path = DEFAULT_HISTORY_DATASET, *, case_id: str = DEFAULT_HISTORY_CASE_ID, cutoff: str = "m04",
) -> HistoricalGroupSeed:
    """校验匿名来源和因果顺序，只返回截至指定消息的公开字段。"""
    records, errors = validate_datasets([path])
    if errors:
        raise ValueError("\n".join(errors))
    case = next((record for record in records if record["id"] == case_id), None)
    if case is None or case["case_type"] != "group_topic_timeline":
        raise ValueError(f"未找到群聊时间线：{case_id}")
    anchor = case.get("timeline_start_local")
    if not isinstance(anchor, str):
        raise ValueError("群聊时间线缺少本地起始时间")
    local_start = time.fromisoformat(anchor)
    if local_start.tzinfo is not None:
        raise ValueError("时间线起点必须是无时区的本地时间")

    seen: set[str] = set()
    prefix: list[HistoricalGroupMessage] = []
    for index, raw in enumerate(case["messages"], start=1):
        message = _public_message(raw, seen)
        prefix.append(message)
        if message.message_alias == cutoff:
            return HistoricalGroupSeed(case_id, date.fromisoformat(case["source"]["capture_date"]), local_start,
                                       cutoff, tuple(prefix), len(case["messages"]) - index)
        seen.add(message.message_alias)
    raise ValueError(f"群聊时间线没有切点：{cutoff}")


def _public_message(raw: dict[str, Any], seen: set[str]) -> HistoricalGroupMessage:
    """只提取安全公开字段，拒绝前缀内无来源的引用。"""
    alias, actor = raw.get("message_alias"), raw.get("actor_alias")
    if (not isinstance(alias, str) or not _ALIAS.fullmatch(alias) or alias in seen
            or not isinstance(actor, str) or not _ALIAS.fullmatch(actor)):
        raise ValueError("群聊消息或发言人别名无效")
    offset, content_type, text = raw["offset_seconds"], raw.get("content_type"), raw["text"]
    if (type(offset) is not int or raw["role"] != "customer"
            or not isinstance(content_type, str) or content_type not in _CONTENT_TYPES):
        raise ValueError(f"{alias} 的角色、类型或时间偏移无效")
    if (content_type == "emoji") == bool(text.strip()):
        raise ValueError(f"{alias} 的正文和消息类型不符")
    reference = raw.get("target_message_alias")
    if content_type == "quote_reply":
        if not isinstance(reference, str) or reference not in seen:
            raise ValueError(f"{alias} 引用了尚未出现的消息")
    elif reference is not None:
        raise ValueError(f"{alias} 的消息类型不允许引用")
    mentions = raw.get("mentioned_actor_aliases", [])
    if not isinstance(mentions, list) or any(not isinstance(item, str) or not _ALIAS.fullmatch(item)
                                             for item in mentions) or len(mentions) != len(set(mentions)):
        raise ValueError(f"{alias} 的提及对象无效")
    return HistoricalGroupMessage(alias, actor, offset, content_type, text, reference, tuple(mentions))
