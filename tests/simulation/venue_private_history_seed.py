"""从已确认的真人私聊时间线选取可交付的历史前缀。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
from typing import Literal


DEFAULT_PRIVATE_HISTORY_DATASET = (
    Path(__file__).resolve().parents[2] / "eval/golden/real_owner_chat_timeline_20260705_20260718.json"
)
DEFAULT_PRIVATE_HISTORY_CASE_ID = "owner_chat_multi_episode_timeline_20260705_20260718_001"
DEFAULT_PRIVATE_HISTORY_EPISODE_ID = "availability_then_smoke_preference"
HOLDOUT_PRIVATE_HISTORY_DATASET = DEFAULT_PRIVATE_HISTORY_DATASET.with_name(
    "real_owner_chat_timeline_20260627_20260719_owner_b.json"
)
HOLDOUT_PRIVATE_HISTORY_CASE_ID = "owner_b_chat_multi_episode_timeline_20260627_20260719_001"
PRIVATE_HISTORY_EPISODES = ("availability", "holdout-friend")
_SAFE_CASE_ID = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]*")
_CONTACT_OR_LINK = re.compile(r"wxid_|微信号|https?://|(?<!\d)1[3-9]\d{9}(?!\d)", re.IGNORECASE)


def load_reviewed_private_episode(name: str, *, cutoff_turn: int | None = None) -> HistoricalPrivateSeed:
    """选择已核实的常用片段或独立留出的老板 B 私聊。"""
    if name == "availability":
        return load_historical_private_seed(cutoff_turn=2 if cutoff_turn is None else cutoff_turn)
    if name == "holdout-friend":
        return load_historical_private_seed(HOLDOUT_PRIVATE_HISTORY_DATASET, case_id=HOLDOUT_PRIVATE_HISTORY_CASE_ID,
            episode_id="friend_as_separate_participant", cutoff_turn=62 if cutoff_turn is None else cutoff_turn)
    raise ValueError(f"未知的已核实私聊片段：{name}")


@dataclass(frozen=True, slots=True)
class HistoricalPrivateMessage:
    """已有私聊的一条真实消息，只携带顺序、角色、时间和正文。"""

    source_message_id: str
    turn: int
    role: Literal["customer", "boss"]
    observed_at: datetime
    offset_seconds: int
    text: str


@dataclass(frozen=True, slots=True)
class HistoricalPrivateSeed:
    """一段在老板实际回复处截断的私聊，历史后续仅以条数保留。"""

    case_id: str
    episode_id: str
    cutoff_turn: int
    prefix: tuple[HistoricalPrivateMessage, ...]
    held_out_count: int

    @property
    def started_at(self) -> datetime:
        """以这个业务片段最早可见的一条消息作为时间锚点。"""
        return self.prefix[0].observed_at

    @property
    def cutoff_at(self) -> datetime:
        """以最后一条投递给顾客的老板回复作为续演的因果起点。"""
        return self.prefix[-1].observed_at


def load_historical_private_seed(
    path: Path = DEFAULT_PRIVATE_HISTORY_DATASET,
    *,
    case_id: str = DEFAULT_PRIVATE_HISTORY_CASE_ID,
    episode_id: str = DEFAULT_PRIVATE_HISTORY_EPISODE_ID,
    cutoff_turn: int = 2,
) -> HistoricalPrivateSeed:
    """只投影选定业务片段的已观察消息，留出真实后续和所有评测信息。"""
    record = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(record, dict) or record.get("kind") != "real_owner_chat_golden"
            or record.get("id") != case_id or not _SAFE_CASE_ID.fullmatch(case_id)):
        raise ValueError(f"未找到已确认的老板私聊来源：{case_id}")
    source = record.get("source")
    if (not isinstance(source, dict) or source.get("channel") != "wechat_screenshot"
            or source.get("timezone") != "Asia/Shanghai"):
        raise ValueError("私聊时间线须保留已核实的微信截图来源及北京时间")
    episodes, messages = record.get("episodes"), record.get("messages")
    if not isinstance(episodes, list) or not isinstance(messages, list):
        raise ValueError("私聊时间线缺少业务片段或消息")
    matches = [episode for episode in episodes if isinstance(episode, dict) and episode.get("id") == episode_id]
    if len(matches) != 1:
        raise ValueError(f"私聊时间线中没有唯一的业务片段：{episode_id}")
    episode = matches[0]
    start_turn, end_turn = episode.get("start_turn"), episode.get("end_turn")
    if any(type(turn) is not int for turn in (start_turn, end_turn, cutoff_turn)) or not (
        1 <= start_turn <= cutoff_turn < end_turn <= len(messages)
    ):
        raise ValueError("私聊切点须在已观察的业务片段内并留出真实续轮")

    prefix: list[HistoricalPrivateMessage] = []
    for turn in range(start_turn, cutoff_turn + 1):
        row = messages[turn - 1]
        if not isinstance(row, dict) or type(row.get("turn")) is not int or row["turn"] != turn:
            raise ValueError(f"私聊来源缺少顺序正确的第 {turn} 条消息")
        role, stamp, text = row.get("role"), row.get("observed_at"), row.get("text")
        if role not in ("customer", "boss") or not isinstance(stamp, str) or not isinstance(text, str) or not text.strip():
            raise ValueError(f"私聊来源第 {turn} 条的角色、时间或正文无效")
        if _CONTACT_OR_LINK.search(text):
            raise ValueError(f"私聊来源第 {turn} 条仍包含联系方式或链接")
        observed_at = datetime.fromisoformat(stamp)
        if observed_at.utcoffset() is None or prefix and observed_at < prefix[-1].observed_at:
            raise ValueError(f"私聊来源第 {turn} 条缺少时区或时间倒序")
        anchor = prefix[0].observed_at if prefix else observed_at
        prefix.append(HistoricalPrivateMessage(f"history-dm:{case_id}:{turn}", turn, role, observed_at,
                                               int((observed_at - anchor).total_seconds()), text))

    first_held_out = messages[cutoff_turn]
    if (prefix[-1].role != "boss" or not isinstance(first_held_out, dict)
            or first_held_out.get("turn") != cutoff_turn + 1 or first_held_out.get("role") != "customer"):
        raise ValueError("私聊切点须是一条后面确有顾客续轮的老板回复")
    return HistoricalPrivateSeed(case_id, episode_id, cutoff_turn, tuple(prefix), end_turn - cutoff_turn)
