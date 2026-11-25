"""将审核过的真人群聊前缀投递到滚动营业演练的聊天账本。"""

from __future__ import annotations

from collections.abc import Collection, Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from .venue_channel import MessageEnvelope, SandboxChannel
from .venue_history_seed import DEFAULT_HISTORY_DATASET, load_historical_group_seed


_DATASET = DEFAULT_HISTORY_DATASET.with_name("real_group_chat_20261122.jsonl")
_CASE_ID = "real_group_smoke_alternative_20261122_001"
_CUTOFF = "m02"
_LEAD_SECONDS = 120


@dataclass(frozen=True, slots=True)
class RollingHistoryDelivery:
    """可供群内关注唤醒使用的最后一条前缀消息及其来源。"""

    last_message_id: str
    last_author_id: str
    last_message_at: datetime
    source_date: str
    relative_offsets_seconds: tuple[int, ...]
    quote_links: int
    held_out_messages: int
    active_archive_authors: int

    def audit(self) -> dict:
        """仅输出来源、投递和时间证据，不输出聊天正文或作者映射。"""
        return {
            "case": _CASE_ID,
            "cutoff": _CUTOFF,
            "source_date": self.source_date,
            "time_basis": "reanchored",
            "lead_seconds": _LEAD_SECONDS,
            "last_message_at": self.last_message_at.isoformat(),
            "last_message_id": self.last_message_id,
            "prefix_messages": len(self.relative_offsets_seconds),
            "relative_offsets_seconds": list(self.relative_offsets_seconds),
            "quote_links": self.quote_links,
            "held_out_messages": self.held_out_messages,
            "active_archive_authors": self.active_archive_authors,
        }


def deliver_rolling_group_history(channel: SandboxChannel, *, roster_ids: Iterable[str],
                                  active_actor_ids: Collection[str], start_at: datetime) -> RollingHistoryDelivery:
    """按原间隔投递前两条群消息，只从聊天账本恢复，旧来话不进入店家业务入口。"""
    if start_at.utcoffset() is None:
        raise ValueError("滚动群聊历史需要带时区的场景起点")
    seed = load_historical_group_seed(_DATASET, case_id=_CASE_ID, cutoff=_CUTOFF)
    roster = tuple(roster_ids)
    active = set(active_actor_ids)
    if (any(not isinstance(member, str) or not member.strip() for member in roster)
            or len(set(roster)) != len(roster) or not active <= set(roster)):
        raise ValueError("滚动群聊历史需要稳定、无重复的同群角色名册")
    author_count = len(seed.actor_aliases)
    if len(roster) < author_count:
        raise ValueError("滚动群聊历史缺少足够的归档作者")
    inactive = tuple(member for member in roster if member not in active)
    selected = inactive[:author_count] if len(inactive) >= author_count else roster[-author_count:]
    authors = dict(zip(seed.actor_aliases, selected, strict=True))
    for member in selected:
        channel.events_for(member)  # 预检全部作者，避免在名册缺项时写下一半前缀。

    last_at = start_at.astimezone(timezone.utc) - timedelta(seconds=_LEAD_SECONDS)
    first_at = last_at - timedelta(seconds=seed.prefix[-1].offset_seconds)
    namespace = f"history:rolling:{seed.case_id}:"
    prefix = tuple(MessageEnvelope(
        message_id=namespace + message.message_alias, sender_id=authors[message.actor_alias], text=message.text,
        sent_at=first_at + timedelta(seconds=message.offset_seconds), kind="group", recipient_id=channel.room_id,
        quoted_message_id=namespace + message.quoted_message_alias if message.quoted_message_alias else None,
    ) for message in seed.prefix)
    recorded = [row for row in channel.events() if row["message_id"].startswith(namespace)]
    if [row["message_id"] for row in recorded] != [message.message_id for message in prefix[:len(recorded)]]:
        raise ValueError("滚动群聊账本含非前缀记录或前缀顺序冲突")
    for row, message in zip(recorded, prefix):
        if (row["direction"], row["kind"], row["sender_id"], row["recipient_id"], row["text"],
                datetime.fromisoformat(row["sent_at"]), row["quoted_message_id"], row["source_message_id"]) != (
                "history", "group", message.sender_id, message.recipient_id, message.text,
                message.sent_at, message.quoted_message_id, None):
            raise ValueError("滚动群聊账本与已审核前缀不一致")
    # 历史群聊写入账本即对群友可见。恢复时已见的原行不用重试渠道回执。
    for message in prefix[len(recorded):]:
        receipt = channel.deliver_group_history(message)
        if receipt["status"] != "delivered" or receipt.get("archive_outputs"):
            raise RuntimeError("历史群消息必须只留在公开聊天账本")

    last = prefix[-1]
    return RollingHistoryDelivery(
        last.message_id, last.sender_id, last.sent_at, seed.capture_date.isoformat(),
        tuple(message.offset_seconds for message in seed.prefix),
        sum(message.quoted_message_alias is not None for message in seed.prefix), seed.held_out_count,
        sum(member in active for member in selected),
    )
