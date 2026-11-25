"""从同一真人群聊切点生成一项因素不同的反事实对照。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time

from .venue_history_seed import HistoricalGroupMessage, HistoricalGroupSeed, load_historical_group_seed


HISTORY_GROUP_VARIANTS = (
    "smoke-pending", "smoke-confirmed", "arrival-2100", "arrival-2140", "question-unquoted", "question-quoted",
)


@dataclass(frozen=True, slots=True)
class HistoricalGroupVariant:
    """真人前缀保持原样，外加材料仅由模拟器正式公开或交给指定角色。"""

    variant_id: str
    condition: str
    seed: HistoricalGroupSeed
    message: HistoricalGroupMessage | None = None
    focus_actor_alias: str | None = None
    earliest_arrival: time | None = None


def load_group_history_variant(name: str) -> HistoricalGroupVariant:
    """仅变更指定条件，切点之后的真人来话仍由原素材留出。"""
    if name not in HISTORY_GROUP_VARIANTS:
        raise ValueError(f"未知的历史群聊对照：{name}")
    if name.startswith("arrival-"):
        seed = load_historical_group_seed(cutoff="m18")
        earliest = time(21, 40 if name == "arrival-2140" else 0)
        return HistoricalGroupVariant(name, earliest.strftime("%H:%M"), seed,
                                      focus_actor_alias="customer_5", earliest_arrival=earliest)

    if name.startswith("smoke-"):
        seed = load_historical_group_seed(cutoff="m04")
        confirmed = name == "smoke-confirmed"
        condition = "confirmed_no_smoking" if confirmed else "pending_confirmation"
        text = "刚才那桌，已经确认无烟。" if confirmed else "刚才那桌，无烟还没确认。"
        speaker = seed.prefix[0].actor_alias  # 发起人补充同一桌的已知或未定条件。
        quoted = seed.prefix[0].message_alias
    else:
        seed = load_historical_group_seed(cutoff="m25")
        quoted = seed.cutoff if name == "question-quoted" else None
        condition = "quoted" if quoted else "unquoted"
        text = "这桌几点开？"
        speaker = "simulated_inquirer"

    message = HistoricalGroupMessage(
        message_alias="simulated-variant", actor_alias=speaker, offset_seconds=seed.prefix[-1].offset_seconds + 1,
        content_type="quote_reply" if quoted else "text", text=text, quoted_message_alias=quoted,
        mentioned_actor_aliases=(),
    )
    return HistoricalGroupVariant(name, condition, seed, message=message)
