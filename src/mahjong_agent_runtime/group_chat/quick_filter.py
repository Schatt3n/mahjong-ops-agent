"""实现群聊场景中的quickfilter能力。

"""

from __future__ import annotations

import re
import unicodedata

from .models import GroupMessage


_LAUGHTER = re.compile(r"^(?:哈|呵|嘿|嘻|[hH]){4,}[啊呀哟哦~～！!。.]?$")
_BUSINESS_SIGNAL = re.compile(
    r"麻将|杭麻|川麻|红中|财敲|cq|无烟|有烟|人齐开|三缺一|二缺二|一缺三|"
    r"173|272|371|几点|几个人|有局|还有位|我来|我打|加我|帮我|组局|约局|"
    r"可以|行|好|ok|打|来|[+＋]1|\d"
)


class QuickFilter:
    """仅剔除明显噪声；语义含糊的自然语言继续交由下游处理。"""

    def should_filter(self, message: GroupMessage) -> bool:
        content_type = str(message.metadata.get("content_type") or "").strip().lower()
        if content_type in {"sticker_xml", "sticker", "pin_event", "unpin_event", "pat_event"}:
            return True
        text = "".join((message.text or "").split())
        if not text:
            return True
        if _BUSINESS_SIGNAL.search(text):
            return False
        if _LAUGHTER.fullmatch(text):
            return True
        if self._emoji_or_punctuation_only(text):
            return True
        return len(text) < 3

    @staticmethod
    def _emoji_or_punctuation_only(text: str) -> bool:
        meaningful = [character for character in text if not character.isspace()]
        return bool(meaningful) and all(
            unicodedata.category(character)[0] in {"P", "S"} for character in meaningful
        )


__all__ = ["QuickFilter"]
