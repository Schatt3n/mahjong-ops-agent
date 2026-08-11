"""实现群聊场景中的ruleengine能力。

"""

from __future__ import annotations

from .models import GroupMessage, L1Result
from .parsing import parse_claim_item_no, parse_game_post


class L1RuleEngine:
    """零模型成本处理显式协议，宁可漏掉也不误写入。"""

    ACK_SIGNALS = ("老板", "帮我", "帮忙", "麻烦", "能不能", "可以吗", "看看", "安排", "挂上")
    ACK_POOL = ("好的", "okk", "收到", "好", "ok", "行")

    def __init__(self, *, bot_id: str) -> None:
        self.bot_id = str(bot_id or "")

    def process(self, msg: GroupMessage) -> L1Result:
        if self.bot_id and msg.sender_external_id == self.bot_id:
            return L1Result(action="ignore")
        parsed = parse_game_post(msg.text, anchor=msg.sent_at)
        if parsed is not None:
            return L1Result(
                action="board_import",
                parsed_game=parsed,
                needs_ack=any(signal in msg.text for signal in self.ACK_SIGNALS),
            )
        item_no = parse_claim_item_no(msg.text)
        if item_no is not None:
            return L1Result(action="claim", item_no=item_no)
        if self._is_emoji_only(msg.text) or self._is_short_noise(msg.text):
            return L1Result(action="ignore")
        return L1Result(action="pass_to_L2")

    @staticmethod
    def _is_emoji_only(text: str) -> bool:
        compact = "".join(str(text or "").split())
        return bool(compact) and not any(character.isalnum() or "\u4e00" <= character <= "\u9fff" for character in compact)

    @staticmethod
    def _is_short_noise(text: str) -> bool:
        return str(text or "").strip().lower() in {"哈", "哈哈", "嗯", "哦", "好", "ok", "okk", "收到"}


__all__ = ["L1RuleEngine"]
