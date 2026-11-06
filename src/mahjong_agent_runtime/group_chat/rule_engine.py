"""只处理完整显式协议，开放式群消息交由语义入口识别业务意图。"""

from __future__ import annotations

from .models import GroupMessage, L1Result
from .parsing import parse_claim_item_no, parse_game_post


class L1RuleEngine:
    """零模型成本处理显式协议，宁可漏掉也不误写入。"""

    def __init__(self, *, bot_id: str) -> None:
        """保存自身通道身份，过滤机器人回声而不推断业务意图。"""
        self.bot_id = str(bot_id or "")

    def process(self, msg: GroupMessage) -> L1Result:
        """只有完整协议可自动执行；自然语言余文不因包含人数或字段而获得授权。"""
        if self.bot_id and msg.sender_external_id == self.bot_id:
            return L1Result(action="ignore")
        parsed = parse_game_post(msg.text, anchor=msg.sent_at)
        if parsed is not None:
            return L1Result(
                action="board_import",
                parsed_game=parsed,
                needs_ack=False,
            )
        item_no = parse_claim_item_no(msg.text)
        if item_no is not None:
            return L1Result(action="claim", item_no=item_no)
        # “好”与表情也可能答复本人当前主题的待确认事项，不能脱离上下文静默丢弃。
        return L1Result(action="pass_to_L2")


__all__ = ["L1RuleEngine"]
