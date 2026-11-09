"""过滤自身回声、识别编号认领，其余群消息交由语义入口理解。"""

from __future__ import annotations

from .models import GroupMessage, L1Result
from .parsing import parse_claim_item_no


class L1RuleEngine:
    """零模型成本处理自身回声与明确的看板编号认领协议。"""

    def __init__(self, *, bot_id: str) -> None:
        """保存自身通道身份，过滤机器人回声而不推断业务意图。"""
        self.bot_id = str(bot_id or "")

    def process(self, msg: GroupMessage) -> L1Result:
        """识别编号认领。群友自发报局即使字段齐全，也要结合上下文判断是否委托店方。"""
        if self.bot_id and msg.sender_external_id == self.bot_id:
            return L1Result(action="ignore")
        item_no = parse_claim_item_no(msg.text)
        if item_no is not None:
            return L1Result(action="claim", item_no=item_no)
        # 完整报局和简短补充都可能接续本人的已授权任务，交给语义层核对上下文。
        return L1Result(action="pass_to_L2")


__all__ = ["L1RuleEngine"]
