"""桌群店家侧协商，只解释实际收到的消息，暂不对接真实微信。"""

from datetime import datetime
import json

from mahjong_agent_runtime.llm import completion_diagnostics, model_response_source
from mahjong_agent_runtime.models import GameStatus

from .venue_channel import AGENT_ID, MessageEnvelope
from .venue_table_opening import VenueTableOpening


_PROPOSAL_PREFIX = "table-opening-offer|"
_CONFIRM_PREFIX = "table-opening-confirm|"
_PROMPT = """你在模拟麻将馆的四人桌群负责约开场。只用本群已送达的话，没收到的同意不能补写。
vote 仅判断 current 这一条顾客消息对 active_proposal 的态度。明确同意该版确切开场用 accept；明确不行或反提其他时间用 decline。问句、估计、条件式接受、对旧版的回复及无开场提议时，一律用 other。引用其他提议不能当成当前的同意。
proposed_start 仅在本群顾客已说出可用时间或给出足够协商信息，且没有提议或当前有人明确拒绝时填写。用当地带时区 ISO 分钟，选一个未来开场，考虑已经说出的到店、离店和可打时长。若当前时间有人同意但尚无拒绝，等其他人，不要为了促成局改时间。无合适时间时为 null，不从身份或画像猜。
只输出 JSON {"vote":"accept|decline|other","proposed_start":null}。"""

class VenueTableShop:
    """给滚动沙盒挂接独立的桌群处理器，同时保留原始判断和真实通道回执。"""

    def __init__(self, client, trace_path) -> None:
        self.client = client
        self.trace_path = trace_path

    def __call__(self, group, reply_to, at):
        game = group.store.require_game(group.game_id)
        if game.status != GameStatus.READY or game.actual_started_at:
            return None
        opening = VenueTableOpening(group)
        records = tuple(row for row in group.delivered_receipts() if row.message.sent_at <= at)
        proposed = [row for row in records if row.message.sender_id == AGENT_ID
                    and row.message.message_id.startswith(_PROPOSAL_PREFIX)]
        for receipt in proposed:
            start = datetime.fromisoformat(receipt.message.message_id.split("|", 2)[1])
            opening.record_proposal(receipt, start_at=start)
        active = opening.current()
        if active and active["committed"]:
            return self._confirmation(group, at, active, records)
        if game.planned_start_at is not None:
            return None
        if reply_to is None:
            return self._message(group, f"table-opening-inquiry:{group.game_id}", at,
                                 "人齐了，大家今晚各自几点能到？咱们在这儿定个开场。")

        positions = {row.message.message_id: index for index, row in enumerate(records)}
        source = reply_to.message
        observed = active if (active and positions[active["proposal_id"]] < positions[source.message_id]
                              and source.quoted_message_id in {None, active["proposal_id"]}) else None
        vote, start = self._decide(group, reply_to, at, records, observed)
        result = None
        if observed and vote in {"accept", "decline"}:
            result = opening.record_vote(reply_to, proposal_id=observed["proposal_id"], accepts=vote == "accept")
        if result == "committed":
            return self._confirmation(group, at, opening.current(), records)
        if (start is None or start <= at or start.second or start.microsecond or len(proposed) >= 3
                or active and (observed is None or result != "declined" or start.isoformat() == active["start_at"])):
            return None
        message_id = f"{_PROPOSAL_PREFIX}{start.isoformat()}|{source.message_id}"
        return opening.proposal_message(message_id, at=at, start_at=start)

    def _decide(self, group, reply, at, records, active):
        """单轮只解释当前发言和同桌公开历史，调用失败继续保留待协商状态。"""
        body = {
            "now": at.isoformat(),
            "current": reply.history_row(),
            "active_proposal": active,
            "history": [
                {"speaker": "shop" if row.message.sender_id == AGENT_ID else row.message.sender_id,
                 "message_id": row.message.message_id, "text": row.message.text,
                 "sent_at": row.message.sent_at.isoformat(), "quoted_message_id": row.message.quoted_message_id}
                for row in records[-24:]
            ],
        }
        # 只上传桌群现有事实；真实顾客的私人出行安排始终由各自角色保留。
        body["current"] = {key: body["current"][key] for key in
                           ("message_id", "sender_id", "text", "sent_at", "quoted_message_id")}
        trace_id = f"table-shop:{group.game_id}:{reply.message.message_id}"
        evidence = {"trace_id": trace_id, "input": body}
        vote, start = "other", None
        try:
            raw = self.client.complete(
                [{"role": "system", "content": _PROMPT},
                 {"role": "user", "content": json.dumps(body, ensure_ascii=False)}],
                trace_id=trace_id, timeout_seconds=25,
            )
            evidence.update(raw=str(raw), response_source=model_response_source(raw, trace_id=trace_id, step_index=1))
            result = json.loads(str(raw))
            if set(result) != {"vote", "proposed_start"} or result["vote"] not in ("accept", "decline", "other"):
                raise ValueError("invalid table shop decision")
            if result["proposed_start"] is not None:
                start = datetime.fromisoformat(result["proposed_start"])
                if start.utcoffset() is None:
                    raise ValueError("table shop proposed opening has no timezone")
                start = start.astimezone(at.tzinfo)
            vote = result["vote"]
        except Exception as exc:
            evidence.update(error=type(exc).__name__, diagnostics=completion_diagnostics(exc))
            vote, start = "other", None
        with self.trace_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(evidence, ensure_ascii=False) + chr(10))
        return vote, start

    def _confirmation(self, group, at, active, records):
        """已落相同牌局时才报定时，重跑复用原群消息。"""
        message_id = f"{_CONFIRM_PREFIX}{active['proposal_id']}"
        if any(row.message.message_id == message_id for row in records):
            return None
        start = datetime.fromisoformat(active["start_at"])
        when = start.strftime("%H:%M" if start.date() == at.date() else "%m月%d日 %H:%M")
        return self._message(group, message_id, at, f"那就 {when} 开，大家按这个时间到哈。")

    @staticmethod
    def _message(group, message_id, at, text):
        return MessageEnvelope(message_id, AGENT_ID, text, at, "group", group.room_id)
