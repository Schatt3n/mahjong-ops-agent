"""沿正式排期发布群看板，按同一牌局核对跨频道的报名与私聊回执。"""

from collections import defaultdict
from datetime import datetime
import re

from mahjong_agent_runtime.group_chat import BOARD_TASK_TYPE
from mahjong_agent_runtime.models import ScheduledTaskStatus

from .venue_channel import AGENT_ID


_SEAT_CODE = re.compile(r"(?<!\S)([0-4])7([0-4])(?!\S)")


class VenueRollingBoard:
    """复用店家库中的发布任务，使用独立聊天账本证明看板确已送达。"""

    def __init__(self, channel):
        """从沙盒通道取得正式业务库，留存当前群和所属会话。"""
        self.channel = channel
        self.store = channel.runtime.store

    def next_due(self) -> datetime | None:
        """找到本群下一个可领取的看板时刻，保留正式任务的租约边界。"""
        due = []
        for task in self.store.scheduled_tasks.values():
            if (task.task_type != BOARD_TASK_TYPE or task.aggregate_type != "group_room"
                    or task.aggregate_id != self.channel.room_id):
                continue
            if task.status == ScheduledTaskStatus.PENDING:
                due.append(task.due_at)
            elif task.status == ScheduledTaskStatus.PROCESSING and task.lease_until is not None:
                due.append(max(task.due_at, task.lease_until))
        return min(due, default=None)

    def publish_due(self, at: datetime) -> dict:
        """到点后使用正式工作器发布，真实授权、失败和重试都由该入口处理。"""
        return self.channel.publish_board(trace_id=f"rolling-board:{at.isoformat()}", sent_at=at)

    def receipts_at(self, at: datetime) -> list[dict]:
        """用业务快照和渠道双账本恢复该时刻已经发布的同一份回执。"""
        records = [message for message in self.channel.events() if message["kind"] == "group"
                   and datetime.fromisoformat(message["sent_at"]) == at]
        verified = self._verified_boards(records)
        return [message for message, _ in verified]

    def latest_quoted_parent(self, recent_group: list[dict], visible_by_id: dict[str, dict]) -> tuple[str, str] | None:
        """找近期同群消息直接引用且本人已经收到的正式看板。"""
        for source in reversed(recent_group):
            if (source.get("kind"), source.get("recipient_id")) != ("group", self.channel.room_id):
                continue
            parent = visible_by_id.get(source.get("quoted_message_id"))
            if (parent is not None and parent["sequence"] < source["sequence"]
                    and datetime.fromisoformat(parent["sent_at"]) <= datetime.fromisoformat(source["sent_at"])
                    and self._verified_boards([parent])):
                return parent["message_id"], source["message_id"]
        return None

    def published_item_expirations(self, records: list[dict]) -> dict[str, list[dict]]:
        """保留真实送达的看板条目及其到期证据，由评审按每条来话的时点筛选。"""
        expired = {row.entity_id: row.occurred_at for row in self.store.transitions
                   if row.entity_type == "game" and row.to_status == "cancelled"
                   and row.reason == "expired_without_full_table"}
        return {message["message_id"]: items for message, snapshot in self._verified_boards(records)
                if (items := [{"item_no": item.item_no, "published_text": item.rendered_text,
                               "cancelled_at": expired[item.game_id].isoformat()}
                              for item in snapshot.items if item.game_id in expired])}

    def audit(self, rows: list[dict], records: list[dict]) -> dict:
        """按建局工具、看板消息、席位事务及真实回执交叉核对。"""
        chat = {row["message_id"]: row for row in records}
        boards = self._boards(records)
        games = self.store.games
        created = {}
        for row in rows:
            source = chat.get(row.get("message_id"))
            if source is None or (source["direction"], source["kind"]) != ("inbound", "private"):
                continue
            for game_id in _written_games(row, {"create_game"}):
                game = games.get(game_id)
                if game is not None and (game.organizer_id, game.conversation_id) == (
                    source["sender_id"], source["conversation_id"],
                ):
                    created.setdefault(game_id, source)

        stages = {}
        for game_id, source in created.items():
            stages[game_id] = [(message, item) for message, item in boards[game_id]
                               if message["sequence"] > source["sequence"]]
        private_replies = defaultdict(list)
        for record in records:
            if (record["direction"], record["kind"], record["sender_id"]) == ("outbound", "private", AGENT_ID):
                private_replies[record["source_message_id"]].append(record)

        by_message = {row["message_id"]: row for row in rows if row.get("message_id") in chat}
        linked = []
        for message_id, row in by_message.items():
            source = chat[message_id]
            if source["direction"] != "inbound":
                continue
            participant = source["sender_id"]
            for game_id in self._joined_games(row, source):
                if game_id not in created or participant == created[game_id]["sender_id"]:
                    continue
                original = self._board_cause(source, row, by_message, chat, stages[game_id])
                if original is None:
                    continue
                prior, prior_item = original
                receipts = [reply for reply in private_replies[message_id]
                            if reply["recipient_id"] == participant
                            and reply["conversation_id"] == self.channel.private_conversation_id(participant)]
                before = _board_claimed(prior_item.rendered_text)
                refresh = next((message for message, item in stages[game_id]
                                if message["sequence"] > source["sequence"] and before is not None
                                and (after := _board_claimed(item.rendered_text)) is not None and after > before), None)
                if receipts and refresh is not None:
                    linked.append({"game_id": game_id, "join_channel": source["kind"], "source_board": prior["message_id"],
                                   "participation_message": message_id, "private_receipt": receipts[0]["message_id"],
                                   "updated_board": refresh["message_id"]})

        return {
            "status": "observed" if linked else "not_observed",
            "private_games_created": len(created),
            "private_games_on_public_board": sum(bool(items) for items in stages.values()),
            "published_boards": len({message["message_id"] for items in boards.values() for message, _ in items}),
            "games_with_end_to_end_link": len({entry["game_id"] for entry in linked}),
            "group_claims_with_private_receipt": sum(entry["join_channel"] == "group" for entry in linked),
            "board_driven_private_joins": sum(entry["join_channel"] == "private" for entry in linked),
            "links": linked,
        }

    def _boards(self, records: list[dict]) -> dict:
        """只采信与渠道归档内容吻合的正式不可变看板快照。"""
        result = defaultdict(list)
        for message, snapshot in self._verified_boards(records):
            for item in snapshot.items:
                result[item.game_id].append((message, item))
        return result

    def _verified_boards(self, records: list[dict]) -> list[tuple]:
        """核对发布消息与 Store 快照的群、会话和正文。"""
        result = []
        for message in records:
            if (message["direction"], message["kind"], message["recipient_id"], message["sender_id"]) != (
                "outbound", "group", self.channel.room_id, AGENT_ID,
            ):
                continue
            snapshot = self.store.get_board_snapshot_by_message_id(self.channel.room_id, message["message_id"])
            if (snapshot is None or snapshot.conversation_id != message["conversation_id"]
                    or snapshot.rendered_text != message["text"]):
                continue
            result.append((message, snapshot))
        return result

    def _joined_games(self, row: dict, source: dict) -> set[str]:
        """群报名以事务为准，私聊报名须有正式写入和已确认状态。"""
        if source["kind"] == "group":
            claim = self.store.get_game_claim_by_source(source["conversation_id"], source["message_id"])
            if claim is not None and claim.customer_id == source["sender_id"] and claim.status == "claimed":
                return {claim.game_id}
            return set()
        return _written_games(row, {"join_game", "record_candidate_reply"}, joined_customer=source["sender_id"])

    @staticmethod
    def _board_cause(source: dict, row: dict, rows: dict, chat: dict, boards: list) -> tuple | None:
        """沿本人的已送达因果链接回看板，跳过别人的私聊和未来消息。"""
        board_messages = {message["message_id"]: (message, item) for message, item in boards
                          if message["sequence"] < source["sequence"]}
        if source["kind"] == "group":
            return board_messages.get(source["quoted_message_id"])
        participant = source["sender_id"]
        while row.get("visible_cause_verified") and (cause := chat.get(row.get("caused_by"))):
            if cause["sequence"] >= source["sequence"]:
                return None
            if cause["kind"] == "group":
                return board_messages.get(cause["message_id"])
            if (cause["direction"], cause["recipient_id"]) != ("outbound", participant):
                return None
            parent = chat.get(cause["source_message_id"])
            if parent is None or parent["sender_id"] != participant or parent["sequence"] >= cause["sequence"]:
                return None
            if parent["kind"] == "group":
                return board_messages.get(parent["quoted_message_id"])
            source, row = parent, rows.get(parent["message_id"], {})
        return None


def _board_claimed(text: str) -> int | None:
    """读取正式四座看板的席位码，时间或桌号变化不能充当席位变化。"""
    matches = _SEAT_CODE.findall(text)
    return int(matches[-1][0]) if matches and sum(map(int, matches[-1])) == 4 else None


def _written_games(row: dict, names: set[str], *, joined_customer: str | None = None) -> set[str]:
    """从成功调用的写入结果提取牌局，查询结果不能充当写入证据。"""
    game_ids = set()
    for tool in row.get("tool_results", ()):
        result = tool.get("result") or {}
        game = result.get("game") or {}
        joined = joined_customer is None or (
            result.get("recorded_status") in {"joined", "confirmed", "accepted"}
            and any(member["customer_id"] == joined_customer and member["status"] in {"joined", "confirmed", "accepted"}
                    for member in game.get("participants", ()))
        )
        if (tool["name"] in names and tool.get("called") is True and tool.get("allowed") is True
                and not tool.get("error") and not tool.get("deduplicated") and game.get("game_id") and joined):
            game_ids.add(game["game_id"])
    return game_ids
