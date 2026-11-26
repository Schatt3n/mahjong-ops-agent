"""牌局小群的独立沙盒账本。成员准入与业务席位使用不同的存储。"""

from collections.abc import Collection
from contextlib import closing, contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from urllib.parse import quote

from mahjong_agent_runtime.models import Game, GameStatus

from .venue_channel import AGENT_ID, OWNER_ID, MessageEnvelope


_TABLE_SIZE = 4


@dataclass(frozen=True, slots=True)
class TableGroupReceipt:
    """同一次原子送达的消息和收件快照，可作为角色唤醒的依据。"""

    message: MessageEnvelope
    conversation_id: str
    recipient_ids: tuple[str, ...]
    sequence: int = 0
    source_message_id: str | None = None

    def history_row(self) -> dict:
        """使用现有角色上下文的群消息形状，保留送达时而非最新的受众。"""
        return {
            "message_id": self.message.message_id,
            "sender_id": self.message.sender_id,
            "recipient_id": self.message.recipient_id,
            "kind": self.message.kind,
            "text": self.message.text,
            "sent_at": self.message.sent_at.isoformat(),
            "quoted_message_id": self.message.quoted_message_id,
            "source_message_id": self.source_message_id,
            "conversation_id": self.conversation_id,
            "visible_to": (self.message.sender_id, *self.recipient_ids),
        }


@dataclass(frozen=True, slots=True)
class CompanionInvitation:
    """报名联络人私下分享的入群邀请，同行本人接受之前没有小群收件权限。"""

    invitation_id: str
    inviter_id: str
    recipient_id: str
    invited_at: datetime
    response_at: datetime | None = None
    accepted: bool | None = None


class VenueTableGroup:
    """按已确认的四席创建沙盒小群。账本不触发任何生产发送或席位写入。"""

    def __init__(
        self, store, archive_path: Path, game_id: str, *, actor_ids: Collection[str], opened_at: datetime,
    ) -> None:
        self.store = store
        self.archive_path = Path(archive_path).resolve()
        store_path = getattr(store, "path", None)
        if store_path is not None and self.archive_path == Path(store_path).resolve():
            raise ValueError("table group requires a separate channel archive")
        self.archive_path.parent.mkdir(parents=True, exist_ok=True)
        self.game_id = game_id
        self.room_id = self.room_id_for_game(game_id)
        self.conversation_id = f"im:{self.room_id}"
        self._actors = frozenset(actor_ids)
        if any(not isinstance(actor, str) or not actor.strip() or actor in {AGENT_ID, OWNER_ID}
               for actor in self._actors):
            raise ValueError("table group requires stable customer actor IDs")
        self._create_schema()
        self._open(_stamp(opened_at))

    @staticmethod
    def room_id_for_game(game_id: str) -> str:
        """用稳定的业务牌局编号派生沙盒桌群地址。"""
        return f"table:{quote(game_id, safe='')}"

    @classmethod
    def opened_rooms(cls, archive_path: Path) -> tuple[tuple[str, datetime], ...]:
        """读取实际已经建成的小群及原始时间，恢复时不用现有席位重新推算。"""
        if not archive_path.is_file():
            return ()
        with closing(sqlite3.connect(archive_path)) as db:
            exists = db.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'table_group_rooms'")
            if exists.fetchone() is None:
                return ()
            return tuple((game_id, datetime.fromisoformat(opened_at)) for game_id, opened_at in db.execute(
                "SELECT game_id, opened_at FROM table_group_rooms ORDER BY opened_at, game_id"))

    @contextmanager
    def _database(self, *, write: bool = False):
        with closing(sqlite3.connect(self.archive_path)) as db, db:
            db.row_factory = sqlite3.Row
            if write:
                db.execute("BEGIN IMMEDIATE")
            yield db

    def _create_schema(self) -> None:
        """把入群权、邀请响应和消息受众分别留档，重开时不从当前名册倒推过去。"""
        with self._database() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS table_group_rooms (
                    game_id TEXT PRIMARY KEY, opened_at TEXT NOT NULL, actor_ids_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS table_group_members (
                    game_id TEXT NOT NULL, actor_id TEXT NOT NULL, joined_at TEXT NOT NULL,
                    PRIMARY KEY (game_id, actor_id)
                );
                CREATE TABLE IF NOT EXISTS table_group_invites (
                    game_id TEXT NOT NULL, invitation_id TEXT NOT NULL, inviter_id TEXT NOT NULL,
                    recipient_id TEXT NOT NULL, invited_at TEXT NOT NULL, response_at TEXT, accepted INTEGER,
                    PRIMARY KEY (game_id, invitation_id)
                );
                CREATE TABLE IF NOT EXISTS table_group_messages (
                    seq INTEGER PRIMARY KEY, game_id TEXT NOT NULL, message_id TEXT NOT NULL, sender_id TEXT NOT NULL,
                    text TEXT NOT NULL, sent_at TEXT NOT NULL, quoted_message_id TEXT, recipients_json TEXT NOT NULL,
                    source_message_id TEXT,
                    UNIQUE (game_id, message_id)
                );
            """)
        with self._database(write=True) as db:
            if "source_message_id" not in {row["name"] for row in db.execute("PRAGMA table_info(table_group_messages)")}:
                db.execute("ALTER TABLE table_group_messages ADD COLUMN source_message_id TEXT")

    def _open(self, opened_at: str) -> None:
        """新建时从权威牌局读取具名和匿名席，重开时复用首次创建及准入结果。"""
        actors = json.dumps(sorted(self._actors))
        with self._database(write=True) as db:
            existing = db.execute(
                "SELECT opened_at, actor_ids_json FROM table_group_rooms WHERE game_id = ?", (self.game_id,),
            ).fetchone()
            if existing is not None:
                if (existing["opened_at"], existing["actor_ids_json"]) != (opened_at, actors):
                    raise ValueError("table group creation conflicts with the saved room")
                return
            named = {AGENT_ID}
            game = self.store.require_game(self.game_id)
            for claim in _confirmed_claims(game, at=datetime.fromisoformat(opened_at)):
                members = {claim["contact_id"], *claim["known_member_ids"]}
                if (members & named or not members <= self._actors
                        or len(members) + claim["anonymous_seat_count"] != claim["seat_count"]):
                    raise ValueError("table game has inconsistent or unregistered named seats")
                named.update(members)
            db.execute("INSERT INTO table_group_rooms VALUES (?, ?, ?)", (self.game_id, opened_at, actors))
            db.executemany("INSERT INTO table_group_members VALUES (?, ?, ?)",
                           ((self.game_id, member, opened_at) for member in sorted(named)))

    def member_ids(self, *, at: datetime | None = None) -> frozenset[str]:
        """按指定虚拟时刻读取小群成员，业务席位仍以 Game 为准。"""
        stamp = _stamp(at) if at is not None else None
        with self._database() as db:
            return frozenset(actor for actor, joined_at in self._members(db).items()
                             if stamp is None or joined_at <= stamp)

    def invite_companion(
        self, invitation_id: str, *, inviter_id: str, recipient_id: str, at: datetime,
    ) -> CompanionInvitation:
        """只允许对应 Party 的报名联络人把匿名席的私下邀请交给已存在的角色。"""
        sent_at = _stamp(at)
        if not invitation_id.strip() or recipient_id not in self._actors:
            raise ValueError("companion invitation requires an existing role and a stable ID")
        with self._database(write=True) as db:
            if (previous := self._invitation_row(db, invitation_id)) is not None:
                if (previous["inviter_id"], previous["recipient_id"], previous["invited_at"]) != (
                    inviter_id, recipient_id, sent_at,
                ):
                    raise ValueError("companion invitation ID conflicts with its first delivery")
                return _invitation(previous)
            members = self._members(db)
            if inviter_id not in members or sent_at < members[inviter_id] or recipient_id in members:
                raise ValueError("invalid table group invitation membership")
            capacity = self._companion_capacity(inviter_id, at)
            pending = db.execute("""
                SELECT inviter_id, recipient_id FROM table_group_invites
                WHERE game_id = ? AND (accepted IS NULL OR accepted = 1)
            """, (self.game_id,)).fetchall()
            if sum(row["inviter_id"] == inviter_id for row in pending) >= capacity:
                raise ValueError("only the party contact may invite within their anonymous seats")
            if any(row["recipient_id"] == recipient_id for row in pending):
                raise ValueError("companion already has an active table invitation")
            db.execute("""
                INSERT INTO table_group_invites (game_id, invitation_id, inviter_id, recipient_id, invited_at)
                VALUES (?, ?, ?, ?, ?)
            """, (self.game_id, invitation_id, inviter_id, recipient_id, sent_at))
            return _invitation(self._invitation_row(db, invitation_id))

    def invitation_for(self, actor_id: str, invitation_id: str) -> CompanionInvitation | None:
        """收件记录仅对分享者与实际同行可见，入群前也可独立核对。"""
        with self._database() as db:
            invite = self._invitation_row(db, invitation_id)
            if invite is None or actor_id not in (invite["inviter_id"], invite["recipient_id"]):
                return None
            return _invitation(invite)

    def respond_companion(
        self, invitation_id: str, *, actor_id: str, at: datetime, accept: bool,
    ) -> CompanionInvitation:
        """通过邀请本人真实发出的接受事件入群，拒绝、沉默和重投保持各自语义。"""
        response_at = _stamp(at)
        if type(accept) is not bool:
            raise ValueError("companion response must explicitly accept or decline")
        with self._database(write=True) as db:
            invite = self._invitation_row(db, invitation_id)
            if invite is None or actor_id != invite["recipient_id"] or response_at < invite["invited_at"]:
                raise ValueError("companion response must come from the invited role after delivery")
            if invite["response_at"] is not None:
                if (invite["response_at"], bool(invite["accepted"])) != (response_at, accept):
                    raise ValueError("companion response conflicts with the recorded outcome")
                return _invitation(invite)
            if accept:
                capacity = self._companion_capacity(invite["inviter_id"], at)
                used = db.execute("""
                    SELECT COUNT(*) FROM table_group_invites WHERE game_id = ? AND inviter_id = ? AND accepted = 1
                """, (self.game_id, invite["inviter_id"])).fetchone()[0]
                members = self._members(db)
                if used >= capacity or actor_id in members or len(members) >= _TABLE_SIZE + 1:
                    raise ValueError("companion no longer has a valid anonymous place in this room")
                db.execute("INSERT INTO table_group_members VALUES (?, ?, ?)", (self.game_id, actor_id, response_at))
            db.execute("""
                UPDATE table_group_invites SET response_at = ?, accepted = ? WHERE game_id = ? AND invitation_id = ?
            """, (response_at, accept, self.game_id, invitation_id))
            return _invitation(self._invitation_row(db, invitation_id))

    def send(self, message: MessageEnvelope, *, source_message_id: str | None = None) -> TableGroupReceipt:
        """用一次事务冻结收件范围，消息编号重投不得改变内容、时间或受众。"""
        sent_at = _stamp(message.sent_at)
        if (message.kind != "group" or message.recipient_id != self.room_id
                or not message.message_id.strip() or not message.text.strip()):
            raise ValueError("invalid table group message")
        payload = (message.sender_id, message.text, sent_at, message.quoted_message_id, source_message_id)
        with self._database(write=True) as db:
            existing = db.execute("""
                SELECT * FROM table_group_messages WHERE game_id = ? AND message_id = ?
            """, (self.game_id, message.message_id)).fetchone()
            if existing is not None:
                if tuple(existing[key] for key in (
                    "sender_id", "text", "sent_at", "quoted_message_id", "source_message_id",
                )) != payload:
                    raise ValueError("table message ID conflicts with its first delivery")
                return self._receipt(existing)
            members = {actor for actor, joined_at in self._members(db).items() if joined_at <= sent_at}
            if message.sender_id not in members:
                raise ValueError("table group sender is not admitted")
            if message.quoted_message_id:
                original = self._visible_rows(db, message.sender_id, message_id=message.quoted_message_id).fetchone()
                if original is None or original["sent_at"] > sent_at:
                    raise ValueError("table group quote is not visible to the sender")
            if source_message_id is not None:
                source = self._visible_rows(db, message.sender_id, message_id=source_message_id).fetchone()
                if not (message.sender_id == AGENT_ID and source is not None
                        and source["sender_id"] != message.sender_id and source["sent_at"] <= sent_at):
                    raise ValueError("table group reply requires an actual received message")
            recipients = tuple(sorted(members - {message.sender_id}))
            cursor = db.execute("""
                INSERT INTO table_group_messages
                    (game_id, message_id, sender_id, text, sent_at, quoted_message_id, source_message_id, recipients_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (self.game_id, message.message_id, *payload, json.dumps(recipients)))
            return TableGroupReceipt(replace(message, sent_at=datetime.fromisoformat(sent_at)),
                                     self.conversation_id, recipients, cursor.lastrowid, source_message_id)

    def delivered_receipts(self) -> tuple[TableGroupReceipt, ...]:
        """供可信调度器恢复收件唤醒，按账本提交顺序返回首次投递范围。"""
        with self._database() as db:
            rows = db.execute("SELECT * FROM table_group_messages WHERE game_id = ? ORDER BY seq", (self.game_id,))
            return tuple(self._receipt(row) for row in rows)

    def events_for(self, actor_id: str) -> list[dict]:
        """返回本人当时送达的历史，晚加入的成员不会取得先前的群消息。"""
        with self._database() as db:
            return [self.archive_row(receipt) for receipt in self._visible_receipts(db, actor_id)]

    def archive_row(self, receipt: TableGroupReceipt) -> dict:
        """使用现有渠道账本形状，保留用于审核和调度的实际送达范围。"""
        message = receipt.message
        return {**receipt.history_row(), "sequence": receipt.sequence, "room_id": self.room_id,
                "table_game_id": self.game_id, "trace_id": f"table:{self.game_id}:{message.message_id}",
                "direction": "outbound" if message.sender_id == AGENT_ID else "inbound"}

    def received(self, actor_id: str, message_id: str) -> TableGroupReceipt | None:
        """仅实际收件人可以用这条回执触发续轮，发送者的自见记录不算来话。"""
        with self._database() as db:
            row = self._visible_rows(db, actor_id, message_id=message_id).fetchone()
            if row is None or row["sender_id"] == actor_id:
                return None
            receipt = self._receipt(row)
            if quote_id := receipt.message.quoted_message_id:
                if self._visible_rows(db, actor_id, message_id=quote_id).fetchone() is None:
                    receipt = replace(receipt, message=replace(receipt.message, quoted_message_id=None))
            if receipt.source_message_id and self._visible_rows(db, actor_id, message_id=receipt.source_message_id).fetchone() is None:
                receipt = replace(receipt, source_message_id=None)
            return receipt

    def _members(self, db) -> dict[str, str]:
        rows = db.execute("SELECT actor_id, joined_at FROM table_group_members WHERE game_id = ?", (self.game_id,))
        return {row["actor_id"]: row["joined_at"] for row in rows}

    def _companion_capacity(self, inviter_id: str, at: datetime) -> int:
        # 入群前重读业务席位，避免失效的旧邀请在原局退席后仍被使用。
        game = self.store.require_game(self.game_id)
        return sum(claim["anonymous_seat_count"] for claim in _confirmed_claims(game, at=at)
                   if claim["contact_id"] == inviter_id)

    def _invitation_row(self, db, invitation_id: str):
        return db.execute("SELECT * FROM table_group_invites WHERE game_id = ? AND invitation_id = ?",
                          (self.game_id, invitation_id)).fetchone()

    def _visible_rows(self, db, actor_id: str, *, message_id: str | None = None):
        return db.execute("""
            SELECT * FROM table_group_messages AS message WHERE game_id = ?
            AND (? IS NULL OR message_id = ?)
            AND (sender_id = ? OR EXISTS (SELECT 1 FROM json_each(message.recipients_json) WHERE value = ?))
            ORDER BY sent_at, seq
        """, (self.game_id, message_id, message_id, actor_id, actor_id))

    def _visible_receipts(self, db, actor_id: str) -> list[TableGroupReceipt]:
        receipts = [self._receipt(row) for row in self._visible_rows(db, actor_id)]
        visible_ids = {receipt.message.message_id for receipt in receipts}
        return [replace(receipt,
                        message=replace(receipt.message, quoted_message_id=receipt.message.quoted_message_id
                                        if receipt.message.quoted_message_id in visible_ids else None),
                        source_message_id=receipt.source_message_id if receipt.source_message_id in visible_ids else None)
                for receipt in receipts]

    def _receipt(self, row) -> TableGroupReceipt:
        message = MessageEnvelope(row["message_id"], row["sender_id"], row["text"],
                                  datetime.fromisoformat(row["sent_at"]), "group", self.room_id, row["quoted_message_id"])
        return TableGroupReceipt(message, self.conversation_id, tuple(json.loads(row["recipients_json"])),
                                 row["seq"], row["source_message_id"])


def _confirmed_claims(game: Game, *, at: datetime) -> list[dict]:
    claims = game.seat_claims()
    if (game.status != GameStatus.READY or game.actual_started_at is not None or at < game.updated_at
            or (game.expires_at is not None and at >= game.expires_at)
            or game.seats_total != _TABLE_SIZE or sum(claim["seat_count"] for claim in claims) != _TABLE_SIZE):
        raise ValueError("table group admission requires four current confirmed seats before play")
    return claims


def _stamp(at: datetime) -> str:
    if not isinstance(at, datetime) or at.utcoffset() is None:
        raise ValueError("table group event needs a timezone")
    return at.astimezone(timezone.utc).isoformat(timespec="microseconds")


def _invitation(row) -> CompanionInvitation:
    return CompanionInvitation(
        row["invitation_id"], row["inviter_id"], row["recipient_id"], datetime.fromisoformat(row["invited_at"]),
        datetime.fromisoformat(row["response_at"]) if row["response_at"] else None,
        bool(row["accepted"]) if row["accepted"] is not None else None,
    )
