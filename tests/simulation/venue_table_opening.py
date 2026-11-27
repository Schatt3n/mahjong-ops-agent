"""人齐开桌群的开场协商，以实际收件支撑同一版时间的四席确认。"""

from dataclasses import replace
from datetime import datetime, time, timedelta
import json

from mahjong_agent_runtime.domains.game_commitments import active_game_participant_ids, ready_commitment_conflicts
from mahjong_agent_runtime.domains.game_domain import (
    START_KIND_ASAP_WHEN_FULL, apply_game_lifecycle, duration_hours_from_requirement, flexible_opening_not_before,
)
from mahjong_agent_runtime.domains.model_context import has_agreed_game_duration
from mahjong_agent_runtime.models import DEFAULT_TZ, GameStatus, StateTransition, using_clock

from .venue_channel import AGENT_ID, MessageEnvelope
from .venue_table_group import TableGroupReceipt, VenueTableGroup
from .venue_scenario import VENUE


TABLE_OPENING_OFFER_PREFIX = "table-opening-offer|"
TABLE_OPENING_OFFER_LIMIT = 3
TABLE_OPENING_CONFIRM_PREFIX = "table-opening-confirm|"


def table_opening_confirmation_text(start: datetime, *, at: datetime) -> str:
    """从同一局确认开场生成对客文本，日期以店里时区表述。"""
    start, at = _local(start), _local(at)
    when = start.strftime("%H:%M" if start.date() == at.date() else "%m月%d日 %H:%M")
    return f"那就 {when} 开，大家按这个时间到哈。"


def _within_venue_hours(start: datetime, end: datetime | None) -> bool:
    """按配置区分正常关门、跨日关门和全天营业。"""
    opens = time.fromisoformat(VENUE["opens"])
    closes = time.fromisoformat(VENUE["closes"])
    if opens == closes:
        return end is None or end > start
    overnight = closes < opens
    business_date = start.date() - timedelta(days=1) if overnight and start.time() < closes else start.date()
    opened_at = datetime.combine(business_date, opens, tzinfo=DEFAULT_TZ)
    closed_at = datetime.combine(business_date + timedelta(days=int(overnight)), closes, tzinfo=DEFAULT_TZ)
    return opened_at <= start < closed_at and (end is None or start < end <= closed_at)


class VenueTableOpening:
    """仿真中从人齐开确定开场的窄入口，保留同一 Game 的原四个席位。"""

    def __init__(self, group: VenueTableGroup) -> None:
        self.group = group
        self.store = group.store
        self.game_id = group.game_id
        with self.store._write_transaction():
            db = self.store._connection
            db.execute("""
                CREATE TABLE IF NOT EXISTS venue_table_opening_proposals (
                    sequence INTEGER PRIMARY KEY, game_id TEXT NOT NULL, proposal_id TEXT NOT NULL,
                    proposed_at TEXT NOT NULL, start_at TEXT NOT NULL, seats_json TEXT NOT NULL,
                    seat_revision TEXT NOT NULL, committed INTEGER NOT NULL DEFAULT 0,
                    UNIQUE (game_id, proposal_id)
                )
            """)
            db.execute("""
                CREATE TABLE IF NOT EXISTS venue_table_opening_votes (
                    game_id TEXT NOT NULL, message_id TEXT NOT NULL, proposal_id TEXT NOT NULL,
                    actor_id TEXT NOT NULL, sent_at TEXT NOT NULL, accepted INTEGER NOT NULL, result TEXT NOT NULL,
                    PRIMARY KEY (game_id, message_id)
                )
            """)

    def proposal_message(self, message_id: str, *, at: datetime, start_at: datetime) -> MessageEnvelope:
        """生成有绝对日期的明确提议，发给当前小群供四人分别表态。"""
        message = self._proposal_envelope(message_id, at=at, start_at=start_at)
        if not self.accepts_bounded_proposal(start_at):
            raise ValueError("table opening proposal falls outside the verified opening or venue hours")
        return message

    def _proposal_envelope(self, message_id: str, *, at: datetime, start_at: datetime) -> MessageEnvelope:
        """回放先核对当时发送的原文，旧版预约不会被当前到期时间改写。"""
        stamp, start = _local(at), _local(start_at)
        if start <= stamp or start.second or start.microsecond:
            raise ValueError("table opening proposal must be a future minute boundary")
        when = start.strftime("%H:%M" if start.date() == stamp.date() else "%m月%d日 %H:%M")
        return MessageEnvelope(message_id, AGENT_ID, f"{when} 开大家能到吗？都行的话就定这个时间。",
                               stamp, "group", self.group.room_id)

    def accepts_bounded_proposal(self, start: datetime, *, same_day_as: datetime | None = None) -> bool:
        """核对牌局、营业与明确时长，跨日修订须仍在同一连续营业时段。"""
        start = _local(start)
        if same_day_as is not None:
            previous = _local(same_day_as)
            if previous.date() != start.date() and not _within_venue_hours(previous, start):
                return False
        return self._within_bounded_opening(self.store.require_game(self.game_id), start)

    @staticmethod
    def _within_bounded_opening(game, start: datetime) -> bool:
        floor = flexible_opening_not_before(game.requirement)
        end = (start + timedelta(hours=duration_hours_from_requirement(game.requirement))
               if has_agreed_game_duration(game.requirement) else None)
        return ((floor is None or floor <= start) and (game.expires_at is None or start < game.expires_at)
                and _within_venue_hours(start, end))

    def record_proposal(self, receipt: TableGroupReceipt, *, start_at: datetime) -> None:
        """只有四席主人实际收到的那份店家提议才成为有效版本。"""
        message = receipt.message
        expected = self._proposal_envelope(message.message_id, at=message.sent_at, start_at=start_at)
        if (replace(message, sent_at=_local(message.sent_at), quoted_message_id=None) != expected
                or receipt not in self.group.delivered_receipts()):
            raise ValueError("table opening proposal needs its exact delivered shop message")
        stamp = _local(message.sent_at)
        start = _local(start_at)
        with self.store._write_transaction():
            db = self.store._connection
            existing = db.execute("""
                SELECT proposed_at, start_at FROM venue_table_opening_proposals WHERE game_id = ? AND proposal_id = ?
            """, (self.game_id, message.message_id)).fetchone()
            if existing is not None:
                if tuple(existing) != (stamp.isoformat(), start.isoformat()):
                    raise ValueError("proposal retry conflicts with its first delivery")
                return
            game = self.store.require_game(self.game_id)
            if not self._within_bounded_opening(game, start):
                raise ValueError("table opening proposal falls outside the verified opening or venue hours")
            seats = active_game_participant_ids(game)
            if (not self._unsettled(game, stamp) or len(seats) != game.seats_total
                    or set(receipt.recipient_ids) != seats
                    or receipt.roster_revision != self.group.roster_revision(at=stamp)):
                raise ValueError("table opening proposal requires delivery to all four current identified seats")
            latest = self._latest()
            if latest is not None and stamp < datetime.fromisoformat(latest["proposed_at"]):
                raise ValueError("table opening proposal is older than the current version")
            db.execute("""
                INSERT INTO venue_table_opening_proposals
                    (game_id, proposal_id, proposed_at, start_at, seats_json, seat_revision)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (self.game_id, message.message_id, stamp.isoformat(), start.isoformat(),
                  json.dumps(sorted(seats)), self._seat_revision(game)))

    def record_vote(self, receipt: TableGroupReceipt, *, proposal_id: str, accepts: bool) -> str:
        """接收上层语义核对的明确表态，核对本人收件及同版提议后才可能写开场。"""
        message = receipt.message
        at = _local(message.sent_at)
        if (type(accepts) is not bool or AGENT_ID not in receipt.recipient_ids
                or self.group.received(AGENT_ID, message.message_id) != receipt):
            raise ValueError("table opening vote needs a completed delivery from the voting customer")
        offered = self.group.received(message.sender_id, proposal_id)
        order = {row.message.message_id: index for index, row in enumerate(self.group.delivered_receipts())}
        if (offered is None or offered.message.sender_id != AGENT_ID or offered.message.sent_at > message.sent_at
                or order[proposal_id] >= order[message.message_id]
                or offered.roster_revision != receipt.roster_revision
                or message.quoted_message_id not in {None, proposal_id}):
            raise ValueError("table opening vote is missing its visible proposal")
        with self.store._write_transaction():
            db = self.store._connection
            old = db.execute("""
                SELECT proposal_id, accepted, result FROM venue_table_opening_votes
                WHERE game_id = ? AND message_id = ?
            """, (self.game_id, message.message_id)).fetchone()
            if old is not None:
                if (old["proposal_id"], bool(old["accepted"])) != (proposal_id, accepts):
                    raise ValueError("table opening vote retry changed its meaning")
                return old["result"]
            result = self._apply_vote(message.sender_id, proposal_id, at, accepts, message.message_id)
            db.execute("INSERT INTO venue_table_opening_votes VALUES (?, ?, ?, ?, ?, ?, ?)",
                       (self.game_id, message.message_id, proposal_id, message.sender_id, at.isoformat(), accepts, result))
            return result

    def current(self) -> dict | None:
        """重开后读取当前提议及各人已核实表态，供店家在同一桌群继续协商。"""
        with self.store._lock:
            proposal = self._latest()
            if proposal is None:
                return None
            game = self.store.require_game(self.game_id)
            if not proposal["committed"] and self._seat_revision(game) != proposal["seat_revision"]:
                return None
            votes = self._votes(proposal["proposal_id"]).values()
            return {"proposal_id": proposal["proposal_id"], "start_at": proposal["start_at"],
                    "accepted_count": sum(votes), "declined_count": sum(not vote for vote in votes),
                    "committed": bool(proposal["committed"])}

    def confirmed_attendance(self, *, at: datetime) -> tuple[TableGroupReceipt, dict[str, TableGroupReceipt]] | None:
        """确认当前阵容的四份正式票，以及店家确实向同阵容送达的落库开场。"""
        with self.store._lock:
            game = self.store.require_game(self.game_id)
            proposal = self._latest()
            if (proposal is None or not proposal["committed"] or not game.planned_start_at
                    or game.planned_start_at != datetime.fromisoformat(proposal["start_at"])):
                return None
            seats = active_game_participant_ids(game)
            if (len(seats) != 4 or seats != set(json.loads(proposal["seats_json"]))
                    or json.loads(proposal["seat_revision"])[1] != self._seat_changes()):
                return None
            proposal_id = proposal["proposal_id"]
            rows = self.store._connection.execute("""
                SELECT actor_id, message_id, accepted, result FROM venue_table_opening_votes
                WHERE game_id = ? AND proposal_id = ? AND result IN ('waiting', 'declined', 'committed')
                ORDER BY sent_at, rowid
            """, (self.game_id, proposal_id)).fetchall()
        votes = {row["actor_id"]: row for row in rows}
        if set(votes) != seats or not all(row["accepted"] for row in votes.values()):
            return None
        final = rows[-1]
        if final["result"] != "committed" or not any(
            row.entity_type == "game_requirement" and row.entity_id == self.game_id
            and row.reason == "venue_table_opening_confirmed" and row.trace_id == final["message_id"]
            for row in self.store.transitions
        ):
            return None
        confirmation = self.group.received(final["actor_id"], f"{TABLE_OPENING_CONFIRM_PREFIX}{proposal_id}")
        revision = self.group.roster_revision(at=at)
        if (confirmation is None or confirmation.message.sender_id != AGENT_ID or confirmation.message.sent_at > at
                or confirmation.message.text != table_opening_confirmation_text(game.planned_start_at, at=confirmation.message.sent_at)
                or confirmation.source_message_id != final["message_id"] or confirmation.roster_revision != revision
                or set(confirmation.recipient_ids) != seats or self.group.member_ids(at=at) != {AGENT_ID, *seats}):
            return None
        ballots = {actor: self.group.received(AGENT_ID, row["message_id"]) for actor, row in votes.items()}
        if any(receipt is None or receipt.message.sender_id != actor or receipt.roster_revision != revision
               or receipt.message.sent_at > confirmation.message.sent_at for actor, receipt in ballots.items()):
            return None
        return confirmation, ballots

    def _apply_vote(self, actor_id: str, proposal_id: str, at: datetime, accepts: bool, trace_id: str) -> str:
        proposal = self._latest()
        if proposal is None or proposal["proposal_id"] != proposal_id:
            return "stale_proposal"
        game = self.store.require_game(self.game_id)
        if proposal["committed"] or not self._unsettled(game, at):
            return "closed"
        start = datetime.fromisoformat(proposal["start_at"])
        if not self._within_bounded_opening(game, start):
            return "outside_opening_window"
        if at >= start:
            return "proposal_elapsed"
        latest_vote_at = self.store._connection.execute("""
            SELECT max(sent_at) FROM venue_table_opening_votes
            WHERE game_id = ? AND proposal_id = ? AND result IN ('waiting', 'declined')
        """, (self.game_id, proposal_id)).fetchone()[0]
        if latest_vote_at is not None and at.isoformat() < latest_vote_at:
            return "stale_vote"
        seats = set(json.loads(proposal["seats_json"]))
        if seats != active_game_participant_ids(game) or self._seat_revision(game) != proposal["seat_revision"]:
            return "seats_changed"
        if actor_id not in seats:
            return "not_a_seat"
        if not accepts:
            return "declined"
        if not all((self._votes(proposal_id) | {actor_id: True}).get(seat) for seat in seats):
            return "waiting"

        # READY 的一般条款仍不可编辑。这里只提交一次已四方同意的、原先未定的开场。
        derived = {"latest_start_at", "lifecycle_expires_at", "lifecycle_ttl_hours", "planned_start_at",
                   "planned_end_at", "recruitment_opens_at", "recruitment_status", "recruitment_lead_hours"}
        requirement = {key: value for key, value in game.requirement.items() if key not in derived}
        requirement.update(start_time_kind="scheduled", start_time=start.strftime("%H:%M"), planned_start_at=start.isoformat())
        updated = replace(game, requirement=requirement, updated_at=at)
        with using_clock(lambda: at):
            apply_game_lifecycle(updated)
            if ready_commitment_conflicts(updated, seats, list(self.store.games.values())):
                return "participant_conflict"
            if any(item.game_id == self.game_id and item.status in {"held", "confirmed", "occupied"}
                   and (item.start_at != start or item.end_at != updated.planned_end_at)
                   for item in self.store.room_reservations.values()):
                return "room_commitment_conflict"
            self.store._save_game(updated)
            self.store._append_transition(StateTransition(
                "game_requirement", self.game_id, "opening_unconfirmed", "opening_confirmed",
                "venue_table_opening_confirmed", trace_id, occurred_at=at,
            ))
            self.store._connection.execute("""
                UPDATE venue_table_opening_proposals SET committed = 1 WHERE game_id = ? AND proposal_id = ?
            """, (self.game_id, proposal_id))
        return "committed"

    def _votes(self, proposal_id: str) -> dict[str, bool]:
        rows = self.store._connection.execute("""
            SELECT actor_id, accepted FROM venue_table_opening_votes
            WHERE game_id = ? AND proposal_id = ? AND result IN ('waiting', 'declined', 'committed')
            ORDER BY sent_at, rowid
        """, (self.game_id, proposal_id))
        return {row["actor_id"]: bool(row["accepted"]) for row in rows}

    def _latest(self):
        return self.store._connection.execute("""
            SELECT * FROM venue_table_opening_proposals WHERE game_id = ? ORDER BY sequence DESC LIMIT 1
        """, (self.game_id,)).fetchone()

    def _seat_revision(self, game) -> str:
        return json.dumps([game.updated_at.isoformat(), self._seat_changes()])

    def _seat_changes(self) -> int:
        changes = (row for row in self.store.transitions if row.entity_type == "game_participant"
                   and row.entity_id.startswith(f"{self.game_id}:"))
        return sum(1 for _ in changes)

    @staticmethod
    def _unsettled(game, at: datetime) -> bool:
        return (game.status == GameStatus.READY and game.seats_total == game.seat_summary()["claimed_seats"] == 4
                and not game.actual_started_at and not game.planned_start_at and game.updated_at <= at
                and (game.expires_at is None or at < game.expires_at)
                and game.requirement.get("start_time_kind") == START_KIND_ASAP_WHEN_FULL)


def _local(at: datetime) -> datetime:
    if at.utcoffset() is None:
        raise ValueError("table opening time must be timezone-aware")
    return at.astimezone(DEFAULT_TZ)
