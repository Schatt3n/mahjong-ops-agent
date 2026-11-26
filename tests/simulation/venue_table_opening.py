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
        stamp, start = _local(at), _local(start_at)
        if start <= stamp or start.second or start.microsecond:
            raise ValueError("table opening proposal must be a future minute boundary")
        if not self.accepts_bounded_proposal(start):
            raise ValueError("table opening proposal falls outside the verified opening or venue hours")
        when = start.strftime("%H:%M" if start.date() == stamp.date() else "%m月%d日 %H:%M")
        return MessageEnvelope(message_id, AGENT_ID, f"{when} 开大家能到吗？都行的话就定这个时间。",
                               stamp, "group", self.group.room_id)

    def accepts_bounded_proposal(self, start: datetime) -> bool:
        """只为有明确下界的试点牌局核对开场和已约定的营业时长。"""
        return self._within_bounded_opening(self.store.require_game(self.game_id), _local(start))

    @staticmethod
    def _within_bounded_opening(game, start: datetime) -> bool:
        floor = flexible_opening_not_before(game.requirement)
        if floor is None:
            return True
        closing = datetime.combine(start.date(), time.fromisoformat(VENUE["closes"]), tzinfo=DEFAULT_TZ)
        end = (start + timedelta(hours=duration_hours_from_requirement(game.requirement))
               if has_agreed_game_duration(game.requirement) else None)
        return (floor.date() == start.date() and floor <= start < closing
                and start.time() >= time.fromisoformat(VENUE["opens"])
                and (game.expires_at is None or start < game.expires_at)
                and (end is None or end <= closing))

    def record_proposal(self, receipt: TableGroupReceipt, *, start_at: datetime) -> None:
        """只有四席主人实际收到的那份店家提议才成为有效版本。"""
        message = receipt.message
        expected = self.proposal_message(message.message_id, at=message.sent_at, start_at=start_at)
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
        changes = (row for row in self.store.transitions if row.entity_type == "game_participant"
                   and row.entity_id.startswith(f"{self.game_id}:"))
        return json.dumps([game.updated_at.isoformat(), sum(1 for _ in changes)])

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
