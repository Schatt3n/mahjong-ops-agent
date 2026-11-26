"""受控演练中的专桌候补和原席截止，正式席位仍由 SQLite 牌局事务管理。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from mahjong_agent_runtime import SQLiteAgentStore
from mahjong_agent_runtime.domains.game_domain import duration_hours_from_requirement
from mahjong_agent_runtime.domains.game_participants import participation_table_terms
from mahjong_agent_runtime.domains.model_context import has_agreed_game_duration
from mahjong_agent_runtime.group_chat.board_engine import BoardEngine
from mahjong_agent_runtime.models import GameStatus, using_clock

from .venue_channel import AGENT_ID, OWNER_ID, SandboxChannel


_PREPLAY = {GameStatus.FORMING, GameStatus.INVITING, GameStatus.READY}
_ACTIVE = {"joined", "confirmed"}


class VenueStandby:
    """一个已登记席位和一个候补。调用方提供受控决定，账本核对真实私聊来源。"""

    def __init__(self, channel: SandboxChannel, run_dir: Path, *, game_id: str,
                 incumbent: str, candidate: str) -> None:
        store = channel.runtime.store
        directory = run_dir.resolve()
        if (not isinstance(store, SQLiteAgentStore) or store.path.resolve() != directory / "agent.sqlite3"
                or channel.archive_path.resolve() != directory / "chat.sqlite3"):
            raise ValueError("standby needs this run's sandbox store and channel")
        if incumbent == candidate or not {incumbent, candidate} <= channel.owner_friends.keys():
            raise ValueError("standby needs two distinct owner friends")
        self.store, self.channel, self.game_id = store, channel, game_id
        self.incumbent, self.candidate = incumbent, candidate
        with store._write_transaction():
            store.require_game(game_id)
            db = store._connection
            db.execute("""
                CREATE TABLE IF NOT EXISTS venue_standby_scope (
                    game_id TEXT PRIMARY KEY, incumbent TEXT NOT NULL, candidate TEXT NOT NULL
                )
            """)
            db.execute("""
                CREATE TABLE IF NOT EXISTS venue_standby_events (
                    sequence INTEGER PRIMARY KEY, game_id TEXT NOT NULL,
                    event_id TEXT NOT NULL UNIQUE, action TEXT NOT NULL, request TEXT NOT NULL, result TEXT NOT NULL,
                    FOREIGN KEY (game_id) REFERENCES venue_standby_scope (game_id)
                )
            """)
            db.execute("INSERT OR IGNORE INTO venue_standby_scope VALUES (?, ?, ?)", (game_id, incumbent, candidate))
            scope = db.execute("SELECT incumbent, candidate FROM venue_standby_scope WHERE game_id = ?", (game_id,)).fetchone()
            if tuple(scope) != (incumbent, candidate):
                raise ValueError("standby scope differs from the existing rehearsal")

    def invite(self, message_id: str, *, at: datetime) -> dict:
        """向第五人送达专桌候补提议，即使满桌也可送达。"""
        return self._announce(message_id, at, acceptance_id=None)

    def offer_seat(self, message_id: str, *, acceptance_id: str, at: datetime) -> dict:
        """确有空席、候补仍接受当前条款时，单独发本人的留位提议。"""
        return self._announce(message_id, at, acceptance_id=acceptance_id)

    def _announce(self, message_id: str, at: datetime, *, acceptance_id: str | None) -> dict:
        stamp = _stamp(at)
        action = "invite" if acceptance_id is None else "offer_seat"

        def prepare(game):
            version = self._version(game)
            if not self._open(game, at):
                return {"status": "closed"}
            if self._seat(game, self.candidate):
                return {"status": "already_confirmed"}
            if acceptance_id is not None:
                acceptance = self._event(acceptance_id)
                if (acceptance is None or acceptance["status"] != "standby"
                        or at < datetime.fromisoformat(acceptance["at"])):
                    return {"status": "missing_standby_acceptance"}
                if acceptance["version"] != version:
                    return {"status": "terms_changed"}
                if not game.remaining_seats():
                    return {"status": "full"}
            return {"status": "seat_offered" if acceptance_id is not None else "invited",
                    "text": _offer_text(game, has_vacancy=acceptance_id is not None), "version": version, "at": stamp}

        result = self._record(message_id, action, {"at": stamp, "acceptance_id": acceptance_id}, prepare)
        if result["status"] in {"invited", "seat_offered"}:
            self.channel.publish_owner(kind="private", recipient_id=self.candidate, text=result["text"],
                                       sent_at=at, message_id=message_id, source_message_id=acceptance_id)
        return result

    def respond(self, message_id: str, action: str, *, reference_id: str | None = None,
                deadline_at: datetime | None = None) -> dict:
        """核对已入账的受控语义，候补同意和腾空后的本人报名分别处理。"""
        if action not in {"accept_standby", "confirm_seat", "set_deadline", "confirm_original", "withdraw"}:
            raise ValueError("unknown standby decision")
        candidate_action = action in {"accept_standby", "confirm_seat"}
        actor = self.candidate if candidate_action else self.incumbent
        source = self._receipt(message_id, actor, from_owner=False)
        if source is None:
            raise ValueError("standby decision requires the acting customer's completed private receipt")
        at = datetime.fromisoformat(source["sent_at"])
        if candidate_action and (not reference_id or source["quoted_message_id"] != reference_id
                                 or self._receipt(reference_id, actor, from_owner=True) is None):
            raise ValueError("candidate must reply to their delivered private offer")
        if action == "confirm_original" and (not reference_id or source["quoted_message_id"] != reference_id):
            raise ValueError("original confirmation needs the documented deadline reference")
        if (action == "set_deadline") != (deadline_at is not None):
            raise ValueError("a documented deadline is required only for that decision")
        request = {"reference_id": reference_id, "deadline_at": _stamp(deadline_at) if deadline_at else None}

        def decide(game):
            version = self._version(game)
            if not self._open(game, at):
                return {"status": "closed"}
            if candidate_action:
                offer = self._event(reference_id)
                expected = "invited" if action == "accept_standby" else "seat_offered"
                if offer is None or offer["status"] != expected:
                    return {"status": "wrong_offer"}
                if offer["version"] != version:
                    return {"status": "terms_changed"}
                if self._seat(game, actor):
                    return {"status": "already_confirmed"}
                if action == "accept_standby":
                    return {"status": "standby", "version": version, "at": source["sent_at"]}
                if not game.remaining_seats():
                    return {"status": "full"}
                if game.planned_start_at is not None and at > game.planned_start_at:
                    return {"status": "late_confirmation_required"}
                return self._write_seat(game, actor, "confirmed", at, source["trace_id"])
            if action == "confirm_original":
                return self._confirm_original(game, reference_id, at)
            seat = self._seat(game, actor)
            if seat is None or seat.seat_count != 1 or seat.anonymous_seat_count or set(seat.known_member_ids) - {actor}:
                return {"status": "no_independent_original_seat"}
            if action == "withdraw":
                return self._write_seat(game, actor, "declined", at, source["trace_id"])
            if (deadline_at <= at or game.planned_start_at and deadline_at > game.planned_start_at
                    or game.expires_at and deadline_at >= game.expires_at):
                return {"status": "deadline_outside_game"}
            return {"status": "deadline_registered", "version": version, "seat_epoch": self._seat_epoch(actor),
                    "deadline_at": request["deadline_at"]}

        return self._record(message_id, action, request, decide)

    def _confirm_original(self, game, deadline_id: str, at: datetime) -> dict:
        deadline = self._event(deadline_id)
        if deadline is None or deadline["status"] != "deadline_registered":
            return {"status": "no_documented_deadline"}
        if at >= datetime.fromisoformat(deadline["deadline_at"]):
            return {"status": "deadline_passed"}
        if deadline_id != self._latest_deadline():
            return {"status": "deadline_superseded"}
        if (deadline["version"] != self._version(game) or deadline["seat_epoch"] != self._seat_epoch(self.incumbent)
                or self._seat(game, self.incumbent) is None):
            return {"status": "original_seat_changed"}
        return {"status": "original_confirmed", "deadline_id": deadline_id}

    def expire(self, deadline_id: str, *, at: datetime) -> dict:
        """只执行本人书面授权的到期释放，不能因沉默自行生成退出。"""
        _stamp(at)
        with self.store._lock:
            deadline = self._event(deadline_id)
        if deadline is None or deadline["status"] != "deadline_registered":
            return {"status": "no_documented_deadline"}
        if at < datetime.fromisoformat(deadline["deadline_at"]):
            return {"status": "not_due"}

        def release(game):
            if not self._open(game, at):
                return {"status": "closed"}
            if deadline_id != self._latest_deadline():
                return {"status": "deadline_superseded"}
            if deadline["version"] != self._version(game) or deadline["seat_epoch"] != self._seat_epoch(self.incumbent):
                return {"status": "original_seat_changed"}
            satisfied = any(row["result"].get("deadline_id") == deadline_id
                            and row["result"]["status"] == "original_confirmed" for row in self._events())
            if satisfied:
                return {"status": "original_confirmed"}
            if self._seat(game, self.incumbent) is None:
                return {"status": "no_independent_original_seat"}
            return self._write_seat(game, self.incumbent, "declined", at, f"standby-deadline:{deadline_id}")

        return self._record(f"standby-deadline:{deadline_id}", "expire", {"deadline_id": deadline_id}, release)

    def events(self) -> list[dict]:
        """导出专桌决定及店家操作，附受控输入引用与提交结果。"""
        with self.store._lock:
            return self._events()

    def _events(self) -> list[dict]:
        rows = self.store._connection.execute(
            "SELECT event_id, action, request, result FROM venue_standby_events WHERE game_id = ? ORDER BY sequence",
            (self.game_id,),
        )
        return [{"event_id": row["event_id"], "action": row["action"], "request": json.loads(row["request"]),
                 "result": json.loads(row["result"])} for row in rows]

    def _event(self, event_id: str | None) -> dict | None:
        row = self.store._connection.execute("SELECT result FROM venue_standby_events WHERE game_id = ? AND event_id = ?",
                                             (self.game_id, event_id)).fetchone()
        return json.loads(row["result"]) if row is not None else None

    def _latest_deadline(self) -> str | None:
        return next((row["event_id"] for row in reversed(self._events())
                     if row["result"]["status"] == "deadline_registered"), None)

    def _record(self, event_id: str, action: str, request: dict, effect) -> dict:
        payload = json.dumps(request, sort_keys=True)
        with self.store._write_transaction():
            db = self.store._connection
            row = db.execute("SELECT game_id, action, request, result FROM venue_standby_events WHERE event_id = ?",
                             (event_id,)).fetchone()
            if row is not None:
                if (row["game_id"], row["action"], row["request"]) != (self.game_id, action, payload):
                    raise ValueError("standby event differs from its first recorded decision")
                return json.loads(row["result"])
            result = effect(self.store.require_game(self.game_id))
            db.execute("INSERT INTO venue_standby_events (game_id, event_id, action, request, result) VALUES (?, ?, ?, ?, ?)",
                       (self.game_id, event_id, action, payload, json.dumps(result, ensure_ascii=False, sort_keys=True)))
            return result

    def _version(self, game) -> dict:
        # 真实桌况的修订代次阻止 A→B→A 时重新使用旧回执。参与者进出不变更桌况代次。
        return {"terms": participation_table_terms(game), "agreed_duration_hours": (
            duration_hours_from_requirement(game.requirement) if has_agreed_game_duration(game.requirement) else None),
                "revision": sum(change.entity_type == "game_requirement" and change.entity_id == game.game_id
                                for change in self.store.transitions)}

    def _seat_epoch(self, customer_id: str) -> int:
        return sum(change.entity_type == "game_participant" and change.entity_id == f"{self.game_id}:{customer_id}"
                   for change in self.store.transitions)

    @staticmethod
    def _open(game, at: datetime) -> bool:
        return (game.status in _PREPLAY and not game.actual_started_at and game.created_at <= at
                and (not game.expires_at or at < game.expires_at))

    @staticmethod
    def _seat(game, customer_id: str):
        return next((person for person in game.participants
                     if person.customer_id == customer_id and person.status in _ACTIVE), None)

    def _write_seat(self, game, actor: str, status: str, at: datetime, trace_id: str) -> dict:
        # 同一 SQLite 事务落正式席位与演练回执。失败后两个写入一起回滚。
        with using_clock(lambda: at):
            updated, transitions = self.store._record_candidate_reply_in_transaction(
                game_id=self.game_id, customer_id=actor, display_name=self.channel.owner_friends[actor],
                status=status, seat_count=1, trace_id=trace_id, expected_participation_terms=participation_table_terms(game))
            if transitions:
                self.store._ensure_game_board_arrangement_in_transaction(
                    self.game_id, trace_id=trace_id, at=at, urgent=True)
        return {"status": "confirmed" if status == "confirmed" else "released", "game_id": self.game_id,
                "actor_id": actor, "trace_id": trace_id, "remaining_seats": updated.remaining_seats(),
                "transitions": [entry.to_dict() for entry in transitions]}

    def _receipt(self, message_id: str, actor: str, *, from_owner: bool):
        sender, recipient, direction = (OWNER_ID, actor, "owner") if from_owner else (actor, AGENT_ID, "inbound")
        with self.channel._database() as db:
            row = db.execute("""
                SELECT message.* FROM venue_messages AS message
                JOIN venue_receipts AS receipt USING (room_id, message_id)
                WHERE message.room_id = ? AND message.message_id = ? AND message.kind = 'private'
                    AND message.direction = ? AND message.sender_id = ? AND message.recipient_id = ?
                    AND message.conversation_id = ? AND receipt.status = 'completed'
            """, (self.channel.room_id, message_id, direction, sender, recipient,
                  self.channel.private_conversation_id(actor))).fetchone()
        return row


def _stamp(at: datetime) -> str:
    if at.utcoffset() is None:
        raise ValueError("standby events require a timezone-aware time")
    return at.astimezone(timezone.utc).isoformat()


def _offer_text(game, *, has_vacancy: bool) -> str:
    requirement = game.requirement
    start = game.planned_start_at.strftime("%m月%d日 %H:%M") if game.planned_start_at else "人齐开"
    name = BoardEngine._display_game_name(requirement) or "牌局"
    stake = requirement.get("stake_label") or "档位待定"
    smoking = {"no_smoking": "无烟", "smoking": "有烟", "any": "烟都可"}
    smoke = smoking.get(requirement.get("smoke_preference"), "烟况待定")
    duration = (f"，打 {duration_hours_from_requirement(requirement):g} 小时"
                if has_agreed_game_duration(requirement) else "")
    terms = f"{start} 的{name}，{stake}档，{smoke}{duration}。"
    return terms + ("有个空位，你现在要留吗？" if has_vacancy else "你愿意先候补吗？有空位我再问你。")
