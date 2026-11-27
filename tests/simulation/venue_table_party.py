"""把同行本人接受的桌群邀请对应到沙盒中已有的一个匿名席位。"""

from dataclasses import replace
from datetime import datetime

from mahjong_agent_runtime import SQLiteAgentStore
from mahjong_agent_runtime.domains.game_commitments import game_commitment_windows_overlap
from mahjong_agent_runtime.domains.game_participants import normalize_game_parties, refresh_requirement_seat_snapshot
from mahjong_agent_runtime.models import CustomerProfile, DEFAULT_TZ, Game, GameParticipant, GameStatus, StateTransition, using_clock

from .venue_table_group import CompanionInvitation, VenueTableGroup


_CLAIMED = {"joined", "confirmed"}


class VenueTableParty:
    """只处理已有 READY 小群，入群和身份绑定各有独立证据。"""

    def __init__(self, group: VenueTableGroup) -> None:
        """业务绑定以真实 SQLite 事务保存，和通道邀请账本分别恢复。"""
        if not isinstance(group.store, SQLiteAgentStore):
            raise ValueError("table companion binding requires the isolated SQLite game store")
        self.group = group
        self.store = group.store
        with self.store._write_transaction():
            self.store._connection.execute("""
                CREATE TABLE IF NOT EXISTS venue_table_companion_bindings (
                    game_id TEXT NOT NULL, invitation_id TEXT NOT NULL,
                    contact_id TEXT NOT NULL, party_id TEXT NOT NULL, recipient_id TEXT NOT NULL,
                    invited_at TEXT NOT NULL, accepted_at TEXT NOT NULL, bound_at TEXT NOT NULL,
                    PRIMARY KEY (game_id, invitation_id), UNIQUE (game_id, recipient_id)
                )
            """)

    def bind_companion(self, invitation_id: str, *, actor_id: str, at: datetime) -> Game:
        """用本人已接受的邀请把一位同行写成独立确认人，整桌仍占四席。"""
        if at.utcoffset() is None:
            raise ValueError("table companion binding time must be timezone-aware")
        stamp = at.astimezone(DEFAULT_TZ)
        invite = self.group.invitation_for(actor_id, invitation_id)
        if (invite is None or invite.recipient_id != actor_id or invite.accepted is not True
                or invite.response_at is None or stamp < invite.response_at
                or actor_id not in self.group.member_ids(at=invite.response_at)):
            raise ValueError("table companion needs its own accepted and admitted invitation")

        with using_clock(lambda: stamp), self.store._write_transaction():
            db = self.store._connection
            game = self.store.require_game(self.group.game_id)
            existing = db.execute("""
                SELECT contact_id, recipient_id, invited_at, accepted_at FROM venue_table_companion_bindings
                WHERE game_id = ? AND invitation_id = ?
            """, (game.game_id, invitation_id)).fetchone()
            source = (invite.inviter_id, actor_id, invite.invited_at.isoformat(), invite.response_at.isoformat())
            if existing is not None:
                if tuple(existing) != source:
                    raise ValueError("table companion binding retry conflicts with its accepted invitation")
                return game

            contact, customer = self._available_seat(game, invite, at=stamp)

            # 正式 Party 由参与者行重建；将一份原匿名席转给同行本人，而不另外占一个席位。
            participants = [replace(row, seat_count=row.seat_count - 1, anonymous_seat_count=row.anonymous_seat_count - 1)
                            if row is contact else row for row in game.participants]
            participants.append(GameParticipant(customer_id=actor_id, display_name=customer.display_name,
                status="confirmed", source="venue_table_companion", known_member_ids=[actor_id],
                registered_by_customer_id=actor_id, joined_at=stamp))
            parties = normalize_game_parties(participants)
            updated = replace(game, participants=participants, parties=parties, updated_at=stamp,
                              requirement=refresh_requirement_seat_snapshot(game.requirement, parties, 0))
            if (updated.seat_summary()["claimed_seats"] != game.seat_summary()["claimed_seats"]
                    or updated.seat_summary()["anonymous_seat_count"] != game.seat_summary()["anonymous_seat_count"] - 1):
                raise ValueError("table companion binding would change the total claimed seats")

            party_id = contact.party_id or f"party_{contact.customer_id}"
            self.store._save_game(updated)
            db.execute("INSERT INTO venue_table_companion_bindings VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                       (game.game_id, invitation_id, source[0], party_id, *source[1:], stamp.isoformat()))
            self.store._append_transition(StateTransition(
                "game_party_companion", f"{game.game_id}:{actor_id}", f"{party_id}:anonymous", "confirmed",
                "venue_table_companion_bound", f"table-companion:{game.game_id}:{invitation_id}", occurred_at=stamp,
            ))
            return updated

    def require_available_seat(self, invitation_id: str, *, actor_id: str, at: datetime) -> None:
        """滚动角色接受前核对原席和撞局，避免拒收后才发现已入群。"""
        invite = self.group.invitation_for(actor_id, invitation_id)
        if invite is None or invite.recipient_id != actor_id or at < invite.invited_at:
            raise ValueError("table companion needs its own delivered invitation")
        self._available_seat(self.store.require_game(self.group.game_id), invite, at=at)

    def bound_invitation(self, actor_id: str) -> CompanionInvitation | None:
        """只有原匿名席绑定和实际同行收件完全相符时才提供身份依据。"""
        with self.store._lock:
            row = self.store._connection.execute("""
                SELECT invitation_id, contact_id, invited_at, accepted_at FROM venue_table_companion_bindings
                WHERE game_id = ? AND recipient_id = ?
            """, (self.group.game_id, actor_id)).fetchone()
        if row is None:
            return None
        invite = self.group.invitation_for(actor_id, row["invitation_id"])
        if (invite is None or invite.recipient_id != actor_id or invite.accepted is not True or not invite.response_at
                or (invite.inviter_id, invite.invited_at.isoformat(), invite.response_at.isoformat())
                    != (row["contact_id"], row["invited_at"], row["accepted_at"])):
            return None
        return invite

    def _available_seat(self, game: Game, invite: CompanionInvitation, *, at: datetime) -> tuple[GameParticipant, CustomerProfile]:
        """入群前与实际业务绑定使用同一套座位资格，绑定时仍以事务内事实为准。"""
        actor_id = invite.recipient_id
        contact = self._eligible_contact(game, invite.inviter_id, invited_at=invite.invited_at, at=at)
        if actor_id in _identified_members(game) or any(row.customer_id == actor_id for row in game.participants):
            raise ValueError("table companion already has a game participant identity")
        if any(other.status == GameStatus.READY and actor_id in _identified_members(other)
               and game_commitment_windows_overlap(game, other) for other in self.store.games.values()):
            raise ValueError("table companion already committed to an overlapping ready game")
        customer = self.store.customers.get(actor_id)
        if customer is None:
            raise ValueError("table companion must be a known sandbox customer")
        return contact, customer

    def _eligible_contact(self, game: Game, contact_id: str, *, invited_at: datetime, at: datetime) -> GameParticipant:
        """原登记人仍有同一条有效匿名席时，才允许同行接管。"""
        if (game.status != GameStatus.READY or game.actual_started_at is not None or game.updated_at > at
                or game.seats_total != game.seat_summary()["claimed_seats"] or game.seats_total != 4
                or game.expires_at is not None and game.expires_at <= at
                or game.planned_start_at is not None and game.planned_start_at < at):
            raise ValueError("table companion needs a full current game before play")
        contact = next((row for row in game.participants if row.customer_id == contact_id and row.status in _CLAIMED), None)
        if (contact is None or contact.joined_at > invited_at or contact.anonymous_seat_count < 1
                or contact.seat_count != len({contact.customer_id, *contact.known_member_ids}) + contact.anonymous_seat_count):
            raise ValueError("table invitation no longer refers to the contact's active anonymous seat")
        if any(row.entity_type == "game_participant" and row.entity_id == f"{game.game_id}:{contact_id}"
               and invited_at <= row.occurred_at <= at for row in self.store.transitions):
            raise ValueError("table invitation predates a change to the contact's seat")
        return contact


def _identified_members(game: Game) -> set[str]:
    """冲突检查也计入历史 Party 中尚未拆分的具名同行。"""
    if game.is_playing:
        return set(game.actual_attendee_ids)
    return {actor for row in game.seat_claims() for actor in (row["contact_id"], *row["known_member_ids"]) if actor}
