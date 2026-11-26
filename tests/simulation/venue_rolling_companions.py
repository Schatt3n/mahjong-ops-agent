"""由桌群真实收件触发报名联系人的同行邀请与本人响应。"""

from datetime import datetime, timedelta

from mahjong_agent_runtime.models import GameStatus

from .venue_actors import CompanionInviteContext
from .venue_channel import AGENT_ID
from .venue_table_group import CompanionInvitation
from .venue_table_party import VenueTableParty
from .venue_table_shop import companion_inquiry_id


TABLE_COMPANION_WAKE = "table_companion_invitation"


class VenueRollingCompanions:
    """角色决定是否分享或接受。渠道邀请与原业务席位仍由各自账本验证。"""

    def __init__(self, channel, journal, actors, *, response_delay: timedelta, cutoff: datetime) -> None:
        self.channel, self.journal, self.actors = channel, journal, actors
        self.response_delay, self.cutoff = response_delay, cutoff

    def context(self, event, *, at: datetime) -> CompanionInviteContext | None:
        """模型只得到本人行动所需的称呼和阶段，受众、席位和双方关系留在沙盒。"""
        group = self.channel.table_group(event.details.get("table_room_id"))
        if group is None:
            return None
        if event.kind == TABLE_COMPANION_WAKE:
            invite = self._received(group, event, at=at)
            return CompanionInviteContext("received", self.actors[invite.inviter_id].name) if invite else None
        recipient_id = self._known_companion(group, event, at=at)
        return CompanionInviteContext("offer", self.actors[recipient_id].name) if recipient_id else None

    def dispatch(self, event, action: str, *, at: datetime) -> dict:
        """已有决定重放相同邀请与响应，确定性过期/冲突保留为未办成。"""
        group = self.channel.table_group(event.details.get("table_room_id"))
        unavailable = {"runtime_status": "table_companion_unavailable"}
        if group is None:
            return unavailable
        if action == "invite_companion" and event.kind != TABLE_COMPANION_WAKE:
            if (recipient_id := self._known_companion(group, event, at=at)) is None:
                return unavailable
            try:
                group.require_current_delivery_roster(at=at)
                invite = group.invite_companion(f"table-invite:{event.event_id}", inviter_id=event.actor_id,
                                                 recipient_id=recipient_id, at=at)
            except ValueError as exc:
                return {**unavailable, "companion_error": str(exc)}
            wake = self.journal.enqueue(actor_id=recipient_id, kind=TABLE_COMPANION_WAKE, origin_id=invite.invitation_id,
                due_at=at + self.response_delay, caused_by=invite.invitation_id, details={"table_room_id": group.room_id})
            return {"runtime_status": "table_companion_invited", "companion_invitation": _archive_invitation(invite),
                    "scheduled_wakes": [wake.event_id]}

        invite = self._received(group, event, at=at)
        if event.kind != TABLE_COMPANION_WAKE or invite is None or action not in {"accept_companion", "decline_companion"}:
            return unavailable
        accept = action == "accept_companion"
        party = VenueTableParty(group) if accept else None
        try:
            if party is not None and invite.response_at is None:
                group.require_current_delivery_roster(at=at)
                party.require_available_seat(invite.invitation_id, actor_id=event.actor_id, at=at)
            reply = group.respond_companion(invite.invitation_id, actor_id=event.actor_id, at=at, accept=accept)
        except ValueError as exc:
            return {**unavailable, "companion_error": str(exc)}
        if party is not None:
            # 若接受后中断，续跑用原收件和响应补做席位事务；不重新问角色或增席。
            party.bind_companion(invite.invitation_id, actor_id=event.actor_id, at=at)
        return {"runtime_status": "table_companion_bound" if accept else "table_companion_declined",
                "companion_invitation": _archive_invitation(reply)}

    def _known_companion(self, group, event, *, at: datetime) -> str | None:
        """只有实际收到本桌提醒的报名联系人，可自主邀请唯一可联系的常搭子。"""
        revision = group.roster_revision(at=at)
        if event.caused_by != companion_inquiry_id(group.game_id, revision) or at + self.response_delay >= self.cutoff:
            return None
        receipt = group.received(event.actor_id, event.caused_by)
        if (receipt is None or receipt.message.sender_id != AGENT_ID or receipt.message.sent_at > at
                or receipt.roster_revision != revision):
            return None
        game = group.store.require_game(group.game_id)
        if (game.status != GameStatus.READY or game.actual_started_at is not None or game.updated_at > at
                or game.expires_at is not None and game.expires_at <= at
                or game.planned_start_at is not None and game.planned_start_at <= at):
            return None
        claims = [claim for claim in game.seat_claims() if claim["contact_id"] == event.actor_id]
        if len(claims) != 1 or claims[0]["anonymous_seat_count"] != 1:
            return None
        actor = self.actors[event.actor_id]
        partners = {edge.target_sender_id for edge in actor.relationships
                    if edge.kind == "regular_partner" and edge.target_sender_id != event.actor_id}
        if len(partners) != 1:
            return None
        partner, = partners
        return partner if partner in self.actors and partner not in group.member_ids(at=at) else None

    def _received(self, group, event, *, at: datetime) -> CompanionInvitation | None:
        """只有邀请本人有本次实际收件，分享者或旁观者无法代办。"""
        if event.caused_by != event.origin_id:
            return None
        invite = group.invitation_for(event.actor_id, event.origin_id)
        if (invite is None or invite.recipient_id != event.actor_id or invite.inviter_id not in self.actors
                or invite.invited_at > at):
            return None
        return invite


def _archive_invitation(invite: CompanionInvitation) -> dict:
    """在角色事件证据中关联渠道卡片，保持私下卡片与群消息分离。"""
    return {"invitation_id": invite.invitation_id, "inviter_id": invite.inviter_id, "recipient_id": invite.recipient_id,
            "invited_at": invite.invited_at.isoformat(), "accepted": invite.accepted,
            "response_at": invite.response_at.isoformat() if invite.response_at else None}
